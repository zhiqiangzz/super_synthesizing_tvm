# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Fuse the reduction chains of a TE program into single-pass reductions.

:func:`fuse` takes a TE program in which reductions over one axis feed each
other (a maximum, then a sum shifted by it; a mean, then moments about it)
and returns the same program with every such chain replaced by one tuple
``te.comm_reducer`` reduction. Per chain it

1. reads the chain off the graph (:mod:`reducer.chain`);
2. judges it (:mod:`reducer.theory`): whether a single pass exists, which
   is when every tensor the chain is needed for is a function of reductions
   that read no earlier result, and how many states are then enough. A
   chain no single pass is known for is left as it is;
3. chooses its states (:mod:`reducer.states`): the results of the program's
   own loops, supplemented where they cannot be merged and replaced where
   they are not defined on every non-empty part of the axis -- one explicit
   procedure, decided by proof (:mod:`reducer.domain`), not a search;
4. derives the reducer of those states (:mod:`reducer.derive`), checks it as
   below and writes it back as TE (:mod:`reducer.build`).

The second and third steps are algebra on the program and nothing else: what
the second says exists does exist over the reals, and the third keeps the
program's own arithmetic wherever that is defined. Floating point is the
business of the fourth.

A derived merge is correct over the reals for two non-empty parts; its
associativity and commutativity follow from that invariant. Execution must
start from a leaf and skip empty parts. The checks are about what that leaves
open, and the chain is left as it is when one of them fails:

* it must not take ``exp`` of an unbounded argument, divide by or take the
  ``log`` of a value of unknown sign, unless the original already does;
* the rebuilt program must agree with the original under MPFR evaluation on
  a few small shapes (this is a check of the code generation);
* compiled, it must meet the configured empirical error budget on adversarial
  inputs (:mod:`accuracy`).

A chain nothing passes for is left as it was, with the reason reported next
to the judgement. :func:`judge` gives the judgement alone, deriving nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from contextlib import nullcontext
from time import monotonic

import numpy as np

import tvm
from tvm import te

from .accuracy import (
    AccuracyConfig,
    AccuracyGate,
    AccuracyReport,
    _Reference,
    extent_names,
    positive_inputs,
    references,
    sample_inputs,
    spec_issues,
)
from .diagnostics import solution_record
from .reducer.build import Rewriter, build_chain
from .reducer.chain import Chain, Skipped, discover_chains
from .reducer.domain import TeSigns
from .reducer.source import chain_source, describe_members, render
from .reducer.states import Selection, Solution, members_of, select
from .reducer.theory import Judgement
from .reducer.theory import judge as judge_chain
from .symbolic.canonicalize import Unsupported

ISSUES = {
    "exp": "takes exp of an argument that is not bounded above",
    "domain": "divides by, or takes the log of, a value of unknown sign",
}


@dataclasses.dataclass(frozen=True)
class Rejected:
    """A derived reducer that was not used."""

    states: tuple[str, ...]
    why: str
    accuracy: AccuracyReport | None = None
    stage: str = ""


@dataclasses.dataclass
class ChainReport:
    chain: Chain
    theory: Judgement  # whether a single pass exists and how many states are enough
    fused: bool = False
    reason: str = ""  # why not, when it is not fused
    solution: Solution | None = None
    accuracy: AccuracyReport | None = None
    rejected: list[Rejected] = dataclasses.field(default_factory=list)  # the reducer, if refused
    selection: Selection | None = None  # how its states were chosen
    attempts: list[dict] = dataclasses.field(default_factory=list)
    chain_id: str = ""

    @property
    def axis(self) -> str:
        return self.chain.jname

    @property
    def members(self) -> tuple[str, ...]:
        """The tensors of the program on the chain."""
        return tuple(m.name for m in self.chain.members if m.op is not None)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.chain.required)

    @property
    def states(self) -> tuple[str, ...]:
        return () if self.solution is None else tuple(c.name for c in self.solution.states)

    @property
    def auxiliaries(self) -> tuple[str, ...]:
        """States the original program does not compute."""
        return () if self.solution is None else tuple(c.name for c in self.solution.auxiliaries)

    def summary(self) -> str:
        head = f"chain over {self.axis} ({', '.join(m.name for m in self.chain.reductions)})"
        if not self.fused:
            return f"{head}: not fused -- {self.reason}"
        text = f"{head}: fused into {len(self.states)} states ({', '.join(self.states)})"
        if self.auxiliaries:
            text += f"; derived: {', '.join(self.auxiliaries)}"
        if self.selection is not None and self.selection.route:
            text += f"; selection: {self.selection.route}"
        return text

    def original(self) -> list[str]:
        """The chain as the program writes it, one definition per line."""
        return describe_members(self.chain)

    def derived(self) -> list[str]:
        """The states the program does not compute as tensors, with their definitions."""
        if self.solution is None:
            return []
        how = {
            "synthesized": "synthesized by cvc5 SyGuS and independently verified",
            "context": "computed in place by the program",
            "piece": "a part of {} that re-bases by itself",
            "closure": "left behind when {} is re-based",
            "hoist": "left when the contexts of {} are taken out of its reduction",
        }
        return [
            f"{c.name} = {render(c.struct, self.chain)}    ({how[c.origin].format(c.source)})"
            for c in self.solution.auxiliaries
        ]

    def source(self, tag: str | None = None) -> str:
        """TE source of the fused reducer (empty when the chain was not fused)."""
        if self.solution is None:
            return ""
        return chain_source(self.chain, self.solution.spec, list(self.states), tag)


@dataclasses.dataclass
class Fused:
    """The result of :func:`fuse`: the rewritten program and what was done to it."""

    outputs: list[te.Tensor]
    inputs: list[te.Tensor]
    chains: list[ChainReport]
    skipped: list[Skipped]
    accuracy: AccuracyReport | None = None

    @property
    def fused(self) -> bool:
        """Was any chain fused?"""
        return any(c.fused for c in self.chains)

    def prim_func(self):
        return te.create_function([*self.inputs, *self.outputs])

    def source(self) -> str:
        """TE source of every fused reducer: what replaces the chains of the program.

        The tensors it reads and the ops that read its results are the
        program's own and are not repeated.
        """
        parts = []
        axes = [c.axis for c in self.chains if c.fused]
        for n, c in enumerate(self.chains):
            if not c.fused:
                parts.append(f"# {c.summary()}")
                continue
            head = [f"# {c.summary()}"]
            head += [f"#   {line}" for line in c.derived()]
            tag = c.axis if axes.count(c.axis) == 1 else f"{c.axis}{n}"
            parts.append("\n".join([*head, c.source(tag)]))
        return "\n\n".join(parts)

    def summary(self) -> str:
        lines = []
        for c in self.chains:
            lines.append(c.summary())
            if c.theory.fusible:  # otherwise the judgement is the reason
                lines.append(f"  theory: {c.theory.summary()}")
        lines += [f"{', '.join(s.members)}: not fused -- {s.reason}" for s in self.skipped]
        if not lines:
            lines = ["no reduction chain in the program"]
        if self.accuracy is not None:
            lines.append(self.accuracy.summary())
        return "\n".join(lines)


class _Agreement:
    """The original in float64 on a few small shapes; a rewritten program must agree.

    Every extent gets its own value, in two different orders, so that swapped
    axes do not go unnoticed; all-ones covers the degenerate reduction.
    """

    def __init__(self, outputs, inputs, declared=(), rtol: float = 1e-6, atol: float = 1e-9):
        self.inputs = list(inputs)
        self.rtol, self.atol = rtol, atol
        names = sorted(extent_names(outputs, inputs)[0])
        positive = positive_inputs(outputs, inputs, declared)
        rng = np.random.default_rng(0)
        self.cases = []
        for values in (
            dict.fromkeys(names, 1),
            {n: 2 + k for k, n in enumerate(names)},
            {n: 2 + k for k, n in enumerate(reversed(names))},
        ):
            extent = _Reference(self.inputs, [], values).extent
            arrays = sample_inputs(self.inputs, positive, extent, rng)
            want = references(outputs, self.inputs, arrays, values)
            if all(np.all(np.isfinite(w)) for w in want):
                self.cases.append((values, arrays, want))

    def __call__(self, candidate) -> bool:
        if not self.cases:
            return False
        for values, arrays, want in self.cases:
            got = references(candidate, self.inputs, arrays, values)
            for g, w in zip(got, want):
                if g.shape != w.shape or not np.allclose(g, w, rtol=self.rtol, atol=self.atol):
                    return False
        return True


def _judge(chain: Chain) -> Judgement:
    return judge_chain(chain, members_of(chain))


def chain_key(chain):
    """Stable within an input workload, independent of allocated symbolic IDs."""
    payload = (
        chain.jname,
        tuple((m.name, m.kind) for m in chain.members),
        tuple(m.name for m in chain.required),
        str(chain.extent),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def _span(experiment, phase):
    return nullcontext() if experiment is None else experiment.span(phase)


def _fuse_chain(
    chain,
    outputs,
    rewriter,
    gate,
    agree,
    max_states,
    accuracy,
    signs,
    retry_rejected_candidates=False,
    experiment=None,
):
    with _span(experiment, "derivation_proof"):
        report = ChainReport(chain, _judge(chain), chain_id=chain_key(chain))
    if not report.theory.fusible:
        report.reason = report.theory.summary()
        return report, rewriter
    accepted_trial = None
    cache = {}
    verdict_cache = {}
    key = report.chain_id
    if experiment is not None:
        experiment.prefixes.setdefault(key, rewriter.fork())

    def accept(sol):
        with _span(experiment, "execution_overhead"):
            return evaluate(sol)

    def evaluate(sol):
        nonlocal accepted_trial
        spec = sol.spec
        names = tuple(c.name for c in sol.states)
        attempt = {
            "attempt_id": len(report.attempts) + 1,
            "proposal_id": sol.proposal_id,
            "source": sol.source,
            "checker_id": sol.checker_id,
            "states": names,
            "algebra": "passed",
            "status": "not_attempted",
            "timings": {},
            "phases": dict.fromkeys(
                ("domain", "static", "build", "agreement", "numerical"), "not_attempted"
            ),
        }
        if experiment is not None:
            experiment.solutions[key] = sol
            attempt["solution"] = solution_record(sol)
        report.attempts.append(attempt)
        start = monotonic()

        def refuse(why, stage, status="failed"):
            attempt.update(status=stage + "_rejection", reason=why)
            attempt["phases"][stage] = status
            report.rejected.append(Rejected(names, why, report.accuracy, stage))
            report.reason = f"the reducer of ({', '.join(names)}) is refused: {why}"
            return False

        def run(stage, fn):
            t = monotonic()
            try:
                with _span(
                    experiment,
                    {
                        "build": "code_generation",
                        "numerical": "accuracy_check",
                        "agreement": "agreement",
                    }.get(stage, "execution_static"),
                ):
                    return fn()
            finally:
                attempt["timings"][stage] = monotonic() - t

        report.accuracy = None
        try:
            if max_states is not None and len(names) > max_states:
                return refuse(f"{len(names)} states exceed budget {max_states}", "domain")
            # A fixed-I evaluation-form comparison must recheck the printed
            # domain, keeping initialization, output formulas and state definitions.
            if experiment is not None and key in experiment.fixed:
                from .reducer.states import _Select
                from .symbolic.canonicalize import subst_domain

                world = experiment.frozen[key][0]
                required = tuple(
                    subst_domain(world.by_pid[m.pid].state, {world.R: world.full})
                    for m in chain.required
                )
                why = _Select(world)._undefined(spec, sol.states, required)
                if why:
                    return refuse(why, "domain", "unknown")
            attempt["phases"]["domain"] = "passed"
            barred = set(gate.required) if gate is not None and accuracy == "filter" else set()
            issues = run("static", lambda: sorted(barred & spec_issues(spec)))
            if issues:
                return refuse(
                    "; ".join(ISSUES[k] for k in issues) + " (the original does not)", "static"
                )
            attempt["phases"]["static"] = "passed" if barred else "not_attempted"
            # Every attempt starts at the same committed prefix. No failed fork
            # can affect subsequent candidates or undo an earlier accepted chain.
            trial = rewriter.fork()

            def build():
                build_chain(chain, spec, trial)
                return [trial.tensor(t) for t in outputs]

            try:
                candidate = run("build", build)
            except (Unsupported, tvm.error.TVMError) as err:
                return refuse(f"cannot be written as TE: {err}", "build")
            attempt["phases"]["build"] = "passed"
            if retry_rejected_candidates or experiment is not None:
                inputs = agree.inputs
                code = te.create_function([*inputs, *candidate]).script()
                compiler = "llvm/default" if experiment is None else experiment.compiler_key
                fingerprint = hashlib.sha256(
                    json.dumps(
                        [
                            code,
                            compiler,
                            accuracy,
                            dataclasses.asdict(gate.config) if gate else None,
                        ]
                    ).encode()
                ).hexdigest()
                attempt["fingerprint"] = fingerprint
                if experiment is not None:
                    attempt["program"] = code
                if fingerprint in cache:
                    previous = cache[fingerprint]
                    report.accuracy = verdict_cache.get(fingerprint)
                    attempt.update(
                        status="duplicate",
                        duplicate_of=previous["attempt_id"],
                        reused_status=previous["status"],
                        accuracy=previous.get("accuracy"),
                    )
                    # Cache belongs to this chain prefix and fixed gate only.
                    return previous["status"] == "accepted"
                cache[fingerprint] = attempt
            try:
                agreed = run("agreement", lambda: agree(candidate))
            except (Unsupported, tvm.error.TVMError, ValueError) as err:
                return refuse(str(err), "agreement")
            if not agreed:
                return refuse(
                    "the rebuilt program differs from the original on small inputs", "agreement"
                )
            attempt["phases"]["agreement"] = "passed"
            try:
                verdict = (
                    run("numerical", lambda: gate.check(candidate)) if gate is not None else None
                )
            except (Unsupported, tvm.error.TVMError, ValueError) as err:
                return refuse(str(err), "numerical")
            report.accuracy = verdict
            if "fingerprint" in attempt:
                verdict_cache[attempt["fingerprint"]] = verdict
            if gate is not None:
                attempt["output_checks"] = getattr(gate, "last_checks", [])
            attempt["accuracy"] = dataclasses.asdict(verdict) if verdict else None
            attempt["numerical_executed"] = bool(verdict and verdict.families)
            if experiment is not None and verdict is not None:
                experiment.checks.append(verdict)
            if verdict is not None:
                attempt["phases"]["numerical"] = "passed" if verdict.ok else "failed"
                if verdict.static:
                    attempt["phases"]["numerical"] = "not_attempted"
                if not verdict.ok and accuracy == "filter":
                    return refuse(verdict.summary(), "static" if verdict.static else "numerical")
            attempt["status"] = "accepted"
            accepted_trial = trial
            report.fused, report.solution, report.reason = True, sol, ""
            if experiment is not None:
                experiment.solutions[key] = sol
            return True
        finally:
            attempt["seconds"] = monotonic() - start

    if experiment is not None and key in experiment.fixed:
        if retry_rejected_candidates:
            raise ValueError("fixed-state evaluation comparisons cannot retry selection")
        sol = experiment.fixed[key]
        accept(sol)
    else:
        with _span(experiment, "derivation_proof"):
            kwargs = {}
            if retry_rejected_candidates:
                kwargs["accept_candidate"] = accept
            if experiment is not None:
                kwargs.update(experiment=experiment, chain_id=key)
            selection = select(chain, members_of(chain), signs, max_states=max_states, **kwargs)
        report.selection = selection
        if selection.solution is None:
            prefix = "execution candidates rejected; " if report.attempts else "no state set: "
            report.reason = prefix + selection.failure
            return report, rewriter
        if not retry_rejected_candidates:
            accept(selection.solution)
    return report, accepted_trial if report.fused else rewriter


def fuse(
    outputs,
    inputs,
    *,
    max_states: int | None = None,
    accuracy: str = "filter",
    accuracy_config: AccuracyConfig | None = None,
    retry_rejected_candidates: bool = False,
) -> Fused:
    """Fuse every reduction chain behind ``outputs`` into a single-pass reduction.

    Parameters
    ----------
    outputs : te.Tensor or list of te.Tensor
        The program, by its outputs.
    inputs : list of te.Tensor
        Its placeholders, in the order the generated function takes them.
    max_states : int, optional
        Further limit the number of state tensors. The default budget is the
        theoretical construction's state count. A chain with no verified
        preserving representation within the budget is left unchanged.
    accuracy : {"filter", "report", "off"}
        ``"filter"`` only accepts a reducer that meets the empirical precision budget
        (see :mod:`accuracy`); ``"report"`` accepts the first one that is
        correct and attaches its accuracy report; ``"off"`` skips the
        comparison (the rebuilt program is still checked against the original
        using high-precision references).

    retry_rejected_candidates : bool
        Continue the existing bounded state search after execution rejects a
        proved candidate. Defaults to False. All candidates reuse the same gate
        inputs; report mode still accepts a failed numerical report.

    Returns
    -------
    Fused
        ``outputs`` rewritten (unchanged where nothing was fused) and one
        report per chain.
    """
    return _fuse(
        outputs,
        inputs,
        max_states=max_states,
        accuracy=accuracy,
        accuracy_config=accuracy_config,
        retry_rejected_candidates=retry_rejected_candidates,
    )


def _fuse(
    outputs,
    inputs,
    *,
    max_states=None,
    accuracy="filter",
    accuracy_config=None,
    retry_rejected_candidates=False,
    experiment=None,
):
    """One implementation for production and explicit internal experiment controls."""
    if accuracy not in ("filter", "report", "off"):
        raise ValueError(f"accuracy must be 'filter', 'report' or 'off', not {accuracy!r}")
    outputs = [outputs] if isinstance(outputs, te.Tensor) else list(outputs)
    inputs = list(inputs)
    chains, skipped = discover_chains(outputs)
    original_outputs = outputs
    if any("two different indices" in s.reason for s in skipped):
        from .reducer.views import specialize

        outputs = specialize(outputs)
        chains, skipped = discover_chains(outputs)
    if experiment is not None:
        if experiment.graph is None:
            experiment.graph = (outputs, chains, skipped)
        else:
            outputs, chains, skipped = experiment.graph
    reports: list[ChainReport] = []
    rewriter = Rewriter()
    if chains:
        config = accuracy_config or AccuracyConfig()
        signs = TeSigns(outputs, config.positive)
        try:
            with _span(experiment, "accuracy_setup"):
                opts = (
                    {}
                    if experiment is None or experiment.compiler is None
                    else {"compiler": experiment.compiler}
                )
                gate = (
                    AccuracyGate(original_outputs, inputs, config, **opts)
                    if accuracy != "off"
                    else None
                )
                if gate is not None and experiment is not None:
                    experiment.gates.append(gate)
                    if experiment.gate_observer is not None:
                        experiment.gate_observer(gate)
            with _span(experiment, "agreement"):
                agree = _Agreement(original_outputs, inputs, config.positive)
                if experiment is not None:
                    experiment.agreements.append(agree)
        except Unsupported as err:
            # nothing is rewritten that cannot be checked against the original
            why = f"the program cannot be evaluated to check a rewrite against ({err})"
            reports = [
                ChainReport(chain, _judge(chain), reason=why, chain_id=chain_key(chain))
                for chain in chains
            ]
            chains = []
        for chain in chains:
            report, rewriter = _fuse_chain(
                chain,
                outputs,
                experiment.prefixes[chain_key(chain)].fork()
                if experiment is not None and chain_key(chain) in experiment.fixed
                else rewriter,
                gate,
                agree,
                max_states,
                accuracy,
                signs,
                retry_rejected_candidates,
                experiment,
            )
            reports.append(report)
    verdict = next((r.accuracy for r in reversed(reports) if r.fused), None)
    result = (
        [rewriter.tensor(t) for t in outputs] if any(r.fused for r in reports) else original_outputs
    )
    return Fused(result, inputs, reports, skipped, verdict)


def judge(outputs) -> list[Judgement]:
    """For every reduction chain behind ``outputs``: does a single pass exist, and
    with how many states?

    Nothing is derived or rewritten, and nothing is run: the answer is read
    off the program by the rules of :mod:`reducer.theory`. A sum lets go of
    whatever multiplies it and does not move with the axis; a maximum of
    whatever is added to it, of a factor that cannot be negative and of a
    function that grows with its argument. Where that frees every reduction
    of the earlier results it reads, those reductions are the states of a
    single pass.
    """
    outputs = [outputs] if isinstance(outputs, te.Tensor) else list(outputs)
    chains, skipped = discover_chains(outputs)
    if any("two different indices" in s.reason for s in skipped):
        from .reducer.views import specialize

        chains, _ = discover_chains(specialize(outputs))
    return [_judge(chain) for chain in chains]


__all__ = ["ChainReport", "Fused", "Rejected", "fuse", "judge"]
