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
2. collects candidate states from the program itself (:mod:`reducer.states`)
   and derives a reducer from each set of them (:mod:`reducer.derive`);
3. takes the first reducer that survives the checks below and writes it back
   as TE (:mod:`reducer.build`).

A derived reducer is correct over the reals by construction and its monoid
laws are verified. The checks are about what that leaves open:

* the merge must be finite against the identity element in floating point;
* it must not take ``exp`` of an unbounded argument, divide by or take the
  ``log`` of a value of unknown sign, unless the original already does;
* the rebuilt program must agree with the original, evaluated in float64 on
  a few small shapes (this is a check of the code generation);
* compiled, it may not be less accurate than the original on adversarial
  inputs (:mod:`accuracy`).

A chain nothing passes for is left as it was, with the reason reported.
"""

from __future__ import annotations

import dataclasses

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
from .reducer.build import Rewriter, build_chain
from .reducer.chain import Chain, Skipped, discover_chains
from .reducer.source import chain_source, describe_members, render
from .reducer.states import Pool, Solution, synthesize
from .reducer.verify import identity_safe
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


@dataclasses.dataclass
class ChainReport:
    chain: Chain
    fused: bool = False
    reason: str = ""  # why not, when it is not fused
    solution: Solution | None = None
    accuracy: AccuracyReport | None = None
    rejected: list[Rejected] = dataclasses.field(default_factory=list)

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
        return text

    def original(self) -> list[str]:
        """The chain as the program writes it, one definition per line."""
        return describe_members(self.chain)

    def derived(self) -> list[str]:
        """The states the program does not compute as tensors, with their definitions."""
        if self.solution is None:
            return []
        how = {
            "context": "computed in place by the program",
            "extent": "the extent of the axis, as far as it has been reduced",
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
        return te.create_prim_func([*self.inputs, *self.outputs])

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
        lines = [c.summary() for c in self.chains]
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

    def __init__(self, outputs, inputs, rtol: float = 1e-6, atol: float = 1e-9) -> None:
        self.inputs = list(inputs)
        self.rtol, self.atol = rtol, atol
        names = sorted(extent_names(outputs, inputs)[0])
        positive = positive_inputs(outputs, inputs)
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


def _why_not(pool: Pool, report: ChainReport, max_states: int) -> str:
    parts = []
    for name in pool.unrebased:
        cand = next(c for c in pool.cands if c.name == name)
        ctx = ", ".join(pool.rebaser.by_pid[p].name for p in pool.rebaser.contexts(cand))
        if name in pool.stuck:
            parts.append(
                f"{name} reads {ctx} from inside its reduction and neither re-basing nor "
                "hoisting takes it out (no finite lifting found)"
            )
        else:
            parts.append(
                f"{name} cannot be re-based around {ctx}, and with {ctx} hoisted out no "
                f"reducer of at most {max_states} states was accepted"
            )
    if report.rejected:
        first = report.rejected[0]
        parts.append(
            f"{len(report.rejected)} derived reducer(s) rejected, e.g. "
            f"({', '.join(first.states)}): {first.why}"
        )
    elif not parts:
        need = ", ".join(report.required)
        parts.append(f"no set of at most {max_states} derived states reproduces {need}")
    return "; ".join(parts)


def _fuse_chain(
    chain: Chain,
    outputs: list[te.Tensor],
    rewriter: Rewriter,
    gate: AccuracyGate | None,
    agree: _Agreement,
    max_states: int,
    accuracy: str,
) -> tuple[ChainReport, Rewriter]:
    report = ChainReport(chain)
    dtype = str(chain.reductions[0].tensor.dtype)
    barred = set(gate.required) if gate is not None and accuracy == "filter" else set()
    pools = [Pool(chain, count) for count in Pool.readings(chain)]
    seen: set = set()
    for pool in pools:
        for sol in synthesize(pool, max_states=max_states, seen=seen):
            spec = sol.spec
            names = tuple(c.name for c in sol.states)

            def reject(why: str, names=names) -> None:
                report.rejected.append(Rejected(names, why))

            if not identity_safe(spec.merge_code, spec.identity, dtype):
                reject("its merge is not finite against the empty state in floating point")
                continue
            issues = sorted(barred & spec_issues(spec))
            if issues:
                reject("; ".join(ISSUES[k] for k in issues) + " (the original does not)")
                continue
            trial = rewriter.fork()
            try:
                build_chain(chain, spec, trial)
                candidate = [trial.tensor(t) for t in outputs]
            except (Unsupported, tvm.error.TVMError) as err:
                reject(f"cannot be written as TE: {err}")
                continue
            if not agree(candidate):
                reject("the rebuilt program differs from the original on small inputs")
                continue
            verdict = gate.check(candidate) if gate is not None else None
            if verdict is not None and not verdict.ok and accuracy == "filter":
                reject(verdict.summary())
                continue
            report.fused, report.solution, report.accuracy = True, sol, verdict
            return report, trial
    report.reason = _why_not(pools[0], report, max_states)
    return report, rewriter


def fuse(
    outputs,
    inputs,
    *,
    max_states: int = 4,
    accuracy: str = "filter",
    accuracy_config: AccuracyConfig | None = None,
) -> Fused:
    """Fuse every reduction chain behind ``outputs`` into a single-pass reduction.

    Parameters
    ----------
    outputs : te.Tensor or list of te.Tensor
        The program, by its outputs.
    inputs : list of te.Tensor
        Its placeholders, in the order the generated function takes them.
    max_states : int
        Largest reducer considered (number of state tensors).
    accuracy : {"filter", "report", "off"}
        ``"filter"`` only accepts a reducer that is as accurate as the original
        (see :mod:`accuracy`); ``"report"`` accepts the first one that is
        correct and attaches its accuracy report; ``"off"`` skips the
        comparison (the rebuilt program is still checked against the original
        in float64).

    Returns
    -------
    Fused
        ``outputs`` rewritten (unchanged where nothing was fused) and one
        report per chain.
    """
    if accuracy not in ("filter", "report", "off"):
        raise ValueError(f"accuracy must be 'filter', 'report' or 'off', not {accuracy!r}")
    outputs = [outputs] if isinstance(outputs, te.Tensor) else list(outputs)
    inputs = list(inputs)
    chains, skipped = discover_chains(outputs)
    reports: list[ChainReport] = []
    rewriter = Rewriter()
    if chains:
        gate = AccuracyGate(outputs, inputs, accuracy_config) if accuracy != "off" else None
        agree = _Agreement(outputs, inputs)
        for chain in chains:
            report, rewriter = _fuse_chain(
                chain, outputs, rewriter, gate, agree, max_states, accuracy
            )
            reports.append(report)
    verdict = next((r.accuracy for r in reversed(reports) if r.fused), None)
    return Fused([rewriter.tensor(t) for t in outputs], inputs, reports, skipped, verdict)


__all__ = ["ChainReport", "Fused", "Rejected", "fuse"]
