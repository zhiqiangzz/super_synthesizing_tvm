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
"""Preserve original reductions; search supplements with proved recovery.

The context-free verification coordinates are B. The maintained tuple A starts
with the original program's reductions. Candidates for additional coordinates
come from B and from rebase coefficients/pieces; a finite search by cardinality
accepts only complete A -> B recovery, a defined inverse, and tuple merge
closure. The upper bound from the theoretical construction is the default
budget, not a promise that a preserving representation exists at that size.

Z3 verifies zero-branch recovery using solver-proven facts about summands (see
smt). Symbolic normalization and SymPy prove the remaining algebraic identities.
Different correction rates generate candidates, never permission to discard
an original sum. Only a demonstrated domain failure on an actual nonempty
segment authorizes a hoist. Search exhaustion/unknown preserves the program
unfused; it is not an impossibility proof. Empty partitions remain an execution
layer concern. Real proofs do not replace the floating-point accuracy gate.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from fractions import Fraction
from itertools import combinations

from ..symbolic import ir, sympy_bridge
from ..symbolic.canonicalize import (
    NONNEG,
    Unsupported,
    contains_node,
    instantiate,
    is_constant,
    map_elems,
    mk_add,
    mk_card,
    mk_mul,
    mk_pow,
    nonneg_alias,
    recanonicalize,
    subst,
    subst_domain,
    walk,
)
from ..symbolic.realize import node_count
from . import domain, smt
from .chain import Chain, partial
from .derive import (
    ReducerSpec,
    atoms_in,
    derive,
    domains,
    merge_of,
    psi_inverts,
    solve_atoms,
    state_outputs,
)
from .rebase import Candidate, Printer, Rebaser, printed_merges, proves
from .rebase import content as _content
from .rebase import scaled as _scaled
from .source import render

MAX_ROUNDS = 64  # decisions only grow; this guards against a mistake, it bounds no search
MAX_SUPPLEMENTS = 16


@dataclasses.dataclass(frozen=True)
class Solution:
    """A derived reducer and the candidates its states are."""

    spec: ReducerSpec
    states: tuple[Candidate, ...]

    @property
    def auxiliaries(self) -> tuple[Candidate, ...]:
        """States the original program does not compute."""
        return tuple(c for c in self.states if c.origin != "member")

    @property
    def hoists(self) -> int:
        return sum(c.depth for c in self.states)


@dataclasses.dataclass(frozen=True)
class Step:
    """One decision of :func:`select` and why it was taken."""

    action: str  # "keep", "supplement", "hoist", "parts", "split" or "drop"
    subject: str
    reason: str

    def summary(self) -> str:
        return f"{self.action} {self.subject}: {self.reason}"


@dataclasses.dataclass
class Selection:
    """The states :func:`select` arrived at, how, and the reducer they give."""

    states: tuple[Candidate, ...] = ()
    steps: list[Step] = dataclasses.field(default_factory=list)
    solution: Solution | None = None
    failure: str = ""


def members_of(chain: Chain, count: bool | None = None) -> list[Candidate]:
    """The members of ``chain`` run on a sub-range ``R`` of the axis, producers first.

    ``count`` picks how they run there where they use the extent of the axis
    as a value (see :func:`chain.partial`); by default as the chain reads it
    (:attr:`Chain.count`).
    """
    count = chain.count if count is None else count
    cands: list[Candidate] = []
    for m in chain.members:  # producers first: a state only reads earlier ones
        struct = partial(m.body, chain.axis, count)
        origin = "member" if m.op is not None else "context"
        cand = Candidate(m.name, m.kind, origin, struct, struct, m.index, m.pid)
        cand.state = _defined(struct, cands)
        cands.append(cand)
    return cands


def _plain(tid: int) -> int:
    """The id of a member read as a value known to be >= 0 (see ``nonneg_alias``)."""
    return tid - NONNEG if NONNEG - 10_000 < tid < NONNEG + 10_000 else tid


def _defined(struct: ir.SymExpr, cands: list[Candidate]) -> ir.SymExpr:
    """``struct`` with every read of a chain member replaced by its definition."""
    by_pid = {c.pid: c for c in cands if c.pid is not None}

    def actual(n: ir.Elem, depth: int):
        src = by_pid.get(_plain(n.tensor))
        if src is None:
            return None
        imap = {c: n.indices[p] for p, c in enumerate(src.index)}
        return instantiate(src.state, imap, depth)

    return map_elems(struct, actual)


def _unwrapped(e: ir.SymExpr) -> tuple[ir.SymExpr, bool]:
    """What is left of ``e`` under the functions the canonical rules took out of its
    maximum (a power, ``exp``, ``log``, a polynomial), and whether there was one.

    A constant factor and terms without a reduction are stepped over as well;
    terms that all hold one and the same reduction (``r³ + r``) are a function
    of it.
    """
    inner = False

    def reduces(x: ir.SymExpr) -> bool:
        return contains_node(x, lambda n: isinstance(n, ir.Reduce))

    while True:
        if isinstance(e, ir.Pow):
            e, inner = e.base, True
        elif isinstance(e, ir.Exp | ir.Log):
            e, inner = e.arg, True
        elif isinstance(e, ir.Mul | ir.Add):
            held = [a for a in e.args if reduces(a)]
            if not held or not all(is_constant(a) for a in e.args if a not in held):
                return e, inner
            if len(held) == 1:
                e = held[0]
                continue
            within = {r for a in held for r in _reductions(a)}
            if len(within) != 1:
                return e, inner
            (e,), inner = within, True
        else:
            return e, inner


def _reductions(e: ir.SymExpr) -> list[ir.Reduce]:
    """The reductions of ``e`` that are not inside another one."""
    out: dict[ir.Reduce, None] = {}
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Reduce):
            out[n] = None
        else:
            stack.extend(n.children())
    return list(out)


def _order(e: ir.SymExpr):
    """A key that orders expressions by their structure, not by when they were made."""
    return (e.size, repr(e))


# ---------------------------------------------------------------------------
# the chain, as select sees it
# ---------------------------------------------------------------------------
class _Chain:
    """Members, re-basing and signs of one chain; what candidates are made from."""

    def __init__(self, chain: Chain, members: list[Candidate], signs: domain.TeSigns) -> None:
        self.chain = chain
        self.members = members
        self.full, self.R, self.A, self.B = domains(chain.axis)
        self.rebaser = Rebaser(members, chain.axis)
        self.by_pid = self.rebaser.by_pid
        self.te = signs
        self._member_sign: dict[int, int] = {}
        self.signs = domain.Signs(self.tensor_sign)
        card = mk_card(self.R)
        self.count = Candidate("count", "sum", "extent", card, card, ())

    # -- signs -------------------------------------------------------------------
    def tensor_sign(self, tid: int) -> int:
        if tid in self.by_pid:
            return self.member_sign(tid)
        if tid >= 0:
            return self.te.tensor(self.chain.lower.placeholder(tid))
        return domain.ANY

    def member_sign(self, pid: int) -> int:
        if pid not in self._member_sign:
            self._member_sign[pid] = domain.ANY  # guards recursion
            m = next(m for m in self.chain.members if m.pid == pid)
            if m.op is not None:
                sign = self.te.tensor(m.tensor)
            else:
                sign = self.signs(self.free(m.body))
            self._member_sign[pid] = sign
        return self._member_sign[pid]

    def state_sign(self, cand: Candidate) -> int:
        if cand.pid is not None and cand.pid in self.by_pid:
            return self.member_sign(cand.pid)
        return self.signs(self.free(cand.struct))

    # -- reading definitions -------------------------------------------------------
    def free(self, struct: ir.SymExpr) -> ir.SymExpr:
        """``struct`` with every context a symbol that does not move with the axis."""

        def fn(n: ir.Elem, depth: int):
            if not self.rebaser.pinned(n):
                return None
            out = ir.elem(n.tensor, n.indices[:-1])
            return nonneg_alias(out) if self.member_sign(n.tensor) >= domain.NONNEG else out

        return map_elems(struct, fn)

    def defined(self, e: ir.SymExpr) -> ir.SymExpr:
        """``e`` with every member read replaced by its definition over ``R``."""
        return _defined(e, self.members)

    def render(self, e: ir.SymExpr) -> str:
        def plain(n: ir.Elem, depth: int):
            tid = _plain(n.tensor)
            return ir.elem(tid, n.indices) if tid != n.tensor else None

        try:
            return render(map_elems(e, plain), self.chain)
        except Unsupported:
            return repr(e)

    def name(self, pid: int) -> str:
        return self.by_pid[pid].name

    # -- what a state can be replaced by, or supplemented with ------------------------
    def hoisted(self, cand: Candidate, contexts: frozenset[int]) -> list[Candidate] | None:
        """``cand`` with ``contexts`` taken out: the coefficients of their monomials."""
        try:
            coeffs = _coefficients(self.rebaser, cand, contexts, self.member_sign)
        except Unsupported:
            coeffs = None
        if coeffs is None:
            return None
        out, seen = [], set()
        for struct in sorted(coeffs, key=_order):
            state = self.defined(struct)
            if isinstance(state, ir.Const):
                continue
            by = _content(state)
            if _scaled(state, by) in seen:  # the coefficient of two monomials
                continue
            seen.add(_scaled(state, by))
            out.append(
                Candidate(
                    f"{cand.name}.h{len(out)}",
                    cand.kind,
                    "hoist",
                    _scaled(state, by),
                    _scaled(struct, by),
                    cand.index,
                    depth=cand.depth + len(contexts),
                    source=cand.name,
                )
            )
        return out

    def parts(self, cand: Candidate) -> list[Candidate]:
        """The reductions ``cand`` is the largest (or smallest) of, or a growing function of."""
        e, inner = cand.struct, False
        if cand.kind in ("max", "min"):
            e, inner = _unwrapped(e)
        if isinstance(e, ir.Max):
            args = e.args
        elif inner and isinstance(e, ir.Reduce):
            args = (e,)
        else:
            return []
        out = []
        for struct in sorted(args, key=_order):
            if not contains_node(struct, lambda x: isinstance(x, ir.Reduce)):
                continue
            state = self.defined(struct)
            by = _content(state)
            out.append(
                Candidate(
                    f"{cand.name}.m{len(out)}",
                    cand.kind,
                    "inner" if inner else "part",
                    _scaled(state, by),
                    _scaled(struct, by),
                    cand.index,
                    depth=cand.depth,
                    source=cand.name,
                )
            )
        return out

    def pieces(self, cand: Candidate) -> list[Candidate]:
        """The parts of a sum that re-base each by itself where the sum does not."""
        terms = self.rebaser.split(cand)
        if terms is None:
            return []
        out = []
        for t in sorted(terms, key=lambda t: _order(self.rebaser.coeff_struct(t))):
            state = self.rebaser.coeff_state(t)
            by = _content(state)
            out.append(
                Candidate(
                    f"{cand.name}.p{len(out)}",
                    cand.kind,
                    "piece",
                    _scaled(state, by),
                    _scaled(self.rebaser.coeff_struct(t), by),
                    cand.index,
                    depth=cand.depth,
                    source=cand.name,
                )
            )
        return out

    def coefficients(self, cand: Candidate) -> list[Candidate]:
        """What re-basing ``cand`` leaves behind: the coefficients of its corrections,
        one reduction before a compound one."""
        d = self.rebaser.decompose(cand)
        if d is None:
            return []
        out = []
        for t in d.all_terms():
            if t.side != "a":
                continue
            try:
                state = self.rebaser.coeff_state(t)
                if isinstance(state, ir.Const):
                    continue
                by = _content(state)
                struct = _scaled(self.rebaser.coeff_struct(t), by)
                state = _scaled(state, by)
            except Unsupported:
                continue
            out.append(
                Candidate(
                    f"{cand.name}.r",
                    cand.kind,
                    "closure",
                    state,
                    struct,
                    cand.index,
                    depth=cand.depth,
                    source=cand.name,
                )
            )
        out.sort(key=lambda c: (len(_reductions(c.state)) > 1, _order(c.state)))
        return out


def _coefficients(
    rebaser: Rebaser, cand: Candidate, hoisted: frozenset[int], sign: Callable[[int], int]
) -> list[ir.SymExpr] | None:
    """``cand`` as Σ (monomial of the hoisted contexts) · coefficient; the coefficients.

    A context that cannot be negative is taken out as such, which is what
    lets it leave a maximum. Under a maximum of several such sums every one
    of them is read the same way. ``None`` when a context does not leave its
    reduction, or when nothing is taken out.
    """
    free: set[ir.Elem] = set()

    def unpin(n: ir.Elem, depth: int):
        if rebaser.pinned(n) and n.tensor in hoisted:
            out = ir.elem(n.tensor, n.indices[:-1])  # no longer moves with j
            if sign(n.tensor) >= domain.NONNEG:
                out = nonneg_alias(out)
            free.add(out)
            return out
        return None

    def moved(f: ir.SymExpr) -> bool:
        return contains_node(f, lambda x: x in free)

    def reduces(f: ir.SymExpr) -> bool:
        return contains_node(f, lambda x: isinstance(x, ir.Reduce | ir.Card))

    coeffs: list[ir.SymExpr] = []
    keys: set[ir.SymExpr] = set()

    def visit(e: ir.SymExpr) -> bool:
        if isinstance(e, ir.Max):
            return all(visit(a) for a in e.args)
        groups: dict[ir.SymExpr, list[ir.SymExpr]] = {}
        for term in e.args if isinstance(e, ir.Add) else (e,):
            factors = term.args if isinstance(term, ir.Mul) else (term,)
            key = [f for f in factors if moved(f)]
            rest = [f for f in factors if not moved(f)]
            inside = [f for f in key if reduces(f)]
            if inside:
                # c · max(u, v) with the contexts next to the reductions of u and v
                if len(inside) > 1 or not isinstance(inside[0], ir.Max) or any(map(reduces, rest)):
                    return False  # the context does not leave the reduction
                keys.update(f for f in key if f is not inside[0])
                if not visit(inside[0]):
                    return False
                continue
            groups.setdefault(mk_mul(*key), []).append(mk_mul(*rest))
        keys.update(groups)
        coeffs.extend(mk_add(*rests) for rests in groups.values())
        return True

    if not visit(map_elems(cand.struct, unpin)) or keys <= {ir.ONE}:
        return None
    return coeffs


# ---------------------------------------------------------------------------
# decisions and the problems that call for them
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _Decisions:
    """What has been decided so far; it only ever grows."""

    hoisted: dict[int, frozenset[int]] = dataclasses.field(default_factory=dict)  # loop: contexts

    def key(self):
        return (tuple(sorted((k, tuple(sorted(v))) for k, v in self.hoisted.items())),)


@dataclasses.dataclass(frozen=True)
class _Fault:
    """Why a state cannot be kept, and the contexts that is blamed on."""

    kind: str  # demonstrated source "domain", "unknown_domain", or inverse "divisor"
    contexts: frozenset[int]
    reason: str


class _Select:
    def __init__(self, world: _Chain, max_states: int | None = None) -> None:
        self.w = world
        self.steps: list[Step] = []
        self.round: list[Step] = []  # supplements of the current round
        self.dec = _Decisions()
        self.loops = [c for c in world.members if c.kind != "value"]
        from .theory import judge

        bound = judge(world.chain, world.members).states
        self.budget = (
            bound
            if max_states is None
            else min(bound if bound is not None else max_states, max_states)
        )
        self.search_failure = ""

    def note(self, action: str, subject: str, reason: str) -> None:
        """Record a step. Supplements belong to the round that adds them: a later
        decision rebuilds the states and they are noted again if still needed."""
        step = Step(action, subject, reason)
        target = self.round if action == "supplement" else self.steps
        if step not in target:
            target.append(step)

    @property
    def trace(self) -> list[Step]:
        """The decisions taken, then the supplements of the states arrived at."""
        return [*self.steps, *self.round]

    # -- maintained tuple A under the decisions ----------------------------------------
    def base(self) -> list[Candidate] | str:
        out: list[Candidate] = []
        for loop in self.loops:
            got = [loop]
            if loop.pid in self.dec.hoisted:
                got = self.w.hoisted(loop, self.dec.hoisted[loop.pid])
                if got is None:
                    return f"the contexts of {loop.name} do not leave its reduction"
                # The original definition was already proved partial. Its
                # replacement coordinates may be represented by their parts.
                got = [p for c in got for p in (self.w.parts(c) or self.w.pieces(c) or [c])]
            out.extend(got)
        return _distinct(out)

    def line(self, cand: Candidate, states: list[Candidate]) -> Candidate | None:
        """The loop of the program a state comes from (``None``: the count, a context)."""
        loops = {c.name: c for c in self.loops}
        made = {c.name: c for c in states}
        cur = cand
        for _ in range(len(states) + 1):
            if cur.name in loops:
                return loops[cur.name]
            nxt = loops.get(cur.source) or made.get(cur.source)
            if nxt is None:
                break
            cur = nxt
        name = cur.source or cur.name  # a part of a replacement: named after what it came from
        while "." in name:
            name = name.rsplit(".", 1)[0]
            if name in loops:
                return loops[name]
        return None

    # -- supplement ---------------------------------------------------------------------
    def supplement(self, states: list[Candidate]):
        """Search supplements by cardinality, accepting only complete recovery.

        Rebase pieces are candidate definitions, not replacements. Their order
        breaks ties, not proof obligations. Failure is not an impossibility proof.
        """
        w = self.w
        targets = {a for c in states for a in atoms_in(c.state, w.R)}
        for m in w.chain.required:
            targets.update(atoms_in(w.by_pid[m.pid].state, w.R))

        def recover(trial):
            trace: dict = {}
            try:
                sts = tuple(c.state for c in trial)
                sol = solve_atoms(sts, w.R, trace)
                if sol is None or any(atoms_in(subst(a, sol), w.R) for a in targets):
                    return None
                signs = self._state_signs(trial)
                if any(self._nonempty(c, k, trial, trace, signs) for k, c in enumerate(trial)):
                    return None
                if psi_inverts(sts, sol, w.R, w.A) is None:
                    return None
                # For the polynomial fragment, independently check uniqueness
                # with two copies of B. Transcendentals use the constructive
                # inverse proof above rather than treating opaque functions as
                # injective by fiat.
                if not any(isinstance(n, ir.Exp | ir.Log) for s in sts for n in walk(s)):
                    if (
                        smt.recovery_status(
                            sts, tuple(sorted(targets)), w.R, tuple(w.state_sign(c) for c in trial)
                        )
                        != "unsat"
                    ):
                        return None
                if any(merge_of(c.state, sol, w.R, w.A, w.B) is None for c in trial):
                    return None
                return sol, trace
            except Unsupported:
                return None

        limit = self.budget if self.budget is not None else len(states) + MAX_SUPPLEMENTS
        remaining = limit - len(states)
        if remaining < 0:
            self.search_failure = "preserving the original states exceeds the state budget"
            return states, None, {}
        got = recover(states)
        if got is not None:
            return states, *got
        pool = []
        if any(isinstance(a, ir.Card) for a in targets):
            pool.append(w.count)
        for owner in states:
            coeffs = w.coefficients(owner)
            pool.extend(
                dataclasses.replace(c, name=f"{owner.name}.r{k}") for k, c in enumerate(coeffs)
            )
            pool.extend(w.pieces(owner))
            pool.extend(w.parts(owner))
        for a in sorted(targets, key=_order):
            if isinstance(a, ir.Reduce):
                pool.append(
                    Candidate(
                        w.render(a),
                        a.kind,
                        "trivial",
                        a,
                        a,
                        tuple(ir.idx(n) for n in sorted(a.free_idx)),
                    )
                )
        have = {_scaled(c.state, _content(c.state)) for c in states}
        pool = [c for c in _distinct(pool) if _scaled(c.state, _content(c.state)) not in have]
        tried = 0
        for size in range(1, min(remaining, len(pool)) + 1):
            for extra in combinations(pool, size):
                tried += 1
                if tried > 256:
                    self.search_failure = (
                        "candidate search limit reached; preservation is not disproven"
                    )
                    return states, None, {}
                trial = [*states, *extra]
                got = recover(trial)
                if got is not None:
                    named, counts = [], {}
                    for c in extra:
                        if c.origin == "closure":
                            k = counts.get(c.source, 0)
                            counts[c.source] = k + 1
                            c = dataclasses.replace(c, name=f"{c.source}.r{k}")
                        named.append(c)
                        self.note(
                            "supplement",
                            f"{c.name} = {w.render(c.struct)}",
                            "complete recovery and tuple merge proved on nonempty parts",
                        )
                    return [*states, *named], *got
        self.search_failure = (
            f"no verified preserving state set within {self.budget} states "
            f"({tried} candidates checked; this is not an impossibility proof)"
        )
        return states, None, {}

    def _state_signs(self, states) -> domain.Signs:
        w = self.w
        sv = {}
        for k, c in enumerate(states):
            s = w.state_sign(c)
            for side in "abm":
                sv[ir.state_var(side, k)] = s
        return domain.Signs(w.tensor_sign, sv)

    def _nonempty(self, c, k, states, trace, signs) -> list[_Fault]:
        w = self.w
        out = []
        contexts = set(w.rebaser.contexts(c))
        # its definition, through its contexts and through the values those read
        todo: list[tuple[ir.SymExpr, int | None]] = [(c.struct, None)]
        seen: set[int] = set()
        while todo:
            struct, via = todo.pop(0)
            free = w.free(struct)
            checked = [free]
            # A context-dependent divisor may still sit inside a max whose
            # monotonicity has not been established. Inspect it before using
            # structural failure as any reason to replace the state.
            checked.extend(
                n
                for n in walk(free)
                if isinstance(n, ir.Pow | ir.Log)
                and any(isinstance(a, ir.Elem) and _plain(a.tensor) in w.by_pid for a in walk(n))
            )
            conditions = dict.fromkeys(c for e in checked for c in domain.conditions(e))
            for cond in conditions:
                if cond.holds(w.signs):
                    continue
                inside = {_plain(n.tensor) for n in walk(cond.operand) if isinstance(n, ir.Elem)}
                blamed = frozenset({via}) if via is not None else frozenset(inside & contexts)
                proof = domain.witness(w.defined(cond.operand), w.R)
                demonstrated = smt.singleton_counterexample(
                    w.defined(cond.operand), cond.kind, w.R, w.tensor_sign
                )
                kind = "domain" if demonstrated else "unknown_domain"
                out.append(_Fault(kind, blamed, _why(w, cond, proof, via)))
            for n in walk(struct):
                pid = _plain(n.tensor) if isinstance(n, ir.Elem) else None
                if pid in w.by_pid and pid not in seen and w.by_pid[pid].kind == "value":
                    seen.add(pid)
                    todo.append((w.by_pid[pid].struct, pid if via is None else via))
        # what is read back from it
        for atom, how in trace.items():
            if not isinstance(atom, ir.Node) or how.state != k:
                continue
            for d in how.divisors:
                if signs(d) == domain.POS or signs(mk_mul(ir.MINUS_ONE, d)) == domain.POS:
                    continue
                refs = {n.k for n in walk(d) if isinstance(n, ir.StateVar)}
                blamed = frozenset(states[i].pid for i in refs if states[i].pid in contexts)
                over = subst(d, {ir.state_var("a", i): states[i].state for i in refs})
                proof = domain.witness(over, w.R)
                why = (
                    f"reading {w.render(atom)} back from it divides by "
                    f"{w.render(over)}, which {_witness_text(w, proof)}"
                )
                out.append(_Fault("divisor", blamed, why))
        return out

    # -- replace ------------------------------------------------------------------------------
    def act(self, c: Candidate, faults: list[_Fault], states, sol) -> str | None:
        """Take the decisions the faults of ``c`` call for; a failure, or ``None``."""
        w = self.w
        loop = self.line(c, states)
        if loop is None:
            return f"{c.name} cannot be kept and is no loop of the program ({faults[0].reason})"
        before = self.dec.key()
        chosen = [f for f in faults if f.kind == "domain"]
        if not chosen:
            return f"preserving {c.name}: {faults[0].reason}; no proof authorizes replacement"
        reason = "; ".join(f.reason for f in chosen)
        every = frozenset(w.rebaser.contexts(loop))
        blamed = frozenset().union(*(f.contexts for f in chosen))
        self._hoist(loop, blamed or every, c, reason)
        if self.dec.key() == before:  # nothing new: take every context out
            if not every or self.dec.hoisted.get(loop.pid) == every:
                return f"{c.name} cannot be kept and nothing replaces it ({reason})"
            self._hoist(loop, every, c, reason)
        return None

    def _hoist(self, loop: Candidate, contexts: frozenset[int], c: Candidate, reason: str) -> None:
        w = self.w
        have = self.dec.hoisted.get(loop.pid, frozenset())
        want = have | contexts
        if not want:
            return
        if w.hoisted(loop, want) is None:  # these do not leave on their own: all of them do
            every = frozenset(w.rebaser.contexts(loop))
            if every != want:
                asked = ", ".join(w.name(p) for p in sorted(want - have))
                reason = f"{reason}; {asked} alone does not leave the reduction"
            want = every
        if want != have:
            self.dec.hoisted[loop.pid] = want
            names = ", ".join(w.name(p) for p in sorted(want))
            self.note("hoist", f"{names} out of {loop.name}", f"{c.name}: {reason}")

    # -- the procedure --------------------------------------------------------------------------
    def run(self) -> Selection:
        if not smt.available():
            return Selection(failure="state proofs require z3-solver (install the superopt extra)")
        for loop in self.loops:
            self.note("keep", loop.name, "the result of a loop of the program")
        for _ in range(MAX_ROUNDS):
            self.round = []
            states = self.base()
            if isinstance(states, str):
                return Selection((), self.trace, None, states)
            # Only a defect in the original definition can authorize a hoist.
            signs = self._state_signs(states)
            original_fault = next(
                (
                    (c, faults)
                    for k, c in enumerate(states)
                    if (faults := self._nonempty(c, k, states, {}, signs))
                ),
                None,
            )
            if original_fault is not None:
                failure = self.act(*original_fault, states, None)
                if failure is not None:
                    return Selection(tuple(states), self.trace, None, failure)
                continue
            states, sol, trace = self.supplement(states)
            if sol is None:
                return Selection(tuple(states), self.trace, None, self.search_failure)
            return self.finish(states)
        return Selection((), self.trace, None, "the selection did not settle")

    def finish(self, states: list[Candidate]) -> Selection:
        """Derive the proved tuple without deleting protected or verification coordinates."""
        w = self.w
        required = tuple(
            subst_domain(w.by_pid[m.pid].state, {w.R: w.full}) for m in w.chain.required
        )
        sts = tuple(c.state for c in states)
        sol = solve_atoms(sts, w.R)
        printer = None if sol is None else Printer(w.rebaser, list(states), states, sol)
        spec = derive(
            sts,
            required,
            w.chain.axis,
            kept=None if printer is None else tuple(printer.kept(c) for c in states),
            preserve=True,
        )
        if spec is None:
            why = "no reducer with its laws proven is derived from these states"
            return Selection(tuple(states), self.trace, None, why)
        printed = printed_merges(printer, spec.merge)
        if printed is not None:
            spec = dataclasses.replace(spec, merge_print=printed)
        written = _written_epilogues(w, spec, sol)
        if written is not None:
            spec = dataclasses.replace(spec, epilogue_print=written)
        undefined = self._undefined(spec, states, required)
        if undefined is not None:
            return Selection(tuple(states), self.trace, None, undefined)
        return Selection(tuple(states), self.trace, Solution(spec, tuple(states)))

    def _undefined(self, spec: ReducerSpec, states, required) -> str | None:
        """Where the reducer is not shown to be defined on non-empty parts: a divisor,
        ``log`` or root argument of its merge, or one its epilogues add to the program's.

        The laws follow from the merge being derived only where it is defined;
        the canonical rules cancel ``x / x`` without asking whether ``x`` is 0.
        """
        w = self.w
        signs = self._state_signs(states)
        for k, m in enumerate(spec.merge_code):
            if any(isinstance(n, ir.IfZero) for n in walk(m)):
                env = {
                    ir.state_var(side, j): subst_domain(c.state, {w.R: dom})
                    for side, dom in (("a", w.A), ("b", w.B))
                    for j, c in enumerate(states)
                }
                if smt.defined(ir.raw_subst(m, env)):
                    continue
                return f"the guarded merge of {states[k].name} has an unproved domain"
            # Inspect the formula as printed: expanding squared corrections
            # can hide the positivity needed for a retained scale context.
            for cond in domain.conditions(m):
                if not cond.holds(signs) and not smt.condition_holds(
                    cond.operand, cond.kind, tuple(signs.states.items())
                ):
                    return (
                        f"the merge of {states[k].name} is not shown to be defined: "
                        f"{_condition_text(cond)} {w.render(cond.operand)}"
                    )
        refs = {}
        for k, c in enumerate(states):
            refs[-(k + 1)] = w.state_sign(c)
        out_signs = domain.Signs(lambda tid: refs.get(tid, w.tensor_sign(tid)))
        names = {ir.elem(-(k + 1), ()): ir.atom(c.name) for k, c in enumerate(states)}
        for e, closed in zip(spec.epilogue_code, required):
            own = [c.operand for c in domain.conditions(closed)]
            for cond in domain.conditions(recanonicalize(e)):
                if cond.holds(out_signs) or self._as_program(cond, own, spec):
                    continue
                shown = subst(
                    cond.operand,
                    {
                        r: names.get(ir.elem(r.tensor, ()), r)
                        for r in walk(cond.operand)
                        if isinstance(r, ir.Elem) and r.tensor < 0
                    },
                )
                return (
                    f"an epilogue is not shown to be defined: "
                    f"{_condition_text(cond)} {w.render(shown)}"
                )
        return None

    def _as_program(self, cond: domain.Condition, own, spec: ReducerSpec) -> bool:
        """Does the program itself, over the whole axis, need what ``cond`` needs?

        It does when the operand is one of the program's own (divisors, ``log``
        and root arguments) times something positive: then both are zero, or
        negative, together.
        """
        w = self.w
        try:
            over = _closed_operand(cond.operand, spec, w)
        except Unsupported:
            return False
        for o in own:
            try:
                ratio = mk_mul(over, mk_pow(o, Fraction(-1)))
            except Unsupported:
                continue
            if w.signs(ratio) == domain.POS or sympy_bridge.positive_ratio(over, o):
                return True
        return False


def select(
    chain: Chain, members: list[Candidate], signs: domain.TeSigns, max_states: int | None = None
) -> Selection:
    """The states of a single pass over ``chain`` and the reducer they give (see the
    module docstring); ``members`` are its members on a part (:func:`members_of`)."""
    return _Select(_Chain(chain, members, signs), max_states).run()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _condition_text(cond: domain.Condition) -> str:
    return {"nonzero": "it divides by", "positive": "it takes the log of"}.get(
        cond.kind, "it takes a root of"
    )


def _closed_operand(operand: ir.SymExpr, spec: ReducerSpec, w: _Chain) -> ir.SymExpr:
    """An epilogue operand over the state outputs, written over the whole axis."""
    mapping = {}
    for n in walk(operand):
        if isinstance(n, ir.Elem) and n.tensor < 0 and -n.tensor - 1 < spec.arity:
            mapping[n] = spec.closed[-n.tensor - 1]
    return subst(operand, mapping)


def _distinct(cands: list[Candidate]) -> list[Candidate]:
    out, seen = [], set()
    for c in cands:
        key = _scaled(c.state, _content(c.state))
        if key in seen or isinstance(c.state, ir.Const):
            continue
        seen.add(key)
        out.append(c)
    return out


def _witness_text(w: _Chain, proof: domain.Witness) -> str:
    if proof.kind == "every single element":
        return "is 0 on every single element"
    if proof.kind == "where zero":
        return f"is 0 where {' or '.join(w.render(r) for r in proof.reads)} is"
    return "is not proven to be non-zero"


def _why(w: _Chain, cond: domain.Condition, proof: domain.Witness, via: int | None) -> str:
    where = f" (through {w.name(via)})" if via is not None else ""
    what = w.render(cond.operand) + where
    if cond.kind == "nonzero":
        return f"it divides by {what}, which {_witness_text(w, proof)}"
    if cond.kind == "positive":
        zero = proof.kind != "not proven"
        tail = _witness_text(w, proof) if zero else "is not proven to be positive"
        return f"it takes the log of {what}, which {tail}"
    return f"it takes a root of {what}, which is not proven to be non-negative"


def _written_epilogues(w: _Chain, spec: ReducerSpec, sol) -> tuple[ir.SymExpr, ...] | None:
    """The epilogues with a function around a maximum kept as the program writes it.

    Over the states ``max_j x`` and ``mean`` the canonical epilogue of
    ``max_j (x - mean)³`` is the expanded cube, four large terms that cancel;
    the cube of ``max_j x - mean`` is what the program computes. It is used
    where it is the shorter of the two, once it is proven equal to the
    canonical one; ``None`` when no epilogue changes.
    """
    try:
        outputs = state_outputs(spec.states, sol, w.chain.axis)
    except Unsupported:
        return None
    out = []
    for m, canonical in zip(w.chain.required, spec.epilogues):
        cand = w.by_pid[m.pid]
        out.append(canonical)
        if cand.kind not in ("max", "min") or not _unwrapped(cand.struct)[1]:
            continue
        try:
            values = {
                red: subst(subst_domain(w.defined(red), {w.R: w.full}), outputs)
                for red in _reductions(cand.struct)
            }
        except Unsupported:
            continue
        written = ir.raw_subst(cand.struct, values)
        if node_count((written,)) < node_count((canonical,)) and proves(written, canonical):
            out[-1] = written
    return None if all(a is b for a, b in zip(out, spec.epilogues)) else tuple(out)


__all__ = ["Selection", "Solution", "Step", "members_of", "select"]
