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
"""Re-basing: merging two partial results whose context has moved.

A later link of a chain reads the *final* value of an earlier one::

    den(R) = Σ_{j∈R} exp(x_j - mx(R))          ss(R) = Σ_{j∈R} (x_j - mean(R))²

Each side of a merge was computed against its own context, ``mx(A)`` and
``mx(B)``; the merged value needs ``mx(A u B)``. Re-basing writes the context
of the union as the side's own plus a correction::

    κ(A u B) = κ(X) + δ_X          (shift)        κ(A u B) = κ(X) · r_X     (scale)

Both are identities, two ways of naming the unknown. Substituted into the
definition, the canonical rules pull the correction out of the partial
reduction of each side::

    Σ_A exp(x - mx_A - δ_A)   = exp(-δ_A) · den(A)
    Σ_A (x - mean_A - δ_A)²   = ss(A) - 2 δ_A · Σ_A (x - mean_A) + δ_A² · |A|

What is left under each power of the correction is a *coefficient*: a
reduction over one side with that side's own context. A coefficient is the
candidate's own value on that side, zero, or something else -- and that
something else is what the reducer is missing: a state the program never
computes (the count, a lower moment). Collecting them is the lifting
(:mod:`states`); nothing here assumes what they look like.

The canonical merge of a derived reducer is an equivalence judge, not a
recipe: it cancels the max shift of a softmax and expands a centred moment
into raw power sums. The merge printed from a decomposition instead keeps
every partial value as the state it is and only adds the corrections. It is
proven equal to the canonical one before it is used.
"""

from __future__ import annotations

import dataclasses
import itertools

from ..dims import DimKey
from ..symbolic import ir
from ..symbolic.canonicalize import (
    Unsupported,
    contains_node,
    instantiate,
    is_constant,
    map_elems,
    mk_add,
    mk_mul,
    positive,
    recanonicalize,
    subst,
    subst_domain,
    term_view,
    walk,
)
from . import verify
from .derive import ReducerSpec, domains, swap_sides

SIDE_OFFSET = {"a": -10_000, "b": -20_000}  # a member's id when pinned to one side of a merge


@dataclasses.dataclass(eq=False)
class Candidate:
    """A possible reducer state: a function of partial reductions over a sub-range ``R``."""

    name: str
    # "sum" / "max" / "min": merged by re-basing; "value": a function of other candidates
    kind: str
    origin: str  # "member", "context", "extent", "part", "closure" or "hoist"
    state: ir.SymExpr  # canonical, down to the boundary tensors
    struct: ir.SymExpr  # the same with reads of chain members kept (pinned) as written
    index: tuple[ir.IndexExpr, ...]  # chain coordinates of its own axes
    pid: int | None = None  # the id other definitions read it by (chain members only)
    depth: int = 0  # how many contexts were taken out of a reduction to obtain it
    source: str = ""  # what it was derived from, for the report

    @property
    def rank(self) -> int:
        return len(self.index)


@dataclasses.dataclass(frozen=True)
class Term:
    """``key · coeff``: a product of corrections times a partial value of one side."""

    side: str | None  # None: a plain constant
    key: ir.SymExpr
    coeff: ir.SymExpr  # over the side's partial reductions, contexts pinned to that side


@dataclasses.dataclass(frozen=True)
class Decomposition:
    """A re-based merge: a sum of terms, or the max of several such sums."""

    terms: tuple[Term, ...] = ()
    parts: tuple[Decomposition, ...] = ()  # non-empty: this is ``max(parts)``
    corr: tuple = ()  # (atom, side, pid, mode) for every correction
    negated: bool = False  # ``-max(parts)``: how a minimum is merged

    def all_terms(self):
        yield from self.terms
        for p in self.parts:
            yield from p.all_terms()


def _is_partial(x: ir.Node) -> bool:
    return isinstance(x, ir.Reduce | ir.Card) and isinstance(x.domain, ir.DSym)


def content(e: ir.SymExpr):
    """The rational factor to divide out: ``-3 Σ f`` and ``Σ f`` are the same state.

    The coefficient of one fixed term, chosen by the term itself so that every
    multiple of ``e`` picks the same one.
    """
    _, terms = term_view(e)
    if not terms:
        return 1
    return terms[min(terms, key=lambda core: core.uid)]


def scaled(e: ir.SymExpr, by) -> ir.SymExpr:
    return e if by == 1 else mk_mul(ir.const(1 / by), e)


def normalised(e: ir.SymExpr) -> ir.SymExpr:
    """``e`` with its rational content divided out."""
    return scaled(e, content(e))


class Rebaser:
    """How each candidate of one chain re-bases; independent of the states chosen."""

    def __init__(self, members: list[Candidate], axis: DimKey) -> None:
        self.axis = axis
        self.by_pid = {c.pid: c for c in members if c.pid is not None}
        _, self.R, a, b = domains(axis)
        self.sides = {"a": a, "b": b}
        self._memo: dict[int, Decomposition | None] = {}

    # -- reading definitions ---------------------------------------------------
    def pinned(self, n: ir.Node) -> bool:
        """A read of a chain member from inside a reduction (it carries the bound index)."""
        return (
            isinstance(n, ir.Elem)
            and n.tensor in self.by_pid
            and len(n.indices) == self.by_pid[n.tensor].rank + 1
        )

    def contexts(self, cand: Candidate) -> list[int]:
        return sorted({n.tensor for n in walk(cand.struct) if self.pinned(n)})

    def side_of(self, domain) -> str | None:
        for side, dom in self.sides.items():
            if domain is dom:
                return side
        return None

    def sides_in(self, e: ir.Node) -> set[str]:
        out = set()
        for n in walk(e):
            if isinstance(n, ir.Reduce | ir.Card):
                s = self.side_of(n.domain)
                if s is not None:
                    out.add(s)
        return out

    def on_side(self, e: ir.SymExpr, side: str) -> ir.SymExpr:
        """A coefficient with every pinned context replaced by its own definition."""
        dom = self.sides[side]

        def actual(n: ir.Elem, depth: int):
            src = self.by_pid.get(n.tensor - SIDE_OFFSET[side])
            if src is None:
                return None
            imap = {c: n.indices[p] for p, c in enumerate(src.index)}
            return instantiate(subst_domain(src.state, {self.R: dom}), imap, depth)

        return map_elems(e, actual)

    def coeff_state(self, term: Term) -> ir.SymExpr:
        """The coefficient as a function of partial reductions over ``R``."""
        return subst_domain(self.on_side(term.coeff, term.side), {self.sides[term.side]: self.R})

    def coeff_struct(self, term: Term) -> ir.SymExpr:
        """The coefficient over ``R`` with its contexts still pinned."""
        off = SIDE_OFFSET[term.side]

        def unpin(n: ir.Elem, depth: int):
            if n.tensor - off in self.by_pid:
                return ir.elem(n.tensor - off, n.indices)
            return None

        return subst_domain(map_elems(term.coeff, unpin), {self.sides[term.side]: self.R})

    # -- decomposition -----------------------------------------------------------
    def decompose(self, cand: Candidate) -> Decomposition | None:
        """The first admissible re-basing of ``cand``: every correction leaves its
        partial reduction, and the correction-free part is ``cand`` itself on
        each side. Additive corrections are tried before multiplicative ones."""
        key = id(cand)
        if key not in self._memo:
            self._memo[key] = None
            contexts = self.contexts(cand)
            for modes in itertools.product(("shift", "scale"), repeat=len(contexts)):
                try:
                    d = self._rebased(cand, dict(zip(contexts, modes)))
                except Unsupported:
                    d = None
                if d is not None and self._admissible(cand, d):
                    self._memo[key] = d
                    break
        return self._memo[key]

    def _admissible(self, cand: Candidate, d: Decomposition) -> bool:
        """Some coefficient is the candidate itself (a minimum is ``-max`` of its negation)."""
        own = normalised(cand.state)
        for t in d.all_terms():
            if t.side != "a":
                continue
            try:
                if normalised(self.coeff_state(t)) is own:
                    return True
            except Unsupported:
                continue
        return False

    def _rebased(self, cand: Candidate, modes: dict[int, str]) -> Decomposition | None:
        union = ir.dunion(self.sides["a"], self.sides["b"])
        corr: dict[ir.Atom, tuple[str, int, str]] = {}  # atom -> (side, pid, mode)
        depth_domain: dict[int, ir.Domain] = {}

        def per_side(n: ir.Elem, depth: int):
            if not self.pinned(n):
                return None
            side = self.side_of(depth_domain.get(depth))
            if side is None:
                raise Unsupported("context outside a partial reduction")
            mode = modes[n.tensor]
            atom = ir.atom(f"{'d' if mode == 'shift' else 'r'}{side}{-n.tensor}")
            corr[atom] = (side, n.tensor, mode)
            pinned = ir.elem(n.tensor + SIDE_OFFSET[side], n.indices)
            return mk_add(pinned, atom) if mode == "shift" else mk_mul(pinned, atom)

        merged = subst_domain(cand.struct, {self.R: union})
        merged = map_elems(merged, per_side, depth_domain)
        table = tuple((atom, *what) for atom, what in corr.items())
        return self._group(merged, corr, table)

    def _group(self, e: ir.SymExpr, corr, table) -> Decomposition | None:
        """``e`` as Σ correction · (a partial value of one side), or a max of such sums."""
        scale, negated = ir.ONE, False
        if isinstance(e, ir.Mul):  # c · max(u, v) = max(c u, c v) for c > 0, -max(-c u, -c v) else
            maxes = [f for f in e.args if isinstance(f, ir.Max)]
            others = [f for f in e.args if not isinstance(f, ir.Max)]
            if len(maxes) == 1 and all(is_constant(f) for f in others):
                c = mk_mul(*others)
                if positive(c):
                    e, scale = maxes[0], c
                elif positive(mk_mul(ir.MINUS_ONE, c)):
                    e, scale, negated = maxes[0], mk_mul(ir.MINUS_ONE, c), True
        if isinstance(e, ir.Max):
            parts = [self._group(mk_mul(scale, a), corr, table) for a in e.args]
            if any(p is None for p in parts):
                return None
            return Decomposition(parts=tuple(parts), corr=table, negated=negated)
        groups: dict[tuple[str | None, ir.SymExpr], list[ir.SymExpr]] = {}
        for term in e.args if isinstance(e, ir.Add) else (e,):
            factors = term.args if isinstance(term, ir.Mul) else (term,)
            key = [f for f in factors if contains_node(f, lambda x: x in corr)]
            rest = [f for f in factors if not contains_node(f, lambda x: x in corr)]
            if any(contains_node(f, _is_partial) for f in key):
                return None  # a correction stuck inside a partial reduction
            sides = {corr[x][0] for f in key for x in walk(f) if x in corr}
            sides |= {s for f in rest for s in self.sides_in(f)}
            if len(sides) > 1:
                return None
            side = next(iter(sides), None)
            groups.setdefault((side, mk_mul(*key)), []).append(mk_mul(*rest))
        terms = tuple(Term(side, key, mk_add(*rests)) for (side, key), rests in groups.items())
        return Decomposition(terms=terms, corr=table)


class Printer:
    """The merges of one chosen state set, printed from the decompositions."""

    def __init__(self, rebaser: Rebaser, pool: list[Candidate], chosen, spec: ReducerSpec, sol):
        self.rb = rebaser
        self.pool = pool
        self.chosen = list(chosen)
        self.spec = spec
        self.atom_map = {}
        for atom, expr in sol.items():
            for side, dom in rebaser.sides.items():
                value = subst_domain(swap_sides(expr, "a", side), {rebaser.R: dom})
                self.atom_map[subst_domain(atom, {rebaser.R: dom})] = value
        self.merged: dict[int, ir.SymExpr | None] = {}  # id(candidate) -> raw merged value

    # -- values of a candidate on one side / merged ------------------------------
    def slot(self, cand: Candidate) -> int | None:
        for k, c in enumerate(self.chosen):
            if c is cand or c.state is cand.state:  # the same function under another name
                return k
        return None

    def side_value(self, cand: Candidate, side: str) -> ir.SymExpr | None:
        """``κ(X)`` over the state variables of side ``X``."""
        k = self.slot(cand)
        if k is not None:
            return ir.state_var(side, k)
        v = subst(subst_domain(cand.state, {self.rb.R: self.rb.sides[side]}), self.atom_map)
        return None if contains_node(v, _is_partial) else v

    def merged_value(self, cand: Candidate) -> ir.SymExpr | None:
        """``κ(A u B)``: a merged slot ``m_k`` or this candidate's own printed merge."""
        k = self.slot(cand)
        if k is not None:
            return ir.state_var("m", k)
        return self.print(cand)

    # -- printing ------------------------------------------------------------------
    def print(self, cand: Candidate) -> ir.SymExpr | None:
        key = id(cand)
        if key in self.merged:
            return self.merged[key]
        self.merged[key] = None  # guards cycles
        try:
            out = self._value(cand) if cand.kind == "value" else self._reduction(cand)
        except Unsupported:
            out = None
        self.merged[key] = out
        return out

    def _value(self, cand: Candidate) -> ir.SymExpr | None:
        """A function of other candidates: the same function of their merged values."""
        mapping: dict[ir.Node, ir.SymExpr] = {}
        for n in walk(cand.struct):
            if isinstance(n, ir.Elem) and n.tensor in self.rb.by_pid:
                src = self.rb.by_pid[n.tensor]
                if tuple(n.indices) != src.index:
                    return None  # read at other indices: not a state of this reducer
                v = self.merged_value(src)
            elif isinstance(n, ir.Card) and n.domain is self.rb.R:
                count = next((c for c in self.pool if c.state is n), None)
                v = self.merged_value(count) if count is not None else None
            elif isinstance(n, ir.Reduce):
                return None  # a value must be a plain function of other tensors
            else:
                continue
            if v is None:
                return None
            mapping[n] = v
        return ir.raw_subst(cand.struct, mapping)

    def _reduction(self, cand: Candidate) -> ir.SymExpr | None:
        d = self.rb.decompose(cand)
        return None if d is None else self._evaluate(d)

    def _evaluate(self, d: Decomposition) -> ir.SymExpr | None:
        if d.parts:
            args = [self._evaluate(p) for p in d.parts]
            if any(a is None for a in args):
                return None
            largest = ir.raw_max(args)
            return ir.raw_mul([ir.MINUS_ONE, largest]) if d.negated else largest
        corr = {atom: (side, pid, mode) for atom, side, pid, mode in d.corr}
        printed = []
        for t in d.terms:
            value = self._state_value(t)
            if value is None:
                return None
            if value is ir.ZERO:
                continue
            printed.append(_raw_product(self._print_key(t.key, corr), value))
        if not printed:
            return ir.ZERO
        return printed[0] if len(printed) == 1 else ir.raw_add(printed)

    def _state_value(self, t: Term) -> ir.SymExpr | None:
        """A coefficient over the state variables of its side: a state monomial."""
        if t.side is None:
            return t.coeff if isinstance(t.coeff, ir.Const) else None
        v = subst(self.rb.on_side(t.coeff, t.side), self.atom_map)
        if contains_node(v, _is_partial) or not _monomial(v):
            return None
        return v

    def _print_key(self, key: ir.SymExpr, corr) -> ir.SymExpr:
        mapping: dict[ir.Node, ir.SymExpr] = {}
        for n in walk(key):
            if isinstance(n, ir.Pow) and n.base in corr and n.exponent == -1:
                side, pid, mode = corr[n.base]
                if mode == "scale":  # 1 / rho = k(X) / k(A u B), finite where k(X) = 0
                    here, there = self._context(pid, side)
                    mapping[n] = ir.raw_mul([here, ir.raw_pow(there, -1)])
        for atom, (side, pid, mode) in corr.items():
            if not contains_node(key, lambda x, atom=atom: x is atom):
                continue
            here, there = self._context(pid, side)
            if mode == "shift":
                mapping[atom] = ir.raw_add([there, ir.raw_mul([ir.MINUS_ONE, here])])
            else:
                mapping[atom] = ir.raw_mul([there, ir.raw_pow(here, -1)])
        return ir.raw_subst(key, mapping)

    def _context(self, pid: int, side: str) -> tuple[ir.SymExpr, ir.SymExpr]:
        cand = self.rb.by_pid[pid]
        here, there = self.side_value(cand, side), self.merged_value(cand)
        if here is None or there is None:
            raise Unsupported(f"context {cand.name} is not expressible in the states")
        return here, there


def _monomial(v: ir.SymExpr) -> bool:
    """A constant times a product of positive integer powers of state variables."""
    factors = v.args if isinstance(v, ir.Mul) else (v,)
    for f in factors:
        if isinstance(f, ir.Const | ir.StateVar):
            continue
        if isinstance(f, ir.Pow) and isinstance(f.base, ir.StateVar):
            if f.exponent.denominator == 1 and f.exponent > 0:
                continue
        return False
    return True


def _raw_product(key: ir.SymExpr, value: ir.SymExpr) -> ir.SymExpr:
    if key is ir.ONE:
        return value
    if value is ir.ONE:
        return key
    factors = list(value.args) if isinstance(value, ir.Mul) else [value]
    if isinstance(factors[0], ir.Const):
        return ir.raw_mul([factors[0], key, *factors[1:]])
    return ir.raw_mul([key, *factors])


def _expand(prints: list[ir.SymExpr]) -> list[ir.SymExpr] | None:
    """Replace merged-slot references ``m_k`` by slot ``k``'s own printed merge."""
    done: dict[int, ir.SymExpr] = {}
    busy: set[int] = set()

    def get(k: int) -> ir.SymExpr | None:
        if k in done:
            return done[k]
        if k in busy:
            return None
        busy.add(k)
        refs = {n for n in walk(prints[k]) if isinstance(n, ir.StateVar) and n.side == "m"}
        mapping = {}
        for r in refs:
            v = get(r.k)
            if v is None:
                return None
            mapping[r] = v
        done[k] = ir.raw_subst(prints[k], mapping)
        busy.discard(k)
        return done[k]

    out = [get(k) for k in range(len(prints))]
    return None if any(x is None for x in out) else out


def proves(printed: ir.SymExpr, canonical: ir.SymExpr) -> bool:
    got = recanonicalize(printed)
    if got is canonical:
        return True
    return verify.equal_modulo_max(got, canonical) or verify.numeric_equal(got, canonical)


def printed_merges(printer: Printer) -> tuple[ir.SymExpr, ...] | None:
    """The merge of every chosen state as it should be evaluated, if all of them
    print and are proven equal to the canonical merges."""
    prints = [printer.print(c) for c in printer.chosen]
    if any(p is None for p in prints):
        return None
    expanded = _expand(prints)
    if expanded is None:
        return None
    if not all(proves(p, m) for p, m in zip(expanded, printer.spec.merge)):
        return None
    return tuple(expanded)


__all__ = [
    "Candidate",
    "Decomposition",
    "Printer",
    "Rebaser",
    "Term",
    "content",
    "normalised",
    "printed_merges",
    "proves",
    "scaled",
]
