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
"""The states a chain can be fused with, and the reducers they give.

Every candidate state is read off the original program; none comes from a
list of known algorithms or a grammar of state shapes.

members
    The chain's own tensors run on a sub-range ``R`` of the axis instead of
    the whole of it: each reduction over ``j`` becomes a partial reduction,
    each use of the extent as a value (``total / n``) becomes ``|R|``. The
    contexts a reduction computes in place are members too.
closure
    What re-basing a member leaves behind (:mod:`rebase`): the coefficients of
    its corrections. ``Σ (x - mean)³`` leaves ``Σ (x - mean)²`` and the count;
    those re-base in turn, down to a fixpoint. This is where states the
    program never computes come from.
hoisting
    A context is the same for the whole reduction, so it can be taken out
    instead of being re-based: ``Σ q (lse - x) = lse · Σ q - Σ q x``. The
    coefficients of the context's monomials are the candidates, and the
    context itself moves to the epilogue. Where re-basing would have to
    correct a value by ``log`` of an empty side, hoisting needs nothing.
    Out of a maximum a context only comes as a shift or as a factor that
    cannot be negative (``max_j (x - mean) / sigma``): what is left is an
    extreme of the data alone.
parts
    A member that is the largest of several reductions (``max_j |x - mean|``
    is the larger of ``max_j (x - mean)`` and ``max_j (mean - x)``) has no
    state of its own to merge: each of those reductions is a candidate.

A state set is a subset of the candidates. :func:`derive` decides whether it
is a reducer; smaller sets come first, then those that hoist less.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator

from ..symbolic import ir
from ..symbolic.canonicalize import (
    Unsupported,
    contains_node,
    instantiate,
    is_constant,
    map_elems,
    mk_add,
    mk_card,
    mk_mul,
    nonneg_alias,
    nonnegative,
    subst_domain,
)
from ..symbolic.realize import node_count
from .chain import Chain, extent_symbol, partial
from .derive import ReducerSpec, derive, domains, solve_atoms, spec_key
from .rebase import Candidate, Printer, Rebaser, printed_merges
from .rebase import content as _content
from .rebase import scaled as _scaled

MAX_CANDIDATES = 24  # bound on the lifting: members plus everything derived from them
MAX_HOIST_DEPTH = 3  # contexts taken out of their reduction along the way to one state


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


class Pool:
    """The candidates of one chain: its members and what lifting adds to them.

    ``count`` picks how the members run on a sub-range where they use the
    extent of the axis as a value (see :func:`chain.partial`).
    """

    def __init__(self, chain: Chain, count: bool = True) -> None:
        self.chain = chain
        self.count = count
        _, self.R, _, _ = domains(chain.axis)
        self.cands: list[Candidate] = []
        for m in chain.members:  # producers first: a state only reads earlier ones
            struct = partial(m.body, chain.axis, count)
            origin = "member" if m.op is not None else "context"
            self.cands.append(Candidate(m.name, m.kind, origin, struct, struct, m.index, m.pid))
            self.cands[-1].state = self._state(struct)
        self.rebaser = Rebaser(self.cands, chain.axis)
        # members reading a context that does not re-base, and of those the ones
        # whose context cannot be hoisted out either
        self.unrebased: list[str] = []
        self.stuck: list[str] = []
        card = mk_card(self.R)
        if any(contains_node(c.struct, lambda x: x is card) for c in self.cands):
            # the extent used as a value is itself a reduction of the program: Σ_j 1
            self._add(Candidate("count", "sum", "extent", card, card, ()))
        self._lift()

    def _add(self, cand: Candidate) -> Candidate | None:
        if isinstance(cand.state, ir.Const) or len(self.cands) >= MAX_CANDIDATES:
            return None
        norm = _scaled(cand.state, _content(cand.state))
        for c in self.cands:
            if _scaled(c.state, _content(c.state)) is norm:  # a multiple of one already there
                c.depth = min(c.depth, cand.depth)
                return None
        self.cands.append(cand)
        return cand

    @staticmethod
    def readings(chain: Chain) -> list[bool]:
        """The ways to run ``chain`` on a sub-range: ``count`` first, and the extent
        as a constant when the members use it as a value at all."""
        ext = extent_symbol(chain.axis)
        used = any(contains_node(m.body, lambda x: x is ext) for m in chain.members)
        return [True, False] if used else [True]

    def _state(self, struct: ir.SymExpr) -> ir.SymExpr:
        """``struct`` with every read of a chain member replaced by its definition."""
        by_pid = {c.pid: c for c in self.cands if c.pid is not None}

        def actual(n: ir.Elem, depth: int):
            src = by_pid.get(n.tensor)
            if src is None:
                return None
            imap = {c: n.indices[p] for p, c in enumerate(src.index)}
            return instantiate(src.state, imap, depth)

        return map_elems(struct, actual)

    def _lift(self) -> None:
        queue = list(self.cands)
        while queue:
            cand = queue.pop(0)
            if cand.kind == "value":
                continue
            queue.extend(self._parts(cand))
            rebased = self.rebaser.decompose(cand) is not None
            queue.extend(self._closure(cand))
            hoisted = self._hoist(cand) if cand.depth < MAX_HOIST_DEPTH else []
            queue.extend(hoisted)
            if cand.origin == "member" and self.rebaser.contexts(cand) and not rebased:
                self.unrebased.append(cand.name)
                if not hoisted:
                    self.stuck.append(cand.name)

    # -- parts -------------------------------------------------------------------
    def _parts(self, cand: Candidate) -> list[Candidate]:
        """The reductions ``cand`` is the largest (or the smallest) of."""
        e = cand.struct
        if isinstance(e, ir.Mul):
            maxes = [f for f in e.args if isinstance(f, ir.Max)]
            if len(maxes) == 1 and all(is_constant(f) for f in e.args if f is not maxes[0]):
                e = maxes[0]
        if not isinstance(e, ir.Max):
            return []
        out = []
        for n, struct in enumerate(e.args):
            if not contains_node(struct, lambda x: isinstance(x, ir.Reduce)):
                continue
            state = self._state(struct)
            by = _content(state)
            new = self._add(
                Candidate(
                    f"{cand.name}.m{n}",
                    cand.kind,
                    "part",
                    _scaled(state, by),
                    _scaled(struct, by),
                    cand.index,
                    depth=cand.depth,
                    source=cand.name,
                )
            )
            if new is not None:
                out.append(new)
        return out

    # -- closure -----------------------------------------------------------------
    def _closure(self, cand: Candidate) -> list[Candidate]:
        d = self.rebaser.decompose(cand)
        if d is None:
            return []
        out = []
        for n, t in enumerate(t for t in d.all_terms() if t.side == "a"):
            try:
                state = self.rebaser.coeff_state(t)
                content = _content(state)
                state = _scaled(state, content)
                struct = _scaled(self.rebaser.coeff_struct(t), content)
            except Unsupported:
                continue
            new = self._add(
                Candidate(
                    f"{cand.name}.r{n}",
                    cand.kind,
                    "closure",
                    state,
                    struct,
                    cand.index,
                    depth=cand.depth,
                    source=cand.name,
                )
            )
            if new is not None:
                out.append(new)
        return out

    # -- hoisting ----------------------------------------------------------------
    def _hoist(self, cand: Candidate) -> list[Candidate]:
        contexts = self.rebaser.contexts(cand)
        out = []
        for size in range(1, min(len(contexts), MAX_HOIST_DEPTH - cand.depth) + 1):
            for chosen in itertools.combinations(contexts, size):
                try:
                    coeffs = self._coefficients(cand, set(chosen))
                except Unsupported:
                    coeffs = None
                for struct in coeffs or ():
                    state = self._state(struct)
                    content = _content(state)
                    new = self._add(
                        Candidate(
                            f"{cand.name}.h{len(out)}",
                            cand.kind,
                            "hoist",
                            _scaled(state, content),
                            _scaled(struct, content),
                            cand.index,
                            depth=cand.depth + size,
                            source=cand.name,
                        )
                    )
                    if new is not None:
                        out.append(new)
        return out

    def _coefficients(self, cand: Candidate, hoisted: set[int]) -> list[ir.SymExpr] | None:
        """``cand`` as Σ (monomial of the hoisted contexts) · coefficient; the coefficients.

        A context whose own definition cannot be negative (a square root, a
        sum of exponentials) is taken out as such, which is what lets it
        leave a maximum. Under a maximum of several such sums every one of
        them is read the same way.
        """
        free: set[ir.Elem] = set()

        def unpin(n: ir.Elem, depth: int):
            if self.rebaser.pinned(n) and n.tensor in hoisted:
                out = ir.elem(n.tensor, n.indices[:-1])  # no longer moves with j
                if nonnegative(self.rebaser.by_pid[n.tensor].state):
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
                    if (
                        len(inside) > 1
                        or not isinstance(inside[0], ir.Max)
                        or any(map(reduces, rest))
                    ):
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
            return None  # stuck, or nothing was hoisted
        return coeffs


def synthesize(
    source: Chain | Pool,
    *,
    max_states: int = 4,
    stats: dict | None = None,
    seen: set | None = None,
) -> Iterator[Solution]:
    """Reducers that reproduce every required member of a chain, most direct first.

    Sets built from what the program computes and what re-basing it leaves
    behind come first: they keep the program's own shifts and centring.
    Hoisted states change how a value is formed (``Σ (x - mean)²`` becomes
    ``Σ x² - (Σ x)² / n``), so they are only reached when nothing before them
    was accepted, and those that take fewer contexts out come before those
    that take more (``Σ q (x - mx)`` keeps the shift by the maximum,
    ``Σ q x`` does not). Within a level smaller sets come first, then smaller
    merges; a reducer whose merge could not be printed in re-based form comes
    last.
    ``seen`` collects the reducers yielded, so that a second pool over the same
    chain does not repeat them.
    """
    pool = source if isinstance(source, Pool) else Pool(source)
    chain = pool.chain
    full, R, _, _ = domains(chain.axis)
    by_pid = pool.rebaser.by_pid
    required = tuple(subst_domain(by_pid[m.pid].state, {R: full}) for m in chain.required)
    seen = set() if seen is None else seen
    for level in range(MAX_HOIST_DEPTH + 1):
        usable = [c for c in pool.cands if c.depth <= level]
        if not any(c.depth == level for c in usable):
            continue
        for size in range(1, max_states + 1):
            found = []
            for chosen in itertools.combinations(usable, size):
                if max(c.depth for c in chosen) != level:
                    continue
                states = tuple(c.state for c in chosen)
                spec = derive(states, required, chain.axis, stats)
                if spec is None:
                    continue
                key = spec_key(spec)
                if key in seen:
                    continue
                seen.add(key)
                printer = Printer(pool.rebaser, pool.cands, chosen, spec, solve_atoms(states, R))
                printed = printed_merges(printer)
                if printed is not None:
                    spec = dataclasses.replace(spec, merge_print=printed)
                rank = (printed is None, node_count(spec.merge_code), len(found))
                found.append((rank, Solution(spec, tuple(chosen))))
            for _, sol in sorted(found, key=lambda x: x[0]):
                yield sol


__all__ = ["Pool", "Solution", "synthesize"]
