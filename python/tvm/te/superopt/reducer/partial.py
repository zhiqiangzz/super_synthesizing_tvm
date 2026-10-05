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
"""Partialisation: reducer states taken from the original program itself.

Run every op of the original program on a sub-range ``R`` of the reduction
axis ``j`` instead of the whole axis: each reduction over ``j`` becomes a
partial reduction over ``R``, each use of the axis extent becomes ``|R|``,
and every tensor in between keeps its own role. The tensors this produces
are candidate reducer states:

* the reductions over ``j`` (``row_max``, ``den``, ``PV``; ``total``, ``ss``);
* the ``j``-free tensors computed from them that later ``j`` reductions read
  back -- the *contexts* (``mean``);
* the count ``|R|`` when the extent is used as a value.

Subsets of these go through the same derivation as grammar states (leaf,
identity, merge, laws, cover, relevance). What partialisation adds is the
*printed* merge. The canonical merge is an equivalence judge, not a recipe:
it cancels the max shift of a softmax and expands the centred second moment
into ``Σx² - (Σx)²/n``. The printed merge instead keeps each side's own
context and corrects it, exactly as the original computes it:

    Σ_{A u B} f(x, κ) = Σ_A f(x, κ_A + δ_A) + Σ_B f(x, κ_B + δ_B),
    δ_X = κ(A u B) - κ(X)       (or κ(A u B) / κ(X) for a scaling context)

``δ`` is pulled out of each partial reduction by the canonical rules, and
what remains under every power of ``δ`` has to be a plain state value of
that side (``Σ_A (x - κ_A)² = M2_A``, ``Σ_A (x - κ_A) = 0``, ``|A| = n_A``).
For the softmax this gives ``exp(c (m_A - m)) · l_A``, for the variance
``M2_A + n_A (mean - mean_A)²``. The printed merge is proven equal to the
canonical one before it is used.
"""

from __future__ import annotations

import dataclasses
import itertools

from tvm import te
from tvm import tirx as tir

from ..dims import DimKey
from ..symbolic import ir
from ..symbolic.canonicalize import (
    Unsupported,
    contains_node,
    instantiate,
    mk_add,
    mk_card,
    mk_mul,
    mk_reduce,
    rebuild,
    recanonicalize,
    subst,
    subst_domain,
)
from ..symbolic.lower import LowerCtx, classify_combiner
from . import verify
from .synth import (
    A_NAME,
    B_NAME,
    R_NAME,
    ReducerSpec,
    SynthesisProblem,
    _derive,
    _spec_key,
    collect_goals,
    solve_atoms,
    swap_sides,
)

PSEUDO = -2000  # tensor ids of candidate tensors inside structured definitions
SIDE_OFFSET = {"a": -10_000, "b": -20_000}  # the same, pinned to one side of a merge


@dataclasses.dataclass
class _Cand:
    name: str
    pid: int
    rank: int
    kind: str  # "sum" / "max" (a reduction over j), "value" (a context), "count"
    state: ir.SymExpr  # canonical partial definition over R, target coordinates
    struct: ir.SymExpr  # the same with candidate reads opaque (see _StructLower)
    index: tuple[ir.IndexExpr, ...]  # target coordinates of its own axes


# ---------------------------------------------------------------------------
# reading the original program
# ---------------------------------------------------------------------------
def _compute_ops(output: te.Tensor) -> list:
    order: list = []

    def visit(op) -> None:
        if not isinstance(op, te.ComputeOp) or any(op.same_as(o) for o in order):
            return
        for t in op.input_tensors:
            visit(t.op)
        order.append(op)

    visit(output.op)
    return order


def _index(ops: list, op) -> int:
    return next(n for n, o in enumerate(ops) if o.same_as(op))


def _candidate_ops(output: te.Tensor, dims, axis: DimKey) -> list[tuple[object, str]]:
    """``(op, kind)`` for the reductions over ``axis`` and the contexts between them."""
    ops = _compute_ops(output)
    n = len(ops)
    kind: list[str | None] = [None] * n
    for k, op in enumerate(ops):
        first = op.body[0]
        if (
            len(op.body) == 1
            and isinstance(first, tir.Reduce)
            and len(first.axis) == 1
            and dims.key(first.axis[0].dom.extent) == axis
        ):
            kind[k] = classify_combiner(first.combiner)
    inputs = [
        [_index(ops, t.op) for t in op.input_tensors if isinstance(t.op, te.ComputeOp)]
        for op in ops
    ]
    after = [set() for _ in range(n)]  # ops reading op k, transitively
    for k in reversed(range(n)):
        for i in inputs[k]:
            after[i] |= {k} | after[k]
    before = [set() for _ in range(n)]
    for k in range(n):
        for i in inputs[k]:
            before[k] |= {i} | before[i]
    out = []
    for k, op in enumerate(ops):
        if kind[k] is not None:
            out.append((op, kind[k]))
            continue
        j_free = all(dims.key(iv.dom.extent) != axis for iv in op.axis)
        feeds = any(kind[x] is not None for x in after[k])
        fed = any(kind[x] is not None for x in before[k])
        plain = not isinstance(op.body[0], tir.Reduce)
        if len(op.body) == 1 and plain and j_free and feeds and fed:
            out.append((op, "value"))
    return out


class _StructLower(LowerCtx):
    """Lower with candidate tensors kept opaque.

    A candidate read inside a ``j`` scope (the reduction over ``j``, or a
    tensor with a ``j`` axis) gets the ``j`` index appended, so that the
    canonical rules cannot pull it out of the reduction it sits in: the
    read is *pinned* there, where the merge will re-base it.
    """

    def __init__(self, dims, axis: DimKey, cands: list) -> None:
        super().__init__(dims)
        self.axis = axis
        self.cands = cands  # list of te ops, index -> PSEUDO - index
        self._jnames: dict[str, bool] = {}

    def pid(self, op) -> int | None:
        for n, c in enumerate(self.cands):
            if c.same_as(op):
                return PSEUDO - n
        return None

    def lower(self, t):
        op = t.op
        if not isinstance(op, te.ComputeOp):
            return super().lower(t)
        saved = self._jnames
        self._jnames = {iv.var.name: self.dims.key(iv.dom.extent) == self.axis for iv in op.axis}
        try:
            return super().lower(t)
        finally:
            self._jnames = saved

    def lower_reduce(self, e, env, level):
        saved = dict(self._jnames)
        for ax in e.axis:
            self._jnames[ax.var.name] = self.dims.key(ax.dom.extent) == self.axis
        try:
            return super().lower_reduce(e, env, level)
        finally:
            self._jnames = saved

    def lower_expr(self, e, env, level):
        if isinstance(e, tir.ProducerLoad):
            pid = self.pid(e.producer.op)
            if pid is not None:
                idx = [self.lower_index(i, env) for i in e.indices]
                pins = [env[nm] for nm, is_j in self._jnames.items() if is_j and nm in env]
                return ir.elem(pid, (*idx, *pins[-1:]))
        return super().lower_expr(e, env, level)

    def own(self, op) -> ir.SymExpr:
        """The op's body over its own indices ``i0 ..`` (candidate reads opaque)."""
        saved = self._jnames
        self._jnames = {iv.var.name: self.dims.key(iv.dom.extent) == self.axis for iv in op.axis}
        try:
            env = {iv.var.name: ir.idx(f"i{n}") for n, iv in enumerate(op.axis)}
            return self.lower_expr(op.body[0], env, 0)
        finally:
            self._jnames = saved


def _coordinates(keys, index_axes: dict[str, DimKey]) -> tuple[ir.Idx, ...] | None:
    """Target index names for axes with extents ``keys`` (outputs first, then nested)."""
    names = sorted(index_axes, key=lambda nm: (not nm.startswith("i"), nm))
    taken: set[str] = set()
    out = []
    for k in keys:
        pick = next((nm for nm in names if index_axes[nm] == k and nm not in taken), None)
        if pick is None:
            return None
        taken.add(pick)
        out.append(ir.idx(pick))
    return tuple(out)


def _partial(e: ir.SymExpr, full: ir.DFull, R: ir.DSym) -> ir.SymExpr:
    """Run on ``R`` instead of the whole axis: domains and the extent-as-value."""
    e = subst_domain(e, {full: R})
    ext = ir.extent_of(full.axis)
    if isinstance(ext, ir.ShapeSym):
        e = subst(e, {ext: mk_card(R)})
    return e


def candidates(problem: SynthesisProblem, ctx) -> list[_Cand]:
    axis = problem.axis
    full, R = ir.dfull(axis), ir.dsym(axis, R_NAME)
    ops = _candidate_ops(ctx.output, ctx.dims, axis)
    if not ops:
        return []
    _, _, index_axes = collect_goals(problem.target, axis)
    for t, key in enumerate(problem.target_keys):
        index_axes.setdefault(f"i{t}", key)
    lower = _StructLower(ctx.dims, axis, [op for op, _ in ops])
    for t in ctx.lower.placeholders:
        lower.tensor_id(t)
    out: list[_Cand] = []
    uses_count = False
    for n, (op, kind) in enumerate(ops):
        keys = tuple(ctx.dims.key(iv.dom.extent) for iv in op.axis)
        coords = _coordinates(keys, index_axes)
        if coords is None:
            continue
        imap = {ir.idx(f"i{p}"): c for p, c in enumerate(coords)}
        try:
            sem = ctx.lower.lower(op.output(0))
            state = _partial(subst(sem.body, imap), full, R)
            struct = _partial(subst(lower.own(op), imap), full, R)
        except Unsupported:
            continue
        uses_count |= contains_node(state, lambda x: isinstance(x, ir.Card))
        out.append(_Cand(op.name, PSEUDO - n, len(keys), kind, state, struct, coords))
    if uses_count:
        card = mk_card(R)
        out.append(_Cand("count", PSEUDO - len(ops), 0, "count", card, card, ()))
    return out


# ---------------------------------------------------------------------------
# re-based merge printing
# ---------------------------------------------------------------------------
class _Printer:
    """Numerically faithful merges for one derived state set."""

    def __init__(self, spec: ReducerSpec, cands: list[_Cand], chosen: list[_Cand], sol) -> None:
        self.spec = spec
        self.by_pid = {c.pid: c for c in cands}
        self.chosen = chosen
        axis = spec.axis
        self.R = ir.dsym(axis, R_NAME)
        self.sides = {"a": ir.dsym(axis, A_NAME), "b": ir.dsym(axis, B_NAME)}
        self.sol = sol
        self.atom_map = {}
        for atom, expr in sol.items():
            for side, dom in self.sides.items():
                self.atom_map[subst_domain(atom, {self.R: dom})] = swap_sides(expr, "a", side)
        self.merged: dict[int, ir.SymExpr | None] = {}  # pid -> raw merged value

    # -- values of a candidate on one side / merged ------------------------
    def slot(self, cand: _Cand) -> int | None:
        for k, c in enumerate(self.chosen):
            if c.pid == cand.pid:
                return k
        for k, st in enumerate(self.spec.states):  # same function under another name
            if st is cand.state:
                return k
        return None

    def side_value(self, cand: _Cand, side: str) -> ir.SymExpr | None:
        """``κ(X)`` over the state variables of side ``X``."""
        k = self.slot(cand)
        if k is not None:
            return ir.state_var(side, k)
        v = subst(subst_domain(cand.state, {self.R: self.sides[side]}), self.atom_map)
        return None if self._partial_left(v) else v

    def merged_value(self, cand: _Cand) -> ir.SymExpr | None:
        """``κ(A u B)``: a merged slot ``m_k`` or this candidate's own printed merge."""
        k = self.slot(cand)
        if k is not None:
            return ir.state_var("m", k)
        return self.print(cand)

    def _partial_left(self, e: ir.Node) -> bool:
        return contains_node(
            e,
            lambda x: isinstance(x, ir.Reduce | ir.Card) and isinstance(x.domain, ir.DSym),
        )

    # -- printing ---------------------------------------------------------
    def print(self, cand: _Cand) -> ir.SymExpr | None:
        if cand.pid in self.merged:
            return self.merged[cand.pid]
        self.merged[cand.pid] = None  # guards cycles
        try:
            out = self._reduction(cand) if cand.kind != "value" else self._value(cand)
        except Unsupported:
            out = None
        self.merged[cand.pid] = out
        return out

    def _value(self, cand: _Cand) -> ir.SymExpr | None:
        """A context: its own expression over the merged values of what it reads."""
        mapping: dict[ir.Node, ir.SymExpr] = {}
        for n in _walk(cand.struct):
            if isinstance(n, ir.Elem) and n.tensor in self.by_pid:
                src = self.by_pid[n.tensor]
                if tuple(n.indices) != src.index:
                    return None  # read at other indices: not a state of this reducer
                v = self.merged_value(src)
                if v is None:
                    return None
                mapping[n] = v
            elif isinstance(n, ir.Card) and n.domain is self.R:
                count = next((c for c in self.by_pid.values() if c.kind == "count"), None)
                v = self.merged_value(count) if count is not None else None
                if v is None:
                    return None
                mapping[n] = v
            elif isinstance(n, ir.Reduce):
                return None  # a context must be a plain function of other tensors
        return ir.raw_subst(cand.struct, mapping)

    def _reduction(self, cand: _Cand) -> ir.SymExpr | None:
        contexts = sorted({n.tensor for n in _walk(cand.struct) if _pinned(n, self.by_pid)})
        for modes in itertools.product(("shift", "scale"), repeat=len(contexts)):
            out = self._rebased(cand, dict(zip(contexts, modes)))
            if out is not None:
                return out
        return None

    def _rebased(self, cand: _Cand, modes: dict[int, str]) -> ir.SymExpr | None:
        union = ir.dunion(self.sides["a"], self.sides["b"])
        corr: dict[ir.Atom, tuple[str, int, str]] = {}  # atom -> (side, pid, mode)

        def per_side(n: ir.Node, depth: int):
            if not isinstance(n, ir.Elem) or not _pinned(n, self.by_pid):
                return None
            side = self._side_of(depth_domain.get(depth))
            if side is None:
                raise Unsupported("context outside a partial reduction")
            mode = modes[n.tensor]
            atom = ir.atom(f"{'d' if mode == 'shift' else 'r'}{side}{-n.tensor}")
            corr[atom] = (side, n.tensor, mode)
            pinned = ir.elem(n.tensor + SIDE_OFFSET[side], n.indices)
            return mk_add(pinned, atom) if mode == "shift" else mk_mul(pinned, atom)

        depth_domain: dict[int, ir.Domain] = {}
        merged = subst_domain(cand.struct, {self.R: union})
        merged = _map_elems(merged, per_side, depth_domain)
        return self._group(merged, corr)

    def _side_of(self, domain) -> str | None:
        for side, dom in self.sides.items():
            if domain is dom:
                return side
        return None

    def _group(self, e: ir.SymExpr, corr) -> ir.SymExpr | None:
        """Print ``e`` as Σ correction(δ) · (a state value of that side)."""
        if isinstance(e, ir.Max):
            args = [self._group(a, corr) for a in e.args]
            return None if any(a is None for a in args) else ir.raw_max(args)
        groups: dict[tuple[str | None, ir.SymExpr], list[ir.SymExpr]] = {}
        for term in e.args if isinstance(e, ir.Add) else (e,):
            factors = term.args if isinstance(term, ir.Mul) else (term,)
            key = [f for f in factors if contains_node(f, lambda x: x in corr)]
            rest = [f for f in factors if not contains_node(f, lambda x: x in corr)]
            if any(self._partial_left(f) for f in key):
                return None  # a correction stuck inside a partial reduction
            sides = {corr[x][0] for f in key for x in _walk(f) if x in corr}
            sides |= {s for f in rest for s in self._sides_in(f)}
            if len(sides) > 1:
                return None
            side = next(iter(sides), None)
            groups.setdefault((side, mk_mul(*key)), []).append(mk_mul(*rest))
        printed = []
        for (side, key), rests in groups.items():
            value = self._state_value(mk_add(*rests), side)
            if value is None:
                return None
            if value is ir.ZERO:
                continue
            printed.append(_raw_product(self._print_key(key, corr), value))
        if not printed:
            return ir.ZERO
        return printed[0] if len(printed) == 1 else ir.raw_add(printed)

    def _sides_in(self, e: ir.Node) -> set[str]:
        out = set()
        for n in _walk(e):
            if isinstance(n, ir.Reduce | ir.Card):
                s = self._side_of(n.domain)
                if s is not None:
                    out.add(s)
        return out

    def _state_value(self, rest: ir.SymExpr, side: str | None) -> ir.SymExpr | None:
        """``rest`` with each pinned context replaced by its value: a state monomial."""
        if side is None:
            return rest if isinstance(rest, ir.Const) else None
        dom = self.sides[side]

        def actual(n: ir.Node, depth: int):
            if not isinstance(n, ir.Elem) or n.tensor - SIDE_OFFSET[side] not in self.by_pid:
                return None
            src = self.by_pid[n.tensor - SIDE_OFFSET[side]]
            value = subst_domain(src.state, {self.R: dom})
            # the definition is over the target coordinates of its own axes
            imap = {c: n.indices[p] for p, c in enumerate(src.index)}
            return instantiate(value, imap, depth)

        v = subst(_map_elems(rest, actual, {}), self.atom_map)
        if self._partial_left(v) or not _monomial(v):
            return None
        return v

    def _print_key(self, key: ir.SymExpr, corr) -> ir.SymExpr:
        mapping: dict[ir.Node, ir.SymExpr] = {}
        for n in _walk(key):
            if isinstance(n, ir.Pow) and n.base in corr and n.exponent == -1:
                side, pid, mode = corr[n.base]
                if mode == "scale":  # 1 / rho = k(X) / k(A u B), finite where k(X) = 0
                    here, there = self._context(pid, side)
                    mapping[n] = ir.raw_mul([here, ir.raw_pow(there, -1)])
        for atom, (side, pid, mode) in corr.items():
            here, there = self._context(pid, side)
            if mode == "shift":
                mapping[atom] = ir.raw_add([there, ir.raw_mul([ir.MINUS_ONE, here])])
            else:
                mapping[atom] = ir.raw_mul([there, ir.raw_pow(here, -1)])
        return ir.raw_subst(key, mapping)

    def _context(self, pid: int, side: str) -> tuple[ir.SymExpr, ir.SymExpr]:
        cand = self.by_pid[pid]
        here, there = self.side_value(cand, side), self.merged_value(cand)
        if here is None or there is None:
            raise Unsupported(f"context {cand.name} is not expressible in the states")
        return here, there


def _pinned(n: ir.Node, by_pid: dict) -> bool:
    return (
        isinstance(n, ir.Elem)
        and n.tensor in by_pid
        and len(n.indices) == by_pid[n.tensor].rank + 1
    )


def _walk(e: ir.Node):
    stack = [e]
    seen = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        yield n
        stack.extend(n.children())


def _map_elems(e: ir.Node, fn, depth_domain: dict, depth: int = 0) -> ir.Node:
    """Rewrite tensor elements with ``fn(elem, depth)``; records each binder's domain."""
    if isinstance(e, ir.Elem):
        out = fn(e, depth)
        return e if out is None else out
    if isinstance(e, ir.Reduce):
        depth_domain[e.level + 1] = e.domain
        body = _map_elems(e.body, fn, depth_domain, e.level + 1)
        return e if body is e.body else mk_reduce(e.kind, e.domain, e.level, body)
    if not e.children() or isinstance(e, ir.Domain):
        return e
    kids = {}
    for c in e.children():
        nc = _map_elems(c, fn, depth_domain, depth)
        if nc is not c:
            kids[c] = nc
    return rebuild(e, kids) if kids else e


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
        refs = {n for n in _walk(prints[k]) if isinstance(n, ir.StateVar) and n.side == "m"}
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


def _proves(printed: ir.SymExpr, canonical: ir.SymExpr) -> bool:
    got = recanonicalize(printed)
    if got is canonical:
        return True
    return verify.equal_modulo_max(got, canonical) or verify._numeric_equal(got, canonical)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def synthesize_partial(problem: SynthesisProblem, ctx, stats: dict | None = None):
    """Reducer specs whose states are partial versions of the original's tensors."""
    if ctx.output is None:
        return []
    cands = candidates(problem, ctx)
    axis = problem.axis
    full, R = ir.dfull(axis), ir.dsym(axis, R_NAME)
    A, B = ir.dsym(axis, A_NAME), ir.dsym(axis, B_NAME)
    _, occurrences, index_axes = collect_goals(problem.target, axis)
    for t, key in enumerate(problem.target_keys):
        index_axes.setdefault(f"i{t}", key)
    specs: list[ReducerSpec] = []
    seen: set = set()
    lo, hi = max(1, problem.min_states), problem.max_states
    for size in range(lo, hi + 1):
        for chosen in itertools.combinations(cands, size):
            states = tuple(c.state for c in chosen)
            if len(set(states)) != len(states):
                continue
            spec = _derive(states, problem, full, R, A, B, occurrences, index_axes, stats)
            if spec is None:
                continue
            key = _spec_key(spec)
            if key in seen:
                continue
            seen.add(key)
            printer = _Printer(spec, cands, list(chosen), solve_atoms(states, R))
            prints = [printer.print(c) for c in chosen]
            expanded = _expand(prints) if all(p is not None for p in prints) else None
            if expanded is not None and all(_proves(p, m) for p, m in zip(expanded, spec.merge)):
                spec = dataclasses.replace(spec, merge_print=tuple(expanded))
                _count(stats, "reducer:partial_printed")
            spec = dataclasses.replace(spec, origin="partial")
            _count(stats, "reducer:partial_specs")
            specs.append(spec)
    return specs


def _count(stats, what, n=1):
    if stats is not None:
        stats[what] = stats.get(what, 0) + n


__all__ = ["candidates", "synthesize_partial"]
