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
"""Reduction chains: reductions over one axis that depend on each other's results.

A *chain* is a set of ``sum``/``max`` reductions over the same axis ``j`` in
which a later one reads the final value of an earlier one from a context that
does not move with ``j``::

    mx[i]  = max_j x[i, j]
    den[i] = sum_j exp(x[i, j] - mx[i])        # reads mx[i]: one more pass over j

Such a program walks ``j`` once per link. Everything this module does is to
read that structure off the TE graph; nothing is rewritten here.

members
    The reductions, plus the *values* between them: tensors without a ``j``
    axis computed from members and read back by one (``mean = total / n``).
    Which tensor an axis belongs to is decided by how it is indexed, never by
    its extent, so a square matrix is no special case. A value the program
    does not name -- ``total[i] / n`` written inside the reduction that uses
    it -- is a member all the same (a *context*).
boundary
    What the members read from outside: placeholders and reductions over other
    axes (``S = Q Kᵀ``, a cost matrix). These stay opaque. Elementwise tensors
    on the way are looked through at the place they are read.
stages
    A reduction that reads another one *along* ``j``, or through a reduction
    over some other axis, cannot share its pass: it starts a later chain and
    sees the earlier one as part of its boundary.
coordinates
    The members of a chain live on a common set of indices (the row of a
    softmax; batch, head and query of an attention), found by following how
    they are indexed where they are read together.
required
    The members the rest of the program still reads once the chain is fused.
"""

from __future__ import annotations

import dataclasses

from tvm import te
from tvm import tirx as tir
from tvm.tirx.stmt_functor import post_order_visit

from ..dims import DimKey, DimTable
from ..symbolic import ir
from ..symbolic.canonicalize import Unsupported, mk_card, subst, subst_domain
from ..symbolic.lower import LowerCtx, classify_combiner, compute_ops
from .derive import domains

PSEUDO = -2000  # tensor ids of chain members inside structured definitions
CONTEXT = -3000  # and of the contexts that have no tensor of their own


@dataclasses.dataclass(eq=False)
class Member:
    """A tensor of the original program that lives on a chain, or a value the
    program computes in place without naming it (``op`` is ``None``)."""

    op: object
    kind: str  # "sum" / "max" (a reduction over j) or "value"
    coords: tuple[int, ...]  # chain coordinate of each of its axes
    pid: int = 0
    # Own definition over the chain coordinates, reads of other members kept as
    # opaque elements (pinned inside the reduction they sit in, see ChainLower).
    body: ir.SymExpr | None = None
    label: str = ""  # the name of a member without a tensor

    @property
    def name(self) -> str:
        return self.label if self.op is None else self.op.name

    @property
    def tensor(self) -> te.Tensor:
        return self.op.output(0)

    @property
    def index(self) -> tuple[ir.Idx, ...]:
        return tuple(ir.idx(f"i{c}") for c in self.coords)

    @property
    def is_reduction(self) -> bool:
        return self.kind != "value"


@dataclasses.dataclass(eq=False)
class Chain:
    axis: DimKey
    extent: object  # extent of the reduction axis
    jname: str
    stage: int
    members: list[Member]  # producers first
    coord_extents: tuple  # extent of each chain coordinate
    coord_names: tuple[str, ...]  # and the loop variable the program uses for it
    lower: ChainLower
    required: list[Member] = dataclasses.field(default_factory=list)

    @property
    def boundary(self) -> list[te.Tensor]:
        return self.lower.placeholders

    @property
    def reductions(self) -> list[Member]:
        return [m for m in self.members if m.is_reduction]

    def member(self, op) -> Member | None:
        return next((m for m in self.members if m.op is not None and m.op.same_as(op)), None)


@dataclasses.dataclass
class Skipped:
    """Reductions that depend on each other but are outside what can be fused."""

    members: list[str]
    reason: str


# ---------------------------------------------------------------------------
# lowering one chain
# ---------------------------------------------------------------------------
class ChainLower(LowerCtx):
    """Lower the members of one chain over its coordinates.

    * A member read where no index moves with ``j`` is a *context*: it stays an
      opaque element. Inside a reduction it gets the ``j`` index appended, so
      that the canonical rules cannot pull it out: the read is *pinned* where
      the merge will re-base it.
    * The elementwise tensors in ``through`` are looked through at the place
      they are read, with the indices they are read at, so what moves with
      ``j`` is decided there. These are the ones that read a member (the
      context has to become visible) and the ones some member reads at every
      loop variable it has: each element is used once, so nothing is computed
      twice and the tensor need not be stored (the score matrix of an
      attention). A tensor read at fewer indices is shared between iterations
      and stays materialised.
    * Anything else is boundary: an opaque element of a tensor from outside.
    * Inside a reduction, an expression that reads a member and does not move
      with ``j`` (``total[i] / n``, ``log(den[i])``) is what the reduction
      actually depends on: it becomes a context of its own, a member without
      a tensor, and is pinned like one.
    """

    def __init__(self, dims: DimTable, members: list[Member], through: list, axis) -> None:
        super().__init__(dims)
        self.members = members
        self.through = through
        # a constant extent is written into the program as a number: reading that
        # number as the extent is what lets ``total / 16`` be a mean
        ext = ir.extent_of(axis)
        self._extent = ext.value if isinstance(ext, ir.Const) and ext.value >= 2 else None
        self._extent_symbol = extent_symbol(axis)
        # elementwise tensors kept opaque although read at every loop variable in scope
        self.full: list = []
        # contexts found inside reductions, with the member whose body they were in
        self.contexts: list[tuple[Member, Member]] = []
        self._site: set[ir.IndexExpr] = set()
        self._current: Member | None = None

    def member(self, op) -> Member | None:
        return next((m for m in self.members if m.op is not None and m.op.same_as(op)), None)

    def lower_expr(self, e, env, level):
        if isinstance(e, tir.IntImm | tir.FloatImm) and e.value == self._extent:
            return self._extent_symbol
        if level and self._is_context(e, env):
            return self._context(e, env, level)
        return super().lower_expr(e, env, level)

    def _is_context(self, e, env) -> bool:
        """A compound expression that reads a member and no loop variable bound by a reduction."""
        inner = e
        while isinstance(inner, tir.Cast):
            inner = inner.value
        if isinstance(inner, tir.ProducerLoad | tir.Var | tir.IntImm | tir.FloatImm | tir.Reduce):
            return False
        found = {"member": False, "moves": False}

        def visit(x) -> None:
            if isinstance(x, tir.Var):
                found["moves"] |= x.name in env and not isinstance(env[x.name], ir.Idx)
            elif isinstance(x, tir.ProducerLoad):
                found["member"] |= self.member(x.producer.op) is not None
            elif isinstance(x, tir.Reduce):
                found["moves"] = True

        post_order_visit(e, visit)
        return found["member"] and not found["moves"]

    def _context(self, e, env, level: int) -> ir.SymExpr:
        names: list[str] = []
        post_order_visit(
            e, lambda x: names.append(x.name) if isinstance(x, tir.Var) and x.name in env else None
        )
        coords = tuple(sorted({int(env[n].name[1:]) for n in names}))
        saved = self._site
        self._site = {ir.idx(f"i{c}") for c in coords}
        try:
            body = super().lower_expr(e, env, 0)  # outside the reduction: reads are not pinned
        finally:
            self._site = saved
        m = next((c for _, c in self.contexts if c.body is body and c.coords == coords), None)
        if m is None:
            n = len(self.contexts)
            m = Member(None, "value", coords, CONTEXT - n, body, f"{self._current.name}.ctx{n}")
            self.contexts.append((self._current, m))
        return ir.elem(m.pid, (*m.index, ir.bidx(level - 1)))

    def lower_load(self, e, env, level):
        t = e.producer
        op = t.op
        idx = tuple(self.lower_index(i, env) for i in e.indices)
        along = any(isinstance(i, ir.BIdx) for i in idx)
        m = self.member(op)
        if m is not None and not along:
            return ir.elem(m.pid, (*idx, ir.bidx(level - 1)) if level else idx)
        if isinstance(op, te.ComputeOp) and is_elementwise(op):
            if any(op.same_as(x) for x in self.through):
                inner = {iv.var.name: i for iv, i in zip(op.axis, idx)}
                if len(inner) != len(op.axis):
                    raise Unsupported("duplicate axis names in compute")
                return self.lower_expr(op.body[0], inner, level)
            site = self._site | ({ir.bidx(level - 1)} if level else set())
            if site <= set(idx):
                self.full.append(op)
        return ir.elem(self.tensor_id(t), idx)

    def own(self, m: Member) -> ir.SymExpr:
        """The member's body over the chain coordinates."""
        env = {iv.var.name: ir.idx(f"i{c}") for iv, c in zip(m.op.axis, m.coords)}
        if len(env) != len(m.op.axis):
            raise Unsupported("duplicate axis names in compute")
        self._site = set(env.values())
        self._current = m
        return self.lower_expr(m.op.body[0], env, 0)


def extent_symbol(axis: DimKey) -> ir.ShapeSym:
    """The extent of ``axis`` as it appears in member bodies: the ``te.var`` itself,
    or a stand-in for the number a constant extent is written as."""
    ext = ir.extent_of(axis)
    return ext if isinstance(ext, ir.ShapeSym) else ir.shape_sym(f"|ax{axis}|")


def partial(e: ir.SymExpr, axis: DimKey, count: bool = True) -> ir.SymExpr:
    """Run on a sub-range ``R`` instead of the whole axis.

    The reductions over the axis become partial reductions. Where the program
    uses the extent as a value there are two readings, both the original
    program when ``R`` is the whole axis: the size of what was reduced
    (``total / n`` is the mean of ``R``; ``count``) or a constant of the
    problem (``a / K`` with ``K`` classes).
    """
    full, R, _, _ = domains(axis)
    e = subst_domain(e, {full: R})
    symbol = extent_symbol(axis)
    if count:
        return subst(e, {symbol: mk_card(R)})
    ext = ir.extent_of(axis)
    return e if symbol is ext else subst(e, {symbol: ext})


# ---------------------------------------------------------------------------
# reading the TE graph
# ---------------------------------------------------------------------------
def reduction_of(op):
    """``(kind, iter_var)`` for a plain ``sum``/``max`` reduction over a single axis."""
    if len(op.body) != 1:
        return None
    body = op.body[0]
    if not isinstance(body, tir.Reduce) or len(body.axis) != 1 or len(body.init) != 0:
        return None
    cond = body.condition
    if not (isinstance(cond, tir.IntImm) and int(cond.value) == 1):
        return None
    kind = classify_combiner(body.combiner)
    return None if kind is None else (kind, body.axis[0])


def is_elementwise(op) -> bool:
    return len(op.body) == 1 and not isinstance(op.body[0], tir.Reduce)


def loads_in(expr) -> list:
    out: list = []
    post_order_visit(expr, lambda x: out.append(x) if isinstance(x, tir.ProducerLoad) else None)
    return out


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class _Graph:
    """The compute ops behind the outputs and how the reductions among them read each other."""

    def __init__(self, outputs, dims: DimTable) -> None:
        self.dims = dims
        self.ops = compute_ops(outputs)
        n = len(self.ops)
        self.red = [reduction_of(op) for op in self.ops]
        self.plain = [is_elementwise(op) for op in self.ops]
        self.key = [None if r is None else dims.key(r[1].dom.extent) for r in self.red]
        self.inputs = [
            sorted({self.index(t.op) for t in op.input_tensors} - {None}) for op in self.ops
        ]
        self.deps: list[set[int]] = [set() for _ in range(n)]  # transitive producers
        for k in range(n):
            for i in self.inputs[k]:
                self.deps[k] |= {i} | self.deps[i]
        self.stage = [0] * n
        self.context: list[set[int]] = [
            set() for _ in range(n)
        ]  # same-stage members read free of j
        self.free: list[set[int]] = [set() for _ in range(n)]  # every producer read free of j
        for k in range(n):
            if self.red[k] is not None:
                self._classify(k)

    def index(self, op) -> int | None:
        return next((i for i, o in enumerate(self.ops) if o.same_as(op)), None)

    def _scan(self, k: int) -> tuple[set[int], set[int]]:
        """Producers reduction ``k`` reads free of / along its own reduction index;
        elementwise producers read along it are looked through."""
        free: set[int] = set()
        along: set[int] = set()

        def walk(expr, jvars: list) -> None:
            for load in loads_in(expr):
                i = self.index(load.producer.op)
                if i is None:
                    continue
                dep = [_uses(ix, jvars) for ix in load.indices]
                if not any(dep):
                    free.add(i)
                elif self.plain[i]:
                    op = self.ops[i]
                    walk(op.body[0], [iv.var for iv, d in zip(op.axis, dep) if d])
                else:
                    along.add(i)

        body = self.ops[k].body[0]
        walk(body.source[0], [body.axis[0].var])
        return free, along

    def through(self, i: int, key: DimKey) -> tuple[set[int], set[int]]:
        """``(reductions over key, other producers)`` met from ``i`` through elementwise ops."""
        if self.red[i] is not None and self.key[i] == key:
            return {i}, set()
        if not self.plain[i]:
            return set(), {i}
        same: set[int] = set()
        other: set[int] = set()
        for x in self.inputs[i]:
            s, o = self.through(x, key)
            same |= s
            other |= o
        return same, other

    def _classify(self, k: int) -> None:
        key = self.key[k]
        free, along = self._scan(k)
        context: set[int] = set()
        behind: set[int] = set()  # producers whose own producers must be complete first
        for f in free:
            same, other = self.through(f, key)
            context |= same
            behind |= other
        behind |= along
        earlier = {x for b in behind for x in (self.deps[b] | {b})}
        earlier = {x for x in earlier if self.red[x] is not None and self.key[x] == key}
        self.stage[k] = max(
            [self.stage[a] for a in context] + [self.stage[a] + 1 for a in earlier] + [0]
        )
        self.context[k] = {a for a in context if self.stage[a] == self.stage[k]}
        self.free[k] = free

    # -- coordinates ----------------------------------------------------------
    def coordinates(self, group: list[int]) -> _UnionFind:
        """Axes that are the same index, among the tensors computed from ``group``.

        Two axes are joined when one tensor is read at the other's loop variable.
        Tensors that owe nothing to the group are data: reading the same data
        twice (``X[a, j] * X[b, j]``) says nothing about ``a`` and ``b``.
        """
        gset = set(group)
        inside = [i in gset or bool(self.deps[i] & gset) for i in range(len(self.ops))]
        uf = _UnionFind()
        for d, op in enumerate(self.ops):
            if not inside[d]:
                continue
            for body in op.body:
                for load in loads_in(body):
                    t = self.index(load.producer.op)
                    if t is None or not inside[t]:
                        continue
                    for p, ix in enumerate(load.indices):
                        for q, iv in enumerate(op.axis):
                            if iv.var.same_as(ix):
                                uf.union((t, p), (d, q))
        return uf


def _uses(expr, variables: list) -> bool:
    """Does an index expression mention one of ``variables``?"""
    if isinstance(expr, tir.IntImm):
        return False
    hit = []
    post_order_visit(
        expr,
        lambda x: (
            hit.append(x)
            if isinstance(x, tir.Var) and any(x.same_as(v) for v in variables)
            else None
        ),
    )
    return bool(hit)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def discover_chains(outputs, dims: DimTable | None = None) -> tuple[list[Chain], list[Skipped]]:
    """The reduction chains of the program behind ``outputs``, earliest stage first."""
    outputs = [outputs] if isinstance(outputs, te.Tensor) else list(outputs)
    dims = dims if dims is not None else DimTable()
    g = _Graph(outputs, dims)
    reds = [k for k in range(len(g.ops)) if g.red[k] is not None]
    chains: list[Chain] = []
    skipped: list[Skipped] = []
    for key, stage in sorted({(g.key[k], g.stage[k]) for k in reds}):
        group = [k for k in reds if g.key[k] == key and g.stage[k] == stage]
        coords = g.coordinates(group)
        comp = _UnionFind()
        for k in group:
            for a in g.context[k]:
                comp.union(k, a)
        # reductions with nothing between them still share a pass when the
        # program reads them at the same index
        rank = {k: len(g.ops[k].axis) for k in group}
        for a in group:
            for b in group:
                if a < b and {coords.find((a, p)) for p in range(rank[a])} & {
                    coords.find((b, p)) for p in range(rank[b])
                }:
                    comp.union(a, b)
        for root in sorted({comp.find(k) for k in group}):
            part = [k for k in group if comp.find(k) == root]
            if len(part) < 2 or not any(g.context[k] & set(part) for k in part):
                continue  # a lone reduction, or independent ones: no chain to fuse
            try:
                chains.append(_build(g, part, key, stage, coords))
            except Unsupported as err:
                skipped.append(Skipped([g.ops[k].name for k in part], str(err)))
    _mark_required(g, chains, outputs)
    chains = [c for c in chains if c.required]
    chains.sort(key=lambda c: (c.stage, g.index(c.members[0].op)))
    return chains, skipped


def _build(g: _Graph, part: list[int], key: DimKey, stage: int, coords: _UnionFind) -> Chain:
    pset = set(part)
    values: set[int] = set()
    for k in part:
        for f in g.free[k]:
            if g.plain[f] and g.through(f, key)[0] & pset:
                values.add(f)
    cands = sorted(pset | values)
    classes = {i: [coords.find((i, p)) for p in range(len(g.ops[i].axis))] for i in cands}
    for i in cands:
        twice = {c for c in classes[i] if classes[i].count(c) > 1}
        if twice:
            shared = [g.ops[x].name for x in cands if x != i and set(classes[x]) & twice]
            raise Unsupported(
                f"{', '.join(shared) or 'a tensor'}: read at two different indices of "
                f"{g.ops[i].name}, which would take two states of one reducer for one tensor"
            )
    every = {c for i in cands for c in classes[i]}
    spanning = [k for k in part if set(classes[k]) == every]
    if not spanning:
        shapes = ", ".join(f"{g.ops[k].name}{list(g.ops[k].axis)}" for k in part)
        raise Unsupported(f"no reduction spans the indices of all the others ({shapes})")
    lead = spanning[-1]
    position = {c: n for n, c in enumerate(classes[lead])}
    lead_op = g.ops[lead]
    extents = tuple(iv.dom.extent for iv in lead_op.axis)
    members = []
    for n, i in enumerate(cands):
        op = g.ops[i]
        where = tuple(position[c] for c in classes[i])
        for iv, c in zip(op.axis, where):
            if g.dims.key(iv.dom.extent) != g.dims.key(extents[c]):
                raise Unsupported(f"{op.name} and {lead_op.name} disagree on an extent")
        kind = g.red[i][0] if g.red[i] is not None else "value"
        members.append(Member(op, kind, where, PSEUDO - n))
    through = [g.ops[i] for i in range(len(g.ops)) if g.plain[i] and g.through(i, key)[0] & pset]
    while True:  # grows by the tensors found to be read once per element
        lower = ChainLower(g.dims, members, through, key)
        for m in members:
            m.body = lower.own(m)
        more = [op for op in lower.full if not any(op.same_as(x) for x in through)]
        if not more:
            break
        through.append(more[0])
    ordered: list[Member] = []  # a context goes before the member it was found in
    for m in members:
        ordered.extend(c for owner, c in lower.contexts if owner is m)
        ordered.append(m)
    members = ordered
    jvar = g.red[part[0]][1]
    names = tuple(iv.var.name for iv in lead_op.axis)
    return Chain(key, jvar.dom.extent, jvar.var.name, stage, members, extents, names, lower)


def _mark_required(g: _Graph, chains: list[Chain], outputs) -> None:
    """Members read by what is left of the program once every chain is fused."""
    seen: list = []
    entered: list[Chain] = []

    def visit(t: te.Tensor) -> None:
        op = t.op
        if any(op.same_as(o) for o in seen):
            return
        seen.append(op)
        for c in chains:
            m = c.member(op)
            if m is None:
                continue
            c.required.append(m)
            if not any(c is e for e in entered):  # the fused reducer reads the boundary
                entered.append(c)
                for b in c.boundary:
                    visit(b)
            return
        if isinstance(op, te.ComputeOp):
            for x in op.input_tensors:
                visit(x)

    for t in outputs:
        visit(t)
    for c in chains:
        c.required.sort(key=lambda m: g.index(m.op))


__all__ = [
    "Chain",
    "ChainLower",
    "Member",
    "Skipped",
    "discover_chains",
    "extent_symbol",
    "is_elementwise",
    "partial",
    "reduction_of",
]
