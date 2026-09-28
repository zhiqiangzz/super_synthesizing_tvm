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
"""The ``comm_reduce`` search operator: synthesise, realise and build.

``params`` of this op are :class:`Realized` reducer specs: a synthesised
:class:`ReducerSpec` whose every leaf has been matched against the *leaf
grammar* (operand elements, constants, ``*`` and optionally ``exp``) so it
can be emitted as a ``tirx.PrimExpr`` inside a real ``te.comm_reducer``.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable

from tvm import te
from tvm import tirx as tir

from ..dims import DimKey
from ..pool import PoolEntry
from ..symbolic import ir
from ..symbolic.canonicalize import instantiate, mk_exp, mk_mul, subst
from ..symbolic.lower import TensorSem
from ..symbolic.realize import RealizeEnv, const_expr, to_prim
from ..target import SearchCtx
from ..tensor_ops import OpSpec, load, shape_of
from .synth import ReducerSpec, SynthesisProblem, synthesize

_SPEC_IDS = itertools.count()


@dataclasses.dataclass(frozen=True)
class Layout:
    axis: DimKey
    out_keys: tuple[DimKey, ...]
    maps: tuple[tuple[int | None, ...], ...]  # per operand, per axis: out pos or None (= j)


@dataclasses.dataclass(frozen=True, eq=False)
class Realized:
    spec: ReducerSpec
    layout: Layout
    leaf_trees: tuple[tuple, ...]
    closed: tuple[ir.SymExpr, ...]  # spec.closed over the layout's output indices
    uid: int = dataclasses.field(default_factory=lambda: next(_SPEC_IDS))

    def __repr__(self) -> str:
        return f"Realized#{self.uid}"


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
def layouts(operands: list[PoolEntry], axis: DimKey, index_keys) -> Iterable[Layout]:
    """Ways to feed ``operands`` to a reducer over ``axis``.

    Every operand must carry the axis; its other axes become output axes,
    shared between operands when their extents match. Output axes are ordered
    like ``index_keys`` (the reducer's index space, matched greedily by
    extent) so the outputs meet downstream consumers without a transpose.
    """
    choices = []
    for e in operands:
        pos = [p for p, k in enumerate(e.shape) if k == axis]
        if not pos:
            return
        choices.append(pos)
    for jpos in itertools.product(*choices):
        out_keys: list[DimKey] = []
        maps = []
        for e, jp in zip(operands, jpos):
            claimed: set[int] = set()
            m: list[int | None] = []
            for p, k in enumerate(e.shape):
                if p == jp:
                    m.append(None)
                    continue
                slot = None
                for o, ok in enumerate(out_keys):
                    if ok == k and o not in claimed:
                        slot = o
                        break
                if slot is None:
                    slot = len(out_keys)
                    out_keys.append(k)
                claimed.add(slot)
                m.append(slot)
            maps.append(tuple(m))
        order: list[int] = []
        for k in index_keys:
            for o, ok in enumerate(out_keys):
                if ok == k and o not in order:
                    order.append(o)
                    break
        order.extend(o for o in range(len(out_keys)) if o not in order)
        new_pos = {o: n for n, o in enumerate(order)}
        out_keys = [out_keys[o] for o in order]
        maps = [tuple(None if o is None else new_pos[o] for o in m) for m in maps]
        yield Layout(axis, tuple(out_keys), tuple(maps))


def index_remap(spec: ReducerSpec, out_keys) -> dict[ir.Idx, ir.Idx] | None:
    """Map the spec's free indices onto output positions with matching extents."""
    claimed: set[int] = set()
    remap: dict[ir.Idx, ir.Idx] = {}
    for name, key in spec.index_space:
        for o, ok in enumerate(out_keys):
            if ok == key and o not in claimed:
                claimed.add(o)
                remap[ir.idx(name)] = ir.idx(f"i{o}")
                break
        else:
            return None
    return remap


def operand_elements(operands: list[PoolEntry], layout: Layout) -> list[ir.SymExpr]:
    out = []
    for e, m in zip(operands, layout.maps):
        imap = {
            ir.idx(f"i{p}"): (ir.bidx(0) if o is None else ir.idx(f"i{o}")) for p, o in enumerate(m)
        }
        out.append(instantiate(e.sem.body, imap, 1))
    return out


# ---------------------------------------------------------------------------
# leaf grammar
# ---------------------------------------------------------------------------
def leaf_table(
    elems: list[ir.SymExpr], consts: list[ir.SymExpr], max_nodes: int, allow_exp: bool
) -> dict[ir.SymExpr, tuple]:
    """All leaf-grammar expressions up to ``max_nodes`` nodes, keyed by canonical form."""
    by_size: dict[int, dict[ir.SymExpr, tuple]] = {1: {}}
    table: dict[ir.SymExpr, tuple] = {}

    def put(size: int, e: ir.SymExpr, tree: tuple) -> None:
        if e not in table:
            table[e] = tree
            by_size.setdefault(size, {})[e] = tree

    for i, e in enumerate(elems):
        put(1, e, ("elem", i))
    for c in [ir.ZERO, ir.ONE, *consts]:
        put(1, c, ("const", c))
    for size in range(2, max_nodes + 1):
        for sa in range(1, size - 1):
            sb = size - 1 - sa
            if sb < sa:
                continue
            for ea, ta in list(by_size.get(sa, {}).items()):
                for eb, tb in list(by_size.get(sb, {}).items()):
                    put(size, mk_mul(ea, eb), ("mul", ta, tb))
        if allow_exp:
            for ea, ta in list(by_size.get(size - 1, {}).items()):
                put(size, mk_exp(ea), ("exp", ta))
    return table


def realize_tree(tree: tuple, elem_fn, dtype: str, dims):
    kind = tree[0]
    if kind == "elem":
        return elem_fn(tree[1])
    if kind == "const":
        return to_prim(tree[1], RealizeEnv(dtype=dtype, dims=dims))
    if kind == "mul":
        return realize_tree(tree[1], elem_fn, dtype, dims) * realize_tree(
            tree[2], elem_fn, dtype, dims
        )
    if kind == "exp":
        return tir.exp(realize_tree(tree[1], elem_fn, dtype, dims))
    raise ValueError(tree)


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------
class CommReduce(OpSpec):
    name = "comm_reduce"
    rank = 11
    arity = 0  # variable
    max_arity = 3
    result_directed = True

    def __init__(self) -> None:
        self._cache: dict[tuple, list[Realized]] = {}
        self._synth_cache: dict[tuple, list[ReducerSpec]] = {}

    def arities(self, prog) -> Iterable[int]:
        return range(1, self.max_arity + 1)  # operand count, not state count

    def params(self, operands, ctx: SearchCtx):
        self.max_arity = ctx.bounds.max_states
        for axis in sorted(ctx.target.reduce_axes):
            specs = self._specs(axis, ctx)
            if not specs:
                continue
            index_keys = list(dict.fromkeys(k for sp in specs for _, k in sp.index_space))
            for layout in layouts(operands, axis, index_keys):
                elems = operand_elements(operands, layout)
                key = (ctx.target.body, axis, tuple(elems), layout.out_keys, layout.maps)
                hit = self._cache.get(key)
                if hit is None:
                    hit = self._realize(operands, layout, elems, specs, ctx)
                    self._cache[key] = hit
                yield from hit

    def _specs(self, axis: DimKey, ctx: SearchCtx) -> list[ReducerSpec]:
        b = ctx.bounds
        skey = (
            ctx.target.body,
            axis,
            b.min_states,
            b.max_states,
            b.max_merge_nodes,
            b.max_leaf_nodes,
            b.max_state_expr_nodes,
            b.max_latent_atoms,
        )
        specs = self._synth_cache.get(skey)
        if specs is None:
            problem = SynthesisProblem(
                target=ctx.target.body,
                axis=axis,
                max_states=ctx.bounds.max_states,
                max_merge_nodes=ctx.bounds.max_merge_nodes,
                target_keys=ctx.target.sem.axis_keys,
                min_states=b.min_states,
                consts=tuple(ctx.target.consts),
                max_leaf_nodes=b.max_leaf_nodes,
                max_state_expr_nodes=b.max_state_expr_nodes,
                max_latent_atoms=b.max_latent_atoms,
            )
            specs = synthesize(problem, ctx.stats)
            self._synth_cache[skey] = specs
        return specs

    def _realize(self, operands, layout: Layout, elems, specs, ctx: SearchCtx) -> list[Realized]:
        table = leaf_table(
            elems, list(ctx.target.consts), ctx.bounds.max_leaf_nodes, ctx.bounds.leaf_exp
        )
        out = []
        for spec in specs:
            remap = index_remap(spec, layout.out_keys)
            if remap is None:
                ctx.count("reducer:index_mismatch")
                continue
            trees = []
            for leaf in spec.leaves:
                t = table.get(subst(leaf, remap))
                if t is None:
                    break
                trees.append(t)
            else:
                # every operand must actually be read by some leaf
                used = set()
                for t in trees:
                    _collect_elems(t, used)
                if used != set(range(len(operands))):
                    ctx.count("reducer:unused_operand")
                    continue
                closed = tuple(subst(c, remap) for c in spec.closed)
                out.append(Realized(spec, layout, tuple(trees), closed))
                continue
            ctx.count("reducer:leaf_unrealizable")
        return out

    def param_key(self, params: Realized):
        return (params.uid,)

    def apply_sem(self, operands, params: Realized, ctx):
        keys = params.layout.out_keys
        return [TensorSem(len(keys), cf, keys, ctx.dtype) for cf in params.closed]

    def build(self, tensors, params: Realized, ctx):
        spec, layout = params.spec, params.layout
        accum = ctx.dtype
        n = spec.arity
        j = te.reduce_axis((0, ctx.dims.extent(layout.axis)), name="j")
        merge_env_a = [ir.state_var("a", k) for k in range(n)]
        merge_env_b = [ir.state_var("b", k) for k in range(n)]

        def fcombine(a, b):
            env = RealizeEnv(dtype=accum, dims=ctx.dims)
            for k in range(n):
                env.state[merge_env_a[k]] = a[k]
                env.state[merge_env_b[k]] = b[k]
            memo: dict = {}
            return tuple(to_prim(spec.merge[k], env, memo) for k in range(n))

        def fidentity(*dtypes):
            return tuple(const_expr(spec.identity[k].value, str(dtypes[k])) for k in range(n))

        reducer = te.comm_reducer(fcombine, fidentity, name="synth")

        def fcompute(*idx):
            def elem_fn(i):
                src = [j if o is None else idx[o] for o in layout.maps[i]]
                return load(tensors[i], src)

            leaves = tuple(realize_tree(t, elem_fn, accum, ctx.dims) for t in params.leaf_trees)
            return reducer(leaves, axis=j)

        outs = te.compute(shape_of(layout.out_keys, ctx), fcompute, name="cr")
        outs = list(outs) if isinstance(outs, tuple) else [outs]
        ctx.lower.register(outs[0].op, tuple(self.apply_sem(None, params, ctx)))
        return outs

    def describe(self, params: Realized) -> str:
        return params.spec.pretty()

    def emit(self, names, entries, outs, params: Realized, ctx, n):
        from ..emit import compute_source, index_names, load_source, shape_source
        from ..symbolic.emit import SourceEnv, const_source, hoisted_source

        spec, layout = params.spec, params.layout
        accum = ctx.dtype
        k = spec.arity
        env = SourceEnv(dtype=accum, dims=ctx.dims)
        for s in range(k):
            env.state[ir.state_var("a", s)] = f"a[{s}]"
            env.state[ir.state_var("b", s)] = f"b[{s}]"
        hoisted, merged = hoisted_source(spec.merge, env, prefix="m")
        lines = [f"def merge{n}(a, b):"]
        lines.extend(f"    {h}" for h in hoisted)
        lines.append("    return (" + ", ".join(merged) + ("," if k == 1 else "") + ")")
        dts = ", ".join(f"t{s}" for s in range(k))
        ident = ", ".join(
            const_source(spec.identity[s].value, f"t{s}", quoted=False) for s in range(k)
        )
        lines.append(f"def identity{n}({dts}):")
        lines.append(f"    return ({ident}{',' if k == 1 else ''})")
        lines.append(f'reducer{n} = te.comm_reducer(merge{n}, identity{n}, name="synth")')
        j = f"j{n}"
        lines.append(f'{j} = te.reduce_axis((0, {ctx.dims.name(layout.axis)}), name="{j}")')
        idx = index_names(len(layout.out_keys))

        def elem_fn(i):
            src = [j if o is None else idx[o] for o in layout.maps[i]]
            return load_source(names[i], src)

        leaves = [_tree_source(t, elem_fn, accum, ctx.dims) for t in params.leaf_trees]
        body = f"reducer{n}(({', '.join(leaves)}{',' if k == 1 else ''}), axis={j})"
        target = ", ".join(outs) if k > 1 else outs[0]
        lines.append(compute_source(target, shape_source(layout.out_keys, ctx), idx, body, "cr"))
        return lines


def _tree_source(tree: tuple, elem_fn, dtype: str, dims) -> str:
    from ..symbolic.emit import SourceEnv, to_source

    kind = tree[0]
    if kind == "elem":
        return elem_fn(tree[1])
    if kind == "const":
        return to_source(tree[1], SourceEnv(dtype=dtype, dims=dims))
    if kind == "mul":
        lhs = _tree_source(tree[1], elem_fn, dtype, dims)
        rhs = _tree_source(tree[2], elem_fn, dtype, dims)
        return f"{lhs} * {rhs}"
    if kind == "exp":
        return f"tir.exp({_tree_source(tree[1], elem_fn, dtype, dims)})"
    raise ValueError(tree)


def _collect_elems(tree: tuple, out: set) -> None:
    if tree[0] == "elem":
        out.add(tree[1])
    elif tree[0] == "mul":
        _collect_elems(tree[1], out)
        _collect_elems(tree[2], out)
    elif tree[0] == "exp":
        _collect_elems(tree[1], out)
