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
"""Tensor-level operators of the search grammar.

Each :class:`OpSpec` knows (a) which parametrisations are *legal* for given
operands (shape / dtype / axis constraints -- the first pruning layer), (b)
the symbolic semantics of its outputs, computed without touching TE, and (c)
how to build the real ``te.compute`` once a program is materialised. Tests
check that (b) and (c) agree.
"""

from __future__ import annotations

from collections.abc import Iterable

from tvm import te

from .dims import DimKey
from .emit import compute_source, index_names, load_source, shape_source
from .pool import PoolEntry
from .symbolic import ir
from .symbolic.canonicalize import (
    instantiate,
    mk_add,
    mk_div,
    mk_exp,
    mk_mul,
    mk_reduce,
    mk_sqrt,
    mk_sub,
)
from .symbolic.emit import SourceEnv, to_source
from .symbolic.lower import TensorSem
from .symbolic.realize import RealizeEnv, to_prim
from .target import SearchCtx


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load(t: te.Tensor, indices):
    return t(*indices)


def shape_of(keys: tuple[DimKey, ...], ctx: SearchCtx) -> tuple:
    return tuple(ctx.dims.extent(k) for k in keys)


def _out_idx(rank: int) -> tuple[ir.Idx, ...]:
    return tuple(ir.idx(f"i{k}") for k in range(rank))


def subseq_matchings(small: tuple, big: tuple) -> Iterable[tuple[int, ...]]:
    """Order-preserving injective maps of ``small`` axes into ``big`` axes with equal keys."""

    def rec(i: int, start: int, acc: list[int]):
        if i == len(small):
            yield tuple(acc)
            return
        for j in range(start, len(big)):
            if big[j] == small[i] and len(big) - j >= len(small) - i:
                acc.append(j)
                yield from rec(i + 1, j + 1, acc)
                acc.pop()

    yield from rec(0, 0, [])


class OpSpec:
    name = "op"
    rank = 0
    arity = 1
    commutative = False
    distinct = False  # require distinct operands (for sub/div)

    def params(self, operands: list[PoolEntry], ctx: SearchCtx) -> Iterable:
        raise NotImplementedError

    def param_key(self, params) -> tuple:
        return (repr(params),)

    def apply_sem(self, operands: list[PoolEntry], params, ctx: SearchCtx) -> list[TensorSem]:
        raise NotImplementedError

    def build(self, tensors: list[te.Tensor], params, ctx: SearchCtx) -> list[te.Tensor]:
        raise NotImplementedError

    def describe(self, params) -> str:
        return repr(params)

    def emit(self, names, entries, outs, params, ctx, n) -> list[str]:
        """Source lines defining ``outs`` from operand variables ``names`` (see emit.py)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
class Matmul(OpSpec):
    """``out = Σ_k A[..k..] * B[..k..]`` with greedy batch-axis matching."""

    name = "matmul"
    rank = 0
    arity = 2
    commutative = True

    def params(self, operands, ctx):
        a, b = operands
        if a.dtype != b.dtype:
            return
        for p, ka in enumerate(a.shape):
            if ka not in ctx.target.reduce_axes:
                continue
            for q, kb in enumerate(b.shape):
                if kb != ka:
                    continue
                batch = []
                used_b: set[int] = set()
                for pa, key in enumerate(a.shape):
                    if pa == p:
                        continue
                    for pb, keyb in enumerate(b.shape):
                        if pb != q and pb not in used_b and keyb == key:
                            batch.append((pa, pb))
                            used_b.add(pb)
                            break
                yield (p, q, tuple(batch))

    def param_key(self, params):
        p, q, batch = params
        return (p, q, batch)

    def layout(self, a: TensorSem, b: TensorSem, params, ctx: SearchCtx):
        """Output axis keys and per-operand axis -> output position maps.

        Output axes are ordered like the target's (matched greedily by extent),
        so that a matmul whose operands happen to sit in "reversed" pool order
        still produces the target's layout; unmatched axes follow.
        """
        p, q, batch = params
        batch_a = {pa: n for n, (pa, _) in enumerate(batch)}
        batch_b = {pb: n for n, (_, pb) in enumerate(batch)}
        keys = [a.axis_keys[pa] for pa, _ in batch]
        map_a: dict[int, int | None] = {}
        map_b: dict[int, int | None] = {}
        for pa in range(a.rank):
            if pa == p:
                map_a[pa] = None
            elif pa in batch_a:
                map_a[pa] = batch_a[pa]
            else:
                map_a[pa] = len(keys)
                keys.append(a.axis_keys[pa])
        for pb in range(b.rank):
            if pb == q:
                map_b[pb] = None
            elif pb in batch_b:
                map_b[pb] = batch_b[pb]
            else:
                map_b[pb] = len(keys)
                keys.append(b.axis_keys[pb])
        order: list[int] = []
        for k in ctx.target.sem.axis_keys:
            for o, ok in enumerate(keys):
                if ok == k and o not in order:
                    order.append(o)
                    break
        order.extend(o for o in range(len(keys)) if o not in order)
        new_pos = {o: n for n, o in enumerate(order)}
        keys = [keys[o] for o in order]
        map_a = {pa: (None if o is None else new_pos[o]) for pa, o in map_a.items()}
        map_b = {pb: (None if o is None else new_pos[o]) for pb, o in map_b.items()}
        return tuple(keys), map_a, map_b, a.axis_keys[p]

    def apply_sem(self, operands, params, ctx):
        a, b = operands[0].sem, operands[1].sem
        keys, map_a, map_b, kkey = self.layout(a, b, params, ctx)
        ia = {
            ir.idx(f"i{pa}"): (ir.bidx(0) if o is None else ir.idx(f"i{o}"))
            for pa, o in map_a.items()
        }
        ib = {
            ir.idx(f"i{pb}"): (ir.bidx(0) if o is None else ir.idx(f"i{o}"))
            for pb, o in map_b.items()
        }
        body = mk_reduce(
            "sum", ir.dfull(kkey), 0, mk_mul(instantiate(a.body, ia, 1), instantiate(b.body, ib, 1))
        )
        return [TensorSem(len(keys), body, keys, a.dtype)]

    def build(self, tensors, params, ctx):
        A, B = tensors
        a, b = ctx.lower.lower(A), ctx.lower.lower(B)
        keys, map_a, map_b, kkey = self.layout(a, b, params, ctx)
        k = te.reduce_axis((0, ctx.dims.extent(kkey)), name="k")

        def fcompute(*idx):
            ia = [k if o is None else idx[o] for _, o in sorted(map_a.items())]
            ib = [k if o is None else idx[o] for _, o in sorted(map_b.items())]
            return te.sum(load(A, ia) * load(B, ib), axis=k)

        return [te.compute(shape_of(keys, ctx), fcompute, name="mm")]

    def describe(self, params):
        p, q, batch = params
        return f"contract a{p}~b{q}, batch {list(batch)}"

    def emit(self, names, entries, outs, params, ctx, n):
        a, b = entries[0].sem, entries[1].sem
        keys, map_a, map_b, kkey = self.layout(a, b, params, ctx)
        idx = index_names(len(keys))
        k = f"k{n}"
        ia = [k if o is None else idx[o] for _, o in sorted(map_a.items())]
        ib = [k if o is None else idx[o] for _, o in sorted(map_b.items())]
        body = f"te.sum({load_source(names[0], ia)} * {load_source(names[1], ib)}, axis={k})"
        return [
            f'{k} = te.reduce_axis((0, {ctx.dims.name(kkey)}), name="{k}")',
            compute_source(outs[0], shape_source(keys, ctx), idx, body, "mm"),
        ]


# ---------------------------------------------------------------------------
# elementwise binary
# ---------------------------------------------------------------------------
class Ewise(OpSpec):
    """Elementwise op with broadcast-by-axis-dropping (``[b,h,i,j] / [b,h,i]``)."""

    arity = 2
    sym = None
    tir_op = None

    def __init__(self, name: str, rank: int, sym, tir_op, commutative: bool):
        self.name = name
        self.rank = rank
        self.sym = staticmethod(sym)
        self.tir_op = staticmethod(tir_op)
        self.commutative = commutative
        # sub/div of a tensor with itself is trivial; add(x, x) is a scale.
        self.distinct = name != "mul"

    def params(self, operands, ctx):
        a, b = operands
        if a.dtype != b.dtype:
            return
        if a.rank == b.rank:
            if a.shape == b.shape:
                yield (0, tuple(range(a.rank)))
            return
        big, small, which = (a, b, 0) if a.rank > b.rank else (b, a, 1)
        for m in subseq_matchings(small.shape, big.shape):
            yield (which, m)

    def param_key(self, params):
        return params

    def apply_sem(self, operands, params, ctx):
        which, mapping = params
        a, b = operands[0].sem, operands[1].sem
        big, small = (a, b) if which == 0 else (b, a)
        imap = {ir.idx(f"i{k}"): ir.idx(f"i{m}") for k, m in enumerate(mapping)}
        small_body = instantiate(small.body, imap, 0)
        lhs, rhs = (big.body, small_body) if which == 0 else (small_body, big.body)
        return [TensorSem(big.rank, self.sym(lhs, rhs), big.axis_keys, big.dtype)]

    def build(self, tensors, params, ctx):
        which, mapping = params
        A, B = tensors
        big, small = (A, B) if which == 0 else (B, A)
        keys = ctx.dims.keys(big.shape)

        def fcompute(*idx):
            vb = big(*idx)
            vs = small(*[idx[m] for m in mapping])
            return self.tir_op(vb, vs) if which == 0 else self.tir_op(vs, vb)

        return [te.compute(shape_of(keys, ctx), fcompute, name=self.name)]

    def describe(self, params):
        which, mapping = params
        return f"broadcast operand {1 - which} via {list(mapping)}"

    def emit(self, names, entries, outs, params, ctx, n):
        which, mapping = params
        big, small = (entries[0], entries[1]) if which == 0 else (entries[1], entries[0])
        big_name, small_name = (names[0], names[1]) if which == 0 else (names[1], names[0])
        idx = index_names(big.rank)
        vb = f"{big_name}[{', '.join(idx)}]"
        vs = f"{small_name}[{', '.join(idx[m] for m in mapping)}]"
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.name]
        body = f"{vb} {op} {vs}" if which == 0 else f"{vs} {op} {vb}"
        return [compute_source(outs[0], shape_source(big.shape, ctx), idx, body, self.name)]


# ---------------------------------------------------------------------------
# scale by a constant from the constant pool
# ---------------------------------------------------------------------------
class Scale(OpSpec):
    name = "scale"
    rank = 5
    arity = 1

    def params(self, operands, ctx):
        yield from ctx.target.consts

    def param_key(self, params):
        return (params.uid,)

    def apply_sem(self, operands, params, ctx):
        a = operands[0].sem
        return [TensorSem(a.rank, mk_mul(params, a.body), a.axis_keys, a.dtype)]

    def build(self, tensors, params, ctx):
        A = tensors[0]
        dtype = str(A.dtype)
        keys = ctx.dims.keys(A.shape)
        c = to_prim(params, RealizeEnv(dtype=dtype, dims=ctx.dims))
        return [te.compute(shape_of(keys, ctx), lambda *idx: A(*idx) * c, name="scale")]

    def describe(self, params):
        return f"by {params}"

    def emit(self, names, entries, outs, params, ctx, n):
        a = entries[0]
        idx = index_names(a.rank)
        c = to_source(params, SourceEnv(dtype=a.dtype, dims=ctx.dims))
        body = f"{names[0]}[{', '.join(idx)}] * {c}"
        return [compute_source(outs[0], shape_source(a.shape, ctx), idx, body, "scale")]


# ---------------------------------------------------------------------------
# unary / cast
# ---------------------------------------------------------------------------
class Unary(OpSpec):
    arity = 1

    def __init__(self, name: str, rank: int, sym, tir_op):
        self.name = name
        self.rank = rank
        self.sym = staticmethod(sym)
        self.tir_op = staticmethod(tir_op)

    def params(self, operands, ctx):
        yield None

    def param_key(self, params):
        return ()

    def apply_sem(self, operands, params, ctx):
        a = operands[0].sem
        return [TensorSem(a.rank, self.sym(a.body), a.axis_keys, a.dtype)]

    def build(self, tensors, params, ctx):
        A = tensors[0]
        keys = ctx.dims.keys(A.shape)
        return [te.compute(shape_of(keys, ctx), lambda *idx: self.tir_op(A(*idx)), name=self.name)]

    def describe(self, params):
        return ""

    def emit(self, names, entries, outs, params, ctx, n):
        a = entries[0]
        idx = index_names(a.rank)
        body = f"tir.{self.name}({names[0]}[{', '.join(idx)}])"
        return [compute_source(outs[0], shape_source(a.shape, ctx), idx, body, self.name)]


# ---------------------------------------------------------------------------
# single reductions
# ---------------------------------------------------------------------------
class ReduceOp(OpSpec):
    arity = 1

    def __init__(self, kind: str, rank: int):
        self.kind = kind
        self.name = kind
        self.rank = rank

    def params(self, operands, ctx):
        a = operands[0]
        for p, key in enumerate(a.shape):
            if key in ctx.target.reduce_axes:
                yield p

    def param_key(self, params):
        return (params,)

    def apply_sem(self, operands, params, ctx):
        a = operands[0].sem
        p = params
        keys = tuple(k for i, k in enumerate(a.axis_keys) if i != p)
        imap = {}
        o = 0
        for i in range(a.rank):
            if i == p:
                imap[ir.idx(f"i{i}")] = ir.bidx(0)
            else:
                imap[ir.idx(f"i{i}")] = ir.idx(f"i{o}")
                o += 1
        body = mk_reduce(self.kind, ir.dfull(a.axis_keys[p]), 0, instantiate(a.body, imap, 1))
        return [TensorSem(a.rank - 1, body, keys, a.dtype)]

    def build(self, tensors, params, ctx):
        A = tensors[0]
        p = params
        akeys = ctx.dims.keys(A.shape)
        keys = tuple(k for i, k in enumerate(akeys) if i != p)
        r = te.reduce_axis((0, ctx.dims.extent(akeys[p])), name="r")
        fn = te.sum if self.kind == "sum" else te.max

        def fcompute(*idx):
            src = []
            o = 0
            for i in range(len(akeys)):
                if i == p:
                    src.append(r)
                else:
                    src.append(idx[o])
                    o += 1
            return fn(A(*src), axis=r)

        return [te.compute(shape_of(keys, ctx), fcompute, name=self.name)]

    def describe(self, params):
        return f"axis {params}"

    def emit(self, names, entries, outs, params, ctx, n):
        a = entries[0]
        p = params
        keys = tuple(k for i, k in enumerate(a.shape) if i != p)
        idx = index_names(a.rank - 1)
        r = f"r{n}"
        src = []
        o = 0
        for i in range(a.rank):
            if i == p:
                src.append(r)
            else:
                src.append(idx[o])
                o += 1
        body = f"te.{self.kind}({names[0]}[{', '.join(src)}], axis={r})"
        return [
            f'{r} = te.reduce_axis((0, {ctx.dims.name(a.shape[p])}), name="{r}")',
            compute_source(outs[0], shape_source(keys, ctx), idx, body, self.name),
        ]


def _tir_add(a, b):
    return a + b


def _tir_sub(a, b):
    return a - b


def _tir_mul(a, b):
    return a * b


def _tir_div(a, b):
    return a / b


def _tir_exp(a):
    from tvm import tirx as tir

    return tir.exp(a)


def _tir_sqrt(a):
    from tvm import tirx as tir

    return tir.sqrt(a)


def default_specs() -> list[OpSpec]:
    return [
        Matmul(),
        Ewise("add", 1, mk_add, _tir_add, True),
        Ewise("sub", 2, mk_sub, _tir_sub, False),
        Ewise("mul", 3, mk_mul, _tir_mul, True),
        Ewise("div", 4, mk_div, _tir_div, False),
        Scale(),
        Unary("exp", 6, mk_exp, _tir_exp),
        Unary("sqrt", 7, mk_sqrt, _tir_sqrt),
        ReduceOp("sum", 9),
        ReduceOp("max", 10),
    ]
