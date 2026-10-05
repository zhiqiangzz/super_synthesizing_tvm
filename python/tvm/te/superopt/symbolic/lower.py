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
"""Lower a ``te.Tensor`` to its symbolic (real-number) semantics.

The result is a :class:`TensorSem` whose body is fully inlined down to the
input placeholders, so that syntactically different TE programs computing the
same function canonicalise to the same node whenever the rewrite theory in
:mod:`canonicalize` can see it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from fractions import Fraction

from tvm import te
from tvm import tirx as tir
from tvm.ir import Call

from ..dims import DimKey, DimTable
from . import ir
from .canonicalize import (
    Unsupported,
    instantiate,
    mk_add,
    mk_div,
    mk_exp,
    mk_log,
    mk_max,
    mk_monoid,
    mk_mul,
    mk_neg,
    mk_reduce,
    mk_sqrt,
    mk_sub,
)


@dataclasses.dataclass(frozen=True)
class TensorSem:
    """Semantics of one tensor: ``body`` over free indices ``i0 .. i{rank-1}``."""

    rank: int
    body: ir.SymExpr
    axis_keys: tuple[DimKey, ...]
    dtype: str

    @property
    def index_vars(self) -> tuple[ir.Idx, ...]:
        return tuple(ir.idx(f"i{k}") for k in range(self.rank))


def dtype_str(t: te.Tensor) -> str:
    return str(t.dtype)


class LowerCtx:
    """Shared state: the extent table, placeholder ids and an optional reducer decoder."""

    def __init__(self, dims: DimTable | None = None) -> None:
        self.dims = dims if dims is not None else DimTable()
        self._placeholders: list[tuple[te.Tensor, int]] = []
        self._memo: dict[tuple[int, int], TensorSem] = {}
        self._ops: list = []
        # Called on an opaque MonoidReduce; may return closed forms per slot.
        self.closed_form: Callable[[ir.MonoidReduce], ir.SymExpr | None] | None = None
        # Ops built by the search, whose semantics are known by construction.
        self._known: list[tuple[object, tuple[TensorSem, ...]]] = []

    def register(self, op, sems: tuple[TensorSem, ...]) -> None:
        """Record the semantics of every output of a freshly built op."""
        self._known.append((op, tuple(sems)))

    # -- placeholder identity -------------------------------------------------
    def tensor_id(self, t: te.Tensor) -> int:
        for known, tid in self._placeholders:
            if known.op.same_as(t.op) and known.value_index == t.value_index:
                return tid
        tid = len(self._placeholders)
        self._placeholders.append((t, tid))
        return tid

    def placeholder(self, tid: int) -> te.Tensor:
        return self._placeholders[tid][0]

    @property
    def placeholders(self) -> list[te.Tensor]:
        return [t for t, _ in self._placeholders]

    def _op_index(self, op) -> int:
        for i, known in enumerate(self._ops):
            if known.same_as(op):
                return i
        self._ops.append(op)
        return len(self._ops) - 1

    # -- lowering -------------------------------------------------------------
    def lower(self, t: te.Tensor) -> TensorSem:
        key = (self._op_index(t.op), int(t.value_index))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        op = t.op
        axis_keys = self.dims.keys(t.shape)
        for known, sems in self._known:
            if known.same_as(op):
                sem = sems[int(t.value_index)]
                self._memo[key] = sem
                return sem
        if isinstance(op, te.PlaceholderOp):
            tid = self.tensor_id(t)
            body = ir.elem(tid, tuple(ir.idx(f"i{k}") for k in range(len(t.shape))))
            sem = TensorSem(len(t.shape), body, axis_keys, dtype_str(t))
        elif isinstance(op, te.ComputeOp):
            env = {iv.var.name: ir.idx(f"i{k}") for k, iv in enumerate(op.axis)}
            if len(env) != len(op.axis):
                raise Unsupported("duplicate axis names in compute")
            body = self.lower_expr(op.body[int(t.value_index)], env, 0)
            sem = TensorSem(len(op.axis), body, axis_keys, dtype_str(t))
        else:
            raise Unsupported(f"operation {type(op).__name__}")
        self._memo[key] = sem
        return sem

    def lower_expr(self, e, env: dict[str, ir.IndexExpr], level: int) -> ir.SymExpr:
        """``env`` maps loop-variable names to index expressions; ``level`` = binder depth."""
        if isinstance(e, tir.FloatImm | tir.IntImm):
            return lower_imm(e)
        if isinstance(e, tir.Var):
            name = e.name
            if name in env:
                raise Unsupported("loop variable used as a real value")
            return ir.shape_sym(name)
        if isinstance(e, tir.Cast):
            return self.lower_expr(e.value, env, level)
        if isinstance(e, tir.Add):
            return mk_add(self.lower_expr(e.a, env, level), self.lower_expr(e.b, env, level))
        if isinstance(e, tir.Sub):
            return mk_sub(self.lower_expr(e.a, env, level), self.lower_expr(e.b, env, level))
        if isinstance(e, tir.Mul):
            return mk_mul(self.lower_expr(e.a, env, level), self.lower_expr(e.b, env, level))
        if isinstance(e, tir.Div):
            return mk_div(self.lower_expr(e.a, env, level), self.lower_expr(e.b, env, level))
        if isinstance(e, tir.Max):
            return mk_max(self.lower_expr(e.a, env, level), self.lower_expr(e.b, env, level))
        if isinstance(e, tir.Min):
            a = self.lower_expr(e.a, env, level)
            b = self.lower_expr(e.b, env, level)
            return mk_neg(mk_max(mk_neg(a), mk_neg(b)))
        if isinstance(e, Call):
            name = e.op.name
            if name == "tirx.exp":
                return mk_exp(self.lower_expr(e.args[0], env, level))
            if name == "tirx.sqrt":
                return mk_sqrt(self.lower_expr(e.args[0], env, level))
            if name == "tirx.log":
                return mk_log(self.lower_expr(e.args[0], env, level))
            if name == "tirx.rsqrt":
                return mk_div(ir.ONE, mk_sqrt(self.lower_expr(e.args[0], env, level)))
            raise Unsupported(f"intrinsic {name}")
        if isinstance(e, tir.ProducerLoad):
            producer = e.producer
            sem = self.lower(producer)
            index_map = {ir.idx(f"i{k}"): self.lower_index(i, env) for k, i in enumerate(e.indices)}
            return instantiate(sem.body, index_map, level)
        if isinstance(e, tir.Reduce):
            return self.lower_reduce(e, env, level)
        raise Unsupported(f"expression {type(e).__name__}")

    def lower_index(self, e, env: dict[str, ir.IndexExpr]) -> ir.IndexExpr:
        if isinstance(e, tir.Var):
            if e.name not in env:
                raise Unsupported(f"free index variable {e.name}")
            return env[e.name]
        if isinstance(e, tir.IntImm):
            return ir.iconst(int(e.value))
        raise Unsupported(f"non-trivial index expression {e}")

    def lower_reduce(self, e, env: dict[str, ir.IndexExpr], level: int) -> ir.SymExpr:
        cond = e.condition
        if not (isinstance(cond, tir.IntImm) and int(cond.value) == 1):
            raise Unsupported("conditional reduction")
        if len(e.init) != 0:
            raise Unsupported("reduction with explicit init")
        axes = list(e.axis)
        inner_env = dict(env)
        for n, ax in enumerate(axes):
            if ax.var.name in inner_env:  # loop variables are resolved by name
                raise Unsupported(f"reduction axis {ax.var.name!r} shadows another loop variable")
            inner_env[ax.var.name] = ir.bidx(level + n)
        depth = level + len(axes)
        kind = classify_combiner(e.combiner)
        if kind is not None:
            body = self.lower_expr(e.source[0], inner_env, depth)
            for n in reversed(range(len(axes))):
                dom = ir.dfull(self.dims.key(axes[n].dom.extent))
                body = mk_reduce(kind, dom, level + n, body)
            return body
        if len(axes) != 1:
            raise Unsupported("tuple reduction over several axes")
        comb = e.combiner
        leaf = tuple(self.lower_expr(s, inner_env, depth) for s in e.source)
        menv: dict[str, ir.IndexExpr] = {}
        merge_map = {}
        for k, (lhs, rhs) in enumerate(zip(comb.lhs, comb.rhs)):
            merge_map[lhs.name] = ir.state_var("a", k)
            merge_map[rhs.name] = ir.state_var("b", k)
        merge = tuple(self._lower_merge(x, merge_map, menv) for x in comb.result)
        identity = tuple(self.lower_expr(x, {}, 0) for x in comb.identity_element)
        dom = ir.dfull(self.dims.key(axes[0].dom.extent))
        node = mk_monoid(dom, level, leaf, merge, identity, int(e.value_index))
        if self.closed_form is not None and isinstance(node, ir.MonoidReduce):
            closed = self.closed_form(node)
            if closed is not None:
                return closed
        return node

    def _lower_merge(self, e, merge_map: dict[str, ir.SymExpr], env) -> ir.SymExpr:
        if isinstance(e, tir.Var) and e.name in merge_map:
            return merge_map[e.name]
        if isinstance(e, tir.Cast):
            return self._lower_merge(e.value, merge_map, env)
        binary = {
            tir.Add: mk_add,
            tir.Sub: mk_sub,
            tir.Mul: mk_mul,
            tir.Div: mk_div,
            tir.Max: mk_max,
        }
        for cls, fn in binary.items():
            if isinstance(e, cls):
                return fn(
                    self._lower_merge(e.a, merge_map, env), self._lower_merge(e.b, merge_map, env)
                )
        if isinstance(e, Call) and e.op.name == "tirx.exp":
            return mk_exp(self._lower_merge(e.args[0], merge_map, env))
        if isinstance(e, Call) and e.op.name == "tirx.sqrt":
            return mk_sqrt(self._lower_merge(e.args[0], merge_map, env))
        if isinstance(e, Call) and e.op.name == "tirx.log":
            return mk_log(self._lower_merge(e.args[0], merge_map, env))
        return self.lower_expr(e, env, 0)


def lower_imm(e) -> ir.Const:
    """Immediates; the dtype's lowest/highest finite value stands for ``-inf``/``+inf``."""
    v = e.value
    if isinstance(e, tir.FloatImm):
        if v == float("-inf") or _is_min_value(e):
            return ir.NEG_INF_C
        if v == float("inf") or _is_max_value(e):
            return ir.POS_INF_C
    return ir.const(Fraction(v))


def classify_combiner(comb) -> str | None:
    """``"sum"`` / ``"max"`` for the standard single-slot reducers, else ``None``."""
    if len(comb.result) != 1:
        return None
    lhs, rhs, res, ident = comb.lhs[0], comb.rhs[0], comb.result[0], comb.identity_element[0]
    if isinstance(res, tir.Add) and _is_pair(res, lhs, rhs) and _imm_value(ident) == 0:
        return "sum"
    if isinstance(res, tir.Max) and _is_pair(res, lhs, rhs) and _is_min_value(ident):
        return "max"
    return None


def _is_pair(res, lhs, rhs) -> bool:
    return (res.a.same_as(lhs) and res.b.same_as(rhs)) or (
        res.a.same_as(rhs) and res.b.same_as(lhs)
    )


def _imm_value(e):
    if isinstance(e, tir.FloatImm | tir.IntImm):
        return e.value
    return None


def _is_min_value(e) -> bool:
    v = _imm_value(e)
    if v is None:
        return False
    if v == float("-inf"):
        return True
    try:
        lowest = tir.min_value(str(e.ty.dtype)).value
    except Exception:  # pragma: no cover - defensive
        return False
    return v == lowest


def _is_max_value(e) -> bool:
    v = _imm_value(e)
    if v is None:
        return False
    if v == float("inf"):
        return True
    try:
        highest = tir.max_value(str(e.ty.dtype)).value
    except Exception:  # pragma: no cover - defensive
        return False
    return v == highest


def lower_tensor(t: te.Tensor, ctx: LowerCtx | None = None) -> TensorSem:
    return (ctx or LowerCtx()).lower(t)
