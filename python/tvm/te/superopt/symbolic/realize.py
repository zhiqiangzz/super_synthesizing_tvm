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
"""Turn scalar symbolic expressions back into ``tirx.PrimExpr``.

Used for the inputs of a reducer, its merge function and identity, and the
epilogues formed from its states. Shared sub-expressions are realised once
per call (the memo acts as SSA/CSE), and :func:`node_count` reports the size
of that SSA form.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from fractions import Fraction

from tvm import tirx as tir

from ..dims import DimTable
from . import ir
from .canonicalize import Unsupported


@dataclasses.dataclass
class RealizeEnv:
    dtype: str
    dims: DimTable | None = None
    index: dict[ir.IndexExpr, object] = dataclasses.field(default_factory=dict)
    state: dict[ir.StateVar, object] = dataclasses.field(default_factory=dict)
    elem: Callable[[int, tuple], object] | None = None
    # symbolic scalars by name (``te.var`` used as a value); extents fall back to ``dims``
    scalars: dict[str, object] = dataclasses.field(default_factory=dict)


def const_expr(value: ir.Number, dtype: str):
    if isinstance(value, float):
        return tir.min_value(dtype) if value < 0 else tir.max_value(dtype)
    return tir.const(float(value), dtype)


def to_prim(e: ir.SymExpr, env: RealizeEnv, memo: dict | None = None):
    """Realise ``e``; ``memo`` (node -> PrimExpr) implements sharing."""
    if memo is None:
        memo = {}
    hit = memo.get(e)
    if hit is not None:
        return hit
    out = _realize(e, env, memo)
    memo[e] = out
    return out


def _realize(e: ir.SymExpr, env: RealizeEnv, memo: dict):
    dt = env.dtype
    if isinstance(e, ir.Const):
        return const_expr(e.value, dt)
    if isinstance(e, ir.ShapeSym):
        if e.name in env.scalars:
            return tir.convert(env.scalars[e.name]).astype(dt)
        if env.dims is None:
            raise Unsupported("shape symbol without a DimTable")
        ext = env.dims.extent(env.dims.key_of_name(e.name))
        return tir.convert(ext).astype(dt)
    if isinstance(e, ir.StateVar):
        return env.state[e]
    if isinstance(e, ir.Elem):
        if env.elem is None:
            raise Unsupported("tensor element without a loader")
        return env.elem(e.tensor, tuple(_realize_index(i, env) for i in e.indices))
    if isinstance(e, ir.Add):
        pos, neg = [], []
        for a in e.args:
            coeff, core = _coeff(a)
            if coeff < 0:
                neg.append(_scaled(-coeff, core, env, memo))
            else:
                pos.append(_scaled(coeff, core, env, memo))
        if not pos:
            pos = [const_expr(Fraction(0), dt)]
        out = pos[0]
        for p in pos[1:]:
            out = out + p
        for n in neg:
            out = out - n
        return out
    if isinstance(e, ir.Mul):
        num, den = [], []
        coeff: ir.Number = Fraction(1)
        for a in e.args:
            if isinstance(a, ir.Const):
                coeff = a.value
            elif isinstance(a, ir.Pow) and a.exponent < 0:
                den.append(to_prim(ir_pow_abs(a), env, memo))
            else:
                num.append(to_prim(a, env, memo))
        out = None
        if coeff != 1 or not num:
            out = const_expr(coeff, dt)
        for n in num:
            out = n if out is None else out * n
        for d in den:
            out = out / d
        return out
    if isinstance(e, ir.Pow):
        p = e.exponent
        base = to_prim(e.base, env, memo)
        if p < 0:
            return const_expr(Fraction(1), dt) / to_prim(ir_pow_abs(e), env, memo)
        if p == Fraction(1, 2):
            return tir.sqrt(base)
        if p.denominator == 1:
            out = base
            for _ in range(p.numerator - 1):
                out = out * base
            return out
        if p.denominator == 2:
            root = tir.sqrt(base)
            out = root
            for _ in range(p.numerator - 1):
                out = out * root
            return out
        raise Unsupported(f"power {p}")
    if isinstance(e, ir.Exp):
        return tir.exp(to_prim(e.arg, env, memo))
    if isinstance(e, ir.Log):
        return tir.log(to_prim(e.arg, env, memo))
    if isinstance(e, ir.Max):
        out = to_prim(e.args[0], env, memo)
        for a in e.args[1:]:
            out = tir.max(out, to_prim(a, env, memo))
        return out
    raise Unsupported(f"cannot realise {type(e).__name__}")


def ir_pow_abs(p: ir.Pow) -> ir.SymExpr:
    from .canonicalize import mk_pow  # local import: avoid cycle at module load

    return mk_pow(p.base, -p.exponent)


def _coeff(a: ir.SymExpr) -> tuple[Fraction, ir.SymExpr]:
    if isinstance(a, ir.Const):
        return a.value, ir.ONE
    if isinstance(a, ir.Mul) and isinstance(a.args[0], ir.Const) and not ir.is_inf(a.args[0]):
        rest = a.args[1:]
        core = rest[0] if len(rest) == 1 else ir.raw_mul(rest)
        return a.args[0].value, core
    return Fraction(1), a


def _scaled(coeff, core: ir.SymExpr, env: RealizeEnv, memo: dict):
    if core is ir.ONE:
        return const_expr(coeff, env.dtype)
    val = to_prim(core, env, memo)
    if coeff == 1:
        return val
    return const_expr(coeff, env.dtype) * val


def _realize_index(i: ir.IndexExpr, env: RealizeEnv):
    if isinstance(i, ir.IConst):
        return tir.const(int(i.value), "int32")
    try:
        return env.index[i]
    except KeyError as err:
        raise Unsupported(f"unbound index {i}") from err


def node_count(exprs, leaf_types=(ir.Const, ir.StateVar, ir.Atom, ir.Elem, ir.ShapeSym)) -> int:
    """Number of distinct non-leaf nodes across ``exprs`` (SSA size with sharing)."""
    seen: set[ir.Node] = set()
    stack = list(exprs)
    while stack:
        n = stack.pop()
        if n in seen or isinstance(n, leaf_types):
            continue
        seen.add(n)
        stack.extend(n.children())
    return len(seen)
