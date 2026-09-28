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
"""Print scalar symbolic expressions as TE/tirx Python source text.

The string twin of :mod:`realize`: same operator mapping, but producing code
such as ``a[1] * tir.exp(a[0] - tir.max(a[0], b[0]))`` for reducer merges and
``tir.const(1.0, "float32") / tir.sqrt(head_dim.astype("float32"))`` for
constants. Sub-expressions used more than once can be hoisted into local
variables so the printed merge function reads like hand-written code.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from fractions import Fraction

from ..dims import DimTable
from . import ir
from .canonicalize import Unsupported, mk_pow


@dataclasses.dataclass
class SourceEnv:
    dtype: str
    dims: DimTable | None = None
    index: dict[ir.IndexExpr, str] = dataclasses.field(default_factory=dict)
    state: dict[ir.StateVar, str] = dataclasses.field(default_factory=dict)
    elem: Callable[[int, tuple[str, ...]], str] | None = None


def const_source(value: ir.Number, dtype: str, quoted: bool = True) -> str:
    """``tir.const`` / ``tir.min_value`` text; ``dtype`` is a literal unless ``quoted=False``."""
    dt = f'"{dtype}"' if quoted else dtype
    if isinstance(value, float):
        return f"tir.{'min' if value < 0 else 'max'}_value({dt})"
    return f"tir.const({float(value)!r}, {dt})"


def _fmt(v: Fraction) -> str:
    return repr(float(v))


def to_source(e: ir.SymExpr, env: SourceEnv, hoist: dict[ir.Node, str] | None = None) -> str:
    """Source text of ``e``; nodes present in ``hoist`` print as their variable name."""
    if hoist and e in hoist:
        return hoist[e]
    dt = env.dtype
    if isinstance(e, ir.Const):
        return const_source(e.value, dt)
    if isinstance(e, ir.ShapeSym):
        return f'{e.name}.astype("{dt}")'
    if isinstance(e, ir.StateVar):
        return env.state[e]
    if isinstance(e, ir.Atom):
        return e.name
    if isinstance(e, ir.Elem):
        if env.elem is None:
            raise Unsupported("tensor element without a loader")
        return env.elem(e.tensor, tuple(_index_source(i, env) for i in e.indices))
    if isinstance(e, ir.Add):
        pos, neg = [], []
        for a in e.args:
            coeff, core = _coeff(a)
            (neg if coeff < 0 else pos).append(_scaled(abs(coeff), core, env, hoist))
        out = " + ".join(pos) if pos else const_source(Fraction(0), dt)
        for n in neg:
            out += f" - {n}"
        return f"({out})" if len(pos) + len(neg) > 1 else out
    if isinstance(e, ir.Mul):
        num, den = [], []
        coeff: ir.Number = Fraction(1)
        for a in e.args:
            if isinstance(a, ir.Const):
                coeff = a.value
            elif isinstance(a, ir.Pow) and a.exponent < 0:
                den.append(_atom(to_source(mk_pow(a.base, -a.exponent), env, hoist)))
            else:
                num.append(_atom(to_source(a, env, hoist)))
        if coeff != 1 or not num:
            num.insert(0, const_source(coeff, dt))
        out = " * ".join(num)
        for d in den:
            out += f" / {d}"
        return f"({out})" if len(num) + len(den) > 1 else out
    if isinstance(e, ir.Pow):
        p = e.exponent
        if p < 0:
            inv = to_source(mk_pow(e.base, -p), env, hoist)
            return f"({const_source(Fraction(1), dt)} / {_atom(inv)})"
        base = to_source(e.base, env, hoist)
        if p == Fraction(1, 2):
            return f"tir.sqrt({base})"
        if p.denominator == 1:
            return "(" + " * ".join([_atom(base)] * p.numerator) + ")"
        if p.denominator == 2:
            root = f"tir.sqrt({base})"
            return "(" + " * ".join([root] * p.numerator) + ")"
        raise Unsupported(f"power {p}")
    if isinstance(e, ir.Exp):
        return f"tir.exp({to_source(e.arg, env, hoist)})"
    if isinstance(e, ir.Log):
        return f"tir.log({to_source(e.arg, env, hoist)})"
    if isinstance(e, ir.Max):
        out = to_source(e.args[0], env, hoist)
        for a in e.args[1:]:
            out = f"tir.max({out}, {to_source(a, env, hoist)})"
        return out
    raise Unsupported(f"cannot print {type(e).__name__}")


def _atom(s: str) -> str:
    """Parenthesise an infix expression when it is used as an operand."""
    if s.startswith("(") and s.endswith(")"):
        return s
    return f"({s})" if any(op in s for op in (" + ", " - ", " * ", " / ")) else s


def _coeff(a: ir.SymExpr) -> tuple[Fraction, ir.SymExpr]:
    if isinstance(a, ir.Const):
        return a.value, ir.ONE
    if isinstance(a, ir.Mul) and isinstance(a.args[0], ir.Const) and not ir.is_inf(a.args[0]):
        rest = a.args[1:]
        core = rest[0] if len(rest) == 1 else ir.raw_mul(rest)
        return a.args[0].value, core
    return Fraction(1), a


def _scaled(coeff, core: ir.SymExpr, env: SourceEnv, hoist) -> str:
    if core is ir.ONE:
        return const_source(coeff, env.dtype)
    val = to_source(core, env, hoist)
    if coeff == 1:
        return val
    return f"{const_source(coeff, env.dtype)} * {_atom(val)}"


def _index_source(i: ir.IndexExpr, env: SourceEnv) -> str:
    if isinstance(i, ir.IConst):
        return str(int(i.value))
    try:
        return env.index[i]
    except KeyError as err:
        raise Unsupported(f"unbound index {i}") from err


def shared_subexpressions(
    exprs, leaf_types=(ir.Const, ir.StateVar, ir.Atom, ir.ShapeSym)
) -> list[ir.Node]:
    """Non-leaf nodes used more than once across ``exprs``, in evaluation order."""
    uses: dict[ir.Node, int] = {}
    order: list[ir.Node] = []

    def visit(n: ir.Node) -> None:
        if isinstance(n, leaf_types) or not n.children():
            return
        if not any(isinstance(x, ir.StateVar | ir.Atom | ir.Elem) for x in _walk(n)):
            return  # a pure constant: printed inline
        if isinstance(n, ir.Mul) and isinstance(n.args[0], ir.Const):
            for a in n.args[1:]:  # coefficient wrapper: printed inline as "c * x" / "x - y"
                visit(a)
            return
        if isinstance(n, ir.Pow) and n.exponent < 0:
            visit(n.base)  # printed as a division of the base
            return
        uses[n] = uses.get(n, 0) + 1
        if uses[n] > 1:
            return
        for c in n.children():
            visit(c)
        order.append(n)

    for e in exprs:
        visit(e)
    return [n for n in order if uses[n] > 1]


def _walk(e: ir.Node):
    stack = [e]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children())


def hoisted_source(exprs, env: SourceEnv, prefix: str = "v") -> tuple[list[str], list[str]]:
    """``(assignments, expressions)``: shared sub-terms become ``prefix{k} = ...`` lines."""
    hoist: dict[ir.Node, str] = {}
    lines: list[str] = []
    for k, n in enumerate(shared_subexpressions(exprs)):
        src = to_source(n, env, hoist)
        name = f"{prefix}{k}"
        lines.append(f"{name} = {src}")
        hoist[n] = name
    return lines, [to_source(e, env, hoist) for e in exprs]
