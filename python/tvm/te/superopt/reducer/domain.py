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
"""Where a state is defined: proofs about every part of the axis, no samples.

A state kept for a single pass must be defined on every non-empty part of
its axis, including a single element or a run of equal elements. Empty
parts are skipped by execution and impose no conditions here.

Does its definition divide by, take the ``log`` of, or take an even root
of something that can be zero (or negative)? The *conditions* are read
outside reductions; inside, they are the program's own conditions on its
data. A condition holds when its operand's sign is proven (:class:`Signs`).
Otherwise a witness is looked for, such as a single element with zero
standard deviation.

Signs come from the program as written (:class:`TeSigns`): ``Σ (x - m)²`` is
non-negative for every ``m``, which its expansion no longer shows; and from
what the caller declares about its inputs (a weight is positive).
"""

from __future__ import annotations

import dataclasses

import tvm_ffi

from tvm import te
from tvm_tirx import tirx as tir
from tvm_tirx.ir import Call, FloatImm, IntImm
from tvm_tirx.ir.prim import expr_deep_equal

from ..loads import is_load, load_tensor
from ..symbolic import ir
from ..symbolic.canonicalize import NONNEG as NONNEG_ID
from ..symbolic.canonicalize import Unsupported, mk_neg
from ..symbolic.lower import classify_combiner

ANY, NONNEG, POS = 0, 1, 2


def _add_signs(signs) -> int:
    signs = list(signs)
    if min(signs) == ANY:
        return ANY
    return POS if max(signs) == POS else NONNEG


# ---------------------------------------------------------------------------
# signs of the program's tensors, from their TE definitions
# ---------------------------------------------------------------------------
class TeSigns:
    """The sign of every tensor of a TE program, from how it is computed.

    ``positive`` names the placeholders the caller declares positive. A
    placeholder read directly as the argument of a ``log`` is positive and
    one under a ``sqrt`` non-negative: the program is not defined otherwise.
    """

    def __init__(self, outputs, positive=()) -> None:
        from ..symbolic.lower import compute_ops

        self.memo: dict = {}
        self.declared: dict[str, int] = dict.fromkeys(positive, POS)
        for op in compute_ops(outputs):
            for body in op.body:
                tvm_ffi.structural_walk(body, self._note_domain)

    def _note_domain(self, e) -> None:
        if not isinstance(e, Call) or e.op.name not in ("prim.log", "prim.sqrt"):
            return
        arg = e.args[0]
        while isinstance(arg, tir.Cast):
            arg = arg.value
        if is_load(arg) and isinstance(load_tensor(arg).op, te.PlaceholderOp):
            sign = POS if e.op.name == "prim.log" else NONNEG
            name = load_tensor(arg).op.name
            self.declared[name] = max(self.declared.get(name, ANY), sign)

    def tensor(self, t: te.Tensor) -> int:
        op = t.op
        for known, sign in self.memo.items():
            if known.same_as(op):
                return sign
        if isinstance(op, te.PlaceholderOp):
            sign = self.declared.get(op.name, ANY)
        elif isinstance(op, te.ComputeOp):
            self.memo[op] = ANY  # guards recursion
            sign = self.expr(op.body[t.value_index])
        else:
            sign = ANY
        self.memo[op] = sign
        return sign

    def expr(self, e) -> int:
        if isinstance(e, FloatImm | IntImm):
            return POS if e.value > 0 else (NONNEG if e.value == 0 else ANY)
        if isinstance(e, tir.Cast):
            return self.expr(e.value)
        if isinstance(e, tir.Var):
            return POS  # an extent used as a value
        if is_load(e):
            return self.tensor(load_tensor(e))
        if isinstance(e, tir.Add):
            return _add_signs((self.expr(e.a), self.expr(e.b)))
        if isinstance(e, tir.Mul):
            if expr_deep_equal(e.a, e.b):  # a square
                return POS if self.expr(e.a) == POS else NONNEG
            return min(self.expr(e.a), self.expr(e.b))
        if isinstance(e, tir.Div):
            return self.expr(e.a) if self.expr(e.b) != ANY else ANY
        if isinstance(e, tir.Max):
            return max(self.expr(e.a), self.expr(e.b))
        if isinstance(e, tir.Min):
            return min(self.expr(e.a), self.expr(e.b))
        if isinstance(e, te.Reduce):
            kind = classify_combiner(e.combiner)
            if kind is None:
                return ANY
            return self.expr(e.source[e.value_index])
        if isinstance(e, Call):
            name = e.op.name
            if name == "prim.exp":
                return POS
            if name == "prim.sqrt":
                return POS if self.expr(e.args[0]) == POS else NONNEG
            if name in ("prim.fabs", "prim.abs"):
                return NONNEG
            if name == "prim.pow":
                base, k = self.expr(e.args[0]), e.args[1]
                if isinstance(k, FloatImm | IntImm):
                    if float(k.value).is_integer() and int(k.value) % 2 == 0:
                        return POS if base == POS else NONNEG
                return base
        return ANY


# ---------------------------------------------------------------------------
# signs over the chain's IR
# ---------------------------------------------------------------------------
class Signs:
    """Signs of expressions over a chain's IR.

    ``tensor(tid)`` gives the sign of the tensor an element of id ``tid``
    reads (a member, a context, a boundary tensor); ``states`` the sign of
    state variables. A reduction is over a non-empty part.
    """

    def __init__(self, tensor, states: dict | None = None) -> None:
        self._tensor = tensor
        self.states = states or {}
        self._memo: dict[ir.Node, int] = {}

    def tensor(self, tid: int) -> int:
        if NONNEG_ID - 10_000 < tid < NONNEG_ID + 10_000:  # read as a value known to be >= 0
            return max(NONNEG, self._tensor(tid - NONNEG_ID))
        return self._tensor(tid)

    def __call__(self, e: ir.SymExpr) -> int:
        hit = self._memo.get(e)
        if hit is None:
            hit = self._memo[e] = self._sign(e)
        return hit

    def _sign(self, e: ir.SymExpr) -> int:
        if isinstance(e, ir.Const):
            v = e.value
            return POS if v > 0 else (NONNEG if v == 0 else ANY)
        if isinstance(e, ir.ShapeSym | ir.Card | ir.Exp):
            return POS
        if isinstance(e, ir.StateVar):
            return self.states.get(e, ANY)
        if isinstance(e, ir.Elem):
            return self.tensor(e.tensor)
        if isinstance(e, ir.Add):
            return _add_signs(self(a) for a in e.args)
        if isinstance(e, ir.Mul):
            return min(self(a) for a in e.args)
        if isinstance(e, ir.Pow):
            p, b = e.exponent, self(e.base)
            if p.denominator == 1 and p.numerator % 2 == 0:
                return POS if b == POS else NONNEG
            if p.denominator % 2 == 0:  # a root, defined where its base is >= 0
                return POS if b == POS else NONNEG
            return b
        if isinstance(e, ir.Max):
            return max(self(a) for a in e.args)
        if isinstance(e, ir.Reduce):
            return self(e.body)
        return ANY


# ---------------------------------------------------------------------------
# conditions and their decision
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Condition:
    """Where an expression is defined: ``operand`` is non-zero / positive / non-negative."""

    kind: str  # "nonzero", "positive" or "nonnegative"
    operand: ir.SymExpr

    def holds(self, sign) -> bool:
        if self.kind == "nonzero":
            return sign(self.operand) == POS or sign(mk_neg(self.operand)) == POS
        if self.kind == "positive":
            return sign(self.operand) == POS
        return sign(self.operand) >= NONNEG


def conditions(e: ir.SymExpr) -> list[Condition]:
    """The conditions of ``e`` outside its reductions, each once."""
    out: dict[tuple, Condition] = {}
    stack = [e]
    seen: set = set()
    while stack:
        n = stack.pop()
        if n in seen or isinstance(n, ir.Reduce | ir.MonoidReduce):
            continue
        seen.add(n)
        if isinstance(n, ir.Pow):
            if n.exponent < 0:
                out.setdefault(("nonzero", n.base), Condition("nonzero", n.base))
            if n.exponent.denominator % 2 == 0:
                out.setdefault(("nonnegative", n.base), Condition("nonnegative", n.base))
        elif isinstance(n, ir.Log):
            out.setdefault(("positive", n.arg), Condition("positive", n.arg))
        stack.extend(n.children())
    return list(out.values())


@dataclasses.dataclass(frozen=True)
class Witness:
    """Why a condition is not proven: what is known about its operand on a short part."""

    kind: str  # "every single element", "where zero" or "not proven"
    reads: tuple[ir.SymExpr, ...] = ()  # for "where zero": the elements whose zero makes it zero


def witness(operand: ir.SymExpr, R: ir.DSym) -> Witness:
    """The operand (written over reductions on ``R``) on a single element."""
    from .derive import singleton

    try:
        one = singleton(operand, R)
    except Unsupported:
        return Witness("every single element")  # already undefined there
    if one is ir.ZERO:
        return Witness("every single element")
    factors = one.args if isinstance(one, ir.Mul) else (one,)
    reads = []
    for f in factors:
        while isinstance(f, ir.Pow) and f.exponent > 0:
            f = f.base
        if isinstance(f, ir.Elem) and f.tensor >= 0:
            reads.append(f)
    if reads:
        return Witness("where zero", tuple(reads))
    return Witness("not proven")


__all__ = [
    "ANY",
    "NONNEG",
    "POS",
    "Condition",
    "Signs",
    "TeSigns",
    "Witness",
    "conditions",
    "witness",
]
