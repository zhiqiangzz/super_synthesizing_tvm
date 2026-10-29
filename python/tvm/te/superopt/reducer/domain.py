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

A state kept for a single pass is evaluated on every part of the axis the
reduction is split into: a single element, a run of equal elements, nothing
at all. Two questions decide whether it can be kept, and both are answered
here by proof:

non-empty parts
    Does its definition divide by, take the ``log`` of, or take an even root
    of something that can be zero (or negative) on some part? The
    *conditions* of an expression are read off its canonical form outside
    the reductions (inside, they are the program's own conditions on its
    data, met wherever the program is defined). A condition holds when the
    sign of its operand is proven (:class:`Signs`); otherwise it is taken
    to fail, and for the report a witness is looked for: the operand on a
    single element (the standard deviation of one element is 0).
the empty part
    What does the merge compute when one side is still the identity element,
    as the generated code does it (a maximum starts at the lowest finite
    value, ``-FLT_MAX``)? :func:`empty_side` evaluates the merge on classes
    of values instead of values -- zero, finite, ``±FLT_MAX``, infinite,
    NaN -- following IEEE arithmetic: ``exp(-FLT_MAX) = 0``,
    ``0 * FLT_MAX = 0``, ``log 0 = -inf``, ``0 * inf = NaN``. The result
    holds for every finite state on the other side.

Signs come from the program as written (:class:`TeSigns`): ``Σ (x - m)²`` is
non-negative for every ``m``, which its expansion no longer shows; and from
what the caller declares about its inputs (a weight is positive).
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

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


# ---------------------------------------------------------------------------
# the empty side, on classes of floating-point values
# ---------------------------------------------------------------------------
Z, P, N, WP, WN, IP, IN, NAN = "0", "+", "-", "+max", "-max", "+inf", "-inf", "nan"
FINITE = frozenset({Z, P, N})
_ORDER = {IN: 0, WN: 1, N: 2, Z: 3, P: 4, WP: 5, IP: 6}
_NEG = {Z: Z, P: N, N: P, WP: WN, WN: WP, IP: IN, IN: IP, NAN: NAN}


def _neg1(x: str) -> str:
    return _NEG[x]


def _add1(x: str, y: str) -> frozenset:
    if NAN in (x, y) or {x, y} == {IP, IN}:
        return frozenset({NAN})
    for inf in (IP, IN):
        if inf in (x, y):
            return frozenset({inf})
    if {x, y} == {WP, WN}:
        return frozenset({Z})  # FLT_MAX - FLT_MAX, exactly
    if x == y and x in (WP, WN):
        return frozenset({IP if x == WP else IN})  # overflows
    for big in (WP, WN):
        if big in (x, y):
            return frozenset({big})  # a finite value is lost against FLT_MAX
    if x == Z:
        return frozenset({y})
    if y == Z:
        return frozenset({x})
    if x == y:
        return frozenset({x})
    return FINITE  # P + N


def _mul1(x: str, y: str) -> frozenset:
    if NAN in (x, y):
        return frozenset({NAN})
    if Z in (x, y):
        other = y if x == Z else x
        return frozenset({NAN}) if other in (IP, IN) else frozenset({Z})
    neg = (x in (N, WN, IN)) != (y in (N, WN, IN))
    if x in (IP, IN) or y in (IP, IN) or (x in (WP, WN) and y in (WP, WN)):
        return frozenset({IN if neg else IP})
    if x in (WP, WN) or y in (WP, WN):  # FLT_MAX times a finite value of any size
        return frozenset({N, WN, IN} if neg else {P, WP, IP})
    return frozenset({N if neg else P})


def _pow1(x: str, p: Fraction) -> frozenset:
    if x == NAN:
        return frozenset({NAN})
    if x in (N, WN, IN):  # the power of the magnitude, with the sign an odd power keeps
        if p.denominator % 2 == 0:
            return frozenset({NAN})  # a root of a negative number
        mag = _pow1(_neg1(x), p)
        return mag if p.numerator % 2 == 0 else frozenset(_neg1(v) for v in mag)
    if p > 0:
        if x == WP:
            return frozenset({WP}) if p == 1 else frozenset({P} if p < 1 else {IP})
        return frozenset({x})  # 0, finite, inf
    return frozenset({Z: {IP}, P: {P}, WP: {Z, P}, IP: {Z}}[x])  # 1 / +0 = +inf


def _exp1(x: str) -> frozenset:
    return frozenset({NAN: {NAN}, IN: {Z}, WN: {Z}, N: {P}, Z: {P}, P: {P}, WP: {IP}, IP: {IP}}[x])


def _log1(x: str) -> frozenset:
    return frozenset(
        {NAN: {NAN}, IN: {NAN}, WN: {NAN}, N: {NAN}, Z: {IN}, P: FINITE, WP: {P}, IP: {IP}}[x]
    )


def _max1(x: str, y: str) -> frozenset:
    if NAN in (x, y):
        return frozenset({NAN})
    if x == y or {x, y} <= FINITE:
        return FINITE if {x, y} == {P, N} else frozenset({max(x, y, key=_ORDER.__getitem__)})
    return frozenset({max(x, y, key=_ORDER.__getitem__)})


def _lift2(f, xs: frozenset, ys: frozenset) -> frozenset:
    out: set = set()
    for x in xs:
        for y in ys:
            out |= f(x, y)
    return frozenset(out)


def classes(e: ir.SymExpr, env: dict[ir.Node, frozenset], memo: dict | None = None) -> frozenset:
    """The classes of values ``e`` (built with any constructors) can take.

    ``env`` gives the classes of the leaves (state variables); a constant is
    its own class.
    """
    memo = {} if memo is None else memo
    hit = memo.get(e)
    if hit is not None:
        return hit
    if e in env:
        out = env[e]
    elif isinstance(e, ir.Const):
        v = e.value
        if isinstance(v, float):
            out = frozenset({IP if v > 0 else IN})
        else:
            out = frozenset({P if v > 0 else (Z if v == 0 else N)})
    elif isinstance(e, ir.Add):
        out = classes(e.args[0], env, memo)
        for a in e.args[1:]:
            out = _lift2(_add1, out, classes(a, env, memo))
    elif isinstance(e, ir.Mul):
        # constant factors first: times ±1 a class is kept exactly (-FLT_MAX stays itself)
        c, out = Fraction(1), None
        for a in e.args:
            if isinstance(a, ir.Const) and not isinstance(a.value, float):
                c *= a.value
                continue
            cls = classes(a, env, memo)
            out = cls if out is None else _lift2(_mul1, out, cls)
        const = frozenset({P if c > 0 else (Z if c == 0 else N)})
        if out is None:
            out = const
        elif c == -1:
            out = frozenset(_neg1(x) for x in out)
        elif c != 1:
            out = _lift2(_mul1, out, const)
    elif isinstance(e, ir.Pow):
        out = frozenset(v for x in classes(e.base, env, memo) for v in _pow1(x, e.exponent))
    elif isinstance(e, ir.Exp):
        out = frozenset(v for x in classes(e.arg, env, memo) for v in _exp1(x))
    elif isinstance(e, ir.Log):
        out = frozenset(v for x in classes(e.arg, env, memo) for v in _log1(x))
    elif isinstance(e, ir.Max):
        out = classes(e.args[0], env, memo)
        for a in e.args[1:]:
            out = _lift2(_max1, out, classes(a, env, memo))
    else:
        out = frozenset({Z, P, N})  # anything else of the other side: finite
    memo[e] = out
    return out


def identity_classes(identity) -> list[frozenset]:
    """The classes of the identity element as the generated code holds it."""
    out = []
    for c in identity:
        v = c.value
        if isinstance(v, float):
            out.append(frozenset({WP if v > 0 else WN}))  # max_value / min_value
        else:
            out.append(frozenset({P if v > 0 else (Z if v == 0 else N)}))
    return out


def sign_classes(sign: int) -> frozenset:
    """The classes of a finite state of a non-empty part with a known sign."""
    return frozenset({P}) if sign == POS else frozenset({Z, P}) if sign == NONNEG else FINITE


def empty_side(expr: ir.SymExpr, empty: str, identity, signs: list[int]) -> frozenset:
    """The classes of ``expr`` (a merge, or part of one) when side ``empty`` holds
    the identity and the other side a finite state with the given signs."""
    other = "b" if empty == "a" else "a"
    env: dict[ir.Node, frozenset] = {}
    for k, cls in enumerate(identity_classes(identity)):
        env[ir.state_var(empty, k)] = cls
        env[ir.state_var(other, k)] = sign_classes(signs[k])
    return classes(expr, env)


def bad(cls: frozenset) -> bool:
    """Not a finite value of a non-empty part: NaN, infinite or ``±FLT_MAX``."""
    return not cls <= FINITE


__all__ = [
    "ANY",
    "NONNEG",
    "POS",
    "Condition",
    "Signs",
    "TeSigns",
    "Witness",
    "bad",
    "classes",
    "conditions",
    "empty_side",
    "witness",
]
