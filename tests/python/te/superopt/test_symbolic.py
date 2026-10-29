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
"""Canonical symbolic semantics and lowering."""

from fractions import Fraction

import pytest
import te_programs

import tvm.testing
from tvm import te
from tvm.te.superopt.dims import DimTable
from tvm.te.superopt.symbolic import LowerCtx, Unsupported, ir
from tvm.te.superopt.symbolic.canonicalize import (
    instantiate,
    mk_add,
    mk_card,
    mk_div,
    mk_exp,
    mk_log,
    mk_max,
    mk_mul,
    mk_neg,
    mk_pow,
    mk_reduce,
    mk_sub,
    nonneg_alias,
    nonnegative,
    positive,
    subst_domain,
    walk,
)
from tvm_tirx import tirx as tir

x, y, z = ir.atom("x"), ir.atom("y"), ir.atom("z")
c2, c3 = ir.const(2), ir.const(3)


def test_canon_flatten_fold_sort():
    assert mk_add(x, mk_add(y, c2), c3) is mk_add(c3, y, c2, x)
    assert mk_add(x, x) is mk_mul(c2, x)
    assert mk_add(x, mk_neg(x)) is ir.ZERO
    assert mk_mul(x, mk_mul(y, c2), ir.ONE) is mk_mul(c2, y, x)
    assert mk_mul(x, ir.ZERO) is ir.ZERO
    assert mk_mul(x, x) is mk_pow(x, 2)
    assert mk_div(x, x) is ir.ONE
    assert mk_pow(mk_pow(x, Fraction(1, 2)), 2) is x
    assert mk_pow(ir.const(4), Fraction(1, 2)) is c2


def test_canon_distributes_products_over_sums():
    assert mk_mul(c2, mk_add(x, y)) is mk_add(mk_mul(c2, x), mk_mul(c2, y))
    assert mk_mul(z, mk_sub(x, y)) is mk_sub(mk_mul(z, x), mk_mul(z, y))


def test_power_of_a_sum_is_expanded_by_the_size_of_the_result():
    """``(x - y)⁷`` has 8 monomials and is expanded at any degree; a sum of ten terms to
    the fourth power has 715 and is kept as a power."""
    seventh = mk_pow(mk_sub(x, y), 7)
    assert isinstance(seventh, ir.Add) and len(seventh.args) == 8
    assert mk_mul(*[mk_sub(x, y)] * 7) is seventh
    wide = mk_add(*[ir.atom(f"t{k}") for k in range(10)])
    assert isinstance(mk_pow(wide, 4), ir.Pow) and isinstance(mk_pow(wide, 2), ir.Add)


def test_canon_exp_merge():
    assert mk_mul(mk_exp(x), mk_exp(y)) is mk_exp(mk_add(x, y))
    assert mk_pow(mk_exp(x), -1) is mk_exp(mk_neg(x))
    assert mk_mul(mk_exp(x), mk_exp(mk_neg(x))) is ir.ONE
    assert mk_exp(ir.ZERO) is ir.ONE
    assert mk_exp(ir.NEG_INF_C) is ir.ZERO


def test_canon_infinities():
    assert mk_max(x, ir.NEG_INF_C) is x
    assert mk_add(x, ir.NEG_INF_C) is ir.NEG_INF_C
    assert mk_mul(ir.POS_INF_C, ir.ZERO) is ir.ZERO
    sym = ir.shape_sym("n")
    assert positive(mk_pow(sym, Fraction(-1, 2)))
    assert mk_mul(ir.NEG_INF_C, mk_pow(sym, Fraction(-1, 2))) is ir.NEG_INF_C


def test_canon_max_common_parts():
    assert mk_max(mk_add(x, z), mk_add(y, z)) is mk_add(mk_max(x, y), z)
    assert mk_max(mk_mul(c2, x), mk_mul(c2, y)) is mk_mul(c2, mk_max(x, y))
    assert mk_max(x, y) is mk_max(y, x, x)


def _elem(t, *idx):
    return ir.elem(t, tuple(idx))


def test_reduce_rules():
    D = ir.dfull(0)
    j = ir.bidx(0)
    f = _elem(0, ir.idx("i0"), j)
    g = _elem(1, ir.idx("i0"), j)
    c = _elem(2, ir.idx("i0"))  # j-free
    # Σ(f + g) = Σf + Σg ; Σ c f = c Σ f
    assert mk_reduce("sum", D, 0, mk_add(f, g)) is mk_add(
        mk_reduce("sum", D, 0, f), mk_reduce("sum", D, 0, g)
    )
    assert mk_reduce("sum", D, 0, mk_mul(c, f)) is mk_mul(c, mk_reduce("sum", D, 0, f))
    # shift extraction: Σ exp(f - c) = exp(-c) Σ exp(f)
    assert mk_reduce("sum", D, 0, mk_exp(mk_sub(f, c))) is mk_mul(
        mk_exp(mk_neg(c)), mk_reduce("sum", D, 0, mk_exp(f))
    )
    # max rules
    assert mk_reduce("max", D, 0, mk_add(f, c)) is mk_add(mk_reduce("max", D, 0, f), c)
    assert mk_reduce("max", D, 0, mk_mul(c2, f)) is mk_mul(c2, mk_reduce("max", D, 0, f))
    assert mk_reduce("max", D, 0, mk_exp(f)) is mk_exp(mk_reduce("max", D, 0, f))
    assert mk_reduce("max", D, 0, mk_max(f, g)) is mk_max(
        mk_reduce("max", D, 0, f), mk_reduce("max", D, 0, g)
    )
    # constant body
    assert mk_reduce("sum", D, 0, c) is mk_mul(c, mk_card(D))
    assert mk_reduce("max", D, 0, c) is c


def test_domain_algebra():
    axis = 0
    j = ir.bidx(0)
    f = _elem(0, ir.idx("i0"), j)
    A, B = ir.dsym(axis, "A"), ir.dsym(axis, "B")
    s = mk_reduce("sum", ir.dsym(axis, "R"), 0, f)
    split = subst_domain(s, {ir.dsym(axis, "R"): ir.dunion(A, B)})
    assert split is mk_add(mk_reduce("sum", A, 0, f), mk_reduce("sum", B, 0, f))
    m = mk_reduce("max", ir.dsym(axis, "R"), 0, f)
    assert subst_domain(m, {ir.dsym(axis, "R"): ir.dunion(A, B)}) is mk_max(
        mk_reduce("max", A, 0, f), mk_reduce("max", B, 0, f)
    )
    assert subst_domain(s, {ir.dsym(axis, "R"): ir.dempty(axis)}) is ir.ZERO
    assert subst_domain(m, {ir.dsym(axis, "R"): ir.dempty(axis)}) is ir.NEG_INF_C


def test_alpha_equivalence_of_nested_reductions():
    D0, D1 = ir.dfull(0), ir.dfull(1)
    inner = mk_reduce("sum", D1, 1, mk_mul(_elem(0, ir.bidx(0), ir.bidx(1)), _elem(1, ir.bidx(1))))
    outer = mk_reduce("sum", D0, 0, mk_exp(inner))
    # rebuilding the same structure yields the same node
    inner2 = mk_reduce("sum", D1, 1, mk_mul(_elem(1, ir.bidx(1)), _elem(0, ir.bidx(0), ir.bidx(1))))
    assert mk_reduce("sum", D0, 0, mk_exp(inner2)) is outer
    # inlining a body under a binder shifts its levels
    body = mk_reduce("sum", D1, 0, _elem(0, ir.idx("i0"), ir.bidx(0)))
    inst = instantiate(body, {ir.idx("i0"): ir.bidx(0)}, 1)
    assert isinstance(inst, ir.Reduce) and inst.level == 1
    assert inst.body is _elem(0, ir.bidx(0), ir.bidx(1))


def test_dimtable_symbolic_vs_concrete():
    n, m = te.var("n"), te.var("m")
    dims = DimTable()
    assert dims.key(n) == dims.key(n)
    assert dims.key(n) != dims.key(m)
    assert dims.key(4) == dims.key(4)
    assert dims.key(4) != dims.key(n)
    assert dims.name(dims.key(n)) == "n" and dims.key_of_name("n") == dims.key(n)


def test_lower_attention_naive_equals_flash_closed_form():
    """The textbook chain canonicalises to the unshifted softmax; the max shift cancels."""
    ins, out = te_programs.attention("naive")
    ctx = LowerCtx()
    sem = ctx.lower(out)
    body = sem.body
    assert sem.rank == 4 and sem.dtype == "float32"
    # exactly two top-level reductions over the key axis and no max anywhere
    reds = [r for r in _all_nodes(body) if isinstance(r, ir.Reduce) and r.level == 0]
    assert sorted(r.kind for r in reds) == ["sum", "sum"]
    assert not any(isinstance(n, ir.Max) for n in _all_nodes(body))
    assert not any(isinstance(n, ir.Reduce) and n.kind == "max" for n in _all_nodes(body))


def test_integer_power_is_the_product():
    """``pow(x, 3)`` and ``x * x * x`` are the same expression; other exponents are not taken."""
    n = te.var("n")
    X = te.placeholder((n,), name="X")

    def lowered(f):
        return ctx.lower(te.compute((n,), lambda i: f(X[i]), name="t")).body

    ctx = LowerCtx()
    cube = lowered(lambda v: tir.power(v, tir.const(3.0, "float32")))
    assert cube is lowered(lambda v: v * v * v) and isinstance(cube, ir.Pow)
    assert cube.exponent == 3
    inverse = lowered(lambda v: tir.power(v, tir.const(-2.0, "float32")))
    assert inverse is lowered(lambda v: tir.const(1.0, "float32") / (v * v))
    assert lowered(lambda v: tir.power(v, tir.const(1.0, "float32"))) is lowered(lambda v: v)
    # of a sum: expanded, like the product
    one = tir.const(1.0, "float32")
    fifth = lowered(lambda v: tir.power(v - one, tir.const(5.0, "float32")))
    assert fifth is lowered(lambda v: (v - one) * (v - one) * (v - one) * (v - one) * (v - one))
    assert isinstance(fifth, ir.Add) and len(fifth.args) == 6
    with pytest.raises(Unsupported, match="not an integer constant"):
        lowered(lambda v: tir.power(v, tir.const(0.5, "float32")))
    with pytest.raises(Unsupported, match="not an integer constant"):
        lowered(lambda v: tir.power(v, v))


def test_even_root_of_an_even_power_keeps_its_sign_lost():
    """``sqrt(x²)`` is ``|x|``, not ``x``: the exponents only merge where that is the same."""
    half = Fraction(1, 2)
    assert mk_pow(mk_pow(x, 2), half) is not x
    assert mk_pow(mk_pow(mk_pow(x, 2), half), 2) is mk_pow(x, 2)  # |x|² = x²
    assert mk_pow(mk_pow(x, 4), half) is mk_pow(x, 2)  # an even result has no sign to lose
    assert mk_pow(mk_pow(x, half), 2) is x  # defined for x >= 0 only, where it is x
    assert mk_pow(mk_pow(x, 3), Fraction(1, 3)) is x  # an odd root keeps the sign
    n = ir.shape_sym("n")
    assert mk_pow(mk_pow(n, 2), half) is n  # an extent is positive


def test_nonnegative_factors_leave_a_maximum():
    """``max_j c f_j = c max_j f_j`` for ``c >= 0``: a root, an even power, a declared tensor."""
    axis = DimTable().key(te.var("m"))
    dom = ir.dfull(axis)
    f = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    c = ir.elem(1, (ir.idx("i0"),))  # a value of unknown sign
    root = mk_pow(c, Fraction(1, 2))
    assert nonnegative(root) and nonnegative(mk_pow(c, 2)) and nonnegative(mk_pow(root, -1))
    assert not positive(root) and not nonnegative(c) and not nonnegative(mk_neg(root))
    plain = mk_reduce("max", dom, 0, f)
    assert mk_reduce("max", dom, 0, mk_mul(root, f)) is mk_mul(root, plain)
    assert mk_reduce("max", dom, 0, mk_div(f, root)) is mk_div(plain, root)
    stuck = mk_reduce("max", dom, 0, mk_mul(c, f))  # c < 0 would turn it into a minimum
    assert isinstance(stuck, ir.Reduce) and stuck.body is mk_mul(c, f)
    known = nonneg_alias(c)
    assert nonnegative(known) and known is not c
    assert mk_reduce("max", dom, 0, mk_mul(known, f)) is mk_mul(known, plain)
    # and out of an elementwise maximum
    assert mk_max(mk_mul(root, x), mk_mul(root, y)) is mk_mul(root, mk_max(x, y))


_AXIS = ir.dfull(DimTable().key(te.var("m")))


def _top(body):
    return mk_reduce("max", _AXIS, 0, body)


def test_growing_functions_leave_a_maximum():
    """``max_j g(f_j) = g(max_j f_j)`` where ``g`` grows with its argument: an odd power,
    a root, ``exp``, ``log``. An even power grows with ``|f|``, the larger of ``f`` and
    ``-f``; under a negative factor it falls with it and has no extreme of ``f`` to go to."""
    f = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    plain = _top(f)
    assert _top(mk_pow(f, 3)) is mk_pow(plain, 3)
    assert _top(mk_pow(f, Fraction(1, 2))) is mk_pow(plain, Fraction(1, 2))
    assert _top(mk_log(f)) is mk_log(plain)
    assert _top(mk_exp(mk_pow(f, 3))) is mk_exp(mk_pow(plain, 3))  # one inside the other
    assert _top(mk_neg(mk_pow(f, 3))) is mk_pow(_top(mk_neg(f)), 3)  # -f³ = (-f)³
    either = mk_max(plain, _top(mk_neg(f)))
    assert _top(mk_pow(f, 2)) is mk_pow(either, 2)
    assert _top(mk_mul(c3, mk_pow(f, 4))) is mk_mul(c3, mk_pow(either, 4))
    known = nonneg_alias(f)  # nothing to take the absolute value of
    assert _top(mk_pow(known, 2)) is mk_pow(_top(known), 2)
    falls = _top(mk_neg(mk_pow(f, 2)))
    assert isinstance(falls, ir.Reduce) and falls.body is mk_neg(mk_pow(f, 2))
    # the same reduction as a sum is left alone: a power does not commute with it
    assert isinstance(mk_reduce("sum", falls.domain, 0, mk_pow(f, 3)), ir.Reduce)


def test_expanded_power_under_a_maximum_is_recognised():
    """A power of a sum is stored expanded; under a maximum it is found again, whatever
    its degree, and is only taken for one if expanding the root gives it back."""
    f = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    c = ir.elem(1, (ir.idx("i0"),))
    d = mk_sub(f, c)
    plain = mk_sub(_top(f), c)  # max_j (f - c)
    either = mk_max(plain, mk_add(_top(mk_neg(f)), c))  # max_j |f - c|
    cube, square = mk_pow(d, 3), mk_pow(d, 2)
    assert isinstance(cube, ir.Add) and isinstance(square, ir.Add)
    assert _top(cube) is mk_pow(plain, 3)
    assert _top(square) is mk_pow(either, 2)
    assert _top(mk_pow(d, 4)) is mk_pow(either, 4)  # a fourth power, not a square of a square
    fifth = ir.ONE
    for _ in range(5):
        fifth = mk_mul(fifth, d)
    assert isinstance(fifth, ir.Add) and _top(fifth) is mk_pow(plain, 5)
    assert _top(mk_mul(c2, square)) is mk_mul(c2, mk_pow(either, 2))
    assert _top(mk_neg(cube)) is mk_pow(mk_add(_top(mk_neg(f)), c), 3)  # (c - f)³
    assert _top(mk_add(square, ir.ONE)) is mk_add(mk_pow(either, 2), ir.ONE)
    # of a quotient: ((f - c) / s)³ with s inside the reduction as well
    s = ir.elem(2, (ir.idx("i0"),))
    z = mk_div(d, s)
    assert _top(mk_pow(z, 3)) is mk_pow(_top(z), 3)

    # -(f - c)² falls with |f - c|: no extreme of f to go to
    (left,) = [n for n in _all_nodes(_top(mk_neg(square))) if isinstance(n, ir.Reduce)]
    assert isinstance(left.body, ir.Add) and left.body is not f


def test_a_polynomial_that_keeps_one_direction_leaves_a_maximum():
    """``max_j h(f_j - c)``, ``h`` a polynomial with constant coefficients: taken at the
    largest ``f`` where ``h`` never falls (``d³ + d``, a cube plus its root, no power), at
    the smallest where it never rises (``-d³ - 2 d``), left in the reduction where it does
    both (``d³ - d``). Decided exactly, a double root of ``h'`` included (``d³ - 3 d² +
    3 d = (d - 1)³ + 1``)."""
    f = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    c = ir.elem(1, (ir.idx("i0"),))
    d = mk_sub(f, c)
    top = mk_sub(_top(f), c)  # max_j (f - c)
    low = mk_sub(mk_neg(_top(mk_neg(f))), c)  # min_j (f - c)
    cube = mk_pow(d, 3)
    assert _top(mk_add(cube, d)) is mk_add(mk_pow(top, 3), top)
    falls = mk_neg(mk_add(cube, mk_mul(c2, d)))
    assert _top(falls) is mk_neg(mk_add(mk_pow(low, 3), mk_mul(c2, low)))
    both = _top(mk_sub(cube, d))
    assert _top(f) not in set(walk(both))
    double = mk_add(cube, mk_mul(ir.const(-3), mk_pow(d, 2)), mk_mul(c3, d))
    want = mk_add(mk_pow(top, 3), mk_mul(ir.const(-3), mk_pow(top, 2)), mk_mul(c3, top))
    assert _top(double) is want


def test_minimum_and_absolute_value_are_written_with_max():
    """``min f = -max(-f)`` and ``|v| = max(v, -v)``: one order operation in the IR."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    ctx = LowerCtx()
    lo = ctx.lower(te.compute((n,), lambda i: te.min(X[i, j1], axis=j1), name="lo")).body
    nmax = te.compute((n,), lambda i: te.max(-X[i, j2], axis=j2), name="nmax")
    neg = ctx.lower(te.compute((n,), lambda i: -nmax[i], name="neg")).body
    assert lo is neg and isinstance(lo, ir.Mul) and lo.args[0] is ir.MINUS_ONE
    (red,) = [a for a in lo.args if isinstance(a, ir.Reduce)]
    assert red.kind == "max" and red.body is mk_neg(ir.elem(0, (ir.idx("i0"), ir.bidx(0))))

    def lowered(f):
        return ctx.lower(te.compute((n, m), lambda i, k: f(X[i, k]), name="t")).body

    v = ir.elem(0, (ir.idx("i0"), ir.idx("i1")))
    assert lowered(tir.abs) is mk_max(v, mk_neg(v))
    assert lowered(lambda u: tir.min(u, tir.const(0.0, "float32"))) is mk_neg(
        mk_max(mk_neg(v), ir.ZERO)
    )
    assert ir.min_args(mk_neg(mk_max(mk_neg(x), mk_neg(y)))) == [x, y]
    assert ir.min_args(mk_neg(mk_max(x, y))) is None


def test_lower_flash_is_opaque_monoid():
    ins, out = te_programs.attention("flash")
    sem = LowerCtx().lower(out)
    monoids = [n for n in _all_nodes(sem.body) if isinstance(n, ir.MonoidReduce)]
    assert monoids and all(len(m.leaf) == 3 and len(m.merge) == 3 for m in monoids)
    assert monoids[0].identity[0] is ir.NEG_INF_C


def _all_nodes(e):
    out, stack = [], [e]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children())
    return out


if __name__ == "__main__":
    tvm.testing.main()
