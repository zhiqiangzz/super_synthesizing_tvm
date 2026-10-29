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
"""Where a state is defined, decided by proof: conditions, signs, witnesses, and the
merge against an empty part on classes of floating-point values."""

from fractions import Fraction

import tvm.testing
from tvm import te
from tvm.te.superopt.dims import DimTable
from tvm.te.superopt.reducer import domain
from tvm.te.superopt.reducer.derive import domains
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import (
    mk_add,
    mk_card,
    mk_div,
    mk_exp,
    mk_log,
    mk_max,
    mk_mul,
    mk_pow,
    mk_reduce,
    mk_sqrt,
    mk_sub,
)
from tvm_tirx import tirx


def _R():
    axis = DimTable().key(te.var("n"))
    return domains(axis)[1]


def _x(tensor: int = 0):
    return ir.elem(tensor, (ir.idx("i0"), ir.bidx(0)))


def _signs(**facts):
    return domain.Signs(lambda tid: facts.get(f"t{tid}", domain.ANY))


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------
def test_conditions_are_read_outside_the_reductions():
    R = _R()
    x = _x()
    s = mk_reduce("sum", R, 0, x)
    e = mk_add(mk_div(ir.ONE, s), mk_log(mk_reduce("sum", R, 0, mk_exp(x))), mk_sqrt(s))
    kinds = {(c.kind, c.operand) for c in domain.conditions(e)}
    assert ("nonzero", s) in kinds and ("nonnegative", s) in kinds
    assert ("positive", mk_reduce("sum", R, 0, mk_exp(x))) in kinds
    # a division inside a reduction is the program's own condition on its data
    inside = mk_reduce("sum", R, 0, mk_div(ir.ONE, x))
    assert domain.conditions(inside) == []


def test_signs_prove_what_cannot_be_zero():
    R = _R()
    x = _x()
    signs = _signs()
    assert signs(mk_card(R)) == domain.POS
    assert signs(mk_reduce("sum", R, 0, mk_exp(x))) == domain.POS
    assert signs(mk_reduce("sum", R, 0, mk_pow(x, 2))) == domain.NONNEG  # 0 where x is
    assert signs(mk_add(mk_reduce("sum", R, 0, mk_pow(x, 2)), ir.const(Fraction(1, 10)))) == (
        domain.POS
    )
    assert signs(mk_reduce("sum", R, 0, x)) == domain.ANY
    # a declared fact about a tensor
    assert _signs(t0=domain.POS)(mk_reduce("sum", R, 0, x)) == domain.POS


def test_conditions_hold_only_when_proven():
    R = _R()
    x = _x()
    signs = _signs()
    den = mk_reduce("sum", R, 0, mk_exp(x))
    assert domain.Condition("nonzero", den).holds(signs)
    assert domain.Condition("positive", den).holds(signs)
    sq = mk_reduce("sum", R, 0, mk_pow(x, 2))
    assert domain.Condition("nonnegative", sq).holds(signs)
    assert not domain.Condition("nonzero", sq).holds(signs)
    assert not domain.Condition("nonzero", mk_reduce("sum", R, 0, x)).holds(signs)
    assert domain.Condition("nonzero", mk_mul(ir.MINUS_ONE, den)).holds(signs)  # negative


# ---------------------------------------------------------------------------
# witnesses: the operand on a single element
# ---------------------------------------------------------------------------
def test_a_deviation_is_zero_on_every_single_element():
    R = _R()
    x, n = _x(), mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)
    var = mk_sub(mk_div(mk_reduce("sum", R, 0, mk_pow(x, 2)), n), mk_pow(mean, 2))
    assert domain.witness(var, R).kind == "every single element"
    assert domain.witness(mk_sqrt(var), R).kind == "every single element"
    hi = mk_reduce("max", R, 0, x)
    lo = mk_mul(ir.MINUS_ONE, mk_reduce("max", R, 0, mk_mul(ir.MINUS_ONE, x)))
    assert domain.witness(mk_sub(hi, lo), R).kind == "every single element"


def test_a_sum_of_squares_is_zero_where_its_element_is():
    R = _R()
    x = _x()
    w = domain.witness(mk_reduce("sum", R, 0, mk_pow(x, 2)), R)
    assert w.kind == "where zero" and w.reads == (x,)
    w = domain.witness(mk_sqrt(mk_reduce("sum", R, 0, mk_pow(x, 2))), R)
    assert w.kind == "where zero"
    assert domain.witness(mk_reduce("sum", R, 0, mk_exp(x)), R).kind == "not proven"


# ---------------------------------------------------------------------------
# signs of the program's tensors, from TE
# ---------------------------------------------------------------------------
def test_te_signs_see_squares_roots_and_declarations():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    W = te.placeholder((n, m), name="W")
    B = te.placeholder((n,), name="B")
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m.astype("float32"), name="mean")
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - mean[i]) * (X[i, j2] - mean[i]), axis=j2), name="ss"
    )
    sd = te.compute((n,), lambda i: tirx.sqrt(ss[i] / m.astype("float32") + 1e-5), name="sd")
    sw = te.compute((n,), lambda i: te.sum(W[i, j3], axis=j3), name="sw")
    lb = te.compute((n,), lambda i: tirx.log(B[i]), name="lb")
    signs = domain.TeSigns([sd, sw, lb], positive=("W",))
    assert signs.tensor(ss) == domain.NONNEG  # a square, whatever the mean
    assert signs.tensor(sd) == domain.POS  # eps keeps it away from 0
    assert signs.tensor(sw) == domain.POS  # declared
    assert signs.tensor(mean) == domain.ANY
    assert signs.tensor(B) == domain.POS  # the program takes its log
    assert domain.TeSigns([sw]).tensor(sw) == domain.ANY  # not declared


# ---------------------------------------------------------------------------
# the empty side: IEEE arithmetic on classes of values
# ---------------------------------------------------------------------------
def test_classes_of_the_operations_that_meet_an_empty_side():
    a, b = ir.state_var("a", 0), ir.state_var("b", 0)
    big = {b: frozenset({domain.WP})}  # the largest float
    zero = {b: frozenset({domain.Z})}
    assert domain.classes(mk_exp(mk_mul(ir.MINUS_ONE, b)), big) == {domain.Z}
    assert domain.classes(mk_mul(a, b), {a: frozenset({domain.Z}), **big}) == {domain.Z}
    assert domain.classes(ir.raw_add([b, b]), big) == {domain.IP}  # overflows
    assert domain.classes(mk_log(b), zero) == {domain.IN}
    inf = {a: frozenset({domain.Z}), b: frozenset({domain.IP})}
    assert domain.classes(mk_mul(a, b), inf) == {domain.NAN}
    assert domain.classes(mk_pow(b, -1), zero) == {domain.IP}  # 1 / +0
    low = {a: frozenset({domain.P}), b: frozenset({domain.WN})}
    assert domain.classes(mk_max(a, b), low) == {domain.P}


def _online_softmax_merge():
    a0, b0, a1, b1 = (ir.state_var(s, k) for k in (0, 1) for s in "ab")
    m = ir.raw_max([a0, b0])

    def rescaled(s0, s1):
        shift = ir.raw_add([m, ir.raw_mul([ir.MINUS_ONE, s0])])
        return ir.raw_mul([ir.raw_exp(ir.raw_mul([ir.MINUS_ONE, shift])), s1])

    return ir.raw_add([rescaled(a0, a1), rescaled(b0, b1)])


def test_online_softmax_is_finite_against_an_empty_side():
    merge = _online_softmax_merge()
    cls = domain.empty_side(merge, "b", (ir.NEG_INF_C, ir.ZERO), [domain.ANY, domain.POS])
    assert not domain.bad(cls)


def test_a_shift_by_log_of_an_empty_sum_meets_zero_as_nan():
    """``Σq · (lse_m - lse_b)`` with ``lse_b = mx_b + log den_b`` on an empty side."""
    a0, a1, a2 = (ir.state_var("a", k) for k in range(3))
    b0, b1, b2 = (ir.state_var("b", k) for k in range(3))
    lse_b = ir.raw_add([b0, ir.raw_log(b1)])
    lse_m = ir.raw_add([ir.raw_max([a0, b0]), ir.raw_log(a1)])
    term = ir.raw_mul([b2, ir.raw_add([lse_m, ir.raw_mul([ir.MINUS_ONE, lse_b])])])
    cls = domain.empty_side(
        term, "b", (ir.NEG_INF_C, ir.ZERO, ir.ZERO), [domain.ANY, domain.POS, domain.ANY]
    )
    assert domain.NAN in cls


def test_a_mean_of_an_empty_part_is_nan():
    total, count = ir.state_var("b", 0), ir.state_var("b", 1)
    mean = ir.raw_mul([total, ir.raw_pow(count, -1)])
    cls = domain.empty_side(mean, "b", (ir.ZERO, ir.ZERO), [domain.ANY, domain.POS])
    assert cls == {domain.NAN}


if __name__ == "__main__":
    tvm.testing.main()
