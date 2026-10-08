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
"""Deriving a reducer (leaf, merge, identity, epilogues) from state definitions."""

import tvm.testing
from tvm import te
from tvm.te.superopt.dims import DimTable
from tvm.te.superopt.reducer.derive import atoms_in, derive, domains
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import (
    mk_add,
    mk_card,
    mk_div,
    mk_exp,
    mk_max,
    mk_mul,
    mk_pow,
    mk_reduce,
    mk_sub,
    subst_domain,
)


def _axis():
    """A fresh symbolic reduction axis and its domains ``(axis, full, R)``."""
    axis = DimTable().key(te.var("n"))
    full, R, _, _ = domains(axis)
    return axis, full, R


def _softmax_value(R):
    """States of ``Σ_j softmax(S)_j V_j`` on a sub-range: running max, mass and weighted sum."""
    i0, i1, j = ir.idx("i0"), ir.idx("i1"), ir.bidx(0)
    s, v = ir.elem(0, (i0, j)), ir.elem(1, (j, i1))
    mx = mk_reduce("max", R, 0, s)
    den = mk_reduce("sum", R, 0, mk_exp(mk_sub(s, mx)))
    num = mk_reduce("sum", R, 0, mk_mul(mk_exp(mk_sub(s, mx)), v))
    return mx, den, num


def test_epilogues_read_the_states_at_their_own_coordinates():
    axis, full, R = _axis()
    mx, den, num = _softmax_value(R)
    required = tuple(subst_domain(x, {R: full}) for x in (den, num))
    spec = derive((mx, den, num), required, axis)
    assert spec is not None and spec.arity == 3
    i0, i1 = ir.idx("i0"), ir.idx("i1")
    assert spec.epilogues == (ir.elem(-2, (i0,)), ir.elem(-3, (i0, i1)))
    # the online-softmax merge: both sides are rescaled to the merged max
    a, b = (lambda k: ir.state_var("a", k)), (lambda k: ir.state_var("b", k))
    m = mk_max(a(0), b(0))
    ra, rb = mk_exp(mk_sub(a(0), m)), mk_exp(mk_sub(b(0), m))
    assert spec.merge[0] is m
    assert spec.merge[1] is mk_add(mk_mul(a(1), ra), mk_mul(b(1), rb))
    assert spec.merge[2] is mk_add(mk_mul(a(2), ra), mk_mul(b(2), rb))
    assert spec.identity == (ir.NEG_INF_C, ir.ZERO, ir.ZERO)
    assert spec.leaves[1] is ir.ONE  # exp(s - s)


def test_normalised_output_is_an_epilogue_of_unnormalised_states():
    axis, full, R = _axis()
    mx, den, num = _softmax_value(R)
    target = subst_domain(mk_div(num, den), {R: full})
    spec = derive((mx, den, num), (target,), axis)
    assert spec is not None
    i0, i1 = ir.idx("i0"), ir.idx("i1")
    assert spec.epilogues == (mk_div(ir.elem(-3, (i0, i1)), ir.elem(-2, (i0,))),)


def test_state_nobody_reads_is_rejected():
    axis, full, R = _axis()
    mx, den, num = _softmax_value(R)
    extra = mk_reduce("sum", R, 0, ir.elem(0, (ir.idx("i0"), ir.bidx(0))))
    required = (subst_domain(num, {R: full}),)
    stats: dict = {}
    assert derive((mx, den, num, extra), required, axis, stats) is None
    assert stats.get("derive:irrelevant_state") == 1
    # den feeds nothing either once only the weighted sum is wanted
    assert derive((mx, den, num), required, axis) is None
    assert derive((mx, num), required, axis) is not None


def test_required_tensor_outside_the_states_is_uncovered():
    axis, full, R = _axis()
    mx, den, _ = _softmax_value(R)
    other = mk_reduce("sum", full, 0, ir.elem(2, (ir.idx("i0"), ir.bidx(0))))
    stats: dict = {}
    assert derive((mx, den), (other,), axis, stats) is None
    assert stats.get("derive:uncovered") == 1


def _moments(R):
    """Count, mean and central moments of ``x`` on a sub-range."""
    x = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    n = mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)

    def central(p):
        return mk_reduce("sum", R, 0, mk_pow(mk_sub(x, mean), p))

    return n, mean, central(2), central(3)


def test_welford_states_and_the_count_as_the_extent():
    axis, full, R = _axis()
    n, mean, m2, _ = _moments(R)
    required = tuple(subst_domain(x, {R: full}) for x in (mean, m2))
    spec = derive((n, mean, m2), required, axis)
    assert spec is not None
    i0 = ir.idx("i0")
    # over the whole axis the count is the extent: no epilogue reads its state
    assert spec.epilogues == (ir.elem(-2, (i0,)), ir.elem(-3, (i0,)))
    assert spec.leaves == (ir.ONE, ir.elem(0, (i0, ir.bidx(0))), ir.ZERO)
    assert spec.identity[0] is ir.ZERO


def test_third_moment_needs_the_second():
    axis, full, R = _axis()
    n, mean, m2, m3 = _moments(R)
    required = (subst_domain(m3, {R: full}),)
    assert derive((n, mean, m3), required, axis) is None  # Σx² has no state to come from
    spec = derive((n, mean, m2, m3), required, axis)
    assert spec is not None and spec.arity == 4


def test_state_made_of_two_reductions_is_solved_for_one_of_them():
    """``Σ_R (3 y + 2) = 3 Σ_R y + 2 |R|`` is one state: it merges by plain addition
    although neither reduction can be recovered from it."""
    axis, full, R = _axis()
    i0 = ir.idx("i0")
    y = ir.elem(0, (i0, ir.bidx(0)))
    s = mk_reduce("sum", R, 0, mk_add(mk_mul(ir.const(3), y), ir.const(2)))
    assert len(atoms_in(s, R)) == 2
    spec = derive((s,), (subst_domain(s, {R: full}),), axis)
    assert spec is not None
    assert spec.merge == (mk_add(ir.state_var("a", 0), ir.state_var("b", 0)),)
    assert spec.epilogues == (ir.elem(-1, (i0,)),)
    assert spec.leaves == (mk_add(mk_mul(ir.const(3), y), ir.const(2)),)
    # over the whole axis the count is the extent, so even Σ y comes back: (s - 2 n) / 3
    alone = subst_domain(mk_reduce("sum", R, 0, y), {R: full})
    (back,) = derive((s,), (alone,), axis).epilogues
    n = ir.shape_sym("n")
    assert back is mk_div(mk_sub(ir.elem(-1, (i0,)), mk_mul(ir.const(2), n)), ir.const(3))
    # a reduction it does not contain stays out of reach
    square = subst_domain(mk_reduce("sum", R, 0, mk_mul(y, y)), {R: full})
    assert derive((s,), (square,), axis) is None


if __name__ == "__main__":
    tvm.testing.main()
