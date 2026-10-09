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
"""Monoid laws of derived reducers and what they cannot see about floating point."""

import itertools

import tvm.testing
from tvm import te
from tvm.te.superopt.dims import DimTable
from tvm.te.superopt.reducer.derive import derive, domains
from tvm.te.superopt.reducer.verify import (
    _SYMMETRY,
    check_laws,
    identity_law,
    identity_safe,
    numeric_equal,
    symmetry_laws,
)
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
    mk_sub,
    subst_domain,
)


def _sv(side, k):
    return ir.state_var(side, k)


def _axis():
    axis = DimTable().key(te.var("n"))
    full, R, _, _ = domains(axis)
    return axis, full, R


def _x():
    return ir.elem(0, (ir.idx("i0"), ir.bidx(0)))


# ---------------------------------------------------------------------------
# laws
# ---------------------------------------------------------------------------
def test_laws_sum_pair_and_rejections():
    a0, a1, b0, b1 = _sv("a", 0), _sv("a", 1), _sv("b", 0), _sv("b", 1)
    assert check_laws((mk_add(a0, b0), mk_add(a1, b1)), (ir.ZERO, ir.ZERO)) == "canonical"
    assert check_laws((mk_sub(a0, b0),), (ir.ZERO,)) is None  # not commutative
    assert check_laws((mk_add(a0, b0),), (ir.ONE,)) is None  # wrong identity
    assert check_laws((mk_max(a0, b0),), (ir.NEG_INF_C,)) == "canonical"
    assert check_laws((mk_max(a0, b0),), (ir.ZERO,)) is None


def test_laws_online_softmax_reference_merge():
    a0, a1, a2 = (_sv("a", k) for k in range(3))
    b0, b1, b2 = (_sv("b", k) for k in range(3))
    m = mk_max(a0, b0)
    ra, rb = mk_exp(mk_sub(a0, m)), mk_exp(mk_sub(b0, m))
    merge = (m, mk_add(mk_mul(a1, ra), mk_mul(b1, rb)), mk_add(mk_mul(a2, ra), mk_mul(b2, rb)))
    assert check_laws(merge, (ir.NEG_INF_C, ir.ZERO, ir.ZERO)) in ("canonical", "case-split")
    # breaking the rescale breaks associativity
    bad = (m, mk_add(mk_mul(a1, ra), b1), mk_add(mk_mul(a2, ra), mk_mul(b2, rb)))
    assert check_laws(bad, (ir.NEG_INF_C, ir.ZERO, ir.ZERO)) is None


def test_rational_identities_are_decided_exactly():
    """A degree-6 identity whose float64 residue is far above any fixed tolerance."""
    a, b, c = _sv("a", 0), _sv("b", 0), _sv("c", 0)
    cube = mk_pow(mk_add(a, b, c), 3)
    lhs = mk_div(mk_mul(cube, cube), mk_pow(mk_add(a, b), 2))
    rhs = mk_div(mk_pow(mk_add(a, b, c), 6), mk_pow(mk_add(a, b), 2))
    assert numeric_equal(lhs, rhs)
    off = mk_add(rhs, mk_div(ir.const(1), ir.const(10**12)))
    assert not numeric_equal(lhs, off)


def test_log_arguments_are_sampled_positive():
    a, b = _sv("a", 0), _sv("b", 0)
    lhs = mk_log(mk_mul(a, b))  # not split by the canonical rules: signs unknown
    rhs = mk_add(mk_log(a), mk_log(b))
    assert lhs is not rhs
    assert numeric_equal(lhs, rhs)
    assert not numeric_equal(lhs, mk_add(rhs, ir.const(1)))


def _moments(R):
    x, n = _x(), mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)
    return (n, mean, *(mk_reduce("sum", R, 0, mk_pow(mk_sub(x, mean), p)) for p in (2, 3)))


def test_third_moment_laws_do_not_depend_on_slot_order():
    axis, full, R = _axis()
    states = _moments(R)
    required = (subst_domain(states[3], {R: full}),)
    for order in itertools.permutations(states):
        assert derive(order, required, axis) is not None


# ---------------------------------------------------------------------------
# the identity element in floating point
# ---------------------------------------------------------------------------
def _smoothed_cross_entropy(R):
    """``Σ_j q_j (lse - x_j)`` and what it is made of, on a sub-range."""
    x = _x()
    q = ir.elem(1, (ir.idx("i0"), ir.bidx(0)))
    mx = mk_reduce("max", R, 0, x)
    den = mk_reduce("sum", R, 0, mk_exp(mk_sub(x, mx)))
    lse = mk_add(mk_log(den), mx)
    sq = mk_reduce("sum", R, 0, q)
    sqx = mk_reduce("sum", R, 0, mk_mul(q, x))
    loss = mk_sub(mk_mul(lse, sq), sqx)
    return mx, den, sq, sqx, loss


def test_loss_as_a_state_obeys_the_laws_but_not_the_arithmetic():
    axis, full, R = _axis()
    mx, den, sq, sqx, loss = _smoothed_cross_entropy(R)
    required = (subst_domain(loss, {R: full}),)
    # the loss itself as a state: its merge needs log-sum-exp of each side, which
    # is log(0) + lowest on a side that is still empty
    direct = derive((mx, den, sq, loss), required, axis)
    assert direct is not None
    assert not identity_safe(direct.merge, direct.identity)
    # Σq and Σqx instead: plain sums, the context only enters the epilogue
    split = derive((mx, den, sq, sqx), required, axis)
    assert split is not None
    assert identity_safe(split.merge, split.identity)


def test_online_softmax_and_welford_meet_the_identity():
    axis, full, R = _axis()
    x = _x()
    v = ir.elem(1, (ir.bidx(0), ir.idx("i1")))
    mx = mk_reduce("max", R, 0, x)
    den = mk_reduce("sum", R, 0, mk_exp(mk_sub(x, mx)))
    num = mk_reduce("sum", R, 0, mk_mul(mk_exp(mk_sub(x, mx)), v))
    spec = derive((mx, den, num), (subst_domain(mk_div(num, den), {R: full}),), axis)
    assert identity_safe(spec.merge, spec.identity)
    n, mean, m2, _ = _moments(R)
    spec = derive((n, mean, m2), (subst_domain(m2, {R: full}),), axis)
    assert identity_safe(spec.merge, spec.identity)
    # a wrong identity is caught here as well as by the laws
    assert not identity_safe(spec.merge, (ir.ONE, ir.ZERO, ir.ZERO))


def test_symmetry_is_judged_once_per_merge():
    """Commutativity and associativity do not depend on the identity element."""
    a, b = ir.state_var("a", 0), ir.state_var("b", 0)
    merge = (mk_max(a, b),)
    _SYMMETRY.pop(merge, None)
    assert symmetry_laws(merge) == "canonical" and merge in _SYMMETRY
    assert identity_law(merge, (ir.NEG_INF_C,)) == "canonical"
    assert identity_law(merge, (ir.ZERO,)) is None  # max(a, 0) is not a
    assert check_laws(merge, (ir.NEG_INF_C,)) == "canonical"
    assert check_laws(merge, (ir.ZERO,)) is None
    lopsided = (mk_sub(a, b),)
    assert symmetry_laws(lopsided) is None and check_laws(lopsided, (ir.ZERO,)) is None


if __name__ == "__main__":
    tvm.testing.main()
