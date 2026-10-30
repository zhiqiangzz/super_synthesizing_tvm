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
"""Equalities by proof, the laws of derived reducers, on non-empty parts."""

import itertools

import tvm.testing
from tvm import te
from tvm.te.superopt.dims import DimTable
from tvm.te.superopt.reducer.derive import (
    derive,
    domains,
    invariant_holds,
    merge_of,
    solve_atoms,
)
from tvm.te.superopt.reducer.verify import prove_equal
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
# equality by proof
# ---------------------------------------------------------------------------
def test_rational_identities_are_proven_by_a_common_denominator():
    """A degree-6 identity the canonical rules do not normalise: SymPy decides it."""
    a, b, c = _sv("a", 0), _sv("b", 0), _sv("c", 0)
    cube = mk_pow(mk_add(a, b, c), 3)
    lhs = mk_div(mk_mul(cube, cube), mk_pow(mk_add(a, b), 2))
    rhs = mk_div(mk_pow(mk_add(a, b, c), 6), mk_pow(mk_add(a, b), 2))
    assert prove_equal(lhs, rhs) in ("canonical", "sympy")
    off = mk_add(rhs, mk_div(ir.const(1), ir.const(10**12)))
    assert prove_equal(lhs, off) is None


def test_welford_merge_written_with_the_merged_mean_is_proven():
    """The printed form of a Welford merge against Σx² - (Σx)²/n: equal as rational functions."""
    ca, cb, ma, mb, sa, sb = (_sv(side, k) for k in (2, 0, 1) for side in "ab")
    m = mk_div(mk_add(mk_mul(ca, ma), mk_mul(cb, mb)), mk_add(ca, cb))
    printed = mk_add(
        sa, sb, mk_mul(ca, mk_pow(mk_sub(m, ma), 2)), mk_mul(cb, mk_pow(mk_sub(m, mb), 2))
    )
    total = mk_add(mk_mul(ca, ma), mk_mul(cb, mb))
    canonical = mk_sub(
        mk_add(sa, sb, mk_mul(ca, mk_pow(ma, 2)), mk_mul(cb, mk_pow(mb, 2))),
        mk_div(mk_pow(total, 2), mk_add(ca, cb)),
    )
    assert printed is not canonical
    assert prove_equal(printed, canonical) == "sympy"


def test_a_maximum_is_split_into_the_cases_of_which_argument_wins():
    a0, a1, b0, b1 = _sv("a", 0), _sv("a", 1), _sv("b", 0), _sv("b", 1)
    m = mk_max(a0, b0)
    written = mk_add(mk_mul(a1, mk_exp(mk_sub(a0, m))), mk_mul(b1, mk_exp(mk_sub(b0, m))))
    # the same with the larger side written out in each case
    other = mk_add(
        mk_mul(a1, mk_exp(mk_sub(a0, mk_max(b0, a0)))), mk_mul(b1, mk_exp(mk_sub(b0, m)))
    )
    assert prove_equal(written, other) is not None
    assert prove_equal(mk_max(a0, b0), a0) is None  # only where a0 wins


def test_what_is_not_proven_is_not_equal():
    """Nothing is sampled: an equality that needs the sign of its variables is refused."""
    a, b = _sv("a", 0), _sv("b", 0)
    lhs = mk_log(mk_mul(a, b))  # log a + log b only for a, b > 0
    rhs = mk_add(mk_log(a), mk_log(b))
    assert lhs is not rhs
    assert prove_equal(lhs, rhs) is None


# ---------------------------------------------------------------------------
# laws on non-empty parts: from the invariant
# ---------------------------------------------------------------------------
def _derived_merge(states, R, axis):
    _, _, A, B = domains(axis)
    sol = solve_atoms(states, R)
    return tuple(merge_of(s, sol, R, A, B) for s in states), A, B


def test_nonempty_merge_laws_rest_on_the_invariant():
    axis, full, R = _axis()
    x = _x()
    states = (
        mk_reduce("max", R, 0, x),
        mk_reduce("sum", R, 0, mk_exp(mk_sub(x, mk_reduce("max", R, 0, x)))),
    )
    merge, A, B = _derived_merge(states, R, axis)
    assert invariant_holds(states, merge, R, A, B) is not None
    # a merge that is not the derived one breaks the invariant
    bad = (merge[0], mk_add(_sv("a", 1), _sv("b", 1)))
    assert invariant_holds(states, bad, R, A, B) is None


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
# states whose empty-part formulas are undefined
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


def test_loss_is_derived_without_defining_log_den_on_an_empty_part():
    axis, full, R = _axis()
    mx, den, sq, _, loss = _smoothed_cross_entropy(R)
    required = (subst_domain(loss, {R: full}),)
    direct = derive((mx, den, sq, loss), required, axis)
    assert direct is not None
    merge, A, B = _derived_merge((mx, den, sq, loss), R, axis)
    assert invariant_holds((mx, den, sq, loss), merge, R, A, B) is not None


if __name__ == "__main__":
    tvm.testing.main()
