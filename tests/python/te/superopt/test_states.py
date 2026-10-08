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
"""Candidate states: the program's own tensors, and what re-basing and hoisting add."""

import itertools

from chain_ops import OPERATORS

import tvm.testing
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.derive import domains
from tvm.te.superopt.reducer.states import Pool, _content, _scaled, synthesize
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
    recanonicalize,
    walk,
)


def _chain(name: str, which: int = 0):
    _, outs = OPERATORS[name].unfused()
    return discover_chains(outs)[0][which]


def _pool(name: str, count: bool = True) -> Pool:
    return Pool(_chain(name), count)


def _x(tensor: int = 0):
    """The first boundary tensor, read along the reduction."""
    return ir.elem(tensor, (ir.idx("i0"), ir.bidx(0)))


def _by_state(pool: Pool, state: ir.SymExpr):
    """The candidate that is ``state`` up to a rational factor."""
    norm = _scaled(state, _content(state))
    return next((c for c in pool.cands if _scaled(c.state, _content(c.state)) is norm), None)


def _names(sol) -> list[str]:
    return [c.name for c in sol.states]


# ---------------------------------------------------------------------------
# shift / scale: the members themselves are the states
# ---------------------------------------------------------------------------
def test_softmax_members_are_the_states_and_the_merge_is_rebased():
    pool = _pool("softmax")
    assert [(c.name, c.origin) for c in pool.cands if c.depth == 0] == [
        ("mx", "member"),
        ("den", "member"),
    ]
    first = next(synthesize(pool))
    assert _names(first) == ["mx", "den"] and not first.auxiliaries
    spec = first.spec
    assert spec.leaves == (_x(), ir.ONE) and spec.identity == (ir.NEG_INF_C, ir.ZERO)
    # printed: each side keeps its own sum and is rescaled by exp(own max - merged max)
    a, b = (lambda k: ir.state_var("a", k)), (lambda k: ir.state_var("b", k))
    m = ir.raw_max([a(0), b(0)])

    def rescaled(s):
        shift = ir.raw_add([m, ir.raw_mul([ir.MINUS_ONE, s(0)])])
        return ir.raw_mul([ir.raw_exp(ir.raw_mul([ir.MINUS_ONE, shift])), s(1)])

    assert spec.merge_print == (m, ir.raw_add([rescaled(a), rescaled(b)]))
    assert recanonicalize(spec.merge_print[1]) is spec.merge[1]


def test_attention_normalised_output_scales_by_the_normaliser():
    """The output re-bases against two contexts at once: shift for the max, scale for the mass."""
    pool = _pool("attention")
    out = next(c for c in pool.cands if c.name == "O")
    d = pool.rebaser.decompose(out)
    modes = {(pool.rebaser.by_pid[pid].name, mode) for _, _, pid, mode in d.corr}
    assert modes == {("row_max", "shift"), ("den", "scale")}
    first = next(synthesize(pool))
    assert _names(first) == ["row_max", "den", "O"] and first.spec.merge_print is not None
    # the unnormalised accumulator of FlashAttention is there as well, by hoisting the mass
    acc = next(c for c in pool.cands if c.origin == "hoist" and c.source == "O")
    assert any(set(_names(s)) >= {"row_max", "den", acc.name} for s in synthesize(pool))


# ---------------------------------------------------------------------------
# closure: states the program never computes
# ---------------------------------------------------------------------------
def _moments(R):
    x, n = _x(), mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)
    return n, mean, [mk_reduce("sum", R, 0, mk_pow(mk_sub(x, mean), p)) for p in (2, 3, 4)]


def test_variance_needs_the_count_and_gets_welford():
    pool = _pool("variance")
    _, R, _, _ = domains(pool.chain.axis)
    count = _by_state(pool, mk_card(R))
    assert count is not None and count.origin == "extent"  # the program divides by it
    first = next(synthesize(pool))
    assert sorted(_names(first)) == ["count", "mean", "ss"]
    assert [c.name for c in first.auxiliaries] == ["count"]
    spec = first.spec
    k = {c.name: n for n, c in enumerate(first.states)}
    assert spec.leaves[k["ss"]] is ir.ZERO and spec.leaves[k["count"]] is ir.ONE
    # the sum of squares is corrected around the merged mean, never expanded into Σx²
    a, b = (lambda n: ir.state_var("a", n)), (lambda n: ir.state_var("b", n))
    na, nb, ma, mb = a(k["count"]), b(k["count"]), a(k["mean"]), b(k["mean"])
    mean = mk_div(mk_add(mk_mul(ma, na), mk_mul(mb, nb)), mk_add(na, nb))
    want = mk_add(
        a(k["ss"]),
        b(k["ss"]),
        mk_mul(na, mk_pow(mk_sub(mean, ma), 2)),
        mk_mul(nb, mk_pow(mk_sub(mean, mb), 2)),
    )
    assert recanonicalize(spec.merge_print[k["ss"]]) is want


def test_third_and_fourth_moments_lift_the_lower_ones():
    pool = _pool("moment3")
    _, R, _, _ = domains(pool.chain.axis)
    n, mean, (m2, m3, m4) = _moments(R)
    lifted = _by_state(pool, m2)
    assert lifted is not None and lifted.origin == "closure" and lifted.source == "s3"
    first = next(synthesize(pool, max_states=4))
    assert {c.origin for c in first.states} == {"member", "extent", "closure"}
    assert set(c.state for c in first.states) >= {m3, lifted.state}
    # without the second moment only raw power sums are left
    assert all(s.hoists for s in synthesize(pool, max_states=3))

    pool = _pool("moment4")
    _, R, _, _ = domains(pool.chain.axis)
    _, _, (m2, m3, _) = _moments(R)
    second, third = _by_state(pool, m2), _by_state(pool, m3)
    assert second is not None and third is not None
    assert {second.origin, third.origin} == {"closure"}
    first = next(synthesize(pool, max_states=5))
    assert len(first.states) == 5 and first.spec.merge_print is not None


def test_every_derived_state_comes_out_of_the_program():
    """No candidate is invented: each is a member (a tensor of the program or a
    context it computes in place), the extent, or a coefficient of one."""
    for name in ("softmax", "attention", "moment4", "covariance", "softmax_entropy"):
        pool = _pool(name)
        by_name = {c.name: c for c in pool.cands}
        for c in pool.cands:
            assert c.origin in ("member", "context", "extent", "closure", "hoist"), c.name
            if c.origin == "closure":
                d = pool.rebaser.decompose(by_name[c.source])
                coeffs = [pool.rebaser.coeff_state(t) for t in d.all_terms() if t.side == "a"]
                assert any(_scaled(s, _content(s)) is c.state for s in coeffs), (name, c.name)
            if c.origin == "hoist":
                parent = by_name[c.source]
                ctx = pool.rebaser.contexts(parent)
                found = False
                for size in range(1, len(ctx) + 1):
                    for chosen in itertools.combinations(ctx, size):
                        for struct in pool._coefficients(parent, set(chosen)) or ():
                            state = pool._state(struct)
                            found |= _scaled(state, _content(state)) is c.state
                assert found, (name, c.name)


# ---------------------------------------------------------------------------
# hoisting: the context moves to the epilogue
# ---------------------------------------------------------------------------
def test_soft_cross_entropy_hoists_the_log_sum_exp():
    pool = _pool("soft_cross_entropy")
    _, R, _, _ = domains(pool.chain.axis)
    x, q = _x(0), _x(1)
    sq = _by_state(pool, mk_reduce("sum", R, 0, q))
    sqx = _by_state(pool, mk_reduce("sum", R, 0, mk_mul(q, x)))
    # Σq is what shifting log(den) leaves behind; Σqx only appears once the contexts are out
    assert sq is not None and sqx is not None and (sq.origin, sqx.origin) == ("closure", "hoist")
    wanted = {"mx", "den", sq.name, sqx.name}
    sol = next(s for s in synthesize(pool) if set(_names(s)) == wanted)
    assert sol.spec.merge_print is not None
    # both are plain sums; the loss is formed at the end: lse * Σq - Σqx
    k = {c.name: n for n, c in enumerate(sol.states)}
    a, b = (lambda n: ir.state_var("a", n)), (lambda n: ir.state_var("b", n))
    for name in (sq.name, sqx.name):
        assert sol.spec.merge[k[name]] is mk_add(a(k[name]), b(k[name]))
    (loss,) = sol.spec.epilogues
    assert isinstance(loss, ir.Add) and not any(isinstance(n, ir.Reduce) for n in walk(loss))


def test_entropy_hoists_the_normaliser_and_keeps_the_shift():
    pool = _pool("softmax_entropy")
    _, R, _, _ = domains(pool.chain.axis)
    x = _x()
    mx = mk_reduce("max", R, 0, x)
    t = mk_reduce("sum", R, 0, mk_mul(mk_sub(x, mx), mk_exp(mk_sub(x, mx))))
    cand = _by_state(pool, t)
    assert cand is not None and cand.origin == "hoist"
    sol = next(s for s in synthesize(pool) if set(_names(s)) == {"mx", "den", cand.name})
    assert sol.spec.merge_print is not None
    k = {c.name: n for n, c in enumerate(sol.states)}
    assert sol.spec.leaves[k[cand.name]] is ir.ZERO  # (x - x) exp(x - x)


def test_states_that_keep_the_programs_own_shape_come_first():
    """The expanded second moment only shows up once nothing re-based is left."""
    pool = _pool("variance")
    sols = list(synthesize(pool))
    hoisted = [n for n, s in enumerate(sols) if s.hoists]
    plain = [n for n, s in enumerate(sols) if not s.hoists]
    assert plain and hoisted and max(plain) < min(hoisted)
    raw = sols[hoisted[0]]
    assert len(raw.states) == 2  # Σx and Σx²: smaller, and still later


# ---------------------------------------------------------------------------
# the extent as a count or as a constant
# ---------------------------------------------------------------------------
def test_label_smoothing_reads_the_extent_as_a_constant():
    chain = _chain("label_smoothing")
    assert Pool.readings(chain) == [True, False]
    assert Pool.readings(_chain("softmax")) == [True]  # the extent is never used as a value
    # a/K as a/|R| would make the smoothed labels depend on the sub-range
    assert list(itertools.islice(synthesize(Pool(chain, count=True)), 1)) == []
    pool = Pool(chain, count=False)
    sols = [s for s in synthesize(pool) if {"mx", "den"} <= set(_names(s))]
    assert sols and len(sols[0].states) == 4 and sols[0].spec.merge_print is not None


# ---------------------------------------------------------------------------
# no finite lifting
# ---------------------------------------------------------------------------
def test_context_under_an_absolute_value_is_stuck():
    pool = _pool("mean_abs_deviation")
    assert pool.stuck == ["dev"] and pool.unrebased == ["dev"]
    assert list(synthesize(pool)) == []


def test_fifth_moment_is_beyond_the_expansion():
    pool = _pool("moment5")
    assert pool.unrebased == ["s5"] and pool.stuck == []  # hoisting still applies ...
    assert list(itertools.islice(synthesize(pool, max_states=4), 1)) == []  # ... with 5 raw sums


def test_max_of_scaled_scores_rebases():
    """``c * max(x)``: the positive factor does not hide the max from the decomposition."""
    pool = _pool("attention")
    row_max = pool.cands[0]
    d = pool.rebaser.decompose(row_max)
    assert d is not None and len(d.parts) == 2
    a, b = (lambda k: ir.state_var("a", k)), (lambda k: ir.state_var("b", k))
    assert next(synthesize(pool)).spec.merge[0] is mk_max(a(0), b(0))


if __name__ == "__main__":
    tvm.testing.main()
