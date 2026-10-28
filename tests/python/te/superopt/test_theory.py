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
"""The judgement that comes before any derivation: does a single pass exist, with how
many states? Read off the program by algebra; nothing is sampled."""

from chain_ops import OPERATORS

import tvm.testing
from tvm import te
from tvm import tirx as tir
from tvm.te.superopt import Judgement, fuse, judge
from tvm.te.superopt.reducer import discover_chains, members_of
from tvm.te.superopt.reducer import judge as judge_chain

F = "float32"


def _rows(name="X", like=None):
    shape = (te.var("n"), te.var("m")) if like is None else like.shape
    return te.placeholder(shape, name=name, dtype=F)


def _red(X, f, name, reducer=te.sum):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: reducer(f(i, j), axis=j), name=name)


def _mean(X):
    total = _red(X, lambda i, j: X[i, j], "total")
    return te.compute((X.shape[0],), lambda i: total[i] / X.shape[1].astype(F), name="mean")


def _sigma(X, mean):
    ss = _red(X, lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    return te.compute((X.shape[0],), lambda i: tir.sqrt(ss[i] / X.shape[1].astype(F)), name="sigma")


def _judged(name: str) -> Judgement:
    (judgement,) = judge(OPERATORS[name].unfused()[1])
    return judgement


# ---------------------------------------------------------------------------
# sums: a factor that does not move with the axis leaves the reduction
# ---------------------------------------------------------------------------
def test_sum_is_judged_by_the_reductions_left_when_its_contexts_are_taken_out():
    """``Σ (x - mean)²`` is ``Σ x² - 2 mean Σ x + mean² Σ 1``: two sums and the count."""
    variance = _judged("variance")
    assert variance.fusible and variance.states == 3 and variance.counted
    assert set(variance.reductions) == {"Σ_j X[i, j]", "Σ_j X[i, j]^2"}
    assert variance.summary().startswith("a single pass with 3 state(s) exists, keeping ")
    softmax = _judged("softmax")  # exp(x - mx) = exp(-mx) exp(x); the maximum is a state too
    assert softmax.states == 2 and not softmax.counted
    assert set(softmax.reductions) == {"max_j X[i, j]", "Σ_j exp(X[i, j])"}
    assert _judged("covariance").states == 4 and _judged("pearson").states == 6


def test_states_judged_enough_are_the_states_of_the_reducers_taken():
    """What is counted is the reductions a context is taken out of, with the contexts
    themselves: the maximum of a softmax stays although it cancels in the entropy."""
    for name, states in (("softmax_entropy", 3), ("attention", 3), ("zscore_max", 4)):
        assert _judged(name).states == states == OPERATORS[name].states
    # a reducer may still take fewer: one state can hold a combination of several sums
    smoothing = _judged("label_smoothing")
    assert smoothing.states == 6 > OPERATORS["label_smoothing"].states == 4


def test_degree_and_domain_do_not_matter_to_algebra():
    """An eighth central moment takes eight sums and the count; a body defined for
    ``|x| <= 1`` only is split like any other. Nothing is evaluated."""
    X = _rows()
    mean = _mean(X)
    eighth = _red(X, lambda i, j: tir.power(X[i, j] - mean[i], tir.const(8.0, F)), "s8")
    (judgement,) = judge(eighth)
    assert judgement.states == 9 and "Σ_j X[i, j]^8" in judgement.reductions
    one = tir.const(1.0, F)
    bounded = _red(
        X, lambda i, j: tir.sqrt(one - X[i, j] * X[i, j]) * (X[i, j] - mean[i]), "bounded"
    )
    (judgement,) = judge(bounded)
    assert judgement.fusible and judgement.states == 4
    assert "Σ_j sqrt(1 - X[i, j]^2)" in judgement.reductions


def test_sum_whose_context_stays_inside_has_no_single_pass_that_is_known():
    """Under an absolute value, as a factor under ``exp``: linearity lets nothing out."""
    deviation = _judged("mean_abs_deviation")
    assert not deviation.fusible and deviation.states is None
    assert deviation.reductions == ("Σ_j X[i, j]",)  # the mean itself is one pass
    (stuck,) = deviation.stuck
    assert (stuck.member, stuck.contexts) == ("dev", ("mean",))
    assert stuck.reduction.startswith("Σ_j max(") and "mean[i] - X[i, j]" in stuck.reduction
    assert deviation.summary() == (
        f"no single pass is known: dev reads mean inside {stuck.reduction}, "
        "and no rule takes it out of the reduction"
    )
    (stuck,) = _judged("logit_norm").stuck
    assert (stuck.member, stuck.contexts) == ("den", ("nrm",))
    # of two contexts the one that can leave does: exp((x - mean) / sigma)
    (stuck,) = _judged("standardised_softmax").stuck
    assert stuck.contexts == ("sigma",) and "mean" not in stuck.reduction
    assert stuck.reduction == "Σ_j exp(X[i, j] / sigma[i])"


# ---------------------------------------------------------------------------
# maxima: a function that grows with its argument leaves the reduction
# ---------------------------------------------------------------------------
def test_maximum_is_judged_by_the_extremes_left():
    """A shift, a factor that cannot be negative, a function that grows: what stays
    under the maximum is the data alone. An even power takes both extremes."""
    score = _judged("zscore_max")  # (x - mean) / sigma, sigma a square root
    assert "max_j X[i, j]" in score.reductions and score.states == 4
    deviation = _judged("max_abs_deviation")
    assert {"max_j X[i, j]", "max_j -X[i, j]"} <= set(deviation.reductions)
    assert _judged("peak_power").reductions == deviation.reductions
    ratio = _judged("peak_to_geomean")  # exp(l - mean)
    assert set(ratio.reductions) == {"Σ_j L[i, j]", "max_j L[i, j]"} and ratio.states == 3


def test_maximum_whose_context_stays_inside_has_no_single_pass_that_is_known():
    X = _rows()
    mean = _mean(X)
    moving = _red(X, lambda i, j: mean[i] * X[i, j] - X[i, j] * X[i, j], "out", te.max)
    (judgement,) = judge(moving)  # attained at the element nearest mean / 2
    assert not judgement.fusible and judgement.stuck[0].contexts == ("mean",)
    assert not _judged("min_sq_deviation").fusible
    # a factor of unknown sign: the largest x or the smallest, and the rules do not say which
    G = _rows("G", like=X)
    g = _red(G, lambda i, j: G[i, j], "g")
    scaled = _red(X, lambda i, j: g[i] * X[i, j], "out", te.max)
    (judgement,) = judge(scaled)
    (stuck,) = judgement.stuck
    assert stuck.contexts == ("g",) and stuck.reduction.startswith("max_j ")


def test_not_known_is_not_the_same_as_impossible():
    """``Σ |x - max x|`` is ``n max x - Σ x`` because a maximum is never below its
    elements: a single pass exists, the rules do not see it and say so in those words."""
    judgement = _judged("abs_from_max")
    assert not judgement.fusible and judgement.summary().startswith("no single pass is known")


def test_a_polynomial_that_never_falls_is_taken_at_the_largest_element():
    """``max ((x - mean)³ + (x - mean))``: ``u³ + u`` never falls, its derivative ``3 u² + 1``
    has no real root, so it is attained at the largest ``x``."""
    judgement = _judged("monotone_sum")
    assert judgement.fusible and judgement.states == 3


# ---------------------------------------------------------------------------
# the judgement in front of the derivation
# ---------------------------------------------------------------------------
def test_judge_reads_the_chains_off_a_program():
    _, outs = OPERATORS["sinkhorn_iteration"].unfused()
    first, second = judge(outs)
    assert first.states == second.states == 2
    X = _rows()
    assert judge(_red(X, lambda i, j: X[i, j], "plain")) == []  # one reduction is no chain
    (chain,) = discover_chains(OPERATORS["variance"].unfused()[1])[0]
    assert judge_chain(chain, members_of(chain)) == _judged("variance")


def test_nothing_is_derived_for_a_chain_without_a_known_single_pass():
    ins, outs = OPERATORS["logit_norm"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert not res.fused and report.rejected == [] and report.solution is None
    assert report.reason == report.theory.summary()
    assert res.summary().count("no single pass is known") == 1  # said once, as the reason


def test_states_chosen_are_as_many_as_judged_enough():
    """Six states for a correlation, five for a fourth moment: nobody has to say so."""
    for name in ("pearson", "moment4", "papr"):
        op = OPERATORS[name]
        ins, outs = op.unfused()
        res = fuse(outs, ins)
        (report,) = res.chains
        assert report.fused and len(report.states) == op.states == report.theory.states
        assert f"theory: a single pass with {op.states} state(s) exists" in res.summary()


def test_single_pass_that_exists_and_is_not_taken_is_reported_as_such():
    """``Σ (a - r q)²`` with ``r = Σ q a / Σ q²``: three states are enough, but ``r`` is
    0 / 0 on a part where ``q`` is 0, so it is taken out and raw sums are left, which cancel."""
    ins, outs = OPERATORS["normalised_projection"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert not res.fused and report.theory.states == 3
    steps = [st.summary() for st in report.selection.steps]
    assert (
        "hoist r out of rem: rem: it divides by qq[i] (through r), which is 0 where Q[i, j] is"
        in steps
    )
    (refused,) = report.rejected
    assert "REJECTED" in refused.why and "offset" in refused.why  # they cancel on shifted data
    assert "theory: a single pass with 3 state(s) exists" in res.summary()


if __name__ == "__main__":
    tvm.testing.main()
