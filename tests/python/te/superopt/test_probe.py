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
"""Probes: whether a finite single pass exists, read off the program without deriving one."""

from decimal import Decimal

from chain_ops import OPERATORS

import tvm.testing
from tvm import te
from tvm import tirx as tir
from tvm.te.superopt import Theory, Verdict, fuse, judge
from tvm.te.superopt.reducer import discover_chains
from tvm.te.superopt.reducer import judge as judge_chain
from tvm.te.superopt.reducer.probe import _rank

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


def _verdict(out, member="out") -> Verdict:
    (theory,) = judge(out)
    verdict = theory.verdict(member)
    assert verdict is not None, theory.summary()
    return verdict


# ---------------------------------------------------------------------------
# sums: the rank of the body over elements and contexts
# ---------------------------------------------------------------------------
def test_rank_of_a_sum_is_the_number_of_sums_to_accumulate():
    X = _rows()
    mx = _red(X, lambda i, j: X[i, j], "mx", te.max)
    den = _red(X, lambda i, j: tir.exp(X[i, j] - mx[i]), "out")
    v = _verdict(den)
    assert (v.kind, v.contexts, v.fusible, v.rank) == ("sum", ("mx",), True, 1)

    for power, rank in ((2, 3), (3, 4), (5, 6)):  # 1, x, .., x^power
        X = _rows()
        mean = _mean(X)
        p = tir.const(float(power), F)
        moment = _red(X, lambda i, j, mean=mean, p=p: tir.power(X[i, j] - mean[i], p), "out")
        v = _verdict(moment)
        assert v.fusible and v.rank == rank, v.summary()


def test_fifth_moment_exists_although_it_is_not_derived():
    """The derivation stops at degree four; the probes do not depend on it."""
    op = OPERATORS["moment5"]
    ins, outs = op.unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert not report.fused and report.theory.fusible is True
    assert report.theory.verdict("s5").rank == 6
    assert "probes: s5 (sum reading mean): separates into 6 term(s)" in res.summary()


def test_a_context_that_scales_the_argument_of_exp_does_not_separate():
    X = _rows()
    ssq = _red(X, lambda i, j: X[i, j] * X[i, j], "ssq")
    nrm = te.compute((X.shape[0],), lambda i: tir.sqrt(ssq[i]), name="nrm")
    out = _red(X, lambda i, j: tir.exp(X[i, j] / nrm[i]), "out")
    v = _verdict(out)
    assert v.fusible is False and v.rank is None and "rank above 8" in v.note
    assert "does not separate" in v.summary()

    # the same context as a shift separates: exp(x - mean) = exp(x) exp(-mean)
    X = _rows()
    mean = _mean(X)
    v = _verdict(_red(X, lambda i, j: tir.exp(X[i, j] - mean[i]), "out"))
    assert v.fusible and v.rank == 1


def test_rank_is_taken_on_pairs_the_program_can_meet():
    """``|x - κ|`` has unbounded rank over all pairs; a maximum is never below an element,
    and there it is ``κ - x``: rank 2. About a mean the absolute value stays."""
    X = _rows()
    mx = _red(X, lambda i, j: X[i, j], "mx", te.max)
    v = _verdict(_red(X, lambda i, j: tir.abs(X[i, j] - mx[i]), "out"))
    assert v.fusible and v.rank == 2

    X = _rows()
    mean = _mean(X)
    v = _verdict(_red(X, lambda i, j: tir.abs(X[i, j] - mean[i]), "out"))
    assert v.fusible is False


def test_probes_are_drawn_on_the_domain_of_the_program():
    """``log(x / mean)`` is not defined on data of both signs: the probes move to positive data."""
    X = _rows()
    mean = _mean(X)
    v = _verdict(_red(X, lambda i, j: tir.log(X[i, j] / mean[i]), "out"))
    assert v.fusible and v.rank == 2  # log x and 1


# ---------------------------------------------------------------------------
# maxima and minima: the element that attains them
# ---------------------------------------------------------------------------
def test_a_maximum_is_decided_by_an_extreme_of_the_data():
    X = _rows()
    mean = _mean(X)
    sigma = _sigma(X, mean)
    v = _verdict(_red(X, lambda i, j: (X[i, j] - mean[i]) / sigma[i], "out", te.max))
    assert (v.kind, v.fusible, v.extremes) == ("max", True, ("the largest X",))
    assert set(v.contexts) == {"mean", "sigma"}

    X = _rows()
    mean = _mean(X)
    v = _verdict(_red(X, lambda i, j: tir.abs(X[i, j] - mean[i]), "out", te.max))
    assert v.fusible and set(v.extremes) == {"the largest X", "the smallest X"}

    X = _rows()
    mean = _mean(X)
    sigma = _sigma(X, mean)
    v = _verdict(_red(X, lambda i, j: (X[i, j] - mean[i]) / sigma[i], "out", te.min))
    assert (v.kind, v.fusible, v.extremes) == ("min", True, ("the smallest X",))


def test_a_maximum_whose_winner_moves_with_the_context_is_not_fusible():
    """``max_j (κ x_j - x_j²)`` is attained by the element closest to ``κ / 2``. As a sum the
    same body has rank 2 and is fusible: the rank says nothing about a maximum."""
    X = _rows()
    mean = _mean(X)
    body = lambda i, j: mean[i] * X[i, j] - X[i, j] * X[i, j]  # noqa: E731
    v = _verdict(_red(X, body, "out", te.max))
    assert v.fusible is False and "probe elements attain it" in v.note

    X = _rows()
    mean = _mean(X)
    body = lambda i, j: mean[i] * X[i, j] - X[i, j] * X[i, j]  # noqa: E731
    v = _verdict(_red(X, body, "out"))
    assert v.fusible and v.rank == 2


def test_smallest_absolute_deviation_is_no_extreme():
    """``min_j |x_j - mean|`` is attained by the element nearest to the mean: any of them."""
    X = _rows()
    mean = _mean(X)
    out = _red(X, lambda i, j: tir.abs(X[i, j] - mean[i]), "out", te.min)
    v = _verdict(out)
    assert (v.kind, v.fusible) == ("min", False) and "moves with mean" in v.summary()
    res = fuse(out, [X])
    assert not res.fused and "(no finite lifting found)" in res.chains[0].reason


def test_extreme_of_a_combination_of_inputs():
    """``max_j (a_j + b_j) - mean``: one element decides, though no input is extreme at it."""
    A = _rows("A")
    B = _rows("B", like=A)
    mean = _mean(A)
    v = _verdict(_red(A, lambda i, j: A[i, j] + B[i, j] - mean[i], "out", te.max))
    assert v.fusible and v.rank == 1


# ---------------------------------------------------------------------------
# the verdict of a whole chain, and where it is reported
# ---------------------------------------------------------------------------
def test_judge_reads_the_chains_off_a_program():
    ins, outs = OPERATORS["attention"].unfused()
    (theory,) = judge(outs)
    assert isinstance(theory, Theory) and theory.fusible is True
    assert [(v.member, v.rank) for v in theory.verdicts] == [("den", 1), ("O", 1)]
    assert theory.verdict("row_max") is None  # reads no context: nothing to wait for

    first, second = judge(OPERATORS["sinkhorn_iteration"].unfused()[1])
    assert first.fusible and second.fusible

    X = _rows()
    assert judge(_red(X, lambda i, j: X[i, j], "total")) == []  # no chain


def test_one_reduction_that_does_not_separate_decides_the_chain():
    ins, outs = OPERATORS["standardised_softmax"].unfused()
    (theory,) = judge(outs)
    assert theory.fusible is False
    assert theory.verdict("ss").fusible and theory.verdict("zden").fusible is False


def test_unfused_report_says_whether_anything_exists():
    """The same failure to derive, told apart by the probes."""
    ins, outs = OPERATORS["abs_from_max"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert not report.fused and report.theory.fusible
    assert "a finite lifting exists by the probes, none was derived" in report.reason
    assert "probes: gap (sum reading mx): separates into 2 term(s)" in res.summary()

    ins, outs = OPERATORS["mean_abs_deviation"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert report.theory.fusible is False and "(no finite lifting found)" in report.reason
    assert "probes: dev (sum reading mean): does not separate" in res.summary()

    # a fused chain is not probed unless asked
    ins, outs = OPERATORS["variance"].unfused()
    res = fuse(outs, ins)
    assert "probes" not in res.summary() and res.chains[0].theory.fusible


def test_probes_agree_between_runs():
    ins, outs = OPERATORS["softmax_entropy"].unfused()
    (chain,), _ = discover_chains(outs)
    assert judge_chain(chain) == judge_chain(chain)
    assert judge_chain(chain, seed=3).fusible is True


def test_rank_by_elimination():
    one, two = Decimal(1), Decimal(2)
    assert _rank([[one, two], [two, Decimal(4)]]) == 1
    assert _rank([[one, two], [two, Decimal(5)]]) == 2
    assert _rank([[Decimal(0), Decimal(0)]]) == 0
    tiny = Decimal(10) ** -150  # far below the precision the rank is decided at
    assert _rank([[one, two], [two, Decimal(4) + tiny]]) == 1


if __name__ == "__main__":
    tvm.testing.main()
