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
"""Precision gate: static exp / domain checks and the differential test."""

import numpy as np
import te_programs

import tvm.testing
from tvm import te
from tvm import tirx as tir
from tvm.te.superopt.accuracy import (
    AccuracyConfig,
    AccuracyGate,
    _OpLower,
    _Scope,
    bounded_above,
    positive_inputs,
    references,
    static_issues,
)
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import mk_add, mk_max, mk_mul, mk_neg, mk_sub


# ---------------------------------------------------------------------------
# programs over shared placeholders (the gate compares programs on the same inputs)
# ---------------------------------------------------------------------------
def _softmax_value(S, V, stable: bool):
    n, m = S.shape
    d = V.shape[1]
    if stable:
        j1 = te.reduce_axis((0, m), "j")
        mx = te.compute((n,), lambda i: te.max(S[i, j1], axis=j1), name="mx")
        e = te.compute((n, m), lambda i, j: tir.exp(S[i, j] - mx[i]), name="e")
    else:
        e = te.compute((n, m), lambda i, j: tir.exp(S[i, j]), name="e")
    j2 = te.reduce_axis((0, m), "j")
    den = te.compute((n,), lambda i: te.sum(e[i, j2], axis=j2), name="den")
    j3 = te.reduce_axis((0, m), "j")
    num = te.compute((n, d), lambda i, k: te.sum(e[i, j3] * V[j3, k], axis=j3), name="num")
    return te.compute((n, d), lambda i, k: num[i, k] / den[i], name="O")


def _variance_two_pass(X):
    n, m = X.shape
    m_f = m.astype("float32")
    j1 = te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m_f, name="mean")
    j2 = te.reduce_axis((0, m), "j")
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - mean[i]) * (X[i, j2] - mean[i]), axis=j2), name="ss"
    )
    return te.compute((n,), lambda i: ss[i] / m_f, name="var")


def _variance_one_pass(X):
    """``Σx²/n - (Σx/n)²``: equal over the reals, cancels catastrophically in floats."""
    n, m = X.shape
    m_f = m.astype("float32")
    j1 = te.reduce_axis((0, m), "j")
    s1 = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="s1")
    j2 = te.reduce_axis((0, m), "j")
    s2 = te.compute((n,), lambda i: te.sum(X[i, j2] * X[i, j2], axis=j2), name="s2")
    return te.compute((n,), lambda i: s2[i] / m_f - s1[i] * s1[i] / (m_f * m_f), name="var")


def _variance_welford(X):
    """Welford's reducer with Chan's merge around the difference of the means."""
    n, m = X.shape

    def merge(a, b):
        cnt = a[0] + b[0]
        delta = b[1] - a[1]
        return (cnt, a[1] + delta * b[0] / cnt, a[2] + b[2] + delta * delta * a[0] * b[0] / cnt)

    def ident(t0, t1, t2):
        return (tir.const(0.0, t0), tir.const(0.0, t1), tir.const(0.0, t2))

    red = te.comm_reducer(merge, ident, name="welford")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, "float32"), tir.const(0.0, "float32")
    cnt, _mean, m2 = te.compute((n,), lambda i: red((one, X[i, j], zero), axis=j), name="st")
    return te.compute((n,), lambda i: m2[i] / cnt[i], name="var")


def _normalise_by_sum(X, positive: bool):
    """``x / Σ_j exp(x)`` (positive divisor) or ``x / Σ_j x`` (sign unknown)."""
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    body = (
        (lambda i: te.sum(tir.exp(X[i, j]), axis=j))
        if positive
        else (lambda i: te.sum(X[i, j], axis=j))
    )
    s = te.compute((n,), body, name="s")
    return te.compute((n, m), lambda i, k: X[i, k] / s[i], name="y")


def _xs():
    n, m = te.var("n"), te.var("m")
    return te.placeholder((n, m), name="X", dtype="float32")


def _logsumexp_inline(X, shifted: bool):
    """``max`` and ``Σ exp(x - max)`` with the exponential written inside the reduction."""
    n, m = X.shape
    j1 = te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(X[i, j1], axis=j1), name="mx")
    j2 = te.reduce_axis((0, m), "j")
    arg = (lambda i: X[i, j2] - mx[i]) if shifted else (lambda i: X[i, j2])
    den = te.compute((n,), lambda i: te.sum(tir.exp(arg(i)), axis=j2), name="den")
    return mx, den


def _weighted_logsumexp(X, W):
    """``log Σ_j w_j exp(x_ij)`` written as ``Σ exp(x + log w)``: ``w`` must be positive."""
    n, m = X.shape
    Z = te.compute((n, m), lambda i, j: X[i, j] + tir.log(W[j]), name="Z")
    mx, den = _logsumexp_inline(Z, shifted=True)
    return te.compute((n,), lambda i: tir.log(den[i]) + mx[i], name="lse")


# ---------------------------------------------------------------------------
# static checks
# ---------------------------------------------------------------------------
def test_static_exp_arguments():
    n, m, d = te.var("n"), te.var("m"), te.var("d")
    S = te.placeholder((n, m), name="S", dtype="float32")
    V = te.placeholder((m, d), name="V", dtype="float32")
    assert static_issues(_softmax_value(S, V, stable=True))["exp"] == []
    assert static_issues(_softmax_value(S, V, stable=False))["exp"] == ["e"]
    # max shift through a scaled score (attention) and inside a merge function
    _, naive = te_programs.attention("naive")
    _, flash = te_programs.attention("flash")
    assert static_issues(naive) == {"exp": [], "domain": []}
    assert static_issues(flash) == {"exp": [], "domain": []}
    # log-space merge: log(exp(a) + exp(b)) is not bounded
    _, logspace = te_programs.softmax_value_logspace()
    assert static_issues(logspace)["exp"] == ["attention"]


def test_static_exp_inside_the_reduction():
    """The shift is judged where the program applies it, not after it has been factored out."""
    X = _xs()
    assert static_issues(_logsumexp_inline(X, shifted=True)) == {"exp": [], "domain": []}
    assert static_issues(_logsumexp_inline(X, shifted=False))["exp"] == ["den"]


def test_static_exp_through_a_stored_tensor():
    """The reducer recomputes the score, the output reads the stored one: the same shift."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype="float32")
    T = te.placeholder((n,), name="T", dtype="float32")
    Z = te.compute((n, m), lambda i, j: X[i, j] / T[i], name="Z")

    def merge(a, b):
        mx = tir.max(a[0], b[0])
        return (mx, a[1] * tir.exp(a[0] - mx) + b[1] * tir.exp(b[0] - mx))

    def ident(t0, t1):
        return (tir.min_value(t0), tir.const(0.0, t1))

    red = te.comm_reducer(merge, ident, name="online")
    j = te.reduce_axis((0, m), "j")
    one = tir.const(1.0, "float32")
    mx, den = te.compute((n,), lambda i: red((X[i, j] / T[i], one), axis=j), name="st")
    P = te.compute((n, m), lambda i, k: tir.exp(Z[i, k] - mx[i]) / den[i], name="P")
    assert static_issues(P)["exp"] == []
    # a tensor the maximum was not taken of is still unbounded
    W = te.placeholder((n, m), name="W", dtype="float32")
    Q = te.compute((n, m), lambda i, k: tir.exp(W[i, k] - mx[i]) / den[i], name="Q")
    assert static_issues(Q)["exp"] == ["Q"]


def test_static_domains():
    X = _xs()
    assert static_issues(_normalise_by_sum(X, positive=True))["domain"] == []
    assert static_issues(_normalise_by_sum(X, positive=False))["domain"] == ["y"]
    # counts and extents are positive; a Welford merge divides by a count that the
    # serial fold makes positive (identity 0 on the left, leaf 1 on the right)
    assert static_issues(_variance_two_pass(X))["domain"] == []
    assert static_issues(_variance_welford(X))["domain"] == []


# ---------------------------------------------------------------------------
# differential test
# ---------------------------------------------------------------------------
def test_reference_evaluates_the_program_as_written():
    X = _xs()
    data = np.random.default_rng(0).standard_normal((3, 7)).astype("float32")
    for program in (_variance_two_pass, _variance_welford):
        (got,) = references(program(X), [X], [data], {"n": 3, "m": 7})
        np.testing.assert_allclose(got, data.astype("float64").var(axis=1), rtol=1e-12)


def test_gate_variance_one_pass_formulas():
    X = _xs()
    gate = AccuracyGate(_variance_two_pass(X), [X])
    assert gate.check(_variance_two_pass(X)).ok
    welford = gate.check(_variance_welford(X))
    assert welford.ok, welford.summary()
    naive = gate.check(_variance_one_pass(X))
    assert not naive.ok and naive.static_ok
    offset = next(f for f in naive.families if f.family == "offset")
    assert not offset.ok and offset.candidate > 1e3 * max(offset.original, offset.inherent)


def test_gate_unshifted_softmax():
    n, m, d = te.var("n"), te.var("m"), te.var("d")
    S = te.placeholder((n, m), name="S", dtype="float32")
    V = te.placeholder((m, d), name="V", dtype="float32")
    gate = AccuracyGate(_softmax_value(S, V, stable=True), [S, V])
    report = gate.check(_softmax_value(S, V, stable=False))
    assert not report.ok and report.static == (("exp", "e"),)
    # judged by the differential test alone it overflows on large scores
    gate.required = []
    report = gate.check(_softmax_value(S, V, stable=False))
    overflow = [f for f in report.families if f.candidate == float("inf")]
    assert not report.ok and overflow and not any(f.ok for f in overflow)
    # an unshifted original sets no static bar
    loose = AccuracyGate(_softmax_value(S, V, stable=False), [S, V])
    assert loose.required == ["domain"]


def test_gate_over_two_outputs():
    X = _xs()
    outs = _logsumexp_inline(X, shifted=True)
    gate = AccuracyGate(outs, [X])
    # a shifted log-sum-exp holds at every magnitude: the two magnitude families count too
    assert [c[0] for c in gate.cases] == ["unit", "offset", "wide", "huge", "tiny"]
    assert gate.required == ["exp", "domain"]
    assert gate.check(outs).ok
    # a wrong second output fails even though the first is exact
    n = X.shape[0]
    off = te.compute((n,), lambda i: outs[1][i] * tir.const(1.01, "float32"), name="off")
    report = gate.check([outs[0], off])
    assert not report.ok and all(f.candidate > 1e-3 for f in report.families)


def test_inputs_are_sampled_on_the_domain_of_the_program():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype="float32")
    W = te.placeholder((m,), name="W", dtype="float32")
    out = _weighted_logsumexp(X, W)
    assert positive_inputs(out, [X, W]) == [False, True]
    gate = AccuracyGate(out, [X, W])
    assert [c[0] for c in gate.cases][:3] == ["unit", "offset", "wide"]  # no family skipped
    assert all(np.all(c[1][1] > 0) for c in gate.cases)
    assert gate.check(out).ok


def test_gate_does_not_pass_vacuously():
    """With no family the original is finite on, nothing was compared: that is not a pass."""
    X = _xs()
    n = X.shape[0]
    bad = te.compute((n,), lambda i: tir.log(tir.const(-1.0, "float32")) + X[i, 0], name="bad")
    gate = AccuracyGate(bad, [X])
    assert gate.cases == []
    report = gate.check(bad)
    assert not report.ok and "no input family" in report.summary()


def test_reference_keeps_reducer_identities_as_written():
    """``0 * min_value`` is 0 in the generated code; ``0 * -inf`` would be NaN."""
    X = _xs()
    n, m = X.shape

    def merge(a, b):
        return (tir.max(a[0], b[0]), a[1] + b[1] + tir.const(0.0, "float32") * a[0])

    def ident(t0, t1):
        return (tir.min_value(t0), tir.const(0.0, t1))

    red = te.comm_reducer(merge, ident, name="r")
    j = te.reduce_axis((0, m), "j")
    mx, total = te.compute((n,), lambda i: red((X[i, j], X[i, j]), axis=j), name="st")
    data = np.random.default_rng(0).standard_normal((2, 5)).astype("float32")
    got_mx, got_total = references([mx, total], [X], [data], {"n": 2, "m": 5})
    np.testing.assert_allclose(got_mx, data.max(axis=1), rtol=1e-12)
    np.testing.assert_allclose(got_total, data.astype("float64").sum(axis=1), rtol=1e-12)


def _softmin_weights(X):
    """``lo = min_j x`` and ``Σ exp(-(x - lo))``: the mirror image of a shifted softmax."""
    n, m = X.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    lo = te.compute((n,), lambda i: te.min(X[i, j1], axis=j1), name="lo")
    w = te.compute((n, m), lambda i, k: tir.exp(lo[i] - X[i, k]), name="w")
    return lo, te.compute((n,), lambda i: te.sum(w[i, j2], axis=j2), name="Z")


def test_static_exp_under_a_minimum_shift():
    X = _xs()
    assert static_issues(_softmin_weights(X)) == {"exp": [], "domain": []}
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(X[i, j], axis=j), name="mx")
    wrong = te.compute((n, m), lambda i, k: tir.exp(mx[i] - X[i, k]), name="wrong")
    assert static_issues(wrong)["exp"] == ["wrong"]  # max - x is not bounded above


def test_one_maximum_bounds_several_terms_of_an_exponent():
    """``exp(a + 2 b - 3 max(a, b))`` is what two rescalings of a merge multiply out to: the
    maximum holds every term down, coefficient for coefficient."""
    lower = _OpLower()
    scope = _Scope(lower, (), {}, 0)
    a, b = ir.state_var("a", 0), ir.state_var("b", 0)
    top = mk_max(a, b)
    assert bounded_above(mk_sub(mk_add(a, mk_mul(ir.const(2), b)), mk_mul(ir.const(3), top)), scope)
    assert not bounded_above(
        mk_sub(mk_add(a, mk_mul(ir.const(3), b)), mk_mul(ir.const(3), top)), scope
    )
    # the mirror image, as a minimum is written: 3 min(a, b) - a - 2 b
    low = mk_neg(mk_max(mk_neg(a), mk_neg(b)))
    assert bounded_above(mk_sub(mk_sub(mk_mul(ir.const(3), low), a), mk_mul(ir.const(2), b)), scope)
    assert not bounded_above(mk_sub(a, low), scope)  # a - min(a, b) grows with a


def test_magnitude_families_count_where_the_original_holds():
    """A norm computed with scaling survives inputs whose squares overflow; a plain sum
    of squares does not, and is not taken for it -- nor held to it when it is the original."""
    X = _xs()
    n, m = X.shape
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    scale = te.compute((n,), lambda i: te.max(tir.abs(X[i, j1]), axis=j1), name="scale")
    ssq = te.compute(
        (n,), lambda i: te.sum((X[i, j2] / scale[i]) * (X[i, j2] / scale[i]), axis=j2), name="ssq"
    )
    scaled = te.compute((n,), lambda i: scale[i] * tir.sqrt(ssq[i]), name="scaled")
    raw = te.compute((n,), lambda i: te.sum(X[i, j3] * X[i, j3], axis=j3), name="raw")
    plain = te.compute((n,), lambda i: tir.sqrt(raw[i]), name="plain")
    gate = AccuracyGate(scaled, [X])
    assert [c[0] for c in gate.cases] == ["unit", "offset", "wide", "huge", "tiny"]
    report = gate.check(plain)
    by_family = {f.family: f for f in report.families}
    assert not report.ok and not by_family["huge"].ok and not by_family["tiny"].ok
    assert by_family["unit"].ok and by_family["offset"].ok and by_family["wide"].ok
    # the plain sum as the original: it overflows there itself, the families do not apply
    loose = AccuracyGate(plain, [X])
    assert [c[0] for c in loose.cases] == ["unit", "offset", "wide"]
    off = AccuracyGate(scaled, [X], AccuracyConfig(magnitudes=False))
    assert [c[0] for c in off.cases] == ["unit", "offset", "wide"] and off.check(plain).ok


def test_declared_positive_inputs():
    """A weight that is only ever summed does not show that it has to be positive."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype="float32")
    W = te.placeholder((n, m), name="W", dtype="float32")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    sw = te.compute((n,), lambda i: te.sum(W[i, j1], axis=j1), name="sw")
    swx = te.compute((n,), lambda i: te.sum(W[i, j2] * X[i, j2], axis=j2), name="swx")
    mean = te.compute((n,), lambda i: swx[i] / sw[i], name="mean")
    assert positive_inputs(mean, [X, W]) == [False, False]
    assert positive_inputs(mean, [X, W], ("W",)) == [False, True]
    gate = AccuracyGate(mean, [X, W], AccuracyConfig(positive=("W",)))
    assert all(np.all(c[1][1] > 0) and np.any(c[1][0] < 0) for c in gate.cases[:1])


if __name__ == "__main__":
    tvm.testing.main()
