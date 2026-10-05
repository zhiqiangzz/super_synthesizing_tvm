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
    AccuracyGate,
    op_issues,
    reference,
    spec_issues,
    static_issues,
)
from tvm.te.superopt.api import make_context
from tvm.te.superopt.config import Bounds
from tvm.te.superopt.reducer import synthesize
from tvm.te.superopt.reducer.partial import synthesize_partial
from tvm.te.superopt.reducer.synth import SynthesisProblem


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
    got = reference(_variance_two_pass(X), [X], [data], {"n": 3, "m": 7})
    np.testing.assert_allclose(got, data.astype("float64").var(axis=1), rtol=1e-12)
    got = reference(_variance_welford(X), [X], [data], {"n": 3, "m": 7})
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


# ---------------------------------------------------------------------------
# the same checks inside the search (filter mode)
# ---------------------------------------------------------------------------
def test_search_time_checks_on_reducers_and_ops():
    ins, out = te_programs.attention("naive")
    ctx, sems = make_context(out, ins, Bounds())
    b = ctx.bounds
    problem = SynthesisProblem(
        ctx.target.body,
        ctx.dims.key(ins[1].shape[2]),
        b.max_states,
        b.max_merge_nodes,
        ctx.target.sem.axis_keys,
        b.min_states,
        ctx.target.consts,
        b.max_leaf_nodes,
        b.max_state_expr_nodes,
        b.max_latent_atoms,
    )
    # the re-based (printed) merge of the partial reducer is judged canonically
    (partial,) = synthesize_partial(problem, ctx)
    assert spec_issues(partial) == set()
    # grammar reducers: the max-shifted ones pass, log-space / unshifted ones do not
    verdicts = [spec_issues(sp) for sp in synthesize(problem)]
    assert set() in verdicts and {"exp"} in [v & {"exp"} for v in verdicts]

    class Entry:  # the part of a pool entry op_issues reads
        def __init__(self, sem):
            self.sem = sem

    assert op_issues("exp", [Entry(sems[0])]) == {"exp"}  # exp(Q) is unbounded
    assert op_issues("div", [Entry(sems[2]), Entry(sems[0])]) == {"domain"}  # Q: either sign
    assert op_issues("log", [Entry(sems[0])]) == {"domain"}


if __name__ == "__main__":
    tvm.testing.main()
