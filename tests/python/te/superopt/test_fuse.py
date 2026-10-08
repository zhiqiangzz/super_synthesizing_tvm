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
"""``fuse``: the rewritten program, the checks a reducer has to pass, the reports."""

import numpy as np
import pytest
from chain_ops import OPERATORS
from chain_ops.run import extents, passes, rel_err, run_llvm, sample

import tvm
import tvm.testing
from tvm import te
from tvm import tirx as tir
from tvm.te.superopt import fuse
from tvm.te.superopt.accuracy import references
from tvm.te.superopt.reducer import Pool, Rewriter, build_chain, synthesize
from tvm.te.superopt.reducer.source import identifier
from tvm.te.superopt.symbolic.lower import compute_ops


def _same_results(ins, outs, res, seed=0, tol=1e-4, **sizes):
    """The original and the fused program agree with the original's float64 value."""
    values = extents(ins, outs, **sizes)
    arrays = sample(ins, outs, values, seed)
    want = references(outs, ins, arrays, values)
    assert rel_err(run_llvm(ins, outs, arrays, values), want) < tol
    assert rel_err(run_llvm(res.inputs, res.outputs, arrays, values), want) < tol


def _tuple_ops(outputs):
    return [op for op in compute_ops(outputs) if len(op.body) > 1]


def _softmax(dtype="float32", m=None):
    n = te.var("n")
    m = te.var("m") if m is None else m
    X = te.placeholder((n, m), name="X", dtype=dtype)
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(X[i, j1], axis=j1), name="mx")
    den = te.compute((n,), lambda i: te.sum(tir.exp(X[i, j2] - mx[i]), axis=j2), name="den")
    P = te.compute((n, m), lambda i, j: tir.exp(X[i, j] - mx[i]) / den[i], name="P")
    return X, mx, den, P


# ---------------------------------------------------------------------------
# the rewritten program
# ---------------------------------------------------------------------------
def test_softmax_becomes_one_tuple_reduction():
    X, mx, den, P = _softmax()
    res = fuse(P, [X])
    (report,) = res.chains
    assert report.fused and report.states == ("mx", "den") and report.auxiliaries == ()
    assert report.members == ("mx", "den") and report.required == ("mx", "den")
    (reducer,) = _tuple_ops(res.outputs)
    assert len(reducer.body) == 2 and reducer.name == "fused_mx_den"
    assert [iv.var.name for iv in reducer.axis] == ["i"]  # the program's own loop variable
    (out,) = res.outputs
    assert out.op.name == "P" and not out.same_as(P)  # rebuilt around the new states
    assert {t.op.name for t in out.op.input_tensors} == {"X", "fused_mx_den"}
    assert passes([P], X.shape[1]) == 2 and passes(res.outputs, X.shape[1]) == 1
    _same_results([X], [P], res)
    tvm.compile(tvm.IRModule({"main": res.prim_func()}), target="llvm")


def test_states_are_outputs_when_the_program_asks_for_them():
    X, mx, den, _ = _softmax()
    res = fuse([den, mx], [X])
    (reducer,) = _tuple_ops(res.outputs)
    # no epilogue in between: the outputs are the reducer's own state tensors
    assert all(t.op.same_as(reducer) for t in res.outputs)
    assert [int(t.value_index) for t in res.outputs] == [1, 0]
    _same_results([X], [den, mx], res)


def test_member_of_lower_rank_reads_its_state_anywhere_along_the_extra_axis():
    """The normaliser of an attention has no value axis; the fused reducer has."""
    ins, (out,) = OPERATORS["attention"].unfused()
    den = next(op for op in compute_ops(out) if op.name == "den").output(0)
    res = fuse([out, den], ins)
    (report,) = res.chains
    assert report.fused and report.required == ("den", "O")
    assert len(res.outputs[1].shape) == 3 and len(_tuple_ops(res.outputs)[0].axis) == 4
    _same_results(ins, [out, den], res, reduce=16)


def test_downstream_tuple_reduction_is_rebuilt():
    """A reducer over another axis that reads the softmax: both of its outputs move."""
    X, mx, den, P = _softmax()
    n, m = X.shape

    def merge(a, b):
        return (a[0] + b[0], tir.max(a[1], b[1]))

    def ident(t0, t1):
        return (tir.const(0.0, t0), tir.min_value(t1))

    red = te.comm_reducer(merge, ident, name="colstat")
    r = te.reduce_axis((0, n), "r")
    total, peak = te.compute((m,), lambda c: red((P[r, c], P[r, c]), axis=r), name="col")
    res = fuse([total, peak], [X])
    assert res.fused and len(_tuple_ops(res.outputs)) == 2
    assert res.outputs[0].op.same_as(res.outputs[1].op) and not res.outputs[0].op.same_as(total.op)
    _same_results([X], [total, peak], res)


def test_two_chains_one_feeding_the_other():
    ins, outs = OPERATORS["sinkhorn_iteration"].unfused()
    res = fuse(outs, ins)
    first, second = res.chains
    assert first.fused and second.fused and (first.axis, second.axis) == ("j", "i")
    # the second reducer reads the first potential, itself rebuilt on the first reducer
    reducers = _tuple_ops(res.outputs)
    assert [op.name for op in reducers] == ["fused_S_max_S_sumexp", "fused_S2_max_S2_sumexp"]
    assert "f" in {t.op.name for t in reducers[1].input_tensors}
    _same_results(ins, outs, res, reduce=24)


def test_program_without_a_chain_is_returned_as_it_is():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j = te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j], axis=j), name="total")
    out = te.compute((n,), lambda i: total[i] * tir.const(2.0, "float32"), name="out")
    res = fuse(out, [X])
    assert not res.fused and res.chains == [] and res.skipped == []
    assert res.outputs[0].same_as(out) and res.accuracy is None
    assert res.summary() == "no reduction chain in the program"


def test_float64_program():
    X, _, _, P = _softmax("float64")
    res = fuse(P, [X])
    assert res.fused and str(res.outputs[0].dtype) == "float64"
    _same_results([X], [P], res, tol=1e-12)


def test_constant_extent():
    """Nothing here depends on the extent being symbolic when it is not used as a value."""
    X, _, _, P = _softmax(m=12)
    res = fuse(P, [X])
    assert res.fused and passes(res.outputs, 12) == 1
    _same_results([X], [P], res)


def _variance(m, mean_inline: bool):
    n = te.var("n")
    X = te.placeholder((n, m), name="X")
    count = tir.const(float(m), "float32") if isinstance(m, int) else m.astype("float32")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    if mean_inline:
        centre = lambda i: total[i] / count  # noqa: E731
    else:
        mean = te.compute((n,), lambda i: total[i] / count, name="mean")
        centre = lambda i: mean[i]  # noqa: E731
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - centre(i)) * (X[i, j2] - centre(i)), axis=j2), name="ss"
    )
    return X, te.compute((n,), lambda i: ss[i] / count, name="var")


@pytest.mark.parametrize("m", ["symbolic", 16])
@pytest.mark.parametrize("mean_inline", [False, True])
def test_variance_is_welford_however_it_is_written(m, mean_inline):
    """A named mean or one written in place, a symbolic extent or a constant one."""
    X, var = _variance(te.var("m") if m == "symbolic" else m, mean_inline)
    res = fuse(var, [X])
    (report,) = res.chains
    assert report.fused and len(report.states) == 3, res.summary()
    origins = sorted(c.origin for c in report.solution.states)
    assert origins == sorted(["context" if mean_inline else "member", "extent", "member"])
    assert report.members == (("total", "ss") if mean_inline else ("total", "mean", "ss"))
    _same_results([X], [var], res)


def test_scores_stored_in_a_tensor_of_their_own():
    """softmax(x / T): the reducer recomputes the scores, the output reads the stored ones."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    T = te.placeholder((n,), name="T")
    Z = te.compute((n, m), lambda i, j: X[i, j] / T[i], name="Z")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(Z[i, j1], axis=j1), name="mx")
    den = te.compute((n,), lambda i: te.sum(tir.exp(Z[i, j2] - mx[i]), axis=j2), name="den")
    P = te.compute((n, m), lambda i, j: tir.exp(Z[i, j] - mx[i]) / den[i], name="P")
    res = fuse(P, [X, T])
    assert res.fused and res.chains[0].states == ("mx", "den"), res.summary()
    _same_results([X, T], [P], res)


# ---------------------------------------------------------------------------
# the checks a reducer has to pass
# ---------------------------------------------------------------------------
def test_expanded_variance_is_kept_out_by_the_accuracy_gate():
    """Two states are enough over the reals: Σx and Σx². In float32 they cancel."""
    ins, outs = OPERATORS["variance"].unfused()
    full = fuse(outs, ins)
    assert full.chains[0].states == ("mean", "ss", "count") and not full.chains[0].rejected
    small = fuse(outs, ins, max_states=2)
    (report,) = small.chains
    assert not small.fused and report.rejected
    assert all("REJECTED" in r.why and "offset" in r.why for r in report.rejected)
    assert "rejected" in report.reason and small.outputs[0].same_as(outs[0])
    # without the gate the same reducer is taken: it is correct, only not accurate
    loose = fuse(outs, ins, max_states=2, accuracy="off")
    assert loose.fused and len(loose.chains[0].states) == 2 and loose.accuracy is None
    values = extents(ins, outs)
    centred = sample(ins, outs, values, seed=3)
    shifted = sample(ins, outs, values, seed=3, shift=1.0e4)
    for arrays, bound in ((centred, 1e-4), (shifted, None)):
        want = references(outs, ins, arrays, values)
        err = rel_err(run_llvm(loose.inputs, loose.outputs, arrays, values), want)
        assert err < bound if bound else err > 1.0  # every digit lost on offset data


def test_unshifted_exponentials_are_refused_statically():
    """Hoisting the maximum out of a softmax leaves ``Σ exp(x)``: smaller, and unbounded."""
    ins, outs = OPERATORS["soft_cross_entropy"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert report.fused and len(report.states) == 4
    reasons = [r.why for r in report.rejected]
    assert any("exp of an argument that is not bounded above" in w for w in reasons)
    assert any("not finite against the empty state" in w for w in reasons)


def test_fewer_contexts_are_taken_out_first():
    """``-Σ q log p``: taking only ``log den`` out leaves ``Σ q (x - mx)``, which keeps the
    shift by the maximum; taking the maximum out as well leaves ``Σ q x``, which has to
    cancel a common offset of the scores against the log-sum-exp."""
    op = OPERATORS["soft_cross_entropy"]
    ins, outs = op.unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    spec = report.solution.spec
    shifted = next(k for k, c in enumerate(report.solution.states) if c.depth == 1)
    assert [c.depth for c in report.solution.states].count(1) == 1
    assert spec.leaves[shifted].size == 1 and spec.leaves[shifted].value == 0  # q (x - x)
    values = extents(ins, outs)
    arrays = sample(ins, outs, values, seed=4, shift=1.0e4)
    want = op.reference(*arrays)
    assert rel_err(run_llvm(res.inputs, res.outputs, arrays, values), want) < 1e-5
    # the two-context form is correct too, and loses four digits on the same data
    plain = next(
        s
        for s in synthesize(Pool(report.chain))
        if max(c.depth for c in s.states) == 2 and {"mx", "den"} <= {c.name for c in s.states}
    )
    trial = Rewriter()
    build_chain(report.chain, plain.spec, trial)
    cancelled = run_llvm(ins, [trial.tensor(t) for t in outs], arrays, values)
    assert 1e-5 < rel_err(cancelled, want) < 1e-2


def test_report_mode_attaches_the_verdict_without_filtering():
    ins, outs = OPERATORS["variance"].unfused()
    res = fuse(outs, ins, max_states=2, accuracy="report")
    assert res.fused and res.accuracy is not None and not res.accuracy.ok
    with pytest.raises(ValueError, match="accuracy must be"):
        fuse(outs, ins, accuracy="strict")


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def test_reports_name_what_was_derived_and_why_something_was_not():
    ins, outs = OPERATORS["moment3"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert len(report.auxiliaries) == 2 and "count" in report.auxiliaries
    assert "fused into 4 states" in res.summary() and "derived:" in res.summary()
    assert [c.origin for c in report.solution.auxiliaries] == ["extent", "closure"]

    ins, outs = OPERATORS["mean_abs_deviation"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert not report.fused and report.states == () and "no finite lifting" in report.reason

    ins, outs = OPERATORS["cov_matrix"].unfused()
    res = fuse(outs, ins)
    assert res.chains == [] and len(res.skipped) == 1
    assert "not fused" in res.summary()


SOURCED = ["logsumexp", "attention", "variance", "moment3", "soft_cross_entropy", "sinkhorn_square"]


@pytest.mark.parametrize("name", SOURCED)
def test_printed_source_is_the_reducer(name):
    """``source()`` is code: run on the program's own tensors, it computes the members
    the program reads -- reducer, merge, identity, inputs and epilogues included."""
    ins, outs = OPERATORS[name].unfused()
    res = fuse(outs, ins)
    assert res.fused
    scope = {"te": te, "tir": tir}
    for t in [*ins, *outs]:
        scope.update({s.name: s for s in t.shape if isinstance(s, tir.Var)})
    values = extents(ins, outs, reduce=9)
    arrays = sample(ins, outs, values, seed=5)
    source = res.source()
    for report in res.chains:  # earliest first: a later chain reads tensors built on this one
        for t in report.chain.boundary:
            scope.setdefault(t.op.name, t)
    exec(compile(source, f"<fused {name}>", "exec"), scope)
    for report in res.chains:
        wanted = [m.tensor for m in report.chain.required]
        printed = [scope[identifier(m.name)] for m in report.chain.required]
        assert not any(p.same_as(w) for p, w in zip(printed, wanted))
        want = references(wanted, ins, arrays, values)
        got = references(printed, ins, arrays, values)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-9)
    # one definition per helper, also with two chains over axes of the same extent
    assert source.count("def merge_") == source.count("def identity_") == len(res.chains)


def test_fused_part_is_kept_when_another_chain_is_not():
    """A softmax and a mean absolute deviation in one program: one fuses, one stays."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j = [te.reduce_axis((0, m), "j") for _ in range(4)]
    mx = te.compute((n,), lambda i: te.max(X[i, j[0]], axis=j[0]), name="mx")
    den = te.compute((n,), lambda i: te.sum(tir.exp(X[i, j[1]] - mx[i]), axis=j[1]), name="den")
    lse = te.compute((n,), lambda i: tir.log(den[i]) + mx[i], name="lse")
    total = te.compute((n,), lambda i: te.sum(X[i, j[2]], axis=j[2]), name="total")
    mean = te.compute((n,), lambda i: total[i] / m.astype("float32"), name="mean")
    dev = te.compute(
        (n,),
        lambda i: te.sum(tir.max(X[i, j[3]] - mean[i], mean[i] - X[i, j[3]]), axis=j[3]),
        name="dev",
    )
    res = fuse([lse, dev], [X])
    assert sorted((c.members, c.fused) for c in res.chains) == [
        (("mx", "den"), True),
        (("total", "mean", "dev"), False),
    ]
    assert res.outputs[1].same_as(dev) and not res.outputs[0].same_as(lse)
    _same_results([X], [lse, dev], res)
    assert np.isfinite(res.accuracy.families[0].candidate)


if __name__ == "__main__":
    tvm.testing.main()
