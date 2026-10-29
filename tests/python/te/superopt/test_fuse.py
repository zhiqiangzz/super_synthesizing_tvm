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
import tvm_tirx
from tvm import te
from tvm.te.superopt import fuse
from tvm.te.superopt.accuracy import AccuracyGate, references, spec_issues
from tvm.te.superopt.reducer import Rewriter, build_chain, discover_chains, members_of
from tvm.te.superopt.reducer.derive import derive, domains
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.source import identifier
from tvm.te.superopt.reducer.states import _Chain
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import (
    mk_exp,
    mk_mul,
    mk_reduce,
    subst_domain,
    walk,
)
from tvm.te.superopt.symbolic.lower import compute_ops
from tvm_tirx import tirx as tir


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
    tvm.compile(tvm_tirx.IRModule({"main": res.prim_func()}), target="llvm")


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
        return (tir.const(0.0, t0), tir.min_value(ty=t1))

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


def test_power_intrinsic_fuses_like_the_product():
    """The third central moment with ``tir.power`` instead of ``d * d * d``."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m.astype("float32"), name="mean")
    cube = tir.const(3.0, "float32")
    s3 = te.compute((n,), lambda i: te.sum(tir.power(X[i, j2] - mean[i], cube), axis=j2), name="s3")
    res = fuse(s3, [X])
    (report,) = res.chains
    assert report.fused and len(report.states) == 4, res.summary()
    assert sorted(c.origin for c in report.solution.states) == [
        "closure",
        "extent",
        "member",
        "member",
    ]
    _same_results([X], [s3], res)


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


def test_minimum_is_fused_like_a_maximum():
    """Weights ``exp(-(x - min x))``: the online softmax mirrored, a running minimum."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    lo = te.compute((n,), lambda i: te.min(X[i, j1], axis=j1), name="lo")
    Z = te.compute((n,), lambda i: te.sum(tir.exp(lo[i] - X[i, j2]), axis=j2), name="Z")
    out = te.compute((n,), lambda i: lo[i] - tir.log(Z[i]), name="softmin")
    res = fuse(out, [X])
    (report,) = res.chains
    assert report.fused and report.states == ("lo", "Z") and not report.rejected
    assert passes([out], m) == 2 and passes(res.outputs, m) == 1
    source = res.source()
    assert "t0 = tir.min(a[0], b[0])" in source and "tir.max_value(ty=t0)" in source
    assert "tir.exp((t0 - a[0]))" in source  # the weights only ever shrink
    _same_results([X], [out], res)


def test_maximum_of_standard_scores_keeps_the_largest_element():
    """``max_j (x - mean) / sigma``: Welford's states and ``max_j (x - mean)`` -- only the
    deviation, zero on a single element, is taken out; the score is formed at the end."""
    ins, outs = OPERATORS["zscore_max"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert report.fused and report.states == ("mean", "ss", "zmax.h0", "count")
    (extreme,) = [c for c in report.solution.states if c.origin == "hoist"]
    assert extreme.source == "zmax" and extreme.depth == 1  # the mean stays a context
    assert "mean[i]" in "".join(report.derived())
    assert passes(outs, ins[0].shape[1]) == 3 and passes(res.outputs, ins[0].shape[1]) == 1
    _same_results(ins, outs, res)


def test_minimum_of_standard_scores_keeps_the_smallest_element():
    """The same chain ending in ``te.min``: the state left is ``max_j -x``, a minimum."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    count = m.astype("float32")
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / count, name="mean")
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - mean[i]) * (X[i, j2] - mean[i]), axis=j2), name="ss"
    )
    sigma = te.compute((n,), lambda i: tir.sqrt(ss[i] / count), name="sigma")
    zmin = te.compute((n,), lambda i: te.min((X[i, j3] - mean[i]) / sigma[i], axis=j3), name="zmin")
    res = fuse(zmin, [X])
    (report,) = res.chains
    assert report.fused and len(report.states) == 4, res.summary()
    assert passes([zmin], m) == 3 and passes(res.outputs, m) == 1
    _same_results([X], [zmin], res)


def test_largest_absolute_deviation_is_two_extremes():
    ins, outs = OPERATORS["max_abs_deviation"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert report.fused and [c.origin for c in report.solution.states].count("part") == 2
    assert "is the largest of" in "".join(report.derived())
    _same_results(ins, outs, res)


def _centred(body, reducer=te.max, name="out"):
    """``reducer_j body(x_j - mean)``, the mean being one more pass before it."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m.astype("float32"), name="mean")
    out = te.compute((n,), lambda i: reducer(body(X[i, j2] - mean[i]), axis=j2), name=name)
    return X, out


@pytest.mark.parametrize(
    "body, reducer, states",
    [
        (lambda d: d * d * d, te.max, 3),  # grows with d: the largest x
        (lambda d: tir.power(d, tir.const(5.0, "float32")), te.max, 3),
        (lambda d: tir.exp(d), te.max, 3),
        (lambda d: tir.exp(d * d * d), te.max, 3),
        (lambda d: d * d * d, te.min, 3),  # the smallest x
        (lambda d: -(d * d * d), te.max, 3),
        (lambda d: d * d * d + d, te.max, 3),  # a polynomial that never falls, no one power
        (lambda d: -(d * d * d) - 2 * d, te.max, 3),  # one that never rises: the smallest x
        (lambda d: d * d, te.max, 4),  # grows with |d|: the largest or the smallest x
        (lambda d: d * d * d * d, te.max, 4),
        (lambda d: 2 * d * d + 1, te.max, 4),
        (lambda d: -(d * d), te.min, 4),  # min -d² = -max d²
    ],
)
def test_function_that_grows_with_its_argument_leaves_a_maximum(body, reducer, states):
    """``max_j g(x_j - mean) = g(max_j (x_j - mean))`` for a ``g`` that grows; an even
    power grows with the absolute value and takes both extremes."""
    X, out = _centred(body, reducer)
    res = fuse(out, [X])
    (report,) = res.chains
    assert report.fused and len(report.states) == states, res.summary()
    assert {c.origin for c in report.solution.auxiliaries} == {"extent", "inner"}
    assert "is a function of" in "".join(report.derived())
    assert passes([out], X.shape[1]) == 2 and passes(res.outputs, X.shape[1]) == 1
    _same_results([X], [out], res)


@pytest.mark.parametrize(
    "body, reducer, inside",
    [
        (lambda d: d * d, te.min, "- X[i, j]^2"),  # nearest the mean
        (lambda d: -(d * d), te.max, "- X[i, j]^2"),
        (lambda d: d * d * d - d, te.max, "X[i, j]^3"),  # falls between -1/√3 and 1/√3
    ],
)
def test_maximum_that_is_no_extreme_is_left_alone(body, reducer, inside):
    X, out = _centred(body, reducer)
    res = fuse(out, [X])
    (report,) = res.chains
    assert not res.fused and not report.theory.fusible and res.outputs[0].same_as(out)
    assert report.reason.startswith("no single pass is known: out reads mean inside max_j")
    assert inside in report.reason and not report.rejected  # nothing was looked for


def test_epilogue_keeps_the_function_as_the_program_writes_it():
    """``max_j ((x - mean) / sigma)³``: the cube grows with its argument and is taken out
    of the maximum, the deviation is taken out of what is left, and the state is
    ``max_j (x - mean)``. The epilogue is the cube of the score again, as the program
    computes it -- not the expanded cube over ``max_j x``, four large terms that cancel."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    count = m.astype("float32")
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / count, name="mean")
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - mean[i]) * (X[i, j2] - mean[i]), axis=j2), name="ss"
    )
    sigma = te.compute((n,), lambda i: tir.sqrt(ss[i] / count), name="sigma")

    def cube(i):
        z = (X[i, j3] - mean[i]) / sigma[i]
        return te.max(z * z * z, axis=j3)

    out = te.compute((n,), cube, name="out")
    res = fuse(out, [X])
    (report,) = res.chains
    assert report.fused and report.states[:2] == ("mean", "ss"), res.summary()
    (extreme,) = [c for c in report.solution.states if c.origin in ("inner", "hoist")]
    assert extreme.source.startswith("out") and "mean[i]" in "".join(report.derived())
    (epilogue,) = report.solution.spec.epilogue_code
    cubes = [x for x in walk(epilogue) if isinstance(x, ir.Pow) and x.exponent == 3]
    assert cubes and not isinstance(epilogue, ir.Add)  # one product, nothing to cancel
    assert passes([out], m) == 3 and passes(res.outputs, m) == 1
    assert res.accuracy is not None and res.accuracy.ok
    _same_results([X], [out], res)


# ---------------------------------------------------------------------------
# the checks a reducer has to pass
# ---------------------------------------------------------------------------
def _reducer_of(name: str, states_of):
    """A reducer built from states chosen by hand (``states_of(world)``), and the program."""
    ins, outs = OPERATORS[name].unfused()
    (chain,) = discover_chains(outs)[0]
    world = _Chain(chain, members_of(chain), TeSigns(outs))
    full, R, _, _ = domains(chain.axis)
    required = tuple(subst_domain(world.by_pid[m.pid].state, {R: full}) for m in chain.required)
    states = states_of(world, R)
    spec = derive(
        tuple(c.state if hasattr(c, "state") else c for c in states), required, chain.axis
    )
    assert spec is not None
    trial = Rewriter()
    build_chain(chain, spec, trial)
    return ins, outs, spec, [trial.tensor(t) for t in outs]


def test_expanded_variance_is_kept_out_by_the_accuracy_gate():
    """Two states are enough over the reals: Σx and Σx². The selection never proposes
    them; built by hand they are correct, and in float32 they cancel."""
    ins, outs = OPERATORS["variance"].unfused()
    full = fuse(outs, ins)
    assert full.chains[0].states == ("mean", "ss", "count") and not full.chains[0].rejected
    capped = fuse(outs, ins, max_states=2)
    (report,) = capped.chains
    assert not capped.fused and "are 3, more than the 2 allowed" in report.reason
    x = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))

    def raw(world, R):
        return (mk_reduce("sum", R, 0, x), mk_reduce("sum", R, 0, mk_mul(x, x)))

    ins, outs, spec, candidate = _reducer_of("variance", raw)
    verdict = AccuracyGate(outs, ins).check(candidate)
    assert not verdict.ok and "offset" in verdict.summary()
    values = extents(ins, outs)
    centred = sample(ins, outs, values, seed=3)
    shifted = sample(ins, outs, values, seed=3, shift=1.0e4)
    for arrays, bound in ((centred, 1e-4), (shifted, None)):
        want = references(outs, ins, arrays, values)
        err = rel_err(run_llvm(ins, candidate, arrays, values), want)
        assert err < bound if bound else err > 1.0  # every digit lost on offset data


def test_unshifted_exponentials_are_refused_statically():
    """Taking the maximum out of a softmax leaves ``Σ exp(x)``: correct, and unbounded."""
    x = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    q = ir.elem(1, (ir.idx("i0"), ir.bidx(0)))

    def plain(world, R):
        return (
            mk_reduce("sum", R, 0, mk_exp(x)),
            mk_reduce("sum", R, 0, q),
            mk_reduce("sum", R, 0, mk_mul(q, x)),
        )

    _, _, spec, _ = _reducer_of("soft_cross_entropy", plain)
    assert "exp" in spec_issues(spec)
    ins, outs = OPERATORS["soft_cross_entropy"].unfused()
    res = fuse(outs, ins)
    assert res.fused and "exp" not in spec_issues(res.chains[0].solution.spec)


def test_only_the_context_at_fault_is_taken_out():
    """``-Σ q log p``: ``log den`` is -inf on an empty part and is taken out; the maximum
    is not at fault and stays, so ``Σ q (x - mx)`` keeps the shift. Taking it out as well
    leaves ``Σ q x``, which has to cancel a common offset of the scores against the
    log-sum-exp."""
    op = OPERATORS["soft_cross_entropy"]
    ins, outs = op.unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    steps = [st.summary() for st in report.selection.steps if st.action == "hoist"]
    assert len(steps) == 1 and steps[0].startswith("hoist loss.ctx0 out of loss")
    spec = report.solution.spec
    world = _Chain(report.chain, members_of(report.chain), TeSigns(outs))
    shifted = next(
        k for k, c in enumerate(report.solution.states) if world.rebaser.contexts(c) and c.depth
    )
    assert spec.leaves[shifted].size == 1 and spec.leaves[shifted].value == 0  # q (x - x)
    values = extents(ins, outs)
    arrays = sample(ins, outs, values, seed=4, shift=1.0e4)
    want = op.reference(*arrays)
    assert rel_err(run_llvm(res.inputs, res.outputs, arrays, values), want) < 1e-5

    # the two-context form is correct too, and loses four digits on the same data
    def both(world, R):
        loss = next(c for c in world.members if c.name == "loss")
        out = world.hoisted(loss, frozenset(world.rebaser.contexts(loss)))
        keep = [c for c in world.members if c.name in ("mx", "den")]
        return [*keep, *out]

    own, _, _, cancelled = _reducer_of("soft_cross_entropy", both)
    assert 1e-5 < rel_err(run_llvm(own, cancelled, arrays, values), want) < 1e-2


def test_scaled_norm_is_not_traded_for_a_sum_of_squares():
    """Fused, the norm with scaling is ``sqrt(Σ x²)`` again: correct, and it overflows where
    the original was written not to. The magnitude families of the gate keep it out."""
    ins, outs = OPERATORS["scaled_norm"].unfused()
    res = fuse(outs, ins)
    (report,) = res.chains
    assert not res.fused and res.outputs[0].same_as(outs[0])
    assert any("huge: inf" in r.why and "tiny" in r.why for r in report.rejected)
    assert report.theory.states == 3  # a single pass exists: the two extremes and Σ x²
    loose = fuse(outs, ins, accuracy="off")
    assert loose.fused
    values = extents(ins, outs)
    (x,) = sample(ins, outs, values, seed=2)
    for scale, fused_ok in ((1.0, True), (1.0e25, False)):
        arrays = [(x * scale).astype("float32")]
        want = [np.sqrt((arrays[0].astype(np.float64) ** 2).sum(axis=1))]
        assert rel_err(run_llvm(ins, outs, arrays, values), want) < 1e-5
        err = rel_err(run_llvm(loose.inputs, loose.outputs, arrays, values), want)
        assert (err < 1e-5) == fused_ok


def test_a_budget_too_small_is_reported_not_raised():
    """A correlation takes six states. Capped at four, the chain is left as it is."""
    ins, outs = OPERATORS["pearson"].unfused()
    res = fuse(outs, ins, max_states=4)
    (report,) = res.chains
    assert not res.fused and "are 6, more than the 4 allowed" in report.reason
    assert report.theory.states == 6 and len(report.selection.states) == 6
    assert "theory: a single pass with 6 state(s) exists" in res.summary()
    # left to the judgement, the reducer has the six states it takes
    assert fuse(outs, ins).chains[0].states == ("mean_x", "mean_y", "sxy", "sxx", "syy", "count")


def test_ops_around_a_chain_are_run_not_analysed():
    """What reads the fused members may use functions the chain analysis knows nothing of,
    as long as the reference can evaluate them; otherwise the program is left alone."""
    X, _, _, P = _softmax()
    n, m = X.shape
    squashed = te.compute((n, m), lambda i, j: tir.tanh(P[i, j]), name="squashed")
    res = fuse(squashed, [X])
    assert res.fused and res.chains[0].states == ("mx", "den")
    _same_results([X], [squashed], res)

    odd = te.compute((n, m), lambda i, j: tir.erf(P[i, j]), name="odd")
    res = fuse(odd, [X])
    (report,) = res.chains
    assert not res.fused and "prim.erf" in report.reason and res.outputs[0].same_as(odd)


def test_report_mode_attaches_the_verdict_without_filtering():
    """The remainder of a normalised projection, with the quotient taken out (it is 0 / 0
    where q is 0), cancels on offset data: refused by default, attached in report mode."""
    ins, outs = OPERATORS["normalised_projection"].unfused()
    assert not fuse(outs, ins).fused
    res = fuse(outs, ins, accuracy="report")
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
    assert not report.fused and report.states == ()
    assert report.reason == report.theory.summary() and "no single pass is known" in report.reason

    ins, outs = OPERATORS["cov_matrix"].unfused()
    res = fuse(outs, ins)
    assert res.chains == [] and len(res.skipped) == 1
    assert "not fused" in res.summary()


SOURCED = [
    "logsumexp",
    "attention",
    "variance",
    "moment3",
    "soft_cross_entropy",
    "sinkhorn_square",
    "heat_capacity",
    "zscore_max",
    "max_abs_deviation",
    "contrastive",
]


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
