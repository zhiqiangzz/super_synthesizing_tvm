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
"""Complete statistical outputs, degenerate domains, and offset accuracy."""

import numpy as np
import pytest
from chain_ops import OPERATORS
from chain_ops.run import run_llvm

from tvm.te.superopt import fuse


@pytest.fixture(scope="module", params=["ledoit_wolf", "ols_hc0", "normalized_policy_loss"])
def program(request):
    op = OPERATORS[request.param]
    ins, outs = op.unfused()
    result = fuse(outs, ins)
    assert result.fused, result.summary()
    return op, ins, outs, result


@pytest.mark.parametrize("n,offset", [(1, 0), (2, 0), (17, 0), (31, 10000)])
def test_complete_outputs(program, n, offset):
    op, ins, outs, result = program
    rng = np.random.default_rng(7)
    dims = {"batch": 2, "samples": n, "features": 3}
    if op.name == "ledoit_wolf":
        arrays = [(offset + rng.normal(size=(2, n, 3))).astype("float32")]
    else:
        arrays = [(offset + rng.normal(size=(2, n))).astype("float32") for _ in ins]
    want = op.reference(*arrays)
    got = run_llvm(result.inputs, result.outputs, arrays, dims)
    hi, ho = op.fused()
    hand = run_llvm(hi, ho, arrays, dims)
    for values in (got, hand):
        for a, b in zip(values, want):
            np.testing.assert_allclose(
                a, b, rtol=0.01 if offset else 1e-4, atol=1e-5, equal_nan=True
            )


def test_zero_and_constant_inputs(program):
    op, ins, outs, result = program
    dims = {"batch": 2, "samples": 4, "features": 3}
    arrays = [
        np.zeros((2, 4, 3) if op.name == "ledoit_wolf" else (2, 4), dtype="float32") for _ in ins
    ]
    if op.name == "ols_hc0":
        arrays[1].fill(2)
    got = run_llvm(result.inputs, result.outputs, arrays, dims)
    for a, b in zip(got, op.reference(*arrays)):
        np.testing.assert_allclose(a, b, atol=1e-7, equal_nan=True)


def test_hc0_preserves_defined_residual_state():
    op = OPERATORS["ols_hc0"]
    ins, outs = op.unfused()
    r = fuse(outs, ins)
    selection = r.chains[0].selection
    assert "meat" in r.chains[0].states
    assert not any(s.action == "hoist" for s in selection.steps)
    assert selection.solution.spec.leaves[2].value == 0


def test_ledoit_single_feature():
    op = OPERATORS["ledoit_wolf"]
    ins, outs = op.unfused()
    r = fuse(outs, ins)
    x = np.array([[[1.0], [2.0], [3.0]]], dtype="float32")
    got = run_llvm(r.inputs, r.outputs, [x], {"batch": 1, "samples": 3, "features": 1})
    for a, b in zip(got, op.reference(x)):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_external_baseline(program):
    """Compare complete outputs against independent library implementations."""
    op, ins, _, result = program
    rng = np.random.default_rng(913)
    dims = {"batch": 2, "samples": 13, "features": 3}
    if op.name == "ledoit_wolf":
        sklearn = pytest.importorskip("sklearn.covariance")
        arrays = [rng.normal(size=(2, 13, 3)).astype("float32")]
        answers = [sklearn.ledoit_wolf(x.astype("float64")) for x in arrays[0]]
        want = tuple(np.asarray(a) for a in zip(*answers))
    elif op.name == "ols_hc0":
        regression = pytest.importorskip("statsmodels.regression.linear_model")
        arrays = [rng.normal(size=(2, 13)).astype("float32") for _ in ins]
        want = (
            np.array(
                [
                    regression.OLS(y.astype("float64"), x.astype("float64")[:, None])
                    .fit()
                    .cov_HC0[0, 0]
                    for x, y in zip(*arrays)
                ]
            ),
        )
    else:
        torch = pytest.importorskip("torch")
        arrays = [rng.normal(size=(2, 13)).astype("float32") for _ in ins]
        a, g = (torch.tensor(v, dtype=torch.float64) for v in arrays)
        # SB3 A2C's normalize_advantage=True objective, per rollout batch.
        z = (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + 1e-8)
        want = (-(z * g).mean(-1).numpy(),)
    for a, b in zip(run_llvm(result.inputs, result.outputs, arrays, dims), want):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    for a, b in zip(op.reference(*arrays), want):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_guarded_source_domain_checks_reduction_bodies(monkeypatch):
    from tvm.te.superopt.reducer import smt
    from tvm.te.superopt.symbolic import ir
    from tvm.te.superopt.symbolic.canonicalize import mk_reduce

    x = ir.elem(93, (ir.bidx(0),))
    r = ir.dsym(914000, "R")
    inv = ir.raw_pow(x, -1)
    guarded = ir.if_zero(x, ir.ZERO, inv)
    assert smt.source_defined(mk_reduce("sum", r, 0, guarded))
    # A second, unguarded divisor inside the same body must still be checked.
    unsafe = ir.raw_add((guarded, inv))
    assert not smt.source_defined(mk_reduce("sum", r, 0, unsafe))
    # UNKNOWN cannot certify source definedness or permit a rewrite.
    monkeypatch.setattr(smt, "defined", lambda e: False)
    assert not smt.source_defined(mk_reduce("sum", r, 0, guarded))


def _direct_states(name, data):
    """Independent centered definitions in the selected slot order."""
    x, y = np.asarray(data, dtype="float64").T
    n = len(x)
    if name == "ols_hc0":
        a, b = x @ x, x @ y
        theta = b / a if a else 0
        residual = y - theta * x
        return np.array(
            [a, b, (x * x * residual * residual).sum(), (x**4).sum(), (x**3 * residual).sum()]
        )
    dx, dy = x - x.mean(), y - y.mean()
    if name == "normalized_policy_loss":
        return np.array([x.sum(), dx @ dx, dx @ dy, n, y.sum()])
    return np.array(
        [
            x.sum(),
            y.sum(),
            (dx * dx * dy * dy).sum(),
            dx @ dy,
            n,
            dx @ dx,
            dy @ dy,
            (dx * dy * dy).sum(),
            (dx * dx * dy).sum(),
        ]
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_merge_trees_and_degenerate_partitions(program, dtype):
    from tvm import te
    from tvm.te.superopt.accuracy import _compile, _run
    from tvm.te.superopt.symbolic import ir
    from tvm.te.superopt.symbolic.realize import RealizeEnv, to_prim

    op, _, _, result = program
    spec = result.chains[0].selection.solution.spec
    size = len(spec.states)
    left, right = (te.placeholder((size,), dtype, name=n) for n in ("left", "right"))
    env = RealizeEnv(
        dtype,
        state={
            ir.state_var(side, k): a[k]
            for side, a in (("a", left), ("b", right))
            for k in range(size)
        },
    )

    def output(e, k):
        return te.compute((), lambda: to_prim(e, env), name=f"merged{k}")

    lib = _compile([left, right], [output(e, k) for k, e in enumerate(spec.merge_code)])

    def merge(a, b):
        return np.array(_run(lib, [a, b], [()] * size, [dtype] * size), dtype=dtype)

    rng = np.random.default_rng(19)
    for data in (
        np.zeros((1, 2)),
        np.ones((7, 2)) * 3,
        np.array([[0, 2], [0, -1], [1, 2], [-2, 4]]),
        rng.normal(size=(17, 2)),
        1000 + rng.normal(size=(17, 2)),
    ):
        data = data.astype(dtype)
        want = _direct_states(op.name, data)
        for mode in ("left", "right", "random"):
            # No empty state: singleton or nonempty chunk initialization only.
            chunks = np.array_split(data, min(len(data), 7))
            states = [_direct_states(op.name, a).astype(dtype) for a in chunks]
            while len(states) > 1:
                k = 0 if mode == "left" else len(states) - 2
                if mode == "random":
                    k = int(rng.integers(len(states) - 1))
                a, b = states[k : k + 2]
                if mode == "random" and rng.integers(2):
                    a, b = b, a
                states[k : k + 2] = [merge(a, b)]
            atol = np.full(size, 0.002 if dtype == "float32" else 1e-6)
            if op.name == "ols_hc0" and want[0]:
                # F = E - theta*D is ill-conditioned when x,y have large means.
                # Rounding theta by one ulp changes F by approximately D*ulp.
                # Check that scale explicitly; keep the original H tolerance.
                atol[4] += 4 * np.finfo(dtype).eps * abs(want[3] * want[1] / want[0])
            np.testing.assert_array_less(
                np.abs(states[0] - want),
                atol + (0.01 if dtype == "float32" else 1e-8) * np.abs(want),
            )
