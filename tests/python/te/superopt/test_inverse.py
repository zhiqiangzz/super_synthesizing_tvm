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
"""Piecewise algebraic recovery and preservation of extrema-based states."""

import numpy as np
import pytest
from chain_ops import OPERATORS
from chain_ops.run import run_llvm

from tvm import te
from tvm.te.superopt import fuse
from tvm.te.superopt.accuracy import _compile, _run
from tvm.te.superopt.reducer import inverse, smt
from tvm.te.superopt.reducer.derive import domains, invariant_holds, solve_atoms
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import (
    mk_add,
    mk_card,
    mk_div,
    mk_max,
    mk_mul,
    mk_pow,
    mk_reduce,
    mk_sub,
)
from tvm.te.superopt.symbolic.realize import RealizeEnv, to_prim


def coordinates(axis):
    R = ir.dsym(axis, "R")
    x = ir.elem(71, (ir.bidx(0),))
    # Exercise the body proof on a different expression, with no operator names.
    y = ir.elem(72, (ir.bidx(0),))
    f = mk_mul(x, y)
    n = mk_card(R)
    s = mk_reduce("sum", R, 0, f)
    q = mk_reduce("sum", R, 0, mk_pow(f, 2))
    hi = mk_reduce("max", R, 0, f)
    neglo = mk_reduce("max", R, 0, mk_mul(ir.MINUS_ONE, f))
    mean = mk_div(s, n)
    peak = mk_pow(mk_max(mk_sub(hi, mean), mk_add(mean, neglo)), 2)
    ss = mk_sub(q, mk_div(mk_pow(s, 2), n))
    return R, (s, peak, ss, n, mk_sub(hi, neglo)), (n, s, q, hi, neglo)


def test_pointwise_extreme_bounds_and_complete_inverse():
    R, states, targets = coordinates(913000)
    n, s, _, hi, neglo = targets
    assert smt.condition_holds(mk_sub(mk_mul(n, hi), s), "nonnegative")
    assert smt.condition_holds(mk_add(mk_mul(n, neglo), s), "nonnegative")
    assert not smt.condition_holds(mk_sub(mk_mul(n, hi), s), "positive")
    assert not smt.condition_holds(mk_sub(s, mk_mul(n, hi)), "nonnegative")
    sol = solve_atoms(states, R)
    assert sol is not None and set(sol) == set(targets)
    env = {ir.state_var("a", k): e for k, e in enumerate(states)}
    assert all(smt.equal(ir.raw_subst(sol[a], env), a) for a in targets)
    assert smt.recovery_status(states, targets, R) == "unsat"


def test_range_or_one_extreme_does_not_supply_the_missing_orientation():
    R, states, targets = coordinates(913001)
    hi, neglo = targets[-2:]
    for extra in (hi, mk_add(hi, neglo)):
        trial = (*states[:-1], extra)
        assert smt.recovery_status(trial, targets, R) == "sat"
        assert inverse.solve(trial, R) is None
    assert inverse.solve(states[:-1], R) is None
    # An actual finite-input collision, not merely a model of the abstraction.
    a, b = np.array([-3, 1, 1, 1]), np.array([-1, -1, -1, 3])
    np.testing.assert_equal(_direct(a)[:4], _direct(b)[:4])
    assert _direct(a)[4] != _direct(b)[4]
    assert _direct(np.append(a, 4))[1] != _direct(np.append(b, 4))[1]


def test_incomplete_branch_coverage_is_rejected(monkeypatch):
    R, states, _ = coordinates(913002)
    original = smt.guarded_inverse
    admitted = []

    def first_branch_only(*args):
        if admitted or not original(*args):
            return False
        admitted.append(True)
        return True

    monkeypatch.setattr(smt, "guarded_inverse", first_branch_only)
    assert inverse.solve(states, R) is None
    assert admitted  # a correct local inverse is insufficient without coverage


def test_unknown_never_certifies_a_piecewise_inverse(monkeypatch):
    R, states, _ = coordinates(913003)
    monkeypatch.setattr(smt.Encoding, "check", lambda *args: smt.z3.unknown)
    assert inverse.solve(states, R) is None


@pytest.fixture(scope="module")
def papr():
    inputs, outputs = OPERATORS["papr"].unfused()
    result = fuse(outputs, inputs)
    assert result.fused and result.accuracy.ok, result.summary()
    return inputs, outputs, result


def test_papr_keeps_originals_within_five_states_and_rejects_four(papr):
    inputs, outputs, result = papr
    selection = result.chains[0].selection
    assert len(selection.states) == 5
    assert [c.name for c in selection.states[:3]] == ["total", "peak", "ss"]
    assert not any(s.action in {"hoist", "parts", "split", "drop"} for s in selection.steps)
    spec = selection.solution.spec
    assert spec.merge_print is not None
    assert spec.epilogues == (ir.elem(-2, (ir.idx("i0"),)), ir.elem(-3, (ir.idx("i0"),)))
    limited = fuse(outputs, inputs, max_states=4)
    assert not limited.fused
    assert "not an impossibility proof" in limited.summary()
    # Congruence rewriting with inverse lemmas must still check the actual merge.
    _, R, A, B = domains(spec.axis)
    sol = solve_atoms(spec.states, R)
    bad = (mk_add(spec.merge_code[0], ir.ONE), *spec.merge_code[1:])
    assert invariant_holds(spec.states, bad, R, A, B, sol) is None


def _direct(x):
    x = np.asarray(x, dtype="float64")
    d = x - x.mean()
    return np.array([x.sum(), (d * d).max(), (d * d).sum(), len(x), x.min() + x.max()])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_compiled_merge_on_degenerate_segments_and_different_trees(papr, dtype):
    spec = papr[2].chains[0].selection.solution.spec
    left, right = (te.placeholder((5,), dtype, name=n) for n in ("left", "right"))
    env = RealizeEnv(
        dtype,
        state={
            ir.state_var(side, k): a[k] for side, a in (("a", left), ("b", right)) for k in range(5)
        },
    )

    def output(e, k):
        return te.compute((), lambda: to_prim(e, env), name=f"merged{k}")

    outputs = [output(e, k) for k, e in enumerate(spec.merge_code)]
    lib = _compile([left, right], outputs)

    def merge(a, b):
        return np.array(_run(lib, [a, b], [()] * 5, [dtype] * 5), dtype=dtype)

    rng = np.random.default_rng(19)
    data = [
        np.array(x, dtype=dtype)
        for x in (
            [0],
            [3, 3, 3, 3],
            [-2, 0, 2],
            [-3, 1, 1, 1, 4],
            [-1, -1, -1, 3, 4],
            [-5, -2, -2, 4, 5],
            [-4, -4, -1, 4, 5],
            10000 + rng.normal(size=31),
            rng.normal(size=31) * 100,
        )
    ]
    for x in data:
        want = _direct(x)
        for mode in ("left", "right", "random"):
            leaves = [_direct(np.array([v])).astype(dtype) for v in x]
            while len(leaves) > 1:
                i = 0 if mode == "left" else len(leaves) - 2
                if mode == "random":
                    i = int(rng.integers(len(leaves) - 1))
                a, b = leaves[i : i + 2]
                if mode == "random" and rng.integers(2):
                    a, b = b, a
                leaves[i : i + 2] = [merge(a, b)]
            np.testing.assert_allclose(
                leaves[0], want, rtol=0.005 if dtype == "float32" else 1e-9, atol=1e-6
            )


def test_generated_program_preserves_source_zero_variance_behavior(papr):
    inputs, outputs, result = papr
    for x in ([0], [3, 3, 3], [-2, 0, 2], [-3, 1, 1, 1, 4]):
        arrays = [np.array([x], dtype="float32")]
        dims = {"n": 1, "m": len(x)}
        got = run_llvm(result.inputs, result.outputs, arrays, dims)
        want = run_llvm(inputs, outputs, arrays, dims)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, equal_nan=True)
