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
"""Solver-backed recovery, zero branches, and retention of original reductions."""

import numpy as np
import pytest
from chain_ops import OPERATORS
from chain_ops.run import run_llvm

from tvm.te.superopt import fuse
from tvm.te.superopt.reducer import smt
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.derive import domains, solve_atoms
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.states import members_of, select
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import mk_mul, mk_pow, mk_reduce, walk
from tvm.te.superopt.symbolic.emit import SourceEnv, hoisted_source


def householder():
    inputs, outputs = OPERATORS["householder"].unfused()
    chain = discover_chains(outputs)[0][0]
    return inputs, outputs, chain


def coordinates():
    _, outputs, chain = householder()
    members = members_of(chain)
    q, v = (c.state for c in members if c.kind != "value")
    _, R, _, _ = domains(chain.axis)
    x = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    e = ir.elem(1, (ir.idx("i0"), ir.bidx(0)))
    u = mk_reduce("sum", R, 0, mk_pow(e, 2))
    c = mk_reduce("sum", R, 0, mk_mul(x, e))
    return q, v, u, c, R


def test_solver_distinguishes_information_loss_from_a_removable_inverse_singularity():
    q, v, u, c, R = coordinates()
    target = (q, u, c)
    assert smt.recovery_status((q, v), target, R) == "sat"
    assert smt.recovery_status((q, v, c), target, R) == "sat"
    assert smt.recovery_status((q, v, u), target, R) == "unsat"
    assert smt.zero_implies(q, c)
    assert not smt.zero_implies(q, u)


def test_zero_implication_is_proved_from_bodies_not_an_operator_name():
    R = ir.dsym(123456, "proof_test")
    x, y = ir.elem(7, (ir.bidx(0),)), ir.elem(8, (ir.bidx(0),))
    # A fourth power and a cubic cross-term, unrelated to the showcase operator.
    q = mk_reduce("sum", R, 0, mk_pow(x, 4))
    c = mk_reduce("sum", R, 0, mk_mul(mk_pow(x, 3), y))
    assert smt.zero_implies(q, c)
    signed = mk_reduce("sum", R, 0, x)
    assert not smt.zero_implies(signed, q)  # cancellation of signed summands


def test_inverse_guard_is_defined_and_two_directions_are_proved():
    q, v, u, c, R = coordinates()
    states = (q, v, u)
    sol = solve_atoms(states, R)
    assert isinstance(sol[c], ir.IfZero)
    env = {ir.state_var("a", k): s for k, s in enumerate(states)}
    for atom in (q, u, c):
        assert smt.equal(ir.raw_subst(sol[atom], env), atom)
    unguarded = sol[c].nonzero
    assert not smt.defined(ir.raw_subst(unguarded, env))
    # A rational identity alone must not hide its hole at zero.
    a = ir.atom("arbitrary_zero")
    quotient = ir.raw_mul((a, ir.raw_pow(a, -1)))
    assert not smt.equal(quotient, ir.ONE)


def test_unknown_is_neither_a_proof_nor_permission_to_hoist(monkeypatch):
    q, _, _, c, _ = coordinates()
    smt.zero_implies.cache_clear()
    monkeypatch.setattr(smt.Encoding, "check", lambda *args: smt.z3.unknown)
    assert not smt.zero_implies(q, c)
    smt.zero_implies.cache_clear()
    smt.body_fact.cache_clear()
    smt.body_zero_implies.cache_clear()


def test_unknown_source_domain_does_not_authorize_a_hoist(monkeypatch):
    _, outputs = OPERATORS["kurtosis"].unfused()
    chain = discover_chains(outputs)[0][0]
    monkeypatch.setattr(smt.Encoding, "check", lambda *args: smt.z3.unknown)
    result = select(chain, members_of(chain), TeSigns(outputs))
    assert result.solution is None
    assert not any(s.action == "hoist" for s in result.steps)


def test_missing_solver_retains_the_program(monkeypatch):
    inputs, outputs, chain = householder()
    monkeypatch.setattr(smt, "z3", None)
    result = select(chain, members_of(chain), TeSigns(outputs))
    assert result.solution is None and "z3-solver" in result.failure
    assert not any(s.action == "hoist" for s in result.steps)


def test_householder_keeps_original_slots_and_outputs_vv_directly():
    _, outputs, chain = householder()
    selection = select(chain, members_of(chain), TeSigns(outputs))
    assert selection.solution is not None, selection.failure
    assert [c.name for c in selection.states] == ["ssq", "vv", "vv.r0"]
    assert not any(s.action in {"hoist", "parts", "split", "drop"} for s in selection.steps)
    spec = selection.solution.spec
    assert spec.proof == "smt"
    assert isinstance(spec.epilogues[0], ir.Elem) and spec.epilogues[0].tensor == -2
    assert any(isinstance(n, ir.IfZero) for e in spec.merge_code for n in walk(e))


def test_a_zero_state_budget_is_enforced():
    _, outputs, chain = householder()
    selection = select(chain, members_of(chain), TeSigns(outputs), max_states=0)
    assert selection.solution is None and "budget" in selection.failure
    assert not any(s.action == "hoist" for s in selection.steps)


def test_branch_local_division_is_not_hoisted_out_of_guard():
    a = ir.state_var("a", 0)
    inv = ir.raw_pow(a, -1)
    guarded = ir.if_zero(a, ir.ZERO, ir.raw_add((inv, inv)))
    env = SourceEnv("float32", state={a: "a[0]"})
    lines, exprs = hoisted_source((guarded, guarded), env)
    assert not lines
    assert all("tir.if_then_else" in e for e in exprs)


@pytest.mark.parametrize(
    "x,e",
    [
        ([0.0], [3.0]),
        ([0.0, 0.0], [3.0, 4.0]),
        ([0.0, 1.0], [3.0, 0.0]),
        ([3.0, 0.0], [2.0, 3.0]),
        ([0.0, 0.0, 2.0, -1.0], [2.0, -3.0, 1.0, 4.0]),
        ([1e-30, 0.0], [1.0, 2.0]),
    ],
)
def test_generated_householder_handles_nonempty_zero_norm_segments(x, e):
    inputs, outputs, _ = householder()
    fused = fuse(outputs, inputs)
    assert fused.fused, fused.summary()
    arrays = [np.array([a], dtype="float32") for a in (x, e)]
    dims = {"n": 1, "m": len(x)}
    got = run_llvm(fused.inputs, fused.outputs, arrays, dims)
    expected = run_llvm(inputs, outputs, arrays, dims)
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-6)
    assert np.all(np.isfinite(got))
