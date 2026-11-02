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
"""Counterexample-guided discovery, independent proofs, and staged selection."""

import pytest
from chain_ops import OPERATORS

from tvm.te.superopt import fuse
from tvm.te.superopt.reducer import smt, synthesis
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.derive import domains, psi_inverts, solve_atoms
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.states import Selection, _Chain, _Select, members_of
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import mk_add, mk_mul, mk_reduce


def problem(axis=915001):
    _, r, a, _ = domains(axis)
    u, v = (mk_reduce("sum", r, 0, ir.elem(i, (ir.bidx(0),))) for i in (31, 32))
    return r, a, u, v


def test_solver_constructs_a_coordinate_and_safe_inverse():
    r, a, u, v = problem()
    original = (mk_add(u, v),)

    def verify(extra):
        states = (*original, *extra)
        sol = solve_atoms(states, r)
        if sol is not None and set(sol) == {u, v} and psi_inverts(states, sol, r, a):
            return sol
        return None

    result = synthesis.synthesize(original, (u, v), r, 1, verify)
    assert result.status == "proved" and result.payload is not None
    assert smt.recovery_status((*original, *result.definitions), (u, v), r) == "unsat"


def test_collisions_refine_the_sketch_instead_of_becoming_a_fake_inverse():
    r, _, u, v = problem(915002)
    # uv plus any single linear coordinate has collisions over arbitrary reals.
    # In particular, choosing u loses v at u=0; division by u is not a repair.
    result = synthesis.synthesize(
        (mk_mul(u, v),), (u, v), r, 1, lambda _: pytest.fail("a collision must not be accepted")
    )
    assert result.status == "exhausted"
    assert result.counterexamples > 0 and result.counterexamples == result.candidates
    assert not result.definitions


def test_injectivity_without_a_constructive_proof_is_not_accepted():
    r, _, u, v = problem(915003)
    result = synthesis.synthesize((mk_add(u, v),), (u, v), r, 1, lambda _: None, max_candidates=3)
    assert result.status != "proved" and result.payload is None


def test_unknown_never_counts_as_independent_verification(monkeypatch):
    r, _, u, v = problem(915004)
    monkeypatch.setattr(smt, "recovery_collision", lambda *a: ("unknown", ()))
    result = synthesis.synthesize((mk_add(u, v),), (u, v), r, 1, lambda _: None, max_candidates=2)
    assert result.status != "proved"
    assert result.counterexamples == 0


def test_resource_and_missing_solver_failures_are_explicit(monkeypatch):
    r, _, u, v = problem(915005)

    def verify(_):
        pytest.fail("a budget failure must not run verification")

    assert synthesis.synthesize((u,), (u, v), r, 0, verify).status == "budget"
    assert synthesis.synthesize((u,), (u, v), r, 1, verify, max_candidates=0).status == "limit"
    monkeypatch.setattr(smt, "z3", None)
    assert synthesis.synthesize((u,), (u, v), r, 1, verify).status == "unavailable"


def test_rebase_merge_proof_failure_enters_cegis_without_locking_supplements(monkeypatch):
    _, outs = OPERATORS["moment3"].unfused()
    chain = discover_chains(outs)[0][0]
    selector = _Select(_Chain(chain, members_of(chain), TeSigns(outs)))
    finish = selector.finish
    rejected = []

    def reject_rebase(states):
        if any(c.origin == "closure" for c in states):
            rejected.append(True)
            return Selection(failure="simulated unproved merge")
        return finish(states)

    monkeypatch.setattr(selector, "finish", reject_rebase)
    result = selector.run()
    assert rejected and selector.route == "cegis"
    assert result.solution is not None
    assert [c.name for c in result.states[:2]] == ["total_mean", "s3"]
    assert not any(c.origin == "closure" for c in result.states)
    assert not any(s.action == "hoist" for s in result.steps)


def test_papr_uses_cegis_and_preserves_original_states():
    ins, outs = OPERATORS["papr"].unfused()
    result = fuse(outs, ins)
    assert result.fused and result.accuracy.ok, result.summary()
    selection = result.chains[0].selection
    assert selection.route == "cegis"
    assert [(s.stage, s.status) for s in selection.search[:2]] == [
        ("original", "unproved"),
        ("rebase", "unproved"),
    ]
    assert selection.search[-1].counterexamples > 0
    assert [c.name for c in selection.states[:3]] == ["total", "peak", "ss"]
    assert not any(s.action == "hoist" for s in selection.steps)
    assert any(c.origin == "synthesized" for c in selection.states)
    assert "Z3 CEGIS" in "\n".join(result.chains[0].derived())


@pytest.mark.parametrize(
    "name,route,hoist",
    [
        ("softmax", "original", False),
        ("householder", "rebase", False),
        ("kurtosis", "rebase", True),
        ("normalized_policy_loss", "rebase", True),
    ],
)
def test_routes_distinguish_source_domain_repairs(name, route, hoist):
    ins, outs = OPERATORS[name].unfused()
    result = fuse(outs, ins)
    assert result.fused, result.summary()
    selection = result.chains[0].selection
    assert selection.route == route
    assert any(s.action == "hoist" for s in selection.steps) == hoist
    assert all(a.stage != "cegis" for a in selection.search)
