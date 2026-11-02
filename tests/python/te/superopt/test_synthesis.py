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
"""Native SyGuS proposals, rebase features, and independent acceptance proofs."""

from fractions import Fraction

import pytest
from chain_ops import OPERATORS

from tvm.te.superopt import fuse
from tvm.te.superopt.reducer import smt, synthesis
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.derive import domains, psi_inverts, solve_atoms
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.states import Selection, _Chain, _Select, members_of
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import mk_add, mk_max, mk_mul, mk_pow, mk_reduce, mk_sub


def problem(axis=915001):
    _, r, a, _ = domains(axis)
    u, v = (mk_reduce("sum", r, 0, ir.elem(i, (ir.bidx(0),))) for i in (31, 32))
    return r, a, u, v


def checker(original, targets, r, a):
    def verify(proposal):
        states = (*original, *proposal.definitions)
        sol = solve_atoms(states, r)
        if (
            sol is not None
            and set(sol) == set(targets)
            and psi_inverts(states, sol, r, a)
            and smt.recovery_status(states, targets, r) == "unsat"
        ):
            return sol
        return None

    return verify


def test_native_solver_constructs_a_coordinate_and_safe_inverse():
    r, a, u, v = problem()
    original = (mk_add(u, v),)
    result = synthesis.synthesize(original, (u, v), r, 1, checker(original, (u, v), r, a))
    assert result.status == "proved" and result.payload is not None
    assert smt.recovery_status((*original, *result.definitions), (u, v), r) == "unsat"


def test_colliding_grammar_is_not_a_fake_inverse():
    r, a, u, v = problem(915002)
    # uv and any one linear coordinate collide over arbitrary reals.
    result = synthesis.synthesize(
        (mk_mul(u, v),),
        (u, v),
        r,
        1,
        checker((mk_mul(u, v),), (u, v), r, a),
        timeout=2,
    )
    assert result.status in {"unknown", "exhausted", "limit"}
    assert not result.definitions


def test_injectivity_without_a_constructive_proof_requests_native_next_solution():
    r, _, u, v = problem(915003)
    seen = []

    def refuse(proposal):
        seen.append(proposal.definitions)

    result = synthesis.synthesize((mk_add(u, v),), (u, v), r, 1, refuse, max_candidates=3)
    assert result.status != "proved" and result.payload is None
    assert len(seen) == 3 and len(set(seen)) == 3


def test_unknown_never_counts_as_independent_verification(monkeypatch):
    r, _, u, v = problem(915004)

    class Unknown:
        def hasSolution(self):
            return False

        def isUnknown(self):
            return True

    class Solver:
        def setOption(self, *args):
            pass

        def checkSynth(self):
            return Unknown()

    monkeypatch.setattr(synthesis, "_problem", lambda *a: (Solver(), [], []))
    result = synthesis.synthesize(
        (mk_add(u, v),),
        (u, v),
        r,
        1,
        lambda _: None,
        max_candidates=2,
    )
    assert result.status != "proved" and result.payload is None


@pytest.mark.parametrize("kwargs,status", [({"max_extra": 0}, "budget"), ({"timeout": 0}, "limit")])
def test_resource_limits_are_not_impossibility(kwargs, status):
    r, _, u, v = problem(915005)
    args = {"max_extra": 1, **kwargs}
    result = synthesis.synthesize((mk_add(u, v),), (u, v), r, verify=lambda _: None, **args)
    assert result.status == status


@pytest.mark.parametrize("module,attribute", [(smt, "z3"), (synthesis, "cvc5")])
def test_missing_solver_is_explicit(monkeypatch, module, attribute):
    r, _, u, v = problem(915006)
    monkeypatch.setattr(module, attribute, None)
    assert synthesis.synthesize((u,), (u, v), r, 1, lambda _: None).status == "unavailable"


def test_nonlinear_rebase_feature_is_preferred_to_raw_coordinates():
    r, a, u, v = problem(915007)
    feature = mk_sub(v, mk_pow(u, 2))
    original = (u,)
    result = synthesis.synthesize(
        original,
        (u, v),
        r,
        1,
        checker(original, (u, v), r, a),
        preferred=(feature,),
    )
    assert result.status == "proved" and result.phase == "rebase"
    assert result.definitions == (feature,)


@pytest.mark.parametrize("kind", ["new_atom", "undefined", "free_value"])
def test_features_must_be_defined_functions_of_b(kind):
    r, _, u, v = problem(915008)
    extra = mk_reduce("sum", r, 0, ir.elem(99, (ir.bidx(0),)))
    feature = extra if kind == "new_atom" else mk_pow(v, -1)
    if kind == "free_value":
        feature = ir.atom("not_in_B")
    result = synthesis.synthesize((u,), (u, v), r, 1, lambda _: None, preferred=(feature,))
    assert result.status == "unsupported"


def test_feature_recipe_reconstructs_nontrivial_combinations():
    r, _, u, v = problem(915009)
    feature = mk_sub(v, mk_pow(u, 2))
    proposal = synthesis.Proposal((feature, u), ((Fraction(2), Fraction(-1)),))
    assert proposal.definitions == (mk_sub(mk_mul(ir.const(2), feature), u),)


def test_native_solver_combines_nonlinear_features_and_preserves_the_recipe():
    r, a, u, v = problem(915011)
    w = mk_reduce("sum", r, 0, ir.elem(33, (ir.bidx(0),)))
    original = (u, mk_max(v, w))
    features = (mk_sub(v, mk_pow(u, 2)), mk_sub(w, mk_pow(u, 2)))
    check = checker(original, (u, v, w), r, a)
    accepted = []

    def verify(proposal):
        proof = check(proposal)
        if proof is not None:
            accepted.append(proposal)
        return proof

    result = synthesis.synthesize(original, (u, v, w), r, 1, verify, preferred=features)
    assert result.status == "proved" and result.phase == "rebase"
    assert accepted[0].features == features
    (row,) = accepted[0].weights
    assert row[0] and row[1] and row[0] == -row[1]


def test_rebase_merge_failure_falls_back_to_native_sygus(monkeypatch):
    _, outs = OPERATORS["moment3"].unfused()
    chain = discover_chains(outs)[0][0]
    selector = _Select(_Chain(chain, members_of(chain), TeSigns(outs)))
    finish = selector.finish
    rejected = []

    def reject_once(states):
        if not rejected and any(c.origin == "closure" for c in states):
            rejected.append(True)
            return Selection(failure="simulated unproved merge")
        return finish(states)

    monkeypatch.setattr(selector, "finish", reject_once)
    result = selector.run()
    assert rejected and selector.route == "sygus"
    assert result.solution is not None
    assert [c.name for c in result.states[:2]] == ["total_mean", "s3"]
    assert not any(s.action == "hoist" for s in result.steps)


def test_papr_uses_sygus_and_preserves_original_states():
    ins, outs = OPERATORS["papr"].unfused()
    result = fuse(outs, ins)
    assert result.fused and result.accuracy.ok, result.summary()
    selection = result.chains[0].selection
    assert selection.route == "sygus"
    assert selection.search[-1].stage == "sygus"
    assert [c.name for c in selection.states[:3]] == ["total", "peak", "ss"]
    assert not any(s.action == "hoist" for s in selection.steps)
    assert "cvc5 SyGuS" in "\n".join(result.chains[0].derived())


@pytest.mark.parametrize(
    "name,route,hoist",
    [
        ("softmax", "original", False),
        ("householder", "sygus", False),
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


def test_sygus_preserves_centered_kurtosis_feature_and_accuracy(monkeypatch):
    finish = _Select.finish
    rejected = []

    def reject_once(self, states):
        if not rejected and any(c.origin == "closure" for c in states):
            rejected.append(True)
            return Selection(failure="exercise native SyGuS after the closure proposal")
        return finish(self, states)

    monkeypatch.setattr(_Select, "finish", reject_once)
    ins, outs = OPERATORS["kurtosis"].unfused()
    result = fuse(outs, ins)
    assert rejected and result.fused and result.accuracy.ok, result.summary()
    selection = result.chains[0].selection
    assert selection.route == "sygus"
    centered = [
        c for c in selection.states if c.origin == "closure" and not isinstance(c.state, ir.Card)
    ]
    assert len(centered) == 1 and centered[0].kind == "sum"
    assert centered[0].struct is not centered[0].state
    assert "rebase" in selection.search[-1].detail


def test_count_is_an_ordinary_b_coordinate_without_rebase_features(monkeypatch):
    monkeypatch.setattr(_Chain, "coefficients", lambda self, owner: [])
    monkeypatch.setattr(_Chain, "pieces", lambda self, owner: [])
    native = synthesis.synthesize
    searches = []

    def search(*args, **kwargs):
        searches.append(kwargs["preferred"])
        return native(*args, **kwargs)

    monkeypatch.setattr(synthesis, "synthesize", search)
    ins, outs = OPERATORS["variance"].unfused()
    result = fuse(outs, ins)
    assert result.fused and result.accuracy.ok, result.summary()
    selection = result.chains[0].selection
    assert searches == [()] and selection.route == "sygus"
    (extra,) = selection.states[2:]
    assert isinstance(extra.state, ir.Card) and extra.origin == "synthesized"


def test_fixed_problem_extent_is_shared_not_a_new_state():
    r, a, u, v = problem(915010)
    extent = ir.shape_sym("fixed_problem_extent")
    original = (mk_mul(u, mk_pow(extent, -1)),)
    feature = mk_mul(v, mk_pow(extent, -1))
    result = synthesis.synthesize(
        original,
        (u, v),
        r,
        1,
        checker(original, (u, v), r, a),
        preferred=(feature,),
    )
    assert result.status == "proved" and result.definitions == (feature,)
    unknown = mk_mul(v, mk_pow(ir.shape_sym("unavailable_extent"), -1))
    rejected = synthesis.synthesize(original, (u, v), r, 1, lambda _: None, preferred=(unknown,))
    assert rejected.status == "unsupported"
