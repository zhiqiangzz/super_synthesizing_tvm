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
"""Execution feedback, controlled search, and frozen-representation comparisons."""

import dataclasses
import importlib
import json
import subprocess
import sys

import pytest
from chain_ops import OPERATORS
from numerical_audit.selection import coverage
from test_synthesis import problem

from tvm.te.superopt import AccuracyConfig, fuse
from tvm.te.superopt.accuracy import AccuracyGate, AccuracyReport
from tvm.te.superopt.diagnostics import Experiment, solution_record
from tvm.te.superopt.fusion import _fuse, chain_key
from tvm.te.superopt.reducer import synthesis
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.states import members_of, select
from tvm.te.superopt.symbolic.canonicalize import mk_add

FUSION = importlib.import_module("tvm.te.superopt.fusion")


def test_rejection_then_acceptance_reuses_gate_and_isolates_forks(monkeypatch):
    inputs, outputs = OPERATORS["variance"].unfused()
    real = AccuracyGate.check
    seen = []

    def check(gate, candidate):
        seen.append((id(gate), tuple(a.tobytes() for case in gate.cases for a in case[1])))
        verdict = real(gate, candidate)
        return (
            dataclasses.replace(verdict, ok=False, note="injected first refusal")
            if len(seen) == 1
            else verdict
        )

    monkeypatch.setattr(AccuracyGate, "check", check)
    result = fuse(
        outputs,
        inputs,
        retry_rejected_candidates=True,
        accuracy_config=AccuracyConfig(families=(("unit", 0.0, 1.0),), magnitudes=False),
    )
    assert result.fused, result.summary()
    report = result.chains[0]
    assert len(seen) == 2 and seen[0] == seen[1]
    assert report.attempts[0]["status"] == "numerical_rejection"
    assert report.rejected[0].accuracy.note == "injected first refusal"
    assert any(a["status"] == "duplicate" for a in report.attempts)
    assert report.attempts[-1]["states"] == report.states
    assert any(c["status"] == "execution_rejected" for c in report.selection.checks)
    # build_chain only appends on a fork: the committed TE has one tuple reducer.
    from test_fuse import _tuple_ops

    assert len(_tuple_ops(result.outputs)) == 1


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("mode", ["filter", "report", "off"])
def test_accuracy_modes_and_default_stop(monkeypatch, mode, retry):
    inputs, outputs = OPERATORS["softmax"].unfused()
    calls = []

    def check(gate, candidate):
        calls.append(candidate)
        return AccuracyReport(False, (), (), "injected rejection")

    monkeypatch.setattr(AccuracyGate, "check", check)
    result = fuse(outputs, inputs, accuracy=mode, retry_rejected_candidates=retry)
    assert len(calls) == (0 if mode == "off" else 1)
    assert result.fused == (mode != "filter")
    if mode == "filter":
        assert result.chains[0].accuracy.note == "injected rejection"
        assert result.outputs[0].same_as(outputs[0])
        if retry:
            assert "execution candidates rejected" in result.chains[0].reason


def test_callback_covers_original_without_committing_rejected_solution():
    inputs, outputs = OPERATORS["softmax"].unfused()
    chain = discover_chains(outputs)[0][0]
    accepted = select(chain, members_of(chain), TeSigns(outputs))
    seen = []
    refused = select(
        chain,
        members_of(chain),
        TeSigns(outputs),
        accept_candidate=lambda sol: seen.append(sol) or False,
    )
    assert accepted.solution and not refused.solution
    assert len(seen) == 1 and solution_record(seen[0]) == solution_record(accepted.solution)
    assert refused.checks[0]["algebra"] == "passed"


def test_partial_fusion_keeps_first_chain_after_second_rejection(monkeypatch):
    inputs, outputs = OPERATORS["sinkhorn_iteration"].unfused()
    real = AccuracyGate.check
    calls = []

    def check(gate, candidate):
        calls.append(True)
        verdict = real(gate, candidate)
        return verdict if len(calls) == 1 else dataclasses.replace(verdict, ok=False)

    monkeypatch.setattr(AccuracyGate, "check", check)
    result = fuse(outputs, inputs, retry_rejected_candidates=True)
    assert [c.fused for c in result.chains] == [True, False]
    from test_fuse import _tuple_ops

    assert len(_tuple_ops(result.outputs)) == 1
    cov = coverage(result, [c.chain_id for c in result.chains])
    assert cov["partial_fusion"] and not cov["complete_fusion"]
    assert cov["accepted_chains"] == 1 and cov["target_chains"] == 2
    assert not coverage(result, [result.chains[0].chain_id, "undiscovered"])["complete_fusion"]


@pytest.mark.parametrize("name", ["variance", "papr", "householder", "kurtosis"])
def test_controlled_modes_share_repaired_ir_and_checker(name):
    inputs, outputs = OPERATORS[name].unfused()
    baseline = Experiment()
    result = _fuse(outputs, inputs, accuracy="off", experiment=baseline)
    key = result.chains[0].chain_id
    for mode in ("common", "reference_only", "enumeration", "single_term"):
        exp = Experiment(mode=mode, frozen=baseline.frozen, graph=baseline.graph)
        other = _fuse(outputs, inputs, accuracy="off", experiment=exp)
        sel = other.chains[0].selection
        assert sel.diagnostics["base"] == result.chains[0].selection.diagnostics["base"]
        assert sel.diagnostics["reference"] == result.chains[0].selection.diagnostics["reference"]
        assert exp.frozen[key] is baseline.frozen[key]
        assert sum(s.checker_calls for s in sel.search) == len(sel.checks)
        assert all(s.checker_calls == 0 for s in sel.search if s.stage == "rebase")
        if sel.solution:
            assert sel.checks[-1]["phases"]["inverse"] == "passed"


def test_whole_closure_over_budget_is_not_a_checker_call():
    _, outputs = OPERATORS["householder"].unfused()
    chain = discover_chains(outputs)[0][0]
    sel = select(chain, members_of(chain), TeSigns(outputs))
    closure = next(s for s in sel.search if s.stage == "rebase")
    assert closure.candidates == 1 and closure.checker_calls == 0
    assert closure.status == "not_attempted"
    limited = select(chain, members_of(chain), TeSigns(outputs), max_states=1)
    assert not limited.checks and "state budget" in limited.failure


def test_controlled_unknown_does_not_fallback(monkeypatch):
    r, _, u, v = problem()

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

    monkeypatch.setattr(synthesis, "_problem", lambda *args: (Solver(), [], []))
    monkeypatch.setattr(synthesis, "_enumerator", lambda *args: pytest.fail("unexpected fallback"))
    for mode in ("common", "reference_only", "single_term"):
        result = synthesis.synthesize((mk_add(u, v),), (u, v), r, 1, lambda _: None, mode=mode)
        assert result.status == "unknown" and result.candidates == 0


@pytest.mark.parametrize("mode", ["common", "enumeration"])
def test_native_next_and_shared_deadline(monkeypatch, mode):
    r, _, u, v = problem()
    proposals, events = [], []

    def verify(proposal):
        proposals.append(proposal)
        return True if len(proposals) == 2 else None

    result = synthesis.synthesize(
        (mk_add(u, v),),
        (u, v),
        r,
        1,
        verify,
        mode=mode,
        observe=lambda kind, **data: events.append((kind, data)),
    )
    assert result.status == "proved" and len(proposals) == 2
    assert proposals[0].definitions != proposals[1].definitions
    assert any(kind == "solver_query" and data["next"] for kind, data in events)
    clock = [0.0]
    monkeypatch.setattr(synthesis, "monotonic", lambda: clock[0])
    calls = []

    def slow(proposal):
        calls.append(proposal)
        clock[0] += 21

    limited = synthesis.synthesize((mk_add(u, v),), (u, v), r, 1, slow, mode=mode, timeout=20)
    assert limited.status == "limit" and len(calls) == 1


def test_fixed_join_changes_only_merge_print_and_no_retry():
    inputs, outputs = OPERATORS["variance"].unfused()
    baseline = Experiment()
    result = _fuse(outputs, inputs, accuracy="off", experiment=baseline)
    chain = result.chains[0]
    sol = chain.solution
    raw = dataclasses.replace(sol, spec=dataclasses.replace(sol.spec, merge_print=None))
    assert raw.states == sol.states and raw.spec.leaves == sol.spec.leaves
    assert raw.spec.merge == sol.spec.merge and raw.spec.epilogue_print == sol.spec.epilogue_print
    exp = Experiment(
        frozen=baseline.frozen,
        graph=baseline.graph,
        fixed={chain.chain_id: raw},
        prefixes=baseline.prefixes,
    )
    other = _fuse(outputs, inputs, accuracy="off", experiment=exp)
    assert other.chains[0].selection is None
    assert other.chains[0].attempts[0]["solution"] == solution_record(raw)
    with pytest.raises(ValueError, match="cannot retry"):
        _fuse(outputs, inputs, accuracy="off", retry_rejected_candidates=True, experiment=exp)


def test_chain_ids_do_not_depend_on_symbol_allocation():
    _, outputs = OPERATORS["variance"].unfused()
    a = chain_key(discover_chains(outputs)[0][0])
    OPERATORS["householder"].unfused()
    _, outputs = OPERATORS["variance"].unfused()
    assert a == chain_key(discover_chains(outputs)[0][0])


# Captured by executing 440b06a2ac in a separate Python process, with its
# unchanged default gate. Hashes cover emitted source, states, routes and verdicts.
BASELINE_440 = {
    "ledoit_wolf": "bed1ab3c4cb4da78f3d7b9c8d0b3953165296a4bb4a10ba5294ec1615a17bb7c",
    "ols_hc0": "7e9ca385ae4dbd67b357cf334d86cd38e8ecf4642b5c4410bc63f8d1e2c2fc10",
    "normalized_policy_loss": "29929c457ca9ab7af9b10c445746996fb04f0bb16c00a4720bfaf5492db6627d",
    "softmax": "18d1d247bc83f4ecd9e00ae37b6cbc07e96cbe8bf9f1893c4f7ac9409c00ddc9",
    "cross_entropy": "d533a2211cc8d298ca33bc11214ba7ca292d93413fa0494bc5a71061bca9b424",
    "softmax_entropy": "ca22fd635bb7ad2e6e6d86b487606320e67a41f9b7cd428e347914b4d2afd09c",
    "attention": "22dd3c80c3fc70384cae95340d4f1c47dc6ff4dc1902bd263b8bd07e5f704b9a",
    "layernorm": "4e7cb821cd8a9a7e7ec699c6c29a24ab0ee6b3a474eb97ad82c785a0672dd953",
    "layernorm_backward": "cbd6c12a93a1006b5e02eb0eff08b94c4d076f0b26518e43bc08b0513978711c",
    "papr": "5a648d671831d505c3fd5e8a88a763d157dc45208636723c0486c857ff536869",
    "householder": "307a15e12ad081ffffb004bd8f8557bbb1447bed310004dd62317bbeafc483f2",
    "sinkhorn_iteration": "c7698f0ee82f080e310f83a73f8fc08960d50a1bda85fd66b64e7c1b0e5ee98a",
    "kurtosis": "d9451497c745a0666e5022cd927ec64b3c7697f9bda325b54d2349b71ce9b877",
    "pearson": "fd52c7ad7fd724c3cec1825ffc49738223d681679a1a22f32fd31b97c6ea7274",
    "ess": "f978d4e6e1cb2253253aa793c9271ee7eaa77c87a3736a16700fc3caec08214e",
    "contrastive": "724ead026d420bd9bb0d2a5b407828af2e739da6bd22e49725ca97d292d291b9",
    "heat_capacity": "64804b9d6193d0cfe577522a4e3475e53b89a2fda62137de7080798875779b2a",
    "normalised_projection": "7a85d64485c7a6f93e75ff7f4715950b49808e8a08560843fa26103b78ac5f4b",
    "scaled_norm": "ed99f6f548c1aacf0c84a22bbec0d42d69a80cdbe7f19b153d8ecbd1a2c187be",
}


def test_default_off_matches_executed_production_baseline():
    # Isolate symbolic allocation and native solver state just as in the baseline
    # capture. Other tests deliberately exercise alternate grammars/solver queries.
    script = r"""
import hashlib, json, sys
from chain_ops import OPERATORS
from tvm.te.superopt import AccuracyConfig, fuse
hashes = {}
for name in json.loads(sys.argv[1]):
    op = OPERATORS[name]
    inputs, outputs = op.unfused()
    result = fuse(outputs, inputs, max_states=op.max_states,
                  accuracy_config=AccuracyConfig(positive=op.positive))
    row = {"fused": [c.fused for c in result.chains],
           "states": [c.states for c in result.chains],
           "routes": [c.selection.route if c.selection else None for c in result.chains],
           "emission": [c.source() for c in result.chains],
           "accuracy": [c.accuracy.ok if c.accuracy else None
                        for c in result.chains if c.fused]}
    hashes[name] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
print(json.dumps(hashes))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(list(BASELINE_440))],
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == BASELINE_440


def test_fixed_join_rewrites_the_specialized_graph(tmp_path):
    from numerical_audit.records import Ledger
    from numerical_audit.selection import fixed_join

    inputs, outputs = OPERATORS["ledoit_wolf"].unfused()
    assert any("two different indices" in s.reason for s in discover_chains(outputs)[1])
    baseline = Experiment()
    result = _fuse(outputs, inputs, experiment=baseline)
    assert result.fused
    ledger = Ledger(tmp_path)
    fixed_join(
        ledger,
        "ledoit_wolf",
        inputs,
        outputs,
        AccuracyConfig(),
        baseline,
        [c.chain_id for c in result.chains],
    )
    programs = list((tmp_path / "programs" / "ledoit_wolf").glob("*/printed/program.tir"))
    assert programs and all("fused_" in path.read_text() for path in programs)
    assert any(row["implementation"] == "printed" and row["executed"] for row in ledger.rows)
