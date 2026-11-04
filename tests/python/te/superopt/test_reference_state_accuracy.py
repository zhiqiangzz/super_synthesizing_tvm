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
"""Actual B, strict LLVM experiments and complete failure recording."""

import dataclasses
import json

import numpy as np
import pytest
from audit_numerical_accuracy import timed_fusion
from numerical_audit.cases import representative_cases
from numerical_audit.programs import TARGET, reference_programs
from numerical_audit.records import Ledger, json_safe, output_record

from tvm.te.superopt import AccuracyConfig, numeric
from tvm.te.superopt.symbolic.canonicalize import Unsupported


@pytest.fixture(scope="module", params=["variance", "softmax"])
def programs(request, tmp_path_factory):
    programs, definitions = reference_programs(request.param)
    root = tmp_path_factory.mktemp(request.param)
    for label, program in programs.items():
        program.compile(root / label)
    return request.param, programs, definitions, root


def test_actual_b_and_proved_reconstruction(programs):
    name, _, definitions, root = programs
    spec = definitions["reference_B"]
    assert spec["merge_invariant_proof"]
    assert all(spec["reconstruction_proofs"])
    assert not spec["gate_applied"]
    assert TARGET["fast-math"] is False
    assert TARGET["fast-math-reassoc"] is False
    assert TARGET["fast-math-contract"] is False
    tir = (root / "reference_B" / "program.tir").read_text()
    assert "range(1, 1 + (m - 1))" in tir
    if name == "variance":
        assert len(spec["states"]) == 3
        assert spec["states"][0] == "Σ_j 1"
        assert "^2" in spec["states"][2]
        assert "m^-1" in spec["reconstruction"][0]
        assert "fused_total_mean_ss_v2" in tir
    else:
        assert len(spec["states"]) == 2
        assert "max" in spec["states"][0] and "exp" in spec["states"][1]
        assert "exp" in spec["reconstruction"][1]
        assert "exp" in spec["static_issues"]
        assert "den" in tir and "T.exp(T.float32(-1.0) *" in tir
    for label in ("source", "reference_B", "maintained"):
        llvm = (root / label / "program.ll").read_text()
        assert "fadd fast" not in llvm and "fmul contract" not in llvm


def test_identical_inputs_and_complete_degenerate_records(programs, tmp_path):
    name, progs, _, _ = programs
    ledger = Ledger(tmp_path)
    for case, array, meta in representative_cases(name):
        if meta["scan"] not in ("singleton", "zeros", "constant") or array.shape[1] > 16:
            continue
        ledger.measure(case, progs["source"], progs, [array], {"n": 3, "m": array.shape[1]}, meta)
    assert len(ledger.rows) == 9
    for case in ledger.cases:
        rows = [r for r in ledger.rows if r["case"] == case]
        assert len({tuple(r["input_hashes"]) for r in rows}) == 1
        assert all(r["executed"] for r in rows)
        assert all(r["reference_precision"] >= 256 for r in rows)
    if name == "variance":
        assert all(r["status"] == "exact_zero" for r in ledger.rows)
        assert all(
            r["zero_comparison"] == "both_zero"
            for r in ledger.rows
            if r["implementation"] != "source"
        )
    ledger.checkpoint()
    assert len((tmp_path / "measurements.jsonl").read_text().splitlines()) == 9


def test_b_executes_despite_static_exp_rejection(programs, tmp_path):
    name, progs, _, _ = programs
    if name != "softmax":
        return
    ledger = Ledger(tmp_path)
    ledger.measure(
        "overflow",
        progs["source"],
        progs,
        [np.full((3, 16), 128, dtype=np.float32)],
        {"n": 3, "m": 16},
        {},
    )
    by = {r["implementation"]: r for r in ledger.rows}
    assert by["reference_B"]["executed"]
    assert by["reference_B"]["status"] == "nonfinite"
    assert by["source"]["status"] == by["maintained"]["status"] == "finite"


def test_zero_and_nonfinite_serialization():
    with numeric.context(256):
        ref = numeric.mp_array(np.array([0, 1, 2], dtype=np.float32))
        row = output_record(np.array([0, 2, np.inf], dtype=np.float32), ref, "float32", 256)
    assert row["nonfinite"] == 1 and row["finite_elements"] == 2
    assert row["exact_zero_elements"] == 1
    assert row["finite_metrics"]["absolute"] == 1
    encoded = json.dumps(json_safe(row), allow_nan=False)
    assert json.loads(encoded)["absolute"] == "Infinity"


def test_reference_failure_and_missing_implementation_are_recorded(programs, tmp_path, monkeypatch):
    from numerical_audit import records

    def fail(*args, **kwargs):
        raise Unsupported("MPFR reference did not converge by 2048 bits")

    monkeypatch.setattr(records, "high_precision_references", fail)
    _, progs, _, _ = programs
    ledger = Ledger(tmp_path)
    ledger.measure(
        "failed",
        progs["source"],
        {"source": progs["source"], "maintained": None},
        [np.zeros((3, 1), dtype=np.float32)],
        {"n": 3, "m": 1},
        {},
    )
    assert ledger.rows[0]["status"] == "reference_failed"
    assert "2048" in ledger.rows[0]["reference_failure"]
    assert ledger.rows[1]["status"] == "not_executed"
    assert ledger.rows[0]["absolute"] is None


def test_real_gate_report_and_phase_capture(programs):
    name, progs, _, _ = programs
    config = AccuracyConfig(reduce_extent=16, magnitudes=False)
    source = progs["source"]
    result, times, observed = timed_fusion(source.inputs, source.outputs, config, None)
    assert result.fused
    assert result.accuracy.ok
    report = dataclasses.asdict(result.accuracy)
    assert dict(report["extents"])["m"] == 16
    assert observed["checks"][-1] == result.accuracy
    assert sum(v for k, v in times.items() if k != "total") == pytest.approx(times["total"])
    for family in report["families"]:
        assert "sensitivity" in family and "inherent" not in family
        assert family["reference_precision"] >= 256
        assert family["sensitivity_samples"] == 2


def test_fixed_scan_cardinality_and_reused_grid():
    assert sum(1 for name in ("variance", "softmax") for _ in representative_cases(name)) == 347
    cases = list(representative_cases("variance"))
    centered = next(x for _, x, m in cases if m.get("offset") == 0 and m.get("seed") == 0)
    offset = next(x for _, x, m in cases if m.get("offset") == 1048576 and m.get("seed") == 0)
    np.testing.assert_array_equal(centered, offset - np.float32(1048576))
    for row in centered:
        assert set((-2, 0, 2)).issubset(row)
