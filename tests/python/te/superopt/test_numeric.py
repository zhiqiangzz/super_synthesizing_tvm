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
"""High-precision oracles, error diagnostics, and fail-closed regression budgets."""

import gmpy2
import numpy as np
import pytest

from tvm import te
from tvm.te.superopt import numeric
from tvm.te.superopt.accuracy import AccuracyConfig, AccuracyGate, high_precision_references
from tvm.te.superopt.symbolic.canonicalize import Unsupported


def _cancellation():
    x = te.placeholder((3,), "float64", name="x")
    out = te.compute((), lambda: (x[0] + x[1]) + x[2], name="sum")
    return x, out


def test_mpfr_reference_resolves_float64_cancellation():
    x, out = _cancellation()
    arrays = [np.array([2.0**53, 1, -(2.0**53)])]
    (value,), bits = high_precision_references(out, [x], arrays, {})
    assert value.item() == 1 and bits >= 256
    assert (arrays[0][0] + arrays[0][1]) + arrays[0][2] == 0


def test_reference_increases_precision_and_refuses_unstable_results():
    x, out = _cancellation()
    arrays = [np.array([2.0**200, 1, -(2.0**200)])]
    (value,), bits = high_precision_references(out, [x], arrays, {})
    assert value.item() == 1 and bits == 512
    with pytest.raises(Unsupported, match="did not stabilize"):
        high_precision_references(out, [x], arrays, {}, max_precision=256)


def test_reference_preserves_quantized_input_and_context():
    x = te.placeholder((1,), "float32", name="x")
    out = te.compute((), lambda: x[0], name="identity")
    value = np.array([0.1], dtype="float32")
    with gmpy2.context(precision=37, round=gmpy2.RoundDown):
        (ref,), _ = high_precision_references(out, [x], [value], {})
        assert gmpy2.get_context().precision == 37
        assert gmpy2.get_context().round == gmpy2.RoundDown
    assert ref.item().as_integer_ratio() == float(value[0]).as_integer_ratio()


def test_float64_error_is_measured_before_reference_rounding():
    with numeric.context(256):
        ref = np.array([gmpy2.mpfr(1) + gmpy2.mpfr(2) ** -54], dtype=object)
    error = numeric.errors(np.array([1.0]), ref, "float64")
    assert error.absolute == 2.0**-54
    assert error.relative > 0 and error.ulps == 0


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_ulp_distance_is_actual_representable_distance(dtype):
    ref = numeric.mp_array(np.array([1.0], dtype=dtype))
    actual = np.nextafter(np.array([1.0], dtype=dtype), np.array([np.inf], dtype=dtype))
    assert numeric.errors(actual, ref, dtype).ulps == 1


def test_small_component_cannot_hide_behind_large_output():
    ref = numeric.mp_array([1e12, 1.0])
    original = np.array([1e12, 1.0])
    actual = np.array([1e12, 2.0])
    metrics = numeric.errors(actual, ref, "float64")
    assert metrics.normwise == 1e-12 and metrics.relative == 1
    assert not numeric.regression_check(
        actual,
        original,
        ref,
        np.zeros(2, dtype=object),
        "float64",
        factor=100,
        relative_eps=64,
        absolute_tolerance=0,
        precision=256,
    )


@pytest.mark.parametrize("actual", [[np.nan], [np.inf], [1e-10]])
def test_zero_reference_and_new_nonfinite_do_not_pass(actual):
    assert not numeric.regression_check(
        np.array(actual),
        np.zeros(1),
        numeric.mp_array([0.0]),
        np.zeros(1, dtype=object),
        "float64",
        factor=100,
        relative_eps=64,
        absolute_tolerance=0,
        precision=256,
    )


def test_missing_mpfr_is_explicit(monkeypatch):
    x, out = _cancellation()
    monkeypatch.setattr(numeric, "gmpy2", None)
    with pytest.raises(Unsupported, match="require gmpy2"):
        high_precision_references(out, [x], [np.ones(3)], {})


def test_gate_report_contains_metrics_and_reference_precision():
    x = te.placeholder((4,), "float32", name="x")
    out = te.compute((4,), lambda i: x[i] * x[i], name="square")
    gate = AccuracyGate(out, [x], AccuracyConfig(magnitudes=False))
    report = gate.check(out)
    assert report.ok
    for case in report.families:
        assert case.reference_precision >= 256
        assert case.sensitivity_samples == 2
        assert len(case.candidate_metrics) == 1
    assert "sampled sensitivity" in report.summary() and "inherent" not in report.summary()


def test_zero_output_of_sinkhorn_has_a_stable_reference():
    from chain_ops import OPERATORS

    inputs, outputs = OPERATORS["sinkhorn_iteration"].unfused()
    arrays = [
        np.zeros((2, 2), "float32"),
        np.zeros((2, 2), "float32"),
        np.full(2, 0.125, "float32"),
        np.full(2, 0.125, "float32"),
        np.zeros(2, "float32"),
    ]
    values, bits = high_precision_references(outputs, inputs, arrays, {"n": 2, "m": 2, "dim": 2})
    assert bits >= 256
    assert np.all(numeric.finite(values[1]))
    assert all(abs(x) < gmpy2.mpfr(2) ** -149 for x in values[1].flat)


def test_reference_work_budget_limits_multidimensional_samples():
    from chain_ops import OPERATORS

    from tvm.te.superopt.accuracy import _var_values

    inputs, outputs = OPERATORS["ledoit_wolf"].unfused()
    cfg = AccuracyConfig()
    values = _var_values(outputs, inputs, cfg)
    assert (
        values["batch"] * values["samples"] * values["features"] ** 2 <= cfg.max_reference_elements
    )
    inputs, outputs = OPERATORS["variance"].unfused()
    assert _var_values(outputs, inputs, cfg)["m"] == cfg.reduce_extent
    x = te.placeholder((20000,), "float32", name="x")
    out = te.compute((20000,), lambda i: x[i], name="identity")
    with pytest.raises(Unsupported, match="fixed reference shape"):
        _var_values(out, [x], cfg)
