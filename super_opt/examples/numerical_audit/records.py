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
"""Lossless experiment ledger: shared quantized inputs and explicit failures."""

import csv
import dataclasses
import hashlib
import json
import math

import numpy as np

from tvm.te.superopt import numeric
from tvm.te.superopt.accuracy import high_precision_references


def json_safe(value):
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def dump(path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def input_hash(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def output_record(actual, reference, dtype, precision):
    """Keep aggregate diagnostics AND finite pairs when an output contains failures."""
    result = dataclasses.asdict(numeric.errors(actual, reference, dtype, precision))
    mask = np.isfinite(actual) & numeric.finite(reference)
    result["finite_elements"] = int(mask.sum())
    result["elements"] = int(actual.size)
    result["reference_nonfinite"] = int(np.count_nonzero(~numeric.finite(reference)))
    result["nan"] = int(np.isnan(actual).sum())
    result["positive_inf"] = int(np.isposinf(actual).sum())
    result["negative_inf"] = int(np.isneginf(actual).sum())
    result["finite_metrics"] = (
        dataclasses.asdict(numeric.errors(actual[mask], reference[mask], dtype, precision))
        if np.any(mask)
        else None
    )
    zeros = numeric.finite(reference) & (reference == 0)
    result["zero_reference_elements"] = int(zeros.sum())
    result["exact_zero_elements"] = int(np.count_nonzero(zeros & (actual == 0)))
    result["exact_equal"] = bool(np.all(mask) and np.all(numeric.mp_array(actual) == reference))
    if result["reference_nonfinite"]:
        result["status"] = "reference_nonfinite"
    elif result["nonfinite"]:
        result["status"] = "nonfinite"
    elif np.all(zeros) and result["exact_equal"]:
        result["status"] = "exact_zero"
    else:
        result["status"] = "finite"
    return result


class Ledger:
    def __init__(self, directory):
        self.directory = directory
        self.rows = []
        self.inputs = {}
        self.cases = {}
        self.reports = {}

    def store_inputs(self, case, arrays):
        keys = []
        for array in arrays:
            key = input_hash(array)
            self.inputs[key] = array.copy()
            keys.append(key)
        self.cases[case] = keys
        return keys

    def measure(self, case, source, programs, arrays, values, metadata, unavailable=None):
        keys = self.store_inputs(case, arrays)
        common = {"case": case, "input_hashes": keys, "extents": values, **metadata}
        want, bits, reference_failure = None, None, ""
        try:
            want, bits = high_precision_references(
                source.outputs, source.inputs, arrays, values, precision=128, max_precision=2048
            )
        except Exception as err:  # Record unsupported domains and nonconvergence, never drop them.
            reference_failure = f"{type(err).__name__}: {err}"
        source_records = []
        for label, program in programs.items():
            failure = (unavailable or {}).get(label, "")
            actual = None
            if program is not None:
                try:
                    actual = program.run(arrays, values)
                except Exception as err:
                    failure = f"{type(err).__name__}: {err}"
            for k, tensor in enumerate(source.outputs):
                row = {
                    **common,
                    "implementation": label,
                    "output_index": k,
                    "output_name": tensor.op.name,
                    "dtype": str(tensor.dtype),
                    "reference_precision": bits,
                    "reference_failure": reference_failure,
                    "executed": actual is not None,
                    "failure": failure,
                    "status": "not_executed" if actual is None else "reference_failed",
                    **dict.fromkeys(("absolute", "relative", "normwise", "ulps", "nonfinite")),
                }
                if actual is not None:
                    row["nonfinite"] = int(np.count_nonzero(~np.isfinite(actual[k])))
                if actual is not None and want is not None:
                    row.update(output_record(actual[k], want[k], str(tensor.dtype), bits))
                if label == "source":
                    source_records.append(row)
                elif source_records:
                    original = source_records[k]["absolute"]
                    error = row["absolute"]
                    if original is not None and error is not None:
                        row["source_error_zero"] = original == 0
                        row["zero_comparison"] = (
                            ("both_zero" if error == 0 else "candidate_nonzero")
                            if original == 0
                            else "source_nonzero"
                        )
                        row["larger_absolute_error_than_source"] = error > original
                self.rows.append(row)
                with (self.directory / "measurements.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
                    )

    def checkpoint(self):
        np.savez_compressed(self.directory / "inputs.npz", **self.inputs)
        dump(self.directory / "input_cases.json", self.cases)
        dump(self.directory / "accuracy_reports.json", self.reports)
        fields = sorted({key for row in self.rows for key in row})
        with (self.directory / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        k: json.dumps(json_safe(v), allow_nan=False)
                        if isinstance(v, dict | list)
                        else json_safe(v)
                        for k, v in row.items()
                    }
                )
