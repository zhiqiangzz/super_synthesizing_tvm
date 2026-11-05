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
"""The GPU benchmark computes the showcase operators the registry defines.

``bench.ops`` restates every showcase operator in torch: the unfused program
and a float64 reference. Both are held here to the float64 numpy reference of
``chain_ops`` on small inputs, so the benchmark cannot drift from the
operators it claims to measure.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the benchmark runs on CUDA", allow_module_level=True)

from bench.ops import SPECS  # noqa: E402
from chain_ops import OPERATORS, showcase  # noqa: E402

SMALL = {
    "softmax": {"n": 5, "m": 300},
    "cross_entropy": {"n": 6, "d": 32, "v": 50},
    "softmax_entropy": {"n": 5, "m": 300},
    "attention": {"b": 2, "h": 3, "s": 40, "d": 16},
    "layernorm": {"n": 5, "m": 300},
    "layernorm_backward": {"n": 5, "m": 300},
    "contrastive": {"n": 37},
    "sinkhorn_iteration": {"n": 30, "m": 20, "d": 3},
    "pearson": {"n": 5, "m": 300},
    "kurtosis": {"n": 5, "m": 300},
    "ess": {"n": 5, "m": 300},
    "papr": {"n": 5, "m": 300},
    "householder": {"n": 5, "m": 300},
    "heat_capacity": {"n": 5, "m": 300},
    "ols_hc0": {"n": 5, "m": 300},
    "normalized_policy_loss": {"n": 5, "m": 300},
    "ledoit_wolf": {"b": 3, "n": 50, "d": 7},
}
# The references take products of matrices in float32 (no TF32): ~1e-8 relative.
FLOAT32_PRODUCTS = {"attention", "cross_entropy"}


def _numpy_inputs(name, inputs):
    """The arguments ``chain_ops`` takes, from the benchmark's tensors."""
    arrays = [t.cpu().double().numpy() for t in inputs]
    if name == "cross_entropy":  # integer labels -> the one-hot targets chain_ops reads
        x, w, labels = arrays
        y = np.zeros((x.shape[0], w.shape[0]))
        y[np.arange(x.shape[0]), labels.astype(int)] = 1.0
        return [x, w, y]
    return arrays


def test_every_showcase_operator_is_benchmarked():
    assert set(SMALL) == set(SPECS) == {op.name for op in showcase()}


@pytest.mark.parametrize("name", sorted(SMALL))
def test_unfused_and_reference_match_the_registry(name):
    spec = SPECS[name]
    gen = torch.Generator(device="cuda").manual_seed(0)
    inputs = spec.inputs(SMALL[name], gen)
    expected = OPERATORS[name].reference(*_numpy_inputs(name, inputs))

    reference = spec.reference(*inputs)
    unfused = spec.unfused(*inputs)
    ref_tol = 1e-6 if name in FLOAT32_PRODUCTS else 1e-9
    assert len(reference) == len(unfused) == len(expected)
    for ref, unf, exp in zip(reference, unfused, expected):
        exp = np.asarray(exp, dtype=np.float64).reshape(ref.shape)
        scale = np.abs(exp).max()
        assert np.abs(ref.cpu().numpy() - exp).max() <= ref_tol * scale, name
        assert np.abs(unf.cpu().double().numpy() - exp).max() <= spec.tol * scale, name
