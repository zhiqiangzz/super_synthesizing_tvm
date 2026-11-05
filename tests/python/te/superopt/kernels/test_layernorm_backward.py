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
"""The layernorm_backward kernel beyond the gate: every layout path, and x far from zero.

The gate's inputs have mean 0. Here the rows of x also sit at an offset of 50
standard deviations, where a float32 Σx² - (Σx)²/m would lose most digits of
the variance; the centred states (Chan's merge, the co-moment re-centred on the
merged mean) must keep the kernel as accurate as the float32 two-pass program.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import layernorm_backward as K

SPEC = SPECS["layernorm_backward"]
FLOOR = 4e-3  # one bf16 rounding of the output, as in kernels.check

CASES = [
    # (n, m, path, vectorised)
    (7, 1000, "held", True),  # one CTA per row, row in registers, idle threads
    (300, 4096, "held", True),  # every thread full
    (3, 4099, "held", False),  # unaligned rows, 17 elements a thread
    (270, 8200, "held", True),  # a cluster of 2 CTAs a row, states merged through DSMEM
    (264, 9001, "held", False),
    (300, 65544, "stream", True),  # too long for a cluster: the row re-read
    (264, 65537, "stream", False),
    (9, 123456, "split", True),  # segments over CTAs, a partial last segment
    (1, 77777, "split", False),
]


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def _check(n, m, offset, variant):
    gen = torch.Generator(device="cuda").manual_seed(0)
    X, G = SPEC.inputs({"n": n, "m": m}, gen)
    X = (X.float() + offset).to(X.dtype)
    (out,) = K.build({"n": n, "m": m}, variant=variant)(X, G)
    (ref,) = SPEC.reference(X, G)
    (unfused,) = SPEC.unfused(X, G)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    assert _err(out, ref) <= max(2.0 * _err(unfused, ref), FLOOR)


@pytest.mark.parametrize("n, m, path, vectorised", CASES)
@pytest.mark.parametrize("offset", [0.0, 50.0])
def test_layernorm_backward_against_float64(n, m, path, vectorised, offset):
    plan = K.plan(n, m)
    assert plan["path"] == path and (plan["vec"] > 1) == vectorised, plan
    _check(n, m, offset, "fused")


@pytest.mark.parametrize("n, m", [(7, 1000), (5, 1001), (9, 123456), (1, 77777)])
def test_unfused_variant(n, m):
    """The ablation: the unfused program's five passes, one kernel each, whole rows or split."""
    _check(n, m, 50.0, "unfused")


def test_split_segments_cover_the_row():
    for n, m in [(4, 16777216), (1, 77777), (9, 123456)]:
        p = K.plan(n, m)
        nvec = m // p["vec"]
        assert p["path"] == "split" and p["s"] > 1
        assert p["seg"] % (p["u"] * K.THREADS) == 0
        assert (p["s"] - 1) * p["seg"] < nvec <= p["s"] * p["seg"]
