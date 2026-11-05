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
"""The pearson kernel beyond the gate: data far from zero, and every layout path.

The gate's inputs have mean 0. Here the rows also sit at an offset of 1000
standard deviations, where Σx² - (Σx)²/m loses every digit in float32 and a
float32 mean of the raw data is off by 1e-4 of the spread; the centred (Chan)
merge of the shifted data must still be as accurate as float32 allows.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import pearson

CASES = [
    # (n, m): vectorised or not, one CTA per row or split across CTAs
    (7, 1000),  # vectorised, one CTA per row
    (5, 1001),  # scalar, one CTA per row
    (1, 77777),  # scalar, split
    (3, 1 << 20),  # vectorised, split, a partial last segment
]


@pytest.mark.parametrize("n, m", CASES)
@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_pearson_against_float64(n, m, offset):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(n, m, generator=gen, device="cuda")
    y = 0.6 * x + 0.8 * torch.randn(n, m, generator=gen, device="cuda")
    X, Y = x + offset, y - offset
    (out,) = pearson.build({"n": n, "m": m})(X, Y)
    (ref,) = SPECS["pearson"].reference(X, Y)
    err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    assert err <= 1e-6, err


def test_plan_covers_the_split_paths():
    assert pearson.plan(16384, 4096)[1] == 1
    vec, splits, _ = pearson.plan(1, 77777)
    assert vec == 1 and splits > 1
    vec, splits, seg = pearson.plan(1, 67108864)
    assert vec == 4 and splits > 1 and (splits - 1) * seg < 67108864 // vec <= splits * seg
