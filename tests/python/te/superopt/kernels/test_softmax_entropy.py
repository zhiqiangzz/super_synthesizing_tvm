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
"""The softmax_entropy kernel beyond the gate: every layout path, and rows the gate's lack.

The gate's rows are N(0, 4) logits. Here also a large common offset (the
state is relative to the running maximum, so it must not matter), peaked rows
(``l`` close to 1, where the epilogue's ``log1p(l - 1)`` matters), and
constant rows, whose entropy is exactly ``log m``.
"""

import math

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import BF16, SPECS
from kernels import softmax_entropy

CASES = [
    # (n, m): vectorised or not, one CTA per row or split across CTAs
    (7, 1000),  # vectorised, one CTA per row
    (5, 1001),  # scalar, one CTA per row
    (1, 77777),  # scalar, split
    (3, (1 << 20) + 40),  # vectorised, split, a shorter last part
]


def _rows(n, m, kind):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(n, m, generator=gen, device="cuda")
    if kind == "gate":
        x = 2.0 * x
    elif kind == "offset":
        x = 2.0 * x + 1000.0
    elif kind == "peaked":
        x = 12.0 * x
    return x.to(BF16)


@pytest.mark.parametrize("n, m", CASES)
@pytest.mark.parametrize("kind", ["gate", "offset", "peaked"])
def test_entropy_against_float64(n, m, kind):
    X = _rows(n, m, kind)
    (out,) = softmax_entropy.build({"n": n, "m": m})(X)
    (ref,) = SPECS["softmax_entropy"].reference(X)
    assert out.shape == (n,) and out.dtype == torch.float32
    err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    assert err <= 1e-6, err


@pytest.mark.parametrize("n, m", CASES)
def test_constant_rows_have_entropy_log_m(n, m):
    X = torch.full((n, m), 3.0, device="cuda", dtype=BF16)
    (out,) = softmax_entropy.build({"n": n, "m": m})(X)
    assert torch.allclose(
        out.double(), torch.full_like(out.double(), math.log(m)), rtol=1e-6, atol=0
    )


def test_splits_cover_both_paths():
    assert softmax_entropy._splits(16384, 32000) == 1
    assert softmax_entropy._splits(4096, 151936) == 1
    assert softmax_entropy._splits(7, 1000) == 1
    assert softmax_entropy._splits(1, 77777) > 1
    assert softmax_entropy._splits(3, (1 << 20) + 40) > 1
