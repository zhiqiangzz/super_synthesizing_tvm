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
"""The heat_capacity kernel beyond the gate: energies far from zero, wide spreads, every path.

The gate's energies are ``2 * randn``. Here they also sit at an offset of
±1000, where a float32 mean of the raw energies is off by ~3e-5 (its ulp is
6e-5) and the merge of absolute means would lose about that much of the
variance; the kernel keeps every mean relative to its state's lowest energy,
which stays small. A spread of 60 makes most weights underflow to 0, which
must merge as weight 0 and not as NaN; a constant row has variance exactly 0.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import heat_capacity

CASES = [
    # (n, m): vectorised or not, one CTA per row or split across CTAs
    (7, 1000),  # vectorised, one CTA per row, idle threads
    (5, 1001),  # scalar, one CTA per row
    (2, 4),  # fewer elements than a warp
    (3, 4099),  # scalar, split, a 3-element last part
    (3, 40000),  # vectorised, split, a partial last part
]


def _err(out, E):
    (ref,) = SPECS["heat_capacity"].reference(E)
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("n, m", CASES)
@pytest.mark.parametrize("offset", [0.0, 1000.0, -1000.0])
def test_heat_capacity_against_float64(n, m, offset):
    gen = torch.Generator(device="cuda").manual_seed(0)
    E = 2.0 * torch.randn(n, m, generator=gen, device="cuda") + offset
    (out,) = heat_capacity.build({"n": n, "m": m})(E)
    assert _err(out, E) <= 1e-6


@pytest.mark.parametrize("n, m", [(5, 1001), (3, 40000)])
def test_underflowing_weights(n, m):
    gen = torch.Generator(device="cuda").manual_seed(1)
    E = 60.0 * torch.randn(n, m, generator=gen, device="cuda")
    (out,) = heat_capacity.build({"n": n, "m": m})(E)
    assert torch.isfinite(out).all()
    assert _err(out, E) <= 1e-6


@pytest.mark.parametrize("n, m", [(2, 4), (3, 4099), (3, 40000)])
def test_constant_rows_have_no_fluctuation(n, m):
    E = torch.full((n, m), -3.25, device="cuda")
    (out,) = heat_capacity.build({"n": n, "m": m})(E)
    assert torch.equal(out, torch.zeros_like(out))


def test_plan_covers_the_split_paths():
    assert heat_capacity.plan(16384, 4096)[0] == 1
    assert heat_capacity.plan(1024, 262144)[0] == 1
    for n, m in [(3, 4099), (1, 77777), (4, 16777216), (1, 67108864)]:
        splits, chunk = heat_capacity.plan(n, m)
        units = m // 4 if m % 4 == 0 else m
        assert splits > 1 and chunk % (heat_capacity.THREADS * heat_capacity.UNROLL) == 0
        assert (splits - 1) * chunk < units <= splits * chunk
