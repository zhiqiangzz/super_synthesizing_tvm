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
"""The layernorm kernel beyond the gate: every layout and tail, and rows far from zero.

The statistics are float32 and the output is rounded to bfloat16 once, so the
error against the float64 reference is that rounding, below the bf16 floor of
the gate (4e-3). The cases hit the resident layout with a shared-memory bank
whose last slot is partly past the row, the rounds layout (a row over the whole
grid, one barrier per row) with a short last slice, a split row whose last
piece is short (16-byte and one-element loads), rows whose mean is far larger
than their spread (where a sum-of-squares formula would cancel), and constant
rows; the unfused variant (three launches) on the same cases.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the kernels run on CUDA", allow_module_level=True)

from bench.ops import SPECS
from kernels import layernorm

BF16_FLOOR = 4e-3
SPEC = SPECS["layernorm"]


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def _run(cfg, x, variant="fused"):
    (out,) = layernorm.build(cfg, variant)(x)
    torch.cuda.synchronize()
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    return out


LAYOUTS = [
    # 512 threads, 2 vectors in registers + 14 in shared memory, the last one partial
    (3, 65528, "resident"),
    (9, 1024, "resident"),  # whole slots only
    # 528 CTAs a row, 4 + 3 vectors a thread; the last CTA's slice is short
    (3, 8 * 811085, "rounds"),
    (2, 262464, "split"),  # 16-byte loads; the last piece is 40 vectors
    (2, 300001, "split"),  # one-element loads, several pieces, the last one short
]


@pytest.mark.parametrize("n, m, kind", LAYOUTS)
@pytest.mark.parametrize("variant", ["fused", "unfused"])
def test_layouts_and_tails(n, m, kind, variant):
    cfg = {"n": n, "m": m}
    if variant == "fused":
        assert layernorm._plan(n, m)["kind"] == kind
    gen = torch.Generator(device="cuda").manual_seed(7)
    (x,) = SPEC.inputs(cfg, gen)
    (ref,) = SPEC.reference(x)
    assert _err(_run(cfg, x, variant), ref) <= BF16_FLOOR
    if kind == "rounds" and variant == "fused":  # its barrier counters carry over between calls
        assert _err(_run(cfg, x, variant), ref) <= BF16_FLOOR


@pytest.mark.parametrize("n, m", [(64, 4096), (4, 16777216 // 64), (3, 4099), (2, 8 * 811085)])
def test_offset_rows(n, m):
    """Mean 300 against a spread of 4: Welford's merge keeps var accurate."""
    cfg = {"n": n, "m": m}
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = (4.0 * torch.randn(n, m, generator=gen, device="cuda") + 300.0).to(torch.bfloat16)
    (ref,) = SPEC.reference(x)
    assert _err(_run(cfg, x), ref) <= BF16_FLOOR


@pytest.mark.parametrize("n, m", [(4, 4096), (2, 1001), (2, 262464), (2, 8 * 811085)])
def test_constant_rows(n, m):
    """No spread: the output is (x - mean) / sqrt(eps) = 0, and nothing is NaN."""
    cfg = {"n": n, "m": m}
    x = torch.full((n, m), 1.5, device="cuda", dtype=torch.bfloat16)
    out = _run(cfg, x).float()
    assert torch.isfinite(out).all()
    assert out.abs().max().item() <= 1e-3
