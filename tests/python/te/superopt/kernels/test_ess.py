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
"""The ess kernel beyond the gate: every load path, split or not, and exact cases.

Rows shorter than the CTA leave threads with no element, whose state
``(NEG, 0, 0)`` must merge as the identity; a row split over CTAs whose length
is not a multiple of the step leaves a tail of single loads in its last part,
and its partial states are merged by the second kernel. Wider log-weights put
nearly all of ``Σ w²`` on a few elements; an offset moves the maximum far from
zero. Equal log-weights give exactly ``m``, one dominant one exactly 1.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import ess

CASES = [
    # (n, m, split): which path
    (4, 3, False),  # scalar, most threads empty
    (2, 36, False),  # 16-byte loads, tail loads only, most threads empty
    (9, 4096, False),  # 16-byte loads, one full step per thread
    (2, 81948, True),  # 16-byte loads, split over CTAs, a tail in the last part
    (3, 20001, True),  # scalar, split over CTAs, a tail in the last part
]


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("n, m, split", CASES)
@pytest.mark.parametrize("scale, offset", [(3.0, 0.0), (10.0, 0.0), (3.0, 100.0)])
def test_ess_against_float64(n, m, split, scale, offset):
    assert (ess.plan(n, m)[0] > 1) == split
    gen = torch.Generator(device="cuda").manual_seed(0)
    LW = torch.randn(n, m, generator=gen, device="cuda") * scale + offset
    (out,) = ess.build({"n": n, "m": m})(LW)
    (ref,) = SPECS["ess"].reference(LW)
    (eager,) = SPECS["ess"].unfused(LW)
    assert _err(out, ref) <= max(2 * _err(eager, ref), 1e-6), (_err(out, ref), _err(eager, ref))


@pytest.mark.parametrize("n, m, split", CASES)
def test_ess_exact_cases(n, m, split):
    LW = torch.full((n, m), -7.5, device="cuda")
    (out,) = ess.build({"n": n, "m": m})(LW)
    assert torch.equal(out, torch.full((n,), float(m), device="cuda")), out  # all weights 1
    LW = torch.randn(n, m, device="cuda")
    LW[:, m // 2] = 200.0  # every other weight underflows: exp(< -190) = 0
    (out,) = ess.build({"n": n, "m": m})(LW)
    assert torch.equal(out, torch.ones(n, device="cuda")), out
