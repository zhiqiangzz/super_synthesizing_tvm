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
"""The kurtosis kernel beyond the gate: every load path, split or not, data off zero.

Rows shorter than the CTA leave threads with no element, whose state (count 0)
must merge as the identity; a row split over CTAs whose length is not a
multiple of the step leaves a tail of single loads in its last segment, and
its partial states are merged by the second kernel. Data at an offset from
zero is where the centred (Pébay) merge matters: the kernel must stay within
the gate's rule of eager PyTorch's error there too.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import kurtosis

CASES = [
    # (n, m, split): which path
    (4, 3, False),  # scalar, most threads empty
    (2, 36, False),  # 16-byte loads, tail loads only, most threads empty
    (9, 1024, False),  # 16-byte loads, one full step per thread
    (2, 81948, True),  # 16-byte loads, split over CTAs, a tail in the last segment
    (3, 20001, True),  # scalar, split over CTAs, a tail in the last segment
]


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("n, m, split", CASES)
@pytest.mark.parametrize("offset", [0.0, 10.0])
def test_kurtosis_against_float64(n, m, split, offset):
    assert kurtosis._plan(n, m, kurtosis.FUSED).split == split
    gen = torch.Generator(device="cuda").manual_seed(0)
    X = torch.randn(n, m, generator=gen, device="cuda") + offset
    (out,) = kurtosis.build({"n": n, "m": m})(X)
    (ref,) = SPECS["kurtosis"].reference(X)
    (eager,) = SPECS["kurtosis"].unfused(X)
    assert _err(out, ref) <= max(2 * _err(eager, ref), 1e-6), (_err(out, ref), _err(eager, ref))


def test_count_exact_past_2_24():
    """A row of 2^24 + 4 elements: a float32 count would round, the int32 count does not."""
    n, m = 1, (1 << 24) + 4
    gen = torch.Generator(device="cuda").manual_seed(1)
    X = torch.randn(n, m, generator=gen, device="cuda")
    (out,) = kurtosis.build({"n": n, "m": m})(X)
    (ref,) = SPECS["kurtosis"].reference(X)
    assert _err(out, ref) <= 1e-6, _err(out, ref)
