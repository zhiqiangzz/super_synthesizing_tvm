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
"""The papr kernel beyond the gate: every layout path, short rows, data off zero.

Rows shorter than a warp leave lanes with no element, whose state (count 0)
must merge as the identity; rows longer than ``WARP_ROW_MAX`` take the
one-CTA-per-row path, vectorised or not. Data at an offset from zero is where
the centred (Chan) merge matters: the kernel must stay within the gate's rule
of eager PyTorch's error there too.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import papr

CASES = [
    # (n, m): which path
    (4, 3),  # warp per row, scalar, most lanes empty
    (2, 36),  # warp per row, vectorised tail only, most lanes empty
    (9, 1024),  # warp per row, vectorised, one full chunk per lane
    (5, 16388),  # CTA per row, vectorised, chunks and a tail
    (3, 20001),  # CTA per row, scalar, chunks and a tail
]


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("n, m", CASES)
@pytest.mark.parametrize("offset", [0.0, 10.0])
def test_papr_against_float64(n, m, offset):
    # the kernel takes a warp per row up to WARP_ROW_MAX, a CTA per row beyond (_kernel)
    gen = torch.Generator(device="cuda").manual_seed(0)
    X = torch.randn(n, m, generator=gen, device="cuda") + offset
    (out,) = papr.build({"n": n, "m": m})(X)
    (ref,) = SPECS["papr"].reference(X)
    (eager,) = SPECS["papr"].unfused(X)
    assert _err(out, ref) <= max(2 * _err(eager, ref), 1e-6), (_err(out, ref), _err(eager, ref))
