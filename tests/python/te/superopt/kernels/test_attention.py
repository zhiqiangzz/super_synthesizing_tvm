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
"""The attention kernel on the tiles the gate's sizes do not reach.

The kernel takes 128 query rows (two warpgroups of 64) and 128 keys per tile,
and a persistent grid of one CTA per SM walks the query tiles. Here: a single
key, fewer keys than one row of an MMA fragment, a whole number of key tiles,
one key past a tile, many tiles with a one-key tail, more query tiles than
CTAs (the Q double buffer and the K/V ring run on across tiles), and both
head dims, for the fused kernel and the three-pass ablation. Scores far apart
(a peaked softmax, where the running maximum jumps between tiles) and equal
scores (a uniform softmax) check the online rescaling. Each output is held to
the gate's rule against the float64 reference, with torch's own program as
the yardstick.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the kernels run on CUDA", allow_module_level=True)

from bench.ops import SPECS  # noqa: E402
from kernels import attention  # noqa: E402

FLOOR = 4e-3  # one bfloat16 rounding of the largest output is up to 2e-3

CASES = [
    (1, 1, 1, 64),
    (1, 2, 8, 128),
    (2, 1, 64, 64),
    (1, 3, 128, 128),
    (1, 1, 129, 64),
    (3, 2, 255, 128),
    (1, 1, 2049, 128),
    (3, 50, 1000, 128),  # 1200 query tiles over 132 CTAs
    (2, 40, 513, 64),
]


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def _check(cfg, inputs, variant="fused"):
    spec = SPECS["attention"]
    (ref,) = spec.reference(*inputs)
    (base,) = spec.unfused(*inputs)
    (out,) = attention.build(cfg, variant=variant)(*inputs)
    assert out.shape == base.shape and out.dtype == base.dtype
    assert torch.isfinite(out.float()).all()
    assert _err(out, ref) <= max(2.0 * _err(base, ref), FLOOR)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, h, s, d", CASES)
def test_tails(b, h, s, d, variant):
    cfg = {"b": b, "h": h, "s": s, "d": d}
    for seed in range(2):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        _check(cfg, SPECS["attention"].inputs(cfg, gen), variant)


@pytest.mark.parametrize("d", [64, 128])
def test_peaked_scores(d):
    """Scores spread over hundreds: the running maximum moves by far more than exp's range."""
    cfg = {"b": 1, "h": 2, "s": 1000, "d": d}
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = SPECS["attention"].inputs(cfg, gen)
    _check(cfg, ((6.0 * q.float()).to(q.dtype), (6.0 * k.float()).to(k.dtype), v))


@pytest.mark.parametrize("s", [77, 1000])
def test_equal_scores(s):
    """K = 0: every key scores 0 and the output is the mean of V."""
    cfg = {"b": 1, "h": 2, "s": s, "d": 64}
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = SPECS["attention"].inputs(cfg, gen)
    (out,) = attention.build(cfg)(q, torch.zeros_like(k), v)
    mean = v.double().mean(dim=2, keepdim=True).expand_as(out)
    assert _err(out, mean) <= FLOOR
