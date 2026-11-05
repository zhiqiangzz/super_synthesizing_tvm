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
"""The householder kernel on the layouts the gate's sizes do not reach.

The gate (``kernels.check``) takes the warp path aligned and not, the CTA path
unaligned, and an unaligned split. Here also: rows shorter than one vector, a
split of aligned rows, a split whose chunks are uneven, and a general ``E``
(the gate's is the first unit vector), each to 1e-6 of the float64 reference.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the kernels run on CUDA", allow_module_level=True)

from bench.ops import SPECS  # noqa: E402
from kernels import householder  # noqa: E402

CFGS = [
    ({"n": 3, "m": 3}, "warp", 1),  # every row inside one vector, or across two
    ({"n": 9, "m": 5}, "warp", 1),
    ({"n": 37, "m": 255}, "warp", 1),
    ({"n": 300, "m": 2047}, "cta", 1),
    ({"n": 3, "m": 36004}, "cta", 4),  # aligned rows, split
    ({"n": 2, "m": 1 << 20}, "cta", 128),
    ({"n": 1, "m": 77781}, "cta", 9),  # uneven chunks, head and tail
]


@pytest.mark.parametrize("cfg, path, splits", CFGS, ids=lambda c: str(c))
@pytest.mark.parametrize("general_e", [False, True])
def test_householder_layouts(cfg, path, splits, general_e):
    plan = householder.plan(cfg["n"], cfg["m"])
    group = {"warp": 32, "cta": householder.THREADS}[path]  # threads per row
    assert (plan["group"], plan["splits"]) == (group, splits), plan
    spec = SPECS["householder"]
    gen = torch.Generator(device="cuda").manual_seed(0)
    X, E = spec.inputs(cfg, gen)
    if general_e:
        E = torch.randn(X.shape, generator=gen, device="cuda")
    (ref,) = spec.reference(X, E)
    (out,) = householder.build(cfg)(X, E)
    assert out.shape == ref.shape and out.dtype == torch.float32
    err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    assert err <= 1e-6, err
