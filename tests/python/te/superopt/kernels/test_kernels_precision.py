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
"""Every TIRx kernel that exists is as accurate as torch.compile on the small sizes.

The full gate (``python -m kernels.check --all --sizes all``) also runs the
benchmark sizes; here only the small, odd ones, which still take every tail
and unaligned path. An operator without a kernel module yet is skipped.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the kernels run on CUDA", allow_module_level=True)

from kernels import OPS, available  # noqa: E402
from kernels.check import check_op  # noqa: E402
from kernels.common.build import compile_sm90  # noqa: E402
from kernels.common.memory import load_row, store_row, vector_width, vectorisable  # noqa: E402
from rich.console import Console  # noqa: E402

from tvm_tirx.script import tirx as T  # noqa: E402

CHECKED = ["common.example_logsumexp", *OPS]


@pytest.mark.parametrize("op", CHECKED)
def test_precision_on_small_sizes(op):
    if not available(op):
        pytest.skip(f"no kernels/{op}.py yet")
    rows = check_op(op, "small", (0, 1, 2), Console(quiet=True))
    failures = [r for r in rows if not r.passed]
    assert not failures, "\n".join(
        f"{r.cfg} seed {r.seed} out {r.output}: err {r.err_tirx} > {r.bound} {r.error}"
        for r in failures
    )


def _double_rows(n, m, dtype):
    """Out = 2 * A through ``load_row`` / ``store_row`` (or element by element when unaligned)."""
    vec = vector_width(dtype)
    vectorised = vectorisable(m, dtype)

    @T.function
    def double(A: T.Tensor((n, m), dtype, align=16), Out: T.Tensor((n, m), dtype, align=16)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(n,), block=(128,)))
        row = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        buf = T.alloc_local((vec,), dtype)
        if vectorised:
            v: T.int32 = tx
            while v < m // vec:
                load_row(buf, A, row, v, vec)
                for k in T.unroll(vec):
                    buf[k] = T.Cast(dtype, T.Cast("float32", buf[k]) * T.float32(2))
                store_row(Out, row, v, buf, vec)
                v += 128
        else:
            j: T.int32 = tx
            while j < m:
                Out[row, j] = T.Cast(dtype, T.Cast("float32", A[row, j]) * T.float32(2))
                j += 128

    return compile_sm90(double)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("m", [1000, 1001, 4096])
def test_row_loads_and_stores(dtype, m):
    n = 3
    a = torch.randn(n, m, device="cuda").to(getattr(torch, dtype))
    out = torch.empty_like(a)
    _double_rows(n, m, dtype)(a, out)
    torch.cuda.synchronize()
    assert torch.equal(out, a * 2)
