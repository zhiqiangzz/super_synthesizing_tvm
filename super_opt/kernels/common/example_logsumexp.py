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
"""A complete one-pass row reduction: log Σ exp of every row, in TIRx. A template.

Not a showcase operator: it is the smallest kernel that has every part the
row reductions need, so each operator's kernel can start from it.

The algorithm is chain_ops' ``logsumexp_fused``: the state is ``(m, l)``, a
running maximum and the sum of ``exp(x - m)``; merging two states keeps the
larger maximum and rescales the other sum by ``exp(m_other - m)``
(``lse_combine``). The result of a row is ``m + log l``.

How it is laid out:

* one CTA of 256 threads (8 warps) per row, and a grid of at most 8 CTAs per
  SM that walks the rows (``row += blocks``);
* each thread reads its row's elements 16 bytes (8 bf16) at a time, folds the
  8 into a state of their own and merges that into its running state; when
  ``m`` is not a multiple of 8 the rows are not 16-byte aligned and it reads
  one element at a time instead (``vectorised``, decided when the kernel is
  built);
* ``cta_allreduce`` merges the 256 states; thread 0 writes the row's result.

``SPEC`` and ``SHAPES`` stand in for ``bench.ops`` and ``bench.shapes``, which
only know the showcase, so that ``python -m kernels.check --op
common.example_logsumexp`` checks this kernel like any other.
"""

from __future__ import annotations

import torch
from bench.ops import BF16, F32, OpSpec

from tvm_tirx.script import tirx as T

from .build import SMS, cached, compile_sm90
from .memory import load_row, vector_width, vectorisable
from .reduce import cta_allreduce

NEG = -3.0e38  # a finite "-inf": the identity of max, and exp(NEG - NEG) is no NaN
THREADS = 256
WARPS = THREADS // 32

KERNEL_META = {
    "op": "example_logsumexp",
    "dtype": "bfloat16 in, float32 out",
    "algorithm": "one pass, state (m, l) = (max, Σ exp(x - max))",
    "structure": "one 256-thread CTA per row, 16-byte loads, warp then CTA all-reduce",
}


@T.inline
def lse_combine(st, ot):
    """(m, l) <- (m, l) ⊕ (m', l'): keep the larger maximum, rescale both sums to it."""
    mx = T.max(st[0], ot[0])
    st[1] = st[1] * T.exp(st[0] - mx) + ot[1] * T.exp(ot[0] - mx)
    st[0] = mx


def _kernel(n: int, m: int):
    vec = vector_width("bfloat16")
    vectorised = vectorisable(m, "bfloat16")
    nvec = m // vec
    blocks = min(n, SMS * 8)

    @T.function
    def logsumexp(A: T.Tensor((n, m), "bfloat16", align=16), Out: T.Tensor((n,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((2,), "float32")  # this thread's (m, l)
        ot = T.alloc_local((2,), "float32")  # the state being merged in
        buf = T.alloc_local((vec,), "bfloat16")
        smem = T.alloc_shared((WARPS, 2), "float32")
        row: T.int32 = bx
        while row < n:
            st[0] = T.float32(NEG)
            st[1] = T.float32(0)
            if vectorised:
                v: T.int32 = tx
                while v < nvec:
                    load_row(buf, A, row, v, vec)
                    ot[0] = T.float32(NEG)
                    for k in T.unroll(vec):
                        ot[0] = T.max(ot[0], T.Cast("float32", buf[k]))
                    ot[1] = T.float32(0)
                    for k in T.unroll(vec):
                        ot[1] = ot[1] + T.exp(T.Cast("float32", buf[k]) - ot[0])
                    lse_combine(st, ot)
                    v += THREADS
            else:
                j: T.int32 = tx
                while j < m:
                    ot[0] = T.Cast("float32", A[row, j])
                    ot[1] = T.float32(1)
                    lse_combine(st, ot)
                    j += THREADS
            cta_allreduce(st, ot, 2, lse_combine, smem, warp, lane, WARPS)
            if tx == 0:
                Out[row] = st[0] + T.log(st[1])
            row += blocks

    return logsumexp


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant == "fused", variant
    n, m = cfg["n"], cfg["m"]
    mod = compile_sm90(_kernel(n, m), name=f"example_logsumexp_{n}x{m}")

    def run(X):
        out = torch.empty(n, device=X.device, dtype=F32)
        mod(X, out)
        return (out,)

    return run


# --- what bench.ops and bench.shapes would say, were this a showcase operator ---
def _inputs(cfg, gen):
    x = torch.randn(cfg["n"], cfg["m"], generator=gen, device="cuda", dtype=F32)
    return ((2.0 * x).to(BF16),)


def _unfused(X):
    x = X.float()
    mx = x.amax(dim=1, keepdim=True)
    return ((mx + (x - mx).exp().sum(dim=1, keepdim=True).log())[:, 0],)


def _reference(X):
    return (torch.logsumexp(X.double(), dim=1),)


SPEC = OpSpec(
    "example_logsumexp",
    BF16,
    _inputs,
    _unfused,
    _reference,
    lambda cfg: cfg["n"] * cfg["m"] * 2 + cfg["n"] * 4,
    tol=1e-3,
    exps=lambda cfg: cfg["n"] * cfg["m"],
)
SHAPES = {
    "short": [{"n": 32768, "m": 4096}],
    "long": [{"n": 2048, "m": 131072}],
}
