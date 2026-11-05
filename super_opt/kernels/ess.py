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
"""The effective sample size of every row of log-weights, ``(Σ w)² / Σ w²``, in one pass.

``w = exp(lw - max lw)``. The algorithm is chain_ops' ``ess_fused`` (the
``squared_weights`` reducer): the state of a set of elements is
``(m, s1, s2)``, their maximum, ``s1 = Σ exp(x - m)`` and
``s2 = Σ exp(2 (x - m))``. Moving a state to a larger maximum ``m'`` scales
``s1`` by ``r = exp(m - m')`` and ``s2`` by ``r²``; merging two states moves
both to the larger maximum and adds (``ess_combine``). The epilogue is
``s1² / s2``; it does not depend on the maximum, so any common scale of the
weights cancels.

A thread folds a batch of loaded elements into its state at once
(``ess_fold``): it takes the batch's maximum, rescales its state once (one
``exp``; exactly 1 when the maximum did not move) and adds ``w`` and ``w²``
of every element, ``w = exp(x - m)``: one ``exp`` per element.

Precision (float32 throughout; the bound is 1e-6 relative):

* the weights use the fast ``__expf``, as Triton does, but the rescale
  factors use ``exp_accurate`` (~0.5 ulp, unbiased): a state is rescaled
  some 10-20 times on its way to the row's total, and ``__expf``'s bias of
  ~-3e-8 per call adds up along that chain (up to 7e-7 on the split rows);
* a batch's ``Σ w`` and ``Σ w²`` are summed as trees before they are added to
  the running sums: added one by one, the many weights far below the maximum
  fall under half an ulp of the running sum and are lost (a bias of up to
  9e-7 on rows of 262144).

How it is laid out:

* ``splits`` CTAs of 256 threads per row (one when there are enough rows to
  fill the GPU), and a grid of at most ``CTAS_PER_SM`` CTAs per SM that walks
  the ``(row, split)`` work items; a split row is cut into parts of a whole
  number of ``THREADS * UNROLL`` vectors, so only the last part has a tail;
* each thread issues ``UNROLL`` 16-byte loads (4 float32 each) before it
  folds them; when ``m`` is not a multiple of 4 the rows are not 16-byte
  aligned and it loads ``UNROLL`` single elements instead (decided when the
  kernel is built);
* ``cta_allreduce`` merges the 256 states; with one split, thread 0 writes
  the result; with several it writes the partial state ``(m, s1, s2)`` to a
  workspace of ``3 * n * splits`` floats, and a second kernel (one CTA per
  row) merges those and writes the result.
"""

from __future__ import annotations

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce

NEG = -3.0e38  # a finite "-inf": the identity of max, and exp(NEG - NEG) is no NaN
THREADS = 256
WARPS = THREADS // 32
UNROLL = 4  # loads in flight per thread before a fold
CTAS_PER_SM = 4  # the split plan aims at this many CTAs per SM
K = 3  # the state (m, s1, s2)
VEC = vector_width("float32")

KERNEL_META = {
    "op": "ess",
    "dtype": "float32",
    "algorithm": "one pass, state (m, s1, s2) = (max, Σ exp(x - max), Σ exp(2 (x - max))); "
    "ess = s1² / s2",
    "structure": "256-thread CTAs, one per row (or per part of a row when rows are few, "
    "merged by a second kernel), 4 x 16-byte loads per thread per step, warp then CTA all-reduce",
}


L2E = 1.4426950408889634  # log2(e)
LN2_HI = 0.693145751953125  # ln 2 in 16 bits, so that j * LN2_HI is exact
LN2_LO = 0.6931471805599453 - LN2_HI


def _fma(a, b, c):
    return T.call_pure_extern("__fmaf_rn", a, b, c, ty="float32")  # T.fma has no CUDA lowering


def exp_fast(x):
    """``__expf``: ``ex2.approx(x * log2 e)``, a few ulp off and biased low by ~3e-8."""
    return T.exp(x)


def exp_accurate(x):
    """``exp`` to ~0.5 ulp, without bias, for ``x <= 0``: ``2^j * p(r)``, ``x = j ln 2 + r``.

    ``j = round(x / ln 2)`` and ``r`` by Cody-Waite (``|r| <= ln 2 / 2``), ``p`` the
    degree-7 Taylor polynomial (error < 1e-8), ``2^j`` exact. ``exp(0)`` is exactly 1,
    and the clamp keeps ``exp(NEG)`` at 0 instead of NaN.
    """
    x = T.max(x, T.float32(-100.0))
    j = T.floor(x * T.float32(L2E) + T.float32(0.5))
    r = _fma(j, T.float32(-LN2_LO), _fma(j, T.float32(-LN2_HI), x))
    p = T.float32(1.0 / 5040)
    for c in (1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1.0, 1.0):
        p = _fma(p, r, T.float32(c))
    return p * T.exp2(j)


def _tree(op, leaf, lo, hi):
    """``leaf(lo) op ... op leaf(hi - 1)`` as a balanced tree: short chains, no term swamped."""
    if hi - lo == 1:
        return leaf(lo)
    mid = (lo + hi) // 2
    return op(_tree(op, leaf, lo, mid), _tree(op, leaf, mid, hi))


def _add(a, b):
    return a + b


@T.inline
def ess_combine(st, ot, exp_r):
    """(m, s1, s2) <- (m, s1, s2) ⊕ (m', s1', s2'): rescale both to the larger maximum and add.

    The reducer's ``exp(m - max)`` and ``exp(m' - max)``: one of the two is
    ``exp(0) = 1``, so only the other, ``r = exp(-|m - m'|)``, is computed.
    """
    r = exp_r(-T.abs(st[0] - ot[0]))
    keep = st[0] >= ot[0]
    st[1] = T.if_then_else(keep, st[1] + ot[1] * r, st[1] * r + ot[1])
    st[2] = T.if_then_else(keep, st[2] + ot[2] * (r * r), st[2] * (r * r) + ot[2])
    st[0] = T.max(st[0], ot[0])


@T.inline
def ess_fold(st, buf, tmp, count, exp_w, exp_r):
    """Fold the ``count`` elements of ``buf`` into ``st``.

    The batch's maximum, one rescale of ``st`` (exactly 1 when the maximum did
    not move), one ``exp`` per element, and the batch's ``Σ w`` and ``Σ w²``
    summed as trees before they are added: added one at a time, the many tiny
    weights far below the maximum fall under half an ulp of the running sum and
    are lost (a bias of ~1e-6 on rows of 262144).
    """
    tmp[0] = T.max(st[0], _tree(T.max, lambda k: buf[k], 0, count))
    tmp[1] = st[0] - tmp[0]
    tmp[1] = exp_r(tmp[1])
    st[0] = tmp[0]
    for k in T.unroll(count):
        buf[k] = exp_w(buf[k] - st[0])
    st[1] = st[1] * tmp[1] + _tree(_add, lambda k: buf[k], 0, count)
    st[2] = st[2] * (tmp[1] * tmp[1]) + _tree(_add, lambda k: buf[k] * buf[k], 0, count)


@T.inline
def load_vec(buf, k, A, row, v):
    """``buf[4k : 4k+4] = A[row, 4v : 4v+4]``: one 16-byte load (``v`` a vector index)."""
    buf[T.Ramp(k * VEC, 1, VEC)] = A[row, T.Ramp(v * VEC, 1, VEC)]


@T.inline
def ess_epilogue(Out, row, item, st):
    """The row's result ``s1² / s2`` (a correctly rounded division: fast-math ``/`` is ~2 ulp)."""
    Out[row] = T.call_pure_extern("__fdiv_rn", st[1] * st[1], st[2], ty="float32")


@T.inline
def ess_store_state(Out, row, item, st):
    """The work item's partial state, one column of the ``(K, items)`` workspace."""
    for j in T.unroll(K):
        Out[j, item] = st[j]


def plan(n: int, m: int) -> tuple[int, int]:
    """``(splits, chunk)``: parts per row, and loads per part (vectors; elements when unaligned)."""
    units = m // VEC if vectorisable(m, "float32") else m
    step = THREADS * UNROLL
    splits = max(1, min(SMS * CTAS_PER_SM // n, -(-units // step)))
    chunk = -(-units // splits)
    chunk = -(-chunk // step) * step  # a whole number of steps: only the last part has a tail
    splits = -(-units // chunk)
    return splits, chunk


def _partial_kernel(n: int, m: int, splits: int, chunk: int, exp_w=exp_fast, exp_r=exp_accurate):
    """Each ``(row, split)`` work item: the state of its part, or (one split) the row's result."""
    vectorised = vectorisable(m, "float32")
    units = m // VEC if vectorised else m
    width = VEC if vectorised else 1
    items = n * splits
    blocks = min(items, SMS * CTAS_PER_SM)
    final = splits == 1
    out_shape = (n,) if final else (K, items)
    store = ess_epilogue if final else ess_store_state  # both branches of a TIRx `if` are parsed

    @T.inline
    def combine(st, ot):
        ess_combine(st, ot, exp_r)

    @T.function
    def ess_partial(LW: T.Tensor((n, m), "float32", align=16), Out: T.Tensor(out_shape, "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")  # this thread's (m, s1, s2)
        ot = T.alloc_local((K,), "float32")  # the state being merged in
        tmp = T.alloc_local((2,), "float32")
        buf = T.alloc_local((UNROLL * width,), "float32")
        smem = T.alloc_shared((WARPS, K), "float32")
        item: T.int32 = bx
        row: T.int32 = 0
        end: T.int32 = 0
        u: T.int32 = 0
        while item < items:
            row = item // splits
            u = (item % splits) * chunk + tx
            end = T.min((item % splits) * chunk + chunk, units)
            st[0] = T.float32(NEG)
            st[1] = T.float32(0)
            st[2] = T.float32(0)
            if vectorised:
                while u + (UNROLL - 1) * THREADS < end:
                    for k in T.unroll(UNROLL):
                        load_vec(buf, k, LW, row, u + k * THREADS)
                    ess_fold(st, buf, tmp, UNROLL * VEC, exp_w, exp_r)
                    u += UNROLL * THREADS
                while u < end:
                    load_vec(buf, 0, LW, row, u)
                    ess_fold(st, buf, tmp, VEC, exp_w, exp_r)
                    u += THREADS
            else:
                while u + (UNROLL - 1) * THREADS < end:
                    for k in T.unroll(UNROLL):
                        buf[k] = LW[row, u + k * THREADS]
                    ess_fold(st, buf, tmp, UNROLL, exp_w, exp_r)
                    u += UNROLL * THREADS
                while u < end:
                    buf[0] = LW[row, u]
                    ess_fold(st, buf, tmp, 1, exp_w, exp_r)
                    u += THREADS
            cta_allreduce(st, ot, K, combine, smem, warp, lane, WARPS)
            if tx == 0:
                store(Out, row, item, st)
            item += blocks

    return ess_partial


def _merge_kernel(n: int, splits: int, exp_r=exp_accurate):
    """One CTA per row merges the row's ``splits`` partial states and writes its result."""

    @T.inline
    def combine(st, ot):
        ess_combine(st, ot, exp_r)

    @T.function
    def ess_merge(P: T.Tensor((K, n * splits), "float32"), Out: T.Tensor((n,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(n,), block=(THREADS,)))
        row = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")
        ot = T.alloc_local((K,), "float32")
        smem = T.alloc_shared((WARPS, K), "float32")
        st[0] = T.float32(NEG)
        st[1] = T.float32(0)
        st[2] = T.float32(0)
        s: T.int32 = tx
        while s < splits:
            for j in T.unroll(K):
                ot[j] = P[j, row * splits + s]
            combine(st, ot)
            s += THREADS
        cta_allreduce(st, ot, K, combine, smem, warp, lane, WARPS)
        if tx == 0:
            ess_epilogue(Out, row, row, st)

    return ess_merge


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant == "fused", variant
    n, m = cfg["n"], cfg["m"]
    splits, chunk = plan(n, m)
    partial = compile_sm90(_partial_kernel(n, m, splits, chunk), name=f"ess_partial_{n}x{m}")
    if splits == 1:

        def run(LW):
            out = torch.empty(n, device=LW.device, dtype=F32)
            partial(LW, out)
            return (out,)

        return run

    merge = compile_sm90(_merge_kernel(n, splits), name=f"ess_merge_{n}x{m}")

    def run(LW):
        part = torch.empty(K, n * splits, device=LW.device, dtype=F32)
        out = torch.empty(n, device=LW.device, dtype=F32)
        partial(LW, part)
        merge(part, out)
        return (out,)

    return run
