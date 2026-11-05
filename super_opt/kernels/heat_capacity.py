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
"""The heat capacity of every row of energies, ``Σ (E - <E>)² w / Σ w``, in one pass.

``w = exp(-(E - min E))`` and ``<E> = Σ E w / Σ w``. The algorithm is
chain_ops' ``heat_capacity_fused`` (the ``boltzmann`` reducer): the state of a
set of elements is ``(lo, z, mu, M2)``, their lowest energy, the sum of their
weights relative to it ``z = Σ exp(lo - E)``, their weighted mean, kept
relative to ``lo`` (``mu = mean - lo``), and their weighted sum of squared
deviations from it ``M2 = Σ w (E - mean)²``. Merging two states
(``hc_combine``) moves both to the lower ``lo`` (``z`` and ``M2`` scale by
``exp(lo - lo_old)``, ``mu`` shifts by ``lo_old - lo``) and merges the means
and ``M2`` the way Welford's parallel algorithm does, weighted by ``z``. The
variance stays centred: no ``Σ E² w`` is formed, which would cancel. The
epilogue is ``M2 / z``.

Why ``mu`` and not the mean itself: a merge's ``M2`` is first-order sensitive
to the rounding of the merged means (``2 Σ z_c (mean_c - mean) δ_c``), and a
float32 mean is rounded by half an ulp of its magnitude. The states that weigh
most hold an energy near ``lo``, so their ``mu`` is small and rounds finely,
where ``mean`` itself would round by an ulp of ``|E|``; at (16384, 4096) this
takes the worst error from 7.5e-7 to 2.8e-7, torch.compile's. The merged
mean also steps from the heavier side's (``mu_a + d z_b / z`` or
``mu_b - d z_a / z``), which rounds less and merges an empty state exactly.

A thread folds a batch of loaded elements into its state at once
(``hc_fold``): the batch's weights are taken relative to ``min(lo, batch)``
(one ``exp`` per element), the batch's own mean and ``M2`` are formed in two
passes over the registers, and the batch's state is merged in. Since the batch
is weighted relative to the merged ``lo``, its factor in the merge is
``exp(0) = 1`` and the running state's is ``1`` unless the minimum moved.

How it is laid out:

* ``splits`` CTAs of 256 threads per row (one when there are enough rows to
  fill the GPU), and a grid of at most ``CTAS_PER_SM`` CTAs per SM that walks
  the ``(row, split)`` work items; a split row is cut into parts of a whole
  number of ``THREADS * UNROLL`` loads, so only the last part has a tail;
* each thread issues ``UNROLL`` 16-byte loads (4 float32 each) before it
  folds them; when ``m`` is not a multiple of 4 the rows are not 16-byte
  aligned and it loads ``UNROLL`` single elements instead (decided when the
  kernel is built);
* ``cta_allreduce`` merges the 256 states; with one split, thread 0 writes
  the result; with several it writes the part's state ``(lo, z, mu, M2)``,
  16 bytes, to a workspace of ``n * splits`` states, and a second kernel (one
  CTA per row) merges those and writes the result.
"""

from __future__ import annotations

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce

POS = 3.0e38  # a finite "+inf": the identity of min, and exp(POS - POS) is 1, no NaN
THREADS = 256
WARPS = THREADS // 32
UNROLL = 4  # loads in flight per thread before a fold
CTAS_PER_SM = 8  # the split plan aims at this many CTAs per SM
K = 4  # the state (lo, z, mu, M2)
VEC = vector_width("float32")
MERGE_THREADS = 256

KERNEL_META = {
    "op": "heat_capacity",
    "dtype": "float32",
    "algorithm": "one pass, state (lo, z, mu, M2) = (min E, Σ exp(lo - E), weighted mean - lo, "
    "weighted Σ (E - mean)²), merged as a z-weighted Welford; cv = M2 / z",
    "structure": "256-thread CTAs, one per row (or per part of a row when rows are few, "
    "merged by a second kernel), 4 x 16-byte loads per thread per fold, warp then CTA all-reduce",
}


@T.inline
def hc_combine(st, ot):
    """(lo, z, mu, M2) <- (lo, z, mu, M2) ⊕ (lo', z', mu', M2').

    Both move to the lower ``lo``: ``z`` and ``M2`` scale by ``exp(lo - lo_old)``
    and ``mu`` (the mean relative to ``lo``) shifts by ``lo_old - lo``. Then the
    means and ``M2`` merge as in Welford's parallel algorithm, weighted by the
    rescaled ``z``; the merged mean steps from the heavier side's mean, by the
    lighter side's share of ``d``. ``st`` is written in an order where every
    right-hand side still reads the old state.

    An empty state ``(POS, 0, 0, 0)`` merges to the other state exactly: its
    weight is 0 and every term it would bring is a finite number times 0.
    """
    lo = T.min(st[0], ot[0])
    ra = T.exp(lo - st[0])
    rb = T.exp(lo - ot[0])
    za = st[1] * ra
    zb = ot[1] * rb
    z = za + zb
    inv = T.if_then_else(z > T.float32(0), T.float32(1) / z, T.float32(0))  # 0 when both are empty
    ga = za * inv
    gb = zb * inv
    mua = st[2] + (st[0] - lo)
    mub = ot[2] + (ot[0] - lo)
    d = mub - mua
    st[3] = st[3] * ra + ot[3] * rb + d * (d * (za * gb))
    st[2] = T.if_then_else(za >= zb, mua + d * gb, mub - d * ga)
    st[1] = z
    st[0] = lo


@T.inline
def hc_fold(st, ot, buf, w, count):
    """Fold the ``count`` elements of ``buf`` into ``st``: one exp per element.

    The batch is weighted relative to ``lo = min(st.lo, batch)``; its mean and
    ``M2`` are formed in two passes over the registers, relative to ``lo``
    (``buf`` is overwritten), and then merged into ``st``.
    """
    ot[0] = st[0]
    for k in T.unroll(count):
        ot[0] = T.min(ot[0], buf[k])
    ot[1] = T.float32(0)
    ot[2] = T.float32(0)
    for k in T.unroll(count):
        buf[k] = buf[k] - ot[0]  # E - lo >= 0
        w[k] = T.exp(-buf[k])
        ot[1] = ot[1] + w[k]
        ot[2] = ot[2] + w[k] * buf[k]
    ot[2] = T.if_then_else(ot[1] > T.float32(0), ot[2] / ot[1], T.float32(0))  # mu = mean - lo
    ot[3] = T.float32(0)
    for k in T.unroll(count):
        buf[k] = buf[k] - ot[2]  # E - mean
        ot[3] = ot[3] + w[k] * buf[k] * buf[k]
    hc_combine(st, ot)


@T.inline
def hc_identity(st):
    st[0] = T.float32(POS)
    st[1] = T.float32(0)
    st[2] = T.float32(0)
    st[3] = T.float32(0)


@T.inline
def hc_epilogue(Out, row, st):
    Out[row] = st[3] / st[1]


@T.inline
def hc_write_result(Out, row, item, st):
    """One split: the row's result."""
    hc_epilogue(Out, row, st)


@T.inline
def hc_write_state(Out, row, item, st):
    """Several splits: the part's state, 16 bytes."""
    Out[item, T.Ramp(0, 1, K)] = st[T.Ramp(0, 1, K)]


def plan(n: int, m: int) -> tuple[int, int]:
    """``(splits, chunk)``: parts per row, and loads per part (vectors; elements when unaligned)."""
    units = m // VEC if vectorisable(m, "float32") else m
    step = THREADS * UNROLL
    splits = max(1, min(SMS * CTAS_PER_SM // n, -(-units // step)))
    chunk = -(-units // splits)
    chunk = -(-chunk // step) * step  # a whole number of steps: only the last part has a tail
    splits = -(-units // chunk)
    return splits, chunk


def _partial_kernel(n: int, m: int, splits: int, chunk: int):
    """Each ``(row, split)`` work item: the state of its part, or (one split) the row's result."""
    vectorised = vectorisable(m, "float32")
    units = m // VEC if vectorised else m
    width = VEC if vectorised else 1
    items = n * splits
    blocks = min(items, SMS * CTAS_PER_SM)
    final = splits == 1
    out_shape = (n,) if final else (items, K)
    # chosen here, not by an `if` in the kernel: the parser builds both arms of an `if`
    write = hc_write_result if final else hc_write_state

    @T.function
    def heat_capacity_partial(
        E: T.Tensor((n, m), "float32", align=16), Out: T.Tensor(out_shape, "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")  # this thread's (lo, z, mu, M2)
        ot = T.alloc_local((K,), "float32")  # the state being merged in
        buf = T.alloc_local((UNROLL * width,), "float32")
        w = T.alloc_local((UNROLL * width,), "float32")
        smem = T.alloc_shared((WARPS, K), "float32")
        item: T.int32 = bx
        row: T.int32 = 0
        end: T.int32 = 0
        u: T.int32 = 0
        while item < items:
            row = item // splits
            u = (item % splits) * chunk + tx
            end = T.min((item % splits) * chunk + chunk, units)
            hc_identity(st)
            if vectorised:
                while u + (UNROLL - 1) * THREADS < end:
                    for k in T.unroll(UNROLL):
                        buf[T.Ramp(k * VEC, 1, VEC)] = E[
                            row, T.Ramp((u + k * THREADS) * VEC, 1, VEC)
                        ]
                    hc_fold(st, ot, buf, w, UNROLL * VEC)
                    u += UNROLL * THREADS
                while u < end:
                    buf[T.Ramp(0, 1, VEC)] = E[row, T.Ramp(u * VEC, 1, VEC)]
                    hc_fold(st, ot, buf, w, VEC)
                    u += THREADS
            else:
                while u + (UNROLL - 1) * THREADS < end:
                    for k in T.unroll(UNROLL):
                        buf[k] = E[row, u + k * THREADS]
                    hc_fold(st, ot, buf, w, UNROLL)
                    u += UNROLL * THREADS
                while u < end:
                    buf[0] = E[row, u]
                    hc_fold(st, ot, buf, w, 1)
                    u += THREADS
            cta_allreduce(st, ot, K, hc_combine, smem, warp, lane, WARPS)
            if tx == 0:
                write(Out, row, item, st)
            item += blocks

    return heat_capacity_partial


def _merge_kernel(n: int, splits: int):
    """One CTA per row merges the row's ``splits`` partial states and writes its result."""

    @T.function
    def heat_capacity_merge(
        P: T.Tensor((n * splits, K), "float32", align=16), Out: T.Tensor((n,), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(n,), block=(MERGE_THREADS,)))
        row = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")
        ot = T.alloc_local((K,), "float32")
        smem = T.alloc_shared((MERGE_THREADS // 32, K), "float32")
        hc_identity(st)
        s: T.int32 = tx
        while s < splits:
            ot[T.Ramp(0, 1, K)] = P[row * splits + s, T.Ramp(0, 1, K)]
            hc_combine(st, ot)
            s += MERGE_THREADS
        cta_allreduce(st, ot, K, hc_combine, smem, warp, lane, MERGE_THREADS // 32)
        if tx == 0:
            hc_epilogue(Out, row, st)

    return heat_capacity_merge


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant == "fused", variant
    n, m = cfg["n"], cfg["m"]
    splits, chunk = plan(n, m)
    partial = compile_sm90(
        _partial_kernel(n, m, splits, chunk), name=f"heat_capacity_partial_{n}x{m}"
    )
    if splits == 1:

        def run(E):
            out = torch.empty(n, device=E.device, dtype=F32)
            partial(E, out)
            return (out,)

        return run

    merge = compile_sm90(_merge_kernel(n, splits), name=f"heat_capacity_merge_{n}x{m}")

    def run(E):
        part = torch.empty(n * splits, K, device=E.device, dtype=F32)
        out = torch.empty(n, device=E.device, dtype=F32)
        partial(E, part)
        merge(part, out)
        return (out,)

    return run
