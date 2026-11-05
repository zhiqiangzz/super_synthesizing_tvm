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
"""softmax(Q Kᵀ / sqrt(d)) V, not causal, FlashAttention-3 style on Hopper, in TIRx.

The algorithm is chain_ops' attention ``_fused(False)``: for every query row
the state is ``(m, l, acc)``, the running maximum of the scores, the sum of
``exp((s - m) / sqrt(d))`` and the unnormalised output ``Σ exp(...) v``.
Folding in a tile of keys with scores ``s_j`` (``online_softmax``):

    m'   = max(m, max_j s_j)
    a    = exp((m - m') / sqrt(d))
    p_j  = exp((s_j - m') / sqrt(d))
    l'   = a l + Σ_j p_j
    acc' = a acc + Σ_j p_j v_j

and the output of a row is ``acc / l``. The exponentials are ``exp2`` of the
score times ``log2(e) / sqrt(d)`` (one FMA and one ``ex2.approx`` each); ``m``
is kept unscaled.

How it is laid out (FlashAttention-3):

* a persistent grid, one CTA per SM, walks the (128-query-row tile, head)
  pairs in order, so the CTAs at work share a few heads' K and V in L2;
* three warpgroups. Warpgroup 0 is the producer: one thread issues TMA loads
  (``cp.async.bulk.tensor``, 128-byte swizzle, rows past ``s`` zero-filled by
  the 3-d tensor map) of Q tiles into two buffers (the next tile's Q arrives
  during this one) and of 128-key K and V tiles into a 2-stage ring that runs
  on across query tiles, all with full/empty mbarriers; ``setmaxnreg`` gives
  its registers to the two consumer warpgroups, 64 query rows each;
* a consumer computes ``S = Q Kᵀ`` with ``wgmma`` (both operands in shared
  memory, fp32 accumulators in registers), the online softmax on the
  accumulators, and ``acc += P V`` with ``wgmma`` whose A operand is P,
  rounded to bf16 in registers (V is an MN-major B operand, read as it lies).
  S and P never leave registers;
* within a warpgroup, ``S_j = Q K_jᵀ`` and ``acc += P_{j-1} V_{j-1}`` are
  issued together and the softmax of ``S_j`` runs while the second is in
  flight (``kv_step`` says what it takes to keep ptxas from serializing
  them); at d = 128 named barriers also alternate the two warpgroups' GEMM
  issue (ping-pong);
* keys past ``s`` in the last tile score ``-3e38`` (a finite "-inf"); that
  step is peeled off, so the others carry no test;
* the normalised output goes through shared memory (swizzled like a TMA
  tile) and out with a TMA store, which clips the rows past ``s``, while the
  next tile starts. One kernel launch.

``variant="unfused"`` is the ablation: three launches of the same kernel, one
per reduction of the unfused program. Pass 1 recomputes S and keeps the row
maximum, pass 2 recomputes S and keeps ``Σ exp``, pass 3 recomputes S, forms
``P = exp(S - m) / l`` and computes ``P V``. Only the row maxima and sums
travel between passes; S is never stored.
"""

from __future__ import annotations

import math

import torch
from bench.ops import BF16, F32

from tvm_tirx.backend.cuda.lang import SMEMPool
from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90

NEG = -3.0e38  # a finite "-inf": the identity of max, and exp(NEG - NEG) is no NaN
BN_FOR = {64: 128, 128: 128}  # keys per tile, by head dim
STAGES_FOR = {64: 2, 128: 2}  # K and V ring depth, by head dim
NC = 2  # consumer warpgroups, 64 query rows each
BM = 64 * NC  # query rows per CTA
THREADS = 128 * (NC + 1)
FULL_MASK = 0xFFFFFFFF
TMA_LOAD = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
TMA_STORE = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
PRODUCER_REGS = 24
CONSUMER_REGS = 240
# named-barrier ping-pong of the two consumer warpgroups' GEMMs (FlashAttention-3); measured:
# 1% faster at d = 128, 2% slower at d = 64, where the softmax is the longer part
PINGPONG_FOR = {64: False, 128: True}
PERSISTENT = True  # a grid of one CTA per SM walks the (query tile, head) pairs

KERNEL_META = {
    "op": "attention",
    "dtype": "bfloat16",
    "algorithm": "one pass over the keys, state (m, l, acc) = (max score, Σ exp, Σ exp·v), "
    "out = acc / l",
    "structure": "FlashAttention-3: persistent CTAs over 128-query-row tiles; a TMA producer "
    "warpgroup and 2 wgmma consumer warpgroups (ping-pong, softmax overlapped with the "
    "GEMMs), 128-key tiles in a 2-stage mbarrier ring, S and P in registers, TMA store",
}


@T.inline
def fmax(a, b):
    """fmaxf: one FMNMX (T.max is a NaN-propagating compare and select)."""
    return T.call_pure_extern("fmaxf", a, b, ty="float32")


@T.inline
def quad_max(x):
    """Max over the 4 lanes that share a row of a wgmma accumulator."""
    y = fmax(x, T.gpu_warp_shuffle_xor(FULL_MASK, x, 1, 32, 32))
    return fmax(y, T.gpu_warp_shuffle_xor(FULL_MASK, y, 2, 32, 32))


@T.inline
def quad_sum(x):
    """Sum over the 4 lanes that share a row of a wgmma accumulator."""
    y = x + T.gpu_warp_shuffle_xor(FULL_MASK, x, 1, 32, 32)
    return y + T.gpu_warp_shuffle_xor(FULL_MASK, y, 2, 32, 32)


def _kernel(b: int, h: int, s: int, d: int, mode: str):
    """``mode``: "fused", or one pass of the ablation: "max", "sum", "out"."""
    assert d in (64, 128), d
    assert mode in ("fused", "max", "sum", "out"), mode
    bh = b * h
    rows = bh * s
    nch = d // 64  # 64-column (128-byte) chunks of a row
    mblocks = (s + BM - 1) // BM
    bn = BN_FOR[d]
    stages = STAGES_FOR[d]
    pingpong = PINGPONG_FOR[d]
    nblocks = (s + bn - 1) // bn
    tail = s % bn != 0
    scale = math.log2(math.e) / math.sqrt(d)  # exp(x / sqrt(d)) = exp2(x * scale)
    kv_bytes = bn * d * 2
    chunk_q = 64 * 128 // 16  # descriptor units (16 bytes) per 64-row chunk of Q
    chunk_kv = bn * 128 // 16  # ... per chunk of a K or V tile
    stage_kv = nch * chunk_kv  # ... per K or V stage
    use_v = mode in ("fused", "out")
    is_fused, is_max, is_out = mode == "fused", mode == "max", mode == "out"
    reads_m = mode in ("sum", "out")
    wg_s = f"wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16"
    wg_o = f"wgmma.mma_async.sync.aligned.m64n{d}k16.f32.bf16.bf16"
    ntiles = mblocks * bh  # (query tile, head) pairs, walked by a persistent grid
    grid = min(ntiles, SMS) if PERSISTENT else ntiles
    # mbarrier slots: Q full, Q empty, K full, V full, K empty, V empty
    b_qf, b_qe = 0, 2  # two Q buffers: query tile k uses buffer k % 2
    b_kf, b_vf, b_ke, b_ve = 4, 4 + stages, 4 + 2 * stages, 4 + 3 * stages
    nbar = 4 + 4 * stages
    q_buf = NC * nch * chunk_q  # descriptor units per Q buffer

    @T.inline
    def gemm_s(sacc, dq, dk, st):
        """S = Q K[st]ᵀ, issued (64 x bn per warpgroup); the first k-step overwrites S."""
        T.ptx.wgmma.fence.sync.aligned()
        for kk in T.unroll(d // 16):
            T.ptx[wg_s](
                *[sacc[i] for i in range(bn // 2)],
                dq + T.Cast("uint64", kk // 4 * chunk_q + kk % 4 * 2),
                dk + T.Cast("uint64", st * stage_kv + kk // 4 * chunk_kv + kk % 4 * 2),
                kk > 0,
                1,
                1,
                0,
                0,
            )
        T.ptx.wgmma.commit_group.sync.aligned()

    @T.inline
    def gemm_o(oacc, pw, dv, st):
        """acc += P V[st], issued; P's words 4 kk .. 4 kk + 3 are the A fragment of keys 16 kk.."""
        T.ptx.wgmma.fence.sync.aligned()
        for kk in T.unroll(bn // 16):
            T.ptx[wg_o](
                *[oacc[i] for i in range(d // 2)],
                pw[kk * 4],
                pw[kk * 4 + 1],
                pw[kk * 4 + 2],
                pw[kk * 4 + 3],
                dv + T.Cast("uint64", st * stage_kv + kk * 128),
                True,
                1,
                1,
                1,
            )
        T.ptx.wgmma.commit_group.sync.aligned()

    @T.inline
    def mask(sacc, n0, t):
        """Keys past s score NEG; called on the last tile only."""
        for i in T.unroll(bn // 2):
            key = n0 + i // 4 * 8 + 2 * t + i % 2
            sacc[i] = T.if_then_else(key < s, sacc[i], T.float32(NEG))

    @T.inline
    def online_softmax(sacc, st, tmp):
        """Fold the tile into (m, l); S becomes p = exp2(s scale - m scale); st[4 + r] = alpha."""
        for r in T.unroll(2):
            tmp[0] = st[r]
            tmp[1] = st[r]
            for jt in T.unroll(bn // 8):
                tmp[0] = fmax(tmp[0], sacc[jt * 4 + r * 2])
                tmp[1] = fmax(tmp[1], sacc[jt * 4 + r * 2 + 1])
            tmp[0] = quad_max(fmax(tmp[0], tmp[1]))
            st[4 + r] = T.exp2((st[r] - tmp[0]) * T.float32(scale))
            st[r] = tmp[0]
            tmp[1] = tmp[0] * T.float32(scale)
            tmp[2] = T.float32(0)
            tmp[3] = T.float32(0)
            for jt in T.unroll(bn // 8):
                sacc[jt * 4 + r * 2] = T.exp2(sacc[jt * 4 + r * 2] * T.float32(scale) - tmp[1])
                sacc[jt * 4 + r * 2 + 1] = T.exp2(
                    sacc[jt * 4 + r * 2 + 1] * T.float32(scale) - tmp[1]
                )
                tmp[2] = tmp[2] + sacc[jt * 4 + r * 2]
                tmp[3] = tmp[3] + sacc[jt * 4 + r * 2 + 1]
            st[2 + r] = st[2 + r] * st[4 + r] + (tmp[2] + tmp[3])

    @T.inline
    def fixed_softmax(sacc, st, tmp):
        """Pass 3: p = exp2(s scale - m scale) / l with the final m (st[0:2]), 1/l (st[2:4])."""
        for r in T.unroll(2):
            tmp[1] = st[r] * T.float32(scale)
            for jt in T.unroll(bn // 8):
                for e in T.unroll(2):
                    i = jt * 4 + r * 2 + e
                    sacc[i] = T.exp2(sacc[i] * T.float32(scale) - tmp[1]) * st[2 + r]

    @T.inline
    def rescale(oacc, st):
        for jt in T.unroll(d // 8):
            for i in T.unroll(4):
                oacc[jt * 4 + i] = oacc[jt * 4 + i] * st[4 + i // 2]

    @T.inline
    def to_p(sacc, pb):
        for i in T.unroll(bn // 2):
            pb[i] = T.Cast("bfloat16", sacc[i])

    @T.inline
    def kv_step(sacc, oacc, pb, st, tmp, desc, bars, n, w, lane, t, j, final, zero, qslot):
        """Step j of a query tile (ring position n): issue S_j and acc += P_{j-1} V_{j-1},
        then the softmax of S_j while the latter runs. No branch between the two waits, or
        ptxas serializes the wgmma (it hoists the second wait above the softmax)."""
        pw = pb.view("uint32")
        cur = n % stages
        prv = (n - 1) % stages
        T.cuda.mbarrier_wait(bars.ptr_to([b_kf + cur]), n // stages % 2)
        if pingpong:
            T.ptx.bar.sync(T.Cast("uint32", 1 + w), T.uint32(256))
        gemm_s(sacc, desc[3], desc[1], cur)
        if is_fused:
            rescale(oacc, st)
        T.cuda.mbarrier_wait(bars.ptr_to([b_vf + prv]), (n - 1) // stages % 2)
        gemm_o(oacc, pw, desc[2], prv)
        if pingpong:
            T.ptx.bar.arrive(T.Cast("uint32", 2 - w), T.uint32(256))
        T.ptx.wgmma.wait_group.sync.aligned(1)  # S_j is done
        T.ptx.mbarrier.arrive.shared.b64(bars.ptr_to([b_ke + cur]), T.uint32(1), pred=lane == 0)
        if final:
            # the last read of this Q tile
            T.ptx.mbarrier.arrive.shared.b64(
                bars.ptr_to([b_qe + qslot]), T.uint32(1), pred=lane == 0
            )
        # The softmax runs while acc += P_{j-1} V_{j-1} is in flight. It sits in a block of
        # its own, under a test ptxas cannot decide (zero is always 0): in one block with the
        # wait, ptxas schedules the wait first (it hoists the stall). P is overwritten under
        # the same test, which keeps P_{j-1} live until after the wait: otherwise ptxas puts
        # softmax values in its registers, still being read, and waits before the softmax.
        if zero == 0:
            if final:
                if tail:
                    mask(sacc, j * bn, t)
            if is_fused:
                online_softmax(sacc, st, tmp)
            else:
                fixed_softmax(sacc, st, tmp)
        T.ptx.wgmma.wait_group.sync.aligned(0)  # acc += P_{j-1} V_{j-1} is done
        T.ptx.mbarrier.arrive.shared.b64(bars.ptr_to([b_ve + prv]), T.uint32(1), pred=lane == 0)
        if zero == 0:
            to_p(sacc, pb)

    @T.function
    def attention(
        Q: T.Tensor((rows, d), "bfloat16", align=16),
        K: T.Tensor((rows, d), "bfloat16", align=16),
        V: T.Tensor((rows, d), "bfloat16", align=16),
        Out: T.Tensor((rows, d), "bfloat16", align=16),
        Mx: T.Tensor((rows,), "float32"),
        Ls: T.Tensor((rows,), "float32"),
        zero: T.int32,  # always 0; see kv_step
    ):
        # 3-d tensor maps (d, s, b h): a box past s is zero-filled on load, clipped on store
        q_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", q_map, "bfloat16", 3, Q.data,
            d, s, bh, d * 2, s * d * 2, 64, 64, 1, 1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        k_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", k_map, "bfloat16", 3, K.data,
            d, s, bh, d * 2, s * d * 2, 64, bn, 1, 1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        v_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", v_map, "bfloat16", 3, V.data,
            d, s, bh, d * 2, s * d * 2, 64, bn, 1, 1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        o_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", o_map, "bfloat16", 3, Out.data,
            d, s, bh, d * 2, s * d * 2, 64, 64, 1, 1, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(grid,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        wg = tx // 128

        pool = SMEMPool()
        sQ = pool.alloc((2, NC, nch, 64, 64), "bfloat16", align=1024)
        sK = pool.alloc((stages, nch, bn, 64), "bfloat16", align=1024)
        sV = pool.alloc((stages if use_v else 1, nch, bn, 64), "bfloat16", align=1024)
        sO = pool.alloc((NC if use_v else 1, nch, 64, 64), "bfloat16", align=1024)
        bars = pool.alloc((nbar,), "uint64", align=8)
        pool.commit()

        if tx == 0:
            for i in T.unroll(2):
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([b_qf + i]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([b_qe + i]), T.uint32(NC * 4))
            for i in T.unroll(stages):
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([b_kf + i]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([b_vf + i]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([b_ke + i]), T.uint32(NC * 4))
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([b_ve + i]), T.uint32(NC * 4))
            T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        if wg == 0:
            # ---------------- producer: one thread issues every TMA load ----------------
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](PRODUCER_REGS)
            if tx == 0:
                it: T.int32 = 0  # K/V tiles loaded so far: the ring position
                k: T.int32 = 0  # query tiles so far: the Q buffer's phase
                tile: T.int32 = bx
                while tile < ntiles:
                    hb = tile // mblocks
                    T.cuda.mbarrier_wait(bars.ptr_to([b_qe + k % 2]), k // 2 % 2 ^ 1)
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        bars.ptr_to([b_qf + k % 2]), T.uint32(BM * d * 2)
                    )
                    for w in T.unroll(NC):
                        for c in T.unroll(nch):
                            T.ptx[TMA_LOAD](
                                sQ.ptr_to([k % 2, w, c, 0, 0]), T.address_of(q_map),
                                c * 64, tile % mblocks * BM + w * 64, hb,
                                bars.ptr_to([b_qf + k % 2]),
                            )  # fmt: skip
                    for j in range(nblocks):
                        stg = (it + j) % stages
                        ph = (it + j) // stages % 2
                        T.cuda.mbarrier_wait(bars.ptr_to([b_ke + stg]), ph ^ 1)
                        T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                            bars.ptr_to([b_kf + stg]), T.uint32(kv_bytes)
                        )
                        for c in T.unroll(nch):
                            T.ptx[TMA_LOAD](
                                sK.ptr_to([stg, c, 0, 0]), T.address_of(k_map),
                                c * 64, j * bn, hb, bars.ptr_to([b_kf + stg]),
                            )  # fmt: skip
                        if use_v:
                            T.cuda.mbarrier_wait(bars.ptr_to([b_ve + stg]), ph ^ 1)
                            T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                bars.ptr_to([b_vf + stg]), T.uint32(kv_bytes)
                            )
                            for c in T.unroll(nch):
                                T.ptx[TMA_LOAD](
                                    sV.ptr_to([stg, c, 0, 0]), T.address_of(v_map),
                                    c * 64, j * bn, hb, bars.ptr_to([b_vf + stg]),
                                )  # fmt: skip
                    it = it + nblocks
                    k = k + 1
                    tile = tile + grid
        else:
            # ---------------- consumers: 64 query rows per warpgroup ----------------
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](CONSUMER_REGS)
            w = wg - 1
            ltx = tx % 128
            wq = ltx // 32
            lane = tx % 32
            g = lane // 4
            t = lane % 4
            sacc = T.alloc_local((bn // 2,), "float32", layout="default")  # S, then p
            oacc = T.alloc_local((d // 2,), "float32", layout="default")  # acc
            pb = T.alloc_local((bn // 2,), "bfloat16", layout="default")  # P, A operand
            st = T.alloc_local((6,), "float32", layout="default")  # m[2], l[2], alpha[2]
            tmp = T.alloc_local((4,), "float32", layout="default")
            desc = T.alloc_local((4,), "uint64", layout="default")  # Q[0], K, V, Q[k % 2]
            pw = pb.view("uint32")
            T.cuda.wgmma.encode_matrix_descriptor(
                T.address_of(desc[0]), sQ.ptr_to([0, w, 0, 0, 0]), 1, 64, 3
            )
            T.cuda.wgmma.encode_matrix_descriptor(
                T.address_of(desc[1]), sK.ptr_to([0, 0, 0, 0]), 1, 64, 3
            )
            T.cuda.wgmma.encode_matrix_descriptor(
                T.address_of(desc[2]), sV.ptr_to([0, 0, 0, 0]), chunk_kv, 64, 3
            )
            if pingpong:
                if w == 1:
                    T.ptx.bar.arrive(T.uint32(1), T.uint32(256))  # warpgroup 0 issues first
            it: T.int32 = 0
            k: T.int32 = 0
            tile: T.int32 = bx
            while tile < ntiles:
                hb = tile // mblocks
                q0 = tile % mblocks * BM
                for r in T.unroll(2):
                    row = q0 + w * 64 + wq * 16 + g + r * 8
                    if reads_m:
                        st[r] = T.if_then_else(
                            row < s, Mx[hb * s + T.min(row, s - 1)], T.float32(0)
                        )
                    else:
                        st[r] = T.float32(NEG)
                    if is_out:
                        st[2 + r] = T.if_then_else(
                            row < s, T.float32(1) / Ls[hb * s + T.min(row, s - 1)], T.float32(0)
                        )
                    else:
                        st[2 + r] = T.float32(0)
                    st[4 + r] = T.float32(1)
                for i in T.unroll(d // 2):
                    oacc[i] = T.float32(0)
                desc[3] = desc[0] + T.Cast("uint64", k % 2 * q_buf)
                T.cuda.mbarrier_wait(bars.ptr_to([b_qf + k % 2]), k // 2 % 2)

                if use_v:
                    # K/V tile 0: S_0 and its softmax
                    cur = it % stages
                    T.cuda.mbarrier_wait(bars.ptr_to([b_kf + cur]), it // stages % 2)
                    if pingpong:
                        T.ptx.bar.sync(T.Cast("uint32", 1 + w), T.uint32(256))
                    gemm_s(sacc, desc[3], desc[1], cur)
                    if pingpong:
                        T.ptx.bar.arrive(T.Cast("uint32", 2 - w), T.uint32(256))
                    T.ptx.wgmma.wait_group.sync.aligned(0)
                    if lane == 0:  # one arrive per warp
                        T.ptx.mbarrier.arrive.shared.b64(bars.ptr_to([b_ke + cur]), T.uint32(1))
                        if nblocks == 1:
                            T.ptx.mbarrier.arrive.shared.b64(
                                bars.ptr_to([b_qe + k % 2]), T.uint32(1)
                            )
                    if tail and nblocks == 1:
                        mask(sacc, 0, t)
                    if is_fused:
                        online_softmax(sacc, st, tmp)
                    else:
                        fixed_softmax(sacc, st, tmp)
                    to_p(sacc, pb)
                    for j in range(1, nblocks - 1):
                        kv_step(
                            sacc,
                            oacc,
                            pb,
                            st,
                            tmp,
                            desc,
                            bars,
                            it + j,
                            w,
                            lane,
                            t,
                            j,
                            False,
                            zero,
                            k % 2,
                        )
                    if nblocks > 1:  # the last step masks the keys past s and frees Q
                        kv_step(sacc, oacc, pb, st, tmp, desc, bars, it + nblocks - 1, w, lane, t,
                                nblocks - 1, True, zero, k % 2)  # fmt: skip
                    last = (it + nblocks - 1) % stages
                    if is_fused:
                        rescale(oacc, st)
                    T.cuda.mbarrier_wait(
                        bars.ptr_to([b_vf + last]), (it + nblocks - 1) // stages % 2
                    )
                    gemm_o(oacc, pw, desc[2], last)
                    T.ptx.wgmma.wait_group.sync.aligned(0)
                    if lane == 0:  # one arrive per warp
                        T.ptx.mbarrier.arrive.shared.b64(bars.ptr_to([b_ve + last]), T.uint32(1))

                    # epilogue: out = acc / l -> shared memory (swizzled) -> TMA store
                    for r in T.unroll(2):
                        if is_fused:
                            tmp[r] = T.float32(1) / quad_sum(st[2 + r])
                        else:
                            tmp[r] = T.float32(1)
                    if ltx == 0:
                        T.ptx.cp.async_.bulk.wait_group.read(0)  # the last store has read sO
                    T.ptx.bar.sync(T.Cast("uint32", 3 + w), T.uint32(128))
                    for jt in T.unroll(d // 8):
                        for i in T.unroll(4):
                            row = wq * 16 + g + i // 2 * 8  # 128-byte swizzle: chunk ^ row % 8
                            sO[w, jt // 8, row, (jt % 8 ^ g) * 8 + 2 * t + i % 2] = T.Cast(
                                "bfloat16", oacc[jt * 4 + i] * tmp[i // 2]
                            )
                    T.ptx.fence.proxy.async_.shared__cta()
                    T.ptx.bar.sync(T.Cast("uint32", 3 + w), T.uint32(128))
                    if ltx == 0:
                        for c in T.unroll(nch):
                            T.ptx[TMA_STORE](
                                T.address_of(o_map), c * 64, q0 + w * 64, hb,
                                sO.ptr_to([w, c, 0, 0]),
                            )  # fmt: skip
                        T.ptx.cp.async_.bulk.commit_group()
                else:
                    # passes 1 and 2 of the ablation: a row reduction of S, nothing else
                    for j in range(nblocks):
                        cur = (it + j) % stages
                        T.cuda.mbarrier_wait(bars.ptr_to([b_kf + cur]), (it + j) // stages % 2)
                        if pingpong:
                            T.ptx.bar.sync(T.Cast("uint32", 1 + w), T.uint32(256))
                        gemm_s(sacc, desc[3], desc[1], cur)
                        if pingpong:
                            T.ptx.bar.arrive(T.Cast("uint32", 2 - w), T.uint32(256))
                        T.ptx.wgmma.wait_group.sync.aligned(0)
                        if lane == 0:  # one arrive per warp
                            T.ptx.mbarrier.arrive.shared.b64(bars.ptr_to([b_ke + cur]), T.uint32(1))
                            if j == nblocks - 1:
                                T.ptx.mbarrier.arrive.shared.b64(
                                    bars.ptr_to([b_qe + k % 2]), T.uint32(1)
                                )
                        if tail:
                            if j == nblocks - 1:
                                mask(sacc, j * bn, t)
                        for r in T.unroll(2):
                            if is_max:
                                for jt in T.unroll(bn // 8):
                                    for e in T.unroll(2):
                                        st[r] = fmax(st[r], sacc[jt * 4 + r * 2 + e])
                            else:
                                tmp[1] = st[r] * T.float32(scale)
                                for jt in T.unroll(bn // 8):
                                    for e in T.unroll(2):
                                        st[2 + r] = st[2 + r] + T.exp2(
                                            sacc[jt * 4 + r * 2 + e] * T.float32(scale) - tmp[1]
                                        )
                    for r in T.unroll(2):
                        row = q0 + w * 64 + wq * 16 + g + r * 8
                        if is_max:
                            tmp[r] = quad_max(st[r])
                        else:
                            tmp[r] = quad_sum(st[2 + r])
                        if t == 0:
                            if row < s:
                                if is_max:
                                    Mx[hb * s + row] = tmp[r]
                                else:
                                    Ls[hb * s + row] = tmp[r]
                it = it + nblocks
                k = k + 1
                tile = tile + grid
            if use_v:
                if ltx == 0:
                    T.ptx.cp.async_.bulk.wait_group(0)

    return attention


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    b, h, s, d = cfg["b"], cfg["h"], cfg["s"], cfg["d"]
    name = f"attention_{b}x{h}x{s}x{d}"
    rows = b * h * s
    if variant == "fused":
        mod = compile_sm90(_kernel(b, h, s, d, "fused"), name=name)
        dummy = torch.empty(rows, device="cuda", dtype=F32)

        def run(Q, K, V):
            out = torch.empty((b, h, s, d), device=Q.device, dtype=BF16)
            mod(Q.view(-1, d), K.view(-1, d), V.view(-1, d), out.view(-1, d), dummy, dummy, 0)
            return (out,)

        return run

    passes = [
        compile_sm90(_kernel(b, h, s, d, mode), name=f"{name}_{mode}")
        for mode in ("max", "sum", "out")
    ]
    mx = torch.empty(rows, device="cuda", dtype=F32)  # row maxima, pass 1 -> 2, 3
    ls = torch.empty(rows, device="cuda", dtype=F32)  # row sums, pass 2 -> 3

    def run_unfused(Q, K, V):
        out = torch.empty((b, h, s, d), device=Q.device, dtype=BF16)
        args = (Q.view(-1, d), K.view(-1, d), V.view(-1, d), out.view(-1, d), mx, ls, 0)
        for mod in passes:
            mod(*args)
        return (out,)

    return run_unfused
