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
"""Cross entropy of a linear classifier against hard labels, in TIRx, on Hopper.

``loss[i] = logsumexp_j(X[i] · W[j]) - X[i] · W[label[i]]`` for bfloat16
``X (n, d)``, ``W (v, d)`` and int64 labels; the logits ``X Wᵀ`` never reach
HBM (the Cut-Cross-Entropy layout).

The algorithm is chain_ops' ``cross_entropy_fused``: one pass over the
vocabulary with the state ``(m, l, t)`` per row -- the running maximum of the
logits, the sum of ``exp(logit - m)``, and the logit of the label (``Σ_j y_j
logit_j`` for a one-hot ``y``). Two states merge by keeping the larger
maximum, rescaling both sums to it, and adding the ``t``; the identity is
``(-3e38, 0, 0)``. The epilogue is ``loss = (m - t) + log l``.

How it is laid out: a Hopper GEMM with the online log-sum-exp in its epilogue.

``main`` -- the logits, ``BM x BN = 128 x 256`` at a time, on the tensor cores
    * a persistent grid of one CTA per SM and three warpgroups. Warpgroup 0 is
      the producer: one thread issues the TMA loads (``cp.async.bulk.tensor``,
      128-byte swizzle; rows past ``n`` or ``v`` and columns past ``d`` are
      zero-filled) of a ``128 x 64`` slice of X and a ``256 x 64`` slice of W
      into a ``STAGES``-deep ring of full/empty mbarriers that runs on across
      tiles; ``setmaxnreg`` hands its registers to the consumers.
    * warpgroups 1 and 2 are the consumers, 64 rows each: per slice, four
      ``wgmma.m64n256k16`` (both operands from shared memory, float32
      accumulators in registers, 128 per thread); one group stays in flight
      while the next slice is awaited.
    * after a tile's last slice, a consumer folds its ``64 x 256`` logits into
      the state of its rows (``fold``): the tile maximum (four lanes share a
      row: two shuffles), one rescale of the running sum, and one ``exp2`` per
      logit of ``(z - m) * log2(e)``. ``z - m`` is exact near the maximum, so
      the maximum itself adds exactly 1 (an FMA ``z * log2(e) - m * log2(e)``
      would add ``2^-δ``, ``δ`` the rounding of ``m * log2(e)``: an error of
      up to one ulp of ``m`` in the loss). The running sum is a float32 pair
      (TwoSum) that takes each tile's two 32-term chain sums per lane, and the
      lanes of a row merge as pairs too, so on rows where one logit dominates
      the tiny rest is not absorbed by the maximum's 1; only the other terms
      of the maximum's own chain are (about ``31 / v`` of the rest). The
      label's logit is picked out only in the tile that holds it (one unsigned
      range test per row and tile); columns ``>= v`` are masked, in the last
      vocabulary tile only.
    * the schedule (``plan``): groups of ``GROUP`` = 12 row blocks; each group
      is cut into ``132 // rows`` vocabulary splits, so every CTA takes one
      (row block, split) per group and walks its split's tiles in order,
      carrying the state in registers. The 12 CTAs of a split read the same W
      tile at the same time and the 11 CTAs of a row block the same X slices,
      so both come from L2 (about 34 MB of HBM per step of 132 tiles). The
      units are dealt longest first to the least loaded CTA: every CTA ends
      within one tile of the mean. After a unit the CTA writes one state per
      row, ``part (n, P, 4) = (m, l, t, c)``, slot = its split (``c`` is the
      sum's compensation).
    * the CTAs of a split are kept in step (``THROTTLE_STEPS``): their
      producers count their progress through the unit, in eighths of a tile,
      on the split's counter in global memory, and none starts step ``s``
      before every sharer has started step ``s - 1``. Left free, they drift
      apart, by up to ten tiles over the 91-tile units at v = 256000; a W
      tile then leaves L2 before its last reader comes, and a tile took 10%
      longer at the end of a call than at its start (traced with
      ``%globaltimer``). In step, every tile runs at the speed of the first:
      a call is 5% faster at v = 256000 and 2% at v = 128256 as bench.run
      times it (8% and 3% for one call on a rested card, before the power
      cap lowers the clock). A wait gives up after ``SPINS`` polls and turns
      the bound off for the rest of the launch, so the kernel never depends
      on its CTAs being co-resident; the last kernel of a call zeroes the
      counters.
``merge`` -- one thread per row
    merges the row's ``P`` states: each slot's sum is scaled to the row's
    maximum by ``exp2`` in float32 (exactly 1 for the slot that holds it) and
    added to a float32 TwoSum pair. Then ``(m - t) + log l`` in float64,
    rounded once (``m - t`` is exact, and 0 when the label is the maximum);
    the H800 runs float64 at 1/64 the rate of float32, so it is spent once per
    row, 32 rows to a warp instruction. Slots a row's group does not use hold
    the identity.

The workspace (``part`` and the counters) is cached per CUDA stream: the
kernels run on torch's current stream (tvm-ffi launches on it), so calls on
one stream run in order, and calls on two streams never share one. A call
made while a CUDA graph is captured gets a workspace of its own.

``variant="unfused"`` is the ablation: the same GEMM kernel once per reduction
of the unfused program, the logits recomputed each time. Pass 1 (``max``)
keeps the row maximum; pass 2 (``sum``) takes the row's maximum from pass 1's
states and keeps ``Σ exp(logit - max)``; pass 3 (``label``) recomputes the
logit of the label, ``X[i] · W[label i]``, with the same tensor-core
instruction in the same k order, so it is bitwise the logit passes 1 and 2
saw: the same kernel with W's rows gathered by label (``cp.async`` by the
producer warpgroup, into the swizzled layout a TMA tile has) and the diagonal
of the ``128 x 128`` product kept. Only per row results (one per row and
split) travel between the passes; a last kernel writes ``(max - t) + log sum``.
"""

from __future__ import annotations

import math

import torch
from bench.ops import F32

from tvm_tirx.backend.cuda.lang import SMEMPool
from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.reduce import FULL_MASK, fmax

NEG = -3.0e38  # a finite "-inf": the identity of max, and exp(NEG - NEG) is no NaN
LOG2E = math.log2(math.e)

# --- the GEMM kernel ---
BM, BN, BK = 128, 256, 64  # CTA tile: rows x vocabulary x hidden slice
STAGES = 4  # TMA ring depth: 4 x 48 KB
NC = 2  # consumer warpgroups, 64 rows each
THREADS = 128 * (NC + 1)
ACC = BN // 2  # float32 accumulators per consumer thread (m64n256)
NJ = BN // 8  # n8 column blocks of a wgmma accumulator
GROUP = 12  # row blocks per schedule group (12 x 11 = 132 CTAs)
PRODUCER_REGS = 40
CONSUMER_REGS = 232
VEC = 8  # bf16 per 16-byte chunk

MERGE_THREADS = 128  # one row per thread
THROTTLE_STEPS = 8  # the CTAs sharing a W tile stay within 1/8 of a tile of each other
SPINS = 20000  # polls (about 1 us each) before a throttle wait gives up

TMA_LOAD = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
WGMMA = f"wgmma.mma_async.sync.aligned.m64n{BN}k16.f32.bf16.bf16"

# 16 bytes global -> shared, asynchronously; the bytes past `bytes` are zero-filled
CP16 = r"""
__forceinline__ __device__ void ce_cp16(void* dst, const void* src, int bytes) {
  unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(src), "r"(bytes)
               : "memory");
}
"""

# 1 once *p >= target, 0 if it is still short after `spins` polls
WAIT_GE = r"""
__forceinline__ __device__ int ce_wait_ge(int* p, int target, int spins) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  while (v < target) {
    if (--spins < 0) return 0;
    __nanosleep(64);
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  }
  return 1;
}
"""

KERNEL_META = {
    "op": "cross_entropy",
    "dtype": "bfloat16 in, int64 labels, float32 out",
    "algorithm": (
        "one pass over the vocabulary, state (m, l, t) = "
        "(max logit, Σ exp(logit - max), logit of the label)"
    ),
    "structure": (
        "Hopper GEMM + online LSE epilogue: persistent CTAs (one per SM), a TMA producer "
        "warpgroup and 2 wgmma consumer warpgroups (m64n256k16, 128x256 logit tiles in "
        "registers, 4-stage mbarrier ring); grouped schedule (12 row blocks x 11 vocabulary "
        "splits) so W and X tiles are shared in L2, the 12 CTAs of a split kept within 1/8 "
        "tile of each other by progress counters; one state per row and split, merged by a "
        "second one-thread-per-row kernel (float32 TwoSum, float64 last step); workspaces "
        "cached per CUDA stream"
    ),
}


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


@T.inline
def two_sum(st, s, c, x, tmp, k):
    """``(st[s], st[c]) += x``: ``st[s]`` takes the rounded sum and ``st[c]`` gathers its
    exact rounding error (TwoSum), so terms far below ``st[s]`` are not lost. ``x`` must not
    read ``st[s]``; ``tmp[k]``, ``tmp[k + 1]`` are scratch."""
    tmp[k] = st[s] + x
    tmp[k + 1] = tmp[k] - st[s]
    st[c] = st[c] + ((st[s] - (tmp[k] - tmp[k + 1])) + (x - tmp[k + 1]))
    st[s] = tmp[k]


@T.inline
def quad_sum2(st, s, c, tmp):
    """``(st[s], st[c])``, a sum and its compensation, summed over the 4 lanes of a quad."""
    for i in T.unroll(2):
        tmp[2] = T.gpu_warp_shuffle_xor(FULL_MASK, st[s], 1 << i, 32, 32)
        tmp[3] = T.gpu_warp_shuffle_xor(FULL_MASK, st[c], 1 << i, 32, 32)
        two_sum(st, s, c, tmp[2], tmp, 4)
        st[c] = st[c] + tmp[3]


def plan(n: int, d: int, v: int, ctas: int = SMS, group: int = GROUP) -> dict:
    """The persistent schedule.

    Row blocks are taken ``group`` at a time; a group of ``rg`` row blocks is cut
    into ``ctas // rg`` contiguous vocabulary splits, which makes ``rg x splits``
    units (row block, first tile, end tile, split, counter, sharers). The units
    of each group go longest first to the least loaded CTA, so ``rounds[c]`` is
    CTA ``c``'s list, in group order, and every CTA ends within one tile of the
    mean. ``P`` is the most splits of any group: the slots of the workspace.
    Each split of each group has its progress counter (``nctr`` in all), shared
    by its ``rg`` units.
    """
    rb, nt, kt = math.ceil(n / BM), math.ceil(v / BN), math.ceil(d / BK)
    load = [0] * ctas
    rounds: list[list[tuple[int, int, int, int, int, int]]] = [[] for _ in range(ctas)]
    P, nctr = 1, 0
    for r0 in range(0, rb, group):
        rg = min(group, rb - r0)
        splits = max(1, min(nt, ctas // rg))
        P = max(P, splits)
        bounds = [s * nt // splits for s in range(splits + 1)]
        units = [
            (r0 + r, bounds[s], bounds[s + 1], s, nctr + s, rg)
            for s in range(splits)
            for r in range(rg)
        ]
        nctr += splits
        units.sort(key=lambda u: u[1] - u[2])  # longest first (stable)
        for u, c in zip(units, sorted(range(ctas), key=lambda c: load[c])):
            rounds[c].append(u)
            load[c] += u[2] - u[1]
    nr = max(len(r) for r in rounds)
    return {
        "rb": rb, "nt": nt, "kt": kt, "P": P, "nr": nr, "nctr": nctr,
        "rounds": rounds, "tiles": max(load),
    }  # fmt: skip


def label_plan(p: dict, ctas: int = SMS) -> dict:
    """The schedule of the ablation's label pass: one unit per row block, dealt round robin.

    A unit is one pseudo-tile (``0, 1``) with no sharers; the workspace shape (``P``,
    ``nctr``) stays that of ``p``, whose ``part`` the pass writes into.
    """
    rounds: list[list[tuple[int, int, int, int, int, int]]] = [[] for _ in range(ctas)]
    for r in range(p["rb"]):
        rounds[r % ctas].append((r, 0, 1, 0, 0, 1))
    nr = max(len(r) for r in rounds)
    return {**p, "rounds": rounds, "nr": nr, "tiles": nr}


def _main(n: int, d: int, v: int, p: dict, mode: str):
    """The GEMM kernel; ``mode``: "fused", or a pass of the ablation, "max", "sum" or "label"."""
    assert mode in ("fused", "max", "sum", "label"), mode
    nt, kt, nr, P, nctr = p["nt"], p["kt"], p["nr"], p["P"], p["nctr"]
    step_k = max(1, kt // THROTTLE_STEPS)  # slices per progress step
    steps = math.ceil(kt / step_k)  # progress steps per tile
    is_fused, is_max, is_sum = mode == "fused", mode == "max", mode == "sum"
    is_label = mode == "label"
    # W comes in BN-row tiles by TMA. The label pass instead gathers, per slice, the W row of
    # each of the block's BM labels into the stage's first BM rows: each of the 128 producer
    # threads copies one row by cp.async (a TMA per row ran 5x slower, bound by TMA issue) and
    # arrives on the slice's full barrier once its copies have landed; TMA brings X only.
    stage_bytes = BM * BK * 2 if is_label else (BM + BN) * BK * 2  # bytes the TMA brings
    full_count = 1 + 128 if is_label else 1  # arrivals per phase of a full barrier
    lag = min(STAGES - 2, kt)  # label pass: a slice is handed over while lag younger ones load
    x_units = BM * BK * 2 // 16  # descriptor units (16 bytes) per stage of X
    w_units = BN * BK * 2 // 16  # ... of W
    tail = v % BN != 0

    @T.inline
    def mask(acc, r, col0, t4):
        """Columns >= v score NEG; called on the last vocabulary tile only."""
        for j in T.unroll(NJ):
            for e in T.unroll(2):
                i = j * 4 + r * 2 + e
                acc[i] = T.if_then_else(col0 + j * 8 + t4 * 2 + e < v, acc[i], T.float32(NEG))

    @T.inline
    def row_max(acc, r, tmp):
        """tmp[0] = this lane's maximum of row half r (two independent chains)."""
        tmp[0] = acc[r * 2]
        tmp[1] = acc[r * 2 + 1]
        for j in T.unroll(NJ - 1):
            tmp[0] = fmax(tmp[0], acc[(j + 1) * 4 + r * 2])
            tmp[1] = fmax(tmp[1], acc[(j + 1) * 4 + r * 2 + 1])
        tmp[0] = fmax(tmp[0], tmp[1])

    @T.inline
    def exp_sum(acc, r, m, tmp):
        """Σ exp2((acc - m) * log2(e)) over this lane's half-row r, in two chains ``e``, each
        kept exactly as ``(tmp[2 + e] - 1) + tmp[4 + e]``; ``acc - m`` is exact near the
        maximum, which counts exactly 1.

        A plain float32 chain rounds every term to the ulp of its running sum: once the
        row's maximum (1) is in it, each later term loses up to 6e-8 and terms under that
        vanish, which is the whole loss of a peaked row. So ``tmp[2 + e]`` starts at 1 and
        takes the terms; no term exceeds 1, so the rounding error of each addition is
        exactly ``term - (new - old)`` (Fast2Sum), which ``tmp[4 + e]`` gathers. The
        offset sum stays below 64, so subtracting the 1 is exact."""
        for e in T.unroll(2):
            tmp[2 + e] = T.float32(1)
            tmp[4 + e] = T.float32(0)
        for j in T.unroll(NJ):
            for e in T.unroll(2):
                tmp[6 + e] = T.exp2((acc[j * 4 + r * 2 + e] - m) * T.float32(LOG2E))
                tmp[8 + e] = tmp[2 + e] + tmp[6 + e]
                tmp[4 + e] = tmp[4 + e] + (tmp[6 + e] - (tmp[8 + e] - tmp[2 + e]))
                tmp[2 + e] = tmp[8 + e]

    @T.inline
    def add_sum(st, r, tmp):
        """Add exp_sum's chains to the row half's sum ``(st[2 + r], st[6 + r])``: their exact
        parts and their gathered errors by TwoSum, then renormalise the pair, so that
        ``st[6 + r]`` stays within an ulp of ``st[2 + r]`` and never gathers the tiny sums of
        many tiles in plain float32 (which would lose them as a chain does)."""
        for e in T.unroll(2):
            two_sum(st, 2 + r, 6 + r, tmp[2 + e] - T.float32(1), tmp, 8)
        two_sum(st, 2 + r, 6 + r, tmp[4] + tmp[5], tmp, 8)
        tmp[6] = st[6 + r]
        st[6 + r] = T.float32(0)
        two_sum(st, 2 + r, 6 + r, tmp[6], tmp, 8)

    @T.inline
    def fold(acc, st, tmp, lab, col0, t4, masked):
        """Fold this warpgroup's 64 x BN logits into the state of the lane's two rows.

        ``st``: m[2], l[2], t[2], c[2] (``c`` the compensation of ``l``)."""
        for r in T.unroll(2):
            if masked:
                mask(acc, r, col0, t4)
            row_max(acc, r, tmp)
            if is_max:
                st[r] = fmax(st[r], tmp[0])
            elif is_fused:
                tmp[0] = fmax(st[r], quad_max(tmp[0]))  # the new maximum of the row
                tmp[1] = T.exp2((st[r] - tmp[0]) * T.float32(LOG2E))
                st[2 + r] = st[2 + r] * tmp[1]
                st[6 + r] = st[6 + r] * tmp[1]
                st[r] = tmp[0]
                exp_sum(acc, r, tmp[0], tmp)
                add_sum(st, r, tmp)
                rel = lab[r] - col0
                if T.Cast("uint32", rel) < T.uint32(BN):  # the label is in this tile
                    for j in T.unroll(NJ):
                        for e in T.unroll(2):
                            st[4 + r] = T.if_then_else(
                                rel == j * 8 + t4 * 2 + e, acc[j * 4 + r * 2 + e], st[4 + r]
                            )
            else:
                exp_sum(acc, r, st[r], tmp)
                add_sum(st, r, tmp)

    @T.inline
    def write_states(Part, st, tmp, row0, slot, t4):
        """One state per row and unit: merge the 4 lanes of the quad, then write ``slot``."""
        for r in T.unroll(2):
            row = row0 + r * 8
            if is_max:
                tmp[0] = quad_max(st[r])
            else:
                tmp[0] = quad_sum(st[4 + r])
                quad_sum2(st, 2 + r, 6 + r, tmp)
            if t4 == 0:
                if row < n:
                    if is_max:
                        Part[row, slot, 0] = tmp[0]
                    elif is_fused:
                        Part[row, slot, 0] = st[r]
                        Part[row, slot, 1] = st[2 + r]
                        Part[row, slot, 2] = tmp[0]
                        Part[row, slot, 3] = st[6 + r]
                    else:
                        Part[row, slot, 1] = st[2 + r]
                        Part[row, slot, 3] = st[6 + r]

    @T.inline
    def write_label(Part, acc, tmp, row0, jw0, g, t4):
        """The label pass: the X row ``i`` of the block met the gathered W row ``i``, so its
        logit is the diagonal of the product, column ``i`` (``jw0 = w * 4 + wq`` is the
        16-row slice of the 128 this warp holds); written to slot 0's ``t``."""
        for r in T.unroll(2):
            row = row0 + r * 8
            tmp[0] = T.float32(0)
            for jw in T.unroll(NC * 4):
                for e in T.unroll(2):
                    tmp[0] = T.if_then_else(
                        jw0 * 2 + g % 2 == jw * 2 + e, acc[(jw * 2 + r) * 4 + r * 2 + e], tmp[0]
                    )
            if t4 == g // 2:
                if row < n:
                    Part[row, 0, 2] = tmp[0]

    @T.function
    def ce_main(
        X: T.Tensor((n, d), "bfloat16", align=16),
        W: T.Tensor((v, d), "bfloat16", align=16),
        L: T.Tensor((n,), "int64"),
        Sched: T.Tensor((SMS, nr, 6), "int32"),
        Part: T.Tensor((n, P, 4), "float32", align=16),
        Prog: T.Tensor((nctr,), "int32"),  # progress of each split; zero at launch
    ):
        # 2-d tensor maps (d, rows): a box past the rows or past d is zero-filled
        x_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", x_map, "bfloat16", 2, X.data,
            d, n, d * 2, BK, BM, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        w_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", w_map, "bfloat16", 2, W.data,
            d, v, d * 2, BK, BN, 1, 1, 0, 3, 2, 0,
        )  # fmt: skip
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(SMS,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        wg = tx // 128

        pool = SMEMPool()
        sX = pool.alloc((STAGES, BM, BK), "bfloat16", align=1024)
        sW = pool.alloc((STAGES, BN, BK), "bfloat16", align=1024)
        bars = pool.alloc((2 * STAGES,), "uint64", align=8)  # full[STAGES], empty[STAGES]
        pool.commit()

        if tx == 0:
            for i in T.unroll(STAGES):
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([i]), T.uint32(full_count))
                T.ptx.mbarrier.init.shared.b64(bars.ptr_to([STAGES + i]), T.uint32(NC * 4))
            T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        if wg == 0:
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](PRODUCER_REGS)
            if is_label:
                # ------ producer: X by TMA (thread 0), W's label rows by cp.async (all) ------
                # Thread tx copies row tx of the gathered W: per slice, 8 chunks of 16 bytes,
                # 128-byte swizzled (chunk c at c ^ tx % 8) as the TMA lays out a tile. When a
                # slice's copies have landed (lag slices later) the thread fences them for the
                # tensor cores, which read through the async proxy, and arrives.
                it: T.int32 = 0  # slices loaded so far: the ring position
                for rd in range(nr):
                    rbk = Sched[bx, rd, 0]
                    for vt in range(Sched[bx, rd, 1], Sched[bx, rd, 2]):  # one, or none
                        wrow = T.Cast("int32", L[T.min(rbk * BM + tx, n - 1)])  # its W row
                        for kb in range(kt):
                            stg = it % STAGES
                            T.cuda.mbarrier_wait(
                                bars.ptr_to([STAGES + stg]), (it // STAGES) % 2 ^ 1
                            )
                            if tx == 0:
                                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                    bars.ptr_to([stg]), T.uint32(stage_bytes)
                                )
                                T.ptx[TMA_LOAD](
                                    sX.ptr_to([stg, 0, 0]), T.address_of(x_map), kb * BK,
                                    rbk * BM, bars.ptr_to([stg]),
                                )  # fmt: skip
                            for c in T.unroll(BK // VEC):
                                col = kb * BK + c * VEC  # chunks past d: zero-filled
                                T.cuda.func_call(
                                    "ce_cp16", sW.ptr_to([stg, tx, (c ^ tx % 8) * VEC]),
                                    T.address_of(W[wrow, T.min(col, d - VEC)]),
                                    T.if_then_else(col < d, 16, 0), CP16,
                                )  # fmt: skip
                            T.ptx.cp.async_.commit_group()
                            if kb >= lag:  # slice it - lag has landed: hand it over
                                T.ptx.cp.async_.wait_group(lag)
                                T.ptx.fence.proxy.async_.shared__cta()
                                T.ptx.mbarrier.arrive.shared.b64(
                                    bars.ptr_to([(it - lag) % STAGES]), T.uint32(1)
                                )
                            it = it + 1
                        T.ptx.cp.async_.wait_group(0)  # and the unit's last lag slices
                        T.ptx.fence.proxy.async_.shared__cta()
                        for j in T.unroll(lag):
                            T.ptx.mbarrier.arrive.shared.b64(
                                bars.ptr_to([(it - lag + j) % STAGES]), T.uint32(1)
                            )
            else:
                # ---------------- producer: one thread issues every TMA load ----------------
                if tx == 0:
                    it: T.int32 = 0  # slices loaded so far: the ring position
                    on: T.int32 = 1  # the drift bound holds until a wait gives up
                    for rd in range(nr):
                        rbk = Sched[bx, rd, 0]
                        vt0 = Sched[bx, rd, 1]
                        ctr = Sched[bx, rd, 4]
                        for vt in range(vt0, Sched[bx, rd, 2]):
                            for kb in range(kt):
                                if kb % step_k == 0:
                                    # progress step `step` of the unit: wait until every
                                    # sharer has started step - 1, then count this one
                                    step = (vt - vt0) * steps + kb // step_k
                                    if on == 1:
                                        if step > 0:
                                            on = T.cuda.func_call(
                                                "ce_wait_ge", T.address_of(Prog[ctr]),
                                                Sched[bx, rd, 5] * step, SPINS, WAIT_GE,
                                                ty="int32",
                                            )  # fmt: skip
                                    T.cuda.atomic_add(T.address_of(Prog[ctr]), T.int32(1))
                                stg = it % STAGES
                                T.cuda.mbarrier_wait(
                                    bars.ptr_to([STAGES + stg]), (it // STAGES) % 2 ^ 1
                                )
                                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                    bars.ptr_to([stg]), T.uint32(stage_bytes)
                                )
                                T.ptx[TMA_LOAD](
                                    sX.ptr_to([stg, 0, 0]), T.address_of(x_map), kb * BK,
                                    rbk * BM, bars.ptr_to([stg]),
                                )  # fmt: skip
                                T.ptx[TMA_LOAD](
                                    sW.ptr_to([stg, 0, 0]), T.address_of(w_map), kb * BK,
                                    vt * BN, bars.ptr_to([stg]),
                                )  # fmt: skip
                                it = it + 1
        else:
            # ---------------- consumers: 64 rows per warpgroup ----------------
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](CONSUMER_REGS)
            w = wg - 1
            wq = tx % 128 // 32
            lane = tx % 32
            g = lane // 4
            t4 = lane % 4
            acc = T.alloc_local((ACC,), "float32", layout="default")
            st = T.alloc_local((8,), "float32", layout="default")  # m[2], l[2], t[2], c[2]
            tmp = T.alloc_local((10,), "float32", layout="default")
            lab = T.alloc_local((2,), "int32", layout="default")
            desc = T.alloc_local((2,), "uint64", layout="default")  # X rows of w, W
            T.cuda.wgmma.encode_matrix_descriptor(
                T.address_of(desc[0]), sX.ptr_to([0, w * 64, 0]), 1, 64, 3
            )
            T.cuda.wgmma.encode_matrix_descriptor(
                T.address_of(desc[1]), sW.ptr_to([0, 0, 0]), 1, 64, 3
            )
            it: T.int32 = 0  # slices consumed so far: the ring position
            for rd in range(nr):
                rbk = Sched[bx, rd, 0]
                vt0 = Sched[bx, rd, 1]
                vt1 = Sched[bx, rd, 2]
                slot = Sched[bx, rd, 3]
                row0 = rbk * BM + w * 64 + wq * 16 + g  # the lane's rows: row0 and row0 + 8
                if vt0 < vt1:
                    for r in T.unroll(2):
                        row = row0 + r * 8
                        lab[r] = T.if_then_else(row < n, T.Cast("int32", L[T.min(row, n - 1)]), -1)
                        st[r] = T.float32(NEG)
                        if is_sum:  # the row maximum, from pass 1's states
                            for s in range(P):
                                st[r] = fmax(st[r], Part[T.min(row, n - 1), s, 0])
                        st[2 + r] = T.float32(0)
                        st[4 + r] = T.float32(0)
                        st[6 + r] = T.float32(0)
                    for vt in range(vt0, vt1):
                        for kb in range(kt):
                            stg = it % STAGES
                            T.cuda.mbarrier_wait(bars.ptr_to([stg]), (it // STAGES) % 2)
                            T.ptx.wgmma.fence.sync.aligned()
                            for kk in T.unroll(BK // 16):
                                T.ptx[WGMMA](
                                    *[acc[i] for i in range(ACC)],
                                    desc[0] + T.Cast("uint64", stg * x_units + kk * 2),
                                    desc[1] + T.Cast("uint64", stg * w_units + kk * 2),
                                    kb * (BK // 16) + kk > 0,  # the first k-step overwrites
                                    1,
                                    1,
                                    0,
                                    0,
                                )
                            T.ptx.wgmma.commit_group.sync.aligned()
                            T.ptx.wgmma.wait_group.sync.aligned(1)  # slice it - 1 is read
                            if kb > 0:
                                T.ptx.mbarrier.arrive.shared.b64(
                                    bars.ptr_to([STAGES + (it + STAGES - 1) % STAGES]),
                                    T.uint32(1),
                                    pred=lane == 0,
                                )
                            it = it + 1
                        T.ptx.wgmma.wait_group.sync.aligned(0)
                        T.ptx.mbarrier.arrive.shared.b64(
                            bars.ptr_to([STAGES + (it + STAGES - 1) % STAGES]),
                            T.uint32(1),
                            pred=lane == 0,
                        )
                        if is_label:
                            write_label(Part, acc, tmp, row0, w * 4 + wq, g, t4)
                        elif tail:
                            if vt == nt - 1:
                                fold(acc, st, tmp, lab, vt * BN, t4, True)
                            else:
                                fold(acc, st, tmp, lab, vt * BN, t4, False)
                        else:
                            fold(acc, st, tmp, lab, vt * BN, t4, False)
                    if not is_label:
                        write_states(Part, st, tmp, row0, slot, t4)

    return ce_main


@T.inline
def zero_counters(Prog, nctr, bx, tx):
    """Block 0 of the call's last kernel resets the progress counters for the next call."""
    if bx == 0:
        i: T.int32 = tx
        while i < nctr:
            Prog[i] = 0
            i += MERGE_THREADS


@T.inline
def write_loss(Loss, row, m, t, lc):
    """``loss = (m - t) + log(l + c)`` in float64, rounded once to float32; ``m - t`` is exact,
    and 0 when the label's logit is the maximum. ``lc``: the sum ``l`` and its compensation."""
    Loss[row] = T.Cast(
        "float32",
        (T.Cast("float64", m) - T.Cast("float64", t))
        + T.log(T.Cast("float64", lc[0]) + T.Cast("float64", lc[1])),
    )


def _merge(n: int, P: int, nctr: int):
    """The fused kernel's epilogue: one thread per row merges the row's ``P`` slot states.

    Each slot's sum is scaled to the row's maximum by ``exp2`` in float32, exactly 1 for the
    slot that holds it, and added to a float32 TwoSum pair. Only ``write_loss`` runs in
    float64, which the H800 runs at 1/64 the rate of float32: with a thread per row that is
    one warp instruction per 32 rows (a warp per row, all in float64, took 47 us at
    n = 8192 against 5.5 us).
    """
    blocks = math.ceil(n / MERGE_THREADS)

    @T.function
    def ce_merge(
        Part: T.Tensor((n, P, 4), "float32", align=16),
        Loss: T.Tensor((n,), "float32"),
        Prog: T.Tensor((nctr,), "int32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(MERGE_THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        ot = T.alloc_local((4,), "float32")  # a slot's (m, l, t, c)
        mx = T.alloc_local((1,), "float32")
        lc = T.alloc_local((3,), "float32")  # l, its compensation, t
        tmp = T.alloc_local((3,), "float32")  # the slot's scale; TwoSum scratch
        zero_counters(Prog, nctr, bx, tx)
        row = bx * MERGE_THREADS + tx
        if row < n:
            mx[0] = T.float32(NEG)
            for s in range(P):
                mx[0] = fmax(mx[0], Part[row, s, 0])
            lc[0] = T.float32(0)
            lc[1] = T.float32(0)
            lc[2] = T.float32(0)
            for s in range(P):
                ot[T.Ramp(0, 1, 4)] = Part[row, s, T.Ramp(0, 1, 4)]
                tmp[0] = T.exp2((ot[0] - mx[0]) * T.float32(LOG2E))  # 0 for an unused slot
                ot[1] = ot[1] * tmp[0]
                two_sum(lc, 0, 1, ot[1], tmp, 1)
                lc[1] = lc[1] + ot[3] * tmp[0]
                lc[2] = lc[2] + ot[2]  # one slot holds t, the others 0
            write_loss(Loss, row, mx[0], lc[2], lc)

    return ce_merge


def _merge_unfused(n: int, P: int, nctr: int):
    """The ablation's epilogue, one thread per row: pass 1's maxima, pass 2's sums (all
    relative to the row maximum already, so they add as they are) and pass 3's label logit.
    Resets both GEMM passes' counters."""
    blocks = math.ceil(n / MERGE_THREADS)

    @T.function
    def ce_merge_unfused(
        Part: T.Tensor((n, P, 4), "float32", align=16),
        Loss: T.Tensor((n,), "float32"),
        ProgMax: T.Tensor((nctr,), "int32"),
        ProgSum: T.Tensor((nctr,), "int32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(MERGE_THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        ot = T.alloc_local((4,), "float32")  # a slot's (m, l, t, c)
        mx = T.alloc_local((1,), "float32")
        lc = T.alloc_local((2,), "float32")  # l, its compensation
        tmp = T.alloc_local((3,), "float32")  # TwoSum scratch
        zero_counters(ProgMax, nctr, bx, tx)
        zero_counters(ProgSum, nctr, bx, tx)
        row = bx * MERGE_THREADS + tx
        if row < n:
            mx[0] = T.float32(NEG)
            lc[0] = T.float32(0)
            lc[1] = T.float32(0)
            for s in range(P):
                ot[T.Ramp(0, 1, 4)] = Part[row, s, T.Ramp(0, 1, 4)]
                mx[0] = fmax(mx[0], ot[0])
                two_sum(lc, 0, 1, ot[1], tmp, 1)
                lc[1] = lc[1] + ot[3]
            write_loss(Loss, row, mx[0], Part[row, 0, 2], lc)

    return ce_merge_unfused


def _schedule_table(p: dict) -> torch.Tensor:
    table = torch.zeros((SMS, p["nr"], 6), dtype=torch.int32)
    for c, rounds in enumerate(p["rounds"]):
        for i, u in enumerate(rounds):
            table[c, i] = torch.tensor(u, dtype=torch.int32)
    return table.cuda()


class Workspace:
    """The scratch of one call: ``part (n, P, 4)``, (m, l, t, c) per row and slot, where slots
    a row's group leaves unused stay the identity; and one zeroed progress counter per split
    for each GEMM pass (the call's last kernel zeroes them again)."""

    def __init__(self, n: int, P: int, nctr: int, passes: int, device):
        self.part = torch.zeros((n, P, 4), device=device, dtype=F32)
        self.part[:, :, 0] = NEG
        self.counters = tuple(
            torch.zeros(nctr, device=device, dtype=torch.int32) for _ in range(passes)
        )


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, d, v = cfg["n"], cfg["d"], cfg["v"]
    assert d % VEC == 0, "rows of X and W must be 16-byte aligned for TMA (d % 8 == 0)"
    assert max(n, v) * d < 2**31, "element offsets are int32"
    p = plan(n, d, v)
    P, nctr = p["P"], p["nctr"]
    name = f"cross_entropy_{n}x{d}x{v}"
    sched = _schedule_table(p)  # read only: shared by every call
    passes = 1 if variant == "fused" else 2
    spaces: dict[int, Workspace] = {}  # CUDA stream handle -> its workspace

    def workspace(device) -> Workspace:
        """The workspace of torch's current stream, which the kernels run on. Calls on one
        stream are ordered, so they share it; a call on another stream gets its own. While
        a CUDA graph is captured, every call gets a new one (the graph keeps it)."""
        if torch.cuda.is_current_stream_capturing():
            return Workspace(n, P, nctr, passes, device)
        key = torch.cuda.current_stream(device).cuda_stream
        ws = spaces.get(key)
        if ws is None:
            ws = spaces[key] = Workspace(n, P, nctr, passes, device)
        return ws

    if variant == "fused":
        main = compile_sm90(_main(n, d, v, p, "fused"), name=f"{name}_main")
        merge = compile_sm90(_merge(n, P, nctr), name=f"cross_entropy_merge_{n}x{P}x{nctr}")

        def run(X, W, labels):
            ws = workspace(X.device)
            (prog,) = ws.counters
            loss = torch.empty(n, device=X.device, dtype=F32)
            main(X, W, labels, sched, ws.part, prog)
            merge(ws.part, loss, prog)
            return (loss,)

        run.workspaces = spaces
        return run

    pl = label_plan(p)
    sched_label = _schedule_table(pl)
    pass_max = compile_sm90(_main(n, d, v, p, "max"), name=f"{name}_max")
    pass_sum = compile_sm90(_main(n, d, v, p, "sum"), name=f"{name}_sum")
    pass_label = compile_sm90(_main(n, d, v, pl, "label"), name=f"{name}_label")
    epilogue = compile_sm90(
        _merge_unfused(n, P, nctr), name=f"cross_entropy_merge_unfused_{n}x{P}x{nctr}"
    )

    def run_unfused(X, W, labels):
        ws = workspace(X.device)
        prog_max, prog_sum = ws.counters
        loss = torch.empty(n, device=X.device, dtype=F32)
        pass_max(X, W, labels, sched, ws.part, prog_max)
        pass_sum(X, W, labels, sched, ws.part, prog_sum)
        pass_label(X, W, labels, sched_label, ws.part, prog_max)  # (no counters used)
        epilogue(ws.part, loss, prog_max, prog_sum)
        return (loss,)

    run_unfused.workspaces = spaces
    return run_unfused
