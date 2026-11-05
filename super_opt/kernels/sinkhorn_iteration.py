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
"""One log-domain Sinkhorn iteration, the cost matrix never stored, in TIRx.

    f_i     = -eps LSE_j((g_j - C_ij) / eps + log b_j)
    g_new_j = -eps LSE_i((f_i - C_ij) / eps + log a_i),   C_ij = |x_i - y_j|²

The algorithm is chain_ops' ``sinkhorn_iteration`` fused (``_iteration(_lse_one_pass)``):
two one-pass online log-sum-exps, the first over the rows of C, the second over
its columns and fed by the new ``f``. Both halves are one reduction, for
``(P, Q, pot, w) = (x, y, g, b)`` and then ``(y, x, f, a)``:

    out_r = |p_r|² - eps ln2 LSE2_c(t_rc),   t_rc = u_c + 2K p_r.q_c,
    u_c   = K (pot_c - |q_c|²) + log2 w_c,    K = log2(e) / eps,

that is ``|p - q|² = |p|² + |q|² - 2 p.q`` with the row's ``|p|²`` taken out of
the log-sum-exp (it is a constant of the row) and everything in base 2, so a
score is one dot product plus ``u_c`` and each exponential one ``ex2.approx``.
The state of a row is ``(m, l)``: a reference ``m`` (never above the row's
largest score) and ``l = Σ 2^(t - m)``. A tile is folded in *lazily*: its sums
``Σ 2^(t - m)`` are added as they are, with no maximum taken per pair, and only
when one exceeds 2^64 (a score outran its reference) is the tile redone
exactly, with a new reference ``max(m, max t)``. The first tile of a range is
folded exactly at once (its rows are fresh). The epilogue, in float64, is
``|p|² - eps (ln2 m + ln l)``; the f half also writes ``u`` of the g half
(``K (f - |x|²) + log2 a``), so the g half reads its column terms ready-made.

Precision: everything is float32 on the CUDA cores (no TF32): the dot products
are FMAs, whose partial sums stay small (their terms change sign), the
per-column ``u`` and the per-row ``|p|²`` are exact to float32 rounding.

How it is laid out: a CTA of 256 threads owns ``BM = 128`` rows and walks a
range of columns tile by tile; a thread scores 8 rows against 8 columns at a
time, the 16 threads of a row merge their states with shuffles, and the row
results are written by one thread per row. Two paths:

* ``small`` (d <= 3, bound by ex2): threads 16 x 16, a thread's rows
  ``ty + 16 i`` with their ``2K p`` in registers; a tile is 256 columns, one
  float4 ``(q, u)`` each in shared memory, double buffered, every thread
  fetching one column of the next tile while this one is scored. A score is
  3 FMAs, and the lazy fold adds one FADD, one ex2 and one FADD per pair; its
  check rides on the tile's barrier (``__syncthreads_or``), so it costs no
  branch, and a tile that fails it is redone by the whole CTA;
* ``gemm`` (d % 4 == 0, bound by FP32 FMAs): 8 warps as 4 x 2, a warp 32 rows
  x 64 columns (lanes 4 x 8); the CTA's rows stay in shared memory, 128-column
  tiles stream in by 16-byte ``cp.async`` (double buffered), and a thread
  accumulates its 8 x 8 dot products from LDS.128 fragments along d (16
  LDS.128 per 256 FMAs). ``u`` of the f half comes from a small ``prep``
  kernel.

When the rows alone give too few CTAs, the columns are split over ``S`` CTAs;
each writes its rows' partial states, and the last CTA of a row block (an
atomic counter) merges them and writes the rows: no second launch.

``variant="unfused"`` keeps the unfused program's passes: per half a ``max``
pass (the row maxima of the scores) and a ``sum`` pass (``Σ 2^(t - max)`` and
the epilogue), each recomputing the scores from x and y, with the same tiles,
loads and split machinery.
"""

from __future__ import annotations

import functools
import math

import torch
from bench.ops import F32, SINKHORN_EPS

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90

EPS = SINKHORN_EPS
LN2 = math.log(2.0)
L2E = 1.0 / LN2
KC = L2E / EPS  # log2 units per unit of cost
TWO_KC = 2.0 * KC
NEG = -3.0e38  # a finite "-inf": the identity of max, and 2^(NEG - NEG) is no NaN
BIG = 2.0**64  # a tile sum above this: a score outran the row's reference
THREADS = 256
TX = 16  # threads along the columns of a tile
TY = THREADS // TX  # threads along the rows
TM = 8  # rows per thread
TN = 8  # columns per thread per tile
BM = TY * TM  # rows per CTA
BN = TX * TN  # columns per tile
VEC = 4  # float32 per 16 bytes
KUNROLL = 2  # 16-byte d-steps per iteration of the gemm path's k loop
# gemm path: 8 warps as 4 (rows) x 2 (columns), a warp 32 rows x 64 columns, its lanes 4 x 8;
# a thread's rows are rb0 + 4 i, its columns cb0 + 8 j
G_RS = 4
G_CS = 8
GEMM_PER_SM = 2  # CTAs per SM of the gemm path (<= 128 registers, 2 x 104 KB of shared memory)
SBN = 2 * BN  # columns per tile of the small path: two halves of BN, one fetched column per thread
FULL_MASK = 0xFFFFFFFF
CP16 = "cp.async.cg.shared.global"
CP4 = "cp.async.ca.shared.global"

KERNEL_META = {
    "op": "sinkhorn_iteration",
    "dtype": "float32",
    "algorithm": "two one-pass online log-sum-exps in base 2 over t = u_c + 2K p.q "
    "(|p|² taken out of the row), state (m, l) with a lazy reference: a tile's Σ 2^(t - m) "
    "is added as is unless it exceeds 2^64; C is recomputed per tile, never stored",
    "structure": "per half: 256-thread CTAs of 128 rows x a column range, tiles double buffered "
    "in shared memory; d=3: rows in registers, 256 (q, u) float4 columns per tile, "
    "3 FMA + FADD + ex2 + FADD per pair, lazy check on the tile barrier; d=64: 8x8 register "
    "tiles from LDS.128 fragments, 128-column cp.async tiles; column splits merged by the last "
    "CTA (atomic counter); 2 launches (d=3) or 3 (d=64, + prep)",
}


# --------------------------------------------------------------------------- states
@T.inline
def fmax(a, b):
    """fmaxf: one instruction (T.max lowers to a NaN-propagating compare and select)."""
    return T.call_pure_extern("fmaxf", a, b, ty="float32")


@T.inline
def lse_merge(st, ot):
    """(m, l) <- (m, l) ⊕ (m', l'): the larger reference, both sums rescaled to it."""
    mx = fmax(st[0], ot[0])
    st[1] = st[1] * T.exp2(st[0] - mx) + ot[1] * T.exp2(ot[0] - mx)
    st[0] = mx


@T.inline
def max_merge(st, ot):
    st[0] = fmax(st[0], ot[0])


@T.inline
def init_fresh(st, Mref, r0, rs, R):
    for i in T.unroll(TM):
        st[2 * i] = T.float32(NEG)
        st[2 * i + 1] = T.float32(0)


@T.inline
def init_ref(st, Mref, r0, rs, R):
    """The sum pass: the reference of a row (``r0 + rs i``) is its maximum, from the max pass."""
    for i in T.unroll(TM):
        st[2 * i] = Mref[T.min(r0 + rs * i, R - 1)]
        st[2 * i + 1] = T.float32(0)


# --------------------------------------------------------------------------- scores
@T.inline
def score_small(xs, qv, i, j):
    """u_c + 2K p.q, the row's 2K p in ``xs`` and the column's (q, u) in ``qv``: 3 FMAs."""
    return (
        qv[4 * j + 3]
        + xs[3 * i] * qv[4 * j]
        + xs[3 * i + 1] * qv[4 * j + 1]
        + xs[3 * i + 2] * qv[4 * j + 2]
    )


@T.inline
def score_gemm(acc, uj, i, j):
    """u_c + 2K p.q from the accumulated dot product: 1 FMA."""
    return acc[i * TN + j] * T.float32(TWO_KC) + uj[j]


# --------------------------------------------------------------------------- folds
@T.inline
def exact_row(st, tmp, score, sa, sb, i):
    """Fold row ``i``'s TN scores with a new reference max(m, max t)."""
    tmp[1] = st[2 * i]
    for j in T.unroll(TN):
        tmp[1] = fmax(tmp[1], score(sa, sb, i, j))
    tmp[0] = st[2 * i + 1] * T.exp2(st[2 * i] - tmp[1])
    for j in T.unroll(TN):
        tmp[0] = tmp[0] + T.exp2(score(sa, sb, i, j) - tmp[1])
    st[2 * i] = tmp[1]
    st[2 * i + 1] = tmp[0]


@T.inline
def fold_lse(st, lt, tmp, score, sa, sb, cb, cs, nc):
    """A full tile, lazily: the sums against the old references, unless one exceeds 2^64."""
    for i in T.unroll(TM):
        lt[i] = T.float32(0)
        for j in T.unroll(TN):
            lt[i] = lt[i] + T.exp2(score(sa, sb, i, j) - st[2 * i])
    tmp[0] = lt[0]
    for i in T.unroll(TM - 1):
        tmp[0] = fmax(tmp[0], lt[i + 1])
    if tmp[0] > T.float32(BIG):  # rare: a fresh row, or a score far above its reference
        for i in T.unroll(TM):
            exact_row(st, tmp, score, sa, sb, i)
    else:
        for i in T.unroll(TM):
            st[2 * i + 1] = st[2 * i + 1] + lt[i]


@T.inline
def fold_lse_tail(st, lt, tmp, score, sa, sb, cb, cs, nc):
    """The last, partial tile: exact, the thread's columns ``cb + cs j >= nc`` left out."""
    for i in T.unroll(TM):
        tmp[1] = st[2 * i]
        for j in T.unroll(TN):
            tmp[1] = fmax(
                tmp[1], T.if_then_else(cb + cs * j < nc, score(sa, sb, i, j), T.float32(NEG))
            )
        tmp[0] = st[2 * i + 1] * T.exp2(st[2 * i] - tmp[1])
        for j in T.unroll(TN):
            tmp[0] = tmp[0] + T.if_then_else(
                cb + cs * j < nc, T.exp2(score(sa, sb, i, j) - tmp[1]), T.float32(0)
            )
        st[2 * i] = tmp[1]
        st[2 * i + 1] = tmp[0]


@T.inline
def fold_max(st, lt, tmp, score, sa, sb, cb, cs, nc):
    for i in T.unroll(TM):
        for j in T.unroll(TN):
            st[2 * i] = fmax(st[2 * i], score(sa, sb, i, j))


@T.inline
def fold_max_tail(st, lt, tmp, score, sa, sb, cb, cs, nc):
    for i in T.unroll(TM):
        for j in T.unroll(TN):
            st[2 * i] = fmax(
                st[2 * i], T.if_then_else(cb + cs * j < nc, score(sa, sb, i, j), T.float32(NEG))
            )


@T.inline
def fold_sum(st, lt, tmp, score, sa, sb, cb, cs, nc):
    for i in T.unroll(TM):
        for j in T.unroll(TN):
            st[2 * i + 1] = st[2 * i + 1] + T.exp2(score(sa, sb, i, j) - st[2 * i])


@T.inline
def fold_sum_tail(st, lt, tmp, score, sa, sb, cb, cs, nc):
    for i in T.unroll(TM):
        for j in T.unroll(TN):
            st[2 * i + 1] = st[2 * i + 1] + T.if_then_else(
                cb + cs * j < nc, T.exp2(score(sa, sb, i, j) - st[2 * i]), T.float32(0)
            )


# mode -> (fold of a full tile, fold of the partial last tile, merge, init)
MODES = {
    "lse": (fold_lse, fold_lse_tail, lse_merge, init_fresh),
    "max": (fold_max, fold_max_tail, max_merge, init_fresh),
    "sum": (fold_sum, fold_sum_tail, lse_merge, init_ref),
}


# --------------------------------------------------------------------------- row results
@T.inline
def row_norm(nd, P, row, d):
    """|p_row|² in float64 (float32 products are exact in float64)."""
    nd[0] = T.float64(0)
    for k in T.serial(d):
        nd[0] = nd[0] + T.Cast("float64", P[row, k]) * T.Cast("float64", P[row, k])


@T.inline
def write_lse(P, Out, U2, W2, row, st, nd, d):
    """out = |p|² - eps (ln2 m + ln l), in float64."""
    row_norm(nd, P, row, d)
    v = T.float64(-EPS) * (
        T.Cast("float64", st[0]) * T.float64(LN2) + T.log(T.Cast("float64", st[1]))
    )
    Out[row] = T.Cast("float32", nd[0] + v)


@T.inline
def write_lse_emit(P, Out, U2, W2, row, st, nd, d):
    """The f half: also the g half's column term u = K (f - |x|²) + log2 a."""
    row_norm(nd, P, row, d)
    v = T.float64(-EPS) * (
        T.Cast("float64", st[0]) * T.float64(LN2) + T.log(T.Cast("float64", st[1]))
    )
    Out[row] = T.Cast("float32", nd[0] + v)
    U2[row] = T.Cast(
        "float32", T.float64(KC) * v + T.log(T.Cast("float64", W2[row])) * T.float64(L2E)
    )


@T.inline
def write_max(P, Out, U2, W2, row, st, nd, d):
    Out[row] = st[0]


@T.inline
def reduce_rows(st, two, oth, RS, tx, ty, merge):
    """Merge each row's 16 states (lanes tx = 0..15 of a half warp); tx == 0 parks them in RS."""
    for i in T.unroll(TM):
        two[0] = st[2 * i]
        two[1] = st[2 * i + 1]
        for s in T.unroll(4):
            oth[0] = T.gpu_warp_shuffle_xor(FULL_MASK, two[0], 8 >> s, 32, 32)
            oth[1] = T.gpu_warp_shuffle_xor(FULL_MASK, two[1], 8 >> s, 32, 32)
            merge(two, oth)
        if tx == 0:
            RS[ty + TY * i, 0] = two[0]
            RS[ty + TY * i, 1] = two[1]


@T.inline
def finish_direct(
    RS, P, Out, U2, W2, Part, Cnt, FLAG, two, oth, nd, tid, r0, rb, sp, write, merge, R, d, S
):
    """One CTA per row block: thread t writes row r0 + t."""
    if tid < BM:
        if r0 + tid < R:
            two[0] = RS[tid, 0]
            two[1] = RS[tid, 1]
            write(P, Out, U2, W2, r0 + tid, two, nd, d)


@T.inline
def finish_split(
    RS, P, Out, U2, W2, Part, Cnt, FLAG, two, oth, nd, tid, r0, rb, sp, write, merge, R, d, S
):
    """S CTAs per row block: park the partial states; the last CTA to arrive merges and writes."""
    if tid < BM:
        if r0 + tid < R:
            Part[sp, 0, r0 + tid] = RS[tid, 0]
            Part[sp, 1, r0 + tid] = RS[tid, 1]
    T.ptx["fence.acq_rel.gpu"]()
    T.cuda.cta_sync()
    if tid == 0:
        FLAG[0] = T.cuda.atomic_add(Cnt.ptr_to([rb]), T.int32(1))
    T.cuda.cta_sync()
    if FLAG[0] == S - 1:
        T.ptx["fence.acq_rel.gpu"]()
        if tid < BM:
            if r0 + tid < R:
                two[0] = Part[0, 0, r0 + tid]
                two[1] = Part[0, 1, r0 + tid]
                for s in T.serial(S - 1):
                    oth[0] = Part[s + 1, 0, r0 + tid]
                    oth[1] = Part[s + 1, 1, r0 + tid]
                    merge(two, oth)
                write(P, Out, U2, W2, r0 + tid, two, nd, d)
        if tid == 0:
            Cnt[rb] = 0  # ready for the next launch


# --------------------------------------------------------------------------- plan
def plan(R: int, Cn: int, per_sm: int, bn: int) -> dict:
    """Row blocks, and over how many CTAs (whole ``bn``-column tiles, none empty) to split columns.

    Picks the split that minimises the busiest SM's work, ``ceil(CTAs / SMS)``
    times the tiles of a CTA plus one for its fixed costs (rows, epilogue,
    merge), among those that give at least ``per_sm`` CTAs per SM (or as many
    as there are tiles); ties go to fewer splits.
    """
    row_blocks = -(-R // BM)
    tiles = -(-Cn // bn)
    best = None
    for s in range(1, tiles + 1):
        per = -(-tiles // s)
        s_eff = -(-tiles // per)
        if row_blocks * s_eff < 0.9 * SMS * per_sm and s_eff < tiles:
            continue
        cost = -(-(row_blocks * s_eff) // SMS) * (per + 1)
        if best is None or cost < best[0]:
            best = (cost, s_eff, per)
    _, splits, per = best
    return {"row_blocks": row_blocks, "splits": splits, "cols_per_split": per * bn}


def _picks(mode: str, emit: bool, splits: int):
    fold_full, fold_tail, merge, init = MODES[mode]
    write = write_max if mode == "max" else (write_lse_emit if emit else write_lse)
    finish = finish_direct if splits == 1 else finish_split
    return fold_full, fold_tail, merge, init, write, finish


# --------------------------------------------------------------------------- small d
@T.inline
def fetch_compute(pre, Q, pot, w, U, c, d):
    """Issue the loads of column c: q, pot, w (the f half computes u itself)."""
    for k in T.unroll(3):
        pre[k] = T.float32(0)
    for k in T.unroll(d):
        pre[k] = Q[c, k]
    pre[3] = pot[c]
    pre[4] = w[c]


@T.inline
def park_compute(pre, Yc, b, tid):
    """Yc[b, tid] = (q, u), u = K (pot - |q|²) + log2 w."""
    pre[3] = T.float32(KC) * (
        pre[3] - (pre[0] * pre[0] + pre[1] * pre[1] + pre[2] * pre[2])
    ) + T.log2(pre[4])
    Yc[b, tid, T.Ramp(0, 1, 4)] = pre[T.Ramp(0, 1, 4)]


@T.inline
def fetch_load(pre, Q, pot, w, U, c, d):
    """Issue the loads of column c: q and its ready-made u (the g half: from the f half's end)."""
    for k in T.unroll(3):
        pre[k] = T.float32(0)
    for k in T.unroll(d):
        pre[k] = Q[c, k]
    pre[3] = U[c]


@T.inline
def park_load(pre, Yc, b, tid):
    Yc[b, tid, T.Ramp(0, 1, 4)] = pre[T.Ramp(0, 1, 4)]


@T.inline
def load_half(qv, Yc, b, h, tx):
    """The thread's TN columns ``h BN + tx + 16 j`` of tile buffer ``b``."""
    for j in T.unroll(TN):
        qv[T.Ramp(4 * j, 1, 4)] = Yc[b, h * BN + tx + TX * j, T.Ramp(0, 1, 4)]


@T.inline
def step_lse(st, lt, tmp, xs, qv, Yc, b, tx, pre, tid, fold_full, park):
    """A full tile of the lse pass, lazily: the sums against the rows' references, with no
    branch; whether any thread's sum exceeded 2^64 is decided at the tile's barrier
    (``__syncthreads_or``), and then the whole CTA redoes the tile exactly (rare: the
    first tile, or a score far above its row's reference)."""
    for i in T.unroll(TM):
        lt[i] = T.float32(0)
    for h in T.unroll(SBN // BN):
        load_half(qv, Yc, b, h, tx)
        for i in T.unroll(TM):
            for j in T.unroll(TN):
                lt[i] = lt[i] + T.exp2(score_small(xs, qv, i, j) - st[2 * i])
    tmp[0] = lt[0]
    for i in T.unroll(TM - 1):
        tmp[0] = fmax(tmp[0], lt[i + 1])
    park(pre, Yc, 1 - b, tid)
    if T.cuda.syncthreads_or(T.Cast("int32", tmp[0] > T.float32(BIG))) != 0:
        for h in T.unroll(SBN // BN):
            load_half(qv, Yc, b, h, tx)
            for i in T.unroll(TM):
                exact_row(st, tmp, score_small, xs, qv, i)
        T.cuda.cta_sync()  # the buffer is refilled next
    else:
        for i in T.unroll(TM):
            st[2 * i + 1] = st[2 * i + 1] + lt[i]


@T.inline
def tile_exact(st, tmp, xs, qv, Yc, b, tx):
    """A full tile folded exactly, half by half (the lse pass's first tile: its rows are fresh)."""
    for h in T.unroll(SBN // BN):
        load_half(qv, Yc, b, h, tx)
        for i in T.unroll(TM):
            exact_row(st, tmp, score_small, xs, qv, i)


@T.inline
def step_plain(st, lt, tmp, xs, qv, Yc, b, tx, pre, tid, fold_full, park):
    """A full tile of the max or sum pass, half by half."""
    for h in T.unroll(SBN // BN):
        load_half(qv, Yc, b, h, tx)
        fold_full(st, lt, tmp, score_small, xs, qv, h * BN + tx, TX, SBN)
    park(pre, Yc, 1 - b, tid)
    T.cuda.cta_sync()


def _small_kernel(R: int, Cn: int, d: int, mode: str, ucompute: bool, emit: bool):
    assert SBN == THREADS
    p = plan(R, Cn, per_sm=2, bn=SBN)
    rb_n, S, cps = p["row_blocks"], p["splits"], p["cols_per_split"]
    blocks = rb_n * S
    fold_full, fold_tail, merge, init, write, finish = _picks(mode, emit, S)
    fetch, park = (fetch_compute, park_compute) if ucompute else (fetch_load, park_load)
    step = step_lse if mode == "lse" else step_plain
    peel = mode == "lse"  # the first tile of the lse pass is folded exactly, without a lazy try

    @T.function
    def sinkhorn_small(
        P: T.Tensor((R, d), "float32"),
        Q: T.Tensor((Cn, d), "float32"),
        pot: T.Tensor((Cn,), "float32"),
        w: T.Tensor((Cn,), "float32"),
        U: T.Tensor((Cn,), "float32"),
        Mref: T.Tensor((R,), "float32"),
        W2: T.Tensor((R,), "float32"),
        Out: T.Tensor((R,), "float32"),
        U2: T.Tensor((R,), "float32"),
        Part: T.Tensor((S, 2, R), "float32"),
        Cnt: T.Tensor((rb_n,), "int32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        tx = tid % TX
        ty = tid // TX
        Yc = T.alloc_shared((2, SBN, 4), "float32")  # (q0, q1, q2, u) per column, double buffered
        RS = T.alloc_shared((BM, 2), "float32")
        FLAG = T.alloc_shared((1,), "int32")
        xs = T.alloc_local((TM * 3,), "float32")  # 2K p of the thread's rows
        qv = T.alloc_local((TN * 4,), "float32")  # the thread's columns of the tile
        pre = T.alloc_local((5,), "float32")  # this thread's column of the next tile
        st = T.alloc_local((2 * TM,), "float32")
        lt = T.alloc_local((TM,), "float32")
        tmp = T.alloc_local((2,), "float32")
        two = T.alloc_local((2,), "float32")
        oth = T.alloc_local((2,), "float32")
        nd = T.alloc_local((1,), "float64")
        rb = bx % rb_n
        sp = bx // rb_n
        r0 = rb * BM
        c_lo = sp * cps
        c_hi = T.min(c_lo + cps, Cn)
        nf = (c_hi - c_lo) // SBN  # full tiles
        for i in T.unroll(TM):
            for k in T.unroll(3):
                xs[3 * i + k] = T.float32(0)
            for k in T.unroll(d):
                xs[3 * i + k] = P[T.min(r0 + ty + TY * i, R - 1), k] * T.float32(TWO_KC)
        init(st, Mref, r0 + ty, TY, R)
        # every thread fetches one column of a tile; columns past the range are loaded
        # clamped (a valid address) and masked in the last tile
        fetch(pre, Q, pot, w, U, T.min(c_lo + tid, Cn - 1), d)
        park(pre, Yc, 0, tid)
        T.cuda.cta_sync()
        t: T.int32 = 0
        if peel:
            if nf > 0:
                fetch(pre, Q, pot, w, U, T.min(c_lo + SBN + tid, Cn - 1), d)
                tile_exact(st, tmp, xs, qv, Yc, 0, tx)
                park(pre, Yc, 1, tid)
                T.cuda.cta_sync()
                t = 1
        while t < nf:
            # the next tile's loads, in flight while this one is scored
            fetch(pre, Q, pot, w, U, T.min(c_lo + (t + 1) * SBN + tid, Cn - 1), d)
            step(st, lt, tmp, xs, qv, Yc, t % 2, tx, pre, tid, fold_full, park)
            t += 1
        if c_lo + nf * SBN < c_hi:
            for h in T.unroll(SBN // BN):
                load_half(qv, Yc, nf % 2, h, tx)
                fold_tail(st, lt, tmp, score_small, xs, qv, h * BN + tx, TX, c_hi - c_lo - nf * SBN)
        reduce_rows(st, two, oth, RS, tx, ty, merge)
        T.cuda.cta_sync()
        finish(
            RS,
            P,
            Out,
            U2,
            W2,
            Part,
            Cnt,
            FLAG,
            two,
            oth,
            nd,
            tid,
            r0,
            rb,
            sp,
            write,
            merge,
            R,
            d,
            S,
        )

    return sinkhorn_small, S, rb_n


# --------------------------------------------------------------------------- gemm (d % 4 == 0)
@T.inline
def load_rows(arena, slot, src, base, nvalid, pitch_chunks, tid):
    """arena[slot, r, :] = src[base + r, :] for r < nvalid (else zeros), 16-byte cp.async."""
    for q in T.unroll(BM * pitch_chunks // THREADS):
        r = (tid + q * THREADS) // pitch_chunks
        v = (tid + q * THREADS) % pitch_chunks
        T.ptx[CP16](
            arena.ptr_to([slot, r, v * VEC]),
            src.ptr_to([base + T.min(r, nvalid - 1), v * VEC]),
            16,
            T.if_then_else(r < nvalid, T.uint32(16), T.uint32(0)),
        )


@T.inline
def load_u(Us, b, U, c0, nvalid, tid):
    if tid < BN:
        T.ptx[CP4](
            Us.ptr_to([b, tid]),
            U.ptr_to([c0 + T.min(tid, nvalid - 1)]),
            4,
            T.if_then_else(tid < nvalid, T.uint32(4), T.uint32(0)),
        )


@T.inline
def dot_tile(acc, xv, yv, uj, arena, Us, slot, b, rb0, cb0, d):
    """acc[i, j] = p_{rb0 + 4i} . q_{cb0 + 8j} over d, from LDS.128 fragments along d."""
    for e in T.unroll(TM * TN):
        acc[e] = T.float32(0)
    for kb in T.serial(d // (VEC * KUNROLL)):
        for ku in T.unroll(KUNROLL):
            for j in T.unroll(TN):
                yv[T.Ramp(VEC * j, 1, VEC)] = arena[
                    slot, cb0 + G_CS * j, T.Ramp(VEC * (kb * KUNROLL + ku), 1, VEC)
                ]
            for i in T.unroll(TM):
                xv[T.Ramp(0, 1, VEC)] = arena[
                    0, rb0 + G_RS * i, T.Ramp(VEC * (kb * KUNROLL + ku), 1, VEC)
                ]
                for e in T.unroll(VEC):  # e outer: consecutive FMAs feed different accumulators
                    for j in T.unroll(TN):
                        acc[i * TN + j] = acc[i * TN + j] + xv[e] * yv[VEC * j + e]
    for j in T.unroll(TN):
        uj[j] = Us[b, cb0 + G_CS * j]


@T.inline
def reduce_rows_gemm(st, two, oth, RS, RSX, cg, wc, rb0, merge):
    """Merge each row's 16 states: 8 lanes (cg) by shuffles, then the 2 warps (wc) via RSX."""
    for i in T.unroll(TM):
        two[0] = st[2 * i]
        two[1] = st[2 * i + 1]
        for s in T.unroll(3):
            oth[0] = T.gpu_warp_shuffle_xor(FULL_MASK, two[0], 4 >> s, 32, 32)
            oth[1] = T.gpu_warp_shuffle_xor(FULL_MASK, two[1], 4 >> s, 32, 32)
            merge(two, oth)
        st[2 * i] = two[0]
        st[2 * i + 1] = two[1]
    if cg == 0:
        if wc == 1:
            for i in T.unroll(TM):
                RSX[rb0 + G_RS * i, 0] = st[2 * i]
                RSX[rb0 + G_RS * i, 1] = st[2 * i + 1]
    T.cuda.cta_sync()
    if cg == 0:
        if wc == 0:
            for i in T.unroll(TM):
                two[0] = st[2 * i]
                two[1] = st[2 * i + 1]
                oth[0] = RSX[rb0 + G_RS * i, 0]
                oth[1] = RSX[rb0 + G_RS * i, 1]
                merge(two, oth)
                RS[rb0 + G_RS * i, 0] = two[0]
                RS[rb0 + G_RS * i, 1] = two[1]


def _gemm_kernel(R: int, Cn: int, d: int, mode: str, emit: bool):
    assert d % VEC == 0
    assert d % (VEC * KUNROLL) == 0
    p = plan(R, Cn, per_sm=GEMM_PER_SM, bn=BN)
    rb_n, S, cps = p["row_blocks"], p["splits"], p["cols_per_split"]
    blocks = rb_n * S
    fold_full, fold_tail, merge, init, write, finish = _picks(mode, emit, S)
    pitch = d + VEC  # a padded row: 16 rows of LDS.128 at one k hit distinct bank groups
    chunks = d // VEC  # 16-byte chunks per row
    assert (BM * chunks) % THREADS == 0

    @T.function
    def sinkhorn_gemm(
        P: T.Tensor((R, d), "float32", align=16),
        Q: T.Tensor((Cn, d), "float32", align=16),
        pot: T.Tensor((Cn,), "float32"),
        w: T.Tensor((Cn,), "float32"),
        U: T.Tensor((Cn,), "float32"),
        Mref: T.Tensor((R,), "float32"),
        W2: T.Tensor((R,), "float32"),
        Out: T.Tensor((R,), "float32"),
        U2: T.Tensor((R,), "float32"),
        Part: T.Tensor((S, 2, R), "float32"),
        Cnt: T.Tensor((rb_n,), "int32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=GEMM_PER_SM),
        )
        bx = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        warp = tid // 32
        lane = tid % 32
        wc = warp // 4
        cg = lane % G_CS
        rb0 = (warp % 4) * 32 + lane // G_CS  # the thread's first row (of the CTA's)
        cb0 = wc * 64 + cg  # its first column (of a tile)
        # slot 0: the CTA's rows of P; slots 1, 2: column tiles of Q, double buffered
        arena = T.alloc_tensor(ty_args=[T.Tensor((3, BM, pitch), "float32", scope="shared.dyn")])
        Us = T.alloc_shared((2, BN), "float32")
        RS = T.alloc_shared((BM, 2), "float32")
        RSX = T.alloc_shared((BM, 2), "float32")
        FLAG = T.alloc_shared((1,), "int32")
        acc = T.alloc_local((TM * TN,), "float32")
        xv = T.alloc_local((VEC,), "float32")
        yv = T.alloc_local((VEC * TN,), "float32")
        uj = T.alloc_local((TN,), "float32")
        st = T.alloc_local((2 * TM,), "float32")
        lt = T.alloc_local((TM,), "float32")
        tmp = T.alloc_local((2,), "float32")
        two = T.alloc_local((2,), "float32")
        oth = T.alloc_local((2,), "float32")
        nd = T.alloc_local((1,), "float64")
        rb = bx % rb_n
        sp = bx // rb_n
        r0 = rb * BM
        c_lo = sp * cps
        c_hi = T.min(c_lo + cps, Cn)
        nf = (c_hi - c_lo) // BN
        init(st, Mref, r0 + rb0, G_RS, R)
        load_rows(arena, 0, P, r0, T.min(BM, R - r0), chunks, tid)
        load_rows(arena, 1, Q, c_lo, T.min(BN, c_hi - c_lo), chunks, tid)
        load_u(Us, 0, U, c_lo, T.min(BN, c_hi - c_lo), tid)
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
        T.cuda.cta_sync()
        t: T.int32 = 0
        while t < nf:
            if c_lo + (t + 1) * BN < c_hi:
                load_rows(
                    arena,
                    1 + (t + 1) % 2,
                    Q,
                    c_lo + (t + 1) * BN,
                    T.min(BN, c_hi - c_lo - (t + 1) * BN),
                    chunks,
                    tid,
                )
                load_u(
                    Us,
                    (t + 1) % 2,
                    U,
                    c_lo + (t + 1) * BN,
                    T.min(BN, c_hi - c_lo - (t + 1) * BN),
                    tid,
                )
                T.ptx.cp.async_.commit_group()
            dot_tile(acc, xv, yv, uj, arena, Us, 1 + t % 2, t % 2, rb0, cb0, d)
            fold_full(st, lt, tmp, score_gemm, acc, uj, cb0, G_CS, BN)
            T.ptx.cp.async_.wait_group(0)
            T.cuda.cta_sync()
            t += 1
        if c_lo + nf * BN < c_hi:
            dot_tile(acc, xv, yv, uj, arena, Us, 1 + nf % 2, nf % 2, rb0, cb0, d)
            fold_tail(st, lt, tmp, score_gemm, acc, uj, cb0, G_CS, c_hi - c_lo - nf * BN)
        reduce_rows_gemm(st, two, oth, RS, RSX, cg, wc, rb0, merge)
        T.cuda.cta_sync()
        finish(
            RS,
            P,
            Out,
            U2,
            W2,
            Part,
            Cnt,
            FLAG,
            two,
            oth,
            nd,
            tid,
            r0,
            rb,
            sp,
            write,
            merge,
            R,
            d,
            S,
        )

    return sinkhorn_gemm, S, rb_n


def _prep_kernel(Cn: int, d: int):
    """u_c = K (pot_c - |q_c|²) + log2 w_c of every column (the gemm path's f half).

    16 lanes per column, each 16 bytes at a time, their partial sums added by a
    butterfly (a pairwise sum: float32 is exact enough here).
    """
    lanes = 16
    steps = -(-(d // VEC) // lanes)  # 16-byte steps per lane
    blocks = -(-Cn * lanes // THREADS)

    @T.function
    def sinkhorn_prep(
        Q: T.Tensor((Cn, d), "float32", align=16),
        pot: T.Tensor((Cn,), "float32"),
        w: T.Tensor((Cn,), "float32"),
        U: T.Tensor((Cn,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        buf = T.alloc_local((VEC,), "float32")
        part = T.alloc_local((1,), "float32")
        c = T.min((bx * THREADS + tid) // lanes, Cn - 1)
        lane = tid % lanes
        part[0] = T.float32(0)
        for v in T.unroll(steps):
            if lane + lanes * v < d // VEC:
                buf[T.Ramp(0, 1, VEC)] = Q[c, T.Ramp(VEC * (lane + lanes * v), 1, VEC)]
                for e in T.unroll(VEC):
                    part[0] = part[0] + buf[e] * buf[e]
        for s in T.unroll(4):
            part[0] = part[0] + T.gpu_warp_shuffle_xor(FULL_MASK, part[0], 8 >> s, 32, 32)
        if lane == 0:
            if (bx * THREADS + tid) // lanes < Cn:
                U[c] = T.float32(KC) * (pot[c] - part[0]) + T.log2(w[c])

    return sinkhorn_prep


# --------------------------------------------------------------------------- host side
def _small_path(d: int) -> bool:
    if d <= 3:
        return True
    assert d % VEC == 0, "d must be at most 3 or a multiple of 4"
    return False


class _Pass:
    """One compiled pass over R rows and Cn columns, with its cached workspaces."""

    def __init__(self, R, Cn, d, mode, ucompute, emit):
        if _small_path(d):
            func, self.S, rb_n = _small_kernel(R, Cn, d, mode, ucompute, emit)
        else:
            func, self.S, rb_n = _gemm_kernel(R, Cn, d, mode, emit)
        tag = ("_u" if ucompute else "") + ("_e" if emit else "")
        name = f"sinkhorn_{mode}{tag}_{R}x{Cn}x{d}_s{self.S}"
        self.mod = compile_sm90(func, name=name)
        self.R, self.Cn, self.rb_n = R, Cn, rb_n
        self.ws = None

    def __call__(self, P, Q, pot=None, w=None, U=None, Mref=None, W2=None, Out=None, U2=None):
        if self.ws is None:
            dev = P.device
            self.ws = {
                "R": torch.zeros(self.R, device=dev, dtype=F32),
                "C": torch.zeros(self.Cn, device=dev, dtype=F32),
                "part": torch.empty((self.S, 2, self.R), device=dev, dtype=F32),
                "cnt": torch.zeros(self.rb_n, device=dev, dtype=torch.int32),
            }
        ws = self.ws
        self.mod(
            P,
            Q,
            ws["C"] if pot is None else pot,
            ws["C"] if w is None else w,
            ws["C"] if U is None else U,
            ws["R"] if Mref is None else Mref,
            ws["R"] if W2 is None else W2,
            ws["R"] if Out is None else Out,
            ws["R"] if U2 is None else U2,
            ws["part"],
            ws["cnt"],
        )


@functools.cache
def _pass(R, Cn, d, mode, ucompute, emit):
    return _Pass(R, Cn, d, mode, ucompute, emit)


@functools.cache
def _prep(Cn, d):
    return compile_sm90(_prep_kernel(Cn, d), name=f"sinkhorn_prep_{Cn}x{d}")


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, m, d = cfg["n"], cfg["m"], cfg["d"]
    assert max(n, m) * d < 2**31, "int32 indexing"
    small = _small_path(d)
    uc = small  # the small path computes the f half's u while it loads a tile
    prep = None if small else _prep(m, d)
    if variant == "fused":
        f_lse = _pass(n, m, d, "lse", uc, True)
        g_lse = _pass(m, n, d, "lse", False, False)
    else:
        f_max = _pass(n, m, d, "max", uc, False)
        f_sum = _pass(n, m, d, "sum", uc, True)
        g_max = _pass(m, n, d, "max", False, False)
        g_sum = _pass(m, n, d, "sum", False, False)
    bufs = {}

    def scratch(dev):
        if dev not in bufs:
            bufs[dev] = {
                "u1": torch.empty(m, device=dev, dtype=F32),
                "u2": torch.empty(n, device=dev, dtype=F32),
                "m1": torch.empty(n, device=dev, dtype=F32),
                "m2": torch.empty(m, device=dev, dtype=F32),
            }
        return bufs[dev]

    def run(x, y, a, b, g):
        s = scratch(x.device)
        f = torch.empty(n, device=x.device, dtype=F32)
        g_new = torch.empty(m, device=x.device, dtype=F32)
        u1 = None
        if prep is not None:
            prep(y, g, b, s["u1"])
            u1 = s["u1"]
        if variant == "fused":
            f_lse(x, y, pot=g, w=b, U=u1, W2=a, Out=f, U2=s["u2"])
            g_lse(y, x, U=s["u2"], Out=g_new)
        else:
            f_max(x, y, pot=g, w=b, U=u1, Out=s["m1"])
            f_sum(x, y, pot=g, w=b, U=u1, Mref=s["m1"], W2=a, Out=f, U2=s["u2"])
            g_max(y, x, U=s["u2"], Out=s["m2"])
            g_sum(y, x, U=s["u2"], Mref=s["m2"], Out=g_new)
        return (f, g_new)

    return run
