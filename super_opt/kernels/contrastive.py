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
"""The symmetric contrastive (CLIP / InfoNCE) loss per example, reading S once.

``rows_i = lse(S[i, :]) - S[i, i]`` and ``cols_i = lse(S[:, i]) - S[i, i]``.
The algorithm is chain_ops' ``contrastive_fused``: one online log-sum-exp
chain per direction, each with the state ``(m, l)``, a reference ``m`` and
``l = Σ 2^(x log2 e - m)`` (base 2: an exponential is one FMA and one
``ex2``). Two states merge by moving both to the larger reference
(``lse_combine``); the epilogue is ``(m + log2 l) ln 2 - S[i, i]``.

The unfused program reads S twice, the column pass with a stride; here every
element is read from HBM once and feeds both chains.

How the fused variant is laid out (two kernels):

* ``partials``: the grid is ``(column blocks, row blocks)``, one full wave
  of 4 CTAs per SM or more. A CTA of 128 threads (4 warps) owns ``TN = 512``
  columns and a row block of ``R`` rows, and walks it ``4 * RC`` rows at a
  time: warp ``w`` takes ``RC = 2`` consecutive rows, and in each lane ``l``
  owns 16 columns: two 16-byte loads per row, all four of a step issued
  together, one step ahead of the math (software prefetch into registers).
  When ``n`` is not a multiple of 8 the rows are not 16-byte aligned and the
  lanes load one element at a time instead.

  - Row chain, exact: the row's maximum over the 512 columns (``fmaxf``
    butterfly) is its reference ``b``; ``e = 2^(x log2 e - b)`` is the
    element's one exponential, summed by a butterfly into ``PR[cb, row]``.
  - Column chain, from the same ``e``: the warp keeps one reference ``W`` for
    all its columns (the largest ``b`` it has seen, rescaling its sums by one
    ``ex2`` when it grows); a column adds ``e * 2^(b - W)``, one FMA, with one
    ``ex2`` per row. So an element costs one exponential, not two.
  - The 4 warps' column states merge through shared memory into
    ``PC[rb, col]``. ``W`` is not the column's own maximum, so a column far
    below the block's largest value could lose its terms to underflow; a
    column whose merged ``l < 2^-60`` makes the CTA redo its columns with
    exact per-column references (never taken on the benchmark's inputs).

* ``merge``: a CTA per 32 indices; each of its 8 warps merges every 8th
  partial (all loads issued at once), then the 8 states per index merge
  through shared memory and the diagonal is subtracted.

The ``unfused`` variant has the unfused program's four passes, each reading
S again: the rows' maxima, the rows' ``Σ exp`` (one 256-thread CTA per row,
4 16-byte loads in flight per thread), the columns' maxima and the columns'
``Σ exp`` (the fused kernel's tile grid, loads and prefetch); only per-row or
per-column vectors (and the column passes' per-row-block partials, merged
like the fused ones) travel between them.
"""

from __future__ import annotations

import math

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce

NEG = -1.0e38  # a finite "-inf"; NEG * log2 e is still finite
L2E = 1.4426950408889634
LN2 = 0.6931471805599453
THRESH = 2.0**-60  # a merged column sum below this takes the exact path
FULL_MASK = 0xFFFFFFFF
THREADS = 128  # tile CTAs: 4 warps
WARPS = THREADS // 32
LANES = 32
VEC = vector_width("bfloat16")  # 8 bf16 = 16 bytes
SEG = 2  # 16-byte chunks per lane per row
VPL = SEG * VEC  # 16 columns per lane
TN = LANES * VPL  # 512 columns per CTA
RC = 2  # rows a warp takes per step
STEP = WARPS * RC  # rows a CTA takes per step
CTAS_PER_SM = 4  # resident tile CTAs per SM (launch bounds: <= 128 registers)
MERGE_WARPS = 8  # merges: a warp per every 8th partial of 32 indices
ROW_UNROLL = 4  # 16-byte loads in flight per thread in the unfused row passes
ROW_CTAS_PER_SM = 8
ROW_THREADS = 256  # unfused row passes: one CTA per row
ROW_WARPS = ROW_THREADS // 32

KERNEL_META = {
    "op": "contrastive",
    "dtype": "bfloat16 in, float32 out",
    "algorithm": "one pass over S, one (m, l) = (reference, Σ 2^(x log2 e - m)) chain per row "
    "and per column; one ex2 per element feeds both (the column's via a per-row factor); "
    "out = (m + log2 l) ln 2 - S[i, i]",
    "structure": "one wave of (512 columns x R rows) tiles, 128-thread CTAs, 4 per SM; 2 rows x "
    "2 16-byte loads per lane per step, prefetched a step ahead; fmaxf/sum butterflies per row, "
    "column states in "
    "registers merged through shared memory; per-row and per-column partials merged by a "
    "second kernel (a warp per every 8th partial)",
}


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def plan(n: int) -> tuple[int, int, int]:
    """``(column blocks, row blocks, rows per row block)`` for an ``n x n`` S.

    The CTA count is a multiple of the resident slots (``SMS * CTAS_PER_SM``)
    where it can be, and at least one full wave of them.
    """
    ncb = _cdiv(n, TN)
    slots = SMS * CTAS_PER_SM
    nrb = slots // math.gcd(slots, ncb)
    while ncb * nrb < slots:
        nrb *= 2
    nrb = max(1, min(nrb, _cdiv(n, STEP)))
    rows = _cdiv(n, nrb)
    return ncb, _cdiv(n, rows), rows


def _fmax(a, b):
    """One ``FMNMX``; ``T.max`` is a NaN-propagating compare and select."""
    return T.call_pure_extern("fmaxf", a, b, ty="float32")


def _fma(a, b, c):
    return T.call_pure_extern("__fmaf_rn", a, b, c, ty="float32")


@T.inline
def lse_combine(st, ot):
    """(m, l) <- (m, l) ⊕ (m', l') in base 2: keep the larger reference, rescale both sums."""
    mx = _fmax(st[0], ot[0])
    st[1] = st[1] * T.exp2(st[0] - mx) + ot[1] * T.exp2(ot[0] - mx)
    st[0] = mx


@T.inline
def max_combine(st, ot):
    st[0] = _fmax(st[0], ot[0])


@T.inline
def add_combine(st, ot):
    st[0] = st[0] + ot[0]


@T.inline
def warp_fmax(a, o, k):
    """a[0:k] <- the maximum over the warp, in every lane."""
    for i in T.unroll(5):
        for j in T.unroll(k):
            o[j] = T.gpu_warp_shuffle_xor(FULL_MASK, a[j], 16 >> i, 32, 32)
        for j in T.unroll(k):
            a[j] = _fmax(a[j], o[j])


@T.inline
def warp_fsum(a, o, k):
    """a[0:k] <- the sum over the warp, in every lane (butterfly order)."""
    for i in T.unroll(5):
        for j in T.unroll(k):
            o[j] = T.gpu_warp_shuffle_xor(FULL_MASK, a[j], 16 >> i, 32, 32)
        for j in T.unroll(k):
            a[j] = a[j] + o[j]


# ---------------------------------------------------------------------------
# the tile kernels' loads: a lane's chunk (r, s) of a step is the 16 bytes of
# S[row0 + r] at columns cb*TN + s*256 + lane*8 (+ 0..7)
# ---------------------------------------------------------------------------
@T.inline
def load_direct(raw, S, row0, cb, lane, n, nvec, col_tail):
    """raw[r, s] = the lane's 16-byte chunk of S[row0 + r], columns cb*TN + s*256 + lane*8 + 0..7.

    All ``RC * SEG`` loads at once, unconditionally: the row (and, with a column
    tail, the column) is clamped into S; ``unpack`` masks what lies outside.
    """
    for r in T.unroll(RC):
        for s in T.unroll(SEG):
            if col_tail:
                raw[r, s, T.Ramp(0, 1, VEC)] = S[
                    T.min(row0 + r, n - 1),
                    T.Ramp(T.min(cb * (TN // VEC) + s * LANES + lane, nvec - 1) * VEC, 1, VEC),
                ]
            else:
                raw[r, s, T.Ramp(0, 1, VEC)] = S[
                    T.min(row0 + r, n - 1),
                    T.Ramp((cb * (TN // VEC) + s * LANES + lane) * VEC, 1, VEC),
                ]


@T.inline
def unpack(x, raw, S, row0, row_end, cb, lane, n, nvec, vectorised, masked):
    """x[r, s*8 + j] = S[row0 + r, cb*TN + s*256 + lane*8 + j] as float32; NEG outside S.

    Vectorised, from the chunks in ``raw`` (selects only when ``masked``: a
    partial step or a column tail); otherwise loaded here one element at a time.
    The callers unpack a step, then load the next one into ``raw``, so its
    loads fly during the step's math (and cannot be sunk to their uses).
    """
    if vectorised:
        for r in T.unroll(RC):
            for s in T.unroll(SEG):
                for j in T.unroll(VEC):
                    if masked:
                        x[r, s * VEC + j] = T.if_then_else(
                            row0 + r < row_end,
                            T.if_then_else(
                                cb * (TN // VEC) + s * LANES + lane < nvec,
                                T.Cast("float32", raw[r, s, j]),
                                T.float32(NEG),
                            ),
                            T.float32(NEG),
                        )
                    else:
                        x[r, s * VEC + j] = T.Cast("float32", raw[r, s, j])
    else:
        for r in T.unroll(RC):
            for s in T.unroll(SEG):
                for j in T.unroll(VEC):
                    if row0 + r < row_end and cb * TN + s * (LANES * VEC) + lane * VEC + j < n:
                        x[r, s * VEC + j] = T.Cast(
                            "float32", S[row0 + r, cb * TN + s * (LANES * VEC) + lane * VEC + j]
                        )
                    else:
                        x[r, s * VEC + j] = T.float32(NEG)


# ---------------------------------------------------------------------------
# fused: one pass, both chains
# ---------------------------------------------------------------------------
@T.inline
def fused_step(
    x,
    raw,
    rm,
    rs,
    fr,
    tmp,
    w,
    pv,
    cl,
    S,
    PR,
    row0,
    r_end,
    more,
    cb,
    lane,
    n,
    nvec,
    vectorised,
    col_tail,
    masked,
):
    """One step of a warp: RC rows x 16 columns into the rows' partials and the columns' sums."""
    unpack(x, raw, S, row0, r_end, cb, lane, n, nvec, vectorised, masked)
    if more:  # the next step's loads, in flight during this step's math
        load_direct(raw, S, row0 + STEP, cb, lane, n, nvec, col_tail)
    for r in T.unroll(RC):
        rm[r] = x[r, 0]
        for k in T.unroll(VPL - 1):
            rm[r] = _fmax(rm[r], x[r, k + 1])
    warp_fmax(rm, tmp, RC)
    # references in base 2; a row past the end gets 0 and its NEG elements give 0
    w[1] = w[0]
    for r in T.unroll(RC):
        if masked:
            rm[r] = T.if_then_else(row0 + r < r_end, rm[r] * T.float32(L2E), T.float32(0))
            w[1] = T.if_then_else(row0 + r < r_end, _fmax(w[1], rm[r]), w[1])
        else:
            rm[r] = rm[r] * T.float32(L2E)
            w[1] = _fmax(w[1], rm[r])
    if w[1] > w[0]:  # the warp's column reference grows: rescale its sums (one ex2)
        pv[0] = T.exp2(w[0] - w[1])
        for k in T.unroll(VPL):
            cl[k] = cl[k] * pv[0]
        w[0] = w[1]
    for r in T.unroll(RC):
        if masked:
            fr[r] = T.if_then_else(row0 + r < r_end, T.exp2(rm[r] - w[0]), T.float32(0))
        else:
            fr[r] = T.exp2(rm[r] - w[0])
    for r in T.unroll(RC):
        rs[r] = T.float32(0)
        for k in T.unroll(VPL):
            e = T.exp2(_fma(x[r, k], T.float32(L2E), T.float32(0) - rm[r]))
            rs[r] = rs[r] + e
            cl[k] = _fma(e, fr[r], cl[k])
    warp_fsum(rs, tmp, RC)
    if lane < RC:  # lane r writes row r's (b, l)
        pv[0] = rm[0]
        pv[1] = rs[0]
        for r in T.unroll(RC - 1):
            pv[0] = T.if_then_else(lane == r + 1, rm[r + 1], pv[0])
            pv[1] = T.if_then_else(lane == r + 1, rs[r + 1], pv[1])
        if masked:
            if row0 + lane < r_end:
                PR[cb, row0 + lane, T.Ramp(0, 1, 2)] = pv[T.Ramp(0, 1, 2)]
        else:
            PR[cb, row0 + lane, T.Ramp(0, 1, 2)] = pv[T.Ramp(0, 1, 2)]


@T.inline
def fused_loop_vec(
    x,
    raw,
    rm,
    rs,
    fr,
    tmp,
    w,
    pv,
    cl,
    S,
    PR,
    r_begin,
    r_end,
    nsteps,
    nfull,
    cb,
    warp,
    lane,
    n,
    nvec,
    col_tail,
):
    """The steps with 16-byte loads, each step's issued during the previous one."""
    load_direct(raw, S, r_begin + warp * RC, cb, lane, n, nvec, col_tail)
    for step in T.serial(nsteps):
        row0 = r_begin + step * STEP + warp * RC
        if step < nfull:  # CTA-uniform: every warp's rows are in the block
            fused_step(
                x,
                raw,
                rm,
                rs,
                fr,
                tmp,
                w,
                pv,
                cl,
                S,
                PR,
                row0,
                r_end,
                step + 1 < nsteps,
                cb,
                lane,
                n,
                nvec,
                True,
                col_tail,
                col_tail,
            )
        else:
            fused_step(
                x,
                raw,
                rm,
                rs,
                fr,
                tmp,
                w,
                pv,
                cl,
                S,
                PR,
                row0,
                r_end,
                step + 1 < nsteps,
                cb,
                lane,
                n,
                nvec,
                True,
                col_tail,
                True,
            )


@T.inline
def fused_loop_scalar(
    x,
    raw,
    rm,
    rs,
    fr,
    tmp,
    w,
    pv,
    cl,
    S,
    PR,
    r_begin,
    r_end,
    nsteps,
    nfull,
    cb,
    warp,
    lane,
    n,
    nvec,
    col_tail,
):
    """The steps, loading one element at a time (rows not 16-byte aligned)."""
    for step in T.serial(nsteps):
        row0 = r_begin + step * STEP + warp * RC
        fused_step(
            x,
            raw,
            rm,
            rs,
            fr,
            tmp,
            w,
            pv,
            cl,
            S,
            PR,
            row0,
            r_end,
            False,
            cb,
            lane,
            n,
            nvec,
            False,
            col_tail,
            True,
        )


def _partials(n: int):
    ncb, nrb, rows = plan(n)
    vectorised = vectorisable(n, "bfloat16")
    nvec = n // VEC
    col_tail = n % TN != 0  # the last column block is partial
    main_loop = fused_loop_vec if vectorised else fused_loop_scalar

    @T.function
    def partials(
        S: T.Tensor((n, n), "bfloat16", align=16),
        PR: T.Tensor((ncb, n, 2), "float32", align=16),
        PC: T.Tensor((nrb, n, 2), "float32", align=16),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(ncb, nrb), block=(THREADS,)),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=CTAS_PER_SM),
        )
        cb = T.cuda.block_idx("x")
        rb = T.cuda.block_idx("y")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        x = T.alloc_local((RC, VPL), "float32")  # this step's RC x 16 elements
        raw = T.alloc_local((RC, SEG, VEC), "bfloat16")
        cm = T.alloc_local((VPL,), "float32")  # exact path: the columns' references
        cl = T.alloc_local((VPL,), "float32")  # the columns' sums
        rm = T.alloc_local((RC,), "float32")  # the rows' maxima, then references b
        rs = T.alloc_local((RC,), "float32")  # the rows' sums
        fr = T.alloc_local((RC,), "float32")  # 2^(b - W): a row's factor for the columns
        tmp = T.alloc_local((RC,), "float32")
        w = T.alloc_local((2,), "float32")  # the warp's column reference W, and its next value
        pv = T.alloc_local((2,), "float32")
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        mh = T.alloc_local((TN // THREADS, WARPS), "float32")
        sl = T.alloc_shared((WARPS, TN), "float32")
        sw = T.alloc_shared((WARPS,), "float32")
        flag = T.alloc_shared((1,), "int32")
        r_begin = rb * rows
        r_end = T.min(n, r_begin + rows)
        nsteps = (r_end - r_begin + (STEP - 1)) // STEP
        nfull = (r_end - r_begin) // STEP  # steps whose rows are all in the block
        for k in T.unroll(VPL):
            cl[k] = T.float32(0)
        w[0] = T.float32(NEG)
        main_loop(
            x,
            raw,
            rm,
            rs,
            fr,
            tmp,
            w,
            pv,
            cl,
            S,
            PR,
            r_begin,
            r_end,
            nsteps,
            nfull,
            cb,
            warp,
            lane,
            n,
            nvec,
            col_tail,
        )
        # merge the warps' column states (W_w, l): every warp moves to the largest W
        for s in T.unroll(SEG):
            for j in T.unroll(VEC):
                sl[warp, s * (LANES * VEC) + lane * VEC + j] = cl[s * VEC + j]
        if lane == 0:
            sw[warp] = w[0]
        if tx == 0:
            flag[0] = 0
        T.cuda.cta_sync()
        w[1] = sw[0]
        for v in T.unroll(WARPS - 1):
            w[1] = _fmax(w[1], sw[v + 1])
        for h in T.unroll(TN // THREADS):
            st[1] = T.float32(0)
            for v in T.unroll(WARPS):
                st[1] = _fma(sl[v, h * THREADS + tx], T.exp2(sw[v] - w[1]), st[1])
            if cb * TN + h * THREADS + tx < n:
                PC[rb, cb * TN + h * THREADS + tx, 0] = w[1]
                PC[rb, cb * TN + h * THREADS + tx, 1] = st[1]
                if st[1] < T.float32(THRESH):
                    flag[0] = 1
        T.cuda.cta_sync()
        if flag[0] != 0:
            # exact path: every column its own reference (two exponentials per element)
            for k in T.unroll(VPL):
                cm[k] = T.float32(NEG)
                cl[k] = T.float32(0)
            for step in T.serial(nsteps):
                row0 = r_begin + step * STEP + warp * RC
                if vectorised:
                    load_direct(raw, S, row0, cb, lane, n, nvec, col_tail)
                unpack(x, raw, S, row0, r_end, cb, lane, n, nvec, vectorised, True)
                for k in T.unroll(VPL):
                    st[0] = cm[k]
                    for r in T.unroll(RC):
                        st[0] = _fmax(st[0], x[r, k] * T.float32(L2E))
                    st[1] = cl[k] * T.exp2(cm[k] - st[0])
                    for r in T.unroll(RC):
                        st[1] = st[1] + T.exp2(_fma(x[r, k], T.float32(L2E), T.float32(0) - st[0]))
                    cm[k] = st[0]
                    cl[k] = st[1]
            # through ``sl`` in two rounds: the references, then the sums
            for s in T.unroll(SEG):
                for j in T.unroll(VEC):
                    sl[warp, s * (LANES * VEC) + lane * VEC + j] = cm[s * VEC + j]
            T.cuda.cta_sync()
            for h in T.unroll(TN // THREADS):
                for v in T.unroll(WARPS):
                    mh[h, v] = sl[v, h * THREADS + tx]
            T.cuda.cta_sync()
            for s in T.unroll(SEG):
                for j in T.unroll(VEC):
                    sl[warp, s * (LANES * VEC) + lane * VEC + j] = cl[s * VEC + j]
            T.cuda.cta_sync()
            for h in T.unroll(TN // THREADS):
                st[0] = T.float32(NEG)
                st[1] = T.float32(0)
                for v in T.unroll(WARPS):
                    ot[0] = mh[h, v]
                    ot[1] = sl[v, h * THREADS + tx]
                    lse_combine(st, ot)
                if cb * TN + h * THREADS + tx < n:
                    PC[rb, cb * TN + h * THREADS + tx, 0] = st[0]
                    PC[rb, cb * TN + h * THREADS + tx, 1] = st[1]

    return partials


def _merge(n: int):
    ncb, nrb, _ = plan(n)
    kr, kc = _cdiv(ncb, MERGE_WARPS), _cdiv(nrb, MERGE_WARPS)

    @T.function
    def merge(
        S: T.Tensor((n, n), "bfloat16", align=16),
        PR: T.Tensor((ncb, n, 2), "float32", align=16),
        PC: T.Tensor((nrb, n, 2), "float32", align=16),
        Rows: T.Tensor((n,), "float32"),
        Cols: T.Tensor((n,), "float32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(_cdiv(n, LANES),), block=(MERGE_WARPS * 32,))
        )
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        pr = T.alloc_local((kr, 2), "float32")
        pc = T.alloc_local((kc, 2), "float32")
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        sm = T.alloc_shared((2, MERGE_WARPS, LANES, 2), "float32")
        i = T.min(bx * LANES + lane, n - 1)
        for t in T.unroll(kr):  # every load first
            pr[t, 0] = T.if_then_else(
                t * MERGE_WARPS + warp < ncb,
                PR[T.min(t * MERGE_WARPS + warp, ncb - 1), i, 0],
                T.float32(NEG),
            )
            pr[t, 1] = T.if_then_else(
                t * MERGE_WARPS + warp < ncb,
                PR[T.min(t * MERGE_WARPS + warp, ncb - 1), i, 1],
                T.float32(0),
            )
        for t in T.unroll(kc):
            pc[t, 0] = T.if_then_else(
                t * MERGE_WARPS + warp < nrb,
                PC[T.min(t * MERGE_WARPS + warp, nrb - 1), i, 0],
                T.float32(NEG),
            )
            pc[t, 1] = T.if_then_else(
                t * MERGE_WARPS + warp < nrb,
                PC[T.min(t * MERGE_WARPS + warp, nrb - 1), i, 1],
                T.float32(0),
            )
        st[0] = pr[0, 0]
        st[1] = pr[0, 1]
        for t in T.unroll(kr - 1):
            ot[0] = pr[t + 1, 0]
            ot[1] = pr[t + 1, 1]
            lse_combine(st, ot)
        sm[0, warp, lane, 0] = st[0]
        sm[0, warp, lane, 1] = st[1]
        st[0] = pc[0, 0]
        st[1] = pc[0, 1]
        for t in T.unroll(kc - 1):
            ot[0] = pc[t + 1, 0]
            ot[1] = pc[t + 1, 1]
            lse_combine(st, ot)
        sm[1, warp, lane, 0] = st[0]
        sm[1, warp, lane, 1] = st[1]
        T.cuda.cta_sync()
        if warp < 2:  # warp 0 the rows, warp 1 the columns
            st[0] = sm[warp, 0, lane, 0]
            st[1] = sm[warp, 0, lane, 1]
            for v in T.unroll(MERGE_WARPS - 1):
                ot[0] = sm[warp, v + 1, lane, 0]
                ot[1] = sm[warp, v + 1, lane, 1]
                lse_combine(st, ot)
            if bx * LANES + lane < n:
                if warp == 0:
                    Rows[i] = (st[0] + T.log2(st[1])) * T.float32(LN2) - T.Cast("float32", S[i, i])
                else:
                    Cols[i] = (st[0] + T.log2(st[1])) * T.float32(LN2) - T.Cast("float32", S[i, i])

    return merge


# ---------------------------------------------------------------------------
# unfused: the unfused program's four passes, each reading S again
# ---------------------------------------------------------------------------
def _row_pass(n: int, kind: str):
    """``max``: Mx[i] = max_j S[i, j]. ``sum``: Out[i] = lse(S[i, :]) - S[i, i] given Mx."""
    vectorised = vectorisable(n, "bfloat16")
    nvec = n // VEC
    blocks = min(n, SMS * ROW_CTAS_PER_SM)
    iters = _cdiv(nvec, ROW_THREADS * ROW_UNROLL)
    exact = nvec % (ROW_THREADS * ROW_UNROLL) == 0  # every load in the row
    is_max = kind == "max"

    @T.function
    def row_pass(
        S: T.Tensor((n, n), "bfloat16", align=16),
        Mx: T.Tensor((n,), "float32"),
        Out: T.Tensor((n,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(ROW_THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        xs = T.alloc_local((ROW_UNROLL, VEC), "float32")
        raw = T.alloc_local((ROW_UNROLL, VEC), "bfloat16")
        st = T.alloc_local((1,), "float32")
        ot = T.alloc_local((1,), "float32")
        bb = T.alloc_local((1,), "float32")
        smem = T.alloc_shared((ROW_WARPS, 1), "float32")
        row: T.int32 = bx
        while row < n:
            if is_max:
                st[0] = T.float32(NEG)
                bb[0] = T.float32(0)
            else:
                st[0] = T.float32(0)
                bb[0] = Mx[row] * T.float32(L2E)
            if vectorised:
                for it in T.serial(iters):
                    for u in T.unroll(ROW_UNROLL):
                        if exact:
                            raw[u, T.Ramp(0, 1, VEC)] = S[
                                row,
                                T.Ramp(((it * ROW_UNROLL + u) * ROW_THREADS + tx) * VEC, 1, VEC),
                            ]
                        else:
                            raw[u, T.Ramp(0, 1, VEC)] = S[
                                row,
                                T.Ramp(
                                    T.min((it * ROW_UNROLL + u) * ROW_THREADS + tx, nvec - 1) * VEC,
                                    1,
                                    VEC,
                                ),
                            ]
                    for u in T.unroll(ROW_UNROLL):
                        for j in T.unroll(VEC):
                            if exact:
                                xs[u, j] = T.Cast("float32", raw[u, j])
                            else:
                                xs[u, j] = T.if_then_else(
                                    (it * ROW_UNROLL + u) * ROW_THREADS + tx < nvec,
                                    T.Cast("float32", raw[u, j]),
                                    T.float32(NEG),
                                )
                    for u in T.unroll(ROW_UNROLL):
                        for j in T.unroll(VEC):
                            if is_max:
                                st[0] = _fmax(st[0], xs[u, j])
                            else:
                                st[0] = st[0] + T.exp2(
                                    _fma(xs[u, j], T.float32(L2E), T.float32(0) - bb[0])
                                )
            else:
                c: T.int32 = tx
                while c < n:
                    if is_max:
                        st[0] = _fmax(st[0], T.Cast("float32", S[row, c]))
                    else:
                        st[0] = st[0] + T.exp2(
                            _fma(T.Cast("float32", S[row, c]), T.float32(L2E), T.float32(0) - bb[0])
                        )
                    c += ROW_THREADS
            if is_max:
                cta_allreduce(st, ot, 1, max_combine, smem, warp, lane, ROW_WARPS)
            else:
                cta_allreduce(st, ot, 1, add_combine, smem, warp, lane, ROW_WARPS)
            if tx == 0:
                if is_max:
                    Mx[row] = st[0]
                else:
                    Out[row] = (bb[0] + T.log2(st[0])) * T.float32(LN2) - T.Cast(
                        "float32", S[row, row]
                    )
            row += blocks

    return row_pass


@T.inline
def col_step(
    x, raw, acc, b, S, row0, r_end, more, cb, lane, n, nvec, vectorised, col_tail, masked, is_max
):
    """One step of a warp in a column pass: RC rows into its 16 columns' maxima or sums."""
    unpack(x, raw, S, row0, r_end, cb, lane, n, nvec, vectorised, masked)
    if more:
        load_direct(raw, S, row0 + STEP, cb, lane, n, nvec, col_tail)
    for k in T.unroll(VPL):
        for r in T.unroll(RC):
            if is_max:
                acc[k] = _fmax(acc[k], x[r, k])
            else:
                acc[k] = acc[k] + T.exp2(_fma(x[r, k], T.float32(L2E), T.float32(0) - b[k]))


@T.inline
def col_loop_vec(
    x, raw, acc, b, S, r_begin, r_end, nsteps, nfull, cb, warp, lane, n, nvec, col_tail, is_max
):
    load_direct(raw, S, r_begin + warp * RC, cb, lane, n, nvec, col_tail)
    for step in T.serial(nsteps):
        row0 = r_begin + step * STEP + warp * RC
        if step < nfull:
            col_step(
                x,
                raw,
                acc,
                b,
                S,
                row0,
                r_end,
                step + 1 < nsteps,
                cb,
                lane,
                n,
                nvec,
                True,
                col_tail,
                col_tail,
                is_max,
            )
        else:
            col_step(
                x,
                raw,
                acc,
                b,
                S,
                row0,
                r_end,
                step + 1 < nsteps,
                cb,
                lane,
                n,
                nvec,
                True,
                col_tail,
                True,
                is_max,
            )


@T.inline
def col_loop_scalar(
    x, raw, acc, b, S, r_begin, r_end, nsteps, nfull, cb, warp, lane, n, nvec, col_tail, is_max
):
    for step in T.serial(nsteps):
        row0 = r_begin + step * STEP + warp * RC
        col_step(
            x, raw, acc, b, S, row0, r_end, False, cb, lane, n, nvec, False, col_tail, True, is_max
        )


def _col_pass(n: int, kind: str):
    """``max``: P[rb, j] = max of S[rows of rb, j]; ``sum``: Σ 2^((x - Cmax[j]) log2 e)."""
    ncb, nrb, rows = plan(n)
    vectorised = vectorisable(n, "bfloat16")
    nvec = n // VEC
    col_tail = n % TN != 0
    is_max = kind == "max"
    main_loop = col_loop_vec if vectorised else col_loop_scalar

    @T.function
    def col_pass(
        S: T.Tensor((n, n), "bfloat16", align=16),
        Cmax: T.Tensor((n,), "float32"),
        P: T.Tensor((nrb, n), "float32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(ncb, nrb), block=(THREADS,)),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=CTAS_PER_SM),
        )
        cb = T.cuda.block_idx("x")
        rb = T.cuda.block_idx("y")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        x = T.alloc_local((RC, VPL), "float32")
        raw = T.alloc_local((RC, SEG, VEC), "bfloat16")
        acc = T.alloc_local((VPL,), "float32")
        b = T.alloc_local((VPL,), "float32")
        sl = T.alloc_shared((WARPS, TN), "float32")
        r_begin = rb * rows
        r_end = T.min(n, r_begin + rows)
        nsteps = (r_end - r_begin + (STEP - 1)) // STEP
        nfull = (r_end - r_begin) // STEP
        for s in T.unroll(SEG):
            for j in T.unroll(VEC):
                if is_max:
                    acc[s * VEC + j] = T.float32(NEG)
                    b[s * VEC + j] = T.float32(0)
                else:
                    acc[s * VEC + j] = T.float32(0)
                    b[s * VEC + j] = T.if_then_else(
                        cb * TN + s * (LANES * VEC) + lane * VEC + j < n,
                        Cmax[T.min(cb * TN + s * (LANES * VEC) + lane * VEC + j, n - 1)]
                        * T.float32(L2E),
                        T.float32(0),
                    )
        main_loop(
            x,
            raw,
            acc,
            b,
            S,
            r_begin,
            r_end,
            nsteps,
            nfull,
            cb,
            warp,
            lane,
            n,
            nvec,
            col_tail,
            is_max,
        )
        for s in T.unroll(SEG):
            for j in T.unroll(VEC):
                sl[warp, s * (LANES * VEC) + lane * VEC + j] = acc[s * VEC + j]
        T.cuda.cta_sync()
        for h in T.unroll(TN // THREADS):
            acc[0] = sl[0, h * THREADS + tx]
            for v in T.unroll(WARPS - 1):
                if is_max:
                    acc[0] = _fmax(acc[0], sl[v + 1, h * THREADS + tx])
                else:
                    acc[0] = acc[0] + sl[v + 1, h * THREADS + tx]
            if cb * TN + h * THREADS + tx < n:
                P[rb, cb * TN + h * THREADS + tx] = acc[0]

    return col_pass


def _col_finish(n: int, kind: str):
    """``max``: Cmax[j] = max_rb Pm[rb, j]; ``out``: Cols[j] = lse(S[:, j]) - S[j, j]."""
    _, nrb, _ = plan(n)
    kc = _cdiv(nrb, MERGE_WARPS)
    is_max = kind == "max"

    @T.function
    def col_finish(
        S: T.Tensor((n, n), "bfloat16", align=16),
        Pm: T.Tensor((nrb, n), "float32"),
        Ps: T.Tensor((nrb, n), "float32"),
        Cmax: T.Tensor((n,), "float32"),
        Cols: T.Tensor((n,), "float32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(_cdiv(n, LANES),), block=(MERGE_WARPS * 32,))
        )
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        pv = T.alloc_local((kc,), "float32")
        acc = T.alloc_local((1,), "float32")
        sm = T.alloc_shared((MERGE_WARPS, LANES), "float32")
        j = T.min(bx * LANES + lane, n - 1)
        for t in T.unroll(kc):  # every load first
            if is_max:
                pv[t] = T.if_then_else(
                    t * MERGE_WARPS + warp < nrb,
                    Pm[T.min(t * MERGE_WARPS + warp, nrb - 1), j],
                    T.float32(NEG),
                )
            else:
                pv[t] = T.if_then_else(
                    t * MERGE_WARPS + warp < nrb,
                    Ps[T.min(t * MERGE_WARPS + warp, nrb - 1), j],
                    T.float32(0),
                )
        acc[0] = pv[0]
        for t in T.unroll(kc - 1):
            if is_max:
                acc[0] = _fmax(acc[0], pv[t + 1])
            else:
                acc[0] = acc[0] + pv[t + 1]
        sm[warp, lane] = acc[0]
        T.cuda.cta_sync()
        if warp == 0:
            acc[0] = sm[0, lane]
            for v in T.unroll(MERGE_WARPS - 1):
                if is_max:
                    acc[0] = _fmax(acc[0], sm[v + 1, lane])
                else:
                    acc[0] = acc[0] + sm[v + 1, lane]
            if bx * LANES + lane < n:
                if is_max:
                    Cmax[j] = acc[0]
                else:
                    Cols[j] = (Cmax[j] * T.float32(L2E) + T.log2(acc[0])) * T.float32(LN2) - T.Cast(
                        "float32", S[j, j]
                    )

    return col_finish


@cached
def build(cfg: dict, variant: str = "fused"):
    n = cfg["n"]
    ncb, nrb, _ = plan(n)
    dev = torch.device("cuda")
    if variant == "fused":
        part = compile_sm90(_partials(n), name=f"contrastive_partials_{n}")
        merge = compile_sm90(_merge(n), name=f"contrastive_merge_{n}")
        pr = torch.empty((ncb, n, 2), device=dev, dtype=F32)  # workspaces, cached with the build
        pc = torch.empty((nrb, n, 2), device=dev, dtype=F32)

        def run(S):
            rows = torch.empty(n, device=S.device, dtype=F32)
            cols = torch.empty(n, device=S.device, dtype=F32)
            part(S, pr, pc)
            merge(S, pr, pc, rows, cols)
            return (rows, cols)

        return run
    assert variant == "unfused", variant
    row_max = compile_sm90(_row_pass(n, "max"), name=f"contrastive_row_max_{n}")
    row_sum = compile_sm90(_row_pass(n, "sum"), name=f"contrastive_row_sum_{n}")
    col_max = compile_sm90(_col_pass(n, "max"), name=f"contrastive_col_max_{n}")
    col_sum = compile_sm90(_col_pass(n, "sum"), name=f"contrastive_col_sum_{n}")
    col_max_merge = compile_sm90(_col_finish(n, "max"), name=f"contrastive_col_max_merge_{n}")
    col_out = compile_sm90(_col_finish(n, "out"), name=f"contrastive_col_out_{n}")
    rmax = torch.empty(n, device=dev, dtype=F32)
    cmax = torch.empty(n, device=dev, dtype=F32)
    pm = torch.empty((nrb, n), device=dev, dtype=F32)
    ps = torch.empty((nrb, n), device=dev, dtype=F32)

    def run_unfused(S):
        rows = torch.empty(n, device=S.device, dtype=F32)
        cols = torch.empty(n, device=S.device, dtype=F32)
        row_max(S, rmax, rows)  # pass 1: the rows' maxima
        row_sum(S, rmax, rows)  # pass 2: the rows' Σ exp, and their outputs
        col_max(S, cmax, pm)  # pass 3: the columns' maxima, per row block ...
        col_max_merge(S, pm, ps, cmax, cols)  # ... merged
        col_sum(S, cmax, ps)  # pass 4: the columns' Σ exp, per row block ...
        col_out(S, pm, ps, cmax, cols)  # ... summed, and the outputs
        return (rows, cols)

    return run_unfused
