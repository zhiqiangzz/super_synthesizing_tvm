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
"""Layer normalisation without affine weights, ``(x - mean) / sqrt(var + eps)``, in TIRx.

The algorithm is chain_ops' ``layernorm_fused``: one pass over the row with
the state ``(count, mean, M2)`` -- the number of elements, their mean, and
the sum of squared deviations from it -- merged pairwise by Chan's form of
Welford's update (``welford_combine``); then the normalise pass,
``y = (x - mean) * rsqrt(M2 / m + eps)``. All of it in float32; the output is
rounded to bfloat16 only at the store.

A thread first reduces the elements it holds in registers -- up to ``K``
16-byte vectors -- to a state of its own by two passes over the registers
(sum, then squared deviations from that mean), and merges that into what it
carries; so there is one division per tile of ``8 K`` elements, not one per
element. ``cta_allreduce`` merges the threads' states.

Three layouts, picked when the kernel is built (``_plan``):

``resident`` (rows of up to 65536 elements, ``m % 8 == 0``; the short and long sizes)
    one CTA per row, one launch over all rows. Every thread loads its vectors of the
    row once, 16 bytes at a time, and keeps them on chip: the statistics and the
    normalise pass both read them there, so HBM is read once and written once. Up to 4
    vectors a thread live in registers (64 registers a thread at most, so 2048 threads
    fit an SM); longer rows take 512 threads with 2 vectors in registers and up to 14 in
    the thread's own slots of shared memory (each merged as a state of 8), so that two
    rows share an SM and one's loads cover the other's reductions.
``rounds`` (few rows, each long enough to spread over the whole grid; the split size)
    one cooperative launch of 4 CTAs per SM, all resident, working on one row at a
    time: CTA ``c`` holds its slice of the row on chip (registers, then shared memory
    filled by ``cp.async``), publishes the slice's state, and waits at the row's
    barrier (an atomic counter that only grows, so it needs no reset); then it merges
    the row's states, normalises its slice, and, slot by slot, loads the next row's
    slice in its place, so one row's stores overlap the next row's loads. HBM is read
    once and written once; what is left is one barrier per row.
``split`` (everything else: unaligned rows, rows too long for either)
    every row cut into ``C`` pieces, one CTA each. ``partial`` reduces each piece to a
    state, ``(n, C, 3)`` float32; ``finish`` merges a row's ``C`` states (every CTA of
    the row in the same order, so they agree), then reads its piece again, newest tile
    first (L2 may still hold it), and normalises it. Unaligned rows take the same
    structure with one-element loads.

``variant="unfused"`` is the ablation: the unfused chain's three passes -- the mean,
then ``Σ (x - mean)²``, then the output -- as three launches that each read X again,
with only per-row partial sums between them, on the same grid and tiles.
"""

from __future__ import annotations

import math

import torch
from bench.ops import LN_EPS

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90, ctas_per_sm
from .common.reduce import cta_allreduce

VEC = 8  # bfloat16 elements per 16-byte access
SPLIT_THREADS = 512
SPLIT_K = 4  # vectors (or groups of 8 elements) per thread per tile
SPLIT_CTAS_PER_SM = 4
# No cache operator on a 16-byte access. A flag handed to an inline function arrives as a TIR
# constant: both branches of vld / vst are built (the if folds away later), so every caller
# passes a uint32 view of its tile.
PLAIN = False

KERNEL_META = {
    "op": "layernorm",
    "dtype": "bfloat16 in, bfloat16 out, float32 statistics",
    "algorithm": (
        "one pass, state (count, mean, M2) merged by Chan/Welford; "
        "then y = (x - mean) * rsqrt(M2/m + eps)"
    ),
    "structure": (
        "resident (m <= 65536): one CTA per row, the row on chip (registers, plus shared "
        "memory past 4 vectors a thread), 16-byte loads, CTA all-reduce, one launch; rounds "
        "(few long rows): one cooperative launch, the grid on one row at a time, each CTA's "
        "slice on chip (cp.async), an atomic barrier per row; split (unaligned or longer "
        "rows): partial states, then merge and normalise"
    ),
}


def _pow2(x: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1, x))))


@T.inline
def welford_combine(st, ot):
    """(c, mean, M2) <- (c, mean, M2) ⊕ (c', mean', M2'); either side may be empty (c = 0)."""
    cnt = st[0] + ot[0]
    w = ot[0] / T.max(cnt, T.float32(1))
    d = ot[1] - st[1]
    st[2] = st[2] + ot[2] + d * d * st[0] * w
    st[1] = st[1] + d * w
    st[0] = cnt


@T.inline
def vld(xs, xv, k, X, row, v, cop):
    """``xs[k, :] = X[row, 8v : 8v + 8]``, one 16-byte load; with ``cop`` as ``ld.global.cs``
    (streaming: evict first, L2 need not keep it). ``xv`` is ``xs`` seen as uint32, which the
    PTX form writes."""
    if cop:
        T.ptx.ld.global_.cs.v4.b32(xv[k, 0], xv[k, 1], xv[k, 2], xv[k, 3], X.ptr_to([row, v * VEC]))
    else:
        xs[k, T.Ramp(0, 1, VEC)] = X[row, T.Ramp(v * VEC, 1, VEC)]


@T.inline
def vst(Y, row, v, xs, xv, k, cop):
    """``Y[row, 8v : 8v + 8] = xs[k, :]``, one 16-byte store; with ``cop`` as ``st.global.cs``."""
    if cop:
        T.ptx.st.global_.cs.v4.b32(Y.ptr_to([row, v * VEC]), xv[k, 0], xv[k, 1], xv[k, 2], xv[k, 3])
    else:
        Y[row, T.Ramp(v * VEC, 1, VEC)] = xs[k, T.Ramp(0, 1, VEC)]


@T.inline
def tile_load(xs, xv, X, row, base, hi, tx, threads, K, vectorised, cop):
    """This thread's slots of the tile at ``base``: vector ``base + k*threads + tx`` in slot ``k``
    (vectorised), or element ``base + (k*VEC + i)*threads + tx`` in slot ``(k, i)``."""
    for k in T.unroll(K):
        if vectorised:
            if base + k * threads + tx < hi:
                vld(xs, xv, k, X, row, base + k * threads + tx, cop)
        else:
            for i in T.unroll(VEC):
                if base + (k * VEC + i) * threads + tx < hi:
                    xs[k, i] = X[row, base + (k * VEC + i) * threads + tx]


@T.inline
def tile_stats(ot, xs, base, hi, tx, threads, K, vectorised):
    """``ot`` <- (count, mean, M2) of this thread's loaded slots, two passes over registers."""
    ot[0] = T.float32(0)
    ot[1] = T.float32(0)
    ot[2] = T.float32(0)
    for k in T.unroll(K):
        if vectorised:
            if base + k * threads + tx < hi:
                ot[0] = ot[0] + T.float32(VEC)
                for i in T.unroll(VEC):
                    ot[1] = ot[1] + T.Cast("float32", xs[k, i])
        else:
            for i in T.unroll(VEC):
                if base + (k * VEC + i) * threads + tx < hi:
                    ot[0] = ot[0] + T.float32(1)
                    ot[1] = ot[1] + T.Cast("float32", xs[k, i])
    ot[1] = ot[1] / T.max(ot[0], T.float32(1))
    for k in T.unroll(K):
        if vectorised:
            if base + k * threads + tx < hi:
                for i in T.unroll(VEC):
                    ot[2] = ot[2] + (T.Cast("float32", xs[k, i]) - ot[1]) * (
                        T.Cast("float32", xs[k, i]) - ot[1]
                    )
        else:
            for i in T.unroll(VEC):
                if base + (k * VEC + i) * threads + tx < hi:
                    ot[2] = ot[2] + (T.Cast("float32", xs[k, i]) - ot[1]) * (
                        T.Cast("float32", xs[k, i]) - ot[1]
                    )


@T.inline
def tile_normalise(Y, xs, xv, row, base, hi, tx, threads, K, vectorised, mean, rstd, cop):
    """Store ``(x - mean) * rstd`` of this thread's slots, rounded to bfloat16."""
    for k in T.unroll(K):
        if vectorised:
            if base + k * threads + tx < hi:
                for i in T.unroll(VEC):
                    xs[k, i] = T.Cast("bfloat16", (T.Cast("float32", xs[k, i]) - mean) * rstd)
                vst(Y, row, base + k * threads + tx, xs, xv, k, cop)
        else:
            for i in T.unroll(VEC):
                if base + (k * VEC + i) * threads + tx < hi:
                    Y[row, base + (k * VEC + i) * threads + tx] = T.Cast(
                        "bfloat16", (T.Cast("float32", xs[k, i]) - mean) * rstd
                    )


@T.inline
def vec_stats(ot, vs):
    """``ot`` <- (8, mean, M2) of one vector of 8 elements."""
    ot[0] = T.float32(VEC)
    ot[1] = T.float32(0)
    ot[2] = T.float32(0)
    for i in T.unroll(VEC):
        ot[1] = ot[1] + T.Cast("float32", vs[0, i])
    ot[1] = ot[1] * T.float32(1.0 / VEC)
    for i in T.unroll(VEC):
        ot[2] = ot[2] + (T.Cast("float32", vs[0, i]) - ot[1]) * (
            T.Cast("float32", vs[0, i]) - ot[1]
        )


@T.inline
def vec_normalise(vs, mean, rstd):
    for i in T.unroll(VEC):
        vs[0, i] = T.Cast("bfloat16", (T.Cast("float32", vs[0, i]) - mean) * rstd)


@T.inline
def finalise(st, m):
    """(c, mean, M2) -> (c, mean, rsqrt(M2 / m + eps)) in place."""
    st[2] = T.rsqrt(st[2] / T.float32(m) + T.float32(LN_EPS))


@T.inline
def add1(st, ot):
    st[0] = st[0] + ot[0]


@T.inline
def add2(st, ot):
    st[0] = st[0] + ot[0]
    st[1] = st[1] + ot[1]


@T.inline
def tile_moment(acc, xs, base, hi, tx, threads, K, vectorised, mean, square):
    """``acc[0] += Σ x`` (``square`` false) or ``Σ (x - mean)²`` over this thread's slots."""
    for k in T.unroll(K):
        if vectorised:
            if base + k * threads + tx < hi:
                for i in T.unroll(VEC):
                    if square:
                        acc[0] = acc[0] + (T.Cast("float32", xs[k, i]) - mean) * (
                            T.Cast("float32", xs[k, i]) - mean
                        )
                    else:
                        acc[0] = acc[0] + T.Cast("float32", xs[k, i])
        else:
            for i in T.unroll(VEC):
                if base + (k * VEC + i) * threads + tx < hi:
                    if square:
                        acc[0] = acc[0] + (T.Cast("float32", xs[k, i]) - mean) * (
                            T.Cast("float32", xs[k, i]) - mean
                        )
                    else:
                        acc[0] = acc[0] + T.Cast("float32", xs[k, i])


# ---------------------------------------------------------------------------
# resident: one CTA per row, the row on chip
# ---------------------------------------------------------------------------
def _resident(
    n: int,
    m: int,
    threads: int,
    kr: int,
    ks: int,
    blocks: int,
    minb: int = 1,
    ld: bool = False,
    st_: bool = False,
):
    """Thread ``tx`` holds vectors ``k*threads + tx``: ``k < kr`` in registers, the next ``ks``
    in its own slots of shared memory (``S[k - kr, tx]``, read back only by itself).
    ``minb`` CTAs must fit on an SM at once (``__launch_bounds__``' second argument);
    ``ld`` / ``st_`` are the cache operators of the loads of X and the stores of Y."""
    nvec = m // VEC
    warps = threads // 32
    s_shape = (ks, threads, VEC) if ks else (1, 1, VEC)
    attrs = T.cuda.KernelAttributes(min_blocks_per_sm=minb)

    @T.function
    def layernorm_resident(
        X: T.Tensor((n, m), "bfloat16", align=16), Y: T.Tensor((n, m), "bfloat16", align=16)
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(blocks,), block=(threads,)), kernel_attrs=attrs
        )
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((3,), "float32")
        ot = T.alloc_local((3,), "float32")
        xs = T.alloc_local((kr, VEC), "bfloat16")
        xv = xs.view("uint32")
        vs = T.alloc_local((1, VEC), "bfloat16")
        vv = vs.view("uint32")
        S = T.alloc_tensor(ty_args=[T.Tensor(s_shape, "bfloat16", scope="shared.dyn")])
        smem = T.alloc_shared((warps, 3), "float32")
        row: T.int32 = bx
        while row < n:
            tile_load(xs, xv, X, row, 0, nvec, tx, threads, kr, True, ld)
            for k in T.unroll(ks):
                if (kr + k) * threads + tx < nvec:
                    vld(vs, vv, 0, X, row, (kr + k) * threads + tx, ld)
                    S[k, tx, T.Ramp(0, 1, VEC)] = vs[0, T.Ramp(0, 1, VEC)]
            tile_stats(st, xs, 0, nvec, tx, threads, kr, True)
            for k in T.unroll(ks):
                if (kr + k) * threads + tx < nvec:
                    vs[0, T.Ramp(0, 1, VEC)] = S[k, tx, T.Ramp(0, 1, VEC)]
                    vec_stats(ot, vs)
                    welford_combine(st, ot)
            cta_allreduce(st, ot, 3, welford_combine, smem, warp, lane, warps)
            finalise(st, m)
            tile_normalise(Y, xs, xv, row, 0, nvec, tx, threads, kr, True, st[1], st[2], st_)
            for k in T.unroll(ks):
                if (kr + k) * threads + tx < nvec:
                    vs[0, T.Ramp(0, 1, VEC)] = S[k, tx, T.Ramp(0, 1, VEC)]
                    vec_normalise(vs, st[1], st[2])
                    vst(Y, row, (kr + k) * threads + tx, vs, vv, 0, st_)
            row += blocks

    return layernorm_resident


# ---------------------------------------------------------------------------
# rounds: the whole grid on one row at a time; one launch, a barrier per row
# ---------------------------------------------------------------------------
def _rounds(
    n: int,
    m: int,
    ctas: int,
    threads: int,
    kr: int,
    ks: int,
    per_sm: int,
    st_cs: bool,
    ld_cs: bool,
):
    """The whole grid works on one row at a time: CTA ``c`` holds vectors
    ``[c*chunk, (c+1)*chunk)`` of the row on chip (``kr`` per thread in registers, ``ks`` in
    its own slots of shared memory, filled by ``cp.async``), so a row is read from HBM once.
    After the row's barrier the CTA normalises its slice and, slot by slot, loads the next
    row's slice in its place: the stores of one row and the loads of the next overlap."""
    units = m // VEC
    warps = threads // 32
    chunk = math.ceil(units / ctas)
    assert (kr + ks) * threads >= chunk, (kr, ks, threads, chunk)
    s_shape = (ks * threads, VEC) if ks else (1, VEC)
    attrs = T.cuda.KernelAttributes(min_blocks_per_sm=per_sm)

    @T.function
    def layernorm_rounds(
        X: T.Tensor((n, m), "bfloat16", align=16),
        Y: T.Tensor((n, m), "bfloat16", align=16),
        P: T.Tensor((n, ctas, 3), "float32"),
        Cnt: T.Tensor((n,), "int32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(ctas,), block=(threads,), cooperative=True),
            kernel_attrs=attrs,
        )
        c = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((3,), "float32")
        ot = T.alloc_local((3,), "float32")
        xs = T.alloc_local((max(kr, 1), VEC), "bfloat16")
        xv = xs.view("uint32")
        vs = T.alloc_local((1, VEC), "bfloat16")
        vv = vs.view("uint32")
        old = T.alloc_local((1,), "int32")
        spin = T.alloc_local((1,), "int32")
        S = T.alloc_tensor(
            ty_args=[T.Tensor(s_shape, "bfloat16", scope="shared.dyn", layout="default")]
        )
        smem = T.alloc_shared((warps, 3), "float32")
        lo = c * chunk
        hi = T.min(lo + chunk, units)
        # row 0 on its way: registers by plain loads, shared memory by cp.async (whose source
        # region must be provably in bounds: the clamps, never taken under the guards, prove it)
        tile_load(xs, xv, X, 0, lo, hi, tx, threads, kr, True, ld_cs)
        for k in T.unroll(ks):
            if lo + (kr + k) * threads + tx < hi:
                T.cuda.tile.apply(
                    T.ptx["cp.async.cg.shared.global"],
                    S[k * threads + tx, 0:VEC],
                    X[
                        0,
                        T.min(lo + (kr + k) * threads + tx, units - 1) * VEC : T.min(
                            lo + (kr + k) * threads + tx, units - 1
                        )
                        * VEC
                        + VEC,
                    ],
                    chunk_bytes=16,
                )
        T.ptx.cp.async_.commit_group()
        r: T.int32 = 0
        while r < n:
            T.ptx.cp.async_.wait_group(0)  # this thread's slots of row r have landed
            tile_stats(st, xs, lo, hi, tx, threads, kr, True)
            for k in T.unroll(ks):
                if lo + (kr + k) * threads + tx < hi:
                    vs[0, T.Ramp(0, 1, VEC)] = S[k * threads + tx, T.Ramp(0, 1, VEC)]
                    vec_stats(ot, vs)
                    welford_combine(st, ot)
            cta_allreduce(st, ot, 3, welford_combine, smem, warp, lane, warps)
            # row r's barrier: publish this slice's state, wait for the other ctas - 1.
            # Cnt only grows: it is a multiple of ctas when a launch starts.
            if tx == 0:
                P[r, c, 0] = st[0]
                P[r, c, 1] = st[1]
                P[r, c, 2] = st[2]
                T.ptx.atom.release.gpu.global_.add.s32(old[0], Cnt.ptr_to([r]), T.int32(1))
                spin[0] = old[0] + 1
                T.cuda.wait_until(
                    spin[0],
                    Cnt.ptr_to([r]),
                    predicate=lambda v: v >= (old[0] // ctas + 1) * ctas,
                    scope="gpu",
                )
            T.cuda.cta_sync()
            st[0] = T.float32(0)
            st[1] = T.float32(0)
            st[2] = T.float32(0)
            j: T.int32 = tx
            while j < ctas:
                ot[0] = P[r, j, 0]
                ot[1] = P[r, j, 1]
                ot[2] = P[r, j, 2]
                welford_combine(st, ot)
                j += threads
            cta_allreduce(st, ot, 3, welford_combine, smem, warp, lane, warps)
            finalise(st, m)
            # normalise row r; each slot, once stored, takes row r + 1
            tile_normalise(Y, xs, xv, r, lo, hi, tx, threads, kr, True, st[1], st[2], st_cs)
            if r + 1 < n:
                tile_load(xs, xv, X, r + 1, lo, hi, tx, threads, kr, True, ld_cs)
            for k in T.unroll(ks):
                if lo + (kr + k) * threads + tx < hi:
                    vs[0, T.Ramp(0, 1, VEC)] = S[k * threads + tx, T.Ramp(0, 1, VEC)]
                    vec_normalise(vs, st[1], st[2])
                    vst(Y, r, lo + (kr + k) * threads + tx, vs, vv, 0, st_cs)
                    if r + 1 < n:
                        T.cuda.tile.apply(
                            T.ptx["cp.async.cg.shared.global"],
                            S[k * threads + tx, 0:VEC],
                            X[
                                T.max(T.min(r + 1, n - 1), 0),
                                T.min(lo + (kr + k) * threads + tx, units - 1) * VEC : T.min(
                                    lo + (kr + k) * threads + tx, units - 1
                                )
                                * VEC
                                + VEC,
                            ],
                            chunk_bytes=16,
                        )
            T.ptx.cp.async_.commit_group()
            r += 1

    return layernorm_rounds


# ---------------------------------------------------------------------------
# split: C CTAs per row, partial states, then merge + normalise (two launches)
# ---------------------------------------------------------------------------
def _split_kernels(n: int, m: int, vectorised: bool, chunks: int, chunk: int, threads: int, K: int):
    warps = threads // 32
    units = m // VEC if vectorised else m  # vectors, or elements
    tile = threads * K if vectorised else threads * K * VEC  # units per CTA per step
    ctas = n * chunks

    @T.function
    def layernorm_partial(
        X: T.Tensor((n, m), "bfloat16", align=16), P: T.Tensor((n, chunks, 3), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(ctas,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((3,), "float32")
        ot = T.alloc_local((3,), "float32")
        xs = T.alloc_local((K, VEC), "bfloat16")
        xv = xs.view("uint32")
        smem = T.alloc_shared((warps, 3), "float32")
        row = bx // chunks
        c = bx % chunks
        hi = T.min(c * chunk + chunk, units)
        st[0] = T.float32(0)
        st[1] = T.float32(0)
        st[2] = T.float32(0)
        base: T.int32 = c * chunk
        while base < hi:
            tile_load(xs, xv, X, row, base, hi, tx, threads, K, vectorised, PLAIN)
            tile_stats(ot, xs, base, hi, tx, threads, K, vectorised)
            welford_combine(st, ot)
            base += tile
        cta_allreduce(st, ot, 3, welford_combine, smem, warp, lane, warps)
        if tx == 0:
            P[row, c, 0] = st[0]
            P[row, c, 1] = st[1]
            P[row, c, 2] = st[2]

    @T.function
    def layernorm_finish(
        X: T.Tensor((n, m), "bfloat16", align=16),
        P: T.Tensor((n, chunks, 3), "float32"),
        Y: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(ctas,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((3,), "float32")
        ot = T.alloc_local((3,), "float32")
        xs = T.alloc_local((K, VEC), "bfloat16")
        xv = xs.view("uint32")
        smem = T.alloc_shared((warps, 3), "float32")
        row = bx // chunks
        c = bx % chunks
        lo = c * chunk
        hi = T.min(lo + chunk, units)
        st[0] = T.float32(0)
        st[1] = T.float32(0)
        st[2] = T.float32(0)
        j: T.int32 = tx
        while j < chunks:
            ot[0] = P[row, j, 0]
            ot[1] = P[row, j, 1]
            ot[2] = P[row, j, 2]
            welford_combine(st, ot)
            j += threads
        cta_allreduce(st, ot, 3, welford_combine, smem, warp, lane, warps)
        finalise(st, m)
        # the piece's tiles newest first: the last ones partial read may still be in L2
        base: T.int32 = lo + (hi - lo - 1) // tile * tile
        while base >= lo:
            tile_load(xs, xv, X, row, base, hi, tx, threads, K, vectorised, PLAIN)
            tile_normalise(
                Y, xs, xv, row, base, hi, tx, threads, K, vectorised, st[1], st[2], PLAIN
            )
            base -= tile

    return layernorm_partial, layernorm_finish


# ---------------------------------------------------------------------------
# unfused: the ablation, one launch per link of the unfused chain
# ---------------------------------------------------------------------------
def _unfused_kernels(
    n: int, m: int, vectorised: bool, chunks: int, chunk: int, threads: int, K: int
):
    """mean, then Σ (x - mean)², then the output: three launches, each reading X again.
    Only per-row partial sums ``(n, chunks)`` travel between them."""
    warps = threads // 32
    units = m // VEC if vectorised else m
    tile = threads * K if vectorised else threads * K * VEC
    ctas = n * chunks
    inv_m = 1.0 / m

    @T.function
    def layernorm_mean_pass(
        X: T.Tensor((n, m), "bfloat16", align=16), S1: T.Tensor((n, chunks), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(ctas,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        acc = T.alloc_local((1,), "float32")
        ot = T.alloc_local((1,), "float32")
        xs = T.alloc_local((K, VEC), "bfloat16")
        xv = xs.view("uint32")
        smem = T.alloc_shared((warps, 1), "float32")
        row = bx // chunks
        c = bx % chunks
        hi = T.min(c * chunk + chunk, units)
        acc[0] = T.float32(0)
        base: T.int32 = c * chunk
        while base < hi:
            tile_load(xs, xv, X, row, base, hi, tx, threads, K, vectorised, PLAIN)
            tile_moment(acc, xs, base, hi, tx, threads, K, vectorised, T.float32(0), False)
            base += tile
        cta_allreduce(acc, ot, 1, add1, smem, warp, lane, warps)
        if tx == 0:
            S1[row, c] = acc[0]

    @T.function
    def layernorm_ss_pass(
        X: T.Tensor((n, m), "bfloat16", align=16),
        S1: T.Tensor((n, chunks), "float32"),
        S2: T.Tensor((n, chunks), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(ctas,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        acc = T.alloc_local((1,), "float32")
        mu = T.alloc_local((1,), "float32")
        ot = T.alloc_local((1,), "float32")
        xs = T.alloc_local((K, VEC), "bfloat16")
        xv = xs.view("uint32")
        smem = T.alloc_shared((warps, 1), "float32")
        row = bx // chunks
        c = bx % chunks
        hi = T.min(c * chunk + chunk, units)
        mu[0] = T.float32(0)
        j: T.int32 = tx
        while j < chunks:
            mu[0] = mu[0] + S1[row, j]
            j += threads
        cta_allreduce(mu, ot, 1, add1, smem, warp, lane, warps)
        mu[0] = mu[0] * T.float32(inv_m)
        acc[0] = T.float32(0)
        base: T.int32 = c * chunk
        while base < hi:
            tile_load(xs, xv, X, row, base, hi, tx, threads, K, vectorised, PLAIN)
            tile_moment(acc, xs, base, hi, tx, threads, K, vectorised, mu[0], True)
            base += tile
        cta_allreduce(acc, ot, 1, add1, smem, warp, lane, warps)
        if tx == 0:
            S2[row, c] = acc[0]

    @T.function
    def layernorm_out_pass(
        X: T.Tensor((n, m), "bfloat16", align=16),
        S1: T.Tensor((n, chunks), "float32"),
        S2: T.Tensor((n, chunks), "float32"),
        Y: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(ctas,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((2,), "float32")
        ot = T.alloc_local((2,), "float32")
        xs = T.alloc_local((K, VEC), "bfloat16")
        xv = xs.view("uint32")
        smem = T.alloc_shared((warps, 2), "float32")
        row = bx // chunks
        c = bx % chunks
        hi = T.min(c * chunk + chunk, units)
        st[0] = T.float32(0)
        st[1] = T.float32(0)
        j: T.int32 = tx
        while j < chunks:
            st[0] = st[0] + S1[row, j]
            st[1] = st[1] + S2[row, j]
            j += threads
        cta_allreduce(st, ot, 2, add2, smem, warp, lane, warps)
        st[0] = st[0] * T.float32(inv_m)
        st[1] = T.rsqrt(st[1] * T.float32(inv_m) + T.float32(LN_EPS))
        base: T.int32 = c * chunk
        while base < hi:
            tile_load(xs, xv, X, row, base, hi, tx, threads, K, vectorised, PLAIN)
            tile_normalise(
                Y, xs, xv, row, base, hi, tx, threads, K, vectorised, st[0], st[1], PLAIN
            )
            base += tile

    return layernorm_mean_pass, layernorm_ss_pass, layernorm_out_pass


# ---------------------------------------------------------------------------
# plans: which layout, and its sizes
# ---------------------------------------------------------------------------
def _sms() -> int:
    """SMs of the current device: the rounds layout's grid must all be resident at once."""
    return min(
        SMS, torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    )


def _resident_shape(m: int):
    """(threads, vectors per thread in registers, in shared memory, CTAs per SM) of a row
    held on chip by one CTA, or None. Up to 4 vectors (32 elements) per thread stay in
    registers; past that, 512 threads keep 2 in registers and the rest (up to 14, 112 KB)
    in shared memory, so that two CTAs -- two rows -- share an SM and one's loads cover
    the other's reductions."""
    if m % VEC:
        return None
    nvec = m // VEC
    threads = min(512, max(64, _pow2(math.ceil(nvec / 4))))
    K = math.ceil(nvec / threads)
    if K <= 4:
        return threads, K, 0, max(1, 1024 // threads)  # at most 64 registers a thread
    if K <= 16:
        return threads, 2, K - 2, 2
    return None


def _split_plan(n: int, m: int, threads: int = SPLIT_THREADS, K: int = SPLIT_K) -> dict:
    vectorised = m % VEC == 0
    units = m // VEC if vectorised else m
    tile = threads * K * (1 if vectorised else VEC)
    target = max(1, math.ceil(SMS * SPLIT_CTAS_PER_SM / n))  # pieces per row to fill the GPU
    chunk = math.ceil(math.ceil(units / target) / tile) * tile  # units per piece, whole tiles
    chunks = math.ceil(units / chunk)
    return {
        "kind": "split",
        "vectorised": vectorised,
        "chunks": chunks,
        "chunk": chunk,
        "threads": threads,
        "K": K,
    }


def _rounds_plan(
    n: int, m: int, sms: int, threads: int = 256, per_sm: int = 4, kr: int = 4
) -> dict | None:
    """The rounds layout, or None when a row does not fit on chip across the grid (or is too
    short to give every thread a vector): ``kr`` vectors per thread in registers, the rest
    of the slice in shared memory, at most 224 KB of it per SM."""
    if m % VEC:
        return None
    ctas = sms * per_sm
    chunk = math.ceil(m // VEC / ctas)
    slots = math.ceil(chunk / threads)
    if chunk < threads or (slots - min(kr, slots)) * threads * 16 * per_sm > 224 * 1024:
        return None
    kr = min(kr, slots)
    return {
        "kind": "rounds",
        "ctas": ctas,
        "threads": threads,
        "kr": kr,
        "ks": slots - kr,
        "per_sm": per_sm,
    }


def _plan(n: int, m: int, variant: str = "fused") -> dict:
    """The layout for ``(n, m)``: few rows long enough to spread over the grid take
    ``rounds``; rows a CTA can hold take ``resident``; the rest (unaligned rows, rows too
    long for either) the two-launch ``split``. ``unfused`` reuses the grid and tiles."""
    sms = _sms()
    shape = _resident_shape(m)
    rounds = _rounds_plan(n, m, sms) if n < sms else None
    if variant == "unfused":
        if shape is not None and rounds is None:
            threads, K = shape[0], shape[1] + shape[2]
            return {
                "kind": "unfused",
                "vectorised": True,
                "chunks": 1,
                "chunk": m // VEC,
                "threads": threads,
                "K": min(K, 4),
            }
        return dict(_split_plan(n, m), kind="unfused")
    if rounds is not None:
        return rounds
    if shape is not None:
        threads, kr, ks, minb = shape
        return {
            "kind": "resident",
            "threads": threads,
            "kr": kr,
            "ks": ks,
            "minb": minb,
            "blocks": n,
        }
    return _split_plan(n, m)


def _build(n: int, m: int, plan: dict):
    """``run(X) -> (Y,)`` for an explicit plan: ``build`` takes ``_plan``'s, tuning others."""
    kind = plan["kind"]
    tag = f"{n}x{m}"
    if kind == "resident":
        mod = compile_sm90(
            _resident(
                n,
                m,
                plan["threads"],
                plan["kr"],
                plan["ks"],
                plan["blocks"],
                plan.get("minb", 1),
                plan.get("ld", False),
                plan.get("st", False),
            ),
            name=f"layernorm_resident_{tag}",
        )

        def run(X):
            Y = torch.empty_like(X)
            mod(X, Y)
            return (Y,)

        return run

    if kind == "rounds":
        ctas = plan["ctas"]
        mod = compile_sm90(
            _rounds(
                n,
                m,
                ctas,
                plan["threads"],
                plan["kr"],
                plan["ks"],
                plan["per_sm"],
                plan.get("st", False),
                plan.get("ld", False),
            ),
            name=f"layernorm_rounds_{tag}",
        )
        # the row barriers need the whole grid resident at once (the cooperative launch
        # refuses otherwise): if per_sm CTAs do not fit an SM, take the two-launch split
        smem = max(plan["ks"], 1) * plan["threads"] * VEC * 2
        if min(ctas_per_sm(mod, plan["threads"], smem).values()) < plan["per_sm"]:
            return _build(n, m, _split_plan(n, m))
        part = torch.empty((n, ctas, 3), device="cuda", dtype=torch.float32)
        count = torch.zeros((n,), device="cuda", dtype=torch.int32)  # only grows: no reset per call

        def run(X):
            Y = torch.empty_like(X)
            mod(X, Y, part, count)
            return (Y,)

        return run

    chunks, chunk, threads, K = plan["chunks"], plan["chunk"], plan["threads"], plan["K"]
    if kind == "split":
        partial, finish = _split_kernels(n, m, plan["vectorised"], chunks, chunk, threads, K)
        mod_p = compile_sm90(partial, name=f"layernorm_partial_{tag}")
        mod_f = compile_sm90(finish, name=f"layernorm_finish_{tag}")
        part = torch.empty((n, chunks, 3), device="cuda", dtype=torch.float32)

        def run(X):
            Y = torch.empty_like(X)
            mod_p(X, part)
            mod_f(X, part, Y)
            return (Y,)

        return run

    assert kind == "unfused", kind
    k_mean, k_ss, k_out = _unfused_kernels(n, m, plan["vectorised"], chunks, chunk, threads, K)
    mod_1 = compile_sm90(k_mean, name=f"layernorm_mean_pass_{tag}")
    mod_2 = compile_sm90(k_ss, name=f"layernorm_ss_pass_{tag}")
    mod_3 = compile_sm90(k_out, name=f"layernorm_out_pass_{tag}")
    s1 = torch.empty((n, chunks), device="cuda", dtype=torch.float32)
    s2 = torch.empty((n, chunks), device="cuda", dtype=torch.float32)

    def run(X):
        Y = torch.empty_like(X)
        mod_1(X, s1)
        mod_2(X, s1, s2)
        mod_3(X, s1, s2, Y)
        return (Y,)

    return run


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, m = cfg["n"], cfg["m"]
    return _build(n, m, _plan(n, m, variant))
