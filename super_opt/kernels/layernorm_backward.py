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
"""The backward pass of layer normalisation, statistics recomputed from x, in TIRx.

``dx = (g - mean(g) - x̂ mean(g x̂)) / sigma`` with ``x̂ = (x - mean) / sigma`` and
``sigma = sqrt(var + 1e-5)``; X and G are bf16 ``(n, m)``, dx is bf16.

The algorithm is chain_ops' ``layernorm_backward_fused`` ("hoist", 5 states).
``mean(g x̂)`` needs the final mean and sigma, so the reducer hoists it: the
state is ``(count, mean, ss, Σ g, Σ g (x - mean))``, where ``ss = Σ (x -
mean)²`` and the last is a co-moment centred, like ``ss``, on the state's own
mean. Merging two states (``ln_combine``) is Chan's update of the mean and
``ss``, and re-centres both co-moments on the merged mean:
``Σ g (x - mean') = Σ g (x - mean) - (mean' - mean) Σ g``. A thread's leaf
state is that of the elements it holds, computed in two passes over its
registers (the sum, then the centred sums). From the row's state the epilogue
is ``dx = rs (g - Σg/m - (x - mean) Σ g (x - mean) rs² / m)``, ``rs = 1 / sigma``.
Everything inside is float32.

How the fused kernel is laid out (256-thread CTAs, 8 warps):

* ``held`` (rows of up to 8 · 8192 elements, n large enough): a row is held on
  chip by a cluster of ``c`` CTAs (``c = 1``: one CTA, no cluster), each with
  a slice of at most 8192 elements: each thread loads its 1-4 vectors of X and
  of G once, 16 bytes at a time, and keeps them in registers. Leaf state,
  ``cta_allreduce``, then (``c > 1``) every CTA reads the ``c`` CTA states from
  the cluster's shared memory (DSMEM) and merges them in rank order, so all
  get the same total; the thread writes dx from its registers with 16-byte
  stores. One kernel; X and G are read once: the floor's traffic.
* ``stream`` (longer rows, n ≥ 2 · SMs): one CTA per row; the thread folds 4
  vectors at a time into its state, then reads the row again to write dx.
* ``split`` (long rows, few of them): each row is cut into ``s`` segments, one
  CTA each. Kernel 1 folds its segment like ``stream`` and writes the
  segment's state to a small ``(n, s, 5)`` float32 workspace; kernel 2 merges
  the row's ``s`` states (strided over the threads, then the all-reduce: every
  CTA of a row gets the same total) and writes dx for its segment, walking the
  segments in the reverse order so that its first reads hit what kernel 1 left
  in L2. Two kernels.

``variant="unfused"`` is the ablation: the unfused program's passes, one kernel
each, with the same loads, grid and reductions: Σx; Σ(x - mean)²; Σg;
Σ g x̂; then dx. Each pass re-reads from HBM what it needs; only per-row
sums travel between passes (the ``(n, s, 4)`` workspace).

When ``m`` is not a multiple of 8 the rows are not 16-byte aligned, and every
path reads and writes one element at a time instead (a "vector" of one).
"""

from __future__ import annotations

import torch
from bench.ops import LN_EPS

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce

THREADS = 256
WARPS = THREADS // 32
K = 5  # (count, mean, ss, Σ g, Σ g (x - mean))
HELD = 32  # elements of X (and of G) a thread keeps in registers, at most
BATCH = 32  # elements a thread folds at a time when the row is streamed
CLUSTER = 8  # CTAs a row may be spread over and still be held on chip (portable maximum)
CTAS_PER_SM = 8

KERNEL_META = {
    "op": "layernorm_backward",
    "dtype": "bfloat16",
    "algorithm": "one pass, state (count, mean, Σ(x-mean)², Σg, Σg(x-mean)), Chan merge "
    "re-centring the co-moment; then dx = rs (g - Σg/m - (x-mean) Σg(x-mean) rs²/m)",
    "structure": "rows held in registers by one CTA (m <= 8192) or a cluster of up to 8 "
    "CTAs merging their states through DSMEM (m <= 65536): one kernel, one read; very long "
    "rows split over CTAs with a partial-state merge (two kernels); 16-byte loads and stores",
}


# --------------------------------------------------------------------------- pieces
@T.inline
def ln_combine(st, ot):
    """st <- st ⊕ ot: Chan's merge of (count, mean, ss); the co-moment re-centred on it."""
    cnt = st[0] + ot[0]
    d = ot[1] - st[1]
    wo = ot[0] / T.max(cnt, T.float32(1.0))  # the other's share of the count; 0 when both are empty
    ws = st[0] / T.max(cnt, T.float32(1.0))
    st[4] = st[4] + ot[4] - d * wo * st[3] + d * ws * ot[3]
    st[2] = st[2] + ot[2] + d * d * st[0] * wo
    st[3] = st[3] + ot[3]
    st[1] = st[1] + d * wo
    st[0] = cnt


@T.inline
def add_combine(st, ot):
    """The unfused passes' sums: four of them side by side."""
    for j in T.unroll(4):
        st[j] = st[j] + ot[j]


@T.inline
def clear(st, k):
    for j in T.unroll(k):
        st[j] = T.float32(0)


@T.inline
def no_exchange(st, ot, cst, addr, c, tx):
    """``cluster_allreduce`` of a one-CTA "cluster": the CTA's state is the row's."""
    pass


@T.inline
def acc_none(acc, xr, gr, j, cf):
    """The dx pass reduces nothing."""
    pass


@T.inline
def ld_vector(xr, X, row, i, v, vec):
    """xr[i * vec : (i + 1) * vec] <- X[row, v * vec : (v + 1) * vec], 16 bytes."""
    xr[T.Ramp(i * vec, 1, vec)] = X[row, T.Ramp(v * vec, 1, vec)]


@T.inline
def ld_scalar(xr, X, row, i, v, vec):
    """xr[i] <- X[row, v]: the unaligned path, a "vector" of one element."""
    xr[i] = X[row, v]


@T.inline
def st_vector(Out, row, v, ob, i, vec):
    Out[row, T.Ramp(v * vec, 1, vec)] = ob[T.Ramp(i * vec, 1, vec)]


@T.inline
def st_scalar(Out, row, v, ob, i, vec):
    Out[row, v] = ob[i]


@T.inline
def load_batch(xr, gr, X, G, row, v0, vend, u, vec, ld, use_x, use_g):
    """xr, gr <- the vectors ``v0 + i * THREADS`` (i < u) of the row that are below ``vend``."""
    for i in T.unroll(u):
        if v0 + i * THREADS < vend:
            if use_x:
                ld(xr, X, row, i, v0 + i * THREADS, vec)
            if use_g:
                ld(gr, G, row, i, v0 + i * THREADS, vec)


@T.inline
def leaf_state(ot, xr, gr, v0, vend, u, vec):
    """ot <- the state of the elements ``load_batch`` loaded: their sums, then the centred sums."""
    clear(ot, K)
    for i in T.unroll(u):
        if v0 + i * THREADS < vend:
            ot[0] = ot[0] + T.float32(vec)
            for k in T.unroll(vec):
                ot[1] = ot[1] + T.Cast("float32", xr[i * vec + k])
                ot[3] = ot[3] + T.Cast("float32", gr[i * vec + k])
    ot[1] = ot[1] / T.max(ot[0], T.float32(1.0))
    for i in T.unroll(u):
        if v0 + i * THREADS < vend:
            for k in T.unroll(vec):
                dev = T.Cast("float32", xr[i * vec + k]) - ot[1]
                ot[2] = ot[2] + dev * dev
                ot[4] = ot[4] + T.Cast("float32", gr[i * vec + k]) * dev


@T.inline
def coefficients(cf, st, m):
    """cf <- (mean, rs, Σg / m, Σg(x - mean) rs² / m) of the row's state ``st``."""
    rs = T.float32(1.0) / T.sqrt(st[2] * T.float32(1.0 / m) + T.float32(LN_EPS))
    cf[0] = st[1]
    cf[1] = rs
    cf[2] = st[3] * T.float32(1.0 / m)
    cf[3] = st[4] * rs * rs * T.float32(1.0 / m)


@T.inline
def dx_batch(Out, ob, xr, gr, row, v0, vend, u, vec, cf, st_):
    """Out[row, vectors v0 + i * THREADS] <- rs (g - a - (x - mean) b), from xr, gr."""
    for i in T.unroll(u):
        if v0 + i * THREADS < vend:
            for k in T.unroll(vec):
                ob[i * vec + k] = T.Cast(
                    "bfloat16",
                    cf[1]
                    * (
                        T.Cast("float32", gr[i * vec + k])
                        - cf[2]
                        - (T.Cast("float32", xr[i * vec + k]) - cf[0]) * cf[3]
                    ),
                )
            st_(Out, row, v0 + i * THREADS, ob, i, vec)


@T.inline
def cluster_sync():
    T.ptx.barrier.cluster.arrive.release.aligned()
    T.ptx.barrier.cluster.wait.acquire.aligned()


@T.inline
def cluster_allreduce(st, ot, cst, addr, c, tx):
    """Every CTA of the cluster holds its CTA state in ``st``; afterwards all hold the total.

    Each CTA publishes its state in its own shared memory ``cst``; after a cluster
    barrier every thread reads the ``c`` states through DSMEM and merges them in
    rank order (the same total everywhere); a second barrier keeps ``cst`` alive
    until every CTA has read it.
    """
    if tx == 0:
        for j in T.unroll(K):
            cst[j] = st[j]
    cluster_sync()
    clear(st, K)
    for r in T.unroll(c):
        for j in T.unroll(K):
            T.ptx.mapa.shared__cluster.u32(
                addr[0], T.cuda.cvta_generic_to_shared(cst.ptr_to([j])), T.uint32(r)
            )
            T.ptx.ld.shared__cluster.f32(ot[j], addr[0])
        ln_combine(st, ot)
    cluster_sync()


def _io(vec):
    return (ld_vector, st_vector) if vec > 1 else (ld_scalar, st_scalar)


# --------------------------------------------------------------------------- fused kernels
def _held_kernel(n: int, m: int, vec: int, u: int, c: int, seg: int, slots: int):
    """A row held in registers by ``c`` CTAs (a cluster when c > 1), ``seg`` vectors each."""
    nvec = m // vec
    ld, st_ = _io(vec)
    launch = T.cuda.LaunchConfig(
        grid=(c * slots,), block=(THREADS,), cluster=(c,) if c > 1 else None
    )
    exchange = cluster_allreduce if c > 1 else no_exchange  # both arms of an `if` are built

    @T.function
    def ln_bwd_held(
        X: T.Tensor((n, m), "bfloat16", align=16),
        G: T.Tensor((n, m), "bfloat16", align=16),
        Out: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=launch)
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")
        ot = T.alloc_local((K,), "float32")
        cf = T.alloc_local((4,), "float32")
        xr = T.alloc_local((u * vec,), "bfloat16")
        gr = T.alloc_local((u * vec,), "bfloat16")
        ob = T.alloc_local((u * vec,), "bfloat16")
        addr = T.alloc_local((1,), "uint32")
        smem = T.alloc_shared((WARPS, K), "float32")
        cst = T.alloc_shared((K,), "float32")
        v0 = (bx % c) * seg + tx
        vend = T.min((bx % c) * seg + seg, nvec)
        row: T.int32 = bx // c
        while row < n:
            load_batch(xr, gr, X, G, row, v0, vend, u, vec, ld, True, True)
            leaf_state(st, xr, gr, v0, vend, u, vec)
            cta_allreduce(st, ot, K, ln_combine, smem, warp, lane, WARPS)
            exchange(st, ot, cst, addr, c, tx)
            coefficients(cf, st, m)
            dx_batch(Out, ob, xr, gr, row, v0, vend, u, vec, cf, st_)
            row += slots

    return ln_bwd_held


def _stream_kernel(n: int, m: int, vec: int, u: int):
    """One CTA per row; the row is folded batch by batch, then read again to write dx."""
    nvec = m // vec
    ld, st_ = _io(vec)
    blocks = min(n, SMS * CTAS_PER_SM)

    @T.function
    def ln_bwd_stream(
        X: T.Tensor((n, m), "bfloat16", align=16),
        G: T.Tensor((n, m), "bfloat16", align=16),
        Out: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((K,), "float32")
        ot = T.alloc_local((K,), "float32")
        cf = T.alloc_local((4,), "float32")
        xr = T.alloc_local((u * vec,), "bfloat16")
        gr = T.alloc_local((u * vec,), "bfloat16")
        ob = T.alloc_local((u * vec,), "bfloat16")
        smem = T.alloc_shared((WARPS, K), "float32")
        row: T.int32 = bx
        while row < n:
            clear(st, K)
            v: T.int32 = tx
            while v < nvec:
                load_batch(xr, gr, X, G, row, v, nvec, u, vec, ld, True, True)
                leaf_state(ot, xr, gr, v, nvec, u, vec)
                ln_combine(st, ot)
                v += u * THREADS
            cta_allreduce(st, ot, K, ln_combine, smem, warp, lane, WARPS)
            coefficients(cf, st, m)
            w: T.int32 = tx
            while w < nvec:
                load_batch(xr, gr, X, G, row, w, nvec, u, vec, ld, True, True)
                dx_batch(Out, ob, xr, gr, row, w, nvec, u, vec, cf, st_)
                w += u * THREADS
            row += blocks

    return ln_bwd_stream


def _split_kernels(n: int, m: int, vec: int, u: int, s: int, seg: int):
    """Rows cut into ``s`` segments of ``seg`` vectors: segment states, then merge and write dx."""
    nvec = m // vec
    ld, st_ = _io(vec)
    total = s * n

    @T.function
    def ln_bwd_partial(
        X: T.Tensor((n, m), "bfloat16", align=16),
        G: T.Tensor((n, m), "bfloat16", align=16),
        P: T.Tensor((n, s, K), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(total,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        sx = bx % s
        row = bx // s
        st = T.alloc_local((K,), "float32")
        ot = T.alloc_local((K,), "float32")
        xr = T.alloc_local((u * vec,), "bfloat16")
        gr = T.alloc_local((u * vec,), "bfloat16")
        smem = T.alloc_shared((WARPS, K), "float32")
        vend = T.min(sx * seg + seg, nvec)
        clear(st, K)
        v: T.int32 = sx * seg + tx
        while v < vend:
            load_batch(xr, gr, X, G, row, v, vend, u, vec, ld, True, True)
            leaf_state(ot, xr, gr, v, vend, u, vec)
            ln_combine(st, ot)
            v += u * THREADS
        cta_allreduce(st, ot, K, ln_combine, smem, warp, lane, WARPS)
        if tx == 0:
            for j in T.unroll(K):
                P[row, sx, j] = st[j]

    @T.function
    def ln_bwd_finish(
        X: T.Tensor((n, m), "bfloat16", align=16),
        G: T.Tensor((n, m), "bfloat16", align=16),
        P: T.Tensor((n, s, K), "float32"),
        Out: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(total,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        sx = (total - 1 - bx) % s  # the reverse of kernel 1's order: L2 still holds its last reads
        row = (total - 1 - bx) // s
        st = T.alloc_local((K,), "float32")
        ot = T.alloc_local((K,), "float32")
        cf = T.alloc_local((4,), "float32")
        xr = T.alloc_local((u * vec,), "bfloat16")
        gr = T.alloc_local((u * vec,), "bfloat16")
        ob = T.alloc_local((u * vec,), "bfloat16")
        smem = T.alloc_shared((WARPS, K), "float32")
        clear(st, K)
        p: T.int32 = tx
        while p < s:
            for j in T.unroll(K):
                ot[j] = P[row, p, j]
            ln_combine(st, ot)
            p += THREADS
        cta_allreduce(st, ot, K, ln_combine, smem, warp, lane, WARPS)
        coefficients(cf, st, m)
        vend = T.min(sx * seg + seg, nvec)
        w: T.int32 = sx * seg + tx
        while w < vend:
            load_batch(xr, gr, X, G, row, w, vend, u, vec, ld, True, True)
            dx_batch(Out, ob, xr, gr, row, w, vend, u, vec, cf, st_)
            w += u * THREADS

    return ln_bwd_partial, ln_bwd_finish


# --------------------------------------------------------------------------- unfused kernels
# The passes of the unfused program: 0 Σx, 1 Σ(x - mean)², 2 Σg, 3 Σ g x̂, then dx.
@T.inline
def acc_x(acc, xr, gr, j, cf):
    acc[0] = acc[0] + T.Cast("float32", xr[j])


@T.inline
def acc_ss(acc, xr, gr, j, cf):
    dev = T.Cast("float32", xr[j]) - cf[0]
    acc[0] = acc[0] + dev * dev


@T.inline
def acc_g(acc, xr, gr, j, cf):
    acc[0] = acc[0] + T.Cast("float32", gr[j])


@T.inline
def acc_gxh(acc, xr, gr, j, cf):
    acc[0] = acc[0] + T.Cast("float32", gr[j]) * ((T.Cast("float32", xr[j]) - cf[0]) * cf[1])


_PASSES = ((acc_x, True, False), (acc_ss, True, False), (acc_g, False, True), (acc_gxh, True, True))


@T.inline
def totals_one(tot, ot, P, row, s, tx, smem, warp, lane):
    for j in T.unroll(4):
        tot[j] = P[row, 0, j]


@T.inline
def totals_split(tot, ot, P, row, s, tx, smem, warp, lane):
    """tot <- the four sums of the row: its segments' partial sums, merged in a fixed order."""
    clear(tot, 4)
    p: T.int32 = tx
    while p < s:
        for j in T.unroll(4):
            tot[j] = tot[j] + P[row, p, j]
        p += THREADS
    cta_allreduce(tot, ot, 4, add_combine, smem, warp, lane, WARPS)


@T.inline
def unfused_coefficients(cf, tot, m):
    """cf <- (mean, rs, Σg / m, Σ g x̂ / m) from the passes' sums, as the unfused program."""
    cf[0] = tot[0] * T.float32(1.0 / m)
    cf[1] = T.float32(1.0) / T.sqrt(tot[1] * T.float32(1.0 / m) + T.float32(LN_EPS))
    cf[2] = tot[2] * T.float32(1.0 / m)
    cf[3] = tot[3] * T.float32(1.0 / m)


def _unfused_kernel(n: int, m: int, vec: int, u: int, s: int, seg: int, stage: int):
    """Pass ``stage`` of the unfused program over every row; s segments a row, one CTA each."""
    nvec = m // vec
    ld, st_ = _io(vec)
    total = s * n
    totals = totals_split if s > 1 else totals_one
    reduce = stage < 4
    acc_fn, use_x, use_g = _PASSES[stage] if reduce else (acc_none, True, True)

    @T.function
    def ln_bwd_pass(
        X: T.Tensor((n, m), "bfloat16", align=16),
        G: T.Tensor((n, m), "bfloat16", align=16),
        P: T.Tensor((n, s, 4), "float32"),
        Out: T.Tensor((n, m), "bfloat16", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(total,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        sx = bx % s
        row = bx // s
        tot = T.alloc_local((4,), "float32")
        ot = T.alloc_local((4,), "float32")
        cf = T.alloc_local((4,), "float32")
        acc = T.alloc_local((4,), "float32")
        xr = T.alloc_local((u * vec,), "bfloat16")
        gr = T.alloc_local((u * vec,), "bfloat16")
        ob = T.alloc_local((u * vec,), "bfloat16")
        smem = T.alloc_shared((WARPS, 4), "float32")
        if stage > 0:
            totals(tot, ot, P, row, s, tx, smem, warp, lane)
            unfused_coefficients(cf, tot, m)
        clear(acc, 4)
        vend = T.min(sx * seg + seg, nvec)
        v: T.int32 = sx * seg + tx
        while v < vend:
            load_batch(xr, gr, X, G, row, v, vend, u, vec, ld, use_x, use_g)
            if reduce:
                for i in T.unroll(u):
                    if v + i * THREADS < vend:
                        for k in T.unroll(vec):
                            acc_fn(acc, xr, gr, i * vec + k, cf)
            else:
                dx_batch(Out, ob, xr, gr, row, v, vend, u, vec, cf, st_)
            v += u * THREADS
        if reduce:
            cta_allreduce(acc, ot, 4, add_combine, smem, warp, lane, WARPS)
            if tx == 0:
                P[row, sx, stage] = acc[0]

    return ln_bwd_pass


# --------------------------------------------------------------------------- plan
def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _segments(n: int, nvec: int, u: int) -> tuple[int, int]:
    """(s, seg): segments a row is cut into for ~8 CTAs per SM, and their length in vectors."""
    batch = u * THREADS  # vectors a CTA folds per step
    batches = _cdiv(nvec, batch)
    want = _cdiv(SMS * CTAS_PER_SM, n)
    per = _cdiv(batches, want)  # batches per segment
    return _cdiv(batches, per), per * batch


def plan(n: int, m: int, variant: str = "fused") -> dict:
    """The path and its sizes, decided from the static ``(n, m)``."""
    vec = vector_width("bfloat16") if vectorisable(m, "bfloat16") else 1
    nvec = m // vec
    stream_u = BATCH // vec
    per_cta = THREADS * (HELD // vec)  # vectors one CTA can hold
    c = _cdiv(nvec, per_cta)
    if variant == "unfused":
        held_like = c == 1 or (c <= CLUSTER and n * c >= SMS) or n >= 2 * SMS
        s, seg = (1, nvec) if held_like else _segments(n, nvec, stream_u)
        return {"path": "unfused", "vec": vec, "u": stream_u, "s": s, "seg": seg}
    if c == 1 or (c <= CLUSTER and n * c >= SMS):
        u = _cdiv(nvec, c * THREADS)
        return {"path": "held", "vec": vec, "u": u, "c": c, "seg": u * THREADS, "slots": n}
    if n >= 2 * SMS:
        return {"path": "stream", "vec": vec, "u": stream_u}
    s, seg = _segments(n, nvec, stream_u)
    return {"path": "split", "vec": vec, "u": stream_u, "s": s, "seg": seg}


def _workspace(shape):
    """A float32 workspace of ``shape`` per device, allocated on first use and then reused."""
    cache = {}

    def get(device):
        if device not in cache:
            cache[device] = torch.empty(shape, device=device, dtype=torch.float32)
        return cache[device]

    return get


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, m = cfg["n"], cfg["m"]
    p = plan(n, m, variant)
    name = f"layernorm_backward_{p['path']}_{n}x{m}"
    vec, u = p["vec"], p["u"]
    if p["path"] == "unfused":
        s, seg = p["s"], p["seg"]
        mods = [
            compile_sm90(_unfused_kernel(n, m, vec, u, s, seg, stage), name=f"{name}_pass{stage}")
            for stage in range(5)
        ]
        sums = _workspace((n, s, 4))

        def run_unfused(X, G):
            out = torch.empty_like(X)
            ws = sums(X.device)
            for mod in mods:
                mod(X, G, ws, out)
            return (out,)

        return run_unfused
    if p["path"] == "split":
        s = p["s"]
        partial, finish = _split_kernels(n, m, vec, u, s, p["seg"])
        mod_partial = compile_sm90(partial, name=name + "_partial")
        mod_finish = compile_sm90(finish, name=name + "_finish")
        states = _workspace((n, s, K))

        def run_split(X, G):
            out = torch.empty_like(X)
            ws = states(X.device)
            mod_partial(X, G, ws)
            mod_finish(X, G, ws, out)
            return (out,)

        return run_split
    if p["path"] == "held":
        func = _held_kernel(n, m, vec, u, p["c"], p["seg"], p["slots"])
    else:
        func = _stream_kernel(n, m, vec, u)
    mod = compile_sm90(func, name=name)

    def run(X, G):
        out = torch.empty_like(X)
        mod(X, G, out)
        return (out,)

    return run
