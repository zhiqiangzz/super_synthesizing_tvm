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
"""The peak-to-average power ratio of every row, mean removed, in one pass. float32.

    papr = max_j (x_j - mean)² / mean_j (x_j - mean)²

The algorithm is chain_ops' ``extremes.papr_fused``: a state of five,
``(cnt, mean, M2, hi, lo)`` -- Welford's count, mean and centred sum of
squares, and the row's largest and smallest element. Merging two states
(``papr_combine``) is Chan's update for the first three and max / min for the
extremes::

    cnt = a.cnt + b.cnt,  d = b.mean - a.mean
    mean = a.mean + d * b.cnt / cnt
    M2   = a.M2 + b.M2 + d² * a.cnt * b.cnt / cnt
    hi   = max(a.hi, b.hi),  lo = min(a.lo, b.lo)

The largest square of ``x - mean`` is at one of the two extremes, so the
epilogue is ``max(hi - mean, mean - lo)² / (M2 / m)``.

How a thread folds its elements: it loads a *chunk* of ``U`` 16-byte vectors
(``4U`` floats; or ``4U`` scalars when the rows are not 16-byte aligned) and
folds it with the same merge along a balanced tree (``papr_fold``). Every
merge in that tree joins two states of equal count, so ``b.cnt / cnt`` is 1/2
and ``a.cnt * b.cnt / cnt`` is ``c / 2``: constants, no division. Only the
merge of a chunk into the thread's running state, and the merges across
threads, divide (``__fdiv_rn``, correctly rounded although the pipeline
compiles with ``--use_fast_math``); the empty state (count 0) merges safely.

How it is laid out (``_kernel``):

* short rows (``m <= WARP_ROW_MAX``, all the benchmark sizes): one warp per
  row, 8 rows per 256-thread CTA, one CTA per 8 rows (the hardware balances
  the last wave); the warp's 32 states merge by a butterfly
  (``warp_allreduce``) and lane 0 writes the result. No shared memory, no
  barrier.
* long rows: one 256-thread CTA per row, a grid of at most 8 CTAs per SM
  walking the rows; ``cta_allreduce`` merges the 256 states.

Either way a row is read once, in coalesced 16-byte vectors (every lane's
``u``-th vector of a chunk is adjacent to its neighbours'), and nothing but
the ``(n,)`` result is written. One kernel launch.

``variant="unfused"`` is the ablation: the unfused program's three passes,
each a launch of the same kernel shape that reads X again -- the row sum (the
mean), ``Σ (x - mean)²``, then ``max (x - mean)²`` and the result. Only the
per-row mean and sum of squares travel between them.
"""

from __future__ import annotations

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce, warp_allreduce

NEG = -3.0e38  # a finite -inf: the identity of max
POS = 3.0e38  # a finite +inf: the identity of min
K = 5  # (cnt, mean, M2, hi, lo)
THREADS = 256
WARPS = THREADS // 32
WARP_ROW_MAX = 16384  # rows up to this long go one warp per row
U = 8  # 16-byte vectors per chunk: a chunk is 4U floats

KERNEL_META = {
    "op": "papr",
    "dtype": "float32",
    "algorithm": "one pass, state (cnt, mean, M2, hi, lo): Chan's merge and both extremes; "
    "max(hi - mean, mean - lo)² / (M2 / m)",
    "structure": "one warp per row (8 rows per 256-thread CTA) for short rows, one CTA per row "
    "for long ones; 16-byte loads in chunks folded by a division-free tree; warp / CTA all-reduce",
}


def _fdiv(a, b):
    """``a / b`` rounded to nearest, whatever ``--use_fast_math`` makes of ``/``."""
    return T.call_pure_extern("__fdiv_rn", a, b, ty="float32")


def _fmax(a, b):
    """One ``FMNMX``; ``T.max`` is a NaN-propagating compare and select."""
    return T.call_pure_extern("fmaxf", a, b, ty="float32")


def _fmin(a, b):
    return T.call_pure_extern("fminf", a, b, ty="float32")


# ---------------------------------------------------------------------------
# the fused state (cnt, mean, M2, hi, lo)
# ---------------------------------------------------------------------------
@T.inline
def papr_combine(st, ot):
    """(cnt, mean, M2, hi, lo) <- st ⊕ ot: Chan's update, max and min. Count 0 is the identity."""
    cnt = st[0] + ot[0]
    r = _fdiv(ot[0], T.max(cnt, T.float32(1)))
    d = ot[1] - st[1]
    st[2] = st[2] + ot[2] + d * d * st[0] * r
    st[1] = st[1] + d * r
    st[0] = cnt
    st[3] = _fmax(st[3], ot[3])
    st[4] = _fmin(st[4], ot[4])


@T.inline
def papr_init(st):
    st[0] = T.float32(0)
    st[1] = T.float32(0)
    st[2] = T.float32(0)
    st[3] = T.float32(NEG)
    st[4] = T.float32(POS)


@T.inline
def papr_leaf(ot, x, cx):
    """The state of one element: (1, x, 0, x, x)."""
    ot[0] = T.float32(1)
    ot[1] = x
    ot[2] = T.float32(0)
    ot[3] = x
    ot[4] = x


@T.inline
def fold_leaves(buf, s, h):
    """States of count 2 from the pairs (buf[i], buf[i + h]): the merge of two elements."""
    for i in T.unroll(h):
        d = buf[i + h] - buf[i]
        s[1, i] = d * d * T.float32(0.5)
        s[0, i] = buf[i] + d * T.float32(0.5)
        s[2, i] = _fmax(buf[i], buf[i + h])
        s[3, i] = _fmin(buf[i], buf[i + h])


@T.inline
def fold_level(s, h, c):
    """State i <- state i ⊕ state i + h, both of count c.

    The merge with equal counts: ``b.cnt / cnt`` is 1/2 and ``a.cnt * b.cnt / cnt`` is ``c / 2``.
    """
    for i in T.unroll(h):
        d = s[0, i + h] - s[0, i]
        s[1, i] = s[1, i] + s[1, i + h] + d * d * T.float32(c * 0.5)
        s[0, i] = s[0, i] + d * T.float32(0.5)
        s[2, i] = _fmax(s[2, i], s[2, i + h])
        s[3, i] = _fmin(s[3, i], s[3, i + h])


@T.inline
def papr_fold(buf, s, ot, n, cx):
    """``ot`` <- the state of ``buf``'s ``n`` elements (a power of two, 2..64): a balanced tree."""
    fold_leaves(buf, s, n // 2)
    if n >= 4:
        fold_level(s, n // 4, 2)
    if n >= 8:
        fold_level(s, n // 8, 4)
    if n >= 16:
        fold_level(s, n // 16, 8)
    if n >= 32:
        fold_level(s, n // 32, 16)
    if n >= 64:
        fold_level(s, n // 64, 32)
    ot[0] = T.float32(n)
    ot[1] = s[0, 0]
    ot[2] = s[1, 0]
    ot[3] = s[2, 0]
    ot[4] = s[3, 0]


@T.inline
def papr_prepare(cx, W, row, n):
    cx[0] = T.float32(0)  # no context: the one pass needs none


@T.inline
def papr_finish(W, row, n, st, m):
    """max(hi - mean, mean - lo)² / (M2 / m)."""
    dmax = _fmax(st[3] - st[1], st[1] - st[4])
    W[row] = _fdiv(dmax * dmax, _fdiv(st[2], T.float32(m)))


# ---------------------------------------------------------------------------
# the unfused passes: one float32 per state; W = (mean, Σ (x - mean)², papr) by rows
# ---------------------------------------------------------------------------
@T.inline
def add_into(buf, i, j):
    buf[i] = buf[i] + buf[j]


@T.inline
def max_into(buf, i, j):
    buf[i] = _fmax(buf[i], buf[j])


@T.inline
def tree(buf, n, into):
    """buf[0] <- buf[0] ∘ ... ∘ buf[n - 1] along a balanced tree (``n`` a power of two, 2..64)."""
    for i in T.unroll(n // 2):
        into(buf, i, i + n // 2)
    if n >= 4:
        for i in T.unroll(n // 4):
            into(buf, i, i + n // 4)
    if n >= 8:
        for i in T.unroll(n // 8):
            into(buf, i, i + n // 8)
    if n >= 16:
        for i in T.unroll(n // 16):
            into(buf, i, i + n // 16)
    if n >= 32:
        for i in T.unroll(n // 32):
            into(buf, i, i + n // 32)
    if n >= 64:
        for i in T.unroll(n // 64):
            into(buf, i, i + n // 64)


@T.inline
def zero_init(st):
    st[0] = T.float32(0)


@T.inline
def neg_init(st):
    st[0] = T.float32(NEG)


@T.inline
def add_combine(st, ot):
    st[0] = st[0] + ot[0]


@T.inline
def max_combine(st, ot):
    st[0] = _fmax(st[0], ot[0])


@T.inline
def sum_leaf(ot, x, cx):
    ot[0] = x


@T.inline
def sq_leaf(ot, x, cx):
    d = x - cx[0]
    ot[0] = d * d


@T.inline
def sum_fold(buf, s, ot, n, cx):
    tree(buf, n, add_into)
    ot[0] = buf[0]


@T.inline
def sq_square(buf, n, cx):
    for i in T.unroll(n):
        buf[i] = (buf[i] - cx[0]) * (buf[i] - cx[0])


@T.inline
def sq_sum_fold(buf, s, ot, n, cx):
    sq_square(buf, n, cx)
    tree(buf, n, add_into)
    ot[0] = buf[0]


@T.inline
def sq_max_fold(buf, s, ot, n, cx):
    sq_square(buf, n, cx)
    tree(buf, n, max_into)
    ot[0] = buf[0]


@T.inline
def no_context(cx, W, row, n):
    cx[0] = T.float32(0)


@T.inline
def mean_context(cx, W, row, n):
    cx[0] = W[row]


@T.inline
def mean_finish(W, row, n, st, m):
    W[row] = _fdiv(st[0], T.float32(m))


@T.inline
def ss_finish(W, row, n, st, m):
    W[n + row] = st[0]


@T.inline
def papr_ratio_finish(W, row, n, st, m):
    """peak / (Σ (x - mean)² / m), as the unfused program writes it."""
    W[2 * n + row] = _fdiv(st[0], _fdiv(W[n + row], T.float32(m)))


# (k, init, leaf, fold, combine, prepare, finish): a pass over the rows
FUSED = (K, papr_init, papr_leaf, papr_fold, papr_combine, papr_prepare, papr_finish)
UNFUSED = (
    (1, zero_init, sum_leaf, sum_fold, add_combine, no_context, mean_finish),
    (1, zero_init, sq_leaf, sq_sum_fold, add_combine, mean_context, ss_finish),
    (1, neg_init, sq_leaf, sq_max_fold, max_combine, mean_context, papr_ratio_finish),
)


# ---------------------------------------------------------------------------
# a pass over the rows: the part every kernel shares
# ---------------------------------------------------------------------------
@T.inline
def row_reduce(A, row, t, p, m, vec, u, vectorised, st, ot, buf, s, cx, init, leaf, fold, combine):
    """``st`` <- the state of thread ``t``'s share of row ``row``, ``p`` threads sharing the row."""
    init(st)
    if vectorised:
        nvec = m // vec
        step = u * p  # vectors per chunk of the p threads
        nfull = nvec // step
        c: T.int32 = 0
        while c < nfull:
            for k in T.unroll(u):
                buf[T.Ramp(k * vec, 1, vec)] = A[row, T.Ramp((c * step + k * p + t) * vec, 1, vec)]
            fold(buf, s, ot, u * vec, cx)
            combine(st, ot)
            c += 1
        v: T.int32 = nfull * step + t
        while v < nvec:
            buf[T.Ramp(0, 1, vec)] = A[row, T.Ramp(v * vec, 1, vec)]
            fold(buf, s, ot, vec, cx)
            combine(st, ot)
            v += p
    else:
        step = u * vec * p  # scalars per chunk of the p threads
        nfull = m // step
        c: T.int32 = 0
        while c < nfull:
            for k in T.unroll(u * vec):
                buf[k] = A[row, c * step + k * p + t]
            fold(buf, s, ot, u * vec, cx)
            combine(st, ot)
            c += 1
        j: T.int32 = nfull * step + t
        while j < m:
            leaf(ot, A[row, j], cx)
            combine(st, ot)
            j += p


def _kernel(n: int, m: int, ops, wrows: int):
    """One pass over the rows of ``A``; ``W`` holds ``wrows`` per-row results, ``(wrows * n,)``."""
    k, init, leaf, fold, combine, prepare, finish = ops
    vec = vector_width("float32")
    vectorised = vectorisable(m, "float32")
    chunk = U * vec
    assert chunk & (chunk - 1) == 0 and 4 <= chunk <= 64, chunk  # what the folds fold

    if m <= WARP_ROW_MAX:
        blocks = (n + WARPS - 1) // WARPS

        @T.function
        def papr(A: T.Tensor((n, m), "float32", align=16), W: T.Tensor((wrows * n,), "float32")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
            bx = T.cuda.block_idx("x")
            tx = T.cuda.thread_idx("x")
            warp = tx // 32
            lane = tx % 32
            st = T.alloc_local((k,), "float32")  # this lane's state
            ot = T.alloc_local((k,), "float32")  # the state being merged in
            cx = T.alloc_local((1,), "float32")  # the row's context (the mean), if any
            buf = T.alloc_local((chunk,), "float32")
            s = T.alloc_local((4, chunk // 2), "float32")  # (mean, M2, hi, lo) of the fold tree
            row: T.int32 = bx * WARPS + warp
            while row < n:
                prepare(cx, W, row, n)
                row_reduce(
                    A,
                    row,
                    lane,
                    32,
                    m,
                    vec,
                    U,
                    vectorised,
                    st,
                    ot,
                    buf,
                    s,
                    cx,
                    init,
                    leaf,
                    fold,
                    combine,
                )
                warp_allreduce(st, ot, k, combine)
                if lane == 0:
                    finish(W, row, n, st, m)
                row += blocks * WARPS

        return papr

    blocks = min(n, SMS * 8)

    @T.function
    def papr(A: T.Tensor((n, m), "float32", align=16), W: T.Tensor((wrows * n,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((k,), "float32")
        ot = T.alloc_local((k,), "float32")
        cx = T.alloc_local((1,), "float32")
        buf = T.alloc_local((chunk,), "float32")
        s = T.alloc_local((4, chunk // 2), "float32")
        smem = T.alloc_shared((WARPS, k), "float32")
        row: T.int32 = bx
        while row < n:
            prepare(cx, W, row, n)
            row_reduce(
                A,
                row,
                tx,
                THREADS,
                m,
                vec,
                U,
                vectorised,
                st,
                ot,
                buf,
                s,
                cx,
                init,
                leaf,
                fold,
                combine,
            )
            cta_allreduce(st, ot, k, combine, smem, warp, lane, WARPS)
            if tx == 0:
                finish(W, row, n, st, m)
            row += blocks

    return papr


@cached
def build(cfg: dict, variant: str = "fused"):
    n, m = cfg["n"], cfg["m"]
    if variant == "fused":
        mod = compile_sm90(_kernel(n, m, FUSED, 1), name=f"papr_{n}x{m}")

        def run(X):
            out = torch.empty(n, device=X.device, dtype=F32)
            mod(X, out)
            return (out,)

        return run

    assert variant == "unfused", variant
    mods = [
        compile_sm90(_kernel(n, m, ops, 3), name=f"papr_unfused{i}_{n}x{m}")
        for i, ops in enumerate(UNFUSED)
    ]

    def run_unfused(X):
        W = torch.empty(3 * n, device=X.device, dtype=F32)  # mean, Σ (x - mean)², papr
        for mod in mods:
            mod(X, W)
        return (W[2 * n :],)

    return run_unfused
