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
"""Pearson correlation of every row of X with the same row of Y, in one pass.

The algorithm is chain_ops' ``pearson_fused``: the state is the six co-moments
``(c, x̄, ȳ, Mxx, Myy, Mxy)`` -- the count, the two means, and the centred
sums Σ(x - x̄)², Σ(y - ȳ)², Σ(x - x̄)(y - ȳ). Two states merge by Chan's rule
(``co_combine``): with ``c = a.c + b.c``, ``r = b.c / c``, ``f = a.c * r``
and ``d = b.mean - a.mean``, the means move by ``d * r`` and every centred
sum gains the cross term ``f * d * d'``. No raw sum of x² is ever formed, so
nothing cancels however long the row. The result of a row is
``Mxy / sqrt(Mxx * Myy)``, evaluated in float64.

Every row is shifted by its first pair, ``(X[row, 0], Y[row, 0])``, as it is
read: the centred sums do not change, but the means the states carry become
``x̄ - X[row, 0]``, of the size of the spread rather than of the data. Without
it a float32 mean of data far from zero (say 1000 ± 1) is off by half an ulp
of 1000, and the cross terms ``f * d * d'`` of the merges inherit that, 1e-4
of the spread; with it the subtraction is exact (Sterbenz) and the means are
as precise as the spread. Every work item of a row uses the same shift, so
their states merge as they are.

How it is laid out (``stream_rows``, shared by both variants):

* a row is cut into ``splits`` contiguous segments; one CTA of 256 threads
  takes a (row, segment) work item, and a grid of at most 8 CTAs per SM walks
  the ``n * splits`` items. ``splits`` is 1 when there are enough rows to fill
  the GPU; otherwise each row is split over enough CTAs to fill it;
* each thread reads its segment 16 bytes (4 float32) at a time, ``UNROLL``
  vectors of X and of Y per step, at a stride of 256 vectors so that a warp's
  loads are contiguous. It folds the ``4 * UNROLL`` pairs into a state of
  their own from registers and merges that into its running state. When ``m``
  is not a multiple of 4 the rows are not 16-byte aligned and it reads one
  element at a time instead (decided when the kernel is built);
* ``cta_allreduce`` merges the 256 states. With ``splits == 1`` thread 0
  writes the row's result. Otherwise thread 0 writes the segment's state to a
  small workspace, fences, and counts the row's finished segments with an
  atomic; the CTA that finishes a row's last segment merges the row's
  ``splits`` states with the same combine, writes the result and resets the
  counter. One launch, no second merge kernel.

``variant="unfused"`` is the ablation with the unfused program's passes
(``bench.ops.SPECS["pearson"].unfused``), on the same machinery: pass 1 the
two means (plain sums), pass 2 re-reads X and Y for the three centred sums
Σ(x - x̄)(y - ȳ), Σ(x - x̄)², Σ(y - ȳ)² and the correlation. Only the
``(n, 2)`` means travel between the passes.
"""

from __future__ import annotations

import dataclasses

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce

THREADS = 256
WARPS = THREADS // 32
UNROLL = 4  # vectors of X (and of Y) folded into one chunk state per step
KMAX = 6  # the largest state: (count, mean_x, mean_y, Mxx, Myy, Mxy)
CTAS_PER_SM = 8
MIN_UNITS_PER_THREAD = 4  # a split segment gives every thread at least this many loads
LINE = 32  # float32 per 128-byte line: a row's segment states start on a line of their own

KERNEL_META = {
    "op": "pearson",
    "dtype": "float32",
    "algorithm": "one pass, Chan/Welford co-moments (count, mean_x, mean_y, Mxx, Myy, Mxy) of "
    "the data shifted by the row's first pair; result Mxy / sqrt(Mxx Myy) in float64",
    "structure": "256-thread CTA per (row, segment), 16-byte loads, 4 vectors of X and Y per "
    "step, warp then CTA all-reduce; short-and-long rows split over CTAs, the last CTA of a "
    "row (atomic counter) merges its segment states: one launch",
}


# --- combines ---------------------------------------------------------------------------
@T.inline
def co_combine(st, ot):
    """st <- st ⊕ ot by Chan's rule; an empty state (count 0) is the identity."""
    cnt = st[0] + ot[0]
    r = ot[0] / T.max(cnt, T.float32(1))
    f = st[0] * r
    dx = ot[1] - st[1]
    dy = ot[2] - st[2]
    st[3] = st[3] + ot[3] + dx * dx * f
    st[4] = st[4] + ot[4] + dy * dy * f
    st[5] = st[5] + ot[5] + dx * dy * f
    st[1] = st[1] + dx * r
    st[2] = st[2] + dy * r
    st[0] = cnt


@T.inline
def add2(st, ot):
    for j in T.unroll(2):
        st[j] = st[j] + ot[j]


@T.inline
def add3(st, ot):
    for j in T.unroll(3):
        st[j] = st[j] + ot[j]


# --- per-row prologues: kk, the two numbers a row's elements are taken relative to -------
@T.inline
def first_pair(kk, X, Y, R, row):
    """The fused shift: the row's first pair."""
    kk[0] = X[row, 0]
    kk[1] = Y[row, 0]


@T.inline
def nothing(kk, X, Y, R, row):
    kk[0] = T.float32(0)
    kk[1] = T.float32(0)


@T.inline
def row_means(kk, X, Y, R, row):
    """The unfused pass 2: the means pass 1 wrote."""
    kk[0] = R[row, 0]
    kk[1] = R[row, 1]


# --- folds: the state of the ``size`` pairs of one step -----------------------------------
@T.inline
def fold_comoments(ot, bx, by, size, kk):
    """ot <- the co-moments of the pairs (bx - kk[0], by - kk[1]): mean, then centred sums."""
    for k in T.unroll(size):
        bx[k] = bx[k] - kk[0]
        by[k] = by[k] - kk[1]
    ot[1] = bx[0]
    ot[2] = by[0]
    for k in T.unroll(size - 1):
        ot[1] = ot[1] + bx[k + 1]
        ot[2] = ot[2] + by[k + 1]
    ot[1] = ot[1] * T.float32(1.0 / size)
    ot[2] = ot[2] * T.float32(1.0 / size)
    ot[0] = T.float32(size)
    ot[3] = T.float32(0)
    ot[4] = T.float32(0)
    ot[5] = T.float32(0)
    for k in T.unroll(size):
        ot[3] = ot[3] + (bx[k] - ot[1]) * (bx[k] - ot[1])
        ot[4] = ot[4] + (by[k] - ot[2]) * (by[k] - ot[2])
        ot[5] = ot[5] + (bx[k] - ot[1]) * (by[k] - ot[2])


@T.inline
def fold_sums(ot, bx, by, size, kk):
    """ot <- (Σ x, Σ y)."""
    ot[0] = bx[0]
    ot[1] = by[0]
    for k in T.unroll(size - 1):
        ot[0] = ot[0] + bx[k + 1]
        ot[1] = ot[1] + by[k + 1]


@T.inline
def fold_centred(ot, bx, by, size, kk):
    """ot <- (Σ(x - x̄)², Σ(y - ȳ)², Σ(x - x̄)(y - ȳ)), with the means in kk."""
    for k in T.unroll(size):
        bx[k] = bx[k] - kk[0]
        by[k] = by[k] - kk[1]
    ot[0] = bx[0] * bx[0]
    ot[1] = by[0] * by[0]
    ot[2] = bx[0] * by[0]
    for k in T.unroll(size - 1):
        ot[0] = ot[0] + bx[k + 1] * bx[k + 1]
        ot[1] = ot[1] + by[k + 1] * by[k + 1]
        ot[2] = ot[2] + bx[k + 1] * by[k + 1]


# --- finishes: a row's output from its state ----------------------------------------------
@T.inline
def corr_of_comoments(Out, row, st, m):
    """Out[row] = Mxy / sqrt(Mxx * Myy), in float64, rounded once to float32."""
    Out[row] = T.Cast(
        "float32",
        T.Cast("float64", st[5]) / T.sqrt(T.Cast("float64", st[3]) * T.Cast("float64", st[4])),
    )


@T.inline
def means_of_sums(Out, row, st, m):
    Out[row, 0] = st[0] / T.float32(m)
    Out[row, 1] = st[1] / T.float32(m)


@T.inline
def corr_of_centred(Out, row, st, m):
    """Out[row] = Σxy / sqrt(Σxx * Σyy), in float64, rounded once to float32."""
    Out[row] = T.Cast(
        "float32",
        T.Cast("float64", st[2]) / T.sqrt(T.Cast("float64", st[0]) * T.Cast("float64", st[1])),
    )


@dataclasses.dataclass(frozen=True)
class Stage:
    """One streaming pass over X and Y: its state, how it starts, folds, merges and ends."""

    name: str
    k: int  # floats in the state
    prologue: object
    fold: object
    combine: object
    finish: object
    out_cols: int  # 0: the output is (n,); c: it is (n, c)
    reads_rows: bool  # takes the (n, 2) result of the previous pass


FUSED = Stage("fused", 6, first_pair, fold_comoments, co_combine, corr_of_comoments, 0, False)
MEANS = Stage("means", 2, nothing, fold_sums, add2, means_of_sums, 2, False)
CENTRED = Stage("centred", 3, row_means, fold_centred, add3, corr_of_centred, 0, True)


# --- loads ----------------------------------------------------------------------------------
@T.inline
def load_vec(buf, u, A, row, v, vec):
    """buf[u*vec : u*vec+vec] = A[row, v*vec : v*vec+vec]: one 16-byte load."""
    buf[T.Ramp(u * vec, 1, vec)] = A[row, T.Ramp(v * vec, 1, vec)]


@T.inline
def load_scalar(buf, u, A, row, v, vec):
    """buf[u] = A[row, v]: the unaligned rows (vec == 1)."""
    buf[u] = A[row, v]


# --- the end of a work item -----------------------------------------------------------------
@T.inline
def end_whole_row(Out, P, C, st, ot, smem, flag, row, w, tx, warp, lane, g):
    """splits == 1: the CTA's state is the row's."""
    if tx == 0:
        g.finish(Out, row, st, g.m)


@T.inline
def end_segment(Out, P, C, st, ot, smem, flag, row, w, tx, warp, lane, g):
    """Publish the segment's state; the CTA that ends the row's last segment merges them all."""
    if tx == 0:
        for j in T.unroll(g.k):
            P[row * g.stride + (w % g.splits) * g.k + j] = st[j]
        T.cuda.thread_fence()
        done = T.cuda.atomic_add(T.address_of(C[row]), T.int32(1))
        flag[0] = T.if_then_else(done == g.splits - 1, 1, 0)
    T.cuda.cta_sync()
    if flag[0] == 1:
        for j in T.unroll(g.k):
            st[j] = T.float32(0)
        p: T.int32 = tx
        while p < g.splits:
            for j in T.unroll(g.k):
                ot[j] = P[row * g.stride + p * g.k + j]
            g.combine(st, ot)
            p += THREADS
        cta_allreduce(st, ot, g.k, g.combine, smem, warp, lane, WARPS)
        if tx == 0:
            g.finish(Out, row, st, g.m)
            C[row] = 0


@T.inline
def stream_rows(X, Y, R, Out, P, C, bx, tx, g):
    """Every (row, segment) work item of the grid-stride walk: fold, all-reduce, end."""
    warp = tx // 32
    lane = tx % 32
    st = T.alloc_local((g.k,), "float32")  # this thread's state
    ot = T.alloc_local((g.k,), "float32")  # the state being merged in
    kk = T.alloc_local((2,), "float32")
    bufx = T.alloc_local((g.chunk,), "float32")
    bufy = T.alloc_local((g.chunk,), "float32")
    smem = T.alloc_shared((WARPS, g.k), "float32")
    flag = T.alloc_shared((1,), "int32")
    w: T.int32 = bx
    while w < g.work:
        row: T.int32 = w // g.splits
        lo: T.int32 = (w % g.splits) * g.seg
        hi: T.int32 = T.min(lo + g.seg, g.units)
        g.prologue(kk, X, Y, R, row)
        for j in T.unroll(g.k):
            st[j] = T.float32(0)
        v: T.int32 = lo + tx
        while v + (UNROLL - 1) * THREADS < hi:
            for u in T.unroll(UNROLL):
                g.load(bufx, u, X, row, v + u * THREADS, g.vec)
                g.load(bufy, u, Y, row, v + u * THREADS, g.vec)
            g.fold(ot, bufx, bufy, g.chunk, kk)
            g.combine(st, ot)
            v += UNROLL * THREADS
        while v < hi:
            g.load(bufx, 0, X, row, v, g.vec)
            g.load(bufy, 0, Y, row, v, g.vec)
            g.fold(ot, bufx, bufy, g.vec, kk)
            g.combine(st, ot)
            v += THREADS
        cta_allreduce(st, ot, g.k, g.combine, smem, warp, lane, WARPS)
        g.end(Out, P, C, st, ot, smem, flag, row, w, tx, warp, lane, g)
        w += g.blocks


@dataclasses.dataclass(frozen=True)
class Plan:
    """The static layout of one stage on one size: everything ``stream_rows`` reads from ``g``."""

    n: int
    m: int
    vec: int  # elements per load: 4 (16 bytes) or 1
    units: int  # loads per row
    splits: int  # segments per row
    seg: int  # loads per segment
    stage: Stage

    @property
    def k(self):
        return self.stage.k

    @property
    def prologue(self):
        return self.stage.prologue

    @property
    def fold(self):
        return self.stage.fold

    @property
    def combine(self):
        return self.stage.combine

    @property
    def finish(self):
        return self.stage.finish

    @property
    def load(self):
        return load_vec if self.vec > 1 else load_scalar

    @property
    def end(self):
        return end_whole_row if self.splits == 1 else end_segment

    @property
    def chunk(self):
        return UNROLL * self.vec

    @property
    def work(self):
        return self.n * self.splits

    @property
    def blocks(self):
        return min(self.work, SMS * CTAS_PER_SM)

    @property
    def stride(self):
        """Floats of workspace per row: its segment states, rounded up to whole lines."""
        return -(-self.splits * self.k // LINE) * LINE


def plan(n: int, m: int) -> tuple[int, int, int]:
    """(vec, splits, seg): the load width, segments per row, and loads per segment."""
    vec = vector_width("float32") if vectorisable(m, "float32") else 1
    units = m // vec
    target = SMS * CTAS_PER_SM
    if n >= 2 * SMS:
        splits = 1
    else:
        splits = max(1, min(-(-target // n), units // (THREADS * MIN_UNITS_PER_THREAD)))
    seg = -(-units // splits)
    seg = -(-seg // THREADS) * THREADS  # whole strides of the CTA
    splits = -(-units // seg)
    return vec, splits, seg


def _kernel(g: Plan):
    """The stage ``g.stage`` as one kernel: (X, Y[, R], P, C, Out)."""
    n, m = g.n, g.m
    out_shape = (n, g.stage.out_cols) if g.stage.out_cols else (n,)
    ws = max(n * g.stride, 1)

    if g.stage.reads_rows:

        @T.function
        def pearson_stage(
            X: T.Tensor((n, m), "float32", align=16),
            Y: T.Tensor((n, m), "float32", align=16),
            R: T.Tensor((n, 2), "float32"),
            P: T.Tensor((ws,), "float32"),
            C: T.Tensor((n,), "int32"),
            Out: T.Tensor(out_shape, "float32"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(g.blocks,), block=(THREADS,)))
            bx = T.cuda.block_idx("x")
            tx = T.cuda.thread_idx("x")
            stream_rows(X, Y, R, Out, P, C, bx, tx, g)

        return pearson_stage

    @T.function
    def pearson_stage(
        X: T.Tensor((n, m), "float32", align=16),
        Y: T.Tensor((n, m), "float32", align=16),
        P: T.Tensor((ws,), "float32"),
        C: T.Tensor((n,), "int32"),
        Out: T.Tensor(out_shape, "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(g.blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        stream_rows(X, Y, X, Out, P, C, bx, tx, g)

    return pearson_stage


def plans(n: int, m: int, variant: str) -> list[Plan]:
    vec, splits, seg = plan(n, m)
    stages = {"fused": [FUSED], "unfused": [MEANS, CENTRED]}[variant]
    return [Plan(n, m, vec, m // vec, splits, seg, s) for s in stages]


class _Workspace:
    """Per-device, per-stage segment states and counters; the counters are zero between calls."""

    def __init__(self, gs: list[Plan]):
        self.gs = gs
        self.bufs = {}

    def get(self, device):
        if device not in self.bufs:
            self.bufs[device] = [
                (
                    torch.empty(max(g.n * g.stride, 1), device=device, dtype=F32),
                    torch.zeros(g.n, device=device, dtype=torch.int32),
                )
                for g in self.gs
            ]
            if len(self.gs) > 1:  # the (n, 2) means between the unfused passes
                self.bufs[device].append(torch.empty(self.gs[0].n, 2, device=device, dtype=F32))
        return self.bufs[device]


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, m = cfg["n"], cfg["m"]
    gs = plans(n, m, variant)
    mods = [compile_sm90(_kernel(g), name=f"pearson_{g.stage.name}_{n}x{m}") for g in gs]
    ws = _Workspace(gs)

    if variant == "fused":
        (fused,) = mods

        def run(X, Y):
            ((P, C),) = ws.get(X.device)
            out = torch.empty(n, device=X.device, dtype=F32)
            fused(X, Y, P, C, out)
            return (out,)

        return run

    means, centred = mods

    def run(X, Y):
        (P1, C1), (P2, C2), M = ws.get(X.device)
        out = torch.empty(n, device=X.device, dtype=F32)
        means(X, Y, P1, C1, M)
        centred(X, Y, M, P2, C2, out)
        return (out,)

    return run
