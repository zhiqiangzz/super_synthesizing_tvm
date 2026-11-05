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
"""Kurtosis of every row, ``m4 / m2²`` (biased central moments), in one pass. float32.

The algorithm is chain_ops' ``kurtosis_fused``: the state of a set of values
is ``(count, mean, M2, M3, M4)``, ``M_k = Σ (x - mean)^k``, and two states are
merged with Pébay's pairwise update (``chain_ops/moments.py``'s ``pebay(4)``).
With ``d = mean_b - mean_a``, ``n = n_a + n_b`` and the weights ``r_a = n_a / n``,
``r_b = n_b / n`` (the same formulas, the powers of ``n`` divided out):

    mean = mean_a + d r_b
    M2   = M2_a + M2_b + d² n_a r_b
    M3   = M3_a + M3_b + d³ n_a r_b (r_a - r_b) + 3 d (r_a M2_b - r_b M2_a)
    M4   = M4_a + M4_b + d⁴ n_a r_b (r_a² - r_a r_b + r_b²)
                       + 6 d² (r_a² M2_b + r_b² M2_a) + 4 d (r_a M3_b - r_b M3_a)

The empty state ``(0, 0, 0, 0, 0)`` is the identity (``n`` is clamped to 1 in
the division, and every correction then has a zero factor). A row's result is
``count · M4 / M2²``.

Precision: the count is an ``int32`` (exact to 2^31; a float32 count is not
past 2^24); the moments are float32 and every update adds its whole correction
to the accumulator in one rounding (``M + (M_b + corrections)``); states are
merged as balanced trees across threads; the result is computed as if rounded
once (``store_kurt``).

How it is laid out (``stream``, shared by both variants):

* a *team* of ``tpr`` threads takes a *task*, a row or a segment of a row:
  a warp per row for short rows (no shared memory, no barrier), up to the
  whole CTA of 256 threads for long ones (``_plan``). A grid of at most the
  resident CTAs walks the tasks.
* rows too few to fill the GPU are split over ``S`` CTAs, all resident at
  once; CTA ``s`` of a row takes the row's CTA-sized steps ``s, s + S, ...``
  (``S`` divides their number where it can, so the CTAs' work is equal).
  The CTA publishes its segment's state, fences,
  and counts the row's finished segments with an atomic; the CTA that
  finishes the last one merges the row's ``S`` states (a balanced tree again),
  writes the result and resets the counter. One launch.
* a thread reads its part ``U = 4`` loads at a time, the loads ``tpr`` apart
  (coalesced across the team); a load is 16 bytes (4 floats) when ``m`` is a
  multiple of 4, else one float. The ``U`` loads are folded into a state of
  their own by two passes in registers (mean, then the centred powers) and
  merged into the thread's running state; the loads left at the end are
  folded one at a time.
* the team's states are merged by a butterfly in each warp, then (teams of
  several warps) the warp states through shared memory by a butterfly over
  the team's warps.

``variant="unfused"`` is the ablation with the unfused program's passes
(``bench.ops.SPECS["kurtosis"].unfused``) on the same machinery: pass 1 the
mean (a sum), pass 2 re-reads X for ``m2 = Σ (x - mean)² / m``, pass 3
re-reads X for ``Σ ((x - mean) / √m2)⁴ / m``. Only the ``(n, 2)`` per-row
``(mean, m2)`` travel between the passes.
"""

from __future__ import annotations

import dataclasses

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable

THREADS = 256
WARPS = THREADS // 32
U = 4  # loads a thread has in flight per step, folded into one state
ROW_STEPS = 8  # a row's team grows past a warp only while every thread keeps this many steps
FILL = 1024  # threads of work per SM below which rows are split over CTAs
CTAS_PER_SM = 4  # CTAs per SM of the walk's grid
SPLIT_CTAS_PER_SM = 4  # segments per SM of a split: all resident at once
MIN_STEPS = 4  # a split segment gives every thread at least this many steps
INTERLEAVE = 1  # split segments take every S-th CTA step of the row (else contiguous ranges)
FULL_MASK = 0xFFFFFFFF
I32 = torch.int32

KERNEL_META = {
    "op": "kurtosis",
    "dtype": "float32",
    "algorithm": "one pass, state (count, mean, M2, M3, M4) merged by Pébay's pairwise "
    "update; result count·M4/M2²",
    "structure": "a warp per short row, a 256-thread CTA per long row, 16-byte loads 4 deep, "
    "chunk states merged per thread, butterfly merges; rows too few to fill the GPU split "
    "over CTAs, the last CTA of a row (atomic counter) merges its segment states: one launch",
}


# --- the states and their merges ---
@T.inline
def pebay_merge(cnt, st, ocnt, ot):
    """(cnt, st) <- (cnt, st) ⊕ (ocnt, ot); st = (mean, M2, M3, M4), cnt an int32 count."""
    na: T.float32 = T.Cast("float32", cnt[0])
    nb: T.float32 = T.Cast("float32", ocnt[0])
    inv: T.float32 = T.float32(1) / T.max(na + nb, T.float32(1))
    ra: T.float32 = na * inv
    rb: T.float32 = nb * inv
    d: T.float32 = ot[0] - st[0]
    d2: T.float32 = d * d
    w: T.float32 = na * rb  # n_a n_b / n
    a2: T.float32 = st[1]
    a3: T.float32 = st[2]
    a4: T.float32 = st[3]
    c4: T.float32 = (
        ot[3]
        + d2 * d2 * w * (ra * ra - ra * rb + rb * rb)
        + T.float32(6) * d2 * (ra * ra * ot[1] + rb * rb * a2)
        + T.float32(4) * d * (ra * ot[2] - rb * a3)
    )
    c3: T.float32 = ot[2] + d2 * d * w * (ra - rb) + T.float32(3) * d * (ra * ot[1] - rb * a2)
    c2: T.float32 = ot[1] + d2 * w
    st[3] = a4 + c4
    st[2] = a3 + c3
    st[1] = a2 + c2
    st[0] = st[0] + d * rb
    cnt[0] = cnt[0] + ocnt[0]


@T.inline
def add_merge(cnt, st, ocnt, ot):
    st[0] = st[0] + ot[0]


@T.inline
def fold_moments(ocnt, ot, buf, k, rv):
    """(ocnt, ot) <- the state of the ``k`` values buf[0:k], by two passes in registers."""
    s: T.float32 = T.float32(0)
    for i in T.unroll(k):
        s = s + buf[i]
    mean: T.float32 = s * T.float32(1.0 / k)
    q2: T.float32 = T.float32(0)
    q3: T.float32 = T.float32(0)
    q4: T.float32 = T.float32(0)
    for i in T.unroll(k):
        dv: T.float32 = buf[i] - mean
        sq: T.float32 = dv * dv
        q2 = q2 + sq
        q3 = q3 + sq * dv
        q4 = q4 + sq * sq
    ocnt[0] = k
    ot[0] = mean
    ot[1] = q2
    ot[2] = q3
    ot[3] = q4


@T.inline
def fold_sum(ocnt, ot, buf, k, rv):
    s: T.float32 = T.float32(0)
    for i in T.unroll(k):
        s = s + buf[i]
    ot[0] = s


@T.inline
def fold_centred_square(ocnt, ot, buf, k, rv):
    s: T.float32 = T.float32(0)
    for i in T.unroll(k):
        dv: T.float32 = buf[i] - rv[0]
        s = s + dv * dv
    ot[0] = s


@T.inline
def fold_standard_fourth(ocnt, ot, buf, k, rv):
    s: T.float32 = T.float32(0)
    for i in T.unroll(k):
        z: T.float32 = (buf[i] - rv[0]) * rv[1]
        z2: T.float32 = z * z
        s = s + z2 * z2
    ot[0] = s


# --- what a pass reads before and writes after its rows ---
def _fma(a, b, c):
    """a·b + c rounded once (CUDA's ``__fmaf_rn``; ``T.fma`` has no CUDA lowering)."""
    return T.call_pure_extern("__fmaf_rn", a, b, c, ty="float32")


def _div(a, b):
    """a / b correctly rounded (``/`` is approximate under ``--use_fast_math``)."""
    return T.call_pure_extern("__fdiv_rn", a, b, ty="float32")


@T.inline
def store_kurt(Out, RV, row, cnt, st, m):
    """Out[row] = count · M4 / M2², as if rounded once.

    ``c M4`` and ``M2²`` are carried exactly as unevaluated sums (``hi + lo``,
    ``p + e``, by FMA), the quotient ``q`` of their leading parts is rounded
    correctly, and its residual ``c M4 - q M2²`` corrects it.
    """
    c: T.float32 = T.Cast("float32", cnt[0])
    hi: T.float32 = c * st[3]
    lo: T.float32 = _fma(c, st[3], -hi)
    p: T.float32 = st[1] * st[1]
    e: T.float32 = _fma(st[1], st[1], -p)
    q: T.float32 = _div(hi, p)
    r: T.float32 = _fma(-q, p, hi) + lo - q * e
    Out[row] = q + r / p


@T.inline
def store_mean(Out, RV, row, cnt, st, m):
    RV[row, 0] = _div(st[0], T.float32(m))


@T.inline
def store_m2(Out, RV, row, cnt, st, m):
    RV[row, 1] = _div(st[0], T.float32(m))


@T.inline
def store_s4(Out, RV, row, cnt, st, m):
    Out[row] = _div(st[0], T.float32(m))


@T.inline
def read_nothing(rv, RV, row):
    rv[0] = T.float32(0)


@T.inline
def read_mean(rv, RV, row):
    rv[0] = RV[row, 0]


@T.inline
def read_mean_rstd(rv, RV, row):
    rv[0] = RV[row, 0]
    rv[1] = _div(T.float32(1), T.call_pure_extern("__fsqrt_rn", RV[row, 1], ty="float32"))


@dataclasses.dataclass(frozen=True)
class Stage:
    """One streaming pass over X: its state, and how it starts, folds, merges and ends a row."""

    name: str
    k: int  # floats in the state (besides the int32 count)
    prologue: object  # prologue(rv, RV, row): the row's values from the previous pass
    fold: object  # fold(ocnt, ot, buf, k, rv): the state of k loaded values
    combine: object  # combine(cnt, st, ocnt, ot): st <- st ⊕ ot
    finish: object  # finish(Out, RV, row, cnt, st, m): the row's result


FUSED = Stage("fused", 4, read_nothing, fold_moments, pebay_merge, store_kurt)
MEAN = Stage("mean", 1, read_nothing, fold_sum, add_merge, store_mean)
M2 = Stage("m2", 1, read_mean, fold_centred_square, add_merge, store_m2)
S4 = Stage("s4", 1, read_mean_rstd, fold_standard_fourth, add_merge, store_s4)
VARIANTS = {"fused": (FUSED,), "unfused": (MEAN, M2, S4)}


# --- merging a team's states ---
@T.inline
def butterfly(cnt, st, ocnt, ot, first, levels, g):
    """Merge with the lanes ``first, first/2, ..., first >> (levels-1)`` away (xor).

    A balanced tree; every lane ends with the merged state of its group.
    """
    for i in T.unroll(levels):
        ocnt[0] = T.gpu_warp_shuffle_xor(FULL_MASK, cnt[0], first >> i, 32, 32)
        for j in T.unroll(g.k):
            ot[j] = T.gpu_warp_shuffle_xor(FULL_MASK, st[j], first >> i, 32, 32)
        g.combine(cnt, st, ocnt, ot)


@T.inline
def merge_warp_team(cnt, st, ocnt, ot, smem, smem_c, warp, lane, g):
    """A team of one warp (or less): a butterfly over its lanes."""
    butterfly(cnt, st, ocnt, ot, g.tpr // 2, g.tpr.bit_length() - 1, g)


@T.inline
def merge_cta_team(cnt, st, ocnt, ot, smem, smem_c, warp, lane, g):
    """A team of ``wpt`` warps: a butterfly per warp, the warp states through shared memory,
    lane l of every warp takes its team's warp ``l % wpt`` and a butterfly over ``wpt``
    lanes merges them. Every thread of the team ends with the team's state. Ends with a
    barrier."""
    butterfly(cnt, st, ocnt, ot, 16, 5, g)
    if lane == 0:
        smem_c[warp] = cnt[0]
        for j in T.unroll(g.k):
            smem[warp, j] = st[j]
    T.cuda.cta_sync()
    src = (warp // g.wpt) * g.wpt + lane % g.wpt
    cnt[0] = smem_c[src]
    for j in T.unroll(g.k):
        st[j] = smem[src, j]
    butterfly(cnt, st, ocnt, ot, g.wpt // 2, g.wpt.bit_length() - 1, g)
    T.cuda.cta_sync()


@T.inline
def set_empty(cnt, st, k):
    cnt[0] = 0
    for j in T.unroll(k):
        st[j] = T.float32(0)


# --- loads ---
@T.inline
def load_step_vec(buf, X, row, v, g):
    """buf[u*w : u*w+w] = the w-float vector v + u*tpr of the row, for u < U (16-byte loads)."""
    for u in T.unroll(g.U):
        buf[T.Ramp(u * g.w, 1, g.w)] = X[row, T.Ramp((v + u * g.tpr) * g.w, 1, g.w)]


@T.inline
def load_one_vec(buf, X, row, v, g):
    buf[T.Ramp(0, 1, g.w)] = X[row, T.Ramp(v * g.w, 1, g.w)]


@T.inline
def load_step_scalar(buf, X, row, v, g):
    for u in T.unroll(g.U):
        buf[u] = X[row, v + u * g.tpr]


@T.inline
def load_one_scalar(buf, X, row, v, g):
    buf[0] = X[row, v]


# --- the end of a task ---
@T.inline
def end_row(Out, RV, P, PC, C, cnt, st, ocnt, ot, smem, smem_c, flag, task, row, tt, warp, lane, g):
    """S == 1: the team's state is the row's."""
    if tt == 0:
        if task < g.tasks:
            g.finish(Out, RV, row, cnt, st, g.m)


@T.inline
def end_segment(
    Out, RV, P, PC, C, cnt, st, ocnt, ot, smem, smem_c, flag, task, row, tt, warp, lane, g
):
    """Publish the segment's state; the CTA that ends the row's last segment merges them all."""
    if tt == 0:
        PC[task] = cnt[0]
        for j in T.unroll(g.k):
            P[task, j] = st[j]
        T.cuda.thread_fence()
        done = T.cuda.atomic_add(T.address_of(C[row]), T.int32(1))
        flag[0] = T.if_then_else(done == g.S - 1, 1, 0)
    T.cuda.cta_sync()
    if flag[0] == 1:
        T.cuda.thread_fence()
        set_empty(cnt, st, g.k)
        p: T.int32 = tt
        while p < g.S:
            ocnt[0] = PC[row * g.S + p]
            for j in T.unroll(g.k):
                ot[j] = P[row * g.S + p, j]
            g.combine(cnt, st, ocnt, ot)
            p += THREADS
        merge_cta_team(cnt, st, ocnt, ot, smem, smem_c, warp, lane, g)
        if tt == 0:
            g.finish(Out, RV, row, cnt, st, g.m)
            C[row] = 0


@T.inline
def stream(X, RV, Out, P, PC, C, bx, tx, g):
    """Every task of the grid-stride walk: fold the team's part of the row, merge, end."""
    warp = tx // 32
    lane = tx % 32
    team = tx // g.tpr
    tt = tx % g.tpr
    cnt = T.alloc_local((1,), "int32")
    st = T.alloc_local((g.k,), "float32")
    ocnt = T.alloc_local((1,), "int32")
    ot = T.alloc_local((g.k,), "float32")
    rv = T.alloc_local((2,), "float32")
    buf = T.alloc_local((g.U * g.w,), "float32")
    smem = T.alloc_shared((WARPS, g.k), "float32")
    smem_c = T.alloc_shared((WARPS,), "int32")
    flag = T.alloc_shared((1,), "int32")
    grp: T.int32 = bx
    while grp < g.groups:
        task: T.int32 = grp * g.teams + team
        tk: T.int32 = T.min(task, g.tasks - 1)
        row: T.int32 = tk // g.S
        lo: T.int32 = (tk % g.S) * g.first
        hi: T.int32 = T.if_then_else(task < g.tasks, T.min(lo + g.span, g.loads), lo)
        g.prologue(rv, RV, row)
        set_empty(cnt, st, g.k)
        v: T.int32 = lo + tt
        while v + (g.U - 1) * g.tpr < hi:
            g.load_step(buf, X, row, v, g)
            g.fold(ocnt, ot, buf, g.U * g.w, rv)
            g.combine(cnt, st, ocnt, ot)
            v += g.vstep
        while v < hi:
            g.load_one(buf, X, row, v, g)
            g.fold(ocnt, ot, buf, g.w, rv)
            g.combine(cnt, st, ocnt, ot)
            v += g.tpr
        g.team_merge(cnt, st, ocnt, ot, smem, smem_c, warp, lane, g)
        g.end(
            Out, RV, P, PC, C, cnt, st, ocnt, ot, smem, smem_c, flag, task, row, tt, warp, lane, g
        )
        grp += g.blocks


# --- the layout ---
@dataclasses.dataclass(frozen=True)
class Plan:
    """The static layout of one pass on one size: everything ``stream`` reads from ``g``."""

    n: int
    m: int
    w: int  # floats per load: 4 (16 bytes) or 1
    loads: int  # loads per row
    tpr: int  # threads per team (a task's threads)
    S: int  # segments per row
    seg: int  # loads per segment
    U: int  # loads per step
    inter: bool  # a split row's segments interleaved by team steps (else contiguous)
    stage: Stage

    teams = property(lambda g: THREADS // g.tpr)  # teams per CTA
    wpt = property(lambda g: max(1, g.tpr // 32))  # warps per team
    tasks = property(lambda g: g.n * g.S)
    groups = property(lambda g: -(-g.tasks // g.teams))  # the CTA-sized steps of the walk
    split = property(lambda g: g.S > 1)
    first = property(lambda g: g.U * g.tpr if g.inter else g.seg)  # segment s starts at s * first
    span = property(lambda g: g.loads if g.inter else g.seg)  # and ends at most span later
    vstep = property(lambda g: g.S * g.U * g.tpr if g.inter else g.U * g.tpr)  # a thread's step
    blocks = property(lambda g: g.groups if g.split else min(g.groups, SMS * CTAS_PER_SM))
    ptasks = property(lambda g: g.tasks if g.split else 1)  # rows of the segment workspace
    k = property(lambda g: g.stage.k)
    prologue = property(lambda g: g.stage.prologue)
    fold = property(lambda g: g.stage.fold)
    combine = property(lambda g: g.stage.combine)
    finish = property(lambda g: g.stage.finish)
    team_merge = property(lambda g: merge_cta_team if g.tpr > 32 else merge_warp_team)
    end = property(lambda g: end_segment if g.split else end_row)
    load_step = property(lambda g: load_step_vec if g.w > 1 else load_step_scalar)
    load_one = property(lambda g: load_one_vec if g.w > 1 else load_one_scalar)


def _plan(n: int, m: int, stage: Stage) -> Plan:
    """A warp per row while a thread would otherwise have fewer than ROW_STEPS steps, up to a
    CTA per row; rows too few to give every SM FILL threads are split over SPLIT_CTAS_PER_SM
    resident CTAs per SM, in equal segments of whole team strides."""
    w = vector_width("float32") if vectorisable(m, "float32") else 1
    loads = m // w
    tpr = 32
    while tpr < THREADS and loads >= 2 * tpr * U * ROW_STEPS:
        tpr *= 2
    S, seg, inter = 1, loads, False
    if n * tpr < SMS * FILL:
        steps = -(-loads // (THREADS * U))  # the CTA's steps in a row
        want = min(SMS * SPLIT_CTAS_PER_SM // n, steps // MIN_STEPS)
        if want > 1:
            tpr = THREADS
            if INTERLEAVE:
                # as many segments as fit, or a few less if that divides the steps evenly
                even = [s for s in range(want, (want * 9) // 10, -1) if steps % s == 0]
                S, seg, inter = (even[0] if even else want), THREADS * U, True
            else:
                seg = -(-(-(-loads // want)) // THREADS) * THREADS
                S = -(-loads // seg)
    return Plan(n, m, w, loads, tpr, S, seg, U, inter, stage)


def _kernel(g: Plan):
    n, m, k, ptasks, blocks = g.n, g.m, g.k, g.ptasks, g.blocks

    @T.function
    def kurtosis_pass(
        X: T.Tensor((n, m), "float32", align=16),
        RV: T.Tensor((n, 2), "float32"),
        Out: T.Tensor((n,), "float32"),
        P: T.Tensor((ptasks, k), "float32"),
        PC: T.Tensor((ptasks,), "int32"),
        C: T.Tensor((n,), "int32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        stream(X, RV, Out, P, PC, C, T.cuda.block_idx("x"), T.cuda.thread_idx("x"), g)

    return kurtosis_pass


@cached
def build(cfg: dict, variant: str = "fused"):
    n, m = cfg["n"], cfg["m"]
    plans = [_plan(n, m, stage) for stage in VARIANTS[variant]]
    mods = {
        g.stage.name: compile_sm90(_kernel(g), name=f"kurtosis_{g.stage.name}_{n}x{m}")
        for g in plans
    }
    dev = torch.device("cuda")
    rows = torch.empty(n, 2, device=dev, dtype=F32)  # (mean, m2) between the unfused passes
    work = [  # segment states, their counts, and the rows' finished-segment counters (kept 0)
        (
            torch.empty(g.ptasks, g.k, device=dev, dtype=F32),
            torch.empty(g.ptasks, device=dev, dtype=I32),
            torch.zeros(n, device=dev, dtype=I32),
        )
        for g in plans
    ]
    passes = list(zip(mods.values(), work))

    def run(X):
        out = torch.empty(n, device=X.device, dtype=F32)
        for mod, (P, PC, C) in passes:
            mod(X, rows, out, P, PC, C)
        return (out,)

    run.modules = mods
    run.plans = plans
    return run
