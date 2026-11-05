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
"""Σ_j (x_j + ‖x‖ e_j)² of every row, in one pass over X and E, in TIRx.

The algorithm is chain_ops' ``householder_fused``: the square expands to
``Σx² + 2‖x‖ Σxe + ‖x‖² Σe²``, so the state of a row is the three inner
products ``(xx, xe, ee)``; two states merge by adding them, and the result of
a row is ``xx + 2 sqrt(xx) xe + xx ee``.

``variant="unfused"`` is the ablation with the unfused program's two passes,
built from the same parts: pass 1 reads X for ``‖x‖ = sqrt(Σx²)`` (one float
per row to HBM), pass 2 reads X and E again for ``Σ(x + ‖x‖ e)²``.

Every pass is one *row pass* (``_row_kernel``), laid out by ``plan``:

* a *group* of threads reduces one row: a warp for short rows
  (``m <= WARP_MAX_M``, 8 rows per 256-thread CTA), the whole 256-thread CTA
  for long ones; a warp or CTA all-reduce, and the group's thread 0 writes;
* the grid walks the rows (``w += stride``) when there are more than
  ``GRID_CAP`` CTAs of work;
* few long rows (``n < SPLIT_BELOW``) are cut into ``splits`` chunks, one group
  per chunk writes its partial state to an ``(n, splits, k)`` workspace, and a
  second kernel (``_finish_kernel``, one warp per row) adds them and applies the
  epilogue.

Every pass reads X and E 16 bytes (4 floats) at a time, for every ``m``: the
tensors are viewed as flat arrays, and a row ``[row*m, row*m + m)`` is its
aligned 16-byte span read as vectors, plus at most 3 elements before it and 3
after it read one at a time (``edges``; none when ``m`` is a multiple of 4).
Each thread issues ``unroll`` independent 16-byte loads per tensor before it
folds them (``accumulate``).
"""

from __future__ import annotations

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.reduce import cta_allreduce, warp_allreduce

VEC = 4  # float32 elements per 16 bytes
THREADS = 256
WARPS = THREADS // 32
CTAS_PER_SM = 2048 // THREADS  # resident CTAs per SM at full occupancy
GRID_CAP = SMS * CTAS_PER_SM  # at most one full wave of CTAs; more work is walked
WARP_MAX_M = 1024  # rows up to this long take one warp each
UNROLL_MAX = 4  # independent 16-byte loads per tensor in flight per thread
MIN_CHUNK = THREADS * VEC * 8  # a split chunk gives each thread at least 8 vectors per tensor
SPLIT_BELOW = SMS * 2  # split rows when there are fewer rows than this
SPLIT_TARGET = SMS * 4  # aim for about this many CTAs when splitting

KERNEL_META = {
    "op": "householder",
    "dtype": "float32",
    "algorithm": "one pass, state (xx, xe, ee) = (Σx², Σxe, Σe²), out = xx + 2 sqrt(xx) xe + xx ee",
    "structure": "one warp per row (m <= 1024) or one 256-thread CTA per row, split over CTAs "
    "plus a merge kernel when rows are few; 4 x 16-byte loads of X and E in flight per thread",
}


# --- the passes: number of states, the fold of one element pair, the epilogue ---
@T.inline
def fold_fused(st, x, e, a):
    st[0] = st[0] + x * x
    st[1] = st[1] + x * e
    st[2] = st[2] + e * e


@T.inline
def final_fused(st):
    return st[0] + T.float32(2.0) * T.sqrt(st[0]) * st[1] + st[0] * st[2]


@T.inline
def fold_ssq(st, x, e, a):
    st[0] = st[0] + x * x


@T.inline
def final_ssq(st):
    return T.sqrt(st[0])


@T.inline
def fold_vv(st, x, e, a):
    st[0] = st[0] + (x + a * e) * (x + a * e)


@T.inline
def final_vv(st):
    return st[0]


@T.inline
def add1(st, ot):
    st[0] = st[0] + ot[0]


@T.inline
def add3(st, ot):
    for j in T.unroll(3):
        st[j] = st[j] + ot[j]


@T.inline
def load_xe(xb, eb, Xf, Ef, v, i):
    """Vector ``v`` of X and E into slot ``i`` of the registers."""
    xb[T.Ramp(i * VEC, 1, VEC)] = Xf[T.Ramp(v * VEC, 1, VEC)]
    eb[T.Ramp(i * VEC, 1, VEC)] = Ef[T.Ramp(v * VEC, 1, VEC)]


@T.inline
def load_x(xb, eb, Xf, Ef, v, i):
    """Vector ``v`` of X only (a pass that does not read E)."""
    xb[T.Ramp(i * VEC, 1, VEC)] = Xf[T.Ramp(v * VEC, 1, VEC)]


@T.inline
def aux_row(A, row):
    return A[row]


@T.inline
def aux_none(A, row):
    return T.float32(0)


# (states, fold, epilogue, loads E, reads the per-row aux input)
MODES = {
    "fused": (3, fold_fused, final_fused, True, False),
    "ssq": (1, fold_ssq, final_ssq, False, False),
    "vv": (1, fold_vv, final_vv, True, True),
}


@T.inline
def edges(st, Xf, Ef, a, lo, hi, s, t, fold):
    """The at most 3 elements before and after the 16-byte span of ``[lo, hi)``, by chunk 0."""
    hs = T.min(((lo + (VEC - 1)) // VEC) * VEC, hi)
    ts = T.max((hi // VEC) * VEC, hs)
    if s == 0:
        if t < hs - lo:
            fold(st, Xf[lo + t], Ef[lo + t], a)
        if t < hi - ts:
            fold(st, Xf[ts + t], Ef[ts + t], a)


@T.inline
def no_edges(st, Xf, Ef, a, lo, hi, s, t, fold):
    """Rows of a multiple of 4 elements are 16-byte spans: nothing to add."""
    T.evaluate(0)


@T.inline
def accumulate(st, Xf, Ef, xb, eb, a, lo, hi, s, splits, t, nthr, unroll, load, fold, edge):
    """Fold chunk ``s`` of ``splits`` of the flat range ``[lo, hi)`` into thread ``t``'s state.

    ``nthr`` threads (``t`` in ``[0, nthr)``) share the range. Its 16-byte
    aligned span ``[4 va, 4 vb)`` is read as vectors, split into ``splits``
    contiguous vector chunks; each thread takes every ``nthr``-th vector,
    ``unroll`` at a time (all loads issued before the folds), then the rest one
    at a time. ``edge`` adds the elements outside the span.
    """
    va = (lo + (VEC - 1)) // VEC
    vb = hi // VEC
    cv = (T.max(vb - va, 0) + (splits - 1)) // splits
    v_hi = T.min(va + s * cv + cv, vb)
    v: T.int32 = va + s * cv + t
    while v + (unroll - 1) * nthr < v_hi:
        for i in T.unroll(unroll):
            load(xb, eb, Xf, Ef, v + i * nthr, i)
        for i in T.unroll(unroll):
            for k in T.unroll(VEC):
                fold(st, xb[i * VEC + k], eb[i * VEC + k], a)
        v += unroll * nthr
    while v < v_hi:
        load(xb, eb, Xf, Ef, v, 0)
        for k in T.unroll(VEC):
            fold(st, xb[k], eb[k], a)
        v += nthr
    edge(st, Xf, Ef, a, lo, hi, s, t, fold)


@T.inline
def write_row(Out, st, w, splits, k, final):
    """Work item ``w`` is a whole row: write its result."""
    Out[w] = final(st)


@T.inline
def write_partial(P, st, w, splits, k, final):
    """Work item ``w`` is chunk ``w % splits`` of row ``w // splits``: write its partial state."""
    for j in T.unroll(k):
        P[w // splits, w % splits, j] = st[j]


@T.inline
def reduce_warp(st, ot, k, combine, smem, warp, lane):
    warp_allreduce(st, ot, k, combine)


@T.inline
def reduce_cta(st, ot, k, combine, smem, warp, lane):
    cta_allreduce(st, ot, k, combine, smem, warp, lane, WARPS)


def plan(n: int, m: int) -> dict:
    """The layout of a row pass over (n, m): threads per row, chunks per row, loads in flight."""
    group = 32 if m <= WARP_MAX_M else THREADS
    splits = 1
    if group == THREADS and n < SPLIT_BELOW:
        splits = max(1, min(-(-SPLIT_TARGET // n), m // MIN_CHUNK))
    per_thread = -(-(m // VEC) // (group * splits))  # vectors per thread per chunk
    unroll = max(1, min(UNROLL_MAX, per_thread))
    return {"group": group, "splits": splits, "unroll": unroll}


def _row_kernel(n: int, m: int, mode: str, group: int, splits: int, unroll: int):
    """One pass over the rows of X (and E): ``mode`` says which states it reduces."""
    k, fold, final, with_e, with_aux = MODES[mode]
    total = n * m
    groups = THREADS // group  # rows (or chunks) per CTA at a time
    work = n * splits
    blocks = min(-(-work // groups), GRID_CAP)
    stride = blocks * groups
    load = load_xe if with_e else load_x
    edge = no_edges if m % VEC == 0 else edges
    aux = aux_row if with_aux else aux_none
    write = write_row if splits == 1 else write_partial
    reduce = reduce_warp if group == 32 else reduce_cta
    combine = add3 if k == 3 else add1

    @T.inline
    def body(X, E, A, Out, bx, tx):
        warp = tx // 32
        lane = tx % 32
        t = tx % group
        Xf = T.decl_tensor(X.data, ty_args=[T.Tensor((total,), "float32", align=16)])
        Ef = T.decl_tensor(E.data, ty_args=[T.Tensor((total,), "float32", align=16)])
        st = T.alloc_local((k,), "float32")
        ot = T.alloc_local((k,), "float32")
        xb = T.alloc_local((unroll * VEC,), "float32")
        eb = T.alloc_local((unroll * VEC,), "float32")
        smem = T.alloc_shared((WARPS, k), "float32")
        w: T.int32 = bx * groups + tx // group
        while w < work:
            for j in T.unroll(k):
                st[j] = T.float32(0)
            row = w // splits
            a: T.float32 = aux(A, row)
            accumulate(
                st,
                Xf,
                Ef,
                xb,
                eb,
                a,
                row * m,
                row * m + m,
                w % splits,
                splits,
                t,
                group,
                unroll,
                load,
                fold,
                edge,
            )
            reduce(st, ot, k, combine, smem, warp, lane)
            if t == 0:
                write(Out, st, w, splits, k, final)
            w += stride

    out_shape = (n,) if splits == 1 else (n, splits, k)
    launch = T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,))
    if mode == "fused":

        @T.function
        def householder_fused(
            X: T.Tensor((n, m), "float32", align=16),
            E: T.Tensor((n, m), "float32", align=16),
            Out: T.Tensor(out_shape, "float32"),
        ):
            T.device_entry(launch=launch)
            body(X, E, Out, Out, T.cuda.block_idx("x"), T.cuda.thread_idx("x"))

        return householder_fused
    if mode == "ssq":

        @T.function
        def householder_ssq(
            X: T.Tensor((n, m), "float32", align=16), Out: T.Tensor(out_shape, "float32")
        ):
            T.device_entry(launch=launch)
            body(X, X, Out, Out, T.cuda.block_idx("x"), T.cuda.thread_idx("x"))

        return householder_ssq

    @T.function
    def householder_vv(
        X: T.Tensor((n, m), "float32", align=16),
        E: T.Tensor((n, m), "float32", align=16),
        A: T.Tensor((n,), "float32"),
        Out: T.Tensor(out_shape, "float32"),
    ):
        T.device_entry(launch=launch)
        body(X, E, A, Out, T.cuda.block_idx("x"), T.cuda.thread_idx("x"))

    return householder_vv


def _finish_kernel(n: int, splits: int, mode: str):
    """Add the ``splits`` partial states of every row (one warp per row) and apply the epilogue."""
    k, _, final, _, _ = MODES[mode]
    combine = add3 if k == 3 else add1
    blocks = min(-(-n // WARPS), GRID_CAP)
    stride = blocks * WARPS

    @T.function
    def householder_finish(P: T.Tensor((n, splits, k), "float32"), Out: T.Tensor((n,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        warp = tx // 32
        lane = tx % 32
        st = T.alloc_local((k,), "float32")
        ot = T.alloc_local((k,), "float32")
        row: T.int32 = bx * WARPS + warp
        while row < n:
            for j in T.unroll(k):
                st[j] = T.float32(0)
            s: T.int32 = lane
            while s < splits:
                for j in T.unroll(k):
                    ot[j] = P[row, s, j]
                combine(st, ot)
                s += 32
            warp_allreduce(st, ot, k, combine)
            if lane == 0:
                Out[row] = final(st)
            row += stride

    return householder_finish


def _pass(n: int, m: int, mode: str):
    """Compile one row pass.

    Returns ``launch(*ins, out)``, which writes the (n,) result to ``out``.
    """
    p = plan(n, m)
    splits = p["splits"]
    tag = f"{n}x{m}_g{p['group']}_s{splits}_u{p['unroll']}"
    mod = compile_sm90(
        _row_kernel(n, m, mode, p["group"], splits, p["unroll"]), name=f"householder_{mode}_{tag}"
    )
    if splits == 1:
        return lambda *args: mod(*args)
    fin = compile_sm90(
        _finish_kernel(n, splits, mode), name=f"householder_finish_{mode}_{n}_s{splits}"
    )
    k = MODES[mode][0]
    part = {}  # the partial states, one workspace per device, reused from call to call

    def launch(*args):
        *ins, out = args
        ws = part.get(out.device)
        if ws is None:
            ws = part[out.device] = torch.empty((n, splits, k), device=out.device, dtype=F32)
        mod(*ins, ws)
        fin(ws, out)

    return launch


@cached
def build(cfg: dict, variant: str = "fused"):
    assert variant in ("fused", "unfused"), variant
    n, m = cfg["n"], cfg["m"]
    assert n * m < 2**31, "flat int32 indexing"

    if variant == "fused":
        fused = _pass(n, m, "fused")

        def run(X, E):
            out = torch.empty(n, device=X.device, dtype=F32)
            fused(X, E, out)
            return (out,)

        return run

    ssq = _pass(n, m, "ssq")
    vv = _pass(n, m, "vv")
    norms = {}  # ‖x‖ of every row, the one per-row result between the passes

    def run_unfused(X, E):
        nrm = norms.get(X.device)
        if nrm is None:
            nrm = norms[X.device] = torch.empty(n, device=X.device, dtype=F32)
        out = torch.empty(n, device=X.device, dtype=F32)
        ssq(X, nrm)
        vv(X, E, nrm, out)
        return (out,)

    return run_unfused
