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
"""The normalized policy loss of every row of (A, G), in one pass.

``loss = -(1/m) Σ_j G_j (A_j - mean A) / scale``, with ``scale`` the sample
standard deviation of the row of A plus 1e-8.

The algorithm is chain_ops' ``normalized_policy_loss_fused``: the moments of a
row in one pass. Here a row's state is a set of sums around a reference ``K``
(the mean of the row's first 32 values, ``row_reference``): ``n``, ``Σa``,
``Σg``, ``Σ (a - K) g`` and ``Σ (a - K)²``. States of parts of a row add up,
and the row's result follows from its sums (``raw_finish``): with
``δ = Σa - m K`` and ``μ = δ / m``, the mean is ``K + μ``,
``M2 = Σ (a - K)² - δ μ`` and the co-moment ``C = Σ (a - ā) g =
Σ (a - K) g - μ Σg``, so the result is ``-C / (m * (sqrt(M2 / (m - 1)) + 1e-8))``
(correctly rounded float32 division and square root).

**Why the sums are exact.** A one-pass fold sees each part of a row before the
row's mean, so whatever it centres a part on is off from the row's mean, by
about ``sd / sqrt(c)`` for a part's own mean of ``c`` pairs, and ``C`` carries
that offset times the part's ``Σ g`` until the parts meet, where it cancels
(exactly, in exact arithmetic). Large ``g`` at samples whose ``a`` is the row's
mean make it large: an RL batch's padded tokens (whitened advantages, A = 0 =
the mean, G = a fill of -1e2 to -1e6 on 10 to 70% of a row), or one outlier
``G``. torch.compile's two passes centre on the row's float32 mean, so their
error is that mean's error times ``Σ g``. With float32 sums the kernel's error
was each part's rounding times its ``Σ g``, several times as much: the fused
kernel failed the gate's rule on padded rows by up to 3.6 times (5.4 at
1 x 77777, 8 at 1 x 33554432), and on an outlier ``G`` by up to 19 times. An
emulation of the kernel's arithmetic (every candidate checked against
torch.compile on these rows) showed that only exact sums suffice everywhere:
an exact mean alone still left 0.7 of the bound on padded rows, and float32
sums per 16-byte vector failed the split size and the outliers. So every sum
here is a float pair (double-float arithmetic in float32: TwoSum, TwoDiff,
TwoProd by FMA, in ``__fadd_rn`` / ``__fsub_rn`` / ``__fmul_rn`` /
``__fmaf_rn``, which neither TVM nor nvcc rewrites), exact to about 2^-44 of
its terms, except ``Σ (a - R)²`` of a lane's step, which is positive and
centred on the lane's own step mean ``R`` and so needs only float32:

* on the streaming paths every lane keeps its running sums, step by step; the
  products are centred exactly on the lane's own step mean (``lane_step``,
  ``lane_close``: no step waits for a team's), so a constant row's are all 0,
  as its loss must be (its scale is the 1e-8 alone, so anything left in C is
  multiplied by 1e8 / m). A warp adds its lanes' sums by exact butterflies at
  the end of its span (``span_state``), and the warps' and segments' states add
  up (``raw_merge``: no products, no divisions);
* on the static path a team folds its whole row, its products centred exactly
  on the row's exact mean (``fold_static``); ``Σ g`` only corrects for the
  mean's low part, so it is a float32 sum.

The exact sums cost about 2 us at the long and split bench sizes (256 x 262144:
179.9 -> 181.9 us, 1 x 33554432: 97.7 -> 100.0 us; 65536 x 256: 48.8 -> 48.3 us),
and the unfused passes' 0.5 to 0.9%: every formulation tried -- TwoSum chains,
Fast2Sum on sums biased by a power of two, centring each step on the warp's
exact mean, loading the next step during a step's arithmetic -- timed the same
or worse, and float32 lane sums with everything else unchanged timed as before.

How it is laid out (``plan``):

* ``static`` (short rows that split evenly over a team of 32, 16 or 8 lanes, a
  power of two of 16-byte vectors a lane, at most ``PER_MAX``; 65536 x 256):
  a team per row, ``256 / tpr`` rows per 256-thread CTA, a CTA per group of
  rows. Every lane loads its whole share at once (16-byte ``ld.global.cs``,
  ``tpr`` vectors apart, so a team's loads are contiguous) and the team folds
  the row; the team's first lane writes the result. No shared memory, no
  barrier, no merge;
* ``team`` (other short rows: unaligned or uneven): a warp per row, each lane
  loading 32 values a step into its running sums (``stream_exact``);
* ``cta`` (long rows): a row is cut into ``S`` segments of whole CTA steps
  (``256 * u`` vectors), as equal as the steps allow; one 256-thread CTA takes
  a (row, segment) work item, and every CTA of the grid is resident
  (``CTAS_PER_SM``, held by the launch bounds). ``S`` is 1 when there are
  enough rows to fill the GPU, else every row is split over enough CTAs to
  fill it. A thread loads 32 values of A and of G a step (``U = 8`` plain
  16-byte loads of each, a CTA's loads contiguous) into its running sums.
  The 8 warps' states then add up (a butterfly over 8 lanes, ``merge_warps_eq``
  when they hold as many pairs, as at every bench size). With ``S > 1`` thread
  0 stores the segment's state and counts the row's finished segments with one
  ``atom.acq_rel.gpu``; the CTA that counts the last one adds the row's ``S``
  states -- in warp 0 alone for ``S <= 32``, by a tree of the CTA when they are
  256 or 512 of equal counts -- writes the result and resets the counter. One
  launch either way.

A last, partial step of a span is masked (``load_tail``): its missing vectors
are not loaded and count for nothing. When ``m`` is not a multiple of 4 the
rows are not 16-byte aligned and every path reads one element at a time
instead (decided when the kernel is built).

The segment states and finished-segment counters are cached per device and
CUDA stream (``kernels.common.workspace``): calls on two streams may overlap on
the GPU, and with one shared counter they merged each other's segments. Calls
on one stream run in order, launch by launch, so they share them (every call
leaves its counters 0).

``variant="unfused"`` is the ablation: the unfused program's three passes on
the same layouts and loads -- the mean of A (a sum), then ``Σ (a - mean)²``
(which gives ``scale``), then the loss sum ``Σ (g (a - mean)) / scale`` -- each
a launch that reads A (and, for the last, G) from HBM again, with the same
exact sums as the fused kernel (the mean a pair; the loss's numerators
``g (a - mean)`` summed exactly and divided by ``m scale`` once; ``mean_team``,
``loss_team``). Only the per-row mean and scale travel
between them, in a buffer each call allocates: one call is three launches,
and two host threads calling on one stream interleave them. Their sums need
no team per step: on the streaming paths each thread keeps its own
(``Stage.lanes`` = 1), merged across the warp or CTA at the end.
"""

from __future__ import annotations

import dataclasses

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.memory import vector_width, vectorisable
from .common.reduce import cta_allreduce
from .common.workspace import workspace

THREADS = 256
WARPS = THREADS // 32
FULL_MASK = 0xFFFFFFFF
I32 = torch.int32
TEAM_ROW_MAX = 2048  # rows up to this long are reduced by a team of at most one warp
PER_MAX = 8  # vectors of A (and of G) a lane of a static team holds at once
TEAM_SIZES = (32, 16, 8)  # lanes per row on the static path, the largest that fits first
# Per step on the streaming paths, a thread of the fused stage loads U_VALUES values of A and of
# G (8 16-byte loads of each, or 32 unaligned ones): a warp folds 1024 pairs a step. The unfused
# passes' threads, which sum alone, load U_TEAM vectors a step on the warp-per-row path and U on
# the long-row path.
U_VALUES = 32
U_TEAM = 4
U = 8
CTAS_PER_SM = 2  # resident 256-thread CTAs per SM on the long-row path (launch bounds)
MIN_STEPS = 4  # a split segment gives every thread at least this many steps
# The 16-byte load (LOADS): short rows stream through L1 and L2 evict-first (``.cs``), long rows
# load plainly. The timer's L2 flush (256 MB zeroed) decides it: it leaves ~30 MB of dirty lines
# that the next kernel's misses write back, ~9.5 us of HBM time that every implementation pays.
# Evict-first lines keep about half of them resident in a call that streams up to ~134 MB
# (65536 x 256: 53 -> 49 us); at 268 MB they are all written back either way (no hint helps
# 1 x 33554432), and at 537 MB evict-first costs 3% (256 x 262144: 179.7 -> 185.8 us). With the
# dirty lines written back before the call, every bench size runs at 100-104% of the HBM floor.
COP_SHORT = 1
COP_LONG = 0

KERNEL_META = {
    "op": "normalized_policy_loss",
    "dtype": "float32",
    "algorithm": "one pass, exact sums (float pairs: TwoSum, TwoDiff, TwoProd) of a, g, "
    "(a - K) g and (a - K)^2 around the row's reference K (the mean of its first 32 values), "
    "each lane's products centred exactly on its own step mean R (Σ (a - R)² in float32); parts' "
    "sums add; mean, M2 and the co-moment C from the row's sums; result "
    "-C / (m (sqrt(M2 / (m - 1)) + 1e-8)); short rows folded whole by a team, centred on the "
    "row's exact mean",
    "structure": "short rows: a team of 32 lanes per row, each lane's share loaded at once "
    "(16-byte ld.global.cs), the team folds the row; long rows: a 256-thread CTA per (row, "
    "segment), all CTAs resident (2 per SM), 16-byte loads 8 deep into each lane's exact running "
    "sums, the warps' sums added; rows too few to fill the GPU are split over CTAs and the last "
    "CTA of a row (one acq_rel atomic) adds them: one launch; workspaces per CUDA stream, the "
    "unfused passes' (mean, scale) rows per call; the unfused passes' sums exact too",
}

# The scratch of the fused stage's streaming fold (``ex``, a local array of EX floats):
# * a lane's step: its sums of a, g and (a - R) g (pairs), Σ d², its mean R and its count;
# * the lane's running sums: Σa, Σg, Σ (a - K) g, Σ (a - K)² (pairs) and its count;
# * the row's reference K (the same in every lane of every CTA of the row);
# * scratch: two pairs, and a pair for shuffles (``team_dd``).
LA, LG, LP, LD, LR, LC = 0, 2, 4, 6, 7, 8
RA, RG, RQ, RM, RN = 9, 11, 13, 15, 17
EK = 18
ET, EU, EH = 19, 21, 23
EX = 25
EA = LA  # fold_static's exact Σa (a pair)


def _fdiv(a, b):
    """``a / b`` rounded to nearest, whatever ``--use_fast_math`` makes of ``/``."""
    return T.call_pure_extern("__fdiv_rn", a, b, ty="float32")


def _fsqrt(a):
    """``sqrt(a)`` rounded to nearest (``--use_fast_math`` makes ``sqrtf`` approximate)."""
    return T.call_pure_extern("__fsqrt_rn", a, ty="float32")


# TwoSum's steps must run as written: these calls are opaque to TVM's simplifier, and nvcc never
# contracts them into an FMA (which would round a product other than TwoSum assumes).
def _fadd(a, b):
    return T.call_pure_extern("__fadd_rn", a, b, ty="float32")


def _fsub(a, b):
    return T.call_pure_extern("__fsub_rn", a, b, ty="float32")


def _fmul(a, b):
    return T.call_pure_extern("__fmul_rn", a, b, ty="float32")


def _fma(a, b, c):
    """``a * b + c`` rounded once: with c = -fl(a b), the exact error of the product (TwoProd)."""
    return T.call_pure_extern("__fmaf_rn", a, b, c, ty="float32")


def _f32_rest(m: int) -> float:
    """m - float32(m): 0 unless the integer m > 2^24 has no float32; then exactly a float32."""
    return float(m - int(torch.tensor(float(m), dtype=F32).item()))


# --- sums of a thread's registers, and of a team's ------------------------------------------
@T.inline
def tree4(acc, r, w):
    """acc[r, 0] <- acc[r, 0] + ... + acc[r, w - 1], as a tree (w is 1 or 4)."""
    if w == 4:
        acc[r, 0] = (acc[r, 0] + acc[r, 1]) + (acc[r, 2] + acc[r, 3])


@T.inline
def team_add(acc, rows, width):
    """acc[r, 0] <- its sum over each aligned group of ``width`` lanes, for every r < rows: a
    butterfly, so every lane of a group ends with the same sums (acc[r, 1] is scratch)."""
    for i in range(T.meta_var(width.bit_length() - 1)):
        for r in T.unroll(rows):
            acc[r, 1] = T.gpu_warp_shuffle_xor(FULL_MASK, acc[r, 0], width >> (i + 1), 32, 32)
        for r in T.unroll(rows):
            acc[r, 0] = acc[r, 0] + acc[r, 1]


# --- float pairs (double-float arithmetic in float32) ---------------------------------------
# A pair (x[i], x[i + 1]) stands for x[i] + x[i + 1]. The low part takes the exact rounding error
# of every sum into the high part (TwoSum) and of every product (TwoProd: fma(a, b, -fl(a b)));
# its own roundings are about 2^-24 of it, so a pair is exact to about 2^-44 of its terms.
@T.inline
def dd_acc(x, i, v):
    """(x[i], x[i + 1]) += v (TwoSum). ``v`` must not read x[i + 1]."""
    s = _fadd(x[i], v)
    b = _fsub(s, x[i])
    x[i + 1] = x[i + 1] + _fadd(_fsub(x[i], _fsub(s, b)), _fsub(v, b))
    x[i] = s


@T.inline
def dd_add(x, i, yh, yl):
    """(x[i], x[i + 1]) += (yh, yl). ``yh`` must not read x[i + 1]."""
    s = _fadd(x[i], yh)
    b = _fsub(s, x[i])
    x[i + 1] = (x[i + 1] + yl) + _fadd(_fsub(x[i], _fsub(s, b)), _fsub(yh, b))
    x[i] = s


@T.inline
def dd_mul(x, i, ah, al, bh, bl):
    """(x[i], x[i + 1]) <- (ah, al) * (bh, bl). The factors must not read x[i] or x[i + 1]."""
    p = _fmul(ah, bh)
    x[i + 1] = _fma(ah, bh, T.float32(0) - p) + (ah * bl + al * bh)
    x[i] = p


@T.inline
def dd_norm(x, i):
    """The pair (x[i], x[i + 1]) with x[i] the float32 rounding of its sum (TwoSum)."""
    s = _fadd(x[i], x[i + 1])
    b = _fsub(s, x[i])
    x[i + 1] = _fadd(_fsub(x[i], _fsub(s, b)), _fsub(x[i + 1], b))
    x[i] = s


@T.inline
def team_dd(x, i, width):
    """The pair (x[i], x[i + 1]) <- the sum of the pairs of each aligned group of ``width`` lanes,
    normalised, the same in every lane of the group: a butterfly of TwoSums (the error of
    ``own + other`` is exact either way round, so both lanes get the same bits). x[EH], x[EH + 1]
    are scratch."""
    for lvl in range(T.meta_var(width.bit_length() - 1)):
        x[EH] = T.gpu_warp_shuffle_xor(FULL_MASK, x[i], width >> (lvl + 1), 32, 32)
        x[EH + 1] = T.gpu_warp_shuffle_xor(FULL_MASK, x[i + 1], width >> (lvl + 1), 32, 32)
        dd_add(x, i, x[EH], x[EH + 1])
    dd_norm(x, i)


@T.inline
def lane_exact_a(x, i, ba, nv):
    """The pair (x[i], x[i + 1]) <- Σ ba[0 .. nv), exactly (a TwoSum chain)."""
    x[i] = ba[0]
    x[i + 1] = T.float32(0)
    for k in T.unroll(nv - 1):
        dd_acc(x, i, ba[k + 1])


# --- the fused stage: a lane's exact running sums ---------------------------------------------
# A lane's step, in two passes over its registers: its exact Σa (a TwoSum chain) gives its own
# mean R; then each value is centred on R exactly (TwoDiff: d + dl = a - R), and the lane adds
# Σ (a - R) g exactly (TwoProd of d g, plus dl g) and Σ g exactly, and Σ d² in float32. Around
# the row's reference K (``row_reference``) its running sums take Σa, Σg,
# Σ (a - K) g += Σ (a - R) g + (R - K) Σ g and
# Σ (a - K)² += Σ d² + c (R - K)² + 2 (R - K) (Σa - c R)
# (exact pairs but Σ d²). The products are centred so that a constant row's are all exactly 0:
# its loss is 0, and its scale is the 1e-8 alone, so any rounding left in C is multiplied by
# 1e8 / m (an uncentred Σ a g - K Σ g, equal in exact arithmetic, missed the gate's rule there by
# 1300 to 2900 times). (Uncentred sums, and Fast2Sum on sums biased by a power of two above
# 64 max|v|, timed the same.)
@T.inline
def lane_step(ex, ba, bg, nv, inv, ok):
    """A lane's step: its ``nv`` pairs in ba, bg (nv a multiple of 4; a missing one is 0 in both
    and ``ok(k)`` = 0), ``inv`` about 1 / max(its count, 1). ex[LA] <- Σa, ex[LP] <- Σ (a - R) g,
    ex[LG] <- Σ g, exactly (pairs); ex[LR] <- R = the lane's float32 mean; ex[LD] <- Σ (a - R)²
    over its values (float32)."""
    ex[LA] = T.float32(0)
    ex[LA + 1] = T.float32(0)
    for k in T.unroll(nv):
        dd_acc(ex, LA, ba[k])
    ex[LR] = (ex[LA] + ex[LA + 1]) * inv
    for j in T.unroll(4):
        ex[LG + j] = T.float32(0)  # the pairs of Σ g and Σ (a - R) g
        ex[ET + j] = T.float32(0)  # 4 float32 partial sums of d² (ET .. EU + 1 are free here)
    for i in T.unroll(nv // 4):
        for j in T.unroll(4):
            d0 = _fsub(ba[i * 4 + j], ex[LR])
            b = _fsub(d0, ba[i * 4 + j])
            dl = _fadd(_fsub(ba[i * 4 + j], _fsub(d0, b)), _fsub(T.float32(0) - ex[LR], b))
            d = d0 * ok(i * 4 + j)
            p = _fmul(d, bg[i * 4 + j])
            dd_acc(ex, LP, p)
            ex[LP + 1] = ex[LP + 1] + _fma(
                dl * ok(i * 4 + j), bg[i * 4 + j], _fma(d, bg[i * 4 + j], T.float32(0) - p)
            )  # fmt: skip
            dd_acc(ex, LG, bg[i * 4 + j])
            ex[ET + j] = ex[ET + j] + d * d
    ex[LD] = (ex[ET] + ex[ET + 1]) + (ex[ET + 2] + ex[ET + 3])


@T.inline
def lane_close(ex, c):
    """The lane's step sums into its running sums (``c`` its count of the step): Σa, Σg,
    Σ (a - K) g += Σ (a - R) g + τ Σg and Σ (a - K)² += Σ d² + c τ² + 2 τ (Σa - c R), with
    τ = R - K exactly."""
    th = _fsub(ex[LR], ex[EK])
    tb = _fsub(th, ex[LR])
    tl = _fadd(_fsub(ex[LR], _fsub(th, tb)), _fsub(T.float32(0) - ex[EK], tb))
    dd_add(ex, RA, ex[LA], ex[LA + 1])
    dd_add(ex, RG, ex[LG], ex[LG + 1])
    dd_add(ex, RQ, ex[LP], ex[LP + 1])
    dd_mul(ex, ET, th, tl, ex[LG], ex[LG + 1])
    dd_add(ex, RQ, ex[ET], ex[ET + 1])
    cr = _fmul(c, ex[LR])
    wr = (ex[LA] - cr) + (ex[LA + 1] - _fma(c, ex[LR], T.float32(0) - cr))  # Σa - c R, tiny
    dd_acc(ex, RM, ex[LD])
    dd_mul(ex, ET, th, tl, th, tl)
    dd_mul(ex, EU, ex[ET], ex[ET + 1], c, T.float32(0))
    dd_add(ex, RM, ex[EU], ex[EU + 1] + T.float32(2) * th * wr)
    ex[RN] = ex[RN] + c


@T.inline
def row_reference(ex, A, row, t, g):
    """ex[EK] <- K, the mean of the row's first 32 values (repeating its last when it is
    shorter): every warp of every CTA of the row computes the same bits, so their sums around K
    add up plainly. Any mean of the row's values serves: the float pairs lose to K's distance from
    the row's mean at most log2(m) of their ~48 bits (Σ (a - K)² - n (ā - K)²)."""
    ex[EK] = A[row, T.min(t % 32, g.m - 1)]
    for i in T.unroll(5):
        ex[EK] = ex[EK] + T.gpu_warp_shuffle_xor(FULL_MASK, ex[EK], 16 >> i, 32, 32)
    ex[EK] = ex[EK] * T.float32(1.0 / 32)


@T.inline
def step_full(ex, ba, bg, nv, g):
    """A whole step (``nv`` pairs a lane) into the lane's running sums."""
    lane_step(ex, ba, bg, nv, T.float32(1.0 / nv), lambda k: T.float32(1))
    lane_close(ex, T.float32(nv))


@T.inline
def span_state(st, ex):
    """st <- the warp's raw state from its lanes' running sums (exact butterflies), the same in
    every lane: (n, Σa, Σg, Σ (a - K) g, Σ (a - K)², K), the sums pairs."""
    team_dd(ex, RA, 32)
    team_dd(ex, RG, 32)
    team_dd(ex, RQ, 32)
    team_dd(ex, RM, 32)
    for i in T.unroll(5):
        ex[RN] = ex[RN] + T.gpu_warp_shuffle_xor(FULL_MASK, ex[RN], 16 >> i, 32, 32)
    st[0] = ex[RN]
    for j in T.unroll(8):
        st[1 + j] = ex[RA + j]
    st[9] = ex[EK]


@T.inline
def fold_static(ot, ba, bg, nv, width, c, rv, acc, ex):
    """ot <- the state of a row a team of ``width`` lanes holds whole (``c`` pairs, a power of two;
    ``nv`` in each lane's ba, bg), the same in every lane: (c, mh, ml, M2, -, -, C, C_lo) for
    ``static_finish``. The row's exact Σa gives its mean mh + ml; each value is centred on mh
    exactly (TwoDiff: d + dl = a - mh) and P = Σ (a - mh) g is summed exactly (TwoProd), so
    C = P - ml Σ g; M2 = Σ d² - c ml² and Σ g are float32 sums. (With float32 products, a row's
    G far from 0 put each d's rounding times g in C: 0.5 of the gate's bound at 9 x 32 for a G
    of 1000; centring G as well put padded rows, whose G is two clusters, back at 0.84.)"""
    lane_exact_a(ex, EA, ba, nv)
    team_dd(ex, EA, width)
    mh = ex[EA] * T.float32(1.0 / c)
    ml = ex[EA + 1] * T.float32(1.0 / c)
    ex[LP] = T.float32(0)
    ex[LP + 1] = T.float32(0)
    w = T.meta_var(min(4, nv))
    for j in T.unroll(w):
        acc[0, j] = T.float32(0)  # Σ d²
        acc[1, j] = T.float32(0)  # Σ g
    for i in T.unroll(nv // w):
        for j in T.unroll(w):
            a_ = ba[i * w + j]
            g_ = bg[i * w + j]
            d0 = _fsub(a_, mh)
            b = _fsub(d0, a_)
            dl = _fadd(_fsub(a_, _fsub(d0, b)), _fsub(T.float32(0) - mh, b))
            p = _fmul(d0, g_)
            dd_acc(ex, LP, p)
            ex[LP + 1] = ex[LP + 1] + _fma(dl, g_, _fma(d0, g_, T.float32(0) - p))
            acc[0, j] = acc[0, j] + d0 * d0
            acc[1, j] = acc[1, j] + g_
    tree4(acc, 0, w)
    tree4(acc, 1, w)
    team_dd(ex, LP, width)
    team_add(acc, 2, width)
    ot[0] = T.float32(c)
    ot[1] = mh
    ot[2] = ml
    ot[3] = acc[0, 0] - T.float32(c) * ml * ml
    ot[6] = ex[LP] - ml * acc[1, 0]
    ot[7] = ex[LP + 1]


@T.inline
def static_finish(Out, RV, row, st, m):
    """Out[row] = -C / (m * (sqrt(M2 / (m - 1)) + 1e-8)) from ``fold_static``'s state."""
    scale = _fsqrt(_fdiv(st[3], T.float32(m - 1))) + T.float32(1e-8)
    Out[row] = T.float32(0) - _fdiv(st[6] + st[7], T.float32(m) * scale)


# --- the fused state (n, Σa, Σg, Σ (a - K) g, Σ (a - K)², K): sums around the row's K ---------
@T.inline
def raw_merge(st, ot):
    """st <- st ⊕ ot: the counts and the pairs add up (both around the row's K). An empty state
    (``set_empty``: count 0, K 0) takes the other's K."""
    st[9] = T.if_then_else(st[0] > T.float32(0), st[9], ot[9])
    st[0] = st[0] + ot[0]
    for j in T.unroll(4):
        dd_add(st, 1 + 2 * j, ot[1 + 2 * j], ot[2 + 2 * j])


@T.inline
def raw_merge_eq(st, ot, c):
    """``raw_merge`` of two states of ``c`` pairs each (the counts are not read)."""
    for j in T.unroll(4):
        dd_add(st, 1 + 2 * j, ot[1 + 2 * j], ot[2 + 2 * j])
    st[0] = T.float32(2 * c)


@T.inline
def raw_finish(Out, RV, row, st, m):
    """Out[row] = -C / (m * (sqrt(M2 / (m - 1)) + 1e-8)) from the row's sums around K: with
    δ = Σa - m K and μ = δ / m (pairs), M2 = Σ (a - K)² - δ μ and C = Σ (a - K) g - μ Σg.
    m is a pair too, fm + fml (fml = 0 unless m > 2^24 has no float32)."""
    fm = T.float32(m)
    fml = T.float32(_f32_rest(m))
    k_ = st[9]
    mk = _fmul(fm, k_)
    mke = _fma(fm, k_, T.float32(0) - mk) + fml * k_
    dh = _fsub(st[1], mk)
    b1 = _fsub(dh, st[1])
    dl = _fadd(_fsub(st[1], _fsub(dh, b1)), _fsub(T.float32(0) - mk, b1)) + (st[2] - mke)
    q1 = _fdiv(dh, fm)
    q2 = _fdiv(_fma(T.float32(0) - q1, fm, dh) + dl - q1 * fml, fm)
    p = _fmul(dh, q1)
    pe = _fma(dh, q1, T.float32(0) - p) + (dh * q2 + dl * q1)
    m2 = (st[7] - p) + (st[8] - pe)
    ug = _fmul(q1, st[3])
    uge = _fma(q1, st[3], T.float32(0) - ug) + (q1 * st[4] + q2 * st[3])
    c2 = _fsub(st[5], ug)
    b3 = _fsub(c2, st[5])
    cl = _fadd(_fsub(st[5], _fsub(c2, b3)), _fsub(T.float32(0) - ug, b3)) + (st[6] - uge)
    scale = _fsqrt(_fdiv(m2, T.float32(m - 1))) + T.float32(1e-8)
    Out[row] = T.float32(0) - _fdiv(c2 + cl, fm * scale)


# --- the unfused passes' states ---------------------------------------------------------------
# The same exact sums as the fused stage's, so the two variants differ only in their passes: the
# mean of A from its exact sum (a pair, mean_h + mean_l: the float32 mean's rounding times Σ g
# cost the unfused program up to 2.5 times the gate's bound on 0/1 advantages at 3 x 4099);
# Σ (a - mean_h)² in float32 (its terms are positive); and the loss's numerators g (a - mean_h)
# summed exactly (TwoDiff, TwoProd), less mean_l Σ g, divided by m scale once. Rounded term by
# term, ((g (a - mean)) / scale, as the unfused program forms them), they missed the gate's rule
# by 1.07 to 1.25 times where G sits 1000 from 0, and their float32 sum by 1.4 to 1.8 times at
# 1 x 33554432 (0/1 advantages, a loss ~1/sqrt(m) of its terms).
@T.inline
def add_merge(st, ot):
    st[0] = st[0] + ot[0]


@T.inline
def add_merge_eq(st, ot, c):
    st[0] = st[0] + ot[0]


@T.inline
def pair_merge(st, ot):
    """st <- st ⊕ ot for the states (pair[, float]) of the mean and loss passes."""
    dd_add(st, 0, ot[0], ot[1])
    st[2] = st[2] + ot[2]


@T.inline
def pair_merge_eq(st, ot, c):
    pair_merge(st, ot)


@T.inline
def square_fold(ot, ba, bg, nv, rv, acc):
    """acc[0, 0] <- Σ (a - mean)² over a lane's values."""
    w = T.meta_var(min(4, nv))
    for j in T.unroll(w):
        acc[0, j] = (ba[j] - rv[0]) * (ba[j] - rv[0])
    for i in T.unroll(nv // w - 1):
        for j in T.unroll(w):
            acc[0, j] = acc[0, j] + (ba[(i + 1) * w + j] - rv[0]) * (ba[(i + 1) * w + j] - rv[0])
    tree4(acc, 0, w)


# The sums of a team's values, the same in every lane (width 1: a thread's own).
@T.inline
def mean_team(ot, ba, bg, nv, width, c, rv, acc, ex):
    """ot <- (Σ a exactly, as a pair; 0)."""
    lane_exact_a(ex, EA, ba, nv)
    team_dd(ex, EA, width)
    ot[0] = ex[EA]
    ot[1] = ex[EA + 1]
    ot[2] = T.float32(0)


@T.inline
def square_team(ot, ba, bg, nv, width, c, rv, acc, ex):
    square_fold(ot, ba, bg, nv, rv, acc)
    team_add(acc, 1, width)
    ot[0] = acc[0, 0]


@T.inline
def loss_team(ot, ba, bg, nv, width, c, rv, acc, ex):
    """ot <- (Σ g (a - mean_h) exactly, as a pair; Σ g in float32), with rv[0] = mean_h: each
    a - mean_h exactly (TwoDiff: d + dl), each product exactly (TwoProd of d g, plus dl g). (The
    first unfused loss pass formed g ((a - mean) (1 / scale)), whose rounded factor takes one value
    per distinct a: with 0/1 advantages its error was the same in every term and grew with the
    row, 4 to 88 times the gate's bound.)"""
    ex[LA] = T.float32(0)
    ex[LA + 1] = T.float32(0)
    acc[0, 0] = T.float32(0)
    for k in T.unroll(nv):
        d0 = _fsub(ba[k], rv[0])
        b = _fsub(d0, ba[k])
        dl = _fadd(_fsub(ba[k], _fsub(d0, b)), _fsub(T.float32(0) - rv[0], b))
        p = _fmul(d0, bg[k])
        dd_acc(ex, LA, p)
        ex[LA + 1] = ex[LA + 1] + _fma(dl, bg[k], _fma(d0, bg[k], T.float32(0) - p))
        acc[0, 0] = acc[0, 0] + bg[k]
    team_dd(ex, LA, width)
    team_add(acc, 1, width)
    ot[0] = ex[LA]
    ot[1] = ex[LA + 1]
    ot[2] = acc[0, 0]


@T.inline
def no_prologue(rv, RV, A, G, row):
    rv[0] = T.float32(0)


@T.inline
def mean_prologue(rv, RV, A, G, row):
    rv[0] = RV[row, 0]


@T.inline
def mean_finish(Out, RV, row, st, m):
    """RV[row, 0:2] <- the row's mean Σa / m as a pair (mean_h, mean_l)."""
    fm = T.float32(m)
    q1 = _fdiv(st[0], fm)
    RV[row, 0] = q1
    RV[row, 1] = _fdiv(
        _fma(T.float32(0) - q1, fm, st[0]) + st[1] - q1 * T.float32(_f32_rest(m)), fm
    )  # fmt: skip


@T.inline
def scale_finish(Out, RV, row, st, m):
    RV[row, 2] = _fsqrt(_fdiv(st[0], T.float32(m - 1))) + T.float32(1e-8)


@T.inline
def loss_finish(Out, RV, row, st, m):
    """Out[row] = -(Σ g (a - mean_h) - mean_l Σ g) / (m scale)."""
    Out[row] = T.float32(0) - _fdiv(
        (st[0] - RV[row, 1] * st[2]) + st[1], T.float32(m) * RV[row, 2]
    )  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Stage:
    """One pass over the rows: its state, and how it starts, folds, merges and ends a row."""

    name: str
    k: int  # floats in the state
    lo: int  # the first entry an equal-count merge reads (the fused count is known there)
    reads_g: bool  # does the pass load G
    # lanes that fold a step together on the streaming paths: 32 (a warp: the fused stage's
    # exact running sums, ``stream_exact``) or 1 (a thread's running sum, ``stream_lanes``)
    lanes: int
    prologue: object  # prologue(rv, RV, A, G, row): the earlier passes' values (unfused)
    # team(ot, ba, bg, nv, width, c, rv, acc, ex): a team's c values (a static team's whole row,
    # or a thread's own vectors), in every lane
    team: object
    merge: object  # merge(st, ot): st <- st ⊕ ot, any counts
    merge_eq: object  # merge_eq(st, ot, c): st <- st ⊕ ot, both of c values
    finish: object  # finish(Out, RV, row, st, m): the row's result from a merged state
    finish_static: object  # the same from a static team's ``team`` state


# The unfused passes' sums need no team per step: each thread keeps its own running sum
# (``stream_lanes``), merged across the warp or the CTA at the end.
FUSED = Stage(
    "fused", 10, 1, True, 32, no_prologue, fold_static, raw_merge, raw_merge_eq, raw_finish,
    static_finish,
)  # fmt: skip
MEAN = Stage(
    "mean", 3, 0, False, 1, no_prologue, mean_team, pair_merge, pair_merge_eq, mean_finish,
    mean_finish,
)  # fmt: skip
SQUARES = Stage(
    "squares", 1, 0, False, 1, mean_prologue, square_team, add_merge, add_merge_eq, scale_finish,
    scale_finish,
)  # fmt: skip
LOSS = Stage(
    "loss", 3, 0, True, 1, mean_prologue, loss_team, pair_merge, pair_merge_eq,
    loss_finish, loss_finish,
)  # fmt: skip
VARIANTS = {"fused": (FUSED,), "unfused": (MEAN, SQUARES, LOSS)}


# --- loads ----------------------------------------------------------------------------------
@T.inline
def load_vec(buf, u, X, row, v, vec):
    """buf[u*vec : u*vec+vec] = X[row, v*vec : v*vec+vec]: one 16-byte load."""
    buf[T.Ramp(u * vec, 1, vec)] = X[row, T.Ramp(v * vec, 1, vec)]


@T.inline
def load_cs(buf, u, X, row, v, vec):
    """The same load as ``ld.global.cs`` (streaming: evict first from L1 and L2)."""
    T.ptx.ld.global_.cs.v4.f32(
        buf[u * 4], buf[u * 4 + 1], buf[u * 4 + 2], buf[u * 4 + 3], X.ptr_to([row, v * 4])
    )


@T.inline
def load_scalar(buf, u, X, row, v, vec):
    """buf[u] = X[row, v]: the unaligned rows (vec == 1)."""
    buf[u] = X[row, v]


LOADS = (load_vec, load_cs)


@T.inline
def set_empty(st, k):
    for j in T.unroll(k):
        st[j] = T.float32(0)


# --- a team's steps over a span of a row ----------------------------------------------------
@T.inline
def load_step(ba, bg, A, G, row, v, p, g):
    """The ``u`` vectors v, v + p, ... of A (and of G) into ba (and bg)."""
    for u in T.unroll(g.u):
        g.load(ba, u, A, row, v + u * p, g.vec)
    if g.stage.reads_g:
        for u in T.unroll(g.u):
            g.load(bg, u, G, row, v + u * p, g.vec)


@T.inline
def load_tail(ba, bg, okv, A, G, row, v, p, hi, g):
    """``load_step`` for a partial step: vector v + u p counts (okv[u] = 1) when it is below hi;
    a missing one is loaded from a valid address and its values are 0 (okv[u] = 0)."""
    for u in T.unroll(g.u):
        okv[u] = T.if_then_else(v + u * p < hi, T.float32(1), T.float32(0))
        g.load(ba, u, A, row, T.min(v + u * p, hi - 1), g.vec)
        g.load(bg, u, G, row, T.min(v + u * p, hi - 1), g.vec)
    for k in T.unroll(g.u * g.vec):
        ba[k] = T.if_then_else(okv[k // g.vec] > T.float32(0), ba[k], T.float32(0))
        bg[k] = T.if_then_else(okv[k // g.vec] > T.float32(0), bg[k], T.float32(0))


@T.inline
def tail_exact(A, G, row, v, p, hi, ex, ba, bg, okv, g):
    """The lane's vectors of the last, partial step (its first: v) into its running sums,
    masked: the lane's count is its own (it may be 0, which adds nothing)."""
    load_tail(ba, bg, okv, A, G, row, v, p, hi, g)
    ex[LC] = okv[0]
    for u in T.unroll(g.u - 1):
        ex[LC] = ex[LC] + okv[u + 1]
    ex[LC] = ex[LC] * T.float32(g.vec)
    lane_step(
        ex, ba, bg, g.u * g.vec, _fdiv(T.float32(1), T.max(ex[LC], T.float32(1))),
        lambda k: okv[k // g.vec],
    )  # fmt: skip
    lane_close(ex, ex[LC])


@T.inline
def no_tail(A, G, row, v, p, hi, ex, ba, bg, okv, g):
    """Every span is whole steps: no partial step is ever left."""
    T.evaluate(0)


@T.inline
def stream_exact(A, G, row, t, p, lo, hi, st, ot, rv, acc, ba, bg, okv, ex, g):
    """st <- the state of the warp's vectors of [lo, hi), the same in every lane (the fused
    stage): step by step, thread t loads the ``u`` vectors base + t, base + t + p, ... and adds
    them to its exact running sums around the row's K (``step_full``); a last, partial step is
    masked. The lanes' sums then make the warp's state (``span_state``)."""
    for j in T.unroll(EX):
        ex[j] = T.float32(0)
    row_reference(ex, A, row, t, g)
    base: T.int32 = lo
    while base + g.u * p <= hi:
        load_step(ba, bg, A, G, row, base + t, p, g)
        step_full(ex, ba, bg, g.u * g.vec, g)
        base += g.u * p
    if base < hi:
        g.tail(A, G, row, base + t, p, hi, ex, ba, bg, okv, g)
    span_state(st, ex)


@T.inline
def fold_one(A, G, row, v, st, ot, rv, acc, ba, bg, ex, g):
    """st <- st + the thread's own sum of vector v alone (the vectors left after the whole steps).

    The caller's loop advances ``v``: an ``@T.inline`` that assigns to an argument binds a
    new local instead (``v += p`` here would never end the loop)."""
    s = g.stage
    g.load(ba, 0, A, row, v, g.vec)
    if s.reads_g:
        g.load(bg, 0, G, row, v, g.vec)
    s.team(ot, ba, bg, g.vec, 1, 0, rv, acc, ex)
    s.merge(st, ot)


@T.inline
def stream_lanes(A, G, row, t, p, lo, hi, st, ot, rv, acc, ba, bg, okv, ex, g):
    """st <- this thread's own sum over its vectors of [lo, hi) (the unfused passes): steps of
    ``u`` vectors ``p`` apart from lo + t, then one at a time what is left of a partial step."""
    s = g.stage
    set_empty(st, s.k)
    v: T.int32 = lo + t
    while v + (g.u - 1) * p < hi:
        load_step(ba, bg, A, G, row, v, p, g)
        s.team(ot, ba, bg, g.u * g.vec, 1, 0, rv, acc, ex)
        s.merge(st, ot)
        v += g.u * p
    while v < hi:
        fold_one(A, G, row, v, st, ot, rv, acc, ba, bg, ex, g)
        v += p


@T.inline
def no_gather(st, ot, width, s):
    """The warp's state is already the same in all its lanes."""
    T.evaluate(0)


@T.inline
def butterfly(st, ot, width, s):
    """Merge the states of each aligned group of ``width`` lanes; every lane gets the group's."""
    for i in range(T.meta_var(width.bit_length() - 1)):
        for j in T.unroll(s.k):
            ot[j] = T.gpu_warp_shuffle_xor(FULL_MASK, st[j], width >> (i + 1), 32, 32)
        s.merge(st, ot)


@T.inline
def butterfly_eq(st, ot, width, c, s):
    """``butterfly`` for states of ``c`` values each: every merge of equal counts."""
    for i in range(T.meta_var(width.bit_length() - 1)):
        for j in T.unroll(s.k - s.lo):
            ot[j + s.lo] = T.gpu_warp_shuffle_xor(FULL_MASK, st[j + s.lo], width >> (i + 1), 32, 32)
        s.merge_eq(st, ot, c << i)


# --- the short-row layouts ------------------------------------------------------------------
@T.inline
def team_static(A, G, RV, Out, P, C, bx, tx, g):
    """A team of ``tpr`` lanes per row, every lane's ``per`` vectors loaded at once; the team
    folds the whole row."""
    s = g.stage
    team = tx // g.tpr
    tl = tx % g.tpr
    st = T.alloc_local((s.k,), "float32")
    rv = T.alloc_local((2,), "float32")
    acc = T.alloc_local((4, 4), "float32")
    ex = T.alloc_local((EX,), "float32")
    ba = T.alloc_local((g.nv,), "float32")
    bg = T.alloc_local((g.nv,), "float32")
    grp: T.int32 = bx
    while grp < g.groups:
        row: T.int32 = T.min(grp * g.teams + team, g.n - 1)
        s.prologue(rv, RV, A, G, row)
        for j in T.unroll(g.per):
            g.load(ba, j, A, row, tl + j * g.tpr, g.vec)
        if s.reads_g:
            for j in T.unroll(g.per):
                g.load(bg, j, G, row, tl + j * g.tpr, g.vec)
        s.team(st, ba, bg, g.nv, g.tpr, g.m, rv, acc, ex)
        if tl == 0:
            if grp * g.teams + team < g.n:
                s.finish_static(Out, RV, row, st, g.m)
        grp += g.blocks


@T.inline
def team_stream(A, G, RV, Out, P, C, bx, tx, g):
    """A warp per row: steps of ``u`` vectors a lane into the warp's exact running sums (the
    unfused sums: each lane's own, merged by a butterfly at the end)."""
    s = g.stage
    team = tx // 32
    tl = tx % 32
    st = T.alloc_local((s.k,), "float32")
    ot = T.alloc_local((s.k,), "float32")
    rv = T.alloc_local((2,), "float32")
    acc = T.alloc_local((4, 4), "float32")
    ex = T.alloc_local((EX,), "float32")
    ba = T.alloc_local((g.u * g.vec,), "float32")
    bg = T.alloc_local((g.u * g.vec,), "float32")
    okv = T.alloc_local((g.u,), "float32")
    grp: T.int32 = bx
    while grp < g.groups:
        row: T.int32 = T.min(grp * g.teams + team, g.n - 1)
        s.prologue(rv, RV, A, G, row)
        g.stream(A, G, row, tl, 32, 0, g.units, st, ot, rv, acc, ba, bg, okv, ex, g)
        g.gather(st, ot, 32, s)
        if tl == 0:
            if grp * g.teams + team < g.n:
                s.finish(Out, RV, row, st, g.m)
        grp += g.blocks


# --- the long-row layout: merging a CTA's states ---------------------------------------------
@T.inline
def merge_warps_eq(st, ot, smem, warp, lane, c, g):
    """Every thread <- the CTA's state, from its warps' (each the same in all its lanes, of ``c``
    values): lane l takes warp l % 8's and a butterfly over 8 lanes merges them, every merge of
    equal counts (``merge_eq``). Ends with a barrier."""
    s = g.stage
    if lane == 0:
        for j in T.unroll(s.k):
            smem[warp, j] = st[j]
    T.cuda.cta_sync()
    for j in T.unroll(s.k):
        st[j] = smem[lane % WARPS, j]
    butterfly_eq(st, ot, WARPS, c, s)
    T.cuda.cta_sync()


@T.inline
def merge_warps_any(st, ot, smem, warp, lane, c, g):
    """``merge_warps_eq`` whatever the warps' counts (``merge``)."""
    s = g.stage
    if lane == 0:
        for j in T.unroll(s.k):
            smem[warp, j] = st[j]
    T.cuda.cta_sync()
    for j in T.unroll(s.k):
        st[j] = smem[lane % WARPS, j]
    butterfly(st, ot, WARPS, s)
    T.cuda.cta_sync()


@T.inline
def reduce_any(st, ot, smem, warp, lane, c, g):
    """Every thread <- the CTA's state from its threads' own, whatever the counts
    (``cta_allreduce``)."""
    cta_allreduce(st, ot, g.stage.k, g.stage.merge, smem, warp, lane, WARPS)


@T.inline
def reduce_eq(st, ot, smem, warp, lane, c, g):
    """Every thread <- the CTA's state when every thread's own state holds ``c`` values: a
    butterfly in each warp, then lane l takes warp l % 8's state and a butterfly over 8 lanes
    merges them. Every merge is of two equal counts (``merge_eq``). Ends with a barrier."""
    s = g.stage
    butterfly_eq(st, ot, 32, c, s)
    if lane == 0:
        for j in T.unroll(s.k):
            smem[warp, j] = st[j]
    T.cuda.cta_sync()
    for j in T.unroll(s.k):
        st[j] = smem[lane % WARPS, j]
    butterfly_eq(st, ot, WARPS, c * 32, s)
    T.cuda.cta_sync()


# --- the end of a work item -----------------------------------------------------------------
@T.inline
def end_row(Out, RV, P, C, st, ot, smem, flag, old, row, q, tx, warp, lane, g):
    """S == 1: the CTA's state is the row's."""
    if tx == 0:
        g.stage.finish(Out, RV, row, st, g.m)


@T.inline
def end_segment(Out, RV, P, C, st, ot, smem, flag, old, row, q, tx, warp, lane, g):
    """Publish the segment's state; the CTA that ends the row's last segment merges them all.

    Thread 0 stores the state and counts the row's finished segments with one acquire-release
    atomic: its release publishes the store, and the CTA that counts the last segment acquires
    the others', so after the barrier its threads read the states with plain loads (the
    pattern of CUTLASS's semaphore)."""
    s = g.stage
    if tx == 0:
        for j in T.unroll(s.k):
            P[(row * g.S + q) * s.k + j] = st[j]
        T.ptx.atom.acq_rel.gpu.global_.add.s32(old[0], C.ptr_to([row]), T.int32(1))
        flag[0] = T.if_then_else(old[0] == g.S - 1, 1, 0)
    T.cuda.cta_sync()
    if flag[0] == 1:
        g.final(Out, RV, P, C, st, ot, smem, row, tx, warp, lane, g)


@T.inline
def final_any(Out, RV, P, C, st, ot, smem, row, tx, warp, lane, g):
    """The row's S states: thread t merges states t, t + 256, ..., then the CTA's (``merge``)."""
    s = g.stage
    set_empty(st, s.k)
    for i in T.unroll(g.slots):
        if i * THREADS + tx < g.S:
            for j in T.unroll(s.k):
                ot[j] = P[(row * g.S + i * THREADS + tx) * s.k + j]
            s.merge(st, ot)
    cta_allreduce(st, ot, s.k, s.merge, smem, warp, lane, WARPS)
    if tx == 0:
        s.finish(Out, RV, row, st, g.m)
        C[row] = 0


@T.inline
def final_warp(Out, RV, P, C, st, ot, smem, row, tx, warp, lane, g):
    """The row's S <= 32 states: lane l of warp 0 takes state l, a butterfly merges them."""
    s = g.stage
    if warp == 0:
        set_empty(st, s.k)
        if lane < g.S:
            for j in T.unroll(s.k):
                st[j] = P[(row * g.S + lane) * s.k + j]
        butterfly(st, ot, g.width, s)
        if lane == 0:
            s.finish(Out, RV, row, st, g.m)
            C[row] = 0


@T.inline
def final_eq(Out, RV, P, C, st, ot, smem, row, tx, warp, lane, g):
    """The row's S = 256 or 512 states of ``cseg`` values each: thread t takes states t (and
    t + 256), then the CTA's; every merge is of equal counts (``merge_eq``)."""
    s = g.stage
    for j in T.unroll(s.k):
        st[j] = P[(row * g.S + tx) * s.k + j]
    for i in range(T.meta_var(g.S // THREADS - 1)):
        for j in T.unroll(s.k):
            ot[j] = P[(row * g.S + (i + 1) * THREADS + tx) * s.k + j]
        s.merge_eq(st, ot, g.cseg)
    reduce_eq(st, ot, smem, warp, lane, g.cseg * (g.S // THREADS), g)
    if tx == 0:
        s.finish(Out, RV, row, st, g.m)
        C[row] = 0


@T.inline
def cta_rows(A, G, RV, Out, P, C, bx, tx, g):
    """A 256-thread CTA per (row, segment) work item; the grid walks the items."""
    s = g.stage
    warp = tx // 32
    lane = tx % 32
    st = T.alloc_local((s.k,), "float32")
    ot = T.alloc_local((s.k,), "float32")
    rv = T.alloc_local((2,), "float32")
    acc = T.alloc_local((4, 4), "float32")
    ex = T.alloc_local((EX,), "float32")
    ba = T.alloc_local((g.u * g.vec,), "float32")
    bg = T.alloc_local((g.u * g.vec,), "float32")
    okv = T.alloc_local((g.u,), "float32")
    old = T.alloc_local((1,), "int32")
    smem = T.alloc_shared((WARPS, s.k), "float32")
    flag = T.alloc_shared((1,), "int32")
    w: T.int32 = bx
    while w < g.work:
        row: T.int32 = w // g.S
        q: T.int32 = w % g.S
        lo: T.int32 = (q * g.steps // g.S) * (THREADS * g.u)
        hi: T.int32 = T.min(((q + 1) * g.steps // g.S) * (THREADS * g.u), g.units)
        s.prologue(rv, RV, A, G, row)
        g.stream(A, G, row, tx, THREADS, lo, hi, st, ot, rv, acc, ba, bg, okv, ex, g)
        g.reduce(st, ot, smem, warp, lane, g.cfold, g)
        g.end(Out, RV, P, C, st, ot, smem, flag, old, row, q, tx, warp, lane, g)
        w += g.blocks


# --- the layout -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Plan:
    """The static layout of one pass on one size: everything the kernel reads from ``g``."""

    n: int
    m: int
    stage: Stage
    vec: int  # elements per load: 4 (16 bytes) or 1
    path: str  # "static", "team" or "cta"
    tpr: int  # lanes per row (team paths), THREADS on the cta path
    per: int  # static path: vectors of A (and of G) per lane
    S: int  # cta path: segments per row
    blocks: int  # CTAs in the grid
    min_blocks: int  # CTAs per SM the launch bounds promise
    threads: int = THREADS  # threads per CTA
    cop: int = 0  # the 16-byte load: plain or .cs (LOADS)
    u: int = U  # streaming paths: vectors of A (and of G) per step

    units = property(lambda g: g.m // g.vec)  # loads per row
    nv = property(lambda g: g.per * g.vec)  # static path: values a lane holds
    teams = property(lambda g: g.threads // g.tpr)  # rows per CTA on the team paths
    groups = property(lambda g: -(-g.n // g.teams))  # CTA-sized row groups
    work = property(lambda g: g.n * g.S)  # cta path: (row, segment) items
    steps = property(lambda g: -(-g.units // (THREADS * g.u)))  # CTA steps per row
    slots = property(lambda g: -(-g.S // THREADS))  # segment states a thread merges at the end
    width = property(lambda g: 1 << (g.S - 1).bit_length())  # final_warp: lanes merged
    stride = property(lambda g: THREADS if g.path == "cta" else 32)  # a thread's vectors apart
    cstep = property(lambda g: g.stage.lanes * g.u * g.vec)  # values the lanes take a step
    # cta path: every segment is whole steps, as many each, so every warp (or thread) holds
    # ``cfold`` values and every segment ``cseg``: then no merge of a CTA's or a row's states
    # divides
    eq = property(lambda g: g.units % (THREADS * g.u) == 0 and g.steps % g.S == 0)
    cfold = property(lambda g: (g.steps // g.S) * g.cstep)
    cseg = property(lambda g: THREADS * (g.steps // g.S) * g.u * g.vec)
    ws = property(lambda g: max(1, g.n * g.S * g.stage.k) if g.S > 1 else 1)
    load = property(lambda g: LOADS[g.cop] if g.vec > 1 else load_scalar)
    body = property(lambda g: {"static": team_static, "team": team_stream, "cta": cta_rows}[g.path])
    stream = property(lambda g: stream_exact if g.stage.lanes == 32 else stream_lanes)
    tail = property(lambda g: tail_exact if g.units % (g.stride * g.u) else no_tail)
    gather = property(lambda g: no_gather if g.stage.lanes == 32 else butterfly)  # team path
    reduce = property(
        lambda g: (merge_warps_eq if g.eq else merge_warps_any)
        if g.stage.lanes == 32
        else (reduce_eq if g.eq else reduce_any)
    )
    end = property(lambda g: end_segment if g.S > 1 else end_row)
    final = property(
        lambda g: final_warp
        if g.S <= 32
        else (final_eq if g.eq and g.S in (THREADS, 2 * THREADS) else final_any)
    )


def _segments(n: int, steps: int, slots: int) -> int:
    """Segments per row: enough work items to fill ``slots`` resident CTAs, each of at least
    MIN_STEPS steps, preferring a count that divides the steps (equal segments)."""
    if n >= slots:
        return 1
    want = max(1, min(slots // n, steps // MIN_STEPS))
    even = [s for s in range(want, (want * 9) // 10, -1) if steps % s == 0]
    return even[0] if even else want


def plan(n: int, m: int, stage: Stage = FUSED, **over) -> Plan:
    """The layout of one pass; ``over`` overrides any choice (for experiments)."""
    vec = vector_width("float32") if vectorisable(m, "float32") else 1
    units = m // vec
    if m <= TEAM_ROW_MAX:

        def even(t):  # t lanes take the row evenly, a power of two of vectors each (fold_static)
            per = units // t
            return units % t == 0 and 1 <= per <= PER_MAX and per & (per - 1) == 0

        fits = [t for t in TEAM_SIZES if even(t)]
        threads = over.get("threads", THREADS)
        cop = over.get("cop", COP_SHORT)
        mb = over.get("min_blocks", 1)
        u = over.get("u", U_VALUES // vec if stage.lanes > 1 else U_TEAM)
        if fits:
            tpr = over.get("tpr", fits[0])
            g = Plan(n, m, stage, vec, "static", tpr, units // tpr, 1, 0, mb, threads, cop, u)
        else:
            g = Plan(n, m, stage, vec, "team", 32, 0, 1, 0, mb, threads, cop, u)
        return dataclasses.replace(g, blocks=over.get("blocks", g.groups))
    per_sm = over.get("min_blocks", CTAS_PER_SM)
    u = over.get("u", U_VALUES // vec if stage.lanes > 1 else U)
    slots = SMS * per_sm
    steps = -(-units // (THREADS * u))
    S = over.get("S", _segments(n, steps, slots))
    blocks = over.get("blocks", min(n * S, slots))
    cop = over.get("cop", COP_LONG)
    return Plan(n, m, stage, vec, "cta", THREADS, 0, S, blocks, per_sm, THREADS, cop, u)


def _kernel(g: Plan):
    """One pass as one kernel: (A, G, RV, Out, P, C)."""
    n, m, body = g.n, g.m, g.body

    @T.function
    def policy_pass(
        A: T.Tensor((n, m), "float32", align=16),
        G: T.Tensor((n, m), "float32", align=16),
        RV: T.Tensor((n, 3), "float32"),
        Out: T.Tensor((n,), "float32"),
        P: T.Tensor((g.ws,), "float32"),
        C: T.Tensor((n,), "int32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(g.blocks,), block=(g.threads,)),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=g.min_blocks),
        )
        body(A, G, RV, Out, P, C, T.cuda.block_idx("x"), T.cuda.thread_idx("x"), g)

    return policy_pass


def compile_plan(g: Plan):
    return compile_sm90(_kernel(g), name=f"normalized_policy_loss_{g.stage.name}_{g.n}x{g.m}")


def make_run(plans, mods):
    """``run(A, G)`` over the passes, on torch's current CUDA stream.

    Each pass's segment states and the rows' finished-segment counters belong to the device and
    stream of the call (``workspace``): a launch uses and resets them, launches on one stream run
    in order, so they share them (every launch leaves the counters 0), calls on two streams may
    overlap and get their own, and a call captured in a CUDA graph gets new ones that the graph
    keeps (with their zeroing). The (mean_h, mean_l, scale) rows the unfused passes hand on from
    one launch to the next are the call's own (``torch.empty``, from the stream-ordered caching
    allocator): one call is three launches, and two host threads calling on one stream
    interleave them, so a shared buffer gave a pass the other call's mean or scale. The fused
    pass reads no rows; it gets a cached placeholder."""
    n = plans[0].n
    key = ("normalized_policy_loss", n, tuple(g.ws for g in plans))
    passes = len(plans)

    def alloc(dev):
        rows = torch.empty(n, 3, device=dev, dtype=F32) if passes == 1 else None
        work = [  # segment states, and the rows' finished-segment counters (kept 0)
            (torch.empty(g.ws, device=dev, dtype=F32), torch.zeros(n, device=dev, dtype=I32))
            for g in plans
        ]
        return rows, work

    def run(A, G):
        dev = A.device
        with torch.cuda.device(dev):  # TVM FFI launches on the current stream of A's device
            rows, work = workspace(key, lambda: alloc(dev))
            if rows is None:
                rows = torch.empty(n, 3, device=dev, dtype=F32)
            out = torch.empty(n, device=dev, dtype=F32)
        for mod, (P, C) in zip(mods, work):
            mod(A, G, rows, out, P, C)
        return (out,)

    run.plans = plans
    run.modules = mods
    return run


@cached
def build(cfg: dict, variant: str = "fused"):
    n, m = cfg["n"], cfg["m"]
    plans = [plan(n, m, stage) for stage in VARIANTS[variant]]
    return make_run(plans, [compile_plan(g) for g in plans])
