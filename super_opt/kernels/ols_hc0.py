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
"""HC0 (White) sandwich variance of the no-intercept OLS slope of every row, in one pass.

For a row of observations ``(x, y)`` the slope is ``theta = Σxy / Σx²`` and the
result is ``Σ x² (y - theta x)² / (Σx²)²``; both are 0 when ``Σx² == 0``.

The algorithm is chain_ops' ``ols_hc0_fused``: the state of a set of pairs is

    (a, b, h, d, e) = (Σx², Σxy, s(a) Σ x² (y - t x)², Σx⁴ / a, Σ x³ (y - t x) / a),

``t = theta(a, b) = b / a`` (0 when ``a == 0``) the set's own slope: the meat
``h`` and ``e`` are kept around the running slope. Two states merge by moving
both to the merged slope ``t = (b1 + b2) / (a1 + a2)``: with ``δi = t - ti``,
``y - t x = (y - ti x) - δi x``, so

    h = r1 h1 + r2 h2 + s δ1 a1 (δ1 d1 - 2 e1) + s δ2 a2 (δ2 d2 - 2 e2)
    e = e1 - δ1 d1 + w ((e2 - δ2 d2) - (e1 - δ1 d1)),    d = d1 + w (d2 - d1)

with ``s = s(a1 + a2)``, ``ri = s / s(ai)``, ``w = a2 / (a1 + a2)``, and ``a``, ``b``
add (``hc0_combine``). The empty state (all zeros) is an exact identity (``w``
is exactly 1 or 0 when a side is empty). A row's result is ``h / (s(a) a²)``,
rounded once (``store_over_product``).

Range. ``s(a) = 2^(4 - E)``, ``E`` the exponent of ``a`` (``_scale``), is the power
of two with ``s a`` in [16, 32): ``h`` is the meat per unit of ``a`` (over a chunk's
16 pairs), but scaled exactly, so the rescalings ``ri`` and the result's
quotients add no rounding and the arithmetic is that of the raw sums. ``d`` and
``e`` are Σx⁴ and Σx³(y - tx) per unit of ``a`` (x²-weighted means of x² and of
x (y - tx)). No value of the state, nor any product formed on the way, has a
degree above 2 in x and y (a chunk's terms are ``(u s) u``, ``u = x (y - t x)``;
the moves ``δ (s a) (δ d - 2 e)``): they have the range of ``x²``, ``x y`` or
``(y - t x)²``. The raw meat has degree 4: its terms fall below 2^-126 when x
and y are both ~1e-9, where ``--use_fast_math`` flushes them to 0 (torch keeps
them as subnormals), and data in such units lost the meat (8.8e-5 of the
result at 1e-9, 70% at 1e-10). Now scaling X, Y or both by a power of two
scales the result exactly (2^-40 to 2^50 on the gate's data; below, the
smallest x² of a sample start to flush), and the error stays within 1e-6 down
to X and Y both ~3e-17, where terms below 2^-126 begin to carry a share of the
meat (torch's own program fails from ~1e-12, where its ``a²`` underflows), and
up to a row's ``Σx²`` of 2^126, where ``1 / a`` is no longer normal (the unfused
program's ``a²`` overflows from ``a > 2^64``; the raw meat from Σ (x r)² > 2^128).
``d`` and ``e`` reach the result only through the moves, so the chunk's ``1 / a``
and the merge's ``w`` are ``rcp.approx`` reciprocals (within an ulp, and a power
of two times the reciprocal of the significand, so the scaling stays exact);
correctly rounded ones changed no error systematically and cost 0.2-1.1% more.

The state carries its slope ``t`` as a sixth float, the correctly rounded
``__fdiv_rn(b, a)`` it was formed with, so a merge divides once (the merged
slope) instead of three times; the values are those of recomputing it.

Precision:

* ``h`` is never formed as a difference of raw sums (``Σx²y² - ...``); every
  state's ``h`` is the residual sum around its own slope, so a merge's
  corrections are of the size of the change of slope. A chunk of pairs read
  by a thread is folded by two passes in registers (its slope, then its
  residuals) with balanced-tree sums.
* ``h`` is a sum, not a mean: a merge adds the sides' (exactly rescaled) ``h``.
  Kept as an ``a``-weighted mean instead, a row whose meat sits in one term (a
  1000-sigma outlier) carried that term through a rounded weight at every
  merge, and lost 2.5 ulp (RMS) of the result.
* ``a`` enters the result squared, and in float32 the roundings of the last,
  largest merges of ``a`` and ``h`` dominate whatever the merge order (~1 ulp
  each, 5e-7 of the result at (4096, 65536)). So ``a``, ``b`` and ``h`` carry
  the exact rounding errors of their merges (TwoSum) as three more floats of
  the state: ``K = 9`` with the slope.
* ``h`` is the meat around the float32 slope of the final state. The finish
  moves it to the slope of the compensated ``a`` and ``b`` (by ``δ a (δ d - 2 e)``,
  ``δ a`` the exact FMA residual ``b - t a`` plus the low parts). For good fits,
  where the residuals are small next to ``y``, that slope error is what dominates
  otherwise.
* the finish is float32 throughout: each quotient is correctly rounded and its
  remainder is exact by FMA, so the low parts (``ca``, ``ch``, the slope
  correction) reach the one final rounding. The H800's float64 divisions
  (once per row, by one lane) held every warp of the short rows 2% longer.
* states are merged as balanced trees (butterflies across lanes and warps).

How it is laid out (``stream``, shared by every pass of both variants):

* a *team* of ``tpr`` threads takes a *task*, a row or a segment of a row: a
  warp per row for short rows (no shared memory, no barrier), the whole CTA
  of 256 threads for long ones. The grid walks the tasks; every kernel is
  compiled with ``__launch_bounds__(256, CTAS_PER_SM)``, so its occupancy is
  known when the grid is sized.
* rows too few to fill the GPU are split over ``S`` CTAs, all resident at once.
  The CTA publishes its segment's state, fences, and counts the row's finished
  segments with an atomic; the CTA that finishes the last one merges the
  row's ``S`` states, writes the result and resets the counter. One launch.
* the workspaces (segment states and counters, and the unfused passes' per-row
  slope) are allocated once per size *and stream*: calls on one stream run one
  after the other and share them, calls on two streams may overlap on the GPU
  and must not (their segments would be counted and merged together). A call
  captured into a CUDA graph gets its own, from the graph's memory pool: the
  graph replays its kernels with them on whatever stream it is replayed on.
* a thread reads its part ``U`` 16-byte vectors of X and of Y at a time, the
  vectors ``tpr`` apart (coalesced across the team), folds the ``4 U`` pairs
  into a chunk state and merges that into its running state; the vectors left
  at the end are folded one at a time. When ``m`` is not a multiple of 4 the
  rows are not 16-byte aligned and it reads one element at a time instead.

``variant="unfused"`` is the ablation with the unfused program's passes
(``bench.ops.SPECS["ols_hc0"].unfused``) on the same machinery: pass 1 reads
X and Y for ``Σx²`` and ``Σxy`` (compensated sums) and keeps the row's slope
and ``Σx²``; pass 2 reads X and Y again for ``Σ x² (y - theta x)²`` around that
slope and writes ``h / a²``. Only per-row ``(theta, a)`` travel between them.
Pass 2 knows the row's ``a``, so it keeps the range of degree 2 with one power
of two per row: it sums ``s (x (y - theta x))²`` as ``(u s) u``, ``s a`` in
[m', 2 m'), ``m'`` the power of two of ``m`` (``_scale``), and divides by ``s a``
and ``a``; every quotient is the unscaled one, bit for bit.
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
U = 4  # 16-byte vectors of X (and of Y) a thread loads per step, folded into one chunk state
ROW_STEPS = 8  # a row's team grows past a warp only while every thread keeps this many steps
FILL = 1024  # threads of work per SM below which rows are split over CTAs
CTAS_PER_SM = 3  # launch bounds: CTAs of 256 threads resident per SM (<= 80 registers)
WALK_CTAS_PER_SM = 0  # grid of the row walk: CTAs per SM, or 0 for one CTA per group of tasks
MIN_STEPS = 4  # a split segment gives every thread at least this many steps
PREFETCH = 0  # issue a step's loads before the previous step's fold (two sets of buffers)
HINT = 0  # cache hint of the 16-byte loads (``ldv``): 0 none, 1 .cs, 2 .nc, 3 L2 evict-first
FULL_MASK = 0xFFFFFFFF
I32 = torch.int32

# the fused state: Σx², Σxy, the meat h = s(a) Σx²(y - tx)², d = Σx⁴ / a, e = Σx³(y - tx) / a,
# the slope, and the TwoSum errors of a, b, h
A, B, H, D, E, TS, CA, CB, CH = range(9)
LG_CHUNK = 4  # s(a) a is in [2^4, 2^5): the fused h is per unit of a, over a chunk's 16 pairs

KERNEL_META = {
    "op": "ols_hc0",
    "dtype": "float32",
    "algorithm": "one pass, state (Σx², Σxy, s Σx²(y-tx)², Σx⁴/Σx², Σx³(y-tx)/Σx², t) around "
    "the state's slope t = Σxy/Σx², s ~ 16/Σx² an exact power of two (no value of degree above "
    "2 in x, y), plus the TwoSum errors of Σx², Σxy, h; merges move both sides to the merged "
    "slope; result h/(s a²) rounded once (float32 quotients with exact FMA remainders), h moved "
    "to the slope of the compensated sums",
    "structure": "a warp per short row, a 256-thread CTA per long row, 16-byte loads, 4 vectors "
    "of X and Y per step folded in registers, butterfly merges; rows too few to fill the GPU "
    "split over resident CTAs, the last CTA of a row (atomic counter) merges its segment "
    "states: one launch",
}


# --- expression helpers (called at parse time, they build PrimExprs) ---------------------
def _div(num, den):
    """num / den, correctly rounded (``--use_fast_math`` otherwise approximates ``/``)."""
    return T.call_pure_extern("__fdiv_rn", num, den, ty="float32")


def _theta(a, b):
    """The slope of a state: b / a, and 0 when a == 0."""
    return T.if_then_else(a == T.float32(0), T.float32(0), _div(b, a))


def _fma(a, b, c):
    """a·b + c rounded once (``T.fma`` has no CUDA lowering)."""
    return T.call_pure_extern("__fmaf_rn", a, b, c, ty="float32")


def _rcp(a):
    """1 / a to within an ulp (``--use_fast_math``: ``rcp.approx``), a power of two times
    the reciprocal of a's significand, so it scales exactly with a power of two."""
    return T.float32(1) / a


def _recip(a):
    """1 / a to within an ulp, and 0 when a == 0."""
    return T.if_then_else(a == T.float32(0), T.float32(0), _rcp(a))


def _weight(sa, oa, aa):
    """oa / aa, the weight of the merged-in side, ``aa = sa + oa``: exactly 1 when the own
    side is empty (``sa == 0``), exactly 0 when the other is, else to within two ulp."""
    return T.if_then_else(sa == T.float32(0), T.float32(1), oa * _rcp(aa))


def _tree(terms):
    """The sum of ``terms`` as a balanced binary tree."""
    terms = list(terms)
    while len(terms) > 1:
        pairs = [terms[i] + terms[i + 1] for i in range(0, len(terms) - 1, 2)]
        terms = pairs + terms[len(pairs) * 2 :]
    return terms[0]


def _sum_sq(bx, size):
    return _tree(bx[k] * bx[k] for k in range(size))


def _sum_prod(bx, by, size):
    return _tree(bx[k] * by[k] for k in range(size))


def _per_a(bx, ra, k):
    """``q = x² / a`` of pair k as ``(x x) ra``, ``ra = 1 / a``: at most 1, of degree 0."""
    return bx[k] * bx[k] * ra


def _two_sum_err(p, q, s):
    """The exact rounding error ``p + q - s`` of ``s = p + q`` (TwoSum)."""
    qq = s - p
    return (p - (s - qq)) + (q - qq)


def _low(q, num, numl, den, denl):
    """The low part of ``(num + numl) / (den + denl)`` given ``q = num / den`` correctly
    rounded: ``(num - q den + numl - q denl) / den``, ``num - q den`` exact by FMA."""
    return (_fma(-q, den, num) + numl - q * denl) / den


def _scale_field(a, lg):
    """The exponent field of ``s = 2^(lg - E)``, ``E`` the exponent of ``a >= 0``: ``s a`` is
    in ``[2^lg, 2^(lg + 1))``. Clamped to the normal floats, [1, 254] (254 when ``a == 0``)."""
    return T.max(T.min(254 + lg - (T.reinterpret(a, ty="int32") >> 23), 254), 1)


def _pow2(field):
    """The power of two with exponent field ``field`` (``2^(field - 127)``), 0 if ``field <= 0``."""
    return T.reinterpret(T.max(field, 0) << 23, ty="float32")


def _scale(a, lg):
    """``s = 2^(lg - E)`` of ``_scale_field``: multiplying by it is exact while the product
    stays a normal float."""
    return _pow2(_scale_field(a, lg))


# --- the fused pass: fold, combine, finish ------------------------------------------------
@T.inline
def fold_fused(ot, bx, by, size, rv):
    """ot <- the state of the ``size`` pairs (bx, by): the chunk's slope, then its residuals;
    ``h`` scaled by ``s`` of the chunk's ``a`` as ``Σ (u s) u``, ``u = x (y - t x)``, and ``d``,
    ``e`` per unit of ``a`` as ``Σ q x²``, ``Σ q u`` with ``q = x² / a`` (``_per_a``): no product
    has a degree above 2.

    Overwrites ``by`` with ``x (y - t x)``.
    """
    ot[A] = _sum_sq(bx, size)
    ot[B] = _sum_prod(bx, by, size)
    ot[TS] = _theta(ot[A], ot[B])
    for k in T.unroll(size):
        by[k] = bx[k] * (by[k] - ot[TS] * bx[k])
    s: T.float32 = _scale(ot[A], LG_CHUNK)
    ot[H] = _tree(by[k] * s * by[k] for k in range(size))
    ra: T.float32 = _recip(ot[A])
    ot[D] = _tree(_per_a(bx, ra, k) * (bx[k] * bx[k]) for k in range(size))
    ot[E] = _tree(_per_a(bx, ra, k) * by[k] for k in range(size))
    ot[CA] = T.float32(0)
    ot[CB] = T.float32(0)
    ot[CH] = T.float32(0)


@T.inline
def hc0_combine(st, ot):
    """st <- st ⊕ ot: both states moved to the merged slope; the empty state is the identity.

    ``h`` is the meat times ``s`` of the state's ``a`` (``_scale``): each side's is rescaled to
    the merged ``a``'s by the exact power of two ``r = s(a_st + a_ot) / s(a_side)``, and the
    moves are ``δ (s a) (δ d - 2 e)`` (``s a`` < 32). ``d`` and ``e`` are per unit of ``a``,
    and the merged ones are the means of the two sides' weighted by ``a``,
    ``w = a_ot / (a_st + a_ot)`` (``_weight``: exactly 1 or 0 when a side is empty).

    ``a``, ``b`` and ``h`` carry the rounding errors of their sums in ``st[CA:CH+1]``
    (TwoSum: with ``c = p + q`` and ``q' = c - p``, ``(p - (c - q')) + (q - q')``
    is exactly ``p + q - c``); ``h``'s is rescaled with it.
    """
    aa: T.float32 = st[A] + ot[A]
    bb: T.float32 = st[B] + ot[B]
    t: T.float32 = _theta(aa, bb)
    w: T.float32 = _weight(st[A], ot[A], aa)
    fa: T.int32 = _scale_field(aa, LG_CHUNK)
    s: T.float32 = _pow2(fa)
    rs: T.float32 = _pow2(fa - _scale_field(st[A], LG_CHUNK) + 127)
    ro: T.float32 = _pow2(fa - _scale_field(ot[A], LG_CHUNK) + 127)
    da: T.float32 = t - st[TS]
    db: T.float32 = t - ot[TS]
    ho: T.float32 = ot[H] * ro
    ch: T.float32 = (
        ho
        + db * (s * ot[A]) * (db * ot[D] - T.float32(2) * ot[E])
        + da * (s * st[A]) * (da * st[D] - T.float32(2) * st[E])
    )
    es: T.float32 = st[E] - da * st[D]  # each side's e moved to the merged slope
    eo: T.float32 = ot[E] - db * ot[D]
    hs: T.float32 = st[H] * rs
    hh: T.float32 = hs + ch
    st[CA] = st[CA] + (ot[CA] + _two_sum_err(st[A], ot[A], aa))
    st[CB] = st[CB] + (ot[CB] + _two_sum_err(st[B], ot[B], bb))
    st[CH] = st[CH] * rs + (ot[CH] * ro + _two_sum_err(hs, ch, hh))
    st[H] = hh
    st[E] = es + w * (eo - es)
    st[D] = st[D] + w * (ot[D] - st[D])
    st[B] = bb
    st[A] = aa
    st[TS] = t


@T.inline
def store_over_product(Out, row, h, hl, a1, al1, a2, al2):
    """Out[row] = (h + hl) / ((a1 + al1) (a2 + al2)), as if rounded once, in float32; 0 when
    a2 == 0.

    ``ali`` is a few ulp of ``ai``; ``h + hl`` is first renormalised (TwoSum) to ``hs + hr``,
    ``hr`` at most half an ulp of ``hs``. It divides by ``a1``, then by ``a2`` (no product
    ``a1 a2`` to overflow or underflow); each quotient ``q = n / a`` is correctly rounded,
    so its remainder ``n - q a`` is exact by FMA, and
    ``(n + nl) / (a + al) = q + (n - q a + nl - q al) / (a + al)`` (``_low``) carries the
    low parts into the one final rounding. As accurate as a float64 epilogue, without its
    float64 divisions.
    """
    hs: T.float32 = h + hl
    hr: T.float32 = _two_sum_err(h, hl, hs)
    q1: T.float32 = _div(hs, a1)
    l1: T.float32 = _low(q1, hs, hr, a1, al1)
    q2: T.float32 = _div(q1, a2)
    Out[row] = T.if_then_else(a2 == T.float32(0), T.float32(0), q2 + _low(q2, q1, l1, a2, al2))


@T.inline
def finish_fused(Out, RV, row, st, g):
    """Out[row] = h / a² around the slope of the compensated sums; 0 when a == 0.

    The state's ``h`` is the meat around its slope ``t``, times ``s`` of its ``a``; the
    compensated sums' slope ``(b + cb) / (a + ca)`` is ``t + δ``, ``δ = res / a``,
    ``res = (b - t a) + (cb - t ca)`` with ``b - t a`` exact by FMA, and ``h`` moves to it
    by ``s res (δ d - 2 e)`` (``d``, ``e`` per unit of ``a``), which joins ``ch`` as the low
    part of ``h``. The result is ``h / ((s a) a)``, the quotients those of the unscaled
    meat by ``a`` twice.
    """
    s: T.float32 = _scale(st[A], LG_CHUNK)
    res: T.float32 = _fma(-st[TS], st[A], st[B]) + (st[CB] - st[TS] * st[CA])
    dt: T.float32 = res / st[A]
    store_over_product(
        Out,
        row,
        st[H],
        st[CH] + (s * res) * (dt * st[D] - T.float32(2) * st[E]),
        s * st[A],
        s * st[CA],
        st[A],
        st[CA],
    )


# --- the unfused passes: (Σx², Σxy), then Σ x² (y - theta x)² -----------------------------
@T.inline
def fold_normal(ot, bx, by, size, rv):
    """ot <- (Σx², Σxy, 0, 0) of the chunk, balanced trees."""
    ot[0] = _sum_sq(bx, size)
    ot[1] = _sum_prod(bx, by, size)
    ot[2] = T.float32(0)
    ot[3] = T.float32(0)


@T.inline
def combine_normal(st, ot):
    """(a, b, ea, eb) <- sums with their TwoSum errors."""
    aa: T.float32 = st[0] + ot[0]
    bb: T.float32 = st[1] + ot[1]
    st[2] = st[2] + (ot[2] + _two_sum_err(st[0], ot[0], aa))
    st[3] = st[3] + (ot[3] + _two_sum_err(st[1], ot[1], bb))
    st[0] = aa
    st[1] = bb


@T.inline
def finish_normal(Out, RV, row, st, g):
    """RV[row] = (theta, a, error of a): theta of the compensated sums, as if rounded once
    (the correctly rounded ``b / a`` and its exact FMA remainder, as ``store_over_product``)."""
    q: T.float32 = _div(st[1], st[0])
    RV[row, 0] = T.if_then_else(
        st[0] == T.float32(0),
        T.float32(0),
        q + (_fma(-q, st[0], st[1]) + st[3] - q * st[2]) / st[0],
    )
    RV[row, 1] = st[0]
    RV[row, 2] = st[2]


@T.inline
def fold_meat(ot, bx, by, size, rv):
    """ot <- (s Σ (x (y - theta x))², 0) of the chunk around the row's slope ``rv[0]``,
    ``s = rv[1]`` the row's power of two (``read_theta``): each term is ``(u s) u``,
    ``u = x (y - theta x)``, so no product has a degree above 2, and every one is the
    unscaled product times s, exactly."""
    for k in T.unroll(size):
        by[k] = bx[k] * (by[k] - rv[0] * bx[k])
    ot[0] = _tree(by[k] * rv[1] * by[k] for k in range(size))
    ot[1] = T.float32(0)


@T.inline
def combine_meat(st, ot):
    hh: T.float32 = st[0] + ot[0]
    st[1] = st[1] + (ot[1] + _two_sum_err(st[0], ot[0], hh))
    st[0] = hh


@T.inline
def finish_meat(Out, RV, row, st, g):
    """Out[row] = h / a², both compensated; 0 when a == 0: ``s h / ((s a) a)``, ``s`` the
    power of two of ``read_theta``, so both quotients are the unscaled ones, bit for bit."""
    s: T.float32 = _scale(RV[row, 1], g.lgm)
    store_over_product(
        Out, row, st[0], st[1], s * RV[row, 1], s * RV[row, 2], RV[row, 1], RV[row, 2]
    )


@T.inline
def read_nothing(rv, RV, row, g):
    rv[0] = T.float32(0)


@T.inline
def read_theta(rv, RV, row, g):
    """The row's slope, and the power of two ``s`` its meat is scaled by: ``s a`` is in
    ``[2^L, 2^(L+1))``, ``L = floor(log2 m)``, so a term ``s x² (y - theta x)²`` is of
    the size of ``(y - theta x)²`` (``x²`` next to the row's mean ``a / m``)."""
    rv[0] = RV[row, 0]
    rv[1] = _scale(RV[row, 1], g.lgm)


@dataclasses.dataclass(frozen=True)
class Stage:
    """One streaming pass over X and Y: its state, and how it starts, folds, merges, ends a row."""

    name: str
    k: int  # floats in the state
    prologue: object  # prologue(rv, RV, row, g): the row's values from the previous pass
    fold: object  # fold(ot, bx, by, size, rv): the state of ``size`` loaded pairs
    combine: object  # combine(st, ot): st <- st ⊕ ot
    finish: object  # finish(Out, RV, row, st, g): the row's result


FUSED = Stage("fused", 9, read_nothing, fold_fused, hc0_combine, finish_fused)
NORMAL = Stage("normal", 4, read_nothing, fold_normal, combine_normal, finish_normal)
MEAT = Stage("meat", 2, read_theta, fold_meat, combine_meat, finish_meat)
VARIANTS = {"fused": (FUSED,), "unfused": (NORMAL, MEAT)}


# --- merging a team's states, as balanced trees ---------------------------------------------
@T.inline
def butterfly(st, ot, first, levels, g):
    """Merge with the lanes ``first, first/2, ..., first >> (levels-1)`` away (xor)."""
    for i in T.unroll(levels):
        for j in T.unroll(g.k):
            ot[j] = T.gpu_warp_shuffle_xor(FULL_MASK, st[j], first >> i, 32, 32)
        g.combine(st, ot)


@T.inline
def merge_warp_team(st, ot, smem, warp, lane, g):
    """A team of one warp (or less): a butterfly over its lanes."""
    butterfly(st, ot, g.tpr // 2, g.tpr.bit_length() - 1, g)


@T.inline
def merge_cta_team(st, ot, smem, warp, lane, g):
    """A team of ``wpt`` warps: a butterfly per warp, the warp states through shared memory,
    lane l of every warp takes its team's warp ``l % wpt`` and a butterfly over ``wpt``
    lanes merges them. Every thread of the team ends with the team's state. Ends with a
    barrier."""
    butterfly(st, ot, 16, 5, g)
    if lane == 0:
        for j in T.unroll(g.k):
            smem[warp, j] = st[j]
    T.cuda.cta_sync()
    src = (warp // g.wpt) * g.wpt + lane % g.wpt
    for j in T.unroll(g.k):
        st[j] = smem[src, j]
    butterfly(st, ot, g.wpt // 2, g.wpt.bit_length() - 1, g)
    T.cuda.cta_sync()


# --- loads ------------------------------------------------------------------------------------
@T.inline
def load_step_vec(bx, by, X, Y, row, v, g, pol):
    """bx/by[u*w : u*w+w] = the w-float vectors v + u*tpr of the row, u < U (16-byte loads)."""
    for u in T.unroll(g.U):
        bx[T.Ramp(u * g.w, 1, g.w)] = X[row, T.Ramp((v + u * g.tpr) * g.w, 1, g.w)]
        by[T.Ramp(u * g.w, 1, g.w)] = Y[row, T.Ramp((v + u * g.tpr) * g.w, 1, g.w)]


@T.inline
def ldv(buf, i, A, row, e, hint, pol):
    """buf[i : i+4] = A[row, e : e+4], one 16-byte load with cache hint ``hint``:
    1 ``ld.global.cs`` (streaming), 2 ``ld.global.nc.L1::no_allocate``,
    3 ``ld.global.L1::no_allocate.L2::cache_hint`` with an evict-first policy."""
    if hint == 1:
        T.ptx.ld.global_.cs.v4.f32(buf[i], buf[i + 1], buf[i + 2], buf[i + 3], A.ptr_to([row, e]))
    elif hint == 2:
        T.ptx.ld.global_.nc.L1__no_allocate.v4.f32(
            buf[i], buf[i + 1], buf[i + 2], buf[i + 3], A.ptr_to([row, e])
        )
    else:
        T.ptx.ld.global_.L1__no_allocate.L2__cache_hint.v4.f32(
            buf[i], buf[i + 1], buf[i + 2], buf[i + 3], A.ptr_to([row, e]), pol[0]
        )


@T.inline
def load_step_hint(bx, by, X, Y, row, v, g, pol):
    for u in T.unroll(g.U):
        ldv(bx, u * 4, X, row, (v + u * g.tpr) * 4, g.hint, pol)
        ldv(by, u * 4, Y, row, (v + u * g.tpr) * 4, g.hint, pol)


@T.inline
def load_one_vec(bx, by, X, Y, row, v, g):
    bx[T.Ramp(0, 1, g.w)] = X[row, T.Ramp(v * g.w, 1, g.w)]
    by[T.Ramp(0, 1, g.w)] = Y[row, T.Ramp(v * g.w, 1, g.w)]


@T.inline
def load_step_scalar(bx, by, X, Y, row, v, g, pol):
    for u in T.unroll(g.U):
        bx[u] = X[row, v + u * g.tpr]
        by[u] = Y[row, v + u * g.tpr]


@T.inline
def load_one_scalar(bx, by, X, Y, row, v, g):
    bx[0] = X[row, v]
    by[0] = Y[row, v]


# --- the end of a task ------------------------------------------------------------------------
@T.inline
def end_row(Out, RV, P, C, st, ot, smem, flag, task, row, tt, warp, lane, g):
    """S == 1: the team's state is the row's."""
    if tt == 0:
        if task < g.tasks:
            g.finish(Out, RV, row, st, g)


@T.inline
def end_segment(Out, RV, P, C, st, ot, smem, flag, task, row, tt, warp, lane, g):
    """Publish the segment's state; the CTA that ends the row's last segment merges them all."""
    if tt == 0:
        for j in T.unroll(g.k):
            P[task, j] = st[j]
        T.cuda.thread_fence()
        done = T.cuda.atomic_add(T.address_of(C[row]), T.int32(1))
        flag[0] = T.if_then_else(done == g.S - 1, 1, 0)
    T.cuda.cta_sync()
    if flag[0] == 1:
        T.cuda.thread_fence()
        for j in T.unroll(g.k):
            st[j] = T.float32(0)
        p: T.int32 = tt
        while p < g.S:
            for j in T.unroll(g.k):
                ot[j] = P[row * g.S + p, j]
            g.combine(st, ot)
            p += THREADS
        merge_cta_team(st, ot, smem, warp, lane, g)
        if tt == 0:
            g.finish(Out, RV, row, st, g)
            C[row] = 0


@T.inline
def steps_plain(bufx, bufy, nx, ny, st, ot, rv, pol, X, Y, row, v, hi, g):
    """The full steps of a task, each loaded right before it is folded; returns past them."""
    while v[0] + (g.U - 1) * g.tpr < hi:
        g.load_step(bufx, bufy, X, Y, row, v[0], g, pol)
        g.fold(ot, bufx, bufy, g.U * g.w, rv)
        g.combine(st, ot)
        v[0] = v[0] + g.U * g.tpr


@T.inline
def steps_prefetch(bufx, bufy, nx, ny, st, ot, rv, pol, X, Y, row, v, hi, g):
    """The full steps of a task, each step's loads issued before the previous step's fold."""
    if v[0] + (g.U - 1) * g.tpr < hi:
        g.load_step(nx, ny, X, Y, row, v[0], g, pol)
    while v[0] + (g.U - 1) * g.tpr < hi:
        for k in T.unroll(g.U * g.w):
            bufx[k] = nx[k]
            bufy[k] = ny[k]
        if v[0] + (2 * g.U - 1) * g.tpr < hi:
            g.load_step(nx, ny, X, Y, row, v[0] + g.U * g.tpr, g, pol)
        g.fold(ot, bufx, bufy, g.U * g.w, rv)
        g.combine(st, ot)
        v[0] = v[0] + g.U * g.tpr


@T.inline
def stream(X, Y, RV, Out, P, C, bx, tx, g):
    """Every task of the grid-stride walk: fold the team's part of the row, merge, end."""
    warp = tx // 32
    lane = tx % 32
    team = tx // g.tpr
    tt = tx % g.tpr
    st = T.alloc_local((g.k,), "float32")  # this thread's state
    ot = T.alloc_local((g.k,), "float32")  # the state being merged in
    rv = T.alloc_local((2,), "float32")  # the row's values from the previous pass
    bufx = T.alloc_local((g.U * g.w,), "float32")
    bufy = T.alloc_local((g.U * g.w,), "float32")
    nx = T.alloc_local((g.U * g.w,), "float32")  # the next step's loads, when prefetching
    ny = T.alloc_local((g.U * g.w,), "float32")
    vv = T.alloc_local((1,), "int32")
    pol = T.alloc_local((1,), "uint64")  # the L2 evict-first policy of hint 3
    if g.hint == 3:
        T.ptx.createpolicy.fractional.L2__evict_first.b64(pol[0])
    smem = T.alloc_shared((WARPS, g.k), "float32")
    flag = T.alloc_shared((1,), "int32")
    grp: T.int32 = bx
    while grp < g.groups:
        task: T.int32 = grp * g.teams + team
        tk: T.int32 = T.min(task, g.tasks - 1)
        row: T.int32 = tk // g.S
        lo: T.int32 = (tk % g.S) * g.seg
        hi: T.int32 = T.if_then_else(task < g.tasks, T.min(lo + g.seg, g.loads), lo)
        g.prologue(rv, RV, row, g)
        for j in T.unroll(g.k):
            st[j] = T.float32(0)
        vv[0] = lo + tt
        g.steps(bufx, bufy, nx, ny, st, ot, rv, pol, X, Y, row, vv, hi, g)
        v: T.int32 = vv[0]
        while v < hi:
            g.load_one(bufx, bufy, X, Y, row, v, g)
            g.fold(ot, bufx, bufy, g.w, rv)
            g.combine(st, ot)
            v += g.tpr
        g.team_merge(st, ot, smem, warp, lane, g)
        g.end(Out, RV, P, C, st, ot, smem, flag, task, row, tt, warp, lane, g)
        grp += g.blocks


# --- the layout -------------------------------------------------------------------------------
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
    walk: int  # CTAs per SM of the row walk's grid, 0 for one CTA per group of tasks
    minb: int  # CTAs per SM the kernel is compiled for (launch bounds)
    prefetch: bool  # a step's loads issued before the previous step's fold
    hint: int  # cache hint of the 16-byte loads (``ldv``), 0 for none
    stage: Stage

    teams = property(lambda g: THREADS // g.tpr)  # teams per CTA
    wpt = property(lambda g: max(1, g.tpr // 32))  # warps per team
    tasks = property(lambda g: g.n * g.S)
    groups = property(lambda g: -(-g.tasks // g.teams))  # the CTA-sized steps of the walk
    split = property(lambda g: g.S > 1)
    blocks = property(lambda g: g.groups if g.split or not g.walk else min(g.groups, SMS * g.walk))
    ptasks = property(lambda g: g.tasks if g.split else 1)  # rows of the segment workspace
    lgm = property(lambda g: g.m.bit_length() - 1)  # floor(log2 m), of the meat's scale s
    k = property(lambda g: g.stage.k)
    prologue = property(lambda g: g.stage.prologue)
    fold = property(lambda g: g.stage.fold)
    combine = property(lambda g: g.stage.combine)
    finish = property(lambda g: g.stage.finish)
    team_merge = property(lambda g: merge_cta_team if g.tpr > 32 else merge_warp_team)
    end = property(lambda g: end_segment if g.split else end_row)
    load_step = property(
        lambda g: (load_step_hint if g.hint else load_step_vec) if g.w > 1 else load_step_scalar
    )
    load_one = property(lambda g: load_one_vec if g.w > 1 else load_one_scalar)
    steps = property(lambda g: steps_prefetch if g.prefetch else steps_plain)


def plan(n: int, m: int, stage: Stage = FUSED) -> Plan:
    """A warp per row while a thread would otherwise have fewer than ROW_STEPS steps, up to a
    CTA per row; rows too few to give every SM FILL threads are split over the resident CTAs
    (CTAS_PER_SM per SM), in segments of whole CTA strides."""
    w = vector_width("float32") if vectorisable(m, "float32") else 1
    loads = m // w
    tpr = 32
    while tpr < THREADS and loads >= 2 * tpr * U * ROW_STEPS:
        tpr *= 2
    S, seg = 1, loads
    if n * tpr < SMS * FILL:
        steps = -(-loads // (THREADS * U))  # the CTA's steps in a row
        want = min(SMS * CTAS_PER_SM // n, steps // MIN_STEPS)
        if want > 1:
            tpr = THREADS
            seg = -(-(-(-loads // want)) // THREADS) * THREADS  # whole strides of the CTA
            S = -(-loads // seg)
    return Plan(
        n, m, w, loads, tpr, S, seg, U, WALK_CTAS_PER_SM, CTAS_PER_SM, bool(PREFETCH), HINT, stage
    )


def _kernel(g: Plan):
    n, m, k, ptasks, blocks = g.n, g.m, g.k, g.ptasks, g.blocks
    attrs = T.cuda.KernelAttributes(min_blocks_per_sm=g.minb)

    @T.function
    def ols_hc0_pass(
        X: T.Tensor((n, m), "float32", align=16),
        Y: T.Tensor((n, m), "float32", align=16),
        RV: T.Tensor((n, 3), "float32"),
        Out: T.Tensor((n,), "float32"),
        P: T.Tensor((ptasks, k), "float32"),
        C: T.Tensor((n,), "int32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(blocks,), block=(THREADS,)), kernel_attrs=attrs
        )
        stream(X, Y, RV, Out, P, C, T.cuda.block_idx("x"), T.cuda.thread_idx("x"), g)

    return ols_hc0_pass


@cached
def build(cfg: dict, variant: str = "fused"):
    n, m = cfg["n"], cfg["m"]
    plans = [plan(n, m, stage) for stage in VARIANTS[variant]]
    mods = {
        g.stage.name: compile_sm90(_kernel(g), name=f"ols_hc0_{g.stage.name}_{n}x{m}")
        for g in plans
    }
    spaces = {}  # (device, stream) -> the workspaces of the eager calls on that stream

    def alloc(dev):
        """New workspaces, on the current stream (the counters' zeroing runs before the
        stream's next launch)."""
        rows = torch.empty(n, 3, device=dev, dtype=F32)  # (theta, a, error of a), pass 1 -> 2
        work = [  # segment states and the rows' finished-segment counters (0 between calls)
            (
                torch.empty(g.ptasks, g.k, device=dev, dtype=F32),
                torch.zeros(n, device=dev, dtype=I32),
            )
            for g in plans
        ]
        return rows, work

    def workspace(dev):
        """The workspaces of a call on ``dev``'s current stream, where its kernels launch.

        Eager calls on one stream run one after the other and share them. Calls on two
        streams may overlap on the GPU: with shared counters one call's segments were counted
        and merged with the other's, and the reset raced the other's atomics.

        A call captured into a CUDA graph gets new ones, from the graph's memory pool, and
        the graph replays the counters' zeroing with it. A graph's kernels keep the buffers
        they were captured with but run on whatever stream the graph is replayed on: graphs
        captured on one stream, replayed at the same time on two, shared that stream's
        workspaces and mixed their segments and per-row slopes.
        """
        with torch.cuda.device(dev):
            if torch.cuda.is_current_stream_capturing():
                return alloc(dev)
        key = (dev.index, torch._C._cuda_getCurrentRawStream(dev.index))
        ws = spaces.get(key)
        if ws is None:
            ws = spaces.setdefault(key, alloc(dev))
        return ws

    def run(X, Y):
        rows, work = workspace(X.device)
        out = torch.empty(n, device=X.device, dtype=F32)
        for mod, (P, C) in zip(mods.values(), work):
            mod(X, Y, rows, out, P, C)
        return (out,)

    run.modules = mods
    run.plans = plans
    run.threads = THREADS
    run.workspaces = spaces
    return run
