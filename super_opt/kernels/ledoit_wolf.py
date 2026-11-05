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
"""Ledoit-Wolf shrinkage of a batch of covariance matrices, in one pass. float32.

For ``X`` of shape ``(b, n, d)`` (``b`` problems of ``n`` samples of ``d``
features) the outputs are the shrunk covariance ``(1 - rho) S + rho mu I``
``(b, d, d)`` and ``rho`` ``(b,)``, with ``S = Zᵀ Z / n``, ``Z`` the centred
samples, ``mu = tr S / d``, ``delta = ‖S‖²_F / d - mu²``,
``beta = min((Σ_j (Σ_k z_jk²)² / n - ‖S‖²_F) / (d n), delta)`` and
``rho = beta / delta``, or 0 where ``beta = 0`` (``bench.ops._shrink``: no spread
to shrink, as for a constant problem, ``n = 1`` or ``S = mu I`` exactly).

The algorithm is chain_ops' ``ledoit_wolf_fused`` (``examples/chain_ops/advanced.py``):
the state of a feature pair ``(k, l)`` over a set of samples is the 9-tuple

    (count, mean_k, mean_l, cxx, cyy, cxy, t21, t12, v)

with ``dx = x_k - mean_k``, ``dy = x_l - mean_l`` and ``cxx = Σ dx²``,
``cyy = Σ dy²``, ``cxy = Σ dx dy``, ``t21 = Σ dx² dy``, ``t12 = Σ dx dy²``,
``v = Σ dx² dy²``: four products per pair and sample. Two states merge by
``shifted()``: both are re-centred on the merged means and added (``lw_merge``,
the TE merge term by term). Then ``S_kl = cxy / n`` and
``Σ_j (Σ_k z_jk²)² = Σ_kl v_kl``. ``count``, ``mean_k`` and ``cxx`` belong to
feature ``k`` alone, and ``(k, l)`` and ``(l, k)`` carry the same information
(``t21_kl = t12_lk``), so only the pairs ``k <= l`` are computed. ``v`` is only
ever summed over the pairs: the merge adds it linearly, no other moment's
correction reads it, and the epilogue needs ``Σ_kl v_kl``. So the kernel sums a
block's ``dx² dy²`` over its pairs as it goes and keeps one compensated sum of
them a thread (below), not one float32 running sum a pair.

The products as one symmetric rank update, a "W-SYRK": per sample put
``W = (dx_0..3, dx²_0..3, dx_4..7, dx²_4..7, ...)`` (W group ``g`` = the 4
features ``4g .. 4g+3``, then their squares); a pair's four products are the
entries of ``W Wᵀ`` at its two features' rows and columns, and the pairs
``k <= l`` are its upper triangle. That is cut into 8 x 16 blocks of W
(4 x 8 feature pairs), one per consumer thread:
super-block ``h`` is W groups ``2h, 2h+1`` (features ``8h .. 8h+7``) and
thread ``(i, h)``, ``i <= 2h``, owns W group ``i`` against super-block ``h``.
``i < 2h`` are whole blocks above the diagonal; ``i = 2h`` is the
super-block's first half against all of it; the rest of the diagonal, the
upper triangle of W group ``2h+1`` with itself, is added by the producer that
stages that group. A thread computes 96 products a sample (``cxy``, ``t21``,
``t12`` of its 32 pairs), and of the 32 ``dx² dy²`` only their weighted sum
``V += Σ_kl w_kl dx_k² dy_l² = rs (2 hi + m lo)`` (``rs`` = the rows' ``Σ dx²``,
``lo`` / ``hi`` = the two column groups' ``Σ dy²``, ``m`` = 1 in a diagonal block,
whose rows are its ``lo`` columns, else 2; ``w`` = 2 off the diagonal), Kahan-
compensated: 32 running sums over up to 4096 samples lost the small ``dx² dy²``
of the bulk next to the large ones of a far minority (20% of the samples at
+50 sd: 1e-5 of ``Σ v``, a systematic loss). After the loop ``V`` is spread
over the block's pairs' ``dx² dy²`` slots, ``V / Σ w`` to each, so that every
pair's ``v`` keeps the size of its own through the centring and the merges, and
the states, merges and epilogues are those of the 9-tuple. The producer's
triangle does the same: 26 products, and its 10 pairs' ``dx² dx²`` as
``(Σ dx²)²``, compensated. A
problem is ``(dw / 16)²`` blocks: exactly 16 / 64 / 256 threads at
d = 32 / 64 / 128, every product computed once (but for 6 of the 32 pairs of a
diagonal block). Consecutive threads take the blocks of super-blocks ``h`` and
``HW - 1 - h`` together (``2 HW`` blocks), so a warp's fragment loads touch at
most 4 column groups (the consumer loop alone runs 5-7% faster than with the
blocks in ``h``-major order).

How it is laid out (``lw_syrk``, warp-specialised, 384 threads, one CTA an SM):

* 4 producer warps (warpgroup 0, ``setmaxnreg`` 136): a producer owns one
  super-block of one slot and 8 items (a W group of a sample) of every step;
  it ``cp.async``-s them ``DIST`` = 2 steps ahead into a raw ring (16-byte
  loads), reads a whole step's items first, takes them relative to a
  reference point ``K`` per feature, squares them, and stores W to a 3-stage
  shared-memory ring. On the side it adds the per-feature ``Σ dx`` and
  ``Σ dx²`` (two-level: by step, then running) and the triangle of its odd W
  group.
* 8 consumer warps (warpgroups 1-2, ``setmaxnreg`` 184): a thread folds its
  8 x 16 block of every sample of the step from the ring: 6 LDS.128, 96 FFMAs
  and ~16 more instructions for ``V`` a sample (128 FFMAs with ``v`` a pair),
  the sums in registers; 2 samples an iteration with the FMAs column by column
  (ptxas schedules that ~4% better than row by row, and the hot loop stays
  small next to the producers').
* What limits it: the FFMA issue rate. cuBLAS SGEMM reaches 78% of the fp32
  peak on this card and the consumer loop alone ~80%; each producer FP32
  instruction costs the consumers ~1.5 issue cycles and each STS.128 ~4, so
  the producers' fixed ~430 instructions a step cost ~25% at d = 32, ~15% at
  d = 64 and ~5% at d = 128.
* The ring's full / empty handshake is one ``mbarrier`` arrival a warp (after
  ``bar.warp.sync``). Each side tests the next stage's barrier early and
  without blocking (``mbarrier.test_wait`` under the step's last FMAs or
  stores) and blocks at the next step only when that test failed: a
  satisfied blocking wait at the top of every step stalled both warps of a
  scheduler at once and cost ~15% at d = 32.
* A CTA serves ``SC`` slots of ``256 / blocks`` threads: ``PP`` problems x
  ``SR`` sample sub-ranges, the sub-ranges of a problem about one ``K`` and
  added in shared memory at the end; ``P`` CTAs split a problem's samples,
  so that about one CTA runs per SM.
* At the end the moments about ``K`` are re-centred on the range's own mean
  (the binomial shift, once per pair) and written as 9-states: pair moments
  ``(cxy, t21, t12, v)`` to ``Pst``, feature states ``(mean, cxx)`` to
  ``Fst``. ``K = x0 +`` the mean of ``MK`` = 32 samples of ``x - x0``, ``x0``
  the problem's first sample (the states carry their means relative to
  ``x0``): one in each of 32 equal strata of the split at a golden-ratio point
  of the stratum's middle half (plain loads in flight with the first steps'
  ``cp.async``; ``cp.async`` into the still unused ring when a producer takes
  more than 4). ``K`` should sit near the whole split's mean: the shift costs
  ``~(1 + ((mean - K) / sd)²)²`` of the float32 digits of the fourth moments
  and adds ``t21``, ``t12`` to ``v`` weighted by ``mean - K``, so the mean of the
  split's first step failed on an offset block or a spike there (up to 30x
  torch.compile's error), evenly spaced samples would alias with data periodic
  at their stride, and a robust location (the mean of the samples within 2 sd)
  put ``K`` in the bulk of data with a shifted minority, where the bulk's small
  ``dx² dy`` of one sign rounded away in ``t21`` (2e-6 of ``Σ v``). An outlier the
  samples hit moves ``K`` by its offset / 32, which the compensated ``V``
  absorbs; outliers at most of the sampled positions themselves still pull it
  off. ``cxx`` (the diagonal of S) is the two-level sum.

When a CTA holds whole problems (``P = 1``, d <= 64) it finishes them itself:
S and ``v`` in shared memory, the double-float sums (each thread sums entries of
one problem), rho and the shrunk matrix (one launch). When a CTA holds one
problem's split of a few (``P <= 4``) and the ``P`` split states fit its shared
memory, the problem's last CTA (an atomic counter, reset by that CTA) stages
them there (16-byte ``cp.async``, all in flight at once), folds each pair from
there starting from split 0's state, and finishes the problem (one launch).
Otherwise ``lw_merge_splits``, CTA = (problem, range of pairs):
each thread folds the ``P`` 9-states of its pairs with ``lw_merge`` in split
order, writes ``S_kl``, and adds its share of ``tr S``, ``‖S‖²`` and ``Σ v`` as
double-float (hi, lo) sums; a CTA publishes its sums, and the last CTA of a
problem (an atomic counter, reset by that CTA) evaluates ``mu``, ``delta``,
``beta``, ``rho`` in double-float arithmetic and rescales the problem's matrix
(two launches). Only partial states reach HBM. Every kernel keeps subnormals
(``compile_sm90(..., ftz=False)``): for data of scale ~1e-9 the fourth powers,
``‖S‖²`` and the low parts of the double-float sums are subnormal, and flushing
them to zero moved rho by ~10% (to 1 at 1e-10).

``variant="unfused"`` is the ablation with the unfused program's passes on the
same machinery, each reading X again: ``lw_mean`` (``Σ (x - x0)`` per feature),
``lw_syrk`` with ``W = z = (x - x0) - mean(x - x0)`` (covsum; its diagonal
two-level, and its producers' partials added as a compensated sum), then
``lw_syrk`` with ``W = z²`` (the fourth-moment sums, folded to one compensated
(hi, lo) number per CTA), then the epilogue, which adds the splits' covsums as
compensated sums. The unfused passes need one product per pair and sample
where the fused reducer needs three and ``V``. Float32 sums there lost 2e-7 of
``S_00`` and 7e-7 of ``Σ v`` with a ramp feature (a time index): rho 1.6e-6.
A problem's blocks span ``NBC`` CTAs at 256 < d <= 512; each of them stages
every W group, so only the first adds the producers' triangles.
"""

from __future__ import annotations

import torch
from bench.ops import F32

from tvm_tirx.script import tirx as T

from .common.build import SMS, cached, compile_sm90
from .common.reduce import cta_allreduce, warp_allreduce
from .common.workspace import workspace

CT = 256  # consumer threads: 8 warps, two warpgroups
PT = 128  # producer threads: one warpgroup
THREADS = CT + PT
WARPS = THREADS // 32
ITEMS = 8  # W groups (8 W values each) a producer stages per step
NBUF = 3  # stages of the shared-memory ring
STAGE = 8192  # floats a stage (32 KB)
ENTRY_REGS = 65536 // THREADS // 8 * 8  # what one CTA a SM gets at launch: 168
PRODUCER_REGS = 136  # setmaxnreg: the producers keep a step's raw items in registers
CONSUMER_REGS = 184  # 128 sums + 24 fragment values + addresses
assert PT * PRODUCER_REGS + CT * CONSUMER_REGS <= THREADS * ENTRY_REGS  # else .inc never returns
FULL0 = 0  # mbarriers of the ring: full[s] at FULL0 + s, empty[s] at EMPTY0 + s
EMPTY0 = FULL0 + NBUF
PBAR = 15
DIST = 2  # steps the producers' raw loads (cp.async) run ahead
NRAW = DIST + 1  # slots of the raw ring
CP16 = "cp.async.cg.shared.global"
CP4 = "cp.async.ca.shared.global"
EPI_THREADS = 256
EPI_WARPS = EPI_THREADS // 32
EPT_MAX = 4  # pair entries a merge-kernel thread folds at most (unrolled): more b, more CTAs
MK_LOG2 = 5
MK = 1 << MK_LOG2  # samples K is the mean of: one in each of MK strata of the split
GOLDEN16 = 40503  # (sqrt(5) - 1) / 2 in units of 2^-16: the samples' jitter within their strata
SMEM_FUSED = NBUF * STAGE + NRAW * ITEMS * PT * 4  # floats of the fused lw_syrk's dynamic smem
NTRI = 36  # the upper triangle of an 8 x 8 outer product
TRI = [(a, c) for a in range(8) for c in range(a, 8)]
TRI_AT = {ac: q for q, ac in enumerate(TRI)}

KERNEL_META = {
    "op": "ledoit_wolf",
    "dtype": "float32",
    "algorithm": "one pass, 9-state (count, mean_k, mean_l, cxx, cyy, cxy, t21, t12, v) per "
    "feature pair merged by the TE shifted() merge; epilogue tr S, ‖S‖², Σv -> rho, shrink",
    "structure": "W-SYRK, W = (dx, dx²) by 4-feature groups: a consumer thread folds an 8x16 W "
    "block (4x8 pairs) of the upper triangle, 6 LDS.128 + 96 FFMA (cxy, t21, t12) a sample and "
    "the block's Σ w dx² dy² as one Kahan-compensated sum, 2 "
    "samples an iteration column by column, super-blocks (h, HW-1-h) paired over consecutive "
    "threads; 4 producer warps cp.async X into a raw ring and write W to a 3-stage "
    "smem ring (per-warp mbarriers, early non-blocking test_wait, setmaxnreg 136/184) and add "
    "the diagonal triangles; slots = problems x sample sub-ranges, 1 CTA/SM; P = 1: finished in "
    "the CTA; P <= 4: the problem's last CTA stages the split states in smem and finishes it (one "
    "launch); else a merge kernel whose last CTA a problem computes rho and rescales",
}


def _fma(a, b, c):
    """a·b + c rounded once (``T.fma`` has no CUDA lowering)."""
    return T.call_pure_extern("__fmaf_rn", a, b, c, ty="float32")


def _div(a, b):
    """a / b correctly rounded (``--use_fast_math`` makes ``/`` approximate)."""
    return T.call_pure_extern("__fdiv_rn", a, b, ty="float32")


def paired_block(j: int, hw: int) -> tuple[int, int]:
    """(i, h) of the block at slot position ``j``, super-blocks paired: ``(h, hw - 1 - h)`` take
    ``2 hw`` consecutive positions, the ``2h + 1`` blocks of ``h`` first. The consumers' mapping
    (``h_of`` / ``i_of`` in ``_syrk_kernel`` build the same as TIR expressions), ``hw > 1``."""
    pi = j // (2 * hw)
    o = j % (2 * hw)
    return (o, pi) if o < 2 * pi + 1 else (o - (2 * pi + 1), hw - 1 - pi)


def _pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def plan(b: int, n: int, d: int, mode: str = "fused") -> dict:
    """W width, blocks, slots, and how a problem's samples are split.

    ``mode``: ``fused`` (W = (dx, dx²)), ``cov`` (W = z) or ``v`` (W = z²).
    """
    fused = mode == "fused"
    dp = max(8 if fused else 16, _pow2(d))  # features, padded with zeros
    assert dp <= 512, "d <= 512"
    dw = 2 * dp if fused else dp  # W values a sample
    hw = dw // 16  # super-blocks
    nb = hw * hw  # 8 x 16 blocks a problem
    nbc = -(-nb // CT)  # CTAs a problem's blocks need
    sc = min(16, CT // nb) if nb <= CT else 1  # slots a CTA
    nv = dw // 8  # W groups a sample
    ppc = PT // (sc * nv)  # producers per (slot, W group)
    assert ppc >= 1 and ppc * sc * nv == PT
    chunk = ITEMS * ppc  # samples a slot per step
    assert sc * chunk * dw == STAGE
    best = None
    sr = sc
    while sr >= 1:
        pp = sc // sr
        groups = -(-b // pp)
        p = max(1, min(SMS // (groups * nbc), 16384 // d))  # the epilogue stages P x d states
        ls = -(-n // (p * sr))
        ls = -(-ls // chunk) * chunk  # samples a slot
        p = -(-n // (sr * ls))  # no empty split
        grid = groups * p * nbc
        steps = ls // chunk
        key = (-(-grid // SMS) * steps, p, grid)
        if best is None or key < best[0]:
            best = (
                key,
                {"sr": sr, "pp": pp, "groups": groups, "splits": p, "ls": ls, "grid": grid},
            )
        sr //= 2
    out = {
        "mode": mode,
        "dp": dp,
        "dw": dw,
        "hw": hw,
        "nb": nb,
        "nbc": nbc,
        "sc": sc,
        "nv": nv,
        "ppc": ppc,
        "chunk": chunk,
    }
    out.update(best[1])
    out["lp"] = out["sr"] * out["ls"]  # samples a split
    out["steps"] = out["ls"] // chunk
    # a CTA that holds whole problems finishes them itself: S, v in shared memory, then rho
    out["epi_in"] = bool(
        fused and out["splits"] == 1 and nbc == 1 and 3 * out["pp"] * dp * dp <= NBUF * STAGE
    )
    # a few splits of one problem a CTA: the problem's last CTA stages the P split states
    # (block and triangle moments, feature states) in its dynamic shared memory, all of it free
    # by then, and finishes the problem; states that do not fit go to the merge kernel
    out["staged"] = out["splits"] * (32 * nb * 4 + hw * 10 * 4 + 2 * d)
    out["last_in"] = bool(
        fused
        and 1 < out["splits"] <= 4
        and nbc == 1
        and out["pp"] == 1
        and out["staged"] <= SMEM_FUSED
    )
    return out


# --- the TE merge ------------------------------------------------------------------------
@T.inline
def shift_corr(cr, s, mx, my):
    """cr[0:6] = what shifted(s) adds to (cxx, cyy, cxy, t21, t12, v) re-centred on (mx, my)."""
    dx: T.float32 = mx - s[1]
    dy: T.float32 = my - s[2]
    nn: T.float32 = s[0]
    ndx: T.float32 = nn * dx
    ndy: T.float32 = nn * dy
    cr[0] = ndx * dx
    cr[1] = ndy * dy
    cr[2] = ndx * dy
    cr[3] = T.float32(-1) * dy * s[3] - T.float32(2) * dx * s[5] - ndx * dx * dy
    cr[4] = T.float32(-1) * dx * s[4] - T.float32(2) * dy * s[5] - ndx * dy * dy
    cr[5] = (
        T.float32(-2) * dy * s[6]
        - T.float32(2) * dx * s[7]
        + dy * dy * s[3]
        + dx * dx * s[4]
        + T.float32(4) * dx * dy * s[5]
        + ndx * dx * dy * dy
    )


@T.inline
def lw_merge(st, ot, ca, cb):
    """st <- st ⊕ ot, chain_ops' merge: both re-centred on the merged means, then added.

    The big running state ``st`` gets its whole update in one rounding:
    ``st + ((ot + corr(ot)) + corr(st))``.
    """
    cnt: T.float32 = st[0] + ot[0]
    w: T.float32 = ot[0] / T.max(cnt, T.float32(1))
    mx: T.float32 = st[1] + (ot[1] - st[1]) * w
    my: T.float32 = st[2] + (ot[2] - st[2]) * w
    shift_corr(ca, st, mx, my)
    shift_corr(cb, ot, mx, my)
    for q in T.unroll(6):
        st[3 + q] = st[3 + q] + ((ot[3 + q] + cb[q]) + ca[q])
    st[0] = cnt
    st[1] = mx
    st[2] = my


@T.inline
def centre(ob, p11, p21, p12, p22, ek, el, p20, p02, cntf):
    """ob = (cxy, t21, t12, v) about the means, from the moments about K: the binomial shift.

    ``ek = mean_k - K_k``, ``p20 = Σ dx_k²`` about K, ``cntf`` the samples.
    """
    se: T.float32 = cntf * ek * el
    ob[0] = p11 - se
    ob[1] = p21 - el * p20 - T.float32(2) * ek * p11 + T.float32(2) * se * ek
    ob[2] = p12 - ek * p02 - T.float32(2) * el * p11 + T.float32(2) * se * el
    ob[3] = (
        p22
        - T.float32(2) * el * p21
        - T.float32(2) * ek * p12
        + el * el * p20
        + ek * ek * p02
        + T.float32(4) * ek * el * p11
        - T.float32(3) * se * ek * el
    )


# --- double-float (hi, lo) helpers for the epilogue -----------------------------------------
@T.inline
def dw_add(out, o, ahi, alo, bhi, blo):
    """(out[o], out[o + 1]) = (ahi + alo) + (bhi + blo), the leading parts summed error-free."""
    s: T.float32 = ahi + bhi
    bb: T.float32 = s - ahi
    err: T.float32 = (ahi - (s - bb)) + (bhi - bb) + (alo + blo)
    hi: T.float32 = s + err
    out[o + 1] = err - (hi - s)
    out[o] = hi


@T.inline
def dw_div(out, hi, lo, y):
    """(out[0], out[1]) = (hi + lo) / y."""
    q: T.float32 = _div(hi, y)
    rr: T.float32 = _fma(T.float32(0) - q, y, hi) + lo
    out[0] = q
    out[1] = rr / y


@T.inline
def dw_combine(st, ot):
    """Three double-float sums at once: st[2i:2i+2] += ot[2i:2i+2]."""
    for i in T.unroll(3):
        dw_add(st, 2 * i, st[2 * i], st[2 * i + 1], ot[2 * i], ot[2 * i + 1])


@T.inline
def sum2_add(acc, o, x):
    """acc[o] + acc[o + 1] += x, Ogita-Rump-Oishi's Sum2: the running sum acc[o] takes x and the
    rounding error of that add (TwoSum, exact) goes to the low part acc[o + 1], which is not
    renormalised. A long sum's dependent chain stays one add a term (dw_add's is six)."""
    s: T.float32 = acc[o] + x
    bb: T.float32 = s - acc[o]
    acc[o + 1] = acc[o + 1] + ((acc[o] - (s - bb)) + (x - bb))
    acc[o] = s


def _rho_parts(d: int, n: int):
    """``lw_rho(sums, dwt)``: ``dwt[6] = (rho, rho mu)`` from the double-float sums
    ``(‖S‖², tr S, Σ v)``, as ``bench.ops._shrink``: ``mu = tr / d``,
    ``delta = ‖S‖² / d - mu²``, ``beta = min((Σv / n - ‖S‖²) / (d n), delta)`` and
    ``rho = 0`` where ``beta = 0``, else ``beta / delta`` (``dwt[0:6]`` are scratch)."""
    DF, NF, DNF = float(d), float(n), float(d * n)

    @T.inline
    def lw_rho(sums, dwt):
        mu = dwt[0]
        nd = dwt[1]
        dl = dwt[2]
        fn = dwt[3]
        gg = dwt[4]
        bn = dwt[5]
        rm = dwt[6]
        dw_div(mu, sums[2], sums[3], T.float32(DF))
        dw_div(nd, sums[0], sums[1], T.float32(DF))
        m2: T.float32 = mu[0] * mu[0]
        m2e: T.float32 = _fma(mu[0], mu[0], T.float32(0) - m2) + T.float32(2) * mu[0] * mu[1]
        dw_add(dl, 0, nd[0], nd[1], T.float32(0) - m2, T.float32(0) - m2e)
        dw_div(fn, sums[4], sums[5], T.float32(NF))
        dw_add(gg, 0, fn[0], fn[1], T.float32(0) - sums[0], T.float32(0) - sums[1])
        dw_div(bn, gg[0], gg[1], T.float32(DNF))
        q0: T.float32 = _div(bn[0], dl[0])
        rr: T.float32 = _fma(T.float32(0) - q0, dl[0], bn[0]) + bn[1] - q0 * dl[1]
        rq: T.float32 = T.if_then_else(dl[0] == T.float32(0), q0, q0 + rr / dl[0])
        # beta = delta where bn >= delta (rho = 1), else bn (rho = bn / delta); beta = 0 means
        # nothing to shrink (S = mu I: a constant problem, one sample), and rho = 0, not 1
        beyond = (bn[0] > dl[0]) | ((bn[0] == dl[0]) & (bn[1] >= dl[1]))
        rho: T.float32 = T.if_then_else(
            beyond,
            T.if_then_else(dl[0] == T.float32(0), T.float32(0), T.float32(1)),
            T.if_then_else(bn[0] == T.float32(0), T.float32(0), rq),
        )
        if d == 1:
            rho = T.float32(0)
        rm[0] = rho
        rm[1] = rho * mu[0]

    return lw_rho


# --- kernel 1: the W-SYRK ------------------------------------------------------------------
def _syrk_kernel(b: int, n: int, d: int, mode: str, msplits: int = 1):
    """``lw_syrk`` for ``mode`` fused / cov / v; ``msplits``: the mean pass's splits (unfused)."""
    pl = plan(b, n, d, mode)
    fused = mode == "fused"
    DW, NB, NBC, SC, NV = pl["dw"], pl["nb"], pl["nbc"], pl["sc"], pl["nv"]  # noqa: F841
    PPC, CHUNK, SR, PP, P = pl["ppc"], pl["chunk"], pl["sr"], pl["pp"], pl["splits"]
    LS, LP, NSTEP, GRID, DPF = pl["ls"], pl["lp"], pl["steps"], pl["grid"], pl["dp"]
    assert PP <= 16
    VEC = 4 if d % 4 == 0 else 1
    GI = 1 if fused else 2  # 4-feature vectors an item (a W group of 8)
    FI = 4 * GI  # features an item
    NS1 = 4 if fused else 0  # Σ dx per feature of a W group (fused)
    NS2 = 4 if fused else (8 if mode == "cov" else 0)  # Σ dx² / Σ z² per feature, two-level
    NPR = NS1 + NS2 + NTRI  # floats per (problem, W group) after the reduction
    NPUB = 2 * (NS1 + NS2) + NTRI  # floats a producer publishes (two W groups)
    NVH = NV // 2  # super-blocks a sample: a producer owns one, both its W groups
    PPC2 = 2 * PPC  # producers per (slot, super-block)
    RAW0 = NBUF * STAGE  # the raw ring after the W ring
    RAWSLOT = ITEMS * PT * FI  # floats a raw slot
    FULL = (d == pl["dp"]) and (n == P * LP) and (b % PP == 0)  # no sample or feature masked
    ACTIVE = min(CT, SC * NB)
    MW = (CT // 2) * 128  # the consumers' merge area (floats)
    assert MW + PT * NPUB <= NBUF * STAGE
    ROWSTRIDE = 2 * NV * 4  # floats a sample of a slot: W groups, plane 0 then plane 1
    SLOTSTRIDE = CHUNK * ROWSTRIDE
    NF = float(n)
    NROUND = SR.bit_length() - 1  # merge rounds of the sub-ranges
    HW = pl["hw"]
    # K's samples: MK of them, spread over the split, shared by the problem's SR * PPC2 producers
    # of a super-block (the first RK of them, KQ samples each)
    RK = min(SR * PPC2, MK)
    KQ = max(1, MK // (SR * PPC2))
    assert RK * KQ == MK and PT * KQ * 8 <= NBUF * STAGE
    BST_N, TST_N, FST_N, PV_N = sizes(b, d, P, NB, HW, NBC, mode)
    EPI_IN = pl["epi_in"]
    # a few splits of one problem a CTA: the problem's last CTA folds and finishes it
    LAST_IN = pl["last_in"]
    if EPI_IN:  # no partial states leave the CTA
        BST_N, TST_N, FST_N = 1, 1, 1
    OUT_N = b * d * d if EPI_IN or LAST_IN else 1
    RHO_N = b if EPI_IN or LAST_IN else 1
    CNT_N = b if LAST_IN else 1
    SQ = PP * DPF * DPF  # floats of one shared matrix (S hi, S lo, v) for the CTA's problems
    NF_ = float(n)
    NVACC = 2 * PP if mode == "v" else 2  # the v pass's Σ v per problem of the CTA, (hi, lo)
    RLO0 = PP * NV * NPR  # cov: the low parts of the reduced diagonal, past the side sums in red
    RLO_N = PP * NV * NS2 if mode == "cov" else 0

    def cslot_of(tid):
        return tid // NB if NB <= CT else tid * 0

    def qb_of(tid, bc):
        return tid % NB if NB <= CT else bc * CT + tid

    PAIRED = NB <= CT and HW > 1

    def h_of(j):
        """Super-block of the block at slot position j (``paired_block``)."""
        if PAIRED:
            pi = j // (2 * HW)
            return T.if_then_else(j % (2 * HW) < 2 * pi + 1, pi, HW - 1 - pi)
        return T.Cast("int32", T.sqrt(T.Cast("float32", j) + T.float32(0.5)))

    def i_of(j):
        """W group (row) of the block at slot position j."""
        if PAIRED:
            pi = j // (2 * HW)
            o = j % (2 * HW)
            return T.if_then_else(o < 2 * pi + 1, o, o - (2 * pi + 1))
        hh = T.Cast("int32", T.sqrt(T.Cast("float32", j) + T.float32(0.5)))
        return j - hh * hh

    # ---------------------------------------------------------------- producer pieces
    # A producer owns super-block u of one slot: item m is W group g = 2u + m % 2 of sample
    # pc + PPC2 * (m // 2) of the step, so every lane works on odd groups on the same items.
    @T.inline
    def cp_item_vec(buf, X, prob, row, g, off):
        for q in T.unroll(GI):
            vi = T.min(g * GI + q, d // 4 - 1)
            T.ptx[CP16](buf.ptr_to([off + 4 * q]), X.ptr_to([prob, row, vi * 4]), 16, T.uint32(16))

    @T.inline
    def cp_item_scalar(buf, X, prob, row, g, off):
        for j in T.unroll(FI):
            f = T.min(g * FI + j, d - 1)
            T.ptx[CP4](buf.ptr_to([off + j]), X.ptr_to([prob, row, f]), 4, T.uint32(4))

    cp_item = cp_item_vec if VEC == 4 else cp_item_scalar

    @T.inline
    def issue_step(buf, X, prob, s0, pc, u, p, t):
        """cp.async the step's items into raw slot t % NRAW, then commit (even when t is past)."""
        if t < NSTEP:
            for m in T.unroll(ITEMS):
                row = T.min(s0 + t * CHUNK + pc + PPC2 * (m // 2), n - 1)
                cp_item(
                    buf,
                    X,
                    prob,
                    row,
                    2 * u + m % 2,
                    RAW0 + (((t % NRAW) * ITEMS + m) * PT + p) * FI,
                )
        T.ptx.cp.async_.commit_group()

    @T.inline
    def read_item(xr, xo, buf, r, m, p):
        for q in T.unroll(FI // 4):
            xr[T.Ramp(xo + 4 * q, 1, 4)] = buf[
                T.Ramp(RAW0 + ((r * ITEMS + m) * PT + p) * FI + 4 * q, 1, 4)
            ]

    @T.inline
    def tri_add_all(tri, w, vt):
        """cov / v: the 36 products of the upper triangle of w wᵀ."""
        for a in T.unroll(8):
            for c in T.unroll(8):
                if c >= a:
                    tri[_tri_index(a, c)] = _fma(w[a], w[c], tri[_tri_index(a, c)])

    @T.inline
    def tri_add_v(tri, w, vt):
        """fused: the 26 products with a dx in them, and the 10 pairs' dx² dx² as one weighted
        sum, compensated as the consumers' V: Σ_{k<=l} w dx_k² dx_l² = (Σ dx²)², w = 2 off the
        diagonal (pack_tri spreads it over the pairs' slots)."""
        for a in T.unroll(4):
            for c in T.unroll(8):
                if c >= a:
                    tri[_tri_index(a, c)] = _fma(w[a], w[c], tri[_tri_index(a, c)])
        rs: T.float32 = (w[4] + w[5]) + (w[6] + w[7])
        vy: T.float32 = _fma(rs, rs, T.float32(0) - vt[1])
        vs: T.float32 = vt[0] + vy
        vt[1] = (vs - vt[0]) - vy
        vt[0] = vs

    tri_add = tri_add_v if fused else tri_add_all

    @T.inline
    def pack_tri_v(tri, vt, u):
        """The triangle's V spread over its pairs' dx² dx² slots as the consumers' V (pack_v):
        V / Σ w to each pair (kk, ll) of weight w > 0, 2 or 1 on the diagonal, 0 past d."""
        f0: T.int32 = 4 * (2 * u + 1)  # the odd W group's first feature
        wsum: T.float32 = T.float32(0)
        for kk in T.unroll(4):
            for ll in T.unroll(4):
                if ll >= kk:
                    wsum = wsum + T.if_then_else(
                        f0 + ll < d,
                        T.if_then_else(kk == ll, T.float32(1), T.float32(2)),
                        T.float32(0),
                    )
        share: T.float32 = T.if_then_else(
            wsum > T.float32(0), _div(vt[0] - vt[1], T.max(wsum, T.float32(1))), T.float32(0)
        )
        for kk in T.unroll(4):
            for ll in T.unroll(4):
                if ll >= kk:
                    tri[_tri_index(4 + kk, 4 + ll)] = T.if_then_else(
                        f0 + ll < d, share, T.float32(0)
                    )

    @T.inline
    def pack_tri_none(tri, vt, u):
        T.evaluate(0)

    pack_tri = pack_tri_v if fused else pack_tri_none

    def _val(x, ok, fval, o):
        """x where the sample and feature exist, else 0 (no select when nothing is masked)."""
        return x if FULL else T.if_then_else(ok & (fval[o] > 0), x, T.float32(0))

    @T.inline
    def w_fused(w, xr, xo, gi, kv, x0v, fval, ok):
        for j in T.unroll(4):
            dx: T.float32 = _val(xr[xo + j] - kv[gi * 4 + j], ok, fval, gi * 4 + j)
            w[j] = dx
            w[4 + j] = dx * dx

    @T.inline
    def w_cov(w, xr, xo, gi, kv, x0v, fval, ok):
        for j in T.unroll(8):
            w[j] = _val((xr[xo + j] - x0v[gi * 8 + j]) - kv[gi * 8 + j], ok, fval, gi * 8 + j)

    @T.inline
    def w_v(w, xr, xo, gi, kv, x0v, fval, ok):
        for j in T.unroll(8):
            zc: T.float32 = (xr[xo + j] - x0v[gi * 8 + j]) - kv[gi * 8 + j]
            z: T.float32 = _val(zc, ok, fval, gi * 8 + j)
            w[j] = z * z

    make_w = {"fused": w_fused, "cov": w_cov, "v": w_v}[mode]
    HB = ITEMS if fused else ITEMS // 2  # items read before they are processed: 32 raw floats

    @T.inline
    def produce(buf, xr, w, kv, x0v, fval, s1, s2c, tri, vt, m, mm, t, stg, slot, s0, pc, u):
        """Item m (its raw values at xr[mm FI ..]): W into the ring, the side sums, the
        triangle of the odd W groups."""
        sm = pc + PPC2 * (m // 2)
        ok = s0 + t * CHUNK + sm < n
        make_w(w, xr, mm * FI, m % 2, kv, x0v, fval, ok)
        o = stg * STAGE + slot * SLOTSTRIDE + sm * ROWSTRIDE
        buf[T.Ramp(o + (2 * u + m % 2) * 4, 1, 4)] = w[T.Ramp(0, 1, 4)]
        buf[T.Ramp(o + (NV + 2 * u + m % 2) * 4, 1, 4)] = w[T.Ramp(4, 1, 4)]
        side(s1, s2c, w, m % 2)
        if m % 2 == 1:
            tri_add(tri, w, vt)

    @T.inline
    def side_fused(s1, s2c, w, gi):
        for j in T.unroll(4):
            s1[gi * 4 + j] = s1[gi * 4 + j] + w[j]
            s2c[gi * 4 + j] = s2c[gi * 4 + j] + w[4 + j]

    @T.inline
    def side_cov(s1, s2c, w, gi):
        for j in T.unroll(8):
            s2c[gi * 8 + j] = _fma(w[j], w[j], s2c[gi * 8 + j])

    @T.inline
    def side_none(s1, s2c, w, gi):
        T.evaluate(0)

    side = {"fused": side_fused, "cov": side_cov, "v": side_none}[mode]

    @T.inline
    def pre_fused(kv, x0v, X, MS, prob, u):
        """Before the first data lands: x0 (the problem's first sample) of both W groups."""
        for gi in T.unroll(2):
            for j in T.unroll(4):
                x0v[gi * 4 + j] = X[prob, 0, T.min((2 * u + gi) * 4 + j, d - 1)]

    def ksample_row(split, cnt, j):
        """Sample ``j`` of K's MK: in stratum ``j`` of the split's MK equal strata, at the point
        ``1/4 + frac(1/2 + j phi) / 2`` of it (golden-ratio jitter within the middle half,
        stratum 0's midpoint first), so K's samples keep off a split's first samples and do not
        alias with data periodic at the strata's stride (``cnt``: the split's samples, int64)."""
        at = j * 65536 + 16384 + (((32768 + j * GOLDEN16) & 65535) >> 1)  # 2^16 a stratum
        return split * LP + T.Cast("int32", (T.Cast("int64", at) * cnt) >> (16 + MK_LOG2))

    @T.inline
    def kload_vec(xr, X, prob, row, u, o):
        for gi in T.unroll(2):
            xr[T.Ramp(o + 4 * gi, 1, 4)] = X[
                prob, row, T.Ramp(T.min(2 * u + gi, d // 4 - 1) * 4, 1, 4)
            ]

    @T.inline
    def kload_scalar(xr, X, prob, row, u, o):
        for jj in T.unroll(8):
            xr[o + jj] = X[prob, row, T.min((2 * u + jj // 4) * 4 + jj % 4, d - 1)]

    kload = kload_vec if VEC == 4 else kload_scalar

    @T.inline
    def ksamples_reg(xr, buf, X, prob, split, u, kr, p):
        """The producer's share of K's samples (``j = kr, kr + RK, ...``), both W groups, into
        its raw-item registers (KQ <= 4): plain loads, in flight with the first steps' cp.async
        (as a cp.async group of their own they cost ~2% at d = 32)."""
        cnt = T.Cast("int64", T.min(LP, n - split * LP))
        for q in T.unroll(KQ):
            kload(xr, X, prob, ksample_row(split, cnt, kr % RK + RK * q), u, q * 8)

    @T.inline
    def ksamples_smem(xr, buf, X, prob, split, u, kr, p):
        """KQ > 4 samples a producer (few producers a problem): cp.async-ed into the W ring,
        unused until the first step, as a group of its own before the first steps'."""
        cnt = T.Cast("int64", T.min(LP, n - split * LP))
        for q in T.unroll(KQ):
            row = ksample_row(split, cnt, kr % RK + RK * q)
            for gi in T.unroll(2):
                cp_item(buf, X, prob, row, 2 * u + gi, (p * KQ + q) * 8 + 4 * gi)
        T.ptx.cp.async_.commit_group()

    @T.inline
    def ksamples_none(xr, buf, X, prob, split, u, kr, p):
        T.evaluate(0)

    @T.inline
    def kshare_reg(x0v, xr, buf, p):
        """y = x - x0 of the producer's samples, to the W ring (unused until the first step)."""
        for q in T.unroll(KQ):
            for j in T.unroll(8):
                xr[q * 8 + j] = xr[q * 8 + j] - x0v[j]
            for gi in T.unroll(2):
                buf[T.Ramp((p * KQ + q) * 8 + 4 * gi, 1, 4)] = xr[T.Ramp(q * 8 + 4 * gi, 1, 4)]

    @T.inline
    def kshare_smem(x0v, xr, buf, p):
        T.ptx.cp.async_.wait_group(DIST)  # the samples' group: the first steps' may still fly
        for q in T.unroll(KQ):
            for j in T.unroll(8):
                buf[(p * KQ + q) * 8 + j] = buf[(p * KQ + q) * 8 + j] - x0v[j]

    KREG = KQ * 8 <= HB * FI  # K's samples fit the producer's raw-item registers (32 floats)
    kshare = kshare_reg if KREG else kshare_smem

    def kat(r):
        """Where sample r of a feature sits, past its (sub-range 0, producer 0) one: sample r is
        the q = r // RK-th of the problem's producer kr = r % RK of the super-block."""
        kr = r % RK
        return (((kr % PPC2) * SC + kr // PPC2) * NVH * KQ + r // RK) * 8

    @T.inline
    def ref_fused(kv, x0v, xr, buf, kmean, fs, pp, sr, pc, u, p):
        """K = x0 + the mean of the split's MK samples of x - x0, both W groups. Each producer
        puts its samples in the W ring, one producer a feature forms K - x0 (named barriers of
        the producers), and every producer of the problem takes the same K.

        K should sit near the split's mean: the centring adds the odd moments t21, t12 to v
        weighted by mean - K, so their rounding counts in proportion. A robust location (the
        mean of the samples within 2 sd) put K in the bulk of data with a shifted minority (20%
        of the samples at +50 sd): the bulk's dx² dy, small and of one sign there, rounded away
        in t21 (2e-6 of Σ v), and before V was compensated so did its dx² dy² (1e-5). The plain
        mean of the stratified samples sits within ~0.2 sd of the split's mean; a rare outlier
        it hits pulls it off by its offset / MK, which the compensated V and the moments about
        K absorb (a block of 32 at +1000 in a 2048-sample split)."""
        kshare(x0v, xr, buf, p)
        T.ptx.bar.sync(T.uint32(PBAR), T.uint32(PT))
        ka = T.alloc_local((4,), "float32")  # 4-way partial sums
        q1: T.int32 = p
        while q1 < PP * DPF:
            ppq = q1 // DPF
            f = q1 % DPF
            base = ((ppq * SR * NVH + f // 8) * KQ) * 8 + f % 8
            for e in T.unroll(4):
                ka[e] = T.float32(0)
            for r in T.unroll(MK):
                ka[r % 4] = ka[r % 4] + buf[base + kat(r)]
            kmean[q1] = ((ka[0] + ka[1]) + (ka[2] + ka[3])) * T.float32(1.0 / MK)
            q1 += PT
        T.ptx.bar.sync(T.uint32(PBAR), T.uint32(PT))
        for j in T.unroll(8):
            kv[j] = x0v[j] + kmean[pp * DPF + 8 * u + j]
            if (sr == 0) & (pc == 0):
                fs[pp, 2, 8 * u + j] = kv[j] - x0v[j]

    @T.inline
    def pre_mean(kv, x0v, X, MS, prob, u):
        """The mean from the mean pass as x0 and m = Σ (x - x0) / n, both W groups: z is then
        (x - x0) - m, exact but for m's rounding, where x - (x0 + m) would carry the rounding
        of a mean of the data's magnitude (global loads only: before the first data lands)."""
        for j in T.unroll(16):
            f = T.min(16 * u + j, d - 1)
            ms: T.float32 = T.float32(0)
            for q in T.unroll(msplits):
                ms = ms + MS[prob * msplits + q, f]
            x0v[j] = X[prob, 0, f]
            kv[j] = _div(ms, T.float32(NF))

    @T.inline
    def ref_none(kv, x0v, xr, buf, kmean, fs, pp, sr, pc, u, p):
        T.evaluate(0)

    pre = pre_fused if fused else pre_mean
    ksamples = (ksamples_reg if KREG else ksamples_smem) if fused else ksamples_none
    ref = ref_fused if fused else ref_none

    @T.inline
    def publish_fused(buf, s1, s2r, tri, p):
        """Σ dx (2 groups), Σ dx² (2 groups), then the odd group's triangle."""
        for j in T.unroll(8):
            buf[MW + p * NPUB + j] = s1[j]
            buf[MW + p * NPUB + 8 + j] = s2r[j]
        for q in T.unroll(NTRI):
            buf[MW + p * NPUB + 16 + q] = tri[q]

    @T.inline
    def publish_cov(buf, s1, s2r, tri, p):
        for j in T.unroll(16):
            buf[MW + p * NPUB + j] = s2r[j]
        for q in T.unroll(NTRI):
            buf[MW + p * NPUB + 16 + q] = tri[q]

    @T.inline
    def publish_v(buf, s1, s2r, tri, p):
        for q in T.unroll(NTRI):
            buf[MW + p * NPUB + q] = tri[q]

    publish = {"fused": publish_fused, "cov": publish_cov, "v": publish_v}[mode]

    # ---------------------------------------------------------------- consumer pieces
    @T.inline
    def frag(rw, cw, buf, base, i, h):
        rw[T.Ramp(0, 1, 4)] = buf[T.Ramp(base + i * 4, 1, 4)]
        rw[T.Ramp(4, 1, 4)] = buf[T.Ramp(base + (NV + i) * 4, 1, 4)]
        cw[T.Ramp(0, 1, 4)] = buf[T.Ramp(base + (2 * h) * 4, 1, 4)]
        cw[T.Ramp(4, 1, 4)] = buf[T.Ramp(base + (NV + 2 * h) * 4, 1, 4)]
        cw[T.Ramp(8, 1, 4)] = buf[T.Ramp(base + (2 * h + 1) * 4, 1, 4)]
        cw[T.Ramp(12, 1, 4)] = buf[T.Ramp(base + (NV + 2 * h + 1) * 4, 1, 4)]

    @T.inline
    def fma_block_all(acc, rw, cw, vk, md):
        """cov / v: the 128 products of the 8 x 16 W block."""
        # column by column (c outer): ptxas schedules this order ~4% better than row by row
        for c in T.unroll(16):
            for r in T.unroll(8):
                acc[r * 16 + c] = acc[r * 16 + c] + rw[r] * cw[c]

    @T.inline
    def fma_block_v(acc, rw, cw, vk, md):
        """fused: the 96 products of cxy, t21, t12 (rows dx and dx² against columns dy, rows dx
        against columns dy²), and the block's 32 dx² dy² as one weighted sum, compensated:
        V += Σ_kl w_kl dx_k² dy_l² = rs (2 hi + md lo), rs = Σ dx² of the rows, lo / hi =
        Σ dy² of the columns' two W groups, md = 1 in a diagonal block (its rows are the lo
        columns, and w = 2, 1, 0 above, on, below the diagonal) and 2 elsewhere. v is only
        ever summed over the pairs (Σ v in the epilogue; the merge adds it linearly and no
        other moment reads it), so the block's sum is all of it that is needed. As 32 float32
        running sums over up to 4096 samples the small terms rounded away: with 20% of the
        samples 50 sd off, the bulk's dx² dy² against the cluster's (50)⁴, 1e-5 of Σ v."""
        for c in T.unroll(16):
            for r in T.unroll(8):
                if (r < 4) | (c % 8 < 4):
                    acc[r * 16 + c] = acc[r * 16 + c] + rw[r] * cw[c]
        rs: T.float32 = (rw[4] + rw[5]) + (rw[6] + rw[7])
        lo: T.float32 = (cw[4] + cw[5]) + (cw[6] + cw[7])
        hi: T.float32 = (cw[12] + cw[13]) + (cw[14] + cw[15])
        vy: T.float32 = _fma(rs, _fma(md, lo, hi + hi), T.float32(0) - vk[1])
        vt: T.float32 = vk[0] + vy
        vk[1] = (vt - vk[0]) - vy
        vk[0] = vt

    fma_block = fma_block_v if fused else fma_block_all

    def _pair_w(k, ell):
        """A pair's weight in the epilogue's Σ v: 2, or 1 on the diagonal; 0 past d or for the
        duplicates k > l of a diagonal block."""
        return T.if_then_else(
            (k <= ell) & (ell < d),
            T.if_then_else(k == ell, T.float32(1), T.float32(2)),
            T.float32(0),
        )

    @T.inline
    def pack_v_fused(acc, vk, i, h):
        """The block's V spread over its pairs' dx² dy² slots, V / Σ w to each of weight w > 0:
        weighted in the epilogue they add V, and each pair's v keeps the size of its own. V in
        one slot rounded at the whole block's ulp through the centring and the merges, 32
        times a pair's: rho 4.6x the gate's bound where Σ v / n - ‖S‖² cancels (beta ~2e-5 of
        it: half of each split's samples at +1000, and K on them)."""
        k0: T.int32 = 4 * i  # locals: the block mapping's expressions, not 64 copies of them
        l0: T.int32 = 8 * h
        wsum: T.float32 = T.float32(0)
        for lp in T.unroll(32):
            wsum = wsum + _pair_w(k0 + lp // 8, l0 + 4 * ((lp // 4) % 2) + lp % 4)
        share: T.float32 = T.if_then_else(
            wsum > T.float32(0), _div(vk[0] - vk[1], T.max(wsum, T.float32(1))), T.float32(0)
        )
        for lp in T.unroll(32):
            wl: T.float32 = _pair_w(k0 + lp // 8, l0 + 4 * ((lp // 4) % 2) + lp % 4)
            acc[(4 + lp // 8) * 16 + ((lp // 4) % 2) * 8 + 4 + lp % 4] = T.if_then_else(
                wl > T.float32(0), share, T.float32(0)
            )

    @T.inline
    def pack_v_none(acc, vk, i, h):
        T.evaluate(0)

    pack_v = pack_v_fused if fused else pack_v_none

    # ---------------------------------------------------------------- after the loop
    @T.inline
    def reduce_side_plain(buf, red, tid):
        """red[(pp, g, j)] = the producers' side sums of W group g, over sub-ranges and samples."""
        for e in T.unroll(-(-(PP * NV * NPR) // THREADS)):
            q0 = tid + e * THREADS
            if q0 < PP * NV * NPR:
                jj: T.int32 = q0 % NPR
                g: T.int32 = (q0 // NPR) % NV
                gi: T.int32 = g % 2
                src: T.int32 = 2 * NS1 + 2 * NS2 + (jj - NS1 - NS2)
                if jj < NS1 + NS2:
                    src = 2 * NS1 + gi * NS2 + (jj - NS1)
                if jj < NS1:
                    src = gi * NS1 + jj
                base: T.int32 = MW + ((q0 // (NPR * NV)) * SR * NVH + g // 2) * NPUB + src
                sacc: T.float32 = T.float32(0)
                for c in T.unroll(PPC2):
                    for s in T.unroll(SR):
                        sacc = sacc + buf[base + ((c * SC + s) * NVH) * NPUB]
                red[q0] = T.if_then_else((jj < NS1 + NS2) | (gi == 1), sacc, T.float32(0))

    @T.inline
    def reduce_side_dw(buf, red, tid):
        """cov: as reduce_side_plain, the partials added as compensated sums (Sum2), and the
        diagonal's (j < NS2) low parts to red[RLO0 + (pp, g, j)]. The diagonal of S is this
        sum of the producers' two-level sums of z²: added in float32 (up to 32 partials) it
        lost 2e-7, and a ramp feature's rho 4.5 times that."""
        tmp = T.alloc_local((2,), "float32")
        for e in T.unroll(-(-(PP * NV * NPR) // THREADS)):
            q0 = tid + e * THREADS
            if q0 < PP * NV * NPR:
                jj: T.int32 = q0 % NPR
                g: T.int32 = (q0 // NPR) % NV
                gi: T.int32 = g % 2
                src: T.int32 = 2 * NS1 + 2 * NS2 + (jj - NS1 - NS2)
                if jj < NS1 + NS2:
                    src = 2 * NS1 + gi * NS2 + (jj - NS1)
                if jj < NS1:
                    src = gi * NS1 + jj
                base: T.int32 = MW + ((q0 // (NPR * NV)) * SR * NVH + g // 2) * NPUB + src
                tmp[0] = T.float32(0)
                tmp[1] = T.float32(0)
                for c in T.unroll(PPC2):
                    for s in T.unroll(SR):
                        sum2_add(tmp, 0, buf[base + ((c * SC + s) * NVH) * NPUB])
                red[q0] = T.if_then_else((jj < NS1 + NS2) | (gi == 1), tmp[0], T.float32(0))
                if jj < NS2:
                    red[RLO0 + (q0 // NPR) * NS2 + jj] = tmp[1]

    reduce_side = reduce_side_dw if mode == "cov" else reduce_side_plain

    @T.inline
    def feature_stats(red, fs, tid, cntf):
        """fs[pp, 0, f] = e = Σ dx / N, fs[pp, 1, f] = Σ dx² (both about K)."""
        q1: T.int32 = tid
        while q1 < PP * DPF:
            f = q1 % DPF
            ppq = q1 // DPF
            base = (ppq * NV + f // 4) * NPR
            fs[ppq, 0, f] = _div(red[base + f % 4], cntf)
            fs[ppq, 1, f] = red[base + 4 + f % 4]
            q1 += THREADS

    @T.inline
    def no_stats(red, fs, tid, cntf):
        T.evaluate(0)

    stats = feature_stats if fused else no_stats

    @T.inline
    def tail_sync(buf, red, fs, tid, cntf):
        """The common part after the loop: side sums reduced, feature statistics."""
        T.cuda.cta_sync()
        reduce_side(buf, red, tid)
        T.cuda.cta_sync()
        stats(red, fs, tid, cntf)
        T.cuda.cta_sync()

    @T.inline
    def merge_rounds(acc, buf, sr, pp, qb, tmp):
        """Add the sub-ranges of a problem (same K): halving rounds through shared memory."""
        for r in T.unroll(NROUND):
            hr = SR >> (r + 1)
            if (sr >= hr) & (sr < 2 * hr):
                wi = (pp * hr + (sr - hr)) * NB + qb
                for j4 in T.unroll(32):
                    buf[T.Ramp((j4 * (CT // 2) + wi) * 4, 1, 4)] = acc[T.Ramp(j4 * 4, 1, 4)]
            T.cuda.cta_sync()
            if sr < hr:
                wi = (pp * hr + sr) * NB + qb
                for j4 in T.unroll(32):
                    tmp[T.Ramp(0, 1, 4)] = buf[T.Ramp((j4 * (CT // 2) + wi) * 4, 1, 4)]
                    for e in T.unroll(4):
                        acc[j4 * 4 + e] = acc[j4 * 4 + e] + tmp[e]
            T.cuda.cta_sync()

    NPP = 32 // SR  # pairs a consumer thread finishes (fused)

    @T.inline
    def scatter_fused(acc, pv, buf, sr, pp, qb, tmp):
        """pv[4j:4j+4] = the 4 moments of pair sr * NPP + j of the block, summed over the
        problem's sub-ranges (same K): one exchange through shared memory."""
        for lp in T.unroll(32):
            kk = lp // 8
            hf = (lp // 4) % 2
            ll = lp % 4
            tmp[0] = acc[kk * 16 + hf * 8 + ll]
            tmp[1] = acc[(4 + kk) * 16 + hf * 8 + ll]
            tmp[2] = acc[kk * 16 + hf * 8 + 4 + ll]
            tmp[3] = acc[(4 + kk) * 16 + hf * 8 + 4 + ll]
            buf[T.Ramp((((pp * 32 + lp) * SR + sr) * NB + qb) * 4, 1, 4)] = tmp[T.Ramp(0, 1, 4)]
        T.cuda.cta_sync()
        for j in T.unroll(NPP):
            for m in T.unroll(4):
                pv[4 * j + m] = T.float32(0)
            for s2 in T.unroll(SR):
                tmp[T.Ramp(0, 1, 4)] = buf[
                    T.Ramp((((pp * 32 + sr * NPP + j) * SR + s2) * NB + qb) * 4, 1, 4)
                ]
                for m in T.unroll(4):
                    pv[4 * j + m] = pv[4 * j + m] + tmp[m]
        T.cuda.cta_sync()

    @T.inline
    def gather_fused(acc, pv, buf, sr, pp, qb, tmp):
        """SR == 1: the thread's own sums are the block's; regroup them by pair."""
        for lp in T.unroll(32):
            kk = lp // 8
            hf = (lp // 4) % 2
            ll = lp % 4
            pv[4 * lp] = acc[kk * 16 + hf * 8 + ll]
            pv[4 * lp + 1] = acc[(4 + kk) * 16 + hf * 8 + ll]
            pv[4 * lp + 2] = acc[kk * 16 + hf * 8 + 4 + ll]
            pv[4 * lp + 3] = acc[(4 + kk) * 16 + hf * 8 + 4 + ll]

    @T.inline
    def idle_two():
        T.cuda.cta_sync()
        T.cuda.cta_sync()

    @T.inline
    def idle_none():
        T.evaluate(0)

    scatter = scatter_fused if SR > 1 else gather_fused
    idle_scatter = idle_two if SR > 1 else idle_none

    @T.inline
    def idle_rounds():
        for r in T.unroll(NROUND):
            T.cuda.cta_sync()
            T.cuda.cta_sync()

    # writes: fused 9-states / covsum entries / Σv contributions
    @T.inline
    def write_block_fused(acc, ob, Bst, fs, problem, split, pp, q, i, h, cntf, vacc):
        """The block's 32 pairs as centred moments: [split][pair of the block][block], coalesced."""
        for kk in T.unroll(4):
            for hf in T.unroll(2):
                for ll in T.unroll(4):
                    k = 4 * i + kk
                    ell = 8 * h + 4 * hf + ll
                    centre(
                        ob,
                        acc[kk * 16 + hf * 8 + ll],
                        acc[(4 + kk) * 16 + hf * 8 + ll],
                        acc[kk * 16 + hf * 8 + 4 + ll],
                        acc[(4 + kk) * 16 + hf * 8 + 4 + ll],
                        fs[pp, 0, k],
                        fs[pp, 0, ell],
                        fs[pp, 1, k],
                        fs[pp, 1, ell],
                        cntf,
                    )
                    lp = kk * 8 + hf * 4 + ll
                    Bst[T.Ramp((((problem * P + split) * 32 + lp) * NB + q) * 4, 1, 4)] = ob[
                        T.Ramp(0, 1, 4)
                    ]

    @T.inline
    def write_block_cov(acc, ob, Bst, fs, problem, split, pp, q, i, h, cntf, vacc):
        for j in T.unroll(128):
            Bst[((problem * P + split) * 128 + j) * NB + q] = acc[j]

    @T.inline
    def write_block_v(acc, ob, Bst, fs, problem, split, pp, q, i, h, cntf, vacc):
        """The block's weighted Σ v, a compensated sum (ob[0:2], Sum2) into vacc[2 pp: 2 pp + 2]:
        one entry, as Σ z_0⁴ of a ramp feature, can be most of it, and adding the other 127 to
        it in float32 cost Σ v 7e-7 (rho twice that)."""
        ob[0] = T.float32(0)
        ob[1] = T.float32(0)
        k0: T.int32 = 8 * i  # locals: the block mapping's expressions, not 128 copies of them
        l0: T.int32 = 16 * h
        for r in T.unroll(8):
            for c in T.unroll(16):
                k = k0 + r
                ell = l0 + c
                wgt: T.float32 = T.if_then_else(
                    (k < d) & (ell < d) & (k <= ell),
                    T.if_then_else(k == ell, T.float32(1), T.float32(2)),
                    T.float32(0),
                )
                sum2_add(ob, 0, wgt * acc[r * 16 + c])
        for e in T.unroll(PP):
            if e == pp:
                dw_add(vacc, 2 * e, vacc[2 * e], vacc[2 * e + 1], ob[0], ob[1])

    write_block = {"fused": write_block_fused, "cov": write_block_cov, "v": write_block_v}[mode]

    @T.inline
    def write_side_fused(red, fs, Tst, Fst, ob, p, group, split, bc, cntf, vacc):
        """Triangle pairs of the odd W groups, and the feature states (p: the producer index)."""
        if bc == 0:
            q2: T.int32 = p
            while q2 < PP * HW * 10:
                ppq = q2 // (HW * 10)
                hh = (q2 // 10) % HW
                g = 2 * hh + 1
                pr = q2 % 10
                kk = _pair_k(pr)
                ll = _pair_l(pr)
                k = 4 * g + kk
                ell = 4 * g + ll
                problem = group * PP + ppq
                if problem < b:
                    base = (ppq * NV + g) * NPR + 8
                    centre(
                        ob,
                        red[base + _tri_index(kk, ll)],
                        red[base + _tri_index(ll, 4 + kk)],
                        red[base + _tri_index(kk, 4 + ll)],
                        red[base + _tri_index(4 + kk, 4 + ll)],
                        fs[ppq, 0, k],
                        fs[ppq, 0, ell],
                        fs[ppq, 1, k],
                        fs[ppq, 1, ell],
                        cntf,
                    )
                    Tst[T.Ramp((((problem * P + split) * HW + hh) * 10 + pr) * 4, 1, 4)] = ob[
                        T.Ramp(0, 1, 4)
                    ]
                q2 += PT
            q3: T.int32 = p
            while q3 < PP * d:
                ppq = q3 // d
                f = q3 % d
                problem = group * PP + ppq
                if problem < b:
                    base = (ppq * NV + f // 4) * NPR
                    e: T.float32 = fs[ppq, 0, f]
                    Fst[((problem * P + split) * d + f) * 2] = fs[ppq, 2, f] + e
                    Fst[((problem * P + split) * d + f) * 2 + 1] = (
                        red[base + 4 + f % 4] - e * red[base + f % 4]
                    )
                q3 += PT

    @T.inline
    def write_side_cov(red, fs, Tst, Fst, ob, p, group, split, bc, cntf, vacc):
        """Triangle entries of the odd W groups, and the two-level diagonal. Every one of a
        problem's NBC CTAs stages all of its W groups, so the block-chunk 0 CTA writes them."""
        if bc == 0:
            q2: T.int32 = p
            while q2 < PP * HW * NTRI:
                ppq = q2 // (HW * NTRI)
                hh = (q2 // NTRI) % HW
                tq = q2 % NTRI
                problem = group * PP + ppq
                if problem < b:
                    Tst[((problem * P + split) * HW + hh) * NTRI + tq] = red[
                        (ppq * NV + 2 * hh + 1) * NPR + 8 + tq
                    ]
                q2 += PT
            q3: T.int32 = p
            while q3 < PP * d:
                ppq = q3 // d
                f = q3 % d
                problem = group * PP + ppq
                if problem < b:  # the diagonal as double-float (hi, lo)
                    Fst[((problem * P + split) * d + f) * 2] = red[
                        (ppq * NV + f // 8) * NPR + f % 8
                    ]
                    Fst[((problem * P + split) * d + f) * 2 + 1] = red[
                        RLO0 + (ppq * NV + f // 8) * NS2 + f % 8
                    ]
                q3 += PT

    @T.inline
    def write_side_v(red, fs, Tst, Fst, ob, p, group, split, bc, cntf, vacc):
        """Σ v of the triangle entries of the odd W groups (weighted: 2 off the diagonal). Every
        one of a problem's NBC CTAs computes the triangles (its producers stage all W groups),
        and the merge kernel adds the Σ v of all NBC, so only the block-chunk 0 CTA adds them."""
        if bc == 0:
            q2: T.int32 = p
            while q2 < PP * HW * NTRI:
                ppq = q2 // (HW * NTRI)
                g = 2 * ((q2 // NTRI) % HW) + 1
                tq: T.int32 = q2 % NTRI
                tr: T.int32 = _tri_row(tq)
                tc: T.int32 = tr + tq - (tr * 8 - (tr * (tr - 1)) // 2)  # _tri_col(tq)
                wgt: T.float32 = T.if_then_else(
                    8 * g + tc < d,
                    T.if_then_else(tr == tc, T.float32(1), T.float32(2)),
                    T.float32(0),
                )
                for e in T.unroll(PP):
                    if e == ppq:
                        sum2_add(vacc, 2 * e, wgt * red[(ppq * NV + g) * NPR + tq])
                q2 += PT

    write_side = {"fused": write_side_fused, "cov": write_side_cov, "v": write_side_v}[mode]

    @T.inline
    def dw_combine_pp(st, ot):
        """The CTA's PP double-float sums at once: st[2e:2e+2] += ot[2e:2e+2]."""
        for e in T.unroll(PP):
            dw_add(st, 2 * e, st[2 * e], st[2 * e + 1], ot[2 * e], ot[2 * e + 1])

    @T.inline
    def v_total_v(vacc, vot, vsm, Pv, tid, group, split, bc):
        """The v pass: the CTA's Σ v per problem, a double-float (hi, lo) per (problem, split,
        block chunk)."""
        warp = tid // 32
        lane = tid % 32
        cta_allreduce(vacc, vot, 2 * PP, dw_combine_pp, vsm, warp, lane, WARPS)
        if tid < PP:
            for e in T.unroll(PP):
                if (e == tid) & (group * PP + tid < b):
                    Pv[(((group * PP + tid) * P + split) * NBC + bc) * 2] = vacc[2 * e]
                    Pv[(((group * PP + tid) * P + split) * NBC + bc) * 2 + 1] = vacc[2 * e + 1]

    @T.inline
    def v_total_none(vacc, vot, vsm, Pv, tid, group, split, bc):
        T.evaluate(0)

    v_total = v_total_v if mode == "v" else v_total_none

    @T.inline
    def put_pair(buf, ob, pp, k, ell, cxxk, cntf):
        """S_kl (hi, lo) to both triangles, w v_kl to the upper one (0 to the lower)."""
        if (k <= ell) & (ell < d):
            sv: T.float32 = T.if_then_else(k == ell, cxxk, ob[0])
            sh: T.float32 = sv * T.float32(1.0 / n)
            sl: T.float32 = _fma(T.float32(0) - sh, T.float32(NF_), sv) * T.float32(1.0 / n)
            o1 = (pp * DPF + k) * DPF + ell
            o2 = (pp * DPF + ell) * DPF + k
            buf[o1] = sh
            buf[o2] = sh
            buf[SQ + o1] = sl
            buf[SQ + o2] = sl
            buf[2 * SQ + o2] = T.float32(0)
            buf[2 * SQ + o1] = T.if_then_else(k == ell, T.float32(1), T.float32(2)) * ob[3]

    @T.inline
    def slice_out(pv, ob, buf, Bst, fs, problem, split, pp, sr, q, i, h, cntf):
        """The thread's pairs of the block, centred: to Bst (P > 1) or to shared memory (P == 1)."""
        for j in T.unroll(NPP):
            lp = sr * NPP + j
            k = 4 * i + lp // 8
            ell = 8 * h + 4 * ((lp // 4) % 2) + lp % 4
            centre(
                ob,
                pv[4 * j],
                pv[4 * j + 1],
                pv[4 * j + 2],
                pv[4 * j + 3],
                fs[pp, 0, k],
                fs[pp, 0, ell],
                fs[pp, 1, k],
                fs[pp, 1, ell],
                cntf,
            )
            pair_out(ob, buf, Bst, fs, problem, split, pp, lp, q, k, ell, cntf)

    @T.inline
    def pair_shared(ob, buf, Bst, fs, problem, split, pp, lp, q, k, ell, cntf):
        e: T.float32 = fs[pp, 0, k]
        put_pair(buf, ob, pp, k, ell, fs[pp, 1, k] - cntf * e * e, cntf)

    @T.inline
    def pair_global(ob, buf, Bst, fs, problem, split, pp, lp, q, k, ell, cntf):
        Bst[T.Ramp((((problem * P + split) * 32 + lp) * NB + q) * 4, 1, 4)] = ob[T.Ramp(0, 1, 4)]

    pair_out = pair_shared if EPI_IN else pair_global

    @T.inline
    def block_in(acc, ob, buf, fs, pp, i, h, cntf):
        """Consumers: the block's pairs centred, S and v into shared memory."""
        for kk in T.unroll(4):
            for hf in T.unroll(2):
                for ll in T.unroll(4):
                    k = 4 * i + kk
                    ell = 8 * h + 4 * hf + ll
                    centre(
                        ob,
                        acc[kk * 16 + hf * 8 + ll],
                        acc[(4 + kk) * 16 + hf * 8 + ll],
                        acc[kk * 16 + hf * 8 + 4 + ll],
                        acc[(4 + kk) * 16 + hf * 8 + 4 + ll],
                        fs[pp, 0, k],
                        fs[pp, 0, ell],
                        fs[pp, 1, k],
                        fs[pp, 1, ell],
                        cntf,
                    )
                    e: T.float32 = fs[pp, 0, k]
                    put_pair(buf, ob, pp, k, ell, fs[pp, 1, k] - cntf * e * e, cntf)

    @T.inline
    def side_in(red, fs, buf, ob, p, group, cntf):
        """Producers: the triangle pairs of the odd W groups into shared memory."""
        q2: T.int32 = p
        while q2 < PP * HW * 10:
            ppq = q2 // (HW * 10)
            hh = (q2 // 10) % HW
            g = 2 * hh + 1
            pr = q2 % 10
            kk = _pair_k(pr)
            ll = _pair_l(pr)
            k = 4 * g + kk
            ell = 4 * g + ll
            base = (ppq * NV + g) * NPR + 8
            centre(
                ob,
                red[base + _tri_index(kk, ll)],
                red[base + _tri_index(ll, 4 + kk)],
                red[base + _tri_index(kk, 4 + ll)],
                red[base + _tri_index(4 + kk, 4 + ll)],
                fs[ppq, 0, k],
                fs[ppq, 0, ell],
                fs[ppq, 1, k],
                fs[ppq, 1, ell],
                cntf,
            )
            e: T.float32 = fs[ppq, 0, k]
            put_pair(buf, ob, ppq, k, ell, fs[ppq, 1, k] - cntf * e * e, cntf)
            q2 += PT

    lw_rho = _rho_parts(d, n)

    @T.inline
    def rho_mu(rsm, e, sums, dwt):
        """rsm[e] = (rho, rho mu) from the double-float (‖S‖², tr S, Σ v)."""
        lw_rho(sums, dwt)
        rm = dwt[6]
        rsm[e, 0] = rm[0]
        rsm[e, 1] = rm[1]

    PPR = min(PP, 2)  # problems reduced together, one per THREADS / PPR threads
    NRD = -(-PP // PPR)  # rounds over the CTA's problems
    TPP = THREADS // PPR  # threads a problem of the round
    WPP = TPP // 32  # their warps
    EPTI = -(-(d * d) // TPP)  # entries a thread
    assert TPP % 32 == 0 and WPP <= 16

    @T.inline
    def finish_in(buf, sums, osum, rsm, esm, dwt, dsm, Out, Rho, tid, group):
        """All threads: per problem (‖S‖², tr S, Σ v) as double-float sums -> rho; then the
        shrunk matrix. PPR problems a round, THREADS / PPR threads each: the sums in registers,
        warp shuffles, then warp 0 combines each problem's warp sums in one half-warp."""
        T.cuda.cta_sync()
        warp = tid // 32
        lane = tid % 32
        ep = tid // TPP  # the thread's problem within a round
        tq = tid % TPP
        for rd in T.serial(NRD):
            e = rd * PPR + ep
            for q in T.unroll(6):
                dsm[q] = T.float32(0)
            for r in T.unroll(EPTI):
                rr = tq + r * TPP
                k = rr // d
                ell = rr % d
                if (rr < d * d) & (e < PP):
                    o = (e * DPF + k) * DPF + ell
                    sh: T.float32 = buf[o]
                    sl: T.float32 = buf[SQ + o]
                    sq: T.float32 = sh * sh
                    sqe: T.float32 = _fma(sh, sh, T.float32(0) - sq) + T.float32(2) * sh * sl
                    vv: T.float32 = buf[2 * SQ + o]
                    dw_add(dsm, 0, dsm[0], dsm[1], sq, sqe)
                    if k == ell:
                        dw_add(dsm, 2, dsm[2], dsm[3], sh, sl)
                    dw_add(dsm, 4, dsm[4], dsm[5], vv, T.float32(0))
            for lv in T.unroll(5):
                for q in T.unroll(6):
                    osum[q] = T.gpu_warp_shuffle_xor(0xFFFFFFFF, dsm[q], 16 >> lv, 32, 32)
                dw_combine(dsm, osum)
            if lane == 0:
                for q in T.unroll(6):
                    esm[warp, q] = dsm[q]
            T.cuda.cta_sync()
            if warp == 0:
                # problem pq's WPP warp partials on lanes 16 pq .. 16 pq + WPP - 1 (zeros up to
                # 16 a half), a 4-level tree within each half
                pq = lane // 16
                wl = lane % 16
                for q in T.unroll(6):
                    dsm[q] = T.if_then_else(
                        (wl < WPP) & (pq < PPR),
                        esm[T.min(pq * WPP + wl, WARPS - 1), q],
                        T.float32(0),
                    )
                for lv in T.unroll(4):
                    for q in T.unroll(6):
                        osum[q] = T.gpu_warp_shuffle_xor(0xFFFFFFFF, dsm[q], 8 >> lv, 32, 32)
                    dw_combine(dsm, osum)
                if (wl == 0) & (pq < PPR) & (rd * PPR + pq < PP):
                    for q in T.unroll(6):
                        sums[q] = dsm[q]
                    rho_mu(rsm, rd * PPR + pq, sums, dwt)
                    if group * PP + rd * PPR + pq < b:
                        Rho[group * PP + rd * PPR + pq] = rsm[rd * PPR + pq, 0]
            T.cuda.cta_sync()
        for r in T.unroll(-(-(PP * d * d) // THREADS)):
            iw = tid + r * THREADS
            if iw < PP * d * d:
                e = iw // (d * d)
                rr = iw % (d * d)
                k = rr // d
                ell = rr % d
                if group * PP + e < b:
                    s0: T.float32 = buf[(e * DPF + k) * DPF + ell]
                    val: T.float32 = _fma(T.float32(0) - rsm[e, 0], s0, s0)
                    Out[(group * PP + e) * d * d + rr] = T.if_then_else(
                        k == ell, val + rsm[e, 1], val
                    )

    @T.inline
    def finish_none(buf, sums, osum, rsm, esm, dwt, dsm, Out, Rho, tid, group):
        T.evaluate(0)

    NEL = 32 * NB + 10 * HW  # pair entries a problem
    EPT = -(-NEL // THREADS)  # entries a thread of the last CTA folds
    entry_l, stage_l = _fold_parts(n, d, pl, THREADS) if LAST_IN else (None, None)
    if LAST_IN:
        stage_all_l = entry_l.stage_all
        entry_smem_l = entry_l.entry_smem
        # the plan staged the same floats, into the whole dynamic buffer (W ring and raw ring)
        assert entry_l.staged_floats == pl["staged"] <= NBUF * STAGE + NRAW * RAWSLOT == SMEM_FUSED

    @T.inline
    def finish_last(buf, sums, osum, rsm, esm, dwt, Out, Rho, Bst, Tst, Fst, Cnt, lst, tid, group):
        """The problem's states are in Bst / Tst / Fst; the last of its P CTAs (the one that
        counts P, and resets the counter for the next call) folds them, sums ‖S‖², tr S and
        Σ v in double-float, evaluates rho and writes the shrunk matrix."""
        T.cuda.thread_fence()
        T.cuda.cta_sync()
        if tid == 0:
            done = T.cuda.atomic_add(T.address_of(Cnt[group]), T.int32(1))
            lst[0] = T.if_then_else(done == P - 1, 1, 0)
            if done == P - 1:
                Cnt[group] = T.int32(0)  # every other CTA of the problem has counted
        T.cuda.cta_sync()
        if lst[0] == 1:
            T.cuda.thread_fence()
            st = T.alloc_local((9,), "float32")
            ot = T.alloc_local((9,), "float32")
            ca = T.alloc_local((6,), "float32")
            cb = T.alloc_local((6,), "float32")
            kv = T.alloc_local((3,), "int32")
            ek = T.alloc_local((EPT,), "int32")
            el = T.alloc_local((EPT,), "int32")
            eh = T.alloc_local((EPT,), "float32")
            stage_all_l(buf, Bst, Tst, Fst, group, tid)
            for q in T.unroll(6):
                sums[q] = T.float32(0)
            for u in T.unroll(EPT):
                entry_smem_l(kv, st, ot, ca, cb, buf, u * THREADS + tid)
                k = kv[0]
                ell = kv[1]
                ek[u] = T.if_then_else(kv[2] == 1, k, -1)
                el[u] = ell
                sv: T.float32 = T.if_then_else(k == ell, st[3], st[5])
                sh: T.float32 = _div(sv, T.float32(NF))
                eh[u] = sh
                if kv[2] == 1:
                    sl: T.float32 = _fma(T.float32(0) - sh, T.float32(NF), sv) / T.float32(NF)
                    wgt: T.float32 = T.if_then_else(k == ell, T.float32(1), T.float32(2))
                    sq: T.float32 = sh * sh
                    sqe: T.float32 = _fma(sh, sh, T.float32(0) - sq) + T.float32(2) * sh * sl
                    dw_add(sums, 0, sums[0], sums[1], wgt * sq, wgt * sqe)
                    if k == ell:
                        dw_add(sums, 2, sums[2], sums[3], sh, sl)
                    dw_add(sums, 4, sums[4], sums[5], wgt * st[8], T.float32(0))
            cta_allreduce(sums, osum, 6, dw_combine, esm, tid // 32, tid % 32, WARPS)
            if tid == 0:
                rho_mu(rsm, 0, sums, dwt)
                Rho[group] = rsm[0, 0]
            T.cuda.cta_sync()
            for u in T.unroll(EPT):
                if ek[u] >= 0:
                    val: T.float32 = _fma(T.float32(0) - rsm[0, 0], eh[u], eh[u])
                    val = T.if_then_else(ek[u] == el[u], val + rsm[0, 1], val)
                    Out[(group * d + ek[u]) * d + el[u]] = val
                    Out[(group * d + el[u]) * d + ek[u]] = val

    @T.inline
    def finish_last_none(
        buf, sums, osum, rsm, esm, dwt, Out, Rho, Bst, Tst, Fst, Cnt, lst, tid, group
    ):
        T.evaluate(0)

    finish = finish_in if EPI_IN else finish_none
    finish2 = finish_last if LAST_IN else finish_last_none

    @T.function
    def lw_syrk(
        X: T.Tensor((b, n, d), "float32", align=16),
        MS: T.Tensor((b * msplits, d), "float32"),
        Bst: T.Tensor((BST_N,), "float32", align=16),
        Tst: T.Tensor((TST_N,), "float32", align=16),
        Fst: T.Tensor((FST_N,), "float32"),
        Pv: T.Tensor((PV_N,), "float32"),
        Out: T.Tensor((OUT_N,), "float32"),
        Rho: T.Tensor((RHO_N,), "float32"),
        Cnt: T.Tensor((CNT_N,), "int32"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(GRID,), block=(THREADS,)),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1),
        )
        bx = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        buf = T.alloc_tensor(
            ty_args=[T.Tensor((NBUF * STAGE + NRAW * RAWSLOT,), "float32", scope="shared.dyn")]
        )
        kmean = T.alloc_shared((PP * DPF,), "float32")  # K - x0 before rounding, per feature
        red = T.alloc_shared((RLO0 + RLO_N,), "float32")
        fs = T.alloc_shared((PP, 3, DPF), "float32")  # e, Σ dx² about K, K - x0
        vsm = T.alloc_shared((WARPS, NVACC), "float32")
        esm = T.alloc_shared((WARPS, 12), "float32")
        dsm = T.alloc_local((12,), "float32")
        rsm = T.alloc_shared((PP, 2), "float32")
        lst = T.alloc_shared((1,), "int32")
        sums = T.alloc_local((6,), "float32")
        osum = T.alloc_local((12,), "float32")
        dwt = (
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
        )
        mbar = T.alloc_shared((2 * NBUF,), "uint64")
        if tid == 0:
            for q in T.unroll(NBUF):
                # one arrival a warp: lane 0 after the warp's bar.warp.sync
                T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([FULL0 + q]), T.uint32(PT // 32))
                T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([EMPTY0 + q]), T.uint32(CT // 32))
            T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        bc = bx % NBC
        split = (bx // NBC) % P
        group = bx // (NBC * P)
        cntf = T.Cast("float32", T.max(T.min(LP, n - split * LP), 1))
        ob = T.alloc_local((4,), "float32")
        vacc = T.alloc_local((NVACC,), "float32")
        vot = T.alloc_local((NVACC,), "float32")
        for e in T.unroll(NVACC):
            vacc[e] = T.float32(0)
        if tid < PT:
            # ======================== producers: warps 0-3 ========================
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](PRODUCER_REGS)
            p = tid
            u = p % NVH
            rest = p // NVH
            slot = rest % SC
            pc = rest // SC
            pp = slot // SR
            sr = slot % SR
            problem = group * PP + pp
            prob = T.min(problem, b - 1)
            s0 = split * LP + sr * LS  # the slot's first sample
            xr = T.alloc_local((HB * FI,), "float32")  # a batch of a step's raw items
            kv = T.alloc_local((2 * FI,), "float32")
            x0v = T.alloc_local((2 * FI,), "float32")
            fval = T.alloc_local((2 * FI,), "float32")
            w = T.alloc_local((8,), "float32")
            s1 = T.alloc_local((8,), "float32")
            s2c = T.alloc_local((16,), "float32")
            s2r = T.alloc_local((16,), "float32")
            tri = T.alloc_local((NTRI,), "float32")
            vt = T.alloc_local((2,), "float32")  # fused: the odd group's Σ w dx² dx², Kahan
            for j in T.unroll(2 * FI):
                fval[j] = T.if_then_else(
                    ((2 * u + j // FI) * FI + j % FI < d) & (problem < b),
                    T.float32(1),
                    T.float32(0),
                )
            for j in T.unroll(8):
                s1[j] = T.float32(0)
            for j in T.unroll(16):
                s2c[j] = T.float32(0)
                s2r[j] = T.float32(0)
            for q in T.unroll(NTRI):
                tri[q] = T.float32(0)
            vt[0] = T.float32(0)
            vt[1] = T.float32(0)
            ksamples(xr, buf, X, prob, split, u, sr * PPC2 + pc, p)  # fused: K's samples
            for t0 in T.unroll(DIST):
                issue_step(buf, X, prob, s0, pc, u, p, t0)
            pre(kv, x0v, X, MS, prob, u)  # global loads in flight with the first steps
            ref(kv, x0v, xr, buf, kmean, fs, pp, sr, pc, u, p)
            free = T.local_scalar("uint32")  # the next stage was seen empty by the early test
            free = T.uint32(0)
            for t in T.serial(NSTEP):
                stg = t % NBUF
                issue_step(buf, X, prob, s0, pc, u, p, t + DIST)
                T.ptx.cp.async_.wait_group(DIST)
                if free == T.uint32(0):
                    T.cuda.mbarrier_wait(mbar.ptr_to([EMPTY0 + stg]), (t // NBUF) % 2 ^ 1)
                for mb in T.unroll(ITEMS // HB):
                    for mm in T.unroll(HB):  # a batch's loads first
                        read_item(xr, mm * FI, buf, t % NRAW, mb * HB + mm, p)
                    for mm in T.unroll(HB):
                        produce(
                            buf,
                            xr,
                            w,
                            kv,
                            x0v,
                            fval,
                            s1,
                            s2c,
                            tri,
                            vt,
                            mb * HB + mm,
                            mm,
                            t,
                            stg,
                            slot,
                            s0,
                            pc,
                            u,
                        )

                T.ptx.mbarrier.test_wait.parity.shared__cta.b64(
                    free, mbar.ptr_to([EMPTY0 + (t + 1) % NBUF]), T.uint32((t + 1) // NBUF % 2 ^ 1)
                )
                for j in T.unroll(16):
                    s2r[j] = s2r[j] + s2c[j]
                    s2c[j] = T.float32(0)
                T.cuda.warp_sync()
                T.ptx.mbarrier.arrive.shared.b64(
                    mbar.ptr_to([FULL0 + stg]), T.uint32(1), pred=tid % 32 == 0
                )
            T.ptx.cp.async_.wait_group(0)
            T.cuda.cta_sync()
            pack_tri(tri, vt, u)
            publish(buf, s1, s2r, tri, p)
            tail_sync(buf, red, fs, tid, cntf)
            if fused:
                idle_scatter()
            else:
                idle_rounds()
            if EPI_IN:
                side_in(red, fs, buf, ob, p, group, cntf)
            else:
                write_side(red, fs, Tst, Fst, ob, p, group, split, bc, cntf, vacc)
            finish(buf, sums, osum, rsm, esm, dwt, dsm, Out, Rho, tid, group)
            finish2(buf, sums, osum, rsm, esm, dwt, Out, Rho, Bst, Tst, Fst, Cnt, lst, tid, group)
        else:
            # ======================== consumers: warps 4-11 ========================
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](CONSUMER_REGS)
            ctid = tid - PT
            acc = T.alloc_local((128,), "float32")
            rw = T.alloc_local((8,), "float32")
            cw = T.alloc_local((16,), "float32")
            tmp = T.alloc_local((4,), "float32")
            vk = T.alloc_local((2,), "float32")  # fused: the block's Σ w dx² dy², Kahan (sum, c)
            for q in T.unroll(128):
                acc[q] = T.float32(0)
            vk[0] = T.float32(0)
            vk[1] = T.float32(0)
            cslot = cslot_of(ctid)
            qj = qb_of(ctid, bc)  # the thread's position in its slot
            h = h_of(qj)
            i = i_of(qj)
            qb = h * h + i  # the block's canonical index: exchanges, stored states, folds
            active = (ctid < ACTIVE) & (qj < NB)
            md: T.float32 = T.if_then_else(i == 2 * h, T.float32(1), T.float32(2))
            ready = T.local_scalar("uint32")  # the stage was seen full by the early test
            ready = T.uint32(0)
            for t in T.serial(NSTEP):
                stg = t % NBUF
                if ready == T.uint32(0):
                    T.cuda.mbarrier_wait(mbar.ptr_to([FULL0 + stg]), (t // NBUF) % 2)
                ready = T.uint32(0)
                if active:
                    base0 = stg * STAGE + cslot * SLOTSTRIDE
                    # 2 samples an iteration: a hot loop small enough to stay in the
                    # instruction cache next to the producers' loop
                    for it in T.serial(CHUNK // 2 - 1):
                        for u in T.unroll(2):
                            frag(rw, cw, buf, base0 + (it * 2 + u) * ROWSTRIDE, i, h)
                            fma_block(acc, rw, cw, vk, md)
                    frag(rw, cw, buf, base0 + (CHUNK - 2) * ROWSTRIDE, i, h)
                    fma_block(acc, rw, cw, vk, md)
                    frag(rw, cw, buf, base0 + (CHUNK - 1) * ROWSTRIDE, i, h)
                    # a satisfied blocking wait at the step's start costs ~15% at d = 32; test the
                    # next stage now, under this sample's FMAs (acquire: its loads come after)
                    T.ptx.mbarrier.test_wait.parity.shared__cta.b64(
                        ready, mbar.ptr_to([FULL0 + (t + 1) % NBUF]), T.uint32((t + 1) // NBUF % 2)
                    )
                    fma_block(acc, rw, cw, vk, md)
                T.cuda.warp_sync()
                T.ptx.mbarrier.arrive.shared.b64(
                    mbar.ptr_to([EMPTY0 + stg]), T.uint32(1), pred=tid % 32 == 0
                )
            T.cuda.cta_sync()
            T.cuda.cta_sync()
            reduce_side(buf, red, tid)
            T.cuda.cta_sync()
            stats(red, fs, tid, cntf)
            T.cuda.cta_sync()
            sr = cslot % SR
            pp = cslot // SR
            problem = group * PP + pp
            if fused:
                pack_v(acc, vk, i, h)
                pv = T.alloc_local((4 * NPP,), "float32")
                scatter(acc, pv, buf, sr, pp, qb, tmp)
                if active & (problem < b):
                    slice_out(pv, ob, buf, Bst, fs, problem, split, pp, sr, qb, i, h, cntf)
            else:
                merge_rounds(acc, buf, sr, pp, qb, tmp)
                if active & (sr == 0) & (problem < b):
                    write_block(acc, ob, Bst, fs, problem, split, pp, qb, i, h, cntf, vacc)
            finish(buf, sums, osum, rsm, esm, dwt, dsm, Out, Rho, tid, group)
            finish2(buf, sums, osum, rsm, esm, dwt, Out, Rho, Bst, Tst, Fst, Cnt, lst, tid, group)
        v_total(vacc, vot, vsm, Pv, tid, group, split, bc)

    return lw_syrk, pl


def _pair_k(pr):
    """k of pair pr of the 10 pairs k <= l of 4 features: (0,0) (0,1) (0,2) (0,3) (1,1) ..."""
    return T.if_then_else(pr < 4, 0, T.if_then_else(pr < 7, 1, T.if_then_else(pr < 9, 2, 3)))


def _pair_l(pr):
    off = T.if_then_else(pr < 4, 0, T.if_then_else(pr < 7, 4, T.if_then_else(pr < 9, 7, 9)))
    return _pair_k(pr) + pr - off


def merge_layout(b: int, d: int, pl: dict, mode: str) -> tuple[int, int, int]:
    """``(NE, EPT, NCE)`` of ``lw_merge_splits``: a problem's ``NE`` stored pair entries, folded
    ``EPT`` a thread (unrolled) by ``NCE`` CTAs of ``EPI_THREADS`` threads. ``EPT`` fills about
    one wave of CTAs, but at most ``EPT_MAX`` (the kernel's code would grow with ``b``: 160 s to
    compile at b = 4096, d = 128) and never past the ``ceil(NE / EPI_THREADS)`` entries a thread
    of a problem's single CTA folds (further iterations would only fold padding)."""
    NB, HW = pl["nb"], pl["hw"]
    ne = 32 * NB + 10 * HW if mode == "fused" else 128 * NB + NTRI * HW + d
    wave = -(-b * ne // (EPI_THREADS * 2 * SMS))
    ept = max(1, min(wave, -(-ne // EPI_THREADS), EPT_MAX))
    return ne, ept, -(-ne // (EPI_THREADS * ept))


def sizes(b: int, d: int, P: int, NB: int, HW: int, NBC: int, mode: str):
    """Element counts of lw_syrk's outputs (1 for the ones a mode does not write)."""
    if mode == "fused":
        return b * P * 32 * NB * 4, b * P * HW * 10 * 4, b * P * d * 2, 1
    if mode == "cov":  # the diagonal as double-float (hi, lo)
        return b * P * 128 * NB, b * P * HW * NTRI, b * P * d * 2, 1
    return 1, 1, 1, b * P * NBC * 2  # Σ v as double-float (hi, lo)


def _tri_index(a, c):
    """Index of entry (a, c), a <= c < 8 (python ints or exprs with a <= c), of TRI."""
    return a * 8 - (a * (a - 1)) // 2 + (c - a)


def _tri_row(tq):
    """Row of triangle entry tq (an expr)."""
    r = 0
    for a in range(1, 8):
        r = r + T.if_then_else(tq >= TRI_AT[(a, a)], 1, 0)
    return r


def _tri_col(tq):
    """Column of triangle entry tq (an expr)."""
    r = _tri_row(tq)
    return r + tq - (r * 8 - (r * (r - 1)) // 2)


# --- the mean pass of the unfused variant ----------------------------------------------------
def _mean_kernel(b: int, n: int, d: int):
    """MS[(problem, split), f] = Σ (x - x0) over the split's samples (x0: the first sample)."""
    VEC = 4 if d % 4 == 0 else 1
    NVX = -(-d // VEC)  # vectors a sample row
    lanes = max(1, 256 // NVX)
    msplits = max(1, min(-(-2 * SMS // b), -(-n // (4 * lanes))))
    per = -(-n // msplits)
    THR = 256
    U = 4
    NVT = -(-NVX // THR)  # vectors a thread when a row has more than 256
    NVM = min(NVX, THR)

    @T.inline
    def ld_vec(xb, o, X, bi, row, vi):
        xb[T.Ramp(o, 1, 4)] = X[bi, row, T.Ramp(vi * 4, 1, 4)]

    @T.inline
    def ld_scalar(xb, o, X, bi, row, vi):
        xb[o] = X[bi, row, vi]

    ld = ld_vec if VEC == 4 else ld_scalar

    @T.function
    def lw_mean(
        X: T.Tensor((b, n, d), "float32", align=16), MS: T.Tensor((b * msplits, d), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(b * msplits,), block=(THR,)))
        bx = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        bi = bx // msplits
        sp = bx % msplits
        part = T.alloc_shared((THR * VEC,), "float32")
        xb = T.alloc_local((U * VEC,), "float32")
        x0 = T.alloc_local((VEC,), "float32")
        sc = T.alloc_local((VEC,), "float32")
        sr = T.alloc_local((VEC,), "float32")
        for vt in T.serial(NVT):
            vi = (tid % NVM) + vt * THR
            lane = tid // NVM
            vic = T.min(vi, NVX - 1)
            for e in T.unroll(VEC):
                sr[e] = T.float32(0)
            ld(x0, 0, X, bi, 0, vic)
            lo = sp * per
            hi = T.min(lo + per, n)
            row: T.int32 = lo + lane
            while row < hi:
                for u in T.unroll(U):
                    ld(xb, u * VEC, X, bi, T.min(row + u * lanes, n - 1), vic)
                for e in T.unroll(VEC):
                    sc[e] = T.float32(0)
                for u in T.unroll(U):
                    for e in T.unroll(VEC):
                        if row + u * lanes < hi:
                            sc[e] = sc[e] + (xb[u * VEC + e] - x0[e])
                for e in T.unroll(VEC):
                    sr[e] = sr[e] + sc[e]
                row += U * lanes
            for e in T.unroll(VEC):
                part[tid * VEC + e] = sr[e]
            T.cuda.cta_sync()
            if (tid < NVM) & (vi < NVX):
                for e in T.unroll(VEC):
                    s: T.float32 = T.float32(0)
                    for l in T.serial(lanes):  # noqa: E741
                        s = s + part[(l * NVM + tid) * VEC + e]
                    if vi * VEC + e < d:
                        MS[bx, vi * VEC + e] = s
            T.cuda.cta_sync()

    return lw_mean, msplits


# --- the fold of a pair's split states (lw_merge_splits; lw_syrk's last CTA) ---------------
def _fold_parts(n: int, d: int, pl: dict, threads: int):
    """``entry(kv, mv, st, ot, ca, cb, fsm, Bst, Tst, Fst, bi, idx)``: pair entry ``idx`` of
    problem ``bi``, its ``P`` 9-states folded with ``lw_merge`` in split order (kv = (k, ell,
    valid), st = the merged state); ``stage(fsm, Fst, bi, tid)``: the problem's feature states
    into shared memory (``threads`` threads, then a CTA barrier)."""
    P, LP, NB, HW = pl["splits"], pl["lp"], pl["nb"], pl["hw"]
    NE = 32 * NB + 10 * HW  # entries a problem
    FSM = P * d * 2
    SPT = -(-FSM // threads)
    BATCH = min(16, P)
    NBT = -(-P // BATCH)

    @T.inline
    def entry_fused(kv, mv, st, ot, ca, cb, fsm, Bst, Tst, Fst, bi, idx):
        """kv = (k, ell, valid); st = the pair's merged 9-state. The padded entries past the
        problem's (idx >= NE: kv[2] = 0) still load: their triangle index is clamped into it."""
        isb: T.bool = idx < 32 * NB
        lp: T.int32 = idx // NB
        q: T.int32 = T.if_then_else(isb, idx % NB, 0)
        h: T.int32 = T.Cast("int32", T.sqrt(T.Cast("float32", q) + T.float32(0.5)))
        i: T.int32 = q - h * h
        hh: T.int32 = T.min(T.max(idx - 32 * NB, 0) // 10, HW - 1)
        pr: T.int32 = T.max(idx - 32 * NB, 0) % 10
        k: T.int32 = 8 * hh + 4 + _pair_k(pr)
        ell: T.int32 = 8 * hh + 4 + _pair_l(pr)
        if isb:
            k = 4 * i + lp // 8
            ell = 8 * h + 4 * ((lp // 4) % 2) + lp % 4
        kv[0] = k
        kv[1] = ell
        kv[2] = T.if_then_else((idx < NE) & (k <= ell) & (ell < d), 1, 0)
        kc: T.int32 = T.min(k, d - 1) * 2
        lc: T.int32 = T.min(ell, d - 1) * 2
        for j in T.unroll(9):
            st[j] = T.float32(0)
        fold(st, ot, ca, cb, mv, fsm, Bst, Tst, bi, isb, lp, q, hh, pr, kc, lc)

    @T.inline
    def fold_one(st, ot, ca, cb, mv, fsm, Bst, Tst, bi, isb, lp, q, hh, pr, kc, lc):
        """P <= 16: every split's moments loaded at once, the split index a constant."""
        if isb:
            for p in T.unroll(P):
                mv[T.Ramp(p * 4, 1, 4)] = Bst[T.Ramp((((bi * P + p) * 32 + lp) * NB + q) * 4, 1, 4)]
        else:
            for p in T.unroll(P):
                mv[T.Ramp(p * 4, 1, 4)] = Tst[
                    T.Ramp((((bi * P + p) * HW + hh) * 10 + pr) * 4, 1, 4)
                ]
        for p in T.unroll(P):
            ot[0] = T.Cast("float32", T.min(LP, n - p * LP))
            ot[1] = fsm[p * d * 2 + kc]
            ot[2] = fsm[p * d * 2 + lc]
            ot[3] = fsm[p * d * 2 + kc + 1]
            ot[4] = fsm[p * d * 2 + lc + 1]
            for j in T.unroll(4):
                ot[5 + j] = mv[p * 4 + j]
            lw_merge(st, ot, ca, cb)

    @T.inline
    def fold_batched(st, ot, ca, cb, mv, fsm, Bst, Tst, bi, isb, lp, q, hh, pr, kc, lc):
        for bt in T.serial(NBT):
            for u in T.unroll(BATCH):
                p = T.min(bt * BATCH + u, P - 1)
                if isb:
                    mv[T.Ramp(u * 4, 1, 4)] = Bst[
                        T.Ramp((((bi * P + p) * 32 + lp) * NB + q) * 4, 1, 4)
                    ]
                else:
                    mv[T.Ramp(u * 4, 1, 4)] = Tst[
                        T.Ramp((((bi * P + p) * HW + hh) * 10 + pr) * 4, 1, 4)
                    ]
            for u in T.unroll(BATCH):
                p = bt * BATCH + u
                if p < P:
                    ot[0] = T.Cast("float32", T.min(LP, n - p * LP))
                    ot[1] = fsm[p * d * 2 + kc]
                    ot[2] = fsm[p * d * 2 + lc]
                    ot[3] = fsm[p * d * 2 + kc + 1]
                    ot[4] = fsm[p * d * 2 + lc + 1]
                    for j in T.unroll(4):
                        ot[5 + j] = mv[u * 4 + j]
                    lw_merge(st, ot, ca, cb)

    fold = fold_one if P <= 16 else fold_batched

    @T.inline
    def stage_fused(fsm, Fst, bi, tid):
        for u in T.unroll(SPT):
            if tid + u * threads < FSM:
                fsm[tid + u * threads] = Fst[bi * FSM + tid + u * threads]
        T.cuda.cta_sync()

    # the whole problem staged (the last CTA of a problem): moments of the blocks, of the
    # triangles, then the feature states
    BSL = P * 32 * NB * 4
    TSL = P * HW * 10 * 4
    FO = BSL + TSL

    @T.inline
    def stage_all(sm, Bst, Tst, Fst, bi, tid):
        """The problem's P split states into shared memory (moments by 16-byte cp.async, all in
        flight together), then a CTA barrier."""
        for r in T.unroll(-(-(BSL // 4) // threads)):
            v = tid + r * threads
            if v < BSL // 4:
                T.ptx[CP16](sm.ptr_to([v * 4]), Bst.ptr_to([bi * BSL + v * 4]), 16, T.uint32(16))
        for r in T.unroll(-(-(TSL // 4) // threads)):
            v = tid + r * threads
            if v < TSL // 4:
                T.ptx[CP16](
                    sm.ptr_to([BSL + v * 4]), Tst.ptr_to([bi * TSL + v * 4]), 16, T.uint32(16)
                )
        T.ptx.cp.async_.commit_group()
        for r in T.unroll(-(-FSM // threads)):
            v = tid + r * threads
            if v < FSM:
                sm[FO + v] = Fst[bi * FSM + v]
        T.ptx.cp.async_.wait_group(0)
        T.cuda.cta_sync()

    @T.inline
    def entry_smem(kv, st, ot, ca, cb, sm, idx):
        """Pair entry idx from the staged problem: st = split 0's state, then splits 1 .. P-1
        merged in order (kv = (k, ell, valid))."""
        isb: T.bool = idx < 32 * NB
        lp: T.int32 = idx // NB
        q: T.int32 = T.if_then_else(isb, idx % NB, 0)
        h: T.int32 = T.Cast("int32", T.sqrt(T.Cast("float32", q) + T.float32(0.5)))
        i: T.int32 = q - h * h
        hh: T.int32 = T.min(T.max(idx - 32 * NB, 0) // 10, HW - 1)
        pr: T.int32 = T.max(idx - 32 * NB, 0) % 10
        k: T.int32 = 8 * hh + 4 + _pair_k(pr)
        ell: T.int32 = 8 * hh + 4 + _pair_l(pr)
        if isb:
            k = 4 * i + lp // 8
            ell = 8 * h + 4 * ((lp // 4) % 2) + lp % 4
        kv[0] = k
        kv[1] = ell
        kv[2] = T.if_then_else((idx < NE) & (k <= ell) & (ell < d), 1, 0)
        kc: T.int32 = T.min(k, d - 1) * 2
        lc: T.int32 = T.min(ell, d - 1) * 2
        mb: T.int32 = T.if_then_else(isb, (T.min(lp, 31) * NB + q) * 4, BSL + (hh * 10 + pr) * 4)
        ms: T.int32 = T.if_then_else(isb, 32 * NB * 4, HW * 10 * 4)
        st[0] = T.Cast("float32", T.min(LP, n))
        st[1] = sm[FO + kc]
        st[2] = sm[FO + lc]
        st[3] = sm[FO + kc + 1]
        st[4] = sm[FO + lc + 1]
        for j in T.unroll(4):
            st[5 + j] = sm[mb + j]
        for pq in T.unroll(P - 1):
            ot[0] = T.Cast("float32", T.min(LP, n - (pq + 1) * LP))
            ot[1] = sm[FO + (pq + 1) * d * 2 + kc]
            ot[2] = sm[FO + (pq + 1) * d * 2 + lc]
            ot[3] = sm[FO + (pq + 1) * d * 2 + kc + 1]
            ot[4] = sm[FO + (pq + 1) * d * 2 + lc + 1]
            for j in T.unroll(4):
                ot[5 + j] = sm[mb + (pq + 1) * ms + j]
            lw_merge(st, ot, ca, cb)

    entry_fused.stage_all = stage_all
    entry_fused.entry_smem = entry_smem
    entry_fused.staged_floats = FO + FSM
    return entry_fused, stage_fused


# --- kernels 2 and 3: fold the splits, then rho and the shrunk matrix ---------------------------
def _merge_kernel(b: int, n: int, d: int, mode: str, pl: dict, pvp: int = 1):
    """``lw_merge_splits``: one thread per stored pair entry of a problem.

    Fused: the ``P`` 9-states of the pair folded with ``lw_merge`` in split order (their
    moments loaded 16 splits at a time, the feature states staged in shared memory);
    unfused: the ``P`` covsum entries added. Writes ``S_kl`` to both triangles of ``Out``
    and the CTA's (``‖S‖²``, ``tr S``, ``Σ v``) double-float sums to ``Part``.
    """
    fused = mode == "fused"
    P, LP, NB, HW, NBC = pl["splits"], pl["lp"], pl["nb"], pl["hw"], pl["nbc"]  # noqa: F841
    BST_N, TST_N, FST_N, _ = sizes(b, d, P, NB, HW, NBC, "fused" if fused else "cov")
    PV_N = b * pvp * 2 if not fused else 1  # the v pass's (hi, lo) per (split, block chunk)
    NE, EPT, NCE = merge_layout(b, d, pl, mode)  # entries a problem, a thread; CTAs a problem
    FSM = P * d * 2 if fused else 1  # the problem's feature states, staged
    SPT = -(-FSM // EPI_THREADS)  # noqa: F841  staged values a thread
    NF = float(n)
    BATCH = min(16, P)
    NBT = -(-P // BATCH)  # noqa: F841

    @T.inline
    def entry_unfused(kv, mv, st, ot, ca, cb, fsm, Bst, Tst, Fst, bi, idx):
        """kv = (k, ell, valid); st[3] = st[5] = the summed covsum, st[8] = 0. The indices are
        locals: as nested expressions (each triangle row a 7-way sum, copied into every use) they
        took the TIRx passes 30-120 s at d > 128."""
        isb: T.bool = idx < 128 * NB
        ist: T.bool = (idx >= 128 * NB) & (idx < 128 * NB + NTRI * HW)
        j: T.int32 = idx // NB
        q: T.int32 = T.if_then_else(isb, idx % NB, 0)
        h: T.int32 = T.Cast("int32", T.sqrt(T.Cast("float32", q) + T.float32(0.5)))
        i: T.int32 = q - h * h
        t: T.int32 = idx - 128 * NB
        hh: T.int32 = t // NTRI
        tq: T.int32 = t % NTRI
        tr: T.int32 = _tri_row(tq)
        tc: T.int32 = tr + tq - (tr * 8 - (tr * (tr - 1)) // 2)  # _tri_col(tq)
        f: T.int32 = idx - 128 * NB - NTRI * HW
        k: T.int32 = f
        ell: T.int32 = f
        if isb:
            k = 8 * i + j // 16
            ell = 16 * h + j % 16
        elif ist:
            k = 16 * hh + 8 + tr
            ell = 16 * hh + 8 + tc
        kv[0] = k
        kv[1] = ell
        kv[2] = T.if_then_else((idx < NE) & (ell < d) & ((k < ell) | ((k == ell) & (f >= 0))), 1, 0)
        # the P splits' covsums as a compensated sum (Sum2; the diagonal's are (hi, lo) already):
        # the diagonal of S, added in float32 over P = 33 splits, lost 2e-7
        s = ot  # (sum, low part) in ot[0:2]; the epilogue takes the low part from ot[1]
        s[0] = T.float32(0)
        s[1] = T.float32(0)
        fc: T.int32 = T.min(T.max(f, 0), d - 1)
        if isb:  # one loop a source, so that each can be unrolled and its loads issued early
            for p in T.serial(P):
                sum2_add(s, 0, Bst[((bi * P + p) * 128 + j) * NB + q])
        elif ist:
            for p in T.serial(P):
                sum2_add(s, 0, Tst[((bi * P + p) * HW + hh) * NTRI + tq])
        else:
            for p in T.serial(P):
                sum2_add(s, 0, Fst[((bi * P + p) * d + fc) * 2])
                s[1] = s[1] + Fst[((bi * P + p) * d + fc) * 2 + 1]
        hs: T.float32 = s[0] + s[1]  # renormalised (Fast2Sum): S_kl's float32 is hs / n
        s[1] = s[1] - (hs - s[0])
        s[0] = hs
        st[3] = s[0]
        st[5] = s[0]
        st[8] = T.float32(0)

    entry_fused, stage_fused = _fold_parts(n, d, pl, EPI_THREADS)
    entry = entry_fused if fused else entry_unfused

    def s_low(sh, sv, ot):
        """The low part of S_kl = sh + low, sh = sv / n rounded: fused, ``sv`` is the folded
        float32 state; unfused, the compensated sum over the splits left its low part in ot[1]."""
        if fused:
            return _fma(T.float32(0) - sh, T.float32(NF), sv) / T.float32(NF)
        return (_fma(T.float32(0) - sh, T.float32(NF), sv) + ot[1]) / T.float32(NF)

    @T.inline
    def stage_none(fsm, Fst, bi, tid):
        T.evaluate(0)

    stage = stage_fused if fused else stage_none

    @T.inline
    def v_add(sums, Pv, bi, tid, c):
        if c == 0:
            pv: T.int32 = tid
            while pv < pvp:  # the v pass's (hi, lo) per (split, block chunk)
                dw_add(
                    sums, 4, sums[4], sums[5], Pv[(bi * pvp + pv) * 2], Pv[(bi * pvp + pv) * 2 + 1]
                )
                pv += EPI_THREADS

    @T.inline
    def v_none(sums, Pv, bi, tid, c):
        T.evaluate(0)

    vsum = v_none if fused else v_add
    lw_rho = _rho_parts(d, n)
    NPL = -(-NCE // 32)  # partial sums a lane of warp 0 adds
    VEC = 4 if (d * d) % 4 == 0 else 1
    PER = 4 * EPI_THREADS  # entries the CTA rescales a round
    NRS = -(-(d * d) // PER)
    RB = min(8, NRS)  # rounds whose loads are issued together
    NRB = -(-NRS // RB)

    @T.inline
    def rescale_scalar(Out, bi, e0, rho, rmu, buf):
        for u in T.unroll(4):
            e = e0 + u
            if e < d * d:
                s0: T.float32 = Out[bi, e // d, e % d]
                val: T.float32 = _fma(T.float32(0) - rho, s0, s0)
                Out[bi, e // d, e % d] = T.if_then_else(e // d == e % d, val + rmu, val)

    @T.inline
    def rescale_all_vec(Out, bi, tid, rho, rmu, buf):
        """RB rounds a batch: every 16-byte load first (clamped), then the guarded stores."""
        for rb in T.serial(NRB):
            for u in T.unroll(RB):
                e0 = T.min((rb * RB + u) * PER + tid * 4, d * d - 4)
                buf[T.Ramp(4 * u, 1, 4)] = Out[bi, e0 // d, T.Ramp(e0 % d, 1, 4)]
            for u in T.unroll(RB):
                e0 = (rb * RB + u) * PER + tid * 4
                if e0 < d * d:
                    for v in T.unroll(4):
                        val: T.float32 = _fma(T.float32(0) - rho, buf[4 * u + v], buf[4 * u + v])
                        buf[4 * u + v] = T.if_then_else(
                            (e0 + v) // d == (e0 + v) % d, val + rmu, val
                        )
                    Out[bi, e0 // d, T.Ramp(e0 % d, 1, 4)] = buf[T.Ramp(4 * u, 1, 4)]

    @T.inline
    def rescale_all_scalar(Out, bi, tid, rho, rmu, buf):
        for r in T.serial(NRS):
            rescale_scalar(Out, bi, r * PER + tid * 4, rho, rmu, buf)

    rescale_all = rescale_all_vec if VEC == 4 else rescale_all_scalar

    @T.inline
    def finish_problem(Part, Out, Rho, sums, osum, dwt, rs, buf, bi, tid):
        """The problem's last CTA: its NCE partial sums -> rho (as chain_ops), then the
        shrunk matrix (1 - rho) S + rho mu I in place."""
        if tid < 32:
            for q in T.unroll(6):
                sums[q] = T.float32(0)
            for m in T.unroll(NPL):
                if tid + 32 * m < NCE:
                    for q in T.unroll(6):
                        osum[q] = Part[bi * NCE + tid + 32 * m, q]
                    dw_combine(sums, osum)
            warp_allreduce(sums, osum, 6, dw_combine)
            if tid == 0:
                lw_rho(sums, dwt)
                rm = dwt[6]
                rs[0] = rm[0]
                rs[1] = rm[1]
                Rho[bi] = rm[0]
        T.cuda.cta_sync()
        rescale_all(Out, bi, tid, rs[0], rs[1], buf)

    @T.function
    def lw_merge_splits(
        Bst: T.Tensor((BST_N,), "float32", align=16),
        Tst: T.Tensor((TST_N,), "float32", align=16),
        Fst: T.Tensor((FST_N,), "float32"),
        Pv: T.Tensor((PV_N,), "float32"),
        Out: T.Tensor((b, d, d), "float32"),
        Part: T.Tensor((b * NCE, 6), "float32"),
        Cnt: T.Tensor((b,), "int32"),
        Rho: T.Tensor((b,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(b * NCE,), block=(EPI_THREADS,)))
        bx = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        warp = tid // 32
        lane = tid % 32
        bi = bx // NCE
        c = bx % NCE
        fsm = T.alloc_tensor(ty_args=[T.Tensor((FSM,), "float32", scope="shared.dyn")])
        st = T.alloc_local((9,), "float32")
        ot = T.alloc_local((9,), "float32")
        ca = T.alloc_local((6,), "float32")
        cb = T.alloc_local((6,), "float32")
        mv = T.alloc_local((BATCH * 4,), "float32")
        kv = T.alloc_local((3,), "int32")
        sums = T.alloc_local((6,), "float32")  # (‖S‖², tr S, Σ v) as (hi, lo) pairs
        osum = T.alloc_local((6,), "float32")
        smem = T.alloc_shared((EPI_WARPS, 6), "float32")
        last = T.alloc_shared((1,), "int32")
        rs = T.alloc_shared((2,), "float32")
        rbuf = T.alloc_local((4 * RB,), "float32")
        dwt = (
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
            T.alloc_local((2,), "float32"),
        )
        for q in T.unroll(6):
            sums[q] = T.float32(0)
        stage(fsm, Fst, bi, tid)
        for u in T.unroll(EPT):  # EPT <= EPT_MAX: the code does not grow with b
            idx = (c * EPT + u) * EPI_THREADS + tid
            entry(kv, mv, st, ot, ca, cb, fsm, Bst, Tst, Fst, bi, idx)
            if kv[2] == 1:
                k = kv[0]
                ell = kv[1]
                sv: T.float32 = T.if_then_else(k == ell, st[3], st[5])
                sh: T.float32 = _div(sv, T.float32(NF))
                sl: T.float32 = s_low(sh, sv, ot)
                Out[bi, k, ell] = sh
                Out[bi, ell, k] = sh
                w: T.float32 = T.if_then_else(k == ell, T.float32(1), T.float32(2))
                sq: T.float32 = sh * sh
                sqe: T.float32 = _fma(sh, sh, T.float32(0) - sq) + T.float32(2) * sh * sl
                dw_add(sums, 0, sums[0], sums[1], w * sq, w * sqe)
                if k == ell:
                    dw_add(sums, 2, sums[2], sums[3], sh, sl)
                dw_add(sums, 4, sums[4], sums[5], w * st[8], T.float32(0))
        vsum(sums, Pv, bi, tid, c)
        cta_allreduce(sums, osum, 6, dw_combine, smem, warp, lane, EPI_WARPS)
        # publish; the problem's last CTA (the one that counts NCE) finishes the problem and
        # resets the counter for the next call
        T.cuda.thread_fence()
        T.cuda.cta_sync()
        if tid == 0:
            for q in T.unroll(6):
                Part[bx, q] = sums[q]
            T.cuda.thread_fence()
            done = T.cuda.atomic_add(T.address_of(Cnt[bi]), T.int32(1))
            last[0] = T.if_then_else(done == NCE - 1, 1, 0)
            if done == NCE - 1:
                Cnt[bi] = T.int32(0)  # every other CTA of the problem has counted
        T.cuda.cta_sync()
        if last[0] == 1:
            T.cuda.thread_fence()
            finish_problem(Part, Out, Rho, sums, osum, dwt, rs, rbuf, bi, tid)

    return lw_merge_splits, NCE


@cached
def build(cfg: dict, variant: str = "fused"):
    """The run function of ``variant`` for these sizes. Its scratch buffers (split states, the
    last-CTA counters) come from ``workspace``: one set per CUDA stream (calls on one stream are
    ordered, and each leaves every counter at 0: the CTA that counts last resets it), fresh
    under CUDA-graph capture. ``run.counters()`` is the current stream's counters.

    Every kernel is compiled keeping subnormals (``ftz=False``): data of scale ~1e-9 has
    subnormal fourth powers and double-float low parts, and flushing them moved rho by ~10%."""
    assert variant in ("fused", "unfused"), variant
    b, n, d = cfg["b"], cfg["n"], cfg["d"]
    assert b * n * d < 2**31, "int32 indexing"
    name = f"{b}x{n}x{d}"
    dev = "cuda"
    key = ("ledoit_wolf", b, n, d, variant)
    if variant == "fused":
        k1, pl = _syrk_kernel(b, n, d, "fused")
        mod1 = compile_sm90(k1, name=f"lw_syrk_{name}", ftz=False)
        k2, nce = _merge_kernel(b, n, d, "fused", pl)
        merge_in = pl["epi_in"] or pl["last_in"]  # finished in lw_syrk: no merge kernel
        mod2 = None if merge_in else compile_sm90(k2, name=f"lw_merge_{name}", ftz=False)
        n1, n2, n3, _ = sizes(b, d, pl["splits"], pl["nb"], pl["hw"], pl["nbc"], "fused")

        def make():
            return (
                torch.empty(n1, device=dev, dtype=F32),  # block pair states
                torch.empty(n2, device=dev, dtype=F32),  # triangle pair states
                torch.empty(n3, device=dev, dtype=F32),  # feature states
                torch.empty(b * nce, 6, device=dev, dtype=F32),  # merge CTAs' partial sums
                torch.zeros(b, device=dev, dtype=torch.int32),  # per problem, 0 between calls
                torch.zeros(1, device=dev, dtype=torch.int32),
                torch.empty(b, d, device=dev, dtype=F32),
                torch.empty(1, device=dev, dtype=F32),
            )

        def counters():
            return workspace(key, make)[4]

        if pl["epi_in"]:

            def run_in(X):
                _, _, _, _, _, icnt, ms, dummy = workspace(key, make)
                out = torch.empty(b, d, d, device=X.device, dtype=F32)
                rho = torch.empty(b, device=X.device, dtype=F32)
                mod1(X, ms, dummy, dummy, dummy, dummy, out.view(-1), rho, icnt)
                return (out, rho)

            run_in.counters = counters
            return run_in

        if pl["last_in"]:

            def run_last(X):
                bst, tst, fst, _, cnt, _, ms, dummy = workspace(key, make)
                out = torch.empty(b, d, d, device=X.device, dtype=F32)
                rho = torch.empty(b, device=X.device, dtype=F32)
                mod1(X, ms, bst, tst, fst, dummy, out.view(-1), rho, cnt)
                return (out, rho)

            run_last.counters = counters
            return run_last

        def run(X):
            bst, tst, fst, part, cnt, icnt, ms, dummy = workspace(key, make)
            out = torch.empty(b, d, d, device=X.device, dtype=F32)
            rho = torch.empty(b, device=X.device, dtype=F32)
            mod1(X, ms, bst, tst, fst, dummy, dummy, dummy, icnt)
            mod2(bst, tst, fst, dummy, out, part, cnt, rho)
            return (out, rho)

        def parts(X):
            """(name, launch) of each kernel, for timing them one by one."""
            bst, tst, fst, part, cnt, icnt, ms, dummy = workspace(key, make)
            out = torch.empty(b, d, d, device=X.device, dtype=F32)
            rho = torch.empty(b, device=X.device, dtype=F32)
            return [
                ("syrk", lambda: mod1(X, ms, bst, tst, fst, dummy, dummy, dummy, icnt)),
                ("merge", lambda: mod2(bst, tst, fst, dummy, out, part, cnt, rho)),
            ]

        run.parts = parts
        run.counters = counters
        return run

    km, msplits = _mean_kernel(b, n, d)
    kc, plc = _syrk_kernel(b, n, d, "cov", msplits)
    kv, plv = _syrk_kernel(b, n, d, "v", msplits)
    assert plc["splits"] == plv["splits"] and plc["lp"] == plv["lp"]
    pvp = plv["splits"] * plv["nbc"]
    ke, nce = _merge_kernel(b, n, d, "cov", plc, pvp)
    modm = compile_sm90(km, name=f"lw_mean_{name}", ftz=False)
    modc = compile_sm90(kc, name=f"lw_cov_{name}", ftz=False)
    modv = compile_sm90(kv, name=f"lw_v_{name}", ftz=False)
    mode_ = compile_sm90(ke, name=f"lw_merge_unfused_{name}", ftz=False)
    n1, n2, n3, _ = sizes(b, d, plc["splits"], plc["nb"], plc["hw"], plc["nbc"], "cov")

    def make_unfused():
        return (
            torch.empty(b * msplits, d, device=dev, dtype=F32),  # the mean pass's sums
            torch.empty(n1, device=dev, dtype=F32),  # covsum blocks
            torch.empty(n2, device=dev, dtype=F32),  # covsum triangles
            torch.empty(n3, device=dev, dtype=F32),  # covsum diagonal, (hi, lo)
            torch.empty(b * pvp * 2, device=dev, dtype=F32),  # the v pass's sums, (hi, lo)
            torch.empty(b * nce, 6, device=dev, dtype=F32),  # merge CTAs' partial sums
            torch.zeros(b, device=dev, dtype=torch.int32),  # per problem, 0 between calls
            torch.empty(1, device=dev, dtype=F32),
            torch.zeros(1, device=dev, dtype=torch.int32),
        )

    def counters_unfused():
        return workspace(key, make_unfused)[6]

    def run_unfused(X):
        ms, cb, ct, cd, pv, part, cnt, d1, i1 = workspace(key, make_unfused)
        out = torch.empty(b, d, d, device=X.device, dtype=F32)
        rho = torch.empty(b, device=X.device, dtype=F32)
        modm(X, ms)
        modc(X, ms, cb, ct, cd, d1, d1, d1, i1)
        modv(X, ms, d1, d1, d1, pv, d1, d1, i1)
        mode_(cb, ct, cd, pv, out, part, cnt, rho)
        return (out, rho)

    def parts(X):
        """(name, launch) of each kernel, for timing them one by one."""
        ms, cb, ct, cd, pv, part, cnt, d1, i1 = workspace(key, make_unfused)
        out = torch.empty(b, d, d, device=X.device, dtype=F32)
        rho = torch.empty(b, device=X.device, dtype=F32)
        return [
            ("mean", lambda: modm(X, ms)),
            ("cov", lambda: modc(X, ms, cb, ct, cd, d1, d1, d1, i1)),
            ("v", lambda: modv(X, ms, d1, d1, d1, pv, d1, d1, i1)),
            ("merge", lambda: mode_(cb, ct, cd, pv, out, part, cnt, rho)),
        ]

    run_unfused.parts = parts
    run_unfused.counters = counters_unfused
    return run_unfused
