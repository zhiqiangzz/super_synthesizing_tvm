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
"""The ledoit_wolf kernels beyond the gate: every W-SYRK layout, splits, and data far from zero.

The gate's inputs have mean 0. Here the features also sit at offsets of
hundreds of standard deviations, where raw moments lose every digit in float32
and a float32 mean of the raw data is off by 1e-5 of the spread; the moments
about the reference point K, the binomial shift and the centred merges (fused),
and the centring as (x - x0) - mean(x - x0) (unfused), must stay as accurate as
float32 allows. The sizes cover one feature (rho = 0), W widths 16 .. 512 (one
8 x 16 block a problem up to 1024 blocks over 4 CTAs), the scalar (d % 4 != 0)
and 16-byte load paths, partial W groups (d not a power of two), the in-CTA
finish (P = 1), the problem's last CTA folding a few splits staged in shared
memory (P = 3) and the merge kernel (P > 4), sub-ranges added in shared memory,
and a last split and step that are partial. Both variants: the one-pass fused
kernel and the unfused passes. Also: the consumers' paired block mapping covers
every block once, and two CUDA streams get workspaces of their own.

Regression tests of the verifiers' findings: rho = 0 where there is no spread
(beta = 0), split states that do not fit the last CTA's shared memory, no reads
past the triangle states (compute-sanitizer), counters back at 0 after every
call, outlier blocks, a spike and periodic outliers (the fused reference point
K), and data of scale 1e-9 / 1e-10 (subnormals kept). Then: problems wider than
256 features, whose blocks span several CTAs of the unfused passes; a minority
of the samples shifted far off the rest (the fused dx² dy² sums and K); a merge
kernel whose code no longer grows with b; and a ramp feature (the unfused
variant's sums). Where float32 itself loses digits the bound is the gate's:
max(2 x the error of the unfused torch program in float32, 1e-6).
"""

import os
import shutil
import subprocess
import sys
import time

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import ledoit_wolf

FLOOR = 1e-6
SPEC = SPECS["ledoit_wolf"]

CASES = [
    # (b, n, d)
    (2, 33, 1),  # one feature: rho = 0
    (3, 37, 5),  # W width 16, scalar loads, one partial step, in-CTA finish
    (2, 700, 16),  # 16-byte loads, sub-ranges, 3 splits folded by the problem's last CTA
    (2, 300, 17),  # partial W groups (d < 32), scalar loads, last-CTA fold
    (1, 2000, 96),  # partial W groups (d < 128), many splits: the merge kernel
    (1, 100, 129),  # W width 512: 1024 blocks over 4 CTAs a problem, scalar loads
    (2, 5001, 40),  # 16-byte loads, many splits, partial last step
]


def _err(out, ref):
    scale = ref.abs().max()
    return (
        ((out.double() - ref).abs().max() / scale).item() if scale > 0 else out.abs().max().item()
    )


def _data(b, n, d, offset, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    (x,) = SPECS["ledoit_wolf"].inputs({"b": b, "n": n, "d": d}, gen)
    shift = offset * (1.0 + torch.rand(b, 1, d, generator=gen, device="cuda"))
    return (x + shift).contiguous()


def _check(outs, refs):
    for out, ref in zip(outs, refs):
        assert out.shape == ref.shape and out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert _err(out, ref) <= FLOOR


def _check_gate(outs, x):
    """The gate's rule, with the unfused torch program in float32 (eager, TF32 off) as the
    yardstick: each output within max(2 x its error, FLOOR) of the float64 reference."""
    refs = SPEC.reference(x)
    yard = SPEC.unfused(x)
    for out, ref, base in zip(outs, refs, yard):
        assert out.shape == ref.shape and torch.isfinite(out).all()
        assert _err(out, ref) <= max(2 * _err(base, ref), FLOOR)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, n, d", CASES)
@pytest.mark.parametrize("offset", [0.0, 300.0])
def test_ledoit_wolf_against_float64(b, n, d, offset, variant):
    X = _data(b, n, d, offset)
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    refs = SPECS["ledoit_wolf"].reference(X)
    for _ in range(2):  # twice: the last-CTA counters only grow across calls
        _check(run(X), refs)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, n, d", [(2, 700, 16), (2, 5001, 40)])
def test_two_streams(b, n, d, variant):
    """Calls on two streams at once: each stream has its own split states and counters."""
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    xs = [_data(b, n, d, 0.0, seed) for seed in (1, 2)]
    refs = [SPECS["ledoit_wolf"].reference(x) for x in xs]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    outs = [[], []]
    for _ in range(3):
        for k in range(2):
            with torch.cuda.stream(streams[k]):
                outs[k].append(run(xs[k]))
    torch.cuda.synchronize()
    for k in range(2):
        for o in outs[k]:
            _check(o, refs[k])


@pytest.mark.parametrize("hw", [2, 4, 8, 16, 32])
def test_paired_blocks_cover_the_triangle(hw):
    """The consumers' block mapping: every block (i, h), i <= 2h, at exactly one position."""
    got = [ledoit_wolf.paired_block(j, hw) for j in range(hw * hw)]
    assert sorted(got) == sorted((i, h) for h in range(hw) for i in range(2 * h + 1))
    for j0 in range(0, hw * hw, 32):  # a warp's positions: at most 4 super-blocks
        assert len({h for _, h in got[j0 : j0 + 32]}) <= 4


@pytest.mark.parametrize("b, n, d", [*CASES, (256, 1024, 32), (64, 4096, 64), (8, 65536, 128)])
def test_plan_covers_the_paths(b, n, d):
    for mode in ("fused", "cov", "v"):
        pl = ledoit_wolf.plan(b, n, d, mode)
        assert pl["sc"] * pl["chunk"] * pl["dw"] == ledoit_wolf.STAGE
        assert pl["chunk"] % 2 == 0  # the consumers fold 2 samples an iteration
        assert pl["ls"] % pl["chunk"] == 0 and (pl["splits"] - 1) * pl["lp"] < n
        assert pl["grid"] == pl["groups"] * pl["splits"] * pl["nbc"]


def test_bench_plans():
    """One wave of 1-CTA/SM blocks at every bench size; d = 32 finishes in the CTA, d = 64 in
    the problem's last CTA (2 splits), d = 128 in the merge kernel (16 splits)."""
    for (b, n, d), splits in [((256, 1024, 32), 1), ((64, 4096, 64), 2), ((8, 65536, 128), 16)]:
        pl = ledoit_wolf.plan(b, n, d)
        assert pl["grid"] <= 132
        assert pl["epi_in"] == (d == 32)
        assert pl["last_in"] == (d == 64)
        assert pl["nbc"] == 1
        assert pl["splits"] == splits


# --- regression tests of the verifiers' findings ----------------------------------------------
@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize(
    "b, n, d",
    [
        (3, 37, 5),  # in-CTA finish: a constant problem and an all-zero one
        (2, 700, 16),  # last-CTA fold: a constant problem
        (2, 5001, 40),  # merge kernel: a constant problem
        (2, 1, 7),  # one sample: S = 0 (in-CTA finish)
        (1, 1, 128),  # one sample: S = 0 (merge kernel)
        (2, 4, 2),  # S = mu I exactly: samples (+-1, 0), (0, +-1)
    ],
)
def test_rho_is_zero_without_spread(b, n, d, variant):
    """beta = min(.., delta) = 0 gives rho = 0 (bench.ops._shrink): a constant or all-zero
    problem, a single sample (delta = beta = 0) and S = mu I (delta = 0 < beta). rho was 1."""
    X = _data(b, n, d, 0.0)
    if n == 1:
        flat = list(range(b))
    elif (n, d) == (4, 2):
        X[0] = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device="cuda")
        flat = [0]
    else:
        X[0] = X[0, 0].clone()  # every sample the first one
        flat = [0]
        if b > 2:
            X[1] = 0.0
            flat.append(1)
    refs = SPEC.reference(X)
    assert (refs[1][flat] == 0).all()
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    out, rho = run(X)
    assert (rho[flat] == 0).all(), rho[flat]
    _check((out, rho), refs)


@pytest.mark.parametrize(
    "b, n, d, splits, last_in",
    [
        (64, 1024, 128, 2, False),  # 2 x 33,664 floats of split states: the merge kernel
        (32, 4096, 128, 4, False),
        (1, 100, 128, 4, False),
        (50, 2000, 100, 2, False),  # partial W groups
        (40, 300, 64, 3, True),  # 25,920 floats: past the W ring, within the whole buffer
        (30, 4096, 64, 4, True),  # 34,560 floats
    ],
)
def test_split_states_fit_or_go_to_the_merge_kernel(b, n, d, splits, last_in):
    """A few splits of a problem (P <= 4) are folded by its last CTA only when their states fit
    the CTA's dynamic shared memory, else by the merge kernel; the build used to assert."""
    pl = ledoit_wolf.plan(b, n, d)
    assert pl["splits"] == splits and pl["pp"] == 1 and pl["nbc"] == 1
    assert pl["last_in"] == last_in == (pl["staged"] <= ledoit_wolf.SMEM_FUSED)
    X = _data(b, n, d, 0.0)
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant="fused")
    _check(run(X), SPEC.reference(X))


def test_last_cta_fold_plans_fit():
    """Over many sizes, every plan that folds in the last CTA stages what fits."""
    for d in (17, 32, 33, 48, 64, 65, 100, 128):
        for b in range(1, 201, 3):
            for n in (100, 300, 1000, 4096, 16384, 65536):
                if b * n * d < 2**31:
                    pl = ledoit_wolf.plan(b, n, d)
                    assert not pl["last_in"] or pl["staged"] <= ledoit_wolf.SMEM_FUSED


_MEMCHECKED = """
import torch
from bench.ops import SPECS
from kernels import ledoit_wolf
for b, n, d in ((2, 1001, 33), (1, 4099, 64)):  # merge kernel: P = 16, then P = 65 (batched)
    gen = torch.Generator(device="cuda").manual_seed(0)
    (x,) = SPECS["ledoit_wolf"].inputs({"b": b, "n": n, "d": d}, gen)
    ledoit_wolf.build({"b": b, "n": n, "d": d}, variant="fused")(x)
    torch.cuda.synchronize()
"""


def test_merge_kernel_reads_in_bounds():
    """compute-sanitizer memcheck of the fused merge kernel, each tensor its own allocation: the
    padded pair entries past a problem's read 16-byte triangle states up to 2.8 KB past Tst."""
    san = shutil.which("compute-sanitizer")
    if san is None:
        pytest.skip("compute-sanitizer is not on PATH")
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
    cmd = [san, "--tool", "memcheck", "--kernel-name", "kns=lw_", "--error-exitcode", "3"]
    res = subprocess.run(
        [*cmd, sys.executable, "-c", _MEMCHECKED],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    log = res.stdout + res.stderr
    assert res.returncode == 0 and "ERROR SUMMARY: 0 errors" in log, log[-3000:]


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, n, d", [(2, 700, 16), (2, 5001, 40)])
def test_counters_return_to_zero(b, n, d, variant):
    """Every call leaves the last-CTA counters at 0 (the CTA that counts last resets its
    problem's), so they never approach the int32 wrap: a grow-only counter elected the wrong
    CTA after ~2^31 / M calls (no CTA, or one that folded unpublished partials)."""
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    xs = [_data(b, n, d, 0.0, seed) for seed in (1, 2)]
    refs = [SPEC.reference(x) for x in xs]
    for k in range(4):
        outs = run(xs[k % 2])
        torch.cuda.synchronize()
        assert (run.counters() == 0).all()
        _check(outs, refs[k % 2])


@pytest.mark.parametrize(
    "b, n, d", [(256, 1024, 32), (64, 4096, 64), (8, 65536, 128), (40, 4096, 64)]
)
def test_reference_point_on_structured_data(b, n, d):
    """Fused: blocks of 8 or 32 samples offset by 30 or 1000 (data sd 1-2) at the problem's
    start, at split 1's and in the middle of split 0 and of split 1, a 1e4 spike at sample 0,
    and every (split / 32)-th sample offset by 30 at the phase of evenly spaced samples. The
    moments are about a reference point K that must sit near each split's mean: the mean of a
    split's first step put it far off on a block there (up to 30x torch.compile's error), and
    K's samples would alias with the periodic data if evenly spaced. A block the samples hit
    pulls their mean off the bulk; with each pair's dx² dy² a float32 running sum the bulk's
    terms then rounded away (up to 7x), which a block's compensated sum of them absorbs."""
    pl = ledoit_wolf.plan(b, n, d)
    lp = pl["lp"]
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant="fused")
    base = _data(b, n, d, 0.0, seed=2)
    starts = sorted({0, lp % n, lp // 2, (lp + 3 + lp // 2) % n})
    for s0 in starts:
        for size, off in ((8, 30.0), (8, 1000.0), (32, 1000.0)):
            x = base.clone()
            x[:, s0 : s0 + size, :] += off
            _check_gate(run(x), x)
    x = base.clone()
    x[:, 0, 0] += 1.0e4
    _check_gate(run(x), x)
    stride = lp // ledoit_wolf.MK
    x = base.clone()
    x[:, stride // 2 :: stride, :] += 30.0
    _check_gate(run(x), x)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, n, d", [(3, 37, 5), (2, 700, 16), (2, 5001, 40)])
@pytest.mark.parametrize("scale", [1e-9, 1e-10])
def test_tiny_scale(b, n, d, scale, variant):
    """Data of scale 1e-9 / 1e-10: the fourth powers, ‖S‖² and the double-float low parts are
    subnormal; flushed to zero (FTZ) they moved rho by ~10% at 1e-9 and to 1 at 1e-10."""
    X = (_data(b, n, d, 0.0) * scale).contiguous()
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    _check_gate(run(X), X)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, n, d", [(2, 600, 300), (1, 1000, 512)])
def test_wide_problems_span_block_chunks(b, n, d, variant):
    """256 < d <= 512: a problem's 8 x 16 blocks span NBC = 4 CTAs of the unfused cov and v
    passes, and every one of them stages all of the problem's W groups. The v pass added the
    producers' triangles' Σ v in all four, and the merge kernel adds the four: rho was 3-5% off
    (0.8137 for 0.7902 at (1, 1000, 512)). The fused kernel (NBC = 16) adds them in one."""
    for mode in ("cov", "v"):
        assert ledoit_wolf.plan(b, n, d, mode)["nbc"] == 4
    X = _data(b, n, d, 0.0)
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    _check(run(X), SPEC.reference(X))


def _shifted_minority(x, kind, seed):
    """A minority of the samples shifted by +50 in every feature (the data's sd is 1-2): every
    k-th sample, or each sample with probability 0.15."""
    if kind == "bernoulli":
        gen = torch.Generator(device="cuda").manual_seed(100 + seed)
        rows = torch.rand(x.shape[0], x.shape[1], 1, generator=gen, device="cuda") < 0.15
        return (x + 50.0 * rows.float()).contiguous()
    y = x.clone()
    y[:, :: int(kind), :] += 50.0
    return y


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, n, d", [(256, 1024, 32), (64, 4096, 64), (8, 65536, 128)])
def test_shifted_minority(b, n, d, variant):
    """Every 5th, 7th or 9th sample, or 15% of them at random, shifted by +50 sd. The fused
    kernel summed each pair's dx² dy² in float32 over up to 4096 samples: next to the
    cluster's (50)⁴ the bulk's terms rounded away, a systematic loss of up to 1e-5 of Σ v (rho
    10x over the bound at (8, 65536, 128)) once a robust K had dropped the cluster and sat in
    the bulk; with K there the bulk's dx² dy also rounded away in t21 (2e-6). A block's dx²
    dy² are now one compensated sum, and K is the plain mean of the split's samples."""
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    for seed in (0, 1):
        x = _data(b, n, d, 0.0, seed)
        for kind in ("5", "7", "9", "bernoulli"):
            y = _shifted_minority(x, kind, seed)
            _check_gate(run(y), y)


@pytest.mark.parametrize("diagonal", [False, True])
def test_block_v_is_the_weighted_sum_of_its_pairs(diagonal):
    """A consumer's V, rs (2 hi + m lo) a sample, is Σ_kl w_kl dx_k² dy_l² over its block's 32
    pairs with the epilogue's weights (2 for k < l, 1 for k = l, 0 for the duplicates k > l of a
    diagonal block), and the producer's (Σ dx²)² that of its triangle's 10 pairs (k <= l)."""
    hw = 4
    gen = torch.Generator().manual_seed(3)
    x2 = torch.rand(8 * hw, generator=gen, dtype=torch.float64)  # dx² of each feature
    for h in range(hw):
        for i in [2 * h] if diagonal else range(2 * h):
            pairs = 0.0
            for kk in range(4):
                for c in range(8):
                    k, ell = 4 * i + kk, 8 * h + c
                    pairs += (2.0 if k < ell else 1.0 if k == ell else 0.0) * x2[k] * x2[ell]
            rs = x2[4 * i : 4 * i + 4].sum()
            lo, hi = x2[8 * h : 8 * h + 4].sum(), x2[8 * h + 4 : 8 * h + 8].sum()
            m = 1.0 if i == 2 * h else 2.0
            assert torch.isclose(rs * (2 * hi + m * lo), pairs, rtol=1e-12)
        tri = x2[8 * h + 4 : 8 * h + 8]
        pairs = sum(
            (2.0 if a < c else 1.0) * tri[a] * tri[c] for a in range(4) for c in range(a, 4)
        )
        assert torch.isclose(tri.sum() ** 2, pairs, rtol=1e-12)


def test_merge_kernel_layout_is_bounded():
    """lw_merge_splits unrolls EPT pair entries a thread: at most EPT_MAX, never past a
    problem's entries, and its NCE CTAs a problem cover them without an empty one. EPT grew
    with b, nearly all of it padding (507 entries a thread for the 33 needed at b = 4096,
    d = 128): the kernel took 160 s to compile there, and over 90 min at b = 100000, d = 32
    (unfused)."""
    for mode in ("fused", "cov"):
        for d in (1, 16, 32, 33, 64, 100, 128, 129, 300):
            for b in (1, 8, 64, 264, 1000, 4096, 100000):
                if b * 64 * d >= 2**31:
                    continue
                pl = ledoit_wolf.plan(b, 64, d, mode)
                ne, ept, nce = ledoit_wolf.merge_layout(b, d, pl, mode)
                per_cta = ept * ledoit_wolf.EPI_THREADS
                assert 1 <= ept <= ledoit_wolf.EPT_MAX
                assert ept <= -(-ne // ledoit_wolf.EPI_THREADS)
                assert (nce - 1) * per_cta < ne <= nce * per_cta


@pytest.mark.parametrize(
    "variant, b, n, d", [("fused", 4096, 128, 128), ("unfused", 100000, 16, 32)]
)
def test_many_problems_build_and_match(variant, b, n, d):
    """The verifier's sizes for a merge kernel unrolled over b: they now build in seconds (the
    bound leaves a 10x margin) and match float64."""
    t0 = time.time()
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    assert time.time() - t0 < 120
    X = _data(b, n, d, 0.0)
    _check(run(X), SPEC.reference(X))


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("b, n, d", [(1, 4099, 64), (3, 50, 7), (2, 1001, 33)])
def test_ramp_feature(b, n, d, variant):
    """Feature 0 a time index 0 .. n - 1, the others the gate's data: S_00 and Σ v are most of
    ‖S‖² and Σ v, and rho carries 2-4.5x their error. The unfused variant added in float32 the
    producers' partial sums of S's diagonal (up to 32) and the splits' covsums (33 at
    (1, 4099, 64)), 2e-7 of S_00, and each block's 128 weighted Σ v entries, 7e-7 of Σ v at
    (3, 50, 7): rho 1.6e-6 and 1.5e-6, over the gate's bound. Those sums are now compensated."""
    run = ledoit_wolf.build({"b": b, "n": n, "d": d}, variant=variant)
    t = torch.arange(n, device="cuda", dtype=torch.float32)
    for seed in (0, 1, 2):
        x = _data(b, n, d, 0.0, seed)
        x[:, :, 0] = t
        _check_gate(run(x), x)
