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
"""The ols_hc0 kernels beyond the gate: good fits, zeros, outliers, scaled data, streams,
CUDA graphs.

The gate's rows are noisy (residuals as large as y). In a good fit the meat
Σ x² (y - θ x)² is a small residual sum, and float32's error in the slope θ
then dominates it: torch's own program loses digits there. The kernel moves
the meat to the slope of compensated sums (float32 quotients whose remainders
are exact by FMA), and must stay within the gate's float32 floor where the
residuals are 1e-3 of y. Rows of zeros give 0 exactly. X, or X and Y together,
scaled by powers of two scale the result exactly, and data in small units
(X and Y ~1e-9 and below, where the meat's terms leave float32's normal range)
stays within the floor. Calls overlapping on two streams do not share
workspaces, nor do graphs of one call replayed on two streams. Every layout
``plan`` picks -- a warp, a team of warps or the CTA per row, a row split over
CTAs with a partial last segment, 16-byte or one-float loads -- runs in both
variants (one fused pass, and the unfused program's two passes), and the
benchmark sizes get the layouts and occupancy they were tuned for.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench import shapes
from bench.ops import SPECS
from kernels import ols_hc0
from kernels.common.build import SMS, ctas_per_sm

VARIANTS = ["fused", "unfused"]
CASES = [
    # (n, m, tpr, split): threads per task (a warp, a team of warps, the CTA), split over CTAs
    (7, 1000, 32, False),  # 16-byte loads, a warp per row: one full step, then single loads
    (5, 1001, 32, False),  # unaligned rows: one float at a time, a warp per row
    (100, 16384, 128, False),  # teams of 4 warps merged through shared memory, 2 per CTA
    (300, 65536, 256, False),  # a CTA per row
    (1, 77777, 256, True),  # unaligned and split, a partial last segment
    (3, 1000004, 256, True),  # 16-byte loads, split, single loads after the last full step
]
SIZES = [(n, m) for n, m, _, _ in CASES]


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def _run(X, Y, variant):
    n, m = X.shape
    (out,) = ols_hc0.build({"n": n, "m": m}, variant=variant)(X, Y)
    (ref,) = SPECS["ols_hc0"].reference(X, Y)
    (base,) = SPECS["ols_hc0"].unfused(X, Y)
    assert out.shape == base.shape and out.dtype == base.dtype
    assert torch.isfinite(out).all()
    return out, ref, base


def _data(n, m, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, m, generator=gen, device="cuda")
    e = torch.randn(n, m, generator=gen, device="cuda")
    return x, e


def _gate_inputs(n, m, seed):
    return SPECS["ols_hc0"].inputs(
        {"n": n, "m": m}, torch.Generator(device="cuda").manual_seed(seed)
    )


def _counters(run):
    """The split counters of every stream's workspace of ``run`` (its eager calls)."""
    return [C for _, work in run.workspaces.values() for _, C in work]


def _released_together(fa, fb):
    """Run ``fa`` and ``fb`` on two streams, both released by one event recorded after a
    sleep, so that their kernels overlap on the GPU."""
    s0, s1, s2 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    ready = torch.cuda.Event()
    with torch.cuda.stream(s0):
        torch.cuda._sleep(2_000_000)
        ready.record(s0)
    s1.wait_event(ready)
    s2.wait_event(ready)
    with torch.cuda.stream(s1):
        a = fa()
    with torch.cuda.stream(s2):
        b = fb()
    torch.cuda.synchronize()
    return a, b


@pytest.mark.parametrize("n, m, tpr, split", CASES)
def test_layout(n, m, tpr, split):
    for stage in (ols_hc0.FUSED, ols_hc0.NORMAL, ols_hc0.MEAT):
        g = ols_hc0.plan(n, m, stage)
        assert (g.tpr, g.split, g.w) == (tpr, split, 4 if m % 4 == 0 else 1)
        assert (g.S - 1) * g.seg < g.loads <= g.S * g.seg
        if split:
            assert g.tasks <= SMS * ols_hc0.CTAS_PER_SM  # every segment's CTA resident at once


def test_bench_layouts_and_occupancy():
    """Short rows a warp each, long rows a CTA each, few rows split to fill every resident
    CTA slot; every kernel keeps the CTAs per SM its grid is sized for."""
    slots = SMS * ols_hc0.CTAS_PER_SM
    want = {"short": (32, False), "long": (256, False), "split": (256, True)}
    for regime, cfg in shapes.configs("ols_hc0"):
        for variant in VARIANTS:
            run = ols_hc0.build(cfg, variant=variant)
            for g in run.plans:
                assert (g.tpr, g.split) == want[regime]
                if g.split:
                    assert 0.9 * slots <= g.tasks <= slots
            for mod in run.modules.values():
                assert min(ctas_per_sm(mod, run.threads).values()) >= ols_hc0.CTAS_PER_SM


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", SIZES)
def test_gate_inputs_against_float64(n, m, variant):
    X, Y = _gate_inputs(n, m, 3)
    out, ref, _ = _run(X, Y, variant)
    assert _err(out, ref) <= 3e-7


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", SIZES)
def test_good_fit_against_float64(n, m, variant):
    x, e = _data(n, m)
    out, ref, base = _run(x, 10.0 * x + 1e-3 * e, variant)
    assert _err(out, ref) <= max(_err(base, ref), 1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", SIZES)
def test_zeros_and_outliers(n, m, variant):
    x, e = _data(n, m, seed=1)
    gen = torch.Generator(device="cuda").manual_seed(2)
    keep = torch.rand(n, m, generator=gen, device="cuda") < 0.1
    x = torch.where(keep, x, 0.0)  # 90% of x exactly 0
    y = 0.5 * x + e
    y[:, m // 2] = 1000.0  # a 1000-sigma outlier in every row
    x[0] = 0.0  # Σx² == 0: the result is 0
    out, ref, base = _run(x, y, variant)
    assert out[0].item() == 0.0
    if n > 1:
        assert _err(out, ref) <= max(2 * _err(base, ref), 1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", SIZES)
def test_power_of_two_scaling_of_x(n, m, variant):
    """HC0 is homogeneous, out(s X, Y) = out(X, Y) / s². With s a power of two every float
    op of the kernel scales exactly, unless a value leaves float32's normal range
    (``--use_fast_math`` flushes subnormals to 0). The fused state's Σx⁴ did for |x| below
    ~3e-10: the gate's inputs scaled by 2^-33 lost 15% of the result. The fused state now
    keeps the range of x², as the unfused program does."""
    X, Y = _gate_inputs(n, m, 5)
    run = ols_hc0.build({"n": n, "m": m}, variant=variant)
    base = run(X, Y)[0].clone()
    for k in (-24, 29, 33, 40):  # |x| ~ 1.7e7, 1.9e-9, 1.2e-10, 9.1e-13
        Xk = X * 2.0**-k
        (out,) = run(Xk, Y)
        (ref,) = SPECS["ols_hc0"].reference(Xk, Y)
        assert _err(out, ref) <= 1e-6, f"X * 2^{-k}"
        assert torch.equal(out, base * 2.0 ** (2 * k)), f"X * 2^{-k}"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", SIZES)
def test_joint_scaling_of_x_and_y(n, m, variant):
    """HC0 is invariant under scaling X and Y together: the slope is unchanged, and the meat
    and a² both scale by s⁴. The meat's terms (x (y - θ x))² have degree 4 and fell below
    2^-126 for data around 1e-9, where ``--use_fast_math`` flushed them to 0: both variants
    lost 8.8e-5 of the result at 1e-9 and 70% at 1e-10. The kernels keep every value of
    degree 2, so with s a power of two the result is the unscaled one bit for bit, and data
    in small or large units stays within the float32 floor."""
    X, Y = _gate_inputs(n, m, 8)
    run = ols_hc0.build({"n": n, "m": m}, variant=variant)
    base = run(X, Y)[0].clone()
    for k in (30, 34, 40, -28, -50):  # |x| ~ 9e-10, 6e-11, 9e-13, 3e8, 1e15
        Xs, Ys = X * 2.0**-k, Y * 2.0**-k
        (out,) = run(Xs, Ys)
        (ref,) = SPECS["ols_hc0"].reference(Xs, Ys)
        assert _err(out, ref) <= 1e-6, f"(X, Y) * 2^{-k}"
        assert torch.equal(out, base), f"(X, Y) * 2^{-k}"
    # natural units, where every value rounds when scaled; and 2^-50, where the smallest x²
    # of a sample start to flush (so the last bits may change, not the error)
    for c in (1e-9, 1e-10, 1e-12, 1e-15, 2.0**-50):
        Xc, Yc = X * c, Y * c
        (out,) = run(Xc, Yc)
        (ref,) = SPECS["ols_hc0"].reference(Xc, Yc)
        assert _err(out, ref) <= 1e-6, f"(X, Y) * {c:g}"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", [(2, 2000000), (1, 77777), (100, 16384)])
def test_overlapping_calls_on_two_streams(n, m, variant):
    """Two calls of one ``run`` released at the same instant on two streams (both wait on an
    event recorded after a sleep) overlap on the GPU. Each must equal its single-stream
    result, and no split counter may be left non-zero for later calls: the workspaces (the
    segment states and counters of split rows, the unfused passes' per-row slope) are per
    stream. Shared ones mixed the two calls' segments and slopes."""
    run = ols_hc0.build({"n": n, "m": m}, variant=variant)
    XA, YA = _gate_inputs(n, m, 6)
    x, e = _data(n, m, seed=7)
    XB, YB = 3.0 * x, e - 2.0  # rows whose results differ from A's
    a0 = run(XA, YA)[0].clone()
    b0 = run(XB, YB)[0].clone()
    torch.cuda.synchronize()  # the side streams wait only on the event, not on these
    for _ in range(6):
        (a,), (b,) = _released_together(lambda: run(XA, YA), lambda: run(XB, YB))
        assert torch.equal(a, a0) and torch.equal(b, b0)
    assert not any(C.any() for C in _counters(run))
    assert torch.equal(run(XA, YA)[0], a0)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", [(2, 2000000), (1, 77777), (100, 16384)])
def test_graphs_replayed_on_two_streams(n, m, variant):
    """Two CUDA graphs of one ``run``, captured on one stream, replayed at the same instant on
    two streams. A graph's kernels keep the buffers they were captured with, but run on the
    stream it is replayed on: when a captured call took its capture stream's workspaces,
    the two graphs shared them, mixed their segments and per-row slopes and left split
    counters non-zero, and an eager call on the capture stream found counters whose zeroing
    had only been captured. A captured call now has workspaces of its own, from the graph's
    memory pool. Every replay, before, during and after the overlapping ones, and every
    eager call must equal the eager results."""
    run = ols_hc0.build({"n": n, "m": m}, variant=variant)
    XA, YA = _gate_inputs(n, m, 6)
    x, e = _data(n, m, seed=7)
    XB, YB = 3.0 * x, e - 2.0  # rows whose results differ from A's
    a0 = run(XA, YA)[0].clone()
    b0 = run(XB, YB)[0].clone()
    cap = torch.cuda.Stream()
    ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga, stream=cap):
        (oa,) = run(XA, YA)
    with torch.cuda.graph(gb, stream=cap):
        (ob,) = run(XB, YB)
    with torch.cuda.stream(cap):  # an eager call on the capture stream, before any replay
        early = run(XA, YA)[0]
    torch.cuda.synchronize()
    assert torch.equal(early, a0)
    for _ in range(6):
        _released_together(ga.replay, gb.replay)
        assert torch.equal(oa, a0) and torch.equal(ob, b0)
    assert not any(C.any() for C in _counters(run))
    gb.replay()
    ga.replay()
    torch.cuda.synchronize()
    assert torch.equal(oa, a0) and torch.equal(ob, b0)
    assert torch.equal(run(XA, YA)[0], a0) and torch.equal(run(XB, YB)[0], b0)
    del ga, gb
    torch.cuda.synchronize()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, m", [(1, 77777), (3, 1000004)])
def test_split_counters_reset_between_calls(n, m, variant):
    """The last CTA of a split row resets the row's counter: after every call every counter
    is 0 again, and the next call, on new inputs, is right."""
    run = ols_hc0.build({"n": n, "m": m}, variant=variant)
    for seed in range(4, 8):
        X, Y = _gate_inputs(n, m, seed)
        out, ref, _ = _run(X, Y, variant)
        assert _err(out, ref) <= 3e-7
        torch.cuda.synchronize()
        assert not any(C.any() for C in _counters(run))
