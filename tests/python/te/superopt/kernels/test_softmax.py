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
"""The softmax kernel on the layouts the gate's sizes do not reach, and on the inputs
its random rows never hold.

The gate (``kernels.check``) takes the resident path on short aligned rows, the
register cache and the split on unaligned ones, and the long, cluster and rounds
paths only at the benchmark sizes, which leave no tails. Here every path is forced
on sizes that leave one: resident rows whose slots are partly empty (one CTA and
clusters of 2 and 4), rounds whose slices are ragged, stream part and all (few CTAs,
few slots, and repeated launches, whose barriers must carry over), unaligned caches,
streams and splits, and the three unfused passes. Each output is held to the gate's
rule against the float64 reference, with torch's own float32 program as the
yardstick.

Then the regressions: masked rows (``-inf`` and ``finfo(bfloat16).min``, as attention
masks write them) and rows far from 0; the rounds' barrier counters, which must not
grow from call to call; and calls of one size on two streams at once, which must not
share a workspace.
"""

import functools

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the kernels run on CUDA", allow_module_level=True)

from bench.ops import SPECS  # noqa: E402
from kernels import softmax  # noqa: E402

FLOOR = 4e-3  # one bfloat16 rounding of the largest output is up to 2e-3
MIN = torch.finfo(torch.bfloat16).min  # -3.39e38, the value HF-style attention masks add
I32_MIN = -(2**31)

CASES = [
    (7, 1000, "resident"),  # short, keep
    (3, 8 * 1500, "resident"),  # 128 threads x 12 slots, the last ones partly empty
    (2, 8 * 5000, "resident"),  # a cluster of 2, ragged
    (2, 8 * 16000, "resident"),  # a cluster of 4, ragged
    (3, 1, "cache"),
    (4, 5, "cache"),
    (3, 2055, "cache"),
    (3, 8199, "cache"),
    (5, 1001, "stream"),
    (3, 77777, "stream"),
    (2, 77776, "split"),
    (3, 100008, "split"),
    (2, 1001, "split"),
    (2, 70001, "split"),
    (7, 1000, "passes"),
    (5, 1001, "passes"),
    (3, 8 * 12345, "passes"),
    (1, 77777, "passes"),
]

# rounds forced onto small sizes: (n, m, threads, CTAs per SM, register slots, cap on the
# shared-memory slots); a cap makes the slices stream part of themselves
ROUNDS = [
    (3, 8 * 12345, 64, 1, 2, None),  # registers only, ragged slices
    (2, 8 * 70001, 128, 2, 4, None),
    (4, 8 * 70001, 64, 1, 0, None),  # shared memory only
    (1, 8 * 300007, 128, 1, 2, 3),  # streams part of every slice
    (3, 8 * 70001, 64, 1, 1, 2),
    (2, 8 * 70001, 128, 2, 0, 1),
]


@functools.cache
def _forced(n, m, path):
    return softmax.build_plan(n, m, softmax.plan(n, m, path))


def _rounds_plan(n, m, threads, per_sm, kr, cap):
    p = softmax._rounds_shape(m, threads, per_sm, kr, cap)
    assert p is not None and p.ks + p.kr > 0 and (cap is None) == (p.sp == 0)
    return p


@functools.cache
def _rounds(n, m, threads, per_sm, kr, cap):
    return softmax.build_plan(n, m, _rounds_plan(n, m, threads, per_sm, kr, cap))


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def _check(run, n, m, seeds=2):
    spec = SPECS["softmax"]
    for seed in range(seeds):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        inputs = spec.inputs({"n": n, "m": m}, gen)
        (ref,) = spec.reference(*inputs)
        (base,) = spec.unfused(*inputs)
        (out,) = run(*inputs)
        assert out.shape == base.shape and out.dtype == base.dtype
        assert torch.isfinite(out.float()).all()
        assert _err(out, ref) <= max(2.0 * _err(base, ref), FLOOR)


@pytest.mark.parametrize("n, m, path", CASES)
def test_forced_path(n, m, path):
    _check(_forced(n, m, path), n, m)


@pytest.mark.parametrize("n, m, threads, per_sm, kr, cap", ROUNDS)
def test_rounds(n, m, threads, per_sm, kr, cap):
    """Every seed is a new launch on the same counters, so the barriers carry over too."""
    _check(_rounds(n, m, threads, per_sm, kr, cap), n, m, seeds=3)


@pytest.mark.parametrize("n, m", [(4, 4096), (132, 77776), (2, 77776), (3, 8 * 70001)])
def test_constant_and_shifted_rows(n, m):
    """Equal scores give 1/m everywhere; scores far from 0 (unshifted, exp overflows) are fine."""
    for variant in ("fused", "unfused"):
        run = softmax.build(dict(n=n, m=m), variant=variant)
        x = torch.zeros(n, m, device="cuda", dtype=torch.bfloat16)
        (out,) = run(x)
        torch.testing.assert_close(
            out.float(), torch.full_like(out.float(), 1.0 / m), rtol=4e-3, atol=0
        )
        gen = torch.Generator(device="cuda").manual_seed(0)
        for offset in (200.0, -200.0):
            y = (2.0 * torch.randn(n, m, generator=gen, device="cuda") + offset).to(torch.bfloat16)
            (out,) = run(y)
            assert torch.isfinite(out.float()).all()
            assert _err(out, torch.softmax(y.double(), dim=1)) <= FLOOR


# ---------------------------------------------------------------------------
# masked rows and rows far from 0
# ---------------------------------------------------------------------------
KINDS = ("causal", "padding", "one-hot", "all-min", "big", "huge")


def _rows(kind, n, m, gen):
    """``n`` rows of ``kind``; every one keeps a finite element, so softmax is finite."""
    x = 2.0 * torch.randn(n, m, generator=gen, device="cuda")
    col = torch.arange(m, device="cuda")[None, :]
    sign = 1.0 - 2.0 * (torch.arange(n, device="cuda")[:, None] % 2)  # +, -, +, ...
    if kind == "causal":  # row i keeps its first 1 + i (m - 1) / (n - 1) scores, -inf after
        keep = 1 + torch.arange(n, device="cuda")[:, None] * (m - 1) // max(n - 1, 1)
        x = x.masked_fill(col >= keep, float("-inf"))
    elif kind == "padding":  # the last 80% masked with finfo.min
        x = x.masked_fill(col >= max(1, m // 5), MIN)
    elif kind == "one-hot":  # one score a row, -inf elsewhere
        hot = torch.randint(0, m, (n, 1), generator=gen, device="cuda")
        x = x.masked_fill(col != hot, float("-inf"))
    elif kind == "all-min":  # a fully masked row: uniform, as in torch
        x = torch.full_like(x, MIN)
    elif kind == "big":  # m log2(e) is rounded by far more than 1
        x = sign * 1e10 * (1.0 + 0.05 * x)
    else:  # "huge": -m log2(e) overflows float32
        assert kind == "huge"
        x = (sign * 2.5e38 * (1.0 + 0.04 * x)).clamp(-3.3e38, 3.3e38)
    return x.to(torch.bfloat16)


def _check_rows(run, n, m):
    spec = SPECS["softmax"]
    gen = torch.Generator(device="cuda").manual_seed(7)
    for kind in KINDS:
        x = _rows(kind, n, m, gen)
        # (the spec's reference adds log Σ to the maximum, which drops it past |m| ~ 1e30)
        ref = torch.softmax(x.double(), dim=1)
        (base,) = spec.unfused(x)
        (out,) = run(x)
        assert torch.isfinite(out.float()).all(), kind
        assert _err(out, ref) <= max(2.0 * _err(base, ref), FLOOR), kind


@pytest.mark.parametrize("n, m, path", CASES)
def test_masked_and_extreme_rows(n, m, path):
    """These rows were NaN on every aligned fused path while a thread's maximum started at
    -3e38 (above finfo.min) and ``exp(x - m)`` was ``ex2(fma(x, log2 e, -m log2 e))``:
    ``-m log2 e`` overflows to +inf past ``|m| = 2.36e38``, and ``ex2(-inf + inf)`` is NaN;
    rows past ``|m| ~ 3e9`` were NaN in both variants, the rounding of ``m log2 e`` being
    larger than the exponent itself."""
    _check_rows(_forced(n, m, path), n, m)


@pytest.mark.parametrize("n, m, threads, per_sm, kr, cap", ROUNDS)
def test_masked_and_extreme_rows_rounds(n, m, threads, per_sm, kr, cap):
    _check_rows(_rounds(n, m, threads, per_sm, kr, cap), n, m)


# ---------------------------------------------------------------------------
# the rounds' barrier counters
# ---------------------------------------------------------------------------
def _workspace(run, stream=None):
    s = stream or torch.cuda.current_stream()
    return run.workspaces[(s.device_index, s.cuda_stream)]


@pytest.mark.parametrize("n, m, threads, per_sm, kr, cap", [ROUNDS[0], ROUNDS[1], ROUNDS[3]])
def test_rounds_barrier_counters_do_not_grow(n, m, threads, per_sm, kr, cap):
    """A call's G arrivals at a row's barrier add 2^31 (CTA 0 adds 2^31 - (G - 1), the others
    1), so between calls the counter holds 0 or 2^31: every call starts from one of the two
    and leaves the other, and no number of calls can wrap it. (It used to grow by G a call
    in int32: the call whose arrivals crossed INT32_MAX hung the whole GPU, after 8.1M
    calls of 8x8388608.)"""
    run = softmax.build_plan(n, m, _rounds_plan(n, m, threads, per_sm, kr, cap))
    x = SPECS["softmax"].inputs({"n": n, "m": m}, torch.Generator("cuda").manual_seed(0))[0]
    (first,) = run(x)
    _, count = _workspace(run)
    for _ in range(4):
        before = count.clone()
        assert ((before == 0) | (before == I32_MIN)).all()
        (out,) = run(x)
        assert torch.equal(count, before ^ I32_MIN)
        assert torch.equal(out.view(torch.int16), first.view(torch.int16))


# ---------------------------------------------------------------------------
# two streams
# ---------------------------------------------------------------------------
STREAMED = [
    (4, 8193, "split"),
    (1, 77777, "passes"),
    (3, 8 * 12345, "passes"),
    (3, 8 * 12345, (64, 1, 2, None)),  # rounds
    (2, 8 * 70001, (128, 2, 0, 1)),  # rounds, streaming part of every slice
]


@pytest.mark.parametrize("n, m, path", STREAMED)
def test_streams_have_their_own_workspaces(n, m, path):
    """Calls of one size on two streams at once must not share a workspace. The split's
    and the passes' per-chunk states and the rounds' states and counters used to be one
    per size: a call overlapping another stream's took the other's states (errors up to
    1e13). Both streams wait on a sleep, are released ``d`` cycles apart over a sweep of
    ``d``, and each output must equal the one its input gives alone, bit for bit (stream
    2's input is shifted by +30, so the other's states cannot pass unnoticed)."""
    run = _rounds(n, m, *path) if isinstance(path, tuple) else _forced(n, m, path)
    gen = torch.Generator(device="cuda").manual_seed(5)
    xa = (2.0 * torch.randn(n, m, generator=gen, device="cuda")).to(torch.bfloat16)
    xb = (2.0 * torch.randn(n, m, generator=gen, device="cuda") + 30.0).to(torch.bfloat16)
    alone_a, alone_b = run(xa)[0].view(torch.int16), run(xb)[0].view(torch.int16)
    torch.cuda.synchronize()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    base = 1_000_000  # cycles: both calls are queued before either stream starts
    for d in range(-48_000, 8_001, 400):
        with torch.cuda.stream(s1):
            torch.cuda._sleep(base + max(0, -d))
        with torch.cuda.stream(s2):
            torch.cuda._sleep(base + max(0, d))
        with torch.cuda.stream(s1):
            (oa,) = run(xa)
        with torch.cuda.stream(s2):
            (ob,) = run(xb)
        torch.cuda.synchronize()
        assert torch.equal(oa.view(torch.int16), alone_a), d
        assert torch.equal(ob.view(torch.int16), alone_b), d
    ws1, ws2 = _workspace(run, s1), _workspace(run, s2)
    ptrs = lambda ws: {t.data_ptr() for t in (ws if isinstance(ws, tuple) else (ws,))}  # noqa: E731
    assert not ptrs(ws1) & ptrs(ws2)


def test_plans_of_the_benchmark_sizes():
    """Short rows keep their exponentials in registers; long rows put ~256 KB of rows on
    every SM (a cluster past 32K); few huge rows take the rounds -- 8M on two CTAs of 512
    an SM, the whole slice on chip; the 64M row on four CTAs of 256, streaming part of
    every slice; the unfused passes fill the GPU."""
    for n, m in [(32768, 4096), (16384, 8192)]:
        p = softmax.plan(n, m)
        assert p.path == "resident" and p.keep and p.ks == 0
    for n, m, cluster in [(4096, 32768, 1), (2048, 131072, 4)]:
        p = softmax.plan(n, m)
        assert p.path == "resident" and p.cluster == cluster
        assert (p.threads, p.kr + p.ks, p.minb) == (256, 16, 4)
    p = softmax.plan(8, 8388608)
    assert p.path == "rounds" and p.sp == 0 and (p.threads, p.ctas) == (512, 264)
    p = softmax.plan(1, 67108864)
    assert p.path == "rounds" and p.sp > 0 and p.ks > 0 and (p.threads, p.ctas) == (256, 528)
    for n, m in [(32768, 4096), (2048, 131072), (8, 8388608), (1, 67108864)]:
        p = softmax._passes_plan(n, m)
        assert n * p.chunks >= 132 and p.chunk % (p.threads * p.u) == 0
