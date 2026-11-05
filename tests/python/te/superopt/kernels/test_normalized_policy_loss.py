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
"""The normalized policy loss kernel beyond the gate: data far from zero, outliers, padded RL
rows, every layout, concurrent calls.

The gate's advantages sit at an offset of a quarter of their spread. Here
they also sit at 1000 standard deviations, where a float32 mean of the raw
data is off by 1e-4 of the spread; the fused kernel's sums (float pairs
around a reference K drawn from the row) must still be as accurate as float32
allows. The error is measured against sqrt(Σ(a - ā)² Σ(g - ḡ)²) / m, the size
of the result's terms, so that a row whose co-moment happens to cancel does
not inflate it.

A row whose first pair is an outlier must be as precise as any other: an
earlier kernel shifted every element of a row by that pair, and so cost such
rows 3 to 60 times torch.compile's error on every layout.

Large G at samples whose A is the row's mean (an RL batch's padded tokens, or
one outlier G) must cost nothing: a one-pass fold's parts are centred off the
row's mean, and with float32 sums their rounding times Σ g failed the gate's
rule by up to 3.6 times on padded rows and 19 times on an outlier G.

The cases reach every layout (a static team, a streaming warp, a CTA per row,
rows split over CTAs) and every way a split row's states are merged at its
end (warp 0 alone, a tree of the CTA for equal counts, any counts), with
counts equal across a CTA and not.
"""

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import normalized_policy_loss as policy

CASES = [
    # (n, m)
    (37, 256),  # static team of 32 lanes, two vectors a lane
    (37, 64),  # static team of 16 lanes, one vector a lane
    (7, 1000),  # streaming warp per row (250 vectors split over no team evenly)
    (5, 1001),  # streaming warp per row, scalar loads
    (300, 4096),  # CTA per row, a partial step only
    (300, 16384),  # CTA per row, two whole steps a thread: the warps' tree of equal counts
    (300, 4099),  # CTA per row, scalar, not split
    (3, 4099),  # scalar, one segment per row
    (3, 1 << 20),  # split into 32: warp 0 merges, equal counts
    (2, (1 << 20) + 4),  # split into 32, a partial last step: unequal counts
    (1, 1 << 22),  # split into 128: the CTA merges the segments, equal counts in a CTA
    (1, (1 << 22) + 4),  # split into 128, a partial last step: any counts throughout
    (1, 1 << 23),  # split into 256 equal segments: a tree of the CTA
]


def _inputs(n, m, offset):
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = 2.0 * torch.randn(n, m, generator=gen, device="cuda") + 0.5 + offset
    g = 0.3 * (a - offset) - 2.0 * torch.randn(n, m, generator=gen, device="cuda").abs()
    return a, g


def _scale(A, G):
    """sqrt(Σ(a - ā)² Σ(g - ḡ)²) / (m sd(a)): the size of the loss were a and g one vector."""
    a, g = A.double(), G.double()
    m = a.shape[1]
    sa = ((a - a.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    sg = ((g - g.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return (sa * sg).sqrt() / (m * (sa / (m - 1)).sqrt())


def _check(n, m, offset, variant):
    A, G = _inputs(n, m, offset)
    run = policy.build({"n": n, "m": m}, variant=variant)
    (ref,) = SPECS["normalized_policy_loss"].reference(A, G)
    for _ in range(2):  # the second call finds the split counters reset
        (out,) = run(A, G)
        err = ((out.double() - ref).abs() / _scale(A, G)).max().item()
        assert err <= 1e-6, err


@pytest.mark.parametrize("n, m", CASES)
@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_policy_loss_against_float64(n, m, offset):
    _check(n, m, offset, "fused")


@pytest.mark.parametrize("n, m", [(37, 256), (5, 1001), (3, 4099), (2, (1 << 20) + 4)])
def test_unfused_against_float64(n, m):
    # the unfused program's float32 mean of the raw data has no shift: no offset here
    _check(n, m, 0.0, "unfused")


# An outlier as every row's first pair, on the gate's inputs: each layout and each end-of-row
# merge, the gate's small sizes, and the bench sizes.
OUTLIER_SIZES = [
    *CASES[:4],  # static teams of 32 and 16 lanes, streaming warps (16-byte and scalar loads)
    (300, 4096),  # CTA per row, a partial step only
    (300, 16384),  # CTA per row, whole steps
    (3, 4099),  # CTA per row, scalar
    (1, 77777),  # split into 2, scalar
    (3, 1 << 20),  # split into 32: warp 0 merges
    (2, 1 << 22),  # split into 128: Chan merges of the segments
    (1, 1 << 23),  # split into 256: a division-free tree
    (65536, 256),
    (256, 262144),
    (1, 33554432),
]
OUTLIERS = [("A", 300.0), ("A", 1e4), ("A", 1e5), ("G", -1e4)]


@pytest.mark.parametrize("n, m", OUTLIER_SIZES)
@pytest.mark.parametrize("where, value", OUTLIERS)
def test_first_pair_outlier(n, m, where, value):
    """No element is trusted to be typical of its row: with an outlier as the first pair, the
    gate's metric max|out - ref| / max|ref| stays within the gate's floor 1e-6, which is the
    gate's bound here (torch.compile's own error on such inputs is 0.1 to 5e-7)."""
    spec = SPECS["normalized_policy_loss"]
    gen = torch.Generator(device="cuda").manual_seed(1)
    A, G = spec.inputs({"n": n, "m": m}, gen)
    (A if where == "A" else G)[:, 0] = value
    (ref,) = spec.reference(A, G)
    (out,) = policy.build({"n": n, "m": m})(A, G)
    err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    assert err <= 1e-6, err


def test_plan_covers_every_path():
    g = policy.plan(65536, 256)
    assert g.path == "static" and (g.tpr, g.per, g.vec) == (32, 2, 4)
    # short rows load evict-first (.cs), long rows plainly (see COP_SHORT / COP_LONG)
    assert g.load is policy.load_cs and policy.plan(256, 262144).load is policy.load_vec
    assert policy.plan(37, 64).tpr == 16
    g = policy.plan(7, 1000)  # 250 vectors: one masked warp step of 256, the whole row
    assert g.path == "team" and (g.u, g.cstep, g.tail) == (8, 1024, policy.tail_exact)
    assert g.stream is policy.stream_exact
    g = policy.plan(5, 1001)
    assert g.path == "team" and (g.vec, g.u, g.cstep) == (1, 32, 1024)
    g = policy.plan(300, 4096)  # 1024 vectors: a masked step only
    assert g.path == "cta" and g.S == 1 and g.end is policy.end_row and g.tail is policy.tail_exact
    g = policy.plan(300, 16384)  # two whole CTA steps: each warp holds 2 x 1024 pairs
    assert g.path == "cta" and g.S == 1 and (g.eq, g.cstep, g.cfold) == (True, 1024, 2048)
    assert (g.stream, g.reduce, g.tail) == (
        policy.stream_exact,
        policy.merge_warps_eq,
        policy.no_tail,
    )
    assert policy.plan(3, 4099).cstep == 1024  # unaligned: 32 values a thread, one at a time
    # the unfused sums: each thread its own (64 values), the CTA's threads merged at the end
    g = policy.plan(300, 16384, policy.MEAN)
    assert (g.stream, g.reduce, g.cfold) == (policy.stream_lanes, policy.reduce_eq, 64)
    assert policy.plan(5, 1001, policy.MEAN).u == policy.U_TEAM
    g = policy.plan(3, 1 << 20)
    assert (g.S, g.eq, g.final, g.reduce) == (32, True, policy.final_warp, policy.merge_warps_eq)
    g = policy.plan(2, (1 << 20) + 4)
    assert (g.S, g.eq, g.final, g.reduce) == (32, False, policy.final_warp, policy.merge_warps_any)
    assert g.tail is policy.tail_exact
    g = policy.plan(1, 1 << 22)
    assert (g.S, g.eq, g.final, g.reduce) == (128, True, policy.final_any, policy.merge_warps_eq)
    g = policy.plan(1, (1 << 22) + 4)
    assert (g.S, g.eq, g.final, g.reduce) == (128, False, policy.final_any, policy.merge_warps_any)
    g = policy.plan(1, 1 << 23)
    assert (g.S, g.eq, g.final) == (256, True, policy.final_eq)
    g = policy.plan(1, 33554432)  # the split bench size
    assert g.vec == 4 and g.eq and g.final is policy.final_eq and g.blocks == g.S
    assert g.load is policy.load_vec and g.tail is policy.no_tail
    g = policy.plan(256, 262144)  # the long bench size: a CTA per row, all resident
    assert g.S == 1 and g.blocks == 256 and g.eq


# A large G at a sample whose A is its row's mean adds next to nothing to the loss, but when a
# fold centred the sample on the mean of its own few pairs, it entered the co-moment as about
# |G| sd / sqrt(c) and cancelled only in the last merges: at a G of 1e4 the fused kernel was 2 to
# 2.7 times the gate's bound at 65536 x 256, and long rows were up to 2.4 times it at larger G.
# Every layout and end-of-row merge, the gate's small sizes and the bench sizes.
PEAK_SIZES = [
    (65536, 256),  # static team of 32 lanes (the short bench size): the team folds the row
    (33, 64),  # static team of 16 lanes
    (7, 1000),  # streaming warp: one masked step, the whole row
    (5, 1001),  # streaming warp, scalar loads
    (4097, 96),  # streaming warp: one masked step of 96 values
    (300, 16384),  # CTA per row, whole steps
    (3, 4099),  # CTA per row, scalar, a masked last step
    (1, 77777),  # split into 2, scalar
    (2, (1 << 20) + 4),  # split into 32, unequal counts
    (1, 1 << 22),  # split into 128, the CTA merges the segments
    (256, 262144),  # the long bench size
    (1, 33554432),  # the split bench size
]


def _peak_inputs(n, m, value, shift=0.0):
    """The gate's inputs (A shifted by ``shift``), and per row one random column where G = value
    and A = the mean of the row's other values."""
    spec = SPECS["normalized_policy_loss"]
    gen = torch.Generator(device="cuda").manual_seed(0)
    A, G = spec.inputs({"n": n, "m": m}, gen)
    A -= shift
    rows = torch.arange(n, device="cuda")
    j = torch.randint(0, m, (n,), generator=gen, device="cuda")
    a = A.double()
    A[rows, j] = ((a.sum(dim=1) - a[rows, j]) / (m - 1)).float()
    G[rows, j] = value
    return A, G


def _gate_err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("n, m", PEAK_SIZES)
@pytest.mark.parametrize("value", [-1e4, 1e4, -1e6, -1e8])
def test_g_peak_at_mean_a(n, m, value):
    """The gate's rule, with the unfused program in float32 eager (torch.compile's arithmetic,
    0.9 to 1.1 times its error here) as the yardstick: err <= max(2 err_float32, 1e-6)."""
    spec = SPECS["normalized_policy_loss"]
    A, G = _peak_inputs(n, m, value)
    (ref,) = spec.reference(A, G)
    err = _gate_err(policy.build({"n": n, "m": m})(A, G)[0], ref)
    err32 = _gate_err(spec.unfused(A, G)[0], ref)
    assert err <= max(2 * err32, 1e-6), (err, err32)


@pytest.mark.parametrize("shift", [0.0, 0.5])
@pytest.mark.parametrize("value", [-1e4, -1e8])
def test_g_peak_short_rows_any_mean(shift, value):
    """Short rows are folded whole by their team, so the bench's short rows stay within the
    gate's rule wherever the row's mean lies, at any G: with zero-mean A (shift 0.5), whose
    float32 mean is the most precise, too."""
    spec = SPECS["normalized_policy_loss"]
    A, G = _peak_inputs(65536, 256, value, shift)
    (ref,) = spec.reference(A, G)
    err = _gate_err(policy.build({"n": 65536, "m": 256})(A, G)[0], ref)
    err32 = _gate_err(spec.unfused(A, G)[0], ref)
    assert err <= max(2 * err32, 1e-6), (err, err32)


# Calls of one size on two CUDA streams at once: with one workspace per size, the split rows'
# segment states and counters were shared, so overlapping calls merged each other's segments
# (39 + 39 of 40 + 40 outputs wrong at 1 x 33554432) and left a counter non-zero, which made the
# next call on one stream wrong too; the unfused passes' per-row (mean, scale) were shared too.
@pytest.mark.parametrize(
    "n, m, variant",
    [
        (1, 33554432, "fused"),
        (1, 9011200, "fused"),
        (65536, 256, "unfused"),
        (256, 262144, "unfused"),
    ],
)
def test_two_streams(n, m, variant):
    spec = SPECS["normalized_policy_loss"]
    run = policy.build({"n": n, "m": m}, variant=variant)
    X, Y, Z = (spec.inputs({"n": n, "m": m}, torch.Generator(device="cuda").manual_seed(s))
               for s in (21, 22, 23))  # fmt: skip
    gold = [run(*t)[0].clone() for t in (X, Y, Z)]  # one at a time, on the default stream
    torch.cuda.synchronize()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    outs = []
    for _ in range(20):
        with torch.cuda.stream(s1):
            outs.append((0, run(*X)[0]))
        with torch.cuda.stream(s2):
            outs.append((1, run(*Y)[0]))
    torch.cuda.synchronize()
    wrong = sum(not torch.equal(o, gold[i]) for i, o in outs)
    assert wrong == 0, f"{wrong} of {len(outs)} outputs differ from the same call alone"
    (o,) = run(*Z)  # alone again, on the default stream
    assert torch.equal(o, gold[2])


# Zero-mean A (the gate's A - 0.5) with a large G at a mean-A sample, on long rows: whatever a
# one-pass fold centres a part of a row on is off from the row's mean, and with float32 sums
# that offset times the G cost 300 x 16384 3.5 to 5.7 times the gate's bound and 256 x 262144
# 1.7 to 19 times it (G = -1e6 to -1e8), where torch.compile's float32 mean of the zero-mean row
# is nearly exact. With exact sums the error is the output's rounding: the floor holds.
@pytest.mark.parametrize("n, m", [(300, 16384), (256, 262144), (2, (1 << 20) + 4)])
@pytest.mark.parametrize("value", [-1e6, -1e8])
def test_g_peak_zero_mean_long_rows(n, m, value):
    spec = SPECS["normalized_policy_loss"]
    A, G = _peak_inputs(n, m, value, shift=0.5)
    (ref,) = spec.reference(A, G)
    err = _gate_err(policy.build({"n": n, "m": m})(A, G)[0], ref)
    assert err <= 1e-6, err


def _padded(n, m, where, frac, fill, seed=0):
    """An RL batch's rows: the gate's advantages whitened over each row's valid tokens (zero
    mean, unit sd, in float32), and A = 0 (the row's mean), G = ``fill`` at the padded ones: the
    last ``frac`` of each row (``where`` = "end") or a random ``frac`` of its tokens."""
    spec = SPECS["normalized_policy_loss"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    A, G = spec.inputs({"n": n, "m": m}, gen)
    if where == "end":
        pad = torch.zeros(n, m, dtype=torch.bool, device="cuda")
        pad[:, m - max(1, int(m * frac)) :] = True
    else:
        pad = torch.rand(n, m, generator=gen, device="cuda") < frac
    valid = ~pad
    cnt = valid.sum(dim=1, keepdim=True).float()
    mu = torch.where(valid, A, 0.0).sum(dim=1, keepdim=True) / cnt
    var = (torch.where(valid, A - mu, 0.0) ** 2).sum(dim=1, keepdim=True) / (cnt - 1)
    A = torch.where(valid, (A - mu) / var.sqrt(), 0.0)
    G = torch.where(valid, G, fill)
    return A.contiguous(), G.contiguous()


# Padded rows put a large G at every padded token, whose A is the row's mean: with float32 sums
# the fused kernel failed the gate's rule there by up to 3.6 times (300 x 16384), 5.4 times
# (1 x 77777) and 8 times (1 x 33554432), from a fill of -1e2 to -1e3 on. Every layout and
# end-of-row merge, the gate's small sizes and the bench sizes, both variants. torch.compile's
# own error here is 1e-6 to 5e-3 of the result (its float32 mean's error times Σ G); with
# exact sums the kernels' error is the output's rounding, whatever the fill: the floor holds.
PAD_SIZES = [
    (65536, 256),  # static team of 32 lanes: the team folds the row
    (37, 64),  # static team of 16 lanes
    (7, 1000),  # streaming warp: one masked step
    (5, 1001),  # streaming warp, scalar loads
    (300, 16384),  # CTA per row, whole steps
    (3, 4099),  # CTA per row, scalar, a masked last step
    (1, 77777),  # split into 2, scalar
    (2, (1 << 20) + 4),  # split into 32, unequal counts
    (1, (1 << 22) + 4),  # split into 128, any counts
    (256, 262144),  # the long bench size
    (1, 33554432),  # the split bench size
]
PAD_CASES = [("end", 0.3, -1e3), ("random", 0.1, -1e4), ("end", 0.7, -1e6), ("random", 0.3, -1e2)]


@pytest.mark.parametrize("n, m", PAD_SIZES)
@pytest.mark.parametrize("where, frac, fill", PAD_CASES)
@pytest.mark.parametrize("variant", ["fused", "unfused"])
def test_padded_rows(n, m, where, frac, fill, variant):
    spec = SPECS["normalized_policy_loss"]
    A, G = _padded(n, m, where, frac, fill)
    (ref,) = spec.reference(A, G)
    err = _gate_err(policy.build({"n": n, "m": m}, variant=variant)(A, G)[0], ref)
    assert err <= 1e-6, err


# Every other row constant (3, 0.5, 1000, -2): its loss is 0 and its scale the 1e-8 alone, so any
# rounding left in its C is multiplied by 1e8 / m. Products centred exactly on each part's own
# mean leave nothing there; an uncentred Σ a g - K Σ g, equal in exact arithmetic, left ~2^-45 of
# K Σ g (1300 to 2900 times the gate's bound). Every layout and end-of-row merge.
CONST_SIZES = [
    (8, 256),  # static team of 32 lanes
    (6, 1001),  # streaming warp, scalar loads, masked steps
    (8, 4096),  # CTA per row, a masked step only
    (8, 16384),  # CTA per row, whole steps
    (8, 262144),  # CTA per row, the long bench size's rows
    (4, (1 << 20) + 4),  # split into 32, unequal counts
]


@pytest.mark.parametrize("n, m", CONST_SIZES)
@pytest.mark.parametrize("variant", ["fused", "unfused"])
def test_constant_rows(n, m, variant):
    spec = SPECS["normalized_policy_loss"]
    A, G = spec.inputs({"n": n, "m": m}, torch.Generator(device="cuda").manual_seed(5))
    vals = torch.tensor([3.0, 0.5, 1000.0, -2.0], device="cuda")
    rows = torch.arange(0, n, 2, device="cuda")
    A[rows] = vals[(rows // 2) % 4][:, None].expand(-1, m)
    (ref,) = spec.reference(A, G)
    err = _gate_err(policy.build({"n": n, "m": m}, variant=variant)(A, G)[0], ref)
    assert err <= 1e-6, err


# Two host threads calling one size on one CUDA stream: one unfused call is three launches that
# hand the rows' (mean, scale) on through a buffer, and with one cached per stream the threads'
# interleaved launches read each other's (260 of 400 outputs wrong in the verifier's run, up to
# 0.8 off). Every output must equal the same call made alone; the fused call (one launch) too.
@pytest.mark.parametrize("n, m", [(65536, 256), (7, 1000), (1, 1 << 23)])
@pytest.mark.parametrize("variant", ["unfused", "fused"])
def test_two_host_threads_one_stream(n, m, variant):
    import sys
    import threading

    spec = SPECS["normalized_policy_loss"]
    run = policy.build({"n": n, "m": m}, variant=variant)
    X, Y = (spec.inputs({"n": n, "m": m}, torch.Generator(device="cuda").manual_seed(s))
            for s in (31, 32))  # fmt: skip
    gold = {0: run(*X)[0].clone(), 1: run(*Y)[0].clone()}
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    outs = {0: [], 1: []}
    start = threading.Barrier(2)

    def worker(k, inp):
        torch.cuda.set_device(device)
        start.wait()
        for _ in range(40):
            outs[k].append(run(*inp)[0])

    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # interleave the threads' launches as often as possible
    try:
        threads = [threading.Thread(target=worker, args=(k, inp)) for k, inp in ((0, X), (1, Y))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(switch)
    torch.cuda.synchronize()
    wrong = sum(not torch.equal(o, gold[k]) for k in outs for o in outs[k])
    assert wrong == 0, f"{wrong} of 80 outputs differ from the same call alone"


# 0/1 advantages (a verifier's rewards), G independent of A: the unfused loss pass formed each
# term as g ((a - mean) (1 / scale)), whose factor takes one value per distinct a, so its
# rounding was the same in every term and grew with the row (4 to 88 times the gate's bound at
# the bench sizes); and its float32 mean and sums missed the rule by 1.4 to 2.5 times. Both
# variants, the gate's rule with the unfused program in float32 eager as the yardstick.
BINARY_CASES = [
    (300, 16384, 0.5, 1),  # unfused was 1.93e-6 (compile 1.9e-7)
    (3, 4099, 0.9, 2),  # unfused was 3.8e-6: the float32 mean (compile 7.5e-7)
    (256, 262144, 0.5, 2),  # unfused was 7.3e-6 (compile 3.2e-7)
    (1, 33554432, 0.1, 2),  # unfused was 7.4e-5 (compile 5.2e-7)
    (1, 33554432, 0.1, 3),  # unfused was 8.8e-5, then 1.4e-6 from its float32 loss sum
]


@pytest.mark.parametrize("n, m, p, seed", BINARY_CASES)
@pytest.mark.parametrize("variant", ["unfused", "fused"])
def test_binary_advantages(n, m, p, seed, variant):
    spec = SPECS["normalized_policy_loss"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    A = (torch.rand(n, m, generator=gen, device="cuda") < p).float()
    G = -2.0 * torch.randn(n, m, generator=gen, device="cuda").abs()
    (ref,) = spec.reference(A, G)
    err = _gate_err(policy.build({"n": n, "m": m}, variant=variant)(A, G)[0], ref)
    err32 = _gate_err(spec.unfused(A, G)[0], ref)
    assert err <= max(2 * err32, 1e-6), (err, err32)
