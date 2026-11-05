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
"""The cross_entropy kernels beyond the gate: every tail, the edge labels, wide logits.

The logits are float32 sums of exact bf16 products, so the loss must agree
with the float64 reference to float32 accuracy, far inside the gate's bound,
for the one-pass kernel and for the three-pass ablation alike. The cases hit a
partial last row block, a hidden size that is not a multiple of the ``BK``
slice (TMA zero-fills the rest), a partial last vocabulary tile, units that
run several tiles (the ring carried across tiles and units), several schedule
groups, and labels in the first and the last (masked-tail) vocabulary columns.
The schedule must cover every (row block, vocabulary tile) exactly once and
keep every CTA within one tile of the mean; the CTAs that share a progress
counter (the drift bound) must walk the same tiles, as many as the counter
expects, and every call must leave the counters at zero for the next.

Regressions: calls of one size on two CUDA streams at once (each stream has
its own workspace); rows whose loss is tiny next to the logits (the label's
logit far above the rest, or every logit of a row equal), where the loss must
be ``(m - t) + log l`` with ``m - t`` exact and the maximum counted exactly
once; and the ablation's label logit, which must be the tensor cores' logit
that its other passes saw.
"""

import math

import pytest
import torch

pytest.importorskip("tvm_tirx")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from bench.ops import SPECS
from kernels import cross_entropy

CASES = [
    # (n, d, v)
    (7, 64, 1001),  # one row block, one tile per unit, a partial last tile
    (300, 72, 777),  # partial last row block, hidden size 72 = BK + 8
    (130, 64, 129),  # a vocabulary of one tile (and one column past BN / 2)
    (2000, 256, 50000),  # 16 row blocks: two groups, units of several tiles
]
VARIANTS = ["fused", "unfused"]


def _inputs(n, d, v, scale, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    X = torch.randn(n, d, generator=gen, device="cuda").bfloat16()
    W = (torch.randn(v, d, generator=gen, device="cuda") * scale).bfloat16()
    labels = torch.randint(0, v, (n,), generator=gen, device="cuda")
    labels[0::3] = 0
    labels[1::3] = v - 1
    return X, W, labels


def _build(n, d, v, variant="fused"):
    return cross_entropy.build({"n": n, "d": d, "v": v}, variant=variant)


def _err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def _counters(run):
    return [c for ws in run.workspaces.values() for c in ws.counters]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, d, v", CASES)
@pytest.mark.parametrize("scale", [0.02, 0.5])
def test_cross_entropy_against_float64(n, d, v, scale, variant):
    X, W, labels = _inputs(n, d, v, scale)
    (out,) = _build(n, d, v, variant)(X, W, labels)
    (ref,) = SPECS["cross_entropy"].reference(X, W, labels)
    err = _err(out, ref)
    assert math.isfinite(err) and err <= 1e-5, err


def test_repeated_calls_agree():
    # the workspace is cached per size: a second call must not see the first's states
    X, W, labels = _inputs(300, 72, 777, 0.5)
    run = _build(300, 72, 777)
    (a,) = run(X, W, labels)
    a = a.clone()
    (b,) = run(X, W, labels)
    assert torch.equal(a, b)


@pytest.mark.parametrize("variant", VARIANTS)
def test_progress_counters_reset_after_each_call(variant):
    # 16 row blocks: two groups, so several counters with 12 and with 4 sharers
    X, W, labels = _inputs(2000, 256, 50000, 0.5)
    run = _build(2000, 256, 50000, variant)
    for _ in range(2):
        run(X, W, labels)
        torch.cuda.synchronize()
        assert all(int(c.abs().sum()) == 0 for c in _counters(run))


@pytest.mark.parametrize(
    "n, d, v",
    [(1, 64, 1001), (300, 72, 777), (2000, 256, 50000)]
    + [(8192, 4096, v) for v in (32000, 128256, 256000)],
)
def test_plan_covers_every_tile_once_and_balances(n, d, v):
    p = cross_entropy.plan(n, d, v)
    assert p["nt"] == math.ceil(v / cross_entropy.BN)
    seen = torch.zeros((p["rb"], p["nt"]), dtype=torch.int32)
    slots = set()
    loads = []
    sharing = {}  # counter -> the units that count on it
    for rounds in p["rounds"]:
        assert len(rounds) <= p["nr"]
        ctrs = [u[4] for u in rounds]
        assert ctrs == sorted(ctrs)  # every CTA walks the groups in order: no wait cycles
        for rb, vt0, vt1, slot, ctr, sharers in rounds:
            assert 0 <= slot < p["P"] and vt0 < vt1
            assert (rb, slot) not in slots  # one writer per row and slot
            slots.add((rb, slot))
            seen[rb, vt0:vt1] += 1
            sharing.setdefault(ctr, []).append((vt0, vt1, sharers))
        loads.append(sum(u[2] - u[1] for u in rounds))
    assert bool((seen == 1).all())
    assert max(loads) == p["tiles"] <= math.ceil(p["rb"] * p["nt"] / len(p["rounds"])) + 1
    assert sorted(sharing) == list(range(p["nctr"]))
    for units in sharing.values():
        # a counter's sharers walk the same tiles, and there are as many as each expects
        assert len(set(units)) == 1 and units[0][2] == len(units)
    # the ablation's label pass: every row block once, one pseudo-tile, no sharers
    pl = cross_entropy.label_plan(p)
    blocks = sorted(u[0] for rounds in pl["rounds"] for u in rounds)
    assert blocks == list(range(p["rb"]))
    assert all(u[1:] == (0, 1, 0, 0, 1) for rounds in pl["rounds"] for u in rounds)
    assert (pl["P"], pl["nctr"]) == (p["P"], p["nctr"])


# --- calls on several CUDA streams ---------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_stream_has_its_own_workspace(variant):
    X, W, labels = _inputs(300, 72, 777, 0.5)
    run = _build(300, 72, 777, variant)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for s in streams:
        with torch.cuda.stream(s):
            run(X, W, labels)
    torch.cuda.synchronize()
    spaces = [run.workspaces[s.cuda_stream] for s in streams]
    assert spaces[0] is not spaces[1]
    assert spaces[0].part.data_ptr() != spaces[1].part.data_ptr()
    ptrs = {c.data_ptr() for ws in spaces for c in ws.counters}
    assert len(ptrs) == 2 * len(spaces[0].counters)


@pytest.mark.parametrize("variant", VARIANTS)
def test_same_size_on_two_streams_at_once(variant):
    # Both streams wait on one event, so the two calls' kernels become ready together and
    # run interleaved; with one shared workspace, the second call's GEMM overwrote the
    # states before the first call's merge read them (every row of one call wrong).
    n, d, v = 2000, 256, 50000
    run = _build(n, d, v, variant)
    a, b = _inputs(n, d, v, 0.5, seed=1), _inputs(n, d, v, 0.5, seed=2)
    (ra,) = run(*a)
    ra = ra.clone()
    (rb,) = run(*b)
    rb = rb.clone()
    s0, s1, s2 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(4):
        torch.cuda.synchronize()
        go = torch.cuda.Event()
        with torch.cuda.stream(s0):
            torch.cuda._sleep(2_000_000)  # about 1 ms: both calls are queued behind it
            go.record()
        s1.wait_event(go)
        s2.wait_event(go)
        with torch.cuda.stream(s1):
            (oa,) = run(*a)
        with torch.cuda.stream(s2):
            (ob,) = run(*b)
        torch.cuda.synchronize()
        assert torch.equal(oa, ra) and torch.equal(ob, rb)
    assert all(int(c.abs().sum()) == 0 for c in _counters(run))


@pytest.mark.parametrize("variant", VARIANTS)
def test_cuda_graph_replays_with_its_own_workspace(variant):
    n, d, v = 300, 72, 777
    run = _build(n, d, v, variant)
    a, b = _inputs(n, d, v, 0.5, seed=3), _inputs(n, d, v, 0.5, seed=4)
    (ra,) = run(*a)
    ra = ra.clone()
    (rb,) = run(*b)
    rb = rb.clone()
    static = [t.clone() for t in a]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (out,) = run(*static)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ra)
    for t, src in zip(static, b):
        t.copy_(src)
    graph.replay()
    (eager,) = run(*a)  # an eager call between replays keeps the stream's own workspace
    torch.cuda.synchronize()
    assert torch.equal(out, rb) and torch.equal(eager, ra)


# --- rows whose loss is tiny next to the logits ---------------------------------------------


def _peaked(n, d, v, margin=30.0, seed=5):
    """Spec inputs with X[i] moved along W[label i]: the label's logit is ~margin above the
    rest, so the loss is ~1e-7 while the row maximum is ~30 (one float32 ulp: 1.9e-6)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    X, W, labels = SPECS["cross_entropy"].inputs({"n": n, "d": d, "v": v}, gen)
    Wl = W[labels].float()
    X = (X.float() + (margin / (Wl * Wl).sum(dim=1, keepdim=True)) * Wl).bfloat16()
    return X, W, labels


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "n, d, v", [(7, 64, 1001), (33, 128, 32003), (5, 256, 50000), (130, 64, 129)]
)
def test_peaked_rows(n, d, v, variant):
    # torch.compile returns 0 on these rows (err 1.0, the gate's bound 2.0). The kernels
    # returned m + log l - t with l = 2^-δ (1 + ε), δ the rounding of m * log2(e), and the
    # ablation took t from a float32 dot product 1e-4 above the tensor cores' logit: errors
    # of 4-45 (fused) and up to 270 (ablation), and negative losses.
    X, W, labels = _peaked(n, d, v)
    (ref,) = SPECS["cross_entropy"].reference(X, W, labels)
    assert float(ref.abs().max()) < 1e-5
    (out,) = _build(n, d, v, variant)(X, W, labels)
    assert float(out.min()) >= 0.0
    assert _err(out, ref) <= 0.25


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, d, v", [(7, 64, 1001), (300, 72, 777), (130, 64, 129)])
@pytest.mark.parametrize("w", [1.0, 1000.0])
def test_rows_of_equal_logits(n, d, v, w, variant):
    # W constant: every logit of a row is the same tensor-core value, so the loss is
    # exactly log v, whatever the size of the logits (here up to ~1e4)
    X, _, labels = _inputs(n, d, v, 0.02)
    W = torch.full((v, d), w, device="cuda", dtype=torch.bfloat16)
    (out,) = _build(n, d, v, variant)(X, W, labels)
    assert torch.equal(out, torch.full_like(out, math.log(v)))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n, d", [(5, 64), (300, 72)])
def test_one_class_loss_is_zero(n, d, variant):
    X, W, labels = _inputs(n, d, 1, 0.5)
    (out,) = _build(n, d, 1, variant)(X, W, labels)
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize(
    "n, d, v",
    [
        (300, 72, 777),  # 2 slices of k, the second zero-filled past d = 72
        (2000, 256, 50000),  # 4 slices: the gather hands slices over while later ones load
        (20000, 136, 1000),  # 157 row blocks: CTAs gather two units, the ring carries on
    ],
)
def test_ablation_label_logit_is_the_tensor_core_logit(n, d, v):
    # pass 3 recomputes X[i] · W[label i] with the instruction and k order of the GEMM
    # passes (W's rows gathered by cp.async into the layout a TMA tile has): bitwise the
    # logit the fused kernel picks out of its accumulators
    X, W, labels = _inputs(n, d, v, 0.5)
    fused, unfused = _build(n, d, v, "fused"), _build(n, d, v, "unfused")
    fused(X, W, labels)
    unfused(X, W, labels)
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream().cuda_stream
    t_fused = fused.workspaces[stream].part[..., 2].sum(dim=1)  # one slot holds t
    t_unfused = unfused.workspaces[stream].part[:, 0, 2]
    assert torch.equal(t_fused, t_unfused)
