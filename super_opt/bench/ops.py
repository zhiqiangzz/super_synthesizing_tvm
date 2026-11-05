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
"""The showcase operators on the GPU: inputs, the unfused program in torch, a reference.

Each :class:`OpSpec` mirrors one operator of ``chain_ops``:

``inputs(cfg, gen)``
    the tensors, on the GPU, in the precision the operator is used at. The LLM
    operators take bfloat16 and accumulate in float32; the scientific ones are
    float32 throughout.
``unfused(*inputs)``
    the operator's ``unfused()`` TE program written in torch, one torch call per
    link of the chain, every intermediate materialised. Inputs are widened to
    float32 once at the start, as the TE program computes in float32; the
    outputs are narrowed to the operator's output precision at the end.
``reference(*inputs)``
    the same function in float64, over row (or column) chunks so the largest
    sizes still fit on the card. Products of matrices are taken in float32
    without TF32: their error (~1e-6 relative) is far below what is checked.
``traffic(cfg)``, ``tc_flops(cfg)``, ``fp32_flops(cfg)``, ``exps(cfg)``
    the least work any implementation must do, for the roofline: HBM traffic
    (every input read once, every output written once), bf16 tensor-core FLOPs,
    float32 FMA FLOPs, and exponentials.

``test_bench_ops.py`` holds ``unfused`` and ``reference`` to the float64 numpy
references of ``chain_ops`` on small inputs, so the two cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import torch

LN_EPS = 1.0e-5  # chain_ops.moments.LN_EPS
SINKHORN_EPS = 0.5  # chain_ops.sinkhorn.EPS
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
CHUNK_BYTES = 1 << 31  # float64 working set of one reference chunk


@dataclasses.dataclass(frozen=True)
class OpSpec:
    name: str
    dtype: torch.dtype  # storage precision of the operator's main tensors
    inputs: Callable[[dict, torch.Generator], tuple]
    unfused: Callable[..., tuple]
    reference: Callable[..., tuple]
    traffic: Callable[[dict], int]
    tol: float  # largest accepted max|out - ref| / max|ref|, per output
    tc_flops: Callable[[dict], int] = lambda cfg: 0  # bf16 tensor-core FLOPs
    fp32_flops: Callable[[dict], int] = lambda cfg: 0  # float32 FMA FLOPs of a product
    exps: Callable[[dict], int] = lambda cfg: 0  # exponentials no single pass can avoid
    note: str = ""


SPECS: dict[str, OpSpec] = {}


def _register(spec: OpSpec) -> OpSpec:
    SPECS[spec.name] = spec
    return spec


def _randn(gen, *size, dtype=F32, scale=1.0, offset=0.0):
    x = torch.randn(*size, generator=gen, device="cuda", dtype=F32)
    return (x * scale + offset).to(dtype)


def _rows(fn, *tensors, width=None):
    """``fn`` over row chunks of equal-height tensors, outputs concatenated along rows.

    ``width`` is the float64 row length ``fn`` works on, when that is not the
    inputs' own (a row of logits is as long as the vocabulary, not the hidden size).
    """
    n = tensors[0].shape[0]
    width = width or len(tensors) * math.prod(tensors[0].shape[1:])
    step = max(1, CHUNK_BYTES // (8 * 4 * width))  # ~4 live float64 temporaries per row
    parts = [fn(*(t[i : i + step] for t in tensors)) for i in range(0, n, step)]
    return tuple(torch.cat(chunk) for chunk in zip(*parts))


def _lse(s, dim):
    """log Σ exp along ``dim``, shifted by the maximum."""
    mx = s.amax(dim=dim, keepdim=True)
    return (mx + (s - mx).exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)


def _rowwise_bytes(dtype_in, dtype_out, n_in=1, out_full=True):
    """Traffic of an (n, m) row operator: ``n_in`` inputs read, a full or per-row output."""

    def traffic(cfg):
        n, m = cfg["n"], cfg["m"]
        out = n * m * dtype_out.itemsize if out_full else n * dtype_out.itemsize
        return n_in * n * m * dtype_in.itemsize + out

    return traffic


# ---------------------------------------------------------------------------
# softmax: p = exp(x - max x) / Σ exp(x - max x)
# ---------------------------------------------------------------------------
def _softmax_unfused(X):
    x = X.float()
    mx = x.amax(dim=1, keepdim=True)
    e = (x - mx).exp()
    den = e.sum(dim=1, keepdim=True)
    return ((e / den).to(X.dtype),)


def _softmax_reference(X):
    def chunk(x):
        x = x.double()
        return ((x - _lse(x, 1)[:, None]).exp(),)

    return _rows(chunk, X)


_register(
    OpSpec(
        "softmax",
        BF16,
        lambda cfg, gen: (_randn(gen, cfg["n"], cfg["m"], dtype=BF16, scale=2.0),),
        _softmax_unfused,
        _softmax_reference,
        _rowwise_bytes(BF16, BF16),
        tol=1e-2,
        exps=lambda cfg: cfg["n"] * cfg["m"],
    )
)


# ---------------------------------------------------------------------------
# cross_entropy: lse(X Wᵀ) - logit of the label
# ---------------------------------------------------------------------------
def _cross_entropy_inputs(cfg, gen):
    n, d, v = cfg["n"], cfg["d"], cfg["v"]
    labels = torch.randint(0, v, (n,), generator=gen, device="cuda")
    return (
        _randn(gen, n, d, dtype=BF16),
        _randn(gen, v, d, dtype=BF16, scale=0.02),
        labels,
    )


def _cross_entropy_unfused(X, W, labels):
    logits = (X @ W.T).float()  # the logits in the model's precision, then widened
    mx = logits.amax(dim=1, keepdim=True)
    e = (logits - mx).exp()
    den = e.sum(dim=1, keepdim=True)
    t = logits.gather(1, labels[:, None])
    return ((mx + den.log() - t)[:, 0],)


def _cross_entropy_reference(X, W, labels):
    w = W.float()

    def chunk(x, lab):
        logits = (x.float() @ w.T).double()
        return (_lse(logits, 1) - logits.gather(1, lab[:, None])[:, 0],)

    return _rows(chunk, X, labels, width=W.shape[0])


_register(
    OpSpec(
        "cross_entropy",
        BF16,
        _cross_entropy_inputs,
        _cross_entropy_unfused,
        _cross_entropy_reference,
        lambda cfg: (cfg["n"] + cfg["v"]) * cfg["d"] * 2 + cfg["n"] * (8 + 4),
        tol=1e-2,
        tc_flops=lambda cfg: 2 * cfg["n"] * cfg["d"] * cfg["v"],
        exps=lambda cfg: cfg["n"] * cfg["v"],
        note=(
            "hard labels: Σ_v y_v logit_v of chain_ops with a one-hot y is the logit "
            "of the label, which is what Liger and CCE take"
        ),
    )
)


# ---------------------------------------------------------------------------
# softmax_entropy: -Σ p log p with p = softmax(x)
# ---------------------------------------------------------------------------
def _entropy_unfused(X):
    x = X.float()
    mx = x.amax(dim=1, keepdim=True)
    e = (x - mx).exp()
    den = e.sum(dim=1, keepdim=True)
    logp = x - mx - den.log()
    return (-((e / den) * logp).sum(dim=1),)


def _entropy_reference(X):
    def chunk(x):
        x = x.double()
        logp = x - _lse(x, 1)[:, None]
        return (-(logp.exp() * logp).sum(dim=1),)

    return _rows(chunk, X)


_register(
    OpSpec(
        "softmax_entropy",
        BF16,
        lambda cfg, gen: (_randn(gen, cfg["n"], cfg["m"], dtype=BF16, scale=2.0),),
        _entropy_unfused,
        _entropy_reference,
        _rowwise_bytes(BF16, F32, out_full=False),
        tol=1e-3,
        exps=lambda cfg: cfg["n"] * cfg["m"],
    )
)


# ---------------------------------------------------------------------------
# attention: softmax(Q Kᵀ / sqrt(d)) V, not causal
# ---------------------------------------------------------------------------
def _attention_inputs(cfg, gen):
    shape = (cfg["b"], cfg["h"], cfg["s"], cfg["d"])
    return tuple(_randn(gen, *shape, dtype=BF16) for _ in range(3))


def _attention_unfused(Q, K, V):
    # the scores in float32, as the TE program keeps them; stored in bfloat16 they
    # cost 1.3e-2 normwise against the reference, past the tolerance
    b, h, s_q, d = Q.shape
    k_t = K.flatten(0, 1).transpose(1, 2)
    s = torch.bmm(Q.flatten(0, 1), k_t, out_dtype=F32).view(b, h, s_q, -1) / math.sqrt(d)
    mx = s.amax(dim=-1, keepdim=True)
    e = (s - mx).exp()
    den = e.sum(dim=-1, keepdim=True)
    p = e / den
    return (p.to(V.dtype) @ V,)


def _attention_reference(Q, K, V):
    out = torch.empty(Q.shape, device=Q.device, dtype=F64)
    scale = 1.0 / math.sqrt(Q.shape[-1])
    for b in range(Q.shape[0]):
        for h in range(Q.shape[1]):
            s = (Q[b, h].float() @ K[b, h].float().T).double() * scale
            p = (s - _lse(s, 1)[:, None]).exp()
            out[b, h] = p @ V[b, h].double()
    return (out,)


_register(
    OpSpec(
        "attention",
        BF16,
        _attention_inputs,
        _attention_unfused,
        _attention_reference,
        lambda cfg: 4 * cfg["b"] * cfg["h"] * cfg["s"] * cfg["d"] * 2,
        tol=1e-2,
        tc_flops=lambda cfg: 4 * cfg["b"] * cfg["h"] * cfg["s"] ** 2 * cfg["d"],
        exps=lambda cfg: cfg["b"] * cfg["h"] * cfg["s"] ** 2,
    )
)


# ---------------------------------------------------------------------------
# layernorm: (x - mean) / sqrt(var + eps), no affine
# ---------------------------------------------------------------------------
def _layernorm_unfused(X):
    x = X.float()
    m = x.shape[1]
    mean = x.sum(dim=1, keepdim=True) / m
    ss = ((x - mean) * (x - mean)).sum(dim=1, keepdim=True)
    return (((x - mean) / (ss / m + LN_EPS).sqrt()).to(X.dtype),)


def _layernorm_reference(X):
    def chunk(x):
        x = x.double()
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        return ((x - mean) / (var + LN_EPS).sqrt(),)

    return _rows(chunk, X)


_register(
    OpSpec(
        "layernorm",
        BF16,
        lambda cfg, gen: (_randn(gen, cfg["n"], cfg["m"], dtype=BF16),),
        _layernorm_unfused,
        _layernorm_reference,
        _rowwise_bytes(BF16, BF16),
        tol=1e-2,
    )
)


# ---------------------------------------------------------------------------
# layernorm_backward: (g - mean(g) - x̂ mean(g x̂)) / sigma, statistics recomputed from x
# ---------------------------------------------------------------------------
def _layernorm_backward_unfused(X, G):
    x, g = X.float(), G.float()
    m = x.shape[1]
    mean = x.sum(dim=1, keepdim=True) / m
    ss = ((x - mean) * (x - mean)).sum(dim=1, keepdim=True)
    sigma = (ss / m + LN_EPS).sqrt()
    xh = (x - mean) / sigma
    sg = g.sum(dim=1, keepdim=True)
    sgx = (g * xh).sum(dim=1, keepdim=True)
    return (((g - sg / m - xh * sgx / m) / sigma).to(X.dtype),)


def _layernorm_backward_reference(X, G):
    def chunk(x, g):
        x, g = x.double(), g.double()
        mean = x.mean(dim=1, keepdim=True)
        sigma = (((x - mean) ** 2).mean(dim=1, keepdim=True) + LN_EPS).sqrt()
        xh = (x - mean) / sigma
        a = g.mean(dim=1, keepdim=True)
        b = (g * xh).mean(dim=1, keepdim=True)
        return ((g - a - xh * b) / sigma,)

    return _rows(chunk, X, G)


_register(
    OpSpec(
        "layernorm_backward",
        BF16,
        lambda cfg, gen: (
            _randn(gen, cfg["n"], cfg["m"], dtype=BF16),
            _randn(gen, cfg["n"], cfg["m"], dtype=BF16),
        ),
        _layernorm_backward_unfused,
        _layernorm_backward_reference,
        _rowwise_bytes(BF16, BF16, n_in=2),
        tol=1e-2,
        note="statistics recomputed from x, as in chain_ops; torch's backward reads saved ones",
    )
)


# ---------------------------------------------------------------------------
# contrastive: lse(S[i, :]) - S[i, i] and lse(S[:, i]) - S[i, i]
# ---------------------------------------------------------------------------
def _contrastive_unfused(S):
    s = S.float()
    diag = s.diagonal()
    mx_r = s.amax(dim=1)
    den_r = (s - mx_r[:, None]).exp().sum(dim=1)
    mx_c = s.amax(dim=0)
    den_c = (s - mx_c[None, :]).exp().sum(dim=0)
    return (mx_r + den_r.log() - diag, mx_c + den_c.log() - diag)


def _contrastive_reference(S):
    diag = S.diagonal().double()
    rows = _rows(lambda s: (_lse(s.double(), 1),), S)[0]
    cols = _rows(lambda s: (_lse(s.double(), 1),), S.T)[0]
    return (rows - diag, cols - diag)


_register(
    OpSpec(
        "contrastive",
        BF16,
        lambda cfg, gen: (_randn(gen, cfg["n"], cfg["n"], dtype=BF16, scale=8.0),),
        _contrastive_unfused,
        _contrastive_reference,
        lambda cfg: cfg["n"] ** 2 * 2 + 2 * cfg["n"] * 4,
        tol=1e-2,
        exps=lambda cfg: 2 * cfg["n"] ** 2,  # shifted by the row's maximum and by the column's
        note="the similarity matrix S is the input (open_clip computes it before the loss)",
    )
)


# ---------------------------------------------------------------------------
# sinkhorn_iteration: the f-update then the g-update, C = |x_i - y_j|² never stored
# ---------------------------------------------------------------------------
def _sinkhorn_inputs(cfg, gen):
    n, m, d = cfg["n"], cfg["m"], cfg["d"]
    x = _randn(gen, n, d)
    y = _randn(gen, m, d, offset=0.5)
    a = torch.full((n,), 1.0 / n, device="cuda")
    b = torch.full((m,), 1.0 / m, device="cuda")
    g = _randn(gen, m, scale=0.1)
    return x, y, a, b, g


def _sinkhorn_unfused(x, y, a, b, g):
    # torch.cdist's |x|² + |y|² - 2 x·y in float32 (TF32 stays off), clamped at 0
    c = ((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * (x @ y.T)).clamp_min(0)
    s = (g[None, :] - c) / SINKHORN_EPS + b.log()[None, :]
    mx = s.amax(dim=1)
    f = -SINKHORN_EPS * (mx + (s - mx[:, None]).exp().sum(dim=1).log())
    s2 = (f[:, None] - c) / SINKHORN_EPS + a.log()[:, None]
    mx2 = s2.amax(dim=0)
    g_new = -SINKHORN_EPS * (mx2 + (s2 - mx2[None, :]).exp().sum(dim=0).log())
    return f, g_new


def _sinkhorn_reference(x, y, a, b, g):
    x64, y64 = x.double(), y.double()
    la, lb = a.double().log(), b.double().log()

    def half(p, q, pot, logw):
        def chunk(pc):
            c = torch.cdist(pc, q, compute_mode="donot_use_mm_for_euclid_dist") ** 2
            return (-SINKHORN_EPS * _lse((pot[None, :] - c) / SINKHORN_EPS + logw[None, :], 1),)

        return _rows(chunk, p, width=q.shape[0])[0]

    f = half(x64, y64, g.double(), lb)
    return f, half(y64, x64, f, la)


_register(
    OpSpec(
        "sinkhorn_iteration",
        F32,
        _sinkhorn_inputs,
        _sinkhorn_unfused,
        _sinkhorn_reference,
        lambda cfg: ((cfg["n"] + cfg["m"]) * (cfg["d"] + 2) + cfg["m"] + cfg["n"]) * 4,
        tol=1e-4,
        fp32_flops=lambda cfg: 2 * 2 * cfg["n"] * cfg["m"] * cfg["d"],
        exps=lambda cfg: 2 * cfg["n"] * cfg["m"],
        note="bound by exp throughput at small d: two exponentials per (i, j)",
    )
)


# ---------------------------------------------------------------------------
# The scientific row reductions, float32 throughout.
# ---------------------------------------------------------------------------
def _pearson_unfused(X, Y):
    m = X.shape[1]
    mx = X.sum(dim=1, keepdim=True) / m
    my = Y.sum(dim=1, keepdim=True) / m
    sxy = ((X - mx) * (Y - my)).sum(dim=1)
    sxx = ((X - mx) * (X - mx)).sum(dim=1)
    syy = ((Y - my) * (Y - my)).sum(dim=1)
    return (sxy / (sxx * syy).sqrt(),)


def _pearson_reference(X, Y):
    def chunk(x, y):
        x, y = x.double(), y.double()
        dx, dy = x - x.mean(dim=1, keepdim=True), y - y.mean(dim=1, keepdim=True)
        return ((dx * dy).sum(1) / ((dx * dx).sum(1) * (dy * dy).sum(1)).sqrt(),)

    return _rows(chunk, X, Y)


def _pearson_inputs(cfg, gen):
    x = _randn(gen, cfg["n"], cfg["m"])
    return x, 0.6 * x + 0.8 * _randn(gen, cfg["n"], cfg["m"])


_register(
    OpSpec(
        "pearson",
        F32,
        _pearson_inputs,
        _pearson_unfused,
        _pearson_reference,
        _rowwise_bytes(F32, F32, n_in=2, out_full=False),
        tol=1e-4,
    )
)


def _kurtosis_unfused(X):
    m = X.shape[1]
    mean = X.sum(dim=1, keepdim=True) / m
    ss = ((X - mean) * (X - mean)).sum(dim=1, keepdim=True)
    m2 = ss / m
    s4 = ((X - mean) / m2.sqrt()).pow(4).sum(dim=1)
    return (s4 / m,)


def _kurtosis_reference(X):
    def chunk(x):
        d = x.double() - x.double().mean(dim=1, keepdim=True)
        return ((d**4).mean(dim=1) / (d**2).mean(dim=1) ** 2,)

    return _rows(chunk, X)


def _ess_unfused(LW):
    mx = LW.amax(dim=1, keepdim=True)
    w = (LW - mx).exp()
    sw = w.sum(dim=1)
    sw2 = (w * w).sum(dim=1)
    return (sw * sw / sw2,)


def _ess_reference(LW):
    def chunk(lw):
        lw = lw.double()
        w = (lw - lw.amax(dim=1, keepdim=True)).exp()
        return (w.sum(dim=1) ** 2 / (w * w).sum(dim=1),)

    return _rows(chunk, LW)


def _papr_unfused(X):
    m = X.shape[1]
    mean = X.sum(dim=1, keepdim=True) / m
    ss = ((X - mean) * (X - mean)).sum(dim=1)
    peak = ((X - mean) * (X - mean)).amax(dim=1)
    return (peak / (ss / m),)


def _papr_reference(X):
    def chunk(x):
        d = (x.double() - x.double().mean(dim=1, keepdim=True)) ** 2
        return (d.amax(dim=1) / d.mean(dim=1),)

    return _rows(chunk, X)


def _householder_inputs(cfg, gen):
    e = torch.zeros(cfg["n"], cfg["m"], device="cuda")
    e[:, 0] = 1.0  # the first unit vector: the reflector of a QR step's leading column
    return _randn(gen, cfg["n"], cfg["m"]), e


def _householder_unfused(X, E):
    nrm = (X * X).sum(dim=1, keepdim=True).sqrt()
    return (((X + nrm * E) * (X + nrm * E)).sum(dim=1),)


def _householder_reference(X, E):
    def chunk(x, e):
        x, e = x.double(), e.double()
        nrm = (x * x).sum(dim=1, keepdim=True).sqrt()
        return (((x + nrm * e) ** 2).sum(dim=1),)

    return _rows(chunk, X, E)


def _heat_capacity_unfused(E):
    lo = E.amin(dim=1, keepdim=True)
    w = (-(E - lo)).exp()
    z = w.sum(dim=1, keepdim=True)
    e1 = (E * w).sum(dim=1, keepdim=True) / z
    cs = ((E - e1) * (E - e1) * w).sum(dim=1, keepdim=True)
    return ((cs / z)[:, 0],)


def _heat_capacity_reference(E):
    def chunk(e):
        e = e.double()
        w = (-(e - e.amin(dim=1, keepdim=True))).exp()
        z = w.sum(dim=1, keepdim=True)
        e1 = (e * w).sum(dim=1, keepdim=True) / z
        return ((((e - e1) ** 2 * w).sum(dim=1, keepdim=True) / z)[:, 0],)

    return _rows(chunk, E)


def _one_input(scale=1.0):
    return lambda cfg, gen: (_randn(gen, cfg["n"], cfg["m"], scale=scale),)


for _spec in (
    OpSpec(
        "kurtosis",
        F32,
        _one_input(),
        _kurtosis_unfused,
        _kurtosis_reference,
        _rowwise_bytes(F32, F32, out_full=False),
        tol=1e-4,
        note="m4 / m2², the biased moment ratio (scipy: fisher=False, bias=True)",
    ),
    OpSpec(
        "ess",
        F32,
        _one_input(scale=3.0),
        _ess_unfused,
        _ess_reference,
        _rowwise_bytes(F32, F32, out_full=False),
        tol=1e-4,
        exps=lambda cfg: cfg["n"] * cfg["m"],
    ),
    OpSpec(
        "papr",
        F32,
        _one_input(),
        _papr_unfused,
        _papr_reference,
        _rowwise_bytes(F32, F32, out_full=False),
        tol=1e-4,
    ),
    OpSpec(
        "householder",
        F32,
        _householder_inputs,
        _householder_unfused,
        _householder_reference,
        _rowwise_bytes(F32, F32, n_in=2, out_full=False),
        tol=1e-4,
    ),
    OpSpec(
        "heat_capacity",
        F32,
        _one_input(scale=2.0),
        _heat_capacity_unfused,
        _heat_capacity_reference,
        _rowwise_bytes(F32, F32, out_full=False),
        tol=1e-4,
        exps=lambda cfg: cfg["n"] * cfg["m"],
    ),
):
    _register(_spec)


# ---------------------------------------------------------------------------
# chain_ops/advanced.py: covariance estimation and normalized policy objectives
# ---------------------------------------------------------------------------
def _ols_hc0_inputs(cfg, gen):
    x = _randn(gen, cfg["n"], cfg["m"])
    noise = _randn(gen, cfg["n"], cfg["m"])
    return x, 0.5 * x + noise * (0.5 + x.abs())  # heteroskedastic: the noise grows with |x|


def _ols_hc0_unfused(X, Y):
    a = (X * X).sum(dim=1)
    c = (X * Y).sum(dim=1)
    theta = torch.where(a == 0, 0.0, c / a)
    r = Y - theta[:, None] * X
    h = (X * X * r * r).sum(dim=1)
    return (torch.where(a == 0, 0.0, h / (a * a)),)


def _ols_hc0_reference(X, Y):
    def chunk(x, y):
        x, y = x.double(), y.double()
        a = (x * x).sum(dim=1)
        theta = torch.where(a == 0, 0.0, (x * y).sum(dim=1) / a)
        h = (x * x * (y - theta[:, None] * x) ** 2).sum(dim=1)
        return (torch.where(a == 0, 0.0, h / (a * a)),)

    return _rows(chunk, X, Y)


def _policy_inputs(cfg, gen):
    # Advantage and log-prob are correlated (~0.38), as a policy's are with its rewards.
    # Independent ones make the loss of a long row ~1/sqrt(m) of its terms (2e-7 at
    # m = 2^25), and then max|out - ref| / max|ref| measures nothing.
    z = _randn(gen, cfg["n"], cfg["m"])
    advantage = 2.0 * z + 0.5
    logprob = -2.0 * _randn(gen, cfg["n"], cfg["m"]).abs() + 0.5 * z
    return advantage, logprob


def _policy_unfused(A, G):
    m = A.shape[1]
    mean = A.sum(dim=1, keepdim=True) / m
    ss = ((A - mean) * (A - mean)).sum(dim=1, keepdim=True)
    scale = (ss / (m - 1)).sqrt() + 1e-8
    return (-(G * (A - mean) / scale).sum(dim=1) / m,)


def _policy_reference(A, G):
    def chunk(a, g):
        a, g = a.double(), g.double()
        scale = a.std(dim=1, correction=1, keepdim=True) + 1e-8
        return (-((a - a.mean(dim=1, keepdim=True)) / scale * g).mean(dim=1),)

    return _rows(chunk, A, G)


def _ledoit_wolf_inputs(cfg, gen):
    b, n, d = cfg["b"], cfg["n"], cfg["d"]
    scales = 1.0 + torch.rand(b, 1, d, generator=gen, device="cuda")  # unequal variances
    return (_randn(gen, b, n, d) * scales,)


def _shrink(cov, fourthsum, n):
    """Ledoit-Wolf's shrinkage of ``cov`` given Σ_j (Σ_k z_jk²)², as chain_ops computes it."""
    d = cov.shape[-1]
    mu = cov.diagonal(dim1=1, dim2=2).sum(dim=-1) / d
    norm = (cov * cov).sum(dim=(1, 2))
    delta = norm / d - mu * mu
    beta = torch.minimum((fourthsum / n - norm) / (d * n), delta)
    rho = torch.zeros_like(beta) if d == 1 else torch.where(beta == 0, 0.0, beta / delta)
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1 - rho)[:, None, None] * cov + (rho * mu)[:, None, None] * eye, rho


def _ledoit_wolf_unfused(X):
    n = X.shape[1]
    z = X - X.sum(dim=1, keepdim=True) / n
    cov = torch.bmm(z.transpose(1, 2), z) / n  # float32, TF32 off
    z2 = z * z
    fourthsum = torch.bmm(z2.transpose(1, 2), z2).sum(dim=(1, 2))  # Σ_j Σ_k,l z_jk² z_jl²
    return _shrink(cov, fourthsum, n)


def _ledoit_wolf_reference(X):
    x = X.double()
    n = x.shape[1]
    z = x - x.mean(dim=1, keepdim=True)
    cov = torch.bmm(z.transpose(1, 2), z) / n
    fourthsum = ((z * z).sum(dim=2) ** 2).sum(dim=1)
    return _shrink(cov, fourthsum, n)


for _spec in (
    OpSpec(
        "ols_hc0",
        F32,
        _ols_hc0_inputs,
        _ols_hc0_unfused,
        _ols_hc0_reference,
        _rowwise_bytes(F32, F32, n_in=2, out_full=False),
        tol=1e-4,
        note="no library computes it on the GPU (statsmodels' HC0 runs on the CPU)",
    ),
    OpSpec(
        "normalized_policy_loss",
        F32,
        _policy_inputs,
        _policy_unfused,
        _policy_reference,
        _rowwise_bytes(F32, F32, n_in=2, out_full=False),
        tol=1e-4,
    ),
    OpSpec(
        "ledoit_wolf",
        F32,
        _ledoit_wolf_inputs,
        _ledoit_wolf_unfused,
        _ledoit_wolf_reference,
        lambda cfg: (cfg["b"] * cfg["n"] * cfg["d"] + cfg["b"] * cfg["d"] ** 2 + cfg["b"]) * 4,
        tol=1e-4,
        fp32_flops=lambda cfg: 4 * cfg["b"] * cfg["n"] * cfg["d"] ** 2,  # Zᵀ Z and (Z²)ᵀ Z²
        note="no library computes it on the GPU (scikit-learn's ledoit_wolf runs on the CPU)",
    ),
):
    _register(_spec)
