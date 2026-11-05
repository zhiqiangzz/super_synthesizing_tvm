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
"""What each showcase operator is measured against.

A baseline is a *builder*: given the operator's inputs it does any setup that
does not belong in the timed region (imports, layout changes, compilation,
saved statistics) and returns a no-argument callable producing the operator's
outputs in the order ``ops.OpSpec.reference`` gives them. A builder that
raises -- a library not installed, a size it does not support -- is reported
with its message and the run goes on.

Every operator has the two unfused baselines:

``torch-unfused``
    the textbook chain, one torch call per link, every intermediate stored;
``torch.compile``
    the same function through TorchInductor. Inductor fuses on its own (a row
    that fits on chip becomes one persistent reduction; long softmax rows get
    its online-softmax rewrite), so this is the compiler's fusion, not an
    unfused program.

On top of these come the idiomatic torch call where torch has one, and the
fused library implementations. Where a library computes something slightly
different from the operator, the builder says so in ``NOTES``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from .ops import LN_EPS, SINKHORN_EPS, SPECS

Builder = Callable[..., Callable[[], tuple]]

NOTES: dict[tuple[str, str], str] = {}


def _note(op: str, impl: str, text: str) -> None:
    NOTES[(op, impl)] = text


def _unfused(op: str) -> Builder:
    fn = SPECS[op].unfused
    return lambda *inputs: lambda: fn(*inputs)


def _compiled(op: str) -> Builder:
    def build(*inputs):
        torch._dynamo.reset()  # a fresh cache per size: no silent fallback to eager
        fn = torch.compile(SPECS[op].unfused, dynamic=False, fullgraph=True)
        fn(*inputs)  # compile outside the timed region
        return lambda: fn(*inputs)

    return build


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------
def _softmax_torch(X):
    return lambda: (torch.softmax(X, dim=1),)


def _softmax_quack(X):
    from quack.softmax import softmax

    return lambda: (softmax(X),)


def _softmax_liger(X):
    from liger_kernel.ops.softmax import LigerSoftmaxFunction

    return lambda: (LigerSoftmaxFunction.apply(X),)


# ---------------------------------------------------------------------------
# cross_entropy
# ---------------------------------------------------------------------------
def _ce_torch(X, W, labels):
    # what a training loop writes: bf16 logits from the head, the loss in float32
    return lambda: (F.cross_entropy((X @ W.T).float(), labels, reduction="none"),)


def _ce_liger(X, W, labels):
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

    return lambda: (liger_fused_linear_cross_entropy(X, W, labels, reduction="none"),)


def _ce_cce(X, W, labels):
    from cut_cross_entropy import linear_cross_entropy

    return lambda: (linear_cross_entropy(X, W, labels, reduction="none"),)


def _ce_quack(X, W, labels):
    from quack.cross_entropy import cross_entropy

    return lambda: (cross_entropy(X @ W.T, labels, reduction="none"),)


_note("cross_entropy", "liger FLCE", "inputs do not require grad, so no gradient is formed")
_note("cross_entropy", "cuBLAS + QuACK CE", "the logits are stored in bf16 between the two")


# ---------------------------------------------------------------------------
# softmax_entropy
# ---------------------------------------------------------------------------
def _entropy_verl(X):
    # verl's entropy_from_logits: lse(x) - Σ softmax(x) x
    def run():
        x = X.float()
        return (torch.logsumexp(x, dim=-1) - (torch.softmax(x, dim=-1) * x).sum(dim=-1),)

    return run


_note("softmax_entropy", "torch (verl)", "verl.utils.torch_functional.entropy_from_logits")


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def _sdpa(backend_name: str) -> Builder:
    def build(Q, K, V):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = getattr(SDPBackend, backend_name)

        def run():
            with sdpa_kernel(backend):
                return (F.scaled_dot_product_attention(Q, K, V),)

        run()  # an unsupported size raises here, not inside the timer
        return run

    return build


def _fa2(Q, K, V):
    from flash_attn import flash_attn_func

    q, k, v = (t.transpose(1, 2).contiguous() for t in (Q, K, V))  # (b, s, h, d)
    return lambda: (flash_attn_func(q, k, v).transpose(1, 2),)


def _fa3(Q, K, V):
    from flash_attn_interface import flash_attn_func

    q, k, v = (t.transpose(1, 2).contiguous() for t in (Q, K, V))
    return lambda: (flash_attn_func(q, k, v).transpose(1, 2),)  # FA3 3.0.0b1 returns out only


def _flex(Q, K, V):
    from torch.nn.attention.flex_attention import flex_attention

    torch._dynamo.reset()
    fn = torch.compile(flex_attention, dynamic=False)
    fn(Q, K, V)
    return lambda: (fn(Q, K, V),)


_note("attention", "FlashAttention-2", "q, k, v given in its (b, s, h, d) layout")
_note("attention", "FlashAttention-3", "q, k, v given in its (b, s, h, d) layout")


# ---------------------------------------------------------------------------
# layernorm and its backward
# ---------------------------------------------------------------------------
def _ln_torch(X):
    shape = (X.shape[1],)
    return lambda: (F.layer_norm(X, shape, eps=LN_EPS),)


def _affine(X):
    m = X.shape[1]
    return (
        torch.ones(m, device=X.device, dtype=X.dtype),
        torch.zeros(m, device=X.device, dtype=X.dtype),
    )


def _ln_liger(X):
    from liger_kernel.ops.layer_norm import LigerLayerNormFunction

    if X.shape[1] > 65536:
        # Liger refuses such rows ("n exceeds the recommended Triton blocksize 65536"),
        # and on one run its compile hung at 18 GB of host memory first
        raise ValueError("Liger LayerNorm takes rows of at most 65536 elements")
    w, b = _affine(X)
    return lambda: (LigerLayerNormFunction.apply(X, w, b, LN_EPS),)


def _ln_apex(X):
    from apex.normalization.fused_layer_norm import fused_layer_norm

    shape = (X.shape[1],)
    return lambda: (fused_layer_norm(X, shape, LN_EPS),)


def _ln_quack(X):
    # QuACK's LayerNorm is its RMSNorm kernel with is_layernorm=True; no weight, no bias
    from quack.rmsnorm import _rmsnorm_fwd

    out = torch.empty_like(X)

    def run():
        _rmsnorm_fwd(X, None, out, None, None, None, None, None, LN_EPS, True, 0.0)
        return (out,)

    return run


_note("layernorm", "liger", "takes weight and bias: given ones and zeros, two extra row reads")
_note("layernorm", "quack", "quack.rmsnorm._rmsnorm_fwd(is_layernorm=True), no affine")


def _ln_backward_torch(X, G):
    # production: the forward pass saved mean and rstd, the backward reads them
    m = X.shape[1]
    _, mean, rstd = torch.ops.aten.native_layer_norm(X, [m], None, None, LN_EPS)
    mask = [True, False, False]
    return lambda: (
        torch.ops.aten.native_layer_norm_backward(G, X, [m], mean, rstd, None, None, mask)[0],
    )


_note(
    "layernorm_backward",
    "torch (saved stats)",
    "reads the mean and rstd its forward saved; the operator recomputes them from x",
)


# ---------------------------------------------------------------------------
# contrastive
# ---------------------------------------------------------------------------
def _clip_loss(S):
    # open_clip.loss.ClipLoss: cross_entropy of the logits and of their transpose
    labels = torch.arange(S.shape[0], device=S.device)
    return lambda: (
        F.cross_entropy(S, labels, reduction="none"),
        F.cross_entropy(S.T, labels, reduction="none"),
    )


_note("contrastive", "open_clip ClipLoss", "reimplemented: the two cross_entropy calls of ClipLoss")


# ---------------------------------------------------------------------------
# sinkhorn_iteration
# ---------------------------------------------------------------------------
def _sinkhorn_keops(x, y, a, b, g):
    from pykeops.torch import LazyTensor

    la, lb = a.log(), b.log()

    def half(p, q, pot, logw):
        p_i = LazyTensor(p[:, None, :])
        q_j = LazyTensor(q[None, :, :])
        c = ((p_i - q_j) ** 2).sum(-1)
        s = (LazyTensor(pot[None, :, None]) - c) / SINKHORN_EPS + LazyTensor(logw[None, :, None])
        return -SINKHORN_EPS * s.logsumexp(dim=1)[:, 0]

    def run():
        f = half(x, y, g, lb)
        return f, half(y, x, f, la)

    return run


_note("sinkhorn_iteration", "KeOps", "online log-sum-exp reductions, as GeomLoss runs them")


def _sinkhorn_flash(x, y, a, b, g):
    # FlashSinkhorn's kernel works on shifted potentials: f = f̂ + |x|², with
    # f̂_i = -eps LSE_j[2 x_i·y_j / eps + (g_j - |y_j|² + eps log b_j) / eps], and the
    # same for g from the new f. TF32 is off: with it the result fails the tolerance.
    from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import flashsinkhorn_lse

    eps = SINKHORN_EPS
    alpha, beta = (x * x).sum(dim=1), (y * y).sum(dim=1)
    gamma, delta = eps * a.log(), eps * b.log()

    def run():
        f = flashsinkhorn_lse(x, y, (g - beta + delta) / eps, eps, allow_tf32=False) + alpha
        bias = (f - alpha + gamma) / eps
        return f, flashsinkhorn_lse(y, x, bias, eps, allow_tf32=False) + beta

    run()  # autotune outside the timed region
    return run


_note(
    "sinkhorn_iteration",
    "FlashSinkhorn",
    "its kernel per half-iteration, TF32 off; squared norms and log-weights once per solve",
)


# ---------------------------------------------------------------------------
# pearson
# ---------------------------------------------------------------------------
def _corrcoef(X, Y):
    if X.shape[0] != 1:
        raise ValueError(
            "torch.corrcoef is the n x n matrix of all pairs: comparable only at n = 1"
        )
    return lambda: (torch.corrcoef(torch.cat([X, Y]))[0, 1:2],)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
LIBRARIES: dict[str, dict[str, Builder]] = {
    "softmax": {"torch": _softmax_torch, "quack": _softmax_quack, "liger": _softmax_liger},
    "cross_entropy": {
        "torch": _ce_torch,
        "liger FLCE": _ce_liger,
        "CCE": _ce_cce,
        "cuBLAS + QuACK CE": _ce_quack,
    },
    "softmax_entropy": {"torch (verl)": _entropy_verl},
    "attention": {
        "FlashAttention-2": _fa2,
        "FlashAttention-3": _fa3,
        "cuDNN SDPA": _sdpa("CUDNN_ATTENTION"),
        "torch SDPA flash": _sdpa("FLASH_ATTENTION"),
        "FlexAttention": _flex,
    },
    "layernorm": {"torch": _ln_torch, "apex": _ln_apex, "liger": _ln_liger, "quack": _ln_quack},
    "layernorm_backward": {"torch (saved stats)": _ln_backward_torch},
    "contrastive": {"open_clip ClipLoss": _clip_loss},
    "sinkhorn_iteration": {"FlashSinkhorn": _sinkhorn_flash, "KeOps": _sinkhorn_keops},
    "pearson": {"torch.corrcoef": _corrcoef},
    "kurtosis": {},
    "ess": {},
    "papr": {},
    "householder": {},
    "heat_capacity": {},
    "ols_hc0": {},
    "normalized_policy_loss": {},
    "ledoit_wolf": {},
}


def _tirx(op: str, cfg: dict, variant: str) -> Builder:
    """Our kernel (``kernels/<op>.py``) at ``cfg``: one-pass ``fused`` or k-pass ``unfused``."""

    def build(*inputs):
        from kernels import load

        run = load(op).build(cfg, variant=variant)
        run(*inputs)  # compile and load outside the timed region
        return lambda: run(*inputs)

    return build


def builders(op: str, cfg: dict | None = None) -> dict[str, Builder]:
    """Every implementation of ``op`` at ``cfg``: ours first, then the unfused baselines,
    torch's call and the libraries. Ours appear once ``kernels/<op>.py`` exists."""
    from kernels import available

    ours = {}
    if cfg is not None and available(op):
        ours = {f"TIRx-{v}": _tirx(op, cfg, v) for v in ("fused", "unfused")}
    return {
        **ours,
        "torch-unfused": _unfused(op),
        "torch.compile": _compiled(op),
        **LIBRARIES[op],
    }


assert set(LIBRARIES) == set(SPECS), set(LIBRARIES) ^ set(SPECS)
