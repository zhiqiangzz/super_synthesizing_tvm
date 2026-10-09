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
"""The softmax family: a maximum, then sums that are shifted by it.

Unfused, every member walks the class axis twice or more: once for the
maximum and once per sum that subtracts it. Fused, all of them are the
*online softmax* reducer (Milakov & Gimelshein): a running maximum and
sums kept relative to it, rescaled by ``exp(old_max - new_max)`` whenever
the maximum moves.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register


def online_softmax(rescaled: int, plain: int = 0):
    """``(max, rescaled sums.., plain sums..)``: the first group is kept relative to the max."""

    def merge(a, b):
        m = tir.max(a[0], b[0])
        ra, rb = tir.exp(a[0] - m), tir.exp(b[0] - m)
        scaled = [a[k] * ra + b[k] * rb for k in range(1, 1 + rescaled)]
        sums = [a[k] + b[k] for k in range(1 + rescaled, 1 + rescaled + plain)]
        return (m, *scaled, *sums)

    def identity(*dtypes):
        return (tir.min_value(dtypes[0]), *(tir.const(0.0, t) for t in dtypes[1:]))

    return te.comm_reducer(merge, identity, name="online_softmax")


def _scores():
    n, m = te.var("n"), te.var("m")
    return te.placeholder((n, m), name="X", dtype=DTYPE)


def two_pass(X):
    """``mx = max_j x`` and ``den = Σ_j exp(x - mx)``, one pass each."""
    n, m = X.shape
    j1 = te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(X[i, j1], axis=j1), name="mx")
    e = te.compute((n, m), lambda i, j: tir.exp(X[i, j] - mx[i]), name="e")
    j2 = te.reduce_axis((0, m), "j")
    den = te.compute((n,), lambda i: te.sum(e[i, j2], axis=j2), name="den")
    return mx, e, den


def one_pass(X):
    """The same two tensors from one online-softmax reduction."""
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    red = online_softmax(1)
    one = tir.const(1.0, DTYPE)
    return te.compute((n,), lambda i: red((X[i, j], one), axis=j), name="lse")


def _np_lse(x):
    x = x.astype(np.float64)
    mx = x.max(axis=-1, keepdims=True)
    return mx, np.exp(x - mx).sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# softmax / log-softmax / log-sum-exp
# ---------------------------------------------------------------------------
def softmax_unfused():
    X = _scores()
    _, e, den = two_pass(X)
    P = te.compute(X.shape, lambda i, j: e[i, j] / den[i], name="P")
    return [X], [P]


def softmax_fused():
    X = _scores()
    mx, den = one_pass(X)
    P = te.compute(X.shape, lambda i, j: tir.exp(X[i, j] - mx[i]) / den[i], name="P")
    return [X], [P]


def softmax_reference(x):
    mx, den = _np_lse(x)
    return (np.exp(x.astype(np.float64) - mx) / den,)


def log_softmax_unfused():
    X = _scores()
    mx, _, den = two_pass(X)
    out = te.compute(X.shape, lambda i, j: X[i, j] - mx[i] - tir.log(den[i]), name="logp")
    return [X], [out]


def log_softmax_fused():
    X = _scores()
    mx, den = one_pass(X)
    out = te.compute(X.shape, lambda i, j: X[i, j] - mx[i] - tir.log(den[i]), name="logp")
    return [X], [out]


def log_softmax_reference(x):
    mx, den = _np_lse(x)
    return (x.astype(np.float64) - mx - np.log(den),)


def logsumexp_unfused():
    X = _scores()
    mx, _, den = two_pass(X)
    lse = te.compute((X.shape[0],), lambda i: tir.log(den[i]) + mx[i], name="lse_out")
    return [X], [lse]


def logsumexp_fused():
    X = _scores()
    mx, den = one_pass(X)
    lse = te.compute((X.shape[0],), lambda i: tir.log(den[i]) + mx[i], name="lse_out")
    return [X], [lse]


def logsumexp_reference(x):
    mx, den = _np_lse(x)
    return ((np.log(den) + mx)[:, 0],)


# ---------------------------------------------------------------------------
# cross entropy against the logits of a linear classifier (Cut Cross-Entropy)
# ---------------------------------------------------------------------------
def _classifier():
    N, D, V = te.var("N"), te.var("D"), te.var("V")
    X = te.placeholder((N, D), name="X", dtype=DTYPE)
    W = te.placeholder((V, D), name="W", dtype=DTYPE)
    Y = te.placeholder((N, V), name="Y", dtype=DTYPE)
    d = te.reduce_axis((0, D), name="d")
    L = te.compute((N, V), lambda i, v: te.sum(X[i, d] * W[v, d], axis=d), name="logits")
    return X, W, Y, L


def cross_entropy_unfused():
    """``loss = lse(logits) - Σ_v y_v logit_v``: the logits are read three times."""
    X, W, Y, L = _classifier()
    N, V = L.shape
    v1 = te.reduce_axis((0, V), name="v")
    mx = te.compute((N,), lambda i: te.max(L[i, v1], axis=v1), name="mx")
    e = te.compute((N, V), lambda i, v: tir.exp(L[i, v] - mx[i]), name="e")
    v2 = te.reduce_axis((0, V), name="v")
    den = te.compute((N,), lambda i: te.sum(e[i, v2], axis=v2), name="den")
    v3 = te.reduce_axis((0, V), name="v")
    t = te.compute((N,), lambda i: te.sum(Y[i, v3] * L[i, v3], axis=v3), name="t")
    loss = te.compute((N,), lambda i: mx[i] + tir.log(den[i]) - t[i], name="loss")
    return [X, W, Y], [loss]


def cross_entropy_fused():
    X, W, Y, L = _classifier()
    N, V = L.shape
    v = te.reduce_axis((0, V), name="v")
    red = online_softmax(1, plain=1)
    one = tir.const(1.0, DTYPE)
    mx, den, t = te.compute(
        (N,), lambda i: red((L[i, v], one, Y[i, v] * L[i, v]), axis=v), name="ce"
    )
    loss = te.compute((N,), lambda i: mx[i] + tir.log(den[i]) - t[i], name="loss")
    return [X, W, Y], [loss]


def cross_entropy_reference(x, w, y):
    logits = x.astype(np.float64) @ w.astype(np.float64).T
    mx, den = _np_lse(logits)
    return ((np.log(den) + mx)[:, 0] - (y.astype(np.float64) * logits).sum(axis=1),)


# ---------------------------------------------------------------------------
# cross entropy by its definition, against soft labels: -Σ_j q_j log p_j
# ---------------------------------------------------------------------------
def _soft_inputs():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype=DTYPE)
    Q = te.placeholder((n, m), name="Q", dtype=DTYPE)
    return X, Q


def _definition(X, q):
    """``-Σ_j q(i, j) log softmax(x)_j`` with ``log softmax`` materialised."""
    n, m = X.shape
    mx, _, den = two_pass(X)
    logp = te.compute(X.shape, lambda i, j: X[i, j] - mx[i] - tir.log(den[i]), name="logp")
    j3 = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.sum(-q(i, j3) * logp[i, j3], axis=j3), name="loss")


def _definition_fused(X, q):
    """``Σ q`` and ``T = Σ q (x - mx)`` ride along; the log of the mass enters at the end.

    ``T`` is kept relative to the running maximum like the mass itself, so that
    a large common offset of the scores never has to cancel:
    ``loss = Σq log(den) - T``.
    """
    n, m = X.shape

    def merge(a, b):
        mx = tir.max(a[0], b[0])
        da, db = a[0] - mx, b[0] - mx
        return (
            mx,
            a[1] * tir.exp(da) + b[1] * tir.exp(db),
            a[2] + b[2],
            a[3] + da * a[2] + b[3] + db * b[2],
        )

    def identity(t0, t1, t2, t3):
        return (tir.min_value(t0), *(tir.const(0.0, t) for t in (t1, t2, t3)))

    red = te.comm_reducer(merge, identity, name="soft_ce")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    _, den, sq, t = te.compute(
        (n,), lambda i: red((X[i, j], one, q(i, j), zero), axis=j), name="ce"
    )
    return te.compute((n,), lambda i: sq[i] * tir.log(den[i]) - t[i], name="loss")


def _np_definition(x, q):
    mx, den = _np_lse(x)
    logp = x.astype(np.float64) - mx - np.log(den)
    return (-(q * logp).sum(axis=1),)


def soft_cross_entropy_unfused():
    X, Q = _soft_inputs()
    return [X, Q], [_definition(X, lambda i, j: Q[i, j])]


def soft_cross_entropy_fused():
    X, Q = _soft_inputs()
    return [X, Q], [_definition_fused(X, lambda i, j: Q[i, j])]


def soft_cross_entropy_reference(x, q):
    return _np_definition(x, q.astype(np.float64))


SMOOTHING = 0.1


def _smoothed(Y):
    """``q = (1 - a) y + a / K`` with ``K`` the number of classes."""
    k = Y.shape[1].astype(DTYPE)
    keep, spread = tir.const(1.0 - SMOOTHING, DTYPE), tir.const(SMOOTHING, DTYPE)
    return lambda i, j: keep * Y[i, j] + spread / k


def label_smoothing_unfused():
    X, Y = _soft_inputs()
    return [X, Y], [_definition(X, _smoothed(Y))]


def label_smoothing_fused():
    X, Y = _soft_inputs()
    return [X, Y], [_definition_fused(X, _smoothed(Y))]


def label_smoothing_reference(x, y):
    q = (1.0 - SMOOTHING) * y.astype(np.float64) + SMOOTHING / y.shape[1]
    return _np_definition(x, q)


# ---------------------------------------------------------------------------
# entropy of the softmax distribution: -Σ_j p_j log p_j
# ---------------------------------------------------------------------------
def entropy_unfused():
    X = _scores()
    n, m = X.shape
    mx, e, den = two_pass(X)
    logp = te.compute(X.shape, lambda i, j: X[i, j] - mx[i] - tir.log(den[i]), name="logp")
    j3 = te.reduce_axis((0, m), "j")
    H = te.compute((n,), lambda i: te.sum(-(e[i, j3] / den[i]) * logp[i, j3], axis=j3), name="H")
    return [X], [H]


def entropy_fused():
    """``T = Σ (x - mx) exp(x - mx)`` next to the mass: ``H = log den - T / den``."""
    X = _scores()
    n, m = X.shape

    def merge(a, b):
        mx = tir.max(a[0], b[0])
        da, db = mx - a[0], mx - b[0]
        ra, rb = tir.exp(-da), tir.exp(-db)
        return (mx, a[1] * ra + b[1] * rb, ra * (a[2] - da * a[1]) + rb * (b[2] - db * b[1]))

    def identity(t0, t1, t2):
        return (tir.min_value(t0), tir.const(0.0, t1), tir.const(0.0, t2))

    red = te.comm_reducer(merge, identity, name="entropy")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    _, den, t = te.compute((n,), lambda i: red((X[i, j], one, zero), axis=j), name="st")
    H = te.compute((n,), lambda i: tir.log(den[i]) - t[i] / den[i], name="H")
    return [X], [H]


def entropy_reference(x):
    mx, den = _np_lse(x)
    logp = x.astype(np.float64) - mx - np.log(den)
    return (-(np.exp(logp) * logp).sum(axis=1),)


# ---------------------------------------------------------------------------
# outside what any finite reducer reaches: the scores are divided by a sum of their own
# ---------------------------------------------------------------------------
def logit_norm_unfused():
    """LogitNorm: ``lse(x / ‖x‖)``. The norm multiplies the score inside the ``exp``."""
    X = _scores()
    n, m = X.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    ssq = te.compute((n,), lambda i: te.sum(X[i, j1] * X[i, j1], axis=j1), name="ssq")
    nrm = te.compute((n,), lambda i: tir.sqrt(ssq[i]), name="nrm")
    den = te.compute((n,), lambda i: te.sum(tir.exp(X[i, j2] / nrm[i]), axis=j2), name="den")
    return [X], [te.compute((n,), lambda i: tir.log(den[i]), name="lse_out")]


def logit_norm_reference(x):
    x = x.astype(np.float64)
    nrm = np.sqrt((x * x).sum(axis=1, keepdims=True))
    return (np.log(np.exp(x / nrm).sum(axis=1)),)


def standardised_softmax_unfused():
    """The normaliser of a softmax over standardised scores: ``Σ exp((x - mean) / sigma)``."""
    X = _scores()
    n, m = X.shape
    count = m.astype(DTYPE)
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / count, name="mean")
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - mean[i]) * (X[i, j2] - mean[i]), axis=j2), name="ss"
    )
    sigma = te.compute((n,), lambda i: tir.sqrt(ss[i] / count), name="sigma")
    return [X], [
        te.compute(
            (n,), lambda i: te.sum(tir.exp((X[i, j3] - mean[i]) / sigma[i]), axis=j3), name="zden"
        )
    ]


def standardised_softmax_reference(x):
    x = x.astype(np.float64)
    z = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return (np.exp(z).sum(axis=1),)


register(
    Operator(
        "softmax",
        "shift/scale",
        "p = exp(x - max x) / Σ exp(x - max x)",
        softmax_unfused,
        softmax_reference,
        softmax_fused,
        states=2,
    )
)
register(
    Operator(
        "log_softmax",
        "shift/scale",
        "log p = x - max x - log Σ exp(x - max x)",
        log_softmax_unfused,
        log_softmax_reference,
        log_softmax_fused,
        states=2,
    )
)
register(
    Operator(
        "logsumexp",
        "shift/scale",
        "max x + log Σ exp(x - max x)",
        logsumexp_unfused,
        logsumexp_reference,
        logsumexp_fused,
        states=2,
    )
)
register(
    Operator(
        "cross_entropy",
        "shift/scale",
        "lse(X Wᵀ) - Σ y (X Wᵀ): the logits never need to be stored (Cut Cross-Entropy)",
        cross_entropy_unfused,
        cross_entropy_reference,
        cross_entropy_fused,
        states=3,
    )
)
register(
    Operator(
        "soft_cross_entropy",
        "hoist",
        "-Σ q log softmax(x) with log softmax materialised",
        soft_cross_entropy_unfused,
        soft_cross_entropy_reference,
        soft_cross_entropy_fused,
        states=4,
    )
)
register(
    Operator(
        "label_smoothing",
        "hoist",
        "-Σ ((1 - a) y + a / K) log softmax(x)",
        label_smoothing_unfused,
        label_smoothing_reference,
        label_smoothing_fused,
        states=4,
    )
)
register(
    Operator(
        "softmax_entropy",
        "hoist",
        "-Σ p log p with p = softmax(x)",
        entropy_unfused,
        entropy_reference,
        entropy_fused,
        states=3,
    )
)
register(
    Operator(
        "logit_norm",
        "negative",
        "log Σ exp(x / ‖x‖): the norm scales the score under the exp",
        logit_norm_unfused,
        logit_norm_reference,
        fusible=False,
        reason="den reads nrm from inside its reduction and neither",
        exists=False,
    )
)
register(
    Operator(
        "standardised_softmax",
        "negative",
        "Σ exp((x - mean) / sigma): the mean can leave the exp, the deviation cannot",
        standardised_softmax_unfused,
        standardised_softmax_reference,
        fusible=False,
        reason="(no finite lifting found)",
        exists=False,
    )
)
