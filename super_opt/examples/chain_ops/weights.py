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
"""Sums under exponential weights ``exp(x - max x)`` or ``exp(-(E - min E))``.

What is summed under the weights varies: their squares (the collision
probability of a softmax, the effective sample size of importance weights),
a second tensor (the Jacobian-vector product of a softmax), energies and
their fluctuation (a heat capacity). The weights always wait for an extreme
of the data, so unfused each of these walks the axis once for the extreme
and once per sum; fused they are the online softmax with more accumulators.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register
from .softmax import one_pass, online_softmax, two_pass


def _rows(name: str = "X", like=None):
    shape = (te.var("n"), te.var("m")) if like is None else like.shape
    return te.placeholder(shape, name=name, dtype=DTYPE)


def _sum(X, f, name: str):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.sum(f(i, j), axis=j), name=name)


def _np_weights(x):
    x = x.astype(np.float64)
    return np.exp(x - x.max(axis=1, keepdims=True))


def squared_weights():
    """``(max, Σ w, Σ w²)``: the squares are rescaled by the square of what rescales the sum."""

    def merge(a, b):
        m = tir.max(a[0], b[0])
        ra, rb = tir.exp(a[0] - m), tir.exp(b[0] - m)
        return (m, a[1] * ra + b[1] * rb, a[2] * ra * ra + b[2] * rb * rb)

    def identity(t0, t1, t2):
        return (tir.min_value(t0), tir.const(0.0, t1), tir.const(0.0, t2))

    return te.comm_reducer(merge, identity, name="squared_weights")


def _squared_states(X):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    one = tir.const(1.0, DTYPE)
    return te.compute((n,), lambda i: squared_weights()((X[i, j], one, one), axis=j), name="st")


# ---------------------------------------------------------------------------
# Σ p² of a softmax (its collision probability; the Rényi entropy of order 2)
# ---------------------------------------------------------------------------
def collision_unfused():
    X = _rows()
    _, e, den = two_pass(X)
    P = te.compute(X.shape, lambda i, k: e[i, k] / den[i], name="P")
    return [X], [_sum(X, lambda i, j: P[i, j] * P[i, j], "coll")]


def collision_fused():
    X = _rows()
    _, den, sq = _squared_states(X)
    return [X], [te.compute((X.shape[0],), lambda i: sq[i] / (den[i] * den[i]), name="coll")]


def collision_reference(x):
    w = _np_weights(x)
    p = w / w.sum(axis=1, keepdims=True)
    return ((p * p).sum(axis=1),)


# ---------------------------------------------------------------------------
# effective sample size of importance weights given by their logarithms
# ---------------------------------------------------------------------------
def ess_unfused():
    LW = _rows("LW")
    n, m = LW.shape
    j1 = te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(LW[i, j1], axis=j1), name="mx")
    w = te.compute(LW.shape, lambda i, k: tir.exp(LW[i, k] - mx[i]), name="w")
    sw = _sum(LW, lambda i, j: w[i, j], "sw")
    sw2 = _sum(LW, lambda i, j: w[i, j] * w[i, j], "sw2")
    return [LW], [te.compute((n,), lambda i: sw[i] * sw[i] / sw2[i], name="ess")]


def ess_fused():
    LW = _rows("LW")
    _, sw, sw2 = _squared_states(LW)
    return [LW], [te.compute((LW.shape[0],), lambda i: sw[i] * sw[i] / sw2[i], name="ess")]


def ess_reference(lw):
    w = _np_weights(lw)
    return (w.sum(axis=1) ** 2 / (w * w).sum(axis=1),)


# ---------------------------------------------------------------------------
# Jacobian-vector product of a softmax: p (g - Σ g p)
# ---------------------------------------------------------------------------
def softmax_jvp_unfused():
    X = _rows("X")
    G = _rows("G", like=X)
    _, e, den = two_pass(X)
    P = te.compute(X.shape, lambda i, k: e[i, k] / den[i], name="P")
    gp = _sum(X, lambda i, j: G[i, j] * P[i, j], "gp")
    return [X, G], [te.compute(X.shape, lambda i, k: P[i, k] * (G[i, k] - gp[i]), name="jvp")]


def softmax_jvp_fused():
    X = _rows("X")
    G = _rows("G", like=X)
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    one = tir.const(1.0, DTYPE)
    mx, den, acc = te.compute(
        (n,), lambda i: online_softmax(2)((X[i, j], one, G[i, j]), axis=j), name="st"
    )
    return [X, G], [
        te.compute(
            X.shape,
            lambda i, k: tir.exp(X[i, k] - mx[i]) / den[i] * (G[i, k] - acc[i] / den[i]),
            name="jvp",
        )
    ]


def softmax_jvp_reference(x, g):
    w = _np_weights(x)
    p = w / w.sum(axis=1, keepdims=True)
    g = g.astype(np.float64)
    return (p * (g - (g * p).sum(axis=1, keepdims=True)),)


# ---------------------------------------------------------------------------
# symmetric contrastive loss per example: a log-sum-exp along rows and along columns
# ---------------------------------------------------------------------------
def _similarity():
    n = te.var("n")
    return te.placeholder((n, n), name="S", dtype=DTYPE)


def contrastive_unfused():
    S = _similarity()
    n = S.shape[0]
    j1, j2 = te.reduce_axis((0, n), "j"), te.reduce_axis((0, n), "j")
    mx_r = te.compute((n,), lambda i: te.max(S[i, j1], axis=j1), name="mx_row")
    den_r = te.compute((n,), lambda i: te.sum(tir.exp(S[i, j2] - mx_r[i]), axis=j2), name="den_row")
    r1, r2 = te.reduce_axis((0, n), "r"), te.reduce_axis((0, n), "r")
    mx_c = te.compute((n,), lambda q: te.max(S[r1, q], axis=r1), name="mx_col")
    den_c = te.compute((n,), lambda q: te.sum(tir.exp(S[r2, q] - mx_c[q]), axis=r2), name="den_col")
    rows = te.compute((n,), lambda i: mx_r[i] + tir.log(den_r[i]) - S[i, i], name="loss_row")
    cols = te.compute((n,), lambda q: mx_c[q] + tir.log(den_c[q]) - S[q, q], name="loss_col")
    return [S], [rows, cols]


def contrastive_fused():
    S = _similarity()
    n = S.shape[0]
    mx_r, den_r = one_pass(S)
    r = te.reduce_axis((0, n), "r")
    one = tir.const(1.0, DTYPE)
    mx_c, den_c = te.compute(
        (n,), lambda q: online_softmax(1)((S[r, q], one), axis=r), name="lse_col"
    )
    rows = te.compute((n,), lambda i: mx_r[i] + tir.log(den_r[i]) - S[i, i], name="loss_row")
    cols = te.compute((n,), lambda q: mx_c[q] + tir.log(den_c[q]) - S[q, q], name="loss_col")
    return [S], [rows, cols]


def contrastive_reference(s):
    s = s.astype(np.float64)
    diag = np.diag(s)

    def lse(axis):
        mx = s.max(axis=axis)
        return mx + np.log(np.exp(s - np.expand_dims(mx, axis)).sum(axis=axis))

    return (lse(1) - diag, lse(0) - diag)


# ---------------------------------------------------------------------------
# heat capacity from energy fluctuations under Boltzmann weights
# ---------------------------------------------------------------------------
def heat_capacity_unfused():
    E = _rows("E")
    n, m = E.shape
    j1 = te.reduce_axis((0, m), "j")
    lo = te.compute((n,), lambda i: te.min(E[i, j1], axis=j1), name="lo")
    w = te.compute(E.shape, lambda i, k: tir.exp(-(E[i, k] - lo[i])), name="w")
    Z = _sum(E, lambda i, j: w[i, j], "Z")
    e1s = _sum(E, lambda i, j: E[i, j] * w[i, j], "e1s")
    e1 = te.compute((n,), lambda i: e1s[i] / Z[i], name="e1")
    cs = _sum(E, lambda i, j: (E[i, j] - e1[i]) * (E[i, j] - e1[i]) * w[i, j], "cs")
    return [E], [te.compute((n,), lambda i: cs[i] / Z[i], name="cv")]


def heat_capacity_fused():
    """A weighted mean and variance whose weights are kept relative to the lowest energy."""
    E = _rows("E")
    n, m = E.shape

    def merge(a, b):
        lo = tir.min(a[0], b[0])
        ra, rb = tir.exp(lo - a[0]), tir.exp(lo - b[0])
        za, zb = a[1] * ra, b[1] * rb
        z = za + zb
        d = b[2] - a[2]
        return (lo, z, a[2] + d * zb / z, a[3] * ra + b[3] * rb + d * d * za * zb / z)

    def identity(t0, t1, t2, t3):
        return (tir.max_value(t0), *(tir.const(0.0, t) for t in (t1, t2, t3)))

    red = te.comm_reducer(merge, identity, name="boltzmann")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    st = te.compute((n,), lambda i: red((E[i, j], one, E[i, j], zero), axis=j), name="st")
    return [E], [te.compute((n,), lambda i: st[3][i] / st[1][i], name="cv")]


def heat_capacity_reference(e):
    e = e.astype(np.float64)
    w = np.exp(-(e - e.min(axis=1, keepdims=True)))
    z = w.sum(axis=1, keepdims=True)
    e1 = (e * w).sum(axis=1, keepdims=True) / z
    return (((e - e1) ** 2 * w).sum(axis=1) / z[:, 0],)


register(
    Operator(
        "collision",
        "hoist",
        "Σ p² with p = softmax(x)",
        collision_unfused,
        collision_reference,
        collision_fused,
        states=3,
    )
)
register(
    Operator(
        "ess",
        "shift/scale",
        "(Σ w)² / Σ w² with w = exp(lw - max lw): the effective sample size",
        ess_unfused,
        ess_reference,
        ess_fused,
        states=3,
    )
)
register(
    Operator(
        "softmax_jvp",
        "shift/scale",
        "p (g - Σ g p) with p = softmax(x): the backward pass of a softmax",
        softmax_jvp_unfused,
        softmax_jvp_reference,
        softmax_jvp_fused,
        states=3,
    )
)
register(
    Operator(
        "contrastive",
        "shift/scale",
        "lse(S[i, :]) - S[i, i] and lse(S[:, i]) - S[i, i]: one chain per direction",
        contrastive_unfused,
        contrastive_reference,
        contrastive_fused,
        states=2,
    )
)
register(
    Operator(
        "heat_capacity",
        "shift/scale",
        "Σ (E - <E>)² w / Σ w with w = exp(-(E - min E))",
        heat_capacity_unfused,
        heat_capacity_reference,
        heat_capacity_fused,
        states=4,
    )
)
