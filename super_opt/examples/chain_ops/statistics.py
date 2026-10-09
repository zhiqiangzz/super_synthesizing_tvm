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
"""Descriptive statistics: sums about a mean that is itself a sum.

A weighted variance, a correlation, a standardised moment, the mean of
min-max scaled data. Unfused they walk the data once for the centre (or the
range) and once more per statistic about it. Fused they are the updating
formulas of West and of Pébay, or plain sums next to a running extreme.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register
from .moments import LN_EPS, pebay


def _rows(name: str = "X", like=None):
    shape = (te.var("n"), te.var("m")) if like is None else like.shape
    return te.placeholder(shape, name=name, dtype=DTYPE)


def _sum(X, f, name: str):
    """``out[i] = Σ_j f(i, j)`` along the rows of ``X``."""
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.sum(f(i, j), axis=j), name=name)


def _mean(X, name: str = "mean"):
    total = _sum(X, lambda i, j: X[i, j], f"total_{name}")
    count = X.shape[1].astype(DTYPE)
    return te.compute((X.shape[0],), lambda i: total[i] / count, name=name)


def _zeros(k: int):
    return lambda *dtypes: tuple(tir.const(0.0, t) for t in dtypes[:k])


# ---------------------------------------------------------------------------
# weighted variance
# ---------------------------------------------------------------------------
def weighted_variance_unfused():
    X = _rows("X")
    W = _rows("W", like=X)
    n = X.shape[0]
    sw = _sum(X, lambda i, j: W[i, j], "sw")
    swx = _sum(X, lambda i, j: W[i, j] * X[i, j], "swx")
    mean = te.compute((n,), lambda i: swx[i] / sw[i], name="mean")
    ss = _sum(X, lambda i, j: W[i, j] * (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    return [X, W], [te.compute((n,), lambda i: ss[i] / sw[i], name="wvar")]


def weighted_variance_fused():
    """West's update: the weight so far takes the place of the count."""
    X = _rows("X")
    W = _rows("W", like=X)
    n, m = X.shape

    def merge(a, b):
        w = a[0] + b[0]
        d = b[1] - a[1]
        return (w, a[1] + d * b[0] / w, a[2] + b[2] + d * d * a[0] * b[0] / w)

    red = te.comm_reducer(merge, _zeros(3), name="west")
    j = te.reduce_axis((0, m), "j")
    zero = tir.const(0.0, DTYPE)
    st = te.compute((n,), lambda i: red((W[i, j], X[i, j], zero), axis=j), name="st")
    return [X, W], [te.compute((n,), lambda i: st[2][i] / st[0][i], name="wvar")]


def weighted_variance_reference(x, w):
    x, w = x.astype(np.float64), w.astype(np.float64)
    sw = w.sum(axis=1)
    mean = (w * x).sum(axis=1) / sw
    return ((w * (x - mean[:, None]) ** 2).sum(axis=1) / sw,)


# ---------------------------------------------------------------------------
# skewness, by its definition: the third moment of the standardised data
# ---------------------------------------------------------------------------
def skewness_unfused():
    return _standardised_moment(3, "skew")


def skewness_fused():
    X = _rows()
    n, m = X.shape
    count = m.astype(DTYPE)
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    st = te.compute((n,), lambda i: pebay(3)((one, X[i, j], zero, zero), axis=j), name="st")

    def skew(i):
        sigma = tir.sqrt(st[2][i] / count)
        return st[3][i] / count / (sigma * sigma * sigma)

    return [X], [te.compute((n,), skew, name="skew")]


def skewness_reference(x):
    x = x.astype(np.float64)
    d = x - x.mean(axis=1, keepdims=True)
    return ((d**3).mean(axis=1) / (d**2).mean(axis=1) ** 1.5,)


def _standardised_moment(power: int, name: str):
    """``Σ ((x - mean) / sqrt(m2))^power / n`` the way it is defined: three passes."""
    X = _rows()
    n, m = X.shape
    count = m.astype(DTYPE)
    mean = _mean(X)
    ss = _sum(X, lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    m2 = te.compute((n,), lambda i: ss[i] / count, name="m2")
    p = tir.const(float(power), DTYPE)
    sp = _sum(X, lambda i, j: tir.power((X[i, j] - mean[i]) / tir.sqrt(m2[i]), p), f"s{power}")
    return [X], [te.compute((n,), lambda i: sp[i] / count, name=name)]


def kurtosis_unfused():
    return _standardised_moment(4, "kurt")


def kurtosis_fused():
    X = _rows()
    n, m = X.shape
    count = m.astype(DTYPE)
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    st = te.compute((n,), lambda i: pebay(4)((one, X[i, j], zero, zero, zero), axis=j), name="st")

    def kurt(i):
        m2 = st[2][i] / count
        return st[4][i] / count / (m2 * m2)

    return [X], [te.compute((n,), kurt, name="kurt")]


def kurtosis_reference(x):
    x = x.astype(np.float64)
    d = x - x.mean(axis=1, keepdims=True)
    return ((d**4).mean(axis=1) / (d**2).mean(axis=1) ** 2,)


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------
def pearson_unfused():
    X = _rows("X")
    Y = _rows("Y", like=X)
    n = X.shape[0]
    mx, my = _mean(X, "mean_x"), _mean(Y, "mean_y")
    sxy = _sum(X, lambda i, j: (X[i, j] - mx[i]) * (Y[i, j] - my[i]), "sxy")
    sxx = _sum(X, lambda i, j: (X[i, j] - mx[i]) * (X[i, j] - mx[i]), "sxx")
    syy = _sum(X, lambda i, j: (Y[i, j] - my[i]) * (Y[i, j] - my[i]), "syy")
    return [X, Y], [te.compute((n,), lambda i: sxy[i] / tir.sqrt(sxx[i] * syy[i]), name="corr")]


def pearson_fused():
    X = _rows("X")
    Y = _rows("Y", like=X)
    n, m = X.shape

    def merge(a, b):
        cnt = a[0] + b[0]
        dx, dy = b[1] - a[1], b[2] - a[2]
        f = a[0] * b[0] / cnt
        return (
            cnt,
            a[1] + dx * b[0] / cnt,
            a[2] + dy * b[0] / cnt,
            a[3] + b[3] + dx * dx * f,
            a[4] + b[4] + dy * dy * f,
            a[5] + b[5] + dx * dy * f,
        )

    red = te.comm_reducer(merge, _zeros(6), name="co_moments")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    st = te.compute(
        (n,), lambda i: red((one, X[i, j], Y[i, j], zero, zero, zero), axis=j), name="st"
    )
    return [X, Y], [
        te.compute((n,), lambda i: st[5][i] / tir.sqrt(st[3][i] * st[4][i]), name="corr")
    ]


def pearson_reference(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    dx, dy = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    return ((dx * dy).sum(axis=1) / np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1)),)


# ---------------------------------------------------------------------------
# Σ exp(x - mean): the partition sum of the centred data
# ---------------------------------------------------------------------------
def exp_centered_unfused():
    X = _rows()
    mean = _mean(X)
    return [X], [_sum(X, lambda i, j: tir.exp(X[i, j] - mean[i]), "zc")]


def exp_centered_fused():
    """The sum is kept relative to the mean so far and rescaled when the mean moves."""
    X = _rows()
    n, m = X.shape

    def merge(a, b):
        cnt = a[0] + b[0]
        mean = a[1] + (b[1] - a[1]) * b[0] / cnt
        return (cnt, mean, a[2] * tir.exp(a[1] - mean) + b[2] * tir.exp(b[1] - mean))

    red = te.comm_reducer(merge, _zeros(3), name="centred_exp")
    j = te.reduce_axis((0, m), "j")
    one = tir.const(1.0, DTYPE)
    st = te.compute((n,), lambda i: red((one, X[i, j], one), axis=j), name="st")
    return [X], [te.compute((n,), lambda i: st[2][i], name="zc")]


def exp_centered_reference(x):
    x = x.astype(np.float64)
    return (np.exp(x - x.mean(axis=1, keepdims=True)).sum(axis=1),)


# ---------------------------------------------------------------------------
# instance normalisation of (batch, channel, length) data
# ---------------------------------------------------------------------------
def _signal():
    nb, ch, ln = te.var("nb"), te.var("ch"), te.var("L")
    return te.placeholder((nb, ch, ln), name="X", dtype=DTYPE)


def _instance_normalise(X, mean, ss):
    count = X.shape[2].astype(DTYPE)
    eps = tir.const(LN_EPS, DTYPE)
    return te.compute(
        X.shape,
        lambda b, c, k: (X[b, c, k] - mean[b, c]) / tir.sqrt(ss[b, c] / count + eps),
        name="Y",
    )


def instance_norm_unfused():
    X = _signal()
    nb, ch, ln = X.shape
    count = ln.astype(DTYPE)
    l1, l2 = te.reduce_axis((0, ln), "l"), te.reduce_axis((0, ln), "l")
    total = te.compute((nb, ch), lambda b, c: te.sum(X[b, c, l1], axis=l1), name="total")
    mean = te.compute((nb, ch), lambda b, c: total[b, c] / count, name="mean")
    ss = te.compute(
        (nb, ch),
        lambda b, c: te.sum((X[b, c, l2] - mean[b, c]) * (X[b, c, l2] - mean[b, c]), axis=l2),
        name="ss",
    )
    return [X], [_instance_normalise(X, mean, ss)]


def instance_norm_fused():
    X = _signal()
    nb, ch, ln = X.shape
    k = te.reduce_axis((0, ln), "l")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    st = te.compute((nb, ch), lambda b, c: pebay(2)((one, X[b, c, k], zero), axis=k), name="st")
    return [X], [_instance_normalise(X, st[1], st[2])]


def instance_norm_reference(x):
    x = x.astype(np.float64)
    mean = x.mean(axis=2, keepdims=True)
    return ((x - mean) / np.sqrt(x.var(axis=2, keepdims=True) + LN_EPS),)


# ---------------------------------------------------------------------------
# the mean of min-max scaled data
# ---------------------------------------------------------------------------
def minmax_mean_unfused():
    X = _rows()
    n, m = X.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    lo = te.compute((n,), lambda i: te.min(X[i, j1], axis=j1), name="lo")
    hi = te.compute((n,), lambda i: te.max(X[i, j2], axis=j2), name="hi")
    scaled = _sum(X, lambda i, j: (X[i, j] - lo[i]) / (hi[i] - lo[i]), "scaled")
    return [X], [te.compute((n,), lambda i: scaled[i] / m.astype(DTYPE), name="avg")]


def minmax_mean_fused():
    """The range and the plain sum: ``Σ (x - lo) / (hi - lo) = (Σ x - n lo) / (hi - lo)``."""
    X = _rows()
    n, m = X.shape
    count = m.astype(DTYPE)

    def merge(a, b):
        return (tir.min(a[0], b[0]), tir.max(a[1], b[1]), a[2] + b[2])

    def identity(t0, t1, t2):
        return (tir.max_value(t0), tir.min_value(t1), tir.const(0.0, t2))

    red = te.comm_reducer(merge, identity, name="range_sum")
    j = te.reduce_axis((0, m), "j")
    lo, hi, total = te.compute((n,), lambda i: red((X[i, j], X[i, j], X[i, j]), axis=j), name="st")
    return [X], [
        te.compute((n,), lambda i: (total[i] - count * lo[i]) / (hi[i] - lo[i]) / count, name="avg")
    ]


def minmax_mean_reference(x):
    x = x.astype(np.float64)
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return (((x - lo) / (hi - lo)).mean(axis=1),)


register(
    Operator(
        "weighted_variance",
        "shift/scale",
        "Σ w (x - mean_w)² / Σ w, weights positive",
        weighted_variance_unfused,
        weighted_variance_reference,
        weighted_variance_fused,
        states=3,
        positive=("W",),
    )
)
register(
    Operator(
        "skewness",
        "hoist",
        "Σ ((x - mean) / sqrt(m2))³ / n: the standard deviation leaves the third moment",
        skewness_unfused,
        skewness_reference,
        skewness_fused,
        states=4,
    )
)
register(
    Operator(
        "kurtosis",
        "hoist",
        "Σ ((x - mean) / sqrt(m2))⁴ / n: found, after a search over sets of five states",
        kurtosis_unfused,
        kurtosis_reference,
        kurtosis_fused,
        states=5,
        max_states=5,
        slow=True,
    )
)
register(
    Operator(
        "pearson",
        "closure",
        "Σ (x - mean_x)(y - mean_y) / sqrt(Σ (x - mean_x)² Σ (y - mean_y)²)",
        pearson_unfused,
        pearson_reference,
        pearson_fused,
        states=6,
        max_states=6,
    )
)
register(
    Operator(
        "exp_centered",
        "closure",
        "Σ exp(x - mean)",
        exp_centered_unfused,
        exp_centered_reference,
        exp_centered_fused,
        states=3,
    )
)
register(
    Operator(
        "instance_norm",
        "closure",
        "(x - mean) / sqrt(var + eps) along the length of (batch, channel, length) data",
        instance_norm_unfused,
        instance_norm_reference,
        instance_norm_fused,
        states=3,
    )
)
register(
    Operator(
        "minmax_mean",
        "hoist",
        "Σ (x - min x) / (max x - min x) / n",
        minmax_mean_unfused,
        minmax_mean_reference,
        minmax_mean_fused,
        states=4,
    )
)
