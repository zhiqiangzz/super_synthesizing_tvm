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
"""Central moments: a mean, then sums of powers of the deviation from it.

Unfused, the mean takes one pass and every moment about it another. Fused
they are the streaming updates of Welford and Pébay: count, mean and the
moments themselves as states, merged around the difference of the two
means. Unlike the softmax family this needs states the program never
computes: the count, and every lower moment (the third needs the second).
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register

LN_EPS = 1.0e-5


def _rows(name: str = "X", like=None):
    shape = (te.var("n"), te.var("m")) if like is None else like.shape
    return te.placeholder(shape, name=name, dtype=DTYPE)


def _mean(X, name: str = "mean"):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j], axis=j), name=f"total_{name}")
    return te.compute((n,), lambda i: total[i] / m.astype(DTYPE), name=name)


def _central(X, mean, power: int, name: str):
    """``Σ_j (x - mean)^power``."""
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    k = tir.const(float(power), DTYPE)
    return te.compute((n,), lambda i: te.sum(tir.power(X[i, j] - mean[i], k), axis=j), name=name)


def _zeros(k: int):
    return lambda *dtypes: tuple(tir.const(0.0, t) for t in dtypes[:k])


def pebay(order: int):
    """``(count, mean, M2 .. M_order)`` with Pébay's pairwise merge."""

    def merge(a, b):
        na, nb = a[0], b[0]
        n = na + nb
        d = b[1] - a[1]
        mean = a[1] + d * nb / n
        m2 = a[2] + b[2] + d * d * na * nb / n
        out = [n, mean, m2]
        if order >= 3:
            m3 = (
                a[3]
                + b[3]
                + d * d * d * na * nb * (na - nb) / (n * n)
                + tir.const(3.0, DTYPE) * d * (na * b[2] - nb * a[2]) / n
            )
            out.append(m3)
        if order >= 4:
            m4 = (
                a[4]
                + b[4]
                + d * d * d * d * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
                + tir.const(6.0, DTYPE) * d * d * (na * na * b[2] + nb * nb * a[2]) / (n * n)
                + tir.const(4.0, DTYPE) * d * (na * b[3] - nb * a[3]) / n
            )
            out.append(m4)
        return tuple(out)

    return te.comm_reducer(merge, _zeros(order + 1), name="pebay")


def _streamed(X, order: int):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    red = pebay(order)
    return te.compute(
        (n,), lambda i: red((one, X[i, j], *([zero] * (order - 1))), axis=j), name="st"
    )


def _np_central(x, power: int):
    x = x.astype(np.float64)
    return ((x - x.mean(axis=1, keepdims=True)) ** power).sum(axis=1)


# ---------------------------------------------------------------------------
# variance, layer normalisation
# ---------------------------------------------------------------------------
def variance_unfused():
    X = _rows()
    n, m = X.shape
    ss = _central(X, _mean(X), 2, "ss")
    return [X], [te.compute((n,), lambda i: ss[i] / m.astype(DTYPE), name="var")]


def variance_fused():
    X = _rows()
    n, m = X.shape
    st = _streamed(X, 2)
    return [X], [te.compute((n,), lambda i: st[2][i] / m.astype(DTYPE), name="var")]


def variance_reference(x):
    return (_np_central(x, 2) / x.shape[1],)


def _normalise(X, mean, ss):
    m = X.shape[1].astype(DTYPE)
    eps = tir.const(LN_EPS, DTYPE)
    return te.compute(
        X.shape, lambda i, j: (X[i, j] - mean[i]) / tir.sqrt(ss[i] / m + eps), name="Y"
    )


def layernorm_unfused():
    X = _rows()
    mean = _mean(X)
    return [X], [_normalise(X, mean, _central(X, mean, 2, "ss"))]


def layernorm_fused():
    X = _rows()
    st = _streamed(X, 2)
    return [X], [_normalise(X, st[1], st[2])]


def layernorm_reference(x):
    x = x.astype(np.float64)
    mean = x.mean(axis=1, keepdims=True)
    return ((x - mean) / np.sqrt(x.var(axis=1, keepdims=True) + LN_EPS),)


# ---------------------------------------------------------------------------
# covariance of two rows
# ---------------------------------------------------------------------------
def covariance_unfused():
    X = _rows("X")
    Y = _rows("Y", like=X)
    n, m = X.shape
    mx, my = _mean(X, "mean_x"), _mean(Y, "mean_y")
    j = te.reduce_axis((0, m), "j")
    co = te.compute(
        (n,), lambda i: te.sum((X[i, j] - mx[i]) * (Y[i, j] - my[i]), axis=j), name="co"
    )
    return [X, Y], [te.compute((n,), lambda i: co[i] / m.astype(DTYPE), name="cov")]


def covariance_fused():
    X = _rows("X")
    Y = _rows("Y", like=X)
    n, m = X.shape

    def merge(a, b):
        cnt = a[0] + b[0]
        dx, dy = b[1] - a[1], b[2] - a[2]
        return (
            cnt,
            a[1] + dx * b[0] / cnt,
            a[2] + dy * b[0] / cnt,
            a[3] + b[3] + dx * dy * a[0] * b[0] / cnt,
        )

    red = te.comm_reducer(merge, _zeros(4), name="co_welford")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    st = te.compute((n,), lambda i: red((one, X[i, j], Y[i, j], zero), axis=j), name="st")
    return [X, Y], [te.compute((n,), lambda i: st[3][i] / m.astype(DTYPE), name="cov")]


def covariance_reference(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    dx, dy = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    return ((dx * dy).sum(axis=1) / x.shape[1],)


# ---------------------------------------------------------------------------
# higher central moments
# ---------------------------------------------------------------------------
def _moment_unfused(power: int):
    X = _rows()
    n, m = X.shape
    s = _central(X, _mean(X), power, f"s{power}")
    return [X], [te.compute((n,), lambda i: s[i] / m.astype(DTYPE), name=f"m{power}")]


def _moment_fused(power: int):
    X = _rows()
    n, m = X.shape
    st = _streamed(X, power)
    return [X], [te.compute((n,), lambda i: st[power][i] / m.astype(DTYPE), name=f"m{power}")]


def _moment_reference(power: int):
    return lambda x: (_np_central(x, power) / x.shape[1],)


# ---------------------------------------------------------------------------
# outside what a finite lifting reaches
# ---------------------------------------------------------------------------
def mad_unfused():
    """Mean absolute deviation: the mean sits under ``|.|`` and cannot be re-based."""
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")
    dev = te.compute(
        (n,), lambda i: te.sum(tir.max(X[i, j] - mean[i], mean[i] - X[i, j]), axis=j), name="dev"
    )
    return [X], [te.compute((n,), lambda i: dev[i] / m.astype(DTYPE), name="mad")]


def mad_reference(x):
    x = x.astype(np.float64)
    return (np.abs(x - x.mean(axis=1, keepdims=True)).mean(axis=1),)


def cov_matrix_unfused():
    """``C[a, b] = Σ_j (x_aj - mean_a)(x_bj - mean_b)``: the mean is read at both indices."""
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")
    C = te.compute(
        (n, n), lambda a, b: te.sum((X[a, j] - mean[a]) * (X[b, j] - mean[b]), axis=j), name="C"
    )
    return [X], [C]


def cov_matrix_reference(x):
    d = x.astype(np.float64) - x.astype(np.float64).mean(axis=1, keepdims=True)
    return (d @ d.T,)


register(
    Operator(
        "variance",
        "closure",
        "Σ (x - mean)² / n",
        variance_unfused,
        variance_reference,
        variance_fused,
        states=3,
    )
)
register(
    Operator(
        "layernorm",
        "closure",
        "(x - mean) / sqrt(var + eps): mean and variance in one pass",
        layernorm_unfused,
        layernorm_reference,
        layernorm_fused,
        states=3,
    )
)
register(
    Operator(
        "covariance",
        "closure",
        "Σ (x - mean_x)(y - mean_y) / n",
        covariance_unfused,
        covariance_reference,
        covariance_fused,
        states=4,
    )
)
register(
    Operator(
        "moment3",
        "closure",
        "Σ (x - mean)³ / n: needs the second moment as a state",
        lambda: _moment_unfused(3),
        _moment_reference(3),
        lambda: _moment_fused(3),
        states=4,
    )
)
register(
    Operator(
        "moment4",
        "closure",
        "Σ (x - mean)⁴ / n: needs the second and the third",
        lambda: _moment_unfused(4),
        _moment_reference(4),
        lambda: _moment_fused(4),
        states=5,
        max_states=5,
    )
)
register(
    Operator(
        "mean_abs_deviation",
        "negative",
        "Σ |x - mean| / n",
        mad_unfused,
        mad_reference,
        fusible=False,
        reason="dev reads mean from inside its reduction and neither",
        exists=False,
    )
)
register(
    Operator(
        "moment5",
        "negative",
        "Σ (x - mean)⁵ / n: beyond the polynomial expansion (degree 4)",
        lambda: _moment_unfused(5),
        _moment_reference(5),
        fusible=False,
        reason="s5 cannot be re-based around mean",
    )
)
register(
    Operator(
        "cov_matrix",
        "negative",
        "Σ_j (x_aj - mean_a)(x_bj - mean_b)",
        cov_matrix_unfused,
        cov_matrix_reference,
        fusible=False,
        reason="read at two different indices of C",
        exists=None,
    )
)
