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
"""Log-domain Sinkhorn: each half-step is a log-sum-exp over the cost matrix.

    f_i = -eps * LSE_j((g_j - C_ij) / eps + log b_j)
    g_j = -eps * LSE_i((f_i - C_ij) / eps + log a_i)

Unfused, a half-step walks its axis twice (maximum, then the shifted sum)
over a materialised score matrix. Fused it is an online log-sum-exp reading
the cost directly. One iteration is two chains, the second over the other
axis and fed by the result of the first.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register
from .softmax import online_softmax

EPS = 0.5


def _inputs(square: bool = False):
    n = te.var("n")
    m = n if square else te.var("m")
    dim = te.var("dim")
    x = te.placeholder((n, dim), name="x", dtype=DTYPE)
    y = te.placeholder((m, dim), name="y", dtype=DTYPE)
    a = te.placeholder((n,), name="a", dtype=DTYPE)
    b = te.placeholder((m,), name="b", dtype=DTYPE)
    g = te.placeholder((m,), name="g", dtype=DTYPE)
    return x, y, a, b, g


def _cost(x, y):
    """``C[i, j] = |x_i - y_j|²``, materialised."""
    n, dim = x.shape
    m = y.shape[0]
    k = te.reduce_axis((0, dim), name="k")
    return te.compute(
        (n, m),
        lambda i, j: te.sum((x[i, k] - y[j, k]) * (x[i, k] - y[j, k]), axis=k),
        name="C",
    )


def _f_scores(C, b, g):
    inv = tir.const(1.0 / EPS, DTYPE)
    return te.compute(C.shape, lambda i, j: (g[j] - C[i, j]) * inv + tir.log(b[j]), name="S")


def _g_scores(C, a, f):
    n, m = C.shape
    inv = tir.const(1.0 / EPS, DTYPE)
    return te.compute((m, n), lambda j, i: (f[i] - C[i, j]) * inv + tir.log(a[i]), name="S2")


def _lse_two_pass(S, name: str, axis_name: str):
    rows, cols = S.shape
    r1 = te.reduce_axis((0, cols), name=axis_name)
    mx = te.compute((rows,), lambda r: te.max(S[r, r1], axis=r1), name=f"{name}_max")
    r2 = te.reduce_axis((0, cols), name=axis_name)
    den = te.compute(
        (rows,), lambda r: te.sum(tir.exp(S[r, r2] - mx[r]), axis=r2), name=f"{name}_sumexp"
    )
    return mx, den


def _lse_one_pass(S, name: str, axis_name: str):
    rows, cols = S.shape
    r = te.reduce_axis((0, cols), name=axis_name)
    red = online_softmax(1)
    one = tir.const(1.0, DTYPE)
    return te.compute((rows,), lambda p: red((S[p, r], one), axis=r), name=f"{name}_lse")


def _potential(mx, den, name: str):
    eps = tir.const(EPS, DTYPE)
    return te.compute(mx.shape, lambda r: -eps * (mx[r] + tir.log(den[r])), name=name)


def _half(lse):
    x, y, _, b, g = _inputs()
    C = _cost(x, y)
    f = _potential(*lse(_f_scores(C, b, g), "S", "j"), "f")
    return [x, y, b, g], [f]


def _iteration(lse, square: bool = False):
    x, y, a, b, g = _inputs(square)
    C = _cost(x, y)
    f = _potential(*lse(_f_scores(C, b, g), "S", "j"), "f")
    g_new = _potential(*lse(_g_scores(C, a, f), "S2", "i"), "g_new")
    return [x, y, a, b, g], [f, g_new]


def _np_lse(s):
    mx = s.max(axis=1)
    return mx + np.log(np.exp(s - mx[:, None]).sum(axis=1))


def _np_cost(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def half_reference(x, y, b, g):
    c = _np_cost(x, y)
    s = (g.astype(np.float64)[None, :] - c) / EPS + np.log(b.astype(np.float64))[None, :]
    return (-EPS * _np_lse(s),)


def iteration_reference(x, y, a, b, g):
    c = _np_cost(x, y)
    (f,) = half_reference(x, y, b, g)
    s2 = (f[None, :] - c.T) / EPS + np.log(a.astype(np.float64))[None, :]
    return f, -EPS * _np_lse(s2)


register(
    Operator(
        "sinkhorn_half",
        "shift/scale",
        "f = -eps LSE_j((g - C) / eps + log b)",
        lambda: _half(_lse_two_pass),
        half_reference,
        lambda: _half(_lse_one_pass),
        states=2,
    )
)
register(
    Operator(
        "sinkhorn_iteration",
        "shift/scale",
        "f-update then g-update: two chains over the two axes of C",
        lambda: _iteration(_lse_two_pass),
        iteration_reference,
        lambda: _iteration(_lse_one_pass),
        states=2,
    )
)
register(
    Operator(
        "sinkhorn_square",
        "shift/scale",
        "the same iteration between point clouds of equal size",
        lambda: _iteration(_lse_two_pass, square=True),
        iteration_reference,
        lambda: _iteration(_lse_one_pass, square=True),
        states=2,
    )
)
