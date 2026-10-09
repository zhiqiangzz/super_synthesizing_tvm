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
"""Backward passes: the forward statistics, then sums of the incoming gradient about them.

The gradient of a layer normalisation needs ``Σ g`` and ``Σ g x̂`` with
``x̂ = (x - mean) / sigma`` -- sums that wait for the mean and the variance
of the forward pass, which is recomputed here. Unfused that is four passes
over the features; fused, the gradient sums ride along with Welford's
update, centred on the running mean.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register
from .moments import LN_EPS


def _rows(name: str = "X", like=None):
    shape = (te.var("n"), te.var("m")) if like is None else like.shape
    return te.placeholder(shape, name=name, dtype=DTYPE)


def _sum(X, f, name: str):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.sum(f(i, j), axis=j), name=name)


def layernorm_backward_unfused():
    X = _rows("X")
    G = _rows("G", like=X)
    n, m = X.shape
    count = m.astype(DTYPE)
    eps = tir.const(LN_EPS, DTYPE)
    total = _sum(X, lambda i, j: X[i, j], "total")
    mean = te.compute((n,), lambda i: total[i] / count, name="mean")
    ss = _sum(X, lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    sigma = te.compute((n,), lambda i: tir.sqrt(ss[i] / count + eps), name="sigma")
    xh = te.compute(X.shape, lambda i, k: (X[i, k] - mean[i]) / sigma[i], name="xh")
    sg = _sum(X, lambda i, j: G[i, j], "sg")
    sgx = _sum(X, lambda i, j: G[i, j] * xh[i, j], "sgx")
    out = te.compute(
        X.shape,
        lambda i, k: (G[i, k] - sg[i] / count - xh[i, k] * sgx[i] / count) / sigma[i],
        name="dx",
    )
    return [X, G], [out]


def layernorm_backward_fused():
    """``(count, mean, ss, Σ g, Σ g (x - mean))``: the last is re-centred like ``ss``."""
    X = _rows("X")
    G = _rows("G", like=X)
    n, m = X.shape
    count = m.astype(DTYPE)
    eps = tir.const(LN_EPS, DTYPE)

    def merge(a, b):
        cnt = a[0] + b[0]
        d = b[1] - a[1]
        mean = a[1] + d * b[0] / cnt
        return (
            cnt,
            mean,
            a[2] + b[2] + d * d * a[0] * b[0] / cnt,
            a[3] + b[3],
            a[4] - (mean - a[1]) * a[3] + b[4] - (mean - b[1]) * b[3],
        )

    def identity(*dtypes):
        return tuple(tir.const(0.0, t) for t in dtypes[:5])

    red = te.comm_reducer(merge, identity, name="ln_backward")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    _, mean, ss, sg, sgc = te.compute(
        (n,), lambda i: red((one, X[i, j], zero, G[i, j], zero), axis=j), name="st"
    )

    def dx(i, k):
        sigma = tir.sqrt(ss[i] / count + eps)
        xh = (X[i, k] - mean[i]) / sigma
        return (G[i, k] - sg[i] / count - xh * (sgc[i] / sigma) / count) / sigma

    return [X, G], [te.compute(X.shape, dx, name="dx")]


def layernorm_backward_reference(x, g):
    x, g = x.astype(np.float64), g.astype(np.float64)
    mean = x.mean(axis=1, keepdims=True)
    sigma = np.sqrt(x.var(axis=1, keepdims=True) + LN_EPS)
    xh = (x - mean) / sigma
    a = g.mean(axis=1, keepdims=True)
    b = (g * xh).mean(axis=1, keepdims=True)
    return ((g - a - xh * b) / sigma,)


register(
    Operator(
        "layernorm_backward",
        "hoist",
        "(g - mean(g) - x̂ mean(g x̂)) / sigma with x̂ = (x - mean) / sigma",
        layernorm_backward_unfused,
        layernorm_backward_reference,
        layernorm_backward_fused,
        states=5,
        max_states=5,
    )
)
