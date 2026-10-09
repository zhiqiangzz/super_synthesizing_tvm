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
"""Chains that end in a maximum or a minimum.

A sum that waits for an earlier result is fused by splitting its body into
an element part and a context part. A maximum cannot be split that way; it
is fused when the element that attains it is the same whatever the context
turns out to be -- an extreme of the data, which can be kept without
knowing the context::

    max_j (x_j - mean) / sigma  =  (max_j x_j - mean) / sigma
    max_j |x_j - mean|          =  max(max_j x_j - mean, mean - min_j x_j)

A sum of absolute deviations from the maximum is the border case the other
way: ``|x - max x|`` does not split for an arbitrary context, but a maximum
is never below its elements, where it is ``max x - x``. A single pass
exists; the derivation does not use that fact and leaves the program alone.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register


def _rows(name: str = "X"):
    return te.placeholder((te.var("n"), te.var("m")), name=name, dtype=DTYPE)


def _mean(X):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j], axis=j), name="total")
    return te.compute((n,), lambda i: total[i] / m.astype(DTYPE), name="mean")


# ---------------------------------------------------------------------------
# the largest standard score
# ---------------------------------------------------------------------------
def zscore_max_unfused():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j1] - mean[i]) * (X[i, j1] - mean[i]), axis=j1), name="ss"
    )
    sigma = te.compute((n,), lambda i: tir.sqrt(ss[i] / m.astype(DTYPE)), name="sigma")
    return [X], [
        te.compute((n,), lambda i: te.max((X[i, j2] - mean[i]) / sigma[i], axis=j2), name="zmax")
    ]


def zscore_max_fused():
    """Welford's update and a running maximum; the score is taken at the end."""
    X = _rows()
    n, m = X.shape

    def merge(a, b):
        cnt = a[0] + b[0]
        d = b[1] - a[1]
        return (
            cnt,
            a[1] + d * b[0] / cnt,
            a[2] + b[2] + d * d * a[0] * b[0] / cnt,
            tir.max(a[3], b[3]),
        )

    def identity(t0, t1, t2, t3):
        return (*(tir.const(0.0, t) for t in (t0, t1, t2)), tir.min_value(t3))

    red = te.comm_reducer(merge, identity, name="moments_max")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    _, mean, ss, mx = te.compute(
        (n,), lambda i: red((one, X[i, j], zero, X[i, j]), axis=j), name="st"
    )
    return [X], [
        te.compute(
            (n,), lambda i: (mx[i] - mean[i]) / tir.sqrt(ss[i] / m.astype(DTYPE)), name="zmax"
        )
    ]


def zscore_max_reference(x):
    x = x.astype(np.float64)
    return (((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)).max(axis=1),)


# ---------------------------------------------------------------------------
# the largest absolute deviation from the mean
# ---------------------------------------------------------------------------
def max_abs_deviation_unfused():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")
    return [X], [
        te.compute((n,), lambda i: te.max(tir.abs(X[i, j] - mean[i]), axis=j), name="dmax")
    ]


def max_abs_deviation_fused():
    """The sum and both extremes: the farthest point from the mean is one of the two."""
    X = _rows()
    n, m = X.shape

    def merge(a, b):
        return (a[0] + b[0], tir.max(a[1], b[1]), tir.min(a[2], b[2]))

    def identity(t0, t1, t2):
        return (tir.const(0.0, t0), tir.min_value(t1), tir.max_value(t2))

    red = te.comm_reducer(merge, identity, name="sum_range")
    j = te.reduce_axis((0, m), "j")
    total, hi, lo = te.compute((n,), lambda i: red((X[i, j], X[i, j], X[i, j]), axis=j), name="st")

    def dmax(i):
        mean = total[i] / m.astype(DTYPE)
        return tir.max(hi[i] - mean, mean - lo[i])

    return [X], [te.compute((n,), dmax, name="dmax")]


def max_abs_deviation_reference(x):
    x = x.astype(np.float64)
    return (np.abs(x - x.mean(axis=1, keepdims=True)).max(axis=1),)


# ---------------------------------------------------------------------------
# Σ |x - max x|: a single pass exists, and is not derived
# ---------------------------------------------------------------------------
def abs_from_max_unfused():
    X = _rows()
    n, m = X.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(X[i, j1], axis=j1), name="mx")
    return [X], [te.compute((n,), lambda i: te.sum(tir.abs(X[i, j2] - mx[i]), axis=j2), name="gap")]


def abs_from_max_reference(x):
    x = x.astype(np.float64)
    return (np.abs(x - x.max(axis=1, keepdims=True)).sum(axis=1),)


register(
    Operator(
        "zscore_max",
        "extreme",
        "max (x - mean) / sigma: the standard deviation is positive, the largest x wins",
        zscore_max_unfused,
        zscore_max_reference,
        zscore_max_fused,
        states=4,
    )
)
register(
    Operator(
        "max_abs_deviation",
        "extreme",
        "max |x - mean|: the largest or the smallest x",
        max_abs_deviation_unfused,
        max_abs_deviation_reference,
        max_abs_deviation_fused,
        states=4,
    )
)
register(
    Operator(
        "abs_from_max",
        "negative",
        "Σ |x - max x| = n max x - Σ x, which takes knowing that x <= max x",
        abs_from_max_unfused,
        abs_from_max_reference,
        fusible=False,
        reason="a finite lifting exists by the probes, none was derived",
    )
)
