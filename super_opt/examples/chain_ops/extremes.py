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

The same holds under a function that grows with its argument, which commutes
with the maximum, and under an even power, which grows with the absolute
value::

    max_j exp(l_j - mean)       =  exp(max_j l_j - mean)
    max_j (x_j - mean)²         =  max(max_j x_j - mean, mean - min_j x_j)²

Three programs mark where this ends; for none of them do the rules show a
single pass. A square under a *minimum* is smallest at the element nearest
the mean, which is no extreme of the data. A sum of two growing functions,
``(x - mean)³ + (x - mean)``, is still largest at the largest element, so a
single pass exists; but it is not one function of one reduction, and the
rules do not see it. A sum of absolute deviations from the maximum does not
split for an arbitrary context, while a maximum is never below its
elements, where ``|x - max x|`` is ``max x - x``: again a single pass
exists, and the rules, which do not know that, leave the program alone.
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


def _sum_and_range(X):
    """``(Σ_j x, max_j x, min_j x)`` in one pass."""
    n, m = X.shape

    def merge(a, b):
        return (a[0] + b[0], tir.max(a[1], b[1]), tir.min(a[2], b[2]))

    def identity(t0, t1, t2):
        return (tir.const(0.0, t0), tir.min_value(t1), tir.max_value(t2))

    red = te.comm_reducer(merge, identity, name="sum_range")
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: red((X[i, j], X[i, j], X[i, j]), axis=j), name="st")


def max_abs_deviation_fused():
    """The sum and both extremes: the farthest point from the mean is one of the two."""
    X = _rows()
    n, m = X.shape
    total, hi, lo = _sum_and_range(X)

    def dmax(i):
        mean = total[i] / m.astype(DTYPE)
        return tir.max(hi[i] - mean, mean - lo[i])

    return [X], [te.compute((n,), dmax, name="dmax")]


def max_abs_deviation_reference(x):
    x = x.astype(np.float64)
    return (np.abs(x - x.mean(axis=1, keepdims=True)).max(axis=1),)


# ---------------------------------------------------------------------------
# the peak power of a signal with its mean removed, and its ratio to the average
# ---------------------------------------------------------------------------
def _square(v):
    return v * v


def peak_power_unfused():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")
    return [X], [
        te.compute((n,), lambda i: te.max(_square(X[i, j] - mean[i]), axis=j), name="peak")
    ]


def peak_power_fused():
    """The sum and both extremes: the largest square is at one of the two."""
    X = _rows()
    n, m = X.shape
    total, hi, lo = _sum_and_range(X)

    def peak(i):
        mean = total[i] / m.astype(DTYPE)
        return _square(tir.max(hi[i] - mean, mean - lo[i]))

    return [X], [te.compute((n,), peak, name="peak")]


def peak_power_reference(x):
    x = x.astype(np.float64)
    return (((x - x.mean(axis=1, keepdims=True)) ** 2).max(axis=1),)


def papr_unfused():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    ss = te.compute((n,), lambda i: te.sum(_square(X[i, j1] - mean[i]), axis=j1), name="ss")
    peak = te.compute((n,), lambda i: te.max(_square(X[i, j2] - mean[i]), axis=j2), name="peak")
    return [X], [te.compute((n,), lambda i: peak[i] / (ss[i] / m.astype(DTYPE)), name="papr")]


def papr_fused():
    """Welford's update and both extremes."""
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
            tir.min(a[4], b[4]),
        )

    def identity(t0, t1, t2, t3, t4):
        zeros = (tir.const(0.0, t) for t in (t0, t1, t2))
        return (*zeros, tir.min_value(t3), tir.max_value(t4))

    red = te.comm_reducer(merge, identity, name="moments_range")
    j = te.reduce_axis((0, m), "j")
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)
    _, mean, ss, hi, lo = te.compute(
        (n,), lambda i: red((one, X[i, j], zero, X[i, j], X[i, j]), axis=j), name="st"
    )

    def papr(i):
        return _square(tir.max(hi[i] - mean[i], mean[i] - lo[i])) / (ss[i] / m.astype(DTYPE))

    return [X], [te.compute((n,), papr, name="papr")]


def papr_reference(x):
    d = x.astype(np.float64)
    d = (d - d.mean(axis=1, keepdims=True)) ** 2
    return (d.max(axis=1) / d.mean(axis=1),)


# ---------------------------------------------------------------------------
# the largest value over the geometric mean, from logarithms
# ---------------------------------------------------------------------------
def peak_to_geomean_unfused():
    L = _rows("L")
    n, m = L.shape
    mean = _mean(L)
    j = te.reduce_axis((0, m), "j")
    return [L], [
        te.compute((n,), lambda i: te.max(tir.exp(L[i, j] - mean[i]), axis=j), name="ratio")
    ]


def peak_to_geomean_fused():
    """The sum and a running maximum; the exponential is taken once, at the end."""
    L = _rows("L")
    n, m = L.shape

    def merge(a, b):
        return (a[0] + b[0], tir.max(a[1], b[1]))

    def identity(t0, t1):
        return (tir.const(0.0, t0), tir.min_value(t1))

    red = te.comm_reducer(merge, identity, name="sum_max")
    j = te.reduce_axis((0, m), "j")
    total, hi = te.compute((n,), lambda i: red((L[i, j], L[i, j]), axis=j), name="st")
    return [L], [
        te.compute((n,), lambda i: tir.exp(hi[i] - total[i] / m.astype(DTYPE)), name="ratio")
    ]


def peak_to_geomean_reference(logs):
    logs = logs.astype(np.float64)
    return (np.exp(logs - logs.mean(axis=1, keepdims=True)).max(axis=1),)


# ---------------------------------------------------------------------------
# where it ends: a square under a minimum, a sum of growing functions
# ---------------------------------------------------------------------------
def min_sq_deviation_unfused():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")
    return [X], [
        te.compute((n,), lambda i: te.min(_square(X[i, j] - mean[i]), axis=j), name="near")
    ]


def min_sq_deviation_reference(x):
    x = x.astype(np.float64)
    return (((x - x.mean(axis=1, keepdims=True)) ** 2).min(axis=1),)


def monotone_sum_unfused():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")

    def body(i):
        d = X[i, j] - mean[i]
        return te.max(d * d * d + d, axis=j)

    return [X], [te.compute((n,), body, name="top")]


def monotone_sum_reference(x):
    d = x.astype(np.float64)
    d = d - d.mean(axis=1, keepdims=True)
    return ((d**3 + d).max(axis=1),)


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
        "peak_power",
        "extreme",
        "max (x - mean)²: a square grows with |x - mean|, the largest or the smallest x",
        peak_power_unfused,
        peak_power_reference,
        peak_power_fused,
        states=4,
    )
)
register(
    Operator(
        "papr",
        "extreme",
        "max (x - mean)² / mean of (x - mean)²: peak-to-average power, the mean removed",
        papr_unfused,
        papr_reference,
        papr_fused,
        states=5,
    )
)
register(
    Operator(
        "peak_to_geomean",
        "extreme",
        "max exp(l - mean l): exp grows with its argument, the largest l wins",
        peak_to_geomean_unfused,
        peak_to_geomean_reference,
        peak_to_geomean_fused,
        states=3,
    )
)
register(
    Operator(
        "min_sq_deviation",
        "negative",
        "min (x - mean)²: attained at the element nearest the mean, which is no extreme",
        min_sq_deviation_unfused,
        min_sq_deviation_reference,
        fusible=False,
        reason="no single pass is known: near reads mean inside",
        judged=False,
    )
)
register(
    Operator(
        "monotone_sum",
        "negative",
        "max ((x - mean)³ + (x - mean)): largest at the largest x, but not one function",
        monotone_sum_unfused,
        monotone_sum_reference,
        fusible=False,
        reason="no single pass is known: top reads mean inside",
        judged=False,
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
        reason="no single pass is known: gap reads mx inside",
        judged=False,
    )
)
