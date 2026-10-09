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
"""Vector kernels of numerical linear algebra: an inner product, then a norm that uses it.

The remainder of a projection (one step of Gram-Schmidt), the squared norm
of a Householder-like update, the cosine of two vectors. Unfused, the
coefficient takes one pass and the norm of what it leaves another. Fused
they are a handful of inner products taken together -- which is also where
their accuracy ends: the remainder is a difference of sums that cancels
when the two vectors are nearly parallel.

The Euclidean norm with scaling is here as a program that is *not* fused:
a single pass exists, but the plain sum of squares it would be made of
overflows where the scaled original does not.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register


def _rows(name: str = "X", like=None):
    shape = (te.var("n"), te.var("m")) if like is None else like.shape
    return te.placeholder(shape, name=name, dtype=DTYPE)


def _sum(X, f, name: str):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.sum(f(i, j), axis=j), name=name)


def _inner_products(X, leaves, name: str = "ip"):
    """``Σ_j leaf_k(i, j)`` for every leaf, in one reduction."""
    n, m = X.shape
    k = len(leaves)

    def merge(a, b):
        return tuple(a[t] + b[t] for t in range(k))

    def identity(*dtypes):
        return tuple(tir.const(0.0, t) for t in dtypes[:k])

    red = te.comm_reducer(merge, identity, name="sums")
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: red(tuple(f(i, j) for f in leaves), axis=j), name=name)


# ---------------------------------------------------------------------------
# what is left of a after its component along q is taken out
# ---------------------------------------------------------------------------
def _remainder(Q, A, r):
    return _sum(Q, lambda i, j: (A[i, j] - r[i] * Q[i, j]) * (A[i, j] - r[i] * Q[i, j]), "rem")


def _qa_aa_qq(Q, A):
    return _inner_products(
        Q,
        [
            lambda i, j: Q[i, j] * A[i, j],
            lambda i, j: A[i, j] * A[i, j],
            lambda i, j: Q[i, j] * Q[i, j],
        ],
    )


def projection_unfused():
    """``q`` of unit length is assumed, as in Gram-Schmidt: ``r = q . a``."""
    Q = _rows("Q")
    A = _rows("A", like=Q)
    r = _sum(Q, lambda i, j: Q[i, j] * A[i, j], "r")
    return [Q, A], [_remainder(Q, A, r)]


def projection_fused():
    """``Σ (a - r q)² = Σ a² - 2 r Σ q a + r² Σ q²`` with ``r = Σ q a``."""
    Q = _rows("Q")
    A = _rows("A", like=Q)
    qa, aa, qq = _qa_aa_qq(Q, A)
    two = tir.const(2.0, DTYPE)
    return [Q, A], [
        te.compute(
            (Q.shape[0],),
            lambda i: aa[i] - two * qa[i] * qa[i] + qa[i] * qa[i] * qq[i],
            name="rem",
        )
    ]


def projection_reference(q, a):
    q, a = q.astype(np.float64), a.astype(np.float64)
    r = (q * a).sum(axis=1, keepdims=True)
    return (((a - r * q) ** 2).sum(axis=1),)


def normalised_projection_unfused():
    Q = _rows("Q")
    A = _rows("A", like=Q)
    qa = _sum(Q, lambda i, j: Q[i, j] * A[i, j], "qa")
    qq = _sum(Q, lambda i, j: Q[i, j] * Q[i, j], "qq")
    r = te.compute((Q.shape[0],), lambda i: qa[i] / qq[i], name="r")
    return [Q, A], [_remainder(Q, A, r)]


def normalised_projection_fused():
    """``Σ (a - r q)² = Σ a² - (Σ q a)² / Σ q²`` with ``r = Σ q a / Σ q²``."""
    Q = _rows("Q")
    A = _rows("A", like=Q)
    qa, aa, qq = _qa_aa_qq(Q, A)
    return [Q, A], [te.compute((Q.shape[0],), lambda i: aa[i] - qa[i] * qa[i] / qq[i], name="rem")]


def normalised_projection_reference(q, a):
    q, a = q.astype(np.float64), a.astype(np.float64)
    r = (q * a).sum(axis=1, keepdims=True) / (q * q).sum(axis=1, keepdims=True)
    return (((a - r * q) ** 2).sum(axis=1),)


# ---------------------------------------------------------------------------
# Σ (x + ‖x‖ e)²: the squared norm of a Householder-like vector
# ---------------------------------------------------------------------------
def householder_unfused():
    X = _rows("X")
    E = _rows("E", like=X)
    ssq = _sum(X, lambda i, j: X[i, j] * X[i, j], "ssq")
    nrm = te.compute((X.shape[0],), lambda i: tir.sqrt(ssq[i]), name="nrm")
    vv = _sum(X, lambda i, j: (X[i, j] + nrm[i] * E[i, j]) * (X[i, j] + nrm[i] * E[i, j]), "vv")
    return [X, E], [vv]


def householder_fused():
    X = _rows("X")
    E = _rows("E", like=X)
    xx, xe, ee = _inner_products(
        X,
        [
            lambda i, j: X[i, j] * X[i, j],
            lambda i, j: X[i, j] * E[i, j],
            lambda i, j: E[i, j] * E[i, j],
        ],
    )
    two = tir.const(2.0, DTYPE)
    return [X, E], [
        te.compute(
            (X.shape[0],),
            lambda i: xx[i] + two * tir.sqrt(xx[i]) * xe[i] + xx[i] * ee[i],
            name="vv",
        )
    ]


def householder_reference(x, e):
    x, e = x.astype(np.float64), e.astype(np.float64)
    nrm = np.sqrt((x * x).sum(axis=1, keepdims=True))
    return (((x + nrm * e) ** 2).sum(axis=1),)


# ---------------------------------------------------------------------------
# cosine of two vectors, each normalised before the product
# ---------------------------------------------------------------------------
def cosine_unfused():
    X = _rows("X")
    Y = _rows("Y", like=X)
    n = X.shape[0]
    sxx = _sum(X, lambda i, j: X[i, j] * X[i, j], "sxx")
    syy = _sum(X, lambda i, j: Y[i, j] * Y[i, j], "syy")
    nx = te.compute((n,), lambda i: tir.sqrt(sxx[i]), name="nx")
    ny = te.compute((n,), lambda i: tir.sqrt(syy[i]), name="ny")
    return [X, Y], [_sum(X, lambda i, j: (X[i, j] / nx[i]) * (Y[i, j] / ny[i]), "cos")]


def cosine_fused():
    X = _rows("X")
    Y = _rows("Y", like=X)
    xx, yy, xy = _inner_products(
        X,
        [
            lambda i, j: X[i, j] * X[i, j],
            lambda i, j: Y[i, j] * Y[i, j],
            lambda i, j: X[i, j] * Y[i, j],
        ],
    )
    return [X, Y], [
        te.compute((X.shape[0],), lambda i: xy[i] / (tir.sqrt(xx[i]) * tir.sqrt(yy[i])), name="cos")
    ]


def cosine_reference(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    nx = np.sqrt((x * x).sum(axis=1, keepdims=True))
    ny = np.sqrt((y * y).sum(axis=1, keepdims=True))
    return (((x / nx) * (y / ny)).sum(axis=1),)


# ---------------------------------------------------------------------------
# Euclidean norm with scaling, the way the reference BLAS describes it
# ---------------------------------------------------------------------------
def scaled_norm_unfused():
    """``scale = max |x|``, then ``scale * sqrt(Σ (x / scale)²)``: no square leaves the range."""
    X = _rows()
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    scale = te.compute((n,), lambda i: te.max(tir.abs(X[i, j]), axis=j), name="scale")
    ssq = _sum(X, lambda i, j: (X[i, j] / scale[i]) * (X[i, j] / scale[i]), "ssq")
    return [X], [te.compute((n,), lambda i: scale[i] * tir.sqrt(ssq[i]), name="nrm2")]


def scaled_norm_reference(x):
    x = x.astype(np.float64)
    return (np.sqrt((x * x).sum(axis=1)),)


register(
    Operator(
        "projection",
        "closure",
        "Σ (a - r q)² with r = Σ q a: the remainder of a Gram-Schmidt step",
        projection_unfused,
        projection_reference,
        projection_fused,
        states=3,
    )
)
register(
    Operator(
        "normalised_projection",
        "shift/scale",
        "Σ (a - r q)² with r = Σ q a / Σ q²",
        normalised_projection_unfused,
        normalised_projection_reference,
        normalised_projection_fused,
        states=3,
    )
)
register(
    Operator(
        "householder",
        "closure",
        "Σ (x + ‖x‖ e)²",
        householder_unfused,
        householder_reference,
        householder_fused,
        states=3,
    )
)
register(
    Operator(
        "cosine",
        "shift/scale",
        "Σ (x / ‖x‖)(y / ‖y‖)",
        cosine_unfused,
        cosine_reference,
        cosine_fused,
        states=3,
    )
)
register(
    Operator(
        "scaled_norm",
        "negative",
        "max |x| · sqrt(Σ (x / max |x|)²): fused, it would overflow where it does not now",
        scaled_norm_unfused,
        scaled_norm_reference,
        fusible=False,
        reason="derived reducer(s) rejected",
    )
)
