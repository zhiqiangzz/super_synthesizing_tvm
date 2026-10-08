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
"""Scaled dot-product attention: ``O = softmax(Q Kᵀ / sqrt(d)) V``.

Unfused, the key axis is walked three times (row maximum, normaliser,
weighted sum of the values) and the attention matrix is materialised.
Fused it is FlashAttention's reduction: the online softmax with the
unnormalised output as a third state, divided by the normaliser at the end.
"""

from __future__ import annotations

import numpy as np

from tvm import te
from tvm import tirx as tir

from . import DTYPE, Operator, register
from .softmax import online_softmax

DIMS = ("batch", "heads", "seq_q", "seq_k", "dim")


def _inputs(bias: bool):
    n_b, n_h, n_q, n_k, dim = (te.var(n) for n in DIMS)
    Q = te.placeholder((n_b, n_h, n_q, dim), name="Q", dtype=DTYPE)
    K = te.placeholder((n_b, n_h, n_k, dim), name="K", dtype=DTYPE)
    V = te.placeholder((n_b, n_h, n_k, dim), name="V", dtype=DTYPE)
    ins = [Q, K, V]
    if bias:
        ins.append(te.placeholder((n_h, n_q, n_k), name="B", dtype=DTYPE))
    return ins


def _scores(ins):
    """``Q Kᵀ / sqrt(d)`` (plus a per-head additive bias: relative positions, a mask)."""
    Q, K = ins[0], ins[1]
    n_b, n_h, n_q, dim = Q.shape
    n_k = K.shape[2]
    d = te.reduce_axis((0, dim), name="d")
    QK = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: te.sum(Q[b, h, i, d] * K[b, h, j, d], axis=d),
        name="QK",
    )
    scale = tir.const(1.0, DTYPE) / tir.sqrt(dim.astype(DTYPE))
    if len(ins) == 3:
        return te.compute(QK.shape, lambda b, h, i, j: QK[b, h, i, j] * scale, name="S")
    B = ins[3]
    return te.compute(QK.shape, lambda b, h, i, j: QK[b, h, i, j] * scale + B[h, i, j], name="S")


def _unfused(bias: bool):
    ins = _inputs(bias)
    V = ins[2]
    S = _scores(ins)
    n_b, n_h, n_q, n_k = S.shape
    dim = V.shape[3]
    j1 = te.reduce_axis((0, n_k), name="j")
    row_max = te.compute(
        (n_b, n_h, n_q), lambda b, h, i: te.max(S[b, h, i, j1], axis=j1), name="row_max"
    )
    exp_S = te.compute(
        S.shape, lambda b, h, i, j: tir.exp(S[b, h, i, j] - row_max[b, h, i]), name="exp_S"
    )
    j2 = te.reduce_axis((0, n_k), name="j")
    den = te.compute(
        (n_b, n_h, n_q), lambda b, h, i: te.sum(exp_S[b, h, i, j2], axis=j2), name="den"
    )
    P = te.compute(S.shape, lambda b, h, i, j: exp_S[b, h, i, j] / den[b, h, i], name="P")
    j3 = te.reduce_axis((0, n_k), name="j")
    O = te.compute(  # noqa: E741
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: te.sum(P[b, h, i, j3] * V[b, h, j3, e], axis=j3),
        name="O",
    )
    return ins, [O]


def _fused(bias: bool):
    ins = _inputs(bias)
    V = ins[2]
    S = _scores(ins)
    n_b, n_h, n_q, n_k = S.shape
    dim = V.shape[3]
    j = te.reduce_axis((0, n_k), name="j")
    red = online_softmax(2)
    one = tir.const(1.0, DTYPE)
    _, den, acc = te.compute(
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: red((S[b, h, i, j], one, V[b, h, j, e]), axis=j),
        name="flash",
    )
    O = te.compute(  # noqa: E741
        (n_b, n_h, n_q, dim), lambda b, h, i, e: acc[b, h, i, e] / den[b, h, i, e], name="O"
    )
    return ins, [O]


def _reference(q, k, v, bias=None):
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    s = np.einsum("bhid,bhjd->bhij", q, k) / np.sqrt(q.shape[-1])
    if bias is not None:
        s = s + bias.astype(np.float64)[None]
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return (np.einsum("bhij,bhje->bhie", p, v),)


register(
    Operator(
        "attention",
        "shift/scale",
        "softmax(Q Kᵀ / sqrt(d)) V",
        lambda: _unfused(False),
        _reference,
        lambda: _fused(False),
        states=3,
    )
)
register(
    Operator(
        "attention_bias",
        "shift/scale",
        "softmax(Q Kᵀ / sqrt(d) + B) V",
        lambda: _unfused(True),
        _reference,
        lambda: _fused(True),
        states=3,
    )
)
