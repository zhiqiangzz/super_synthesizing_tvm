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
"""TE programs with symbolic shapes shared by the tests of the symbolic layer."""

from __future__ import annotations

from tvm import te
from tvm import tirx as tir

ATTN_DIMS = ("batch", "num_heads", "seqlen_q", "seqlen_k", "head_dim")


def attention(kind: str = "naive", dtype: str = "float32"):
    """``O = softmax(Q Kᵀ / sqrt(d)) V`` as the textbook chain or the online-softmax form.

    Mirrors ``super_opt/examples/flashattn_te.py`` but returns ``(inputs, output)``.
    """
    n_b, n_h, n_q, n_k, dim = (te.var(n) for n in ATTN_DIMS)
    Q = te.placeholder((n_b, n_h, n_q, dim), name="Q", dtype=dtype)
    K = te.placeholder((n_b, n_h, n_k, dim), name="K", dtype=dtype)
    V = te.placeholder((n_b, n_h, n_k, dim), name="V", dtype=dtype)
    d = te.reduce_axis((0, dim), name="d")
    S = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: te.sum(Q[b, h, i, d] * K[b, h, j, d], axis=d),
        name="S",
    )
    scale = tir.const(1.0, dtype) / tir.sqrt(dim.astype(dtype))
    if kind == "naive":
        j1 = te.reduce_axis((0, n_k), name="j")
        row_max = te.compute(
            (n_b, n_h, n_q), lambda b, h, i: te.max(S[b, h, i, j1], axis=j1), name="row_max"
        )
        exp_S = te.compute(
            (n_b, n_h, n_q, n_k),
            lambda b, h, i, j: tir.exp((S[b, h, i, j] - row_max[b, h, i]) * scale),
            name="exp_S",
        )
        j2 = te.reduce_axis((0, n_k), name="j")
        den = te.compute(
            (n_b, n_h, n_q), lambda b, h, i: te.sum(exp_S[b, h, i, j2], axis=j2), name="den"
        )
        P = te.compute(
            (n_b, n_h, n_q, n_k), lambda b, h, i, j: exp_S[b, h, i, j] / den[b, h, i], name="P"
        )
        j3 = te.reduce_axis((0, n_k), name="j")
        PV = te.compute(
            (n_b, n_h, n_q, dim),
            lambda b, h, i, e: te.sum(P[b, h, i, j3] * V[b, h, j3, e], axis=j3),
            name="PV",
        )
        return [Q, K, V], PV

    def merge(a, b):
        m = tir.max(a[0], b[0])
        ra = tir.exp(a[0] - m)
        rb = tir.exp(b[0] - m)
        return (m, a[1] * ra + b[1] * rb, a[2] * ra + b[2] * rb)

    def ident(t0, t1, t2):
        return (tir.min_value(t0), tir.const(0.0, t1), tir.const(0.0, t2))

    online_softmax = te.comm_reducer(merge, ident, name="online_softmax")
    j = te.reduce_axis((0, n_k), name="j")
    _m, denom, acc = te.compute(
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: online_softmax(
            (S[b, h, i, j] * scale, tir.const(1.0, dtype), V[b, h, j, e]), axis=j
        ),
        name="softmax_state",
    )
    O = te.compute(  # noqa: E741
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: acc[b, h, i, e] / denom[b, h, i, e],
        name="O",
    )
    return [Q, K, V], O


def softmax_value_logspace():
    """``softmax(S) @ V`` as a 2-state ``(log Σ exp, normalised output)`` reducer.

    The merge keeps ``z = log(Σ exp s)`` and ``o = Σ exp(s) v / Σ exp s`` and
    needs no finalisation: ``o`` is the answer.
    """
    n, m, d = te.var("n"), te.var("m"), te.var("d")
    S = te.placeholder((n, m), name="S", dtype="float32")
    V = te.placeholder((m, d), name="V", dtype="float32")
    k = te.reduce_axis((0, m), name="k")

    def fcombine(x, y):
        z_x, o_x = x
        z_y, o_y = y
        ex, ey = tir.exp(z_x), tir.exp(z_y)
        den = ex + ey
        return tir.log(den), (ex * o_x + ey * o_y) / den

    def fidentity(t_z, t_o):
        return tir.min_value(t_z), tir.const(0.0, t_o)

    reducer = te.comm_reducer(fcombine, fidentity, name="attention_reducer")
    _z, out = te.compute((n, d), lambda i, e: reducer((S[i, k], V[k, e]), axis=k), name="attention")
    return [S, V], out
