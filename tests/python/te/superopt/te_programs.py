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
"""TE programs with symbolic shapes shared by the superoptimizer tests."""

from __future__ import annotations

import numpy as np

import tvm
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


def attention_reference(Q, K, V):
    """float64 numpy reference of scaled dot-product attention."""
    Q, K, V = (x.astype("float64") for x in (Q, K, V))
    S = np.einsum("bhid,bhjd->bhij", Q, K) / np.sqrt(Q.shape[-1])
    P = np.exp(S - S.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    return np.einsum("bhij,bhje->bhie", P, V)


def scaled_matmul():
    """``alpha * (A @ B)`` with symbolic shapes (STENSO's reorder example)."""
    n, m, p = te.var("n"), te.var("m"), te.var("p")
    A = te.placeholder((n, m), name="A", dtype="float32")
    B = te.placeholder((m, p), name="B", dtype="float32")
    k = te.reduce_axis((0, m), "k")
    AB = te.compute((n, p), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="AB")
    alpha = tir.const(1.0, "float32") / tir.sqrt(m.astype("float32"))
    O = te.compute((n, p), lambda i, j: AB[i, j] * alpha, name="O")  # noqa: E741
    return [A, B], O


def softmax_rows():
    """Row softmax written with the max shift."""
    n, m = te.var("n"), te.var("m")
    S = te.placeholder((n, m), name="S", dtype="float32")
    j1 = te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(S[i, j1], axis=j1), name="mx")
    e = te.compute((n, m), lambda i, j: tir.exp(S[i, j] - mx[i]), name="e")
    j2 = te.reduce_axis((0, m), "j")
    den = te.compute((n,), lambda i: te.sum(e[i, j2], axis=j2), name="den")
    P = te.compute((n, m), lambda i, j: e[i, j] / den[i], name="P")
    return [S], P


def sum_pair():
    """``T[i] = Σ_j X[i,j] + Σ_j Y[i,j]``: the simplest tuple-reducer target."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype="float32")
    Y = te.placeholder((n, m), name="Y", dtype="float32")
    j1 = te.reduce_axis((0, m), "j")
    sx = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="sx")
    j2 = te.reduce_axis((0, m), "j")
    sy = te.compute((n,), lambda i: te.sum(Y[i, j2], axis=j2), name="sy")
    T = te.compute((n,), lambda i: sx[i] + sy[i], name="T")
    return [X, Y], T


def softmax_value(stable: bool = True):
    """``Σ_j exp(c S_ij) V_je / Σ_j exp(c S_ij)`` from an explicit score matrix."""
    n, m, d = te.var("n"), te.var("m"), te.var("d")
    S = te.placeholder((n, m), name="S", dtype="float32")
    V = te.placeholder((m, d), name="V", dtype="float32")
    if stable:
        j1 = te.reduce_axis((0, m), "j")
        mx = te.compute((n,), lambda i: te.max(S[i, j1], axis=j1), name="mx")
        e = te.compute((n, m), lambda i, j: tir.exp(S[i, j] - mx[i]), name="e")
    else:
        e = te.compute((n, m), lambda i, j: tir.exp(S[i, j]), name="e")
    j2 = te.reduce_axis((0, m), "j")
    den = te.compute((n,), lambda i: te.sum(e[i, j2], axis=j2), name="den")
    j3 = te.reduce_axis((0, m), "j")
    num = te.compute((n, d), lambda i, k: te.sum(e[i, j3] * V[j3, k], axis=j3), name="num")
    O = te.compute((n, d), lambda i, k: num[i, k] / den[i], name="O")  # noqa: E741
    return [S, V], O


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


def attention_projected(kind: str = "naive"):
    """Attention followed by an output projection: ``Y = softmax(Q Kᵀ c) V W``."""
    (Q, K, V), out = attention(kind)
    d_model = te.var("d_model")
    W = te.placeholder((out.shape[3], d_model), name="W", dtype="float32")
    e = te.reduce_axis((0, out.shape[3]), name="e")
    Y = te.compute(
        (out.shape[0], out.shape[1], out.shape[2], d_model),
        lambda b, h, i, f: te.sum(out[b, h, i, e] * W[e, f], axis=e),
        name="Y",
    )
    return [Q, K, V, W], Y


def variance_rows():
    """Row variance ``Σ_j (x_ij - mean_i)^2 / m`` with the mean materialised (Welford's target)."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype="float32")
    m_f = m.astype("float32")
    j1 = te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m_f, name="mean")
    sq = te.compute((n, m), lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), name="sq")
    j2 = te.reduce_axis((0, m), "j")
    ss = te.compute((n,), lambda i: te.sum(sq[i, j2], axis=j2), name="ss")
    var = te.compute((n,), lambda i: ss[i] / m_f, name="var")
    return [X], var


def run_llvm(inputs, output, arrays, out_dtype):
    """Compile ``output`` for LLVM and run it on numpy ``arrays``; returns the result."""
    mod = tvm.IRModule({"main": te.create_prim_func([*inputs, output])})
    lib = tvm.compile(mod, target="llvm")
    out_shape = _infer_out_shape(inputs, output, arrays)
    out = np.zeros(out_shape, dtype=out_dtype)
    args = [tvm.runtime.tensor(x) for x in (*arrays, out)]
    lib["main"](*args)
    return args[-1].numpy()


def _infer_out_shape(inputs, output, arrays):
    env = {}
    for t, a in zip(inputs, arrays):
        for s, v in zip(t.shape, a.shape):
            if isinstance(s, tir.Var):
                env[s.name] = v
    shape = []
    for s in output.shape:
        shape.append(env[s.name] if isinstance(s, tir.Var) else int(s.value))
    return tuple(shape)
