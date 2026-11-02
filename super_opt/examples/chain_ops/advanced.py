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
"""Covariance estimation and normalized policy objectives."""

import numpy as np

from tvm import te
from tvm_tirx import tirx as tir

from . import DTYPE, Operator, Showcase, register


def _sum(shape, extent, body, name):
    j = te.reduce_axis((0, extent), "sample")
    return te.compute(shape, lambda *idx: te.sum(body(*idx, j), axis=j), name=name)


def ols_hc0_unfused():
    b, n = te.var("batch"), te.var("samples")
    x, y = (te.placeholder((b, n), name=k) for k in ("X", "Y"))
    a = _sum((b,), n, lambda i, j: x[i, j] * x[i, j], "xx")
    c = _sum((b,), n, lambda i, j: x[i, j] * y[i, j], "xy")
    theta = te.compute(
        (b,),
        lambda i: tir.if_then_else(a[i] == 0, tir.const(0.0, DTYPE), c[i] / a[i]),
        name="theta",
    )

    def term(i, j):
        r = y[i, j] - theta[i] * x[i, j]
        return x[i, j] * x[i, j] * r * r

    h = _sum((b,), n, term, "meat")
    out = te.compute(
        (b,),
        lambda i: tir.if_then_else(a[i] == 0, tir.const(0.0, DTYPE), h[i] / (a[i] * a[i])),
        name="hc0",
    )
    return [x, y], [out]


def ols_hc0_reference(x, y):
    x, y = x.astype("float64"), y.astype("float64")
    a = (x * x).sum(-1)
    theta = np.divide((x * y).sum(-1), a, out=np.zeros_like(a), where=a != 0)
    h = (x * x * (y - theta[:, None] * x) ** 2).sum(-1)
    return (np.divide(h, a * a, out=np.zeros_like(a), where=a != 0),)


def normalized_policy_loss_unfused():
    b, n = te.var("batch"), te.var("samples")
    a, g = (te.placeholder((b, n), name=k) for k in ("Advantage", "LogProb"))
    total = _sum((b,), n, lambda i, j: a[i, j], "total")
    mean = te.compute((b,), lambda i: total[i] / n.astype(DTYPE), name="mean")

    def square(i, j):
        d = a[i, j] - mean[i]
        return d * d

    ss = _sum((b,), n, square, "ss")
    scale = te.compute((b,), lambda i: tir.sqrt(ss[i] / (n.astype(DTYPE) - 1)) + 1e-8, name="scale")
    loss = _sum((b,), n, lambda i, j: g[i, j] * (a[i, j] - mean[i]) / scale[i], "loss_sum")
    out = te.compute((b,), lambda i: -loss[i] / n.astype(DTYPE), name="policy_loss")
    return [a, g], [out]


def normalized_policy_loss_reference(a, g):
    a, g = a.astype("float64"), g.astype("float64")
    with np.errstate(all="ignore"):
        scale = a.std(axis=-1, ddof=1, keepdims=True) + 1e-8
        return (-((a - a.mean(-1, keepdims=True)) / scale * g).mean(-1),)


def ledoit_wolf_unfused():
    b, n, d = te.var("batch"), te.var("samples"), te.var("features")
    x = te.placeholder((b, n, d), name="X")
    total = _sum((b, d), n, lambda i, k, j: x[i, j, k], "total")
    mean = te.compute((b, d), lambda i, k: total[i, k] / n.astype(DTYPE), name="mean")
    centered = te.compute((b, n, d), lambda i, j, k: x[i, j, k] - mean[i, k], name="centered")
    covsum = _sum(
        (b, d, d), n, lambda i, k, ell, j: centered[i, j, k] * centered[i, j, ell], "covsum"
    )

    def fourth(i, k, ell, j):
        z = centered[i, j, k] * centered[i, j, ell]
        return z * z

    v = _sum((b, d, d), n, fourth, "v22")
    cov = te.compute((b, d, d), lambda i, k, ell: covsum[i, k, ell] / n.astype(DTYPE), name="cov")
    k = te.reduce_axis((0, d), "feature")
    trace = te.compute((b,), lambda i: te.sum(cov[i, k, k], axis=k), name="trace")
    k, ell = te.reduce_axis((0, d), "k"), te.reduce_axis((0, d), "ell")
    norm = te.compute(
        (b,), lambda i: te.sum(cov[i, k, ell] * cov[i, k, ell], axis=[k, ell]), name="norm"
    )
    k, ell = te.reduce_axis((0, d), "k"), te.reduce_axis((0, d), "ell")
    fourthsum = te.compute((b,), lambda i: te.sum(v[i, k, ell], axis=[k, ell]), name="fourthsum")
    mu = te.compute((b,), lambda i: trace[i] / d.astype(DTYPE), name="mu")
    delta = te.compute((b,), lambda i: norm[i] / d.astype(DTYPE) - mu[i] * mu[i], name="delta")
    beta = te.compute(
        (b,),
        lambda i: tir.min(
            (fourthsum[i] / n.astype(DTYPE) - norm[i]) / (d.astype(DTYPE) * n.astype(DTYPE)),
            delta[i],
        ),
        name="beta",
    )
    rho = te.compute(
        (b,),
        lambda i: tir.if_then_else(
            d == 1,
            tir.const(0.0, DTYPE),
            tir.if_then_else(beta[i] == 0, tir.const(0.0, DTYPE), beta[i] / delta[i]),
        ),
        name="shrinkage",
    )
    out = te.compute(
        (b, d, d),
        lambda i, k, ell: (1 - rho[i]) * cov[i, k, ell]
        + tir.if_then_else(k == ell, rho[i] * mu[i], tir.const(0.0, DTYPE)),
        name="ledoit_wolf",
    )
    return [x], [out, rho]


def ledoit_wolf_reference(x):
    x = x.astype("float64")
    n, d = x.shape[1:]
    z = x - x.mean(1, keepdims=True)
    s = np.einsum("bni,bnj->bij", z, z) / n
    mu = np.trace(s, axis1=1, axis2=2) / d
    norm = (s * s).sum((1, 2))
    delta = norm / d - mu * mu
    beta = np.minimum(((z * z).sum(2) ** 2).sum(1) / n - norm, delta * d * n) / (d * n)
    rho = (
        np.divide(beta, delta, out=np.zeros_like(beta), where=beta != 0)
        if d > 1
        else np.zeros_like(beta)
    )
    return ((1 - rho[:, None, None]) * s + rho[:, None, None] * mu[:, None, None] * np.eye(d), rho)


def _fold(shape, n, leaf, merge, name):
    j = te.reduce_axis((1, n), "sample")

    def body(*idx):
        init = leaf(*idx, 0)

        def identity(*types):
            return init

        red = te.comm_reducer(merge, identity, name=name)
        return red(leaf(*idx, j), axis=j)

    states = te.compute(shape, body, name=name, tag="superopt.nonempty")

    def finish(*idx):
        return tuple(tir.if_then_else(n == 1, v, s(*idx)) for v, s in zip(leaf(*idx, 0), states))

    return te.compute(shape, finish, name=name + "_nonempty")


def ols_hc0_fused():
    inputs, _ = ols_hc0_unfused()
    x, y = inputs
    b, n = x.shape
    zero = tir.const(0.0, DTYPE)

    def theta(a):
        return tir.if_then_else(a[0] == 0, zero, a[1] / a[0])

    def merge(a, b):
        aa, bb = a[0] + b[0], a[1] + b[1]
        t = tir.if_then_else(aa == 0, zero, bb / aa)
        da, db = t - theta(a), t - theta(b)
        return (
            aa,
            bb,
            a[2] + b[2] - 2 * da * a[4] - 2 * db * b[4] + da * da * a[3] + db * db * b[3],
            a[3] + b[3],
            a[4] + b[4] - da * a[3] - db * b[3],
        )

    def leaf(i, j):
        q = x[i, j] * x[i, j]
        return (q, x[i, j] * y[i, j], zero, q * q, zero)

    st = _fold((b,), n, leaf, merge, "hc0_hand")
    return inputs, [
        te.compute(
            (b,),
            lambda i: tir.if_then_else(st[0][i] == 0, zero, st[2][i] / (st[0][i] * st[0][i])),
            name="hc0",
        )
    ]


def normalized_policy_loss_fused():
    inputs, _ = normalized_policy_loss_unfused()
    a, g = inputs
    b, n = a.shape
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)

    def merge(a, b):
        count = a[0] + b[0]
        da = b[1] - a[1]
        dg = b[3] - a[3]
        w = a[0] * b[0] / count
        return (
            count,
            a[1] + da * b[0] / count,
            a[2] + b[2] + da * da * w,
            a[3] + dg * b[0] / count,
            a[4] + b[4] + da * dg * w,
        )

    st = _fold((b,), n, lambda i, j: (one, a[i, j], zero, g[i, j], zero), merge, "policy_hand")
    return inputs, [
        te.compute(
            (b,),
            lambda i: -st[4][i]
            / (n.astype(DTYPE) * (tir.sqrt(st[2][i] / (n.astype(DTYPE) - 1)) + 1e-8)),
            name="policy_loss",
        )
    ]


def ledoit_wolf_fused():
    from tvm.te.superopt.reducer.build import Rewriter
    from tvm.te.superopt.symbolic.lower import compute_ops

    inputs, outputs = ledoit_wolf_unfused()
    x = inputs[0]
    b, n, d = x.shape
    one, zero = tir.const(1.0, DTYPE), tir.const(0.0, DTYPE)

    def merge(a, b):
        count = a[0] + b[0]
        mx = a[1] + (b[1] - a[1]) * b[0] / count
        my = a[2] + (b[2] - a[2]) * b[0] / count

        def shifted(s):
            dx, dy = mx - s[1], my - s[2]
            nn = s[0]
            cxx, cyy, cxy, t21, t12, v = s[3:]
            return (
                cxx + nn * dx * dx,
                cyy + nn * dy * dy,
                cxy + nn * dx * dy,
                t21 - dy * cxx - 2 * dx * cxy - nn * dx * dx * dy,
                t12 - dx * cyy - 2 * dy * cxy - nn * dx * dy * dy,
                v
                - 2 * dy * t21
                - 2 * dx * t12
                + dy * dy * cxx
                + dx * dx * cyy
                + 4 * dx * dy * cxy
                + nn * dx * dx * dy * dy,
            )

        sa, sb = shifted(a), shifted(b)
        return (count, mx, my, *(u + v for u, v in zip(sa, sb)))

    st = _fold(
        (b, d, d),
        n,
        lambda i, k, ell, j: (one, x[i, j, k], x[i, j, ell], zero, zero, zero, zero, zero, zero),
        merge,
        "lw_hand",
    )
    rewrite = Rewriter()
    for op in compute_ops(outputs):
        if op.name in ("covsum", "v22"):
            rewrite.replace(op.output(0), st[5 if op.name == "covsum" else 8])
    return inputs, [rewrite.tensor(t) for t in outputs]


for name, doc, unfused, reference, hand, states, domain in (
    (
        "ledoit_wolf",
        "Centered Ledoit-Wolf covariance and shrinkage",
        ledoit_wolf_unfused,
        ledoit_wolf_reference,
        ledoit_wolf_fused,
        9,
        "statistical learning: covariance shrinkage",
    ),
    (
        "ols_hc0",
        "Scalar no-intercept OLS HC0 covariance",
        ols_hc0_unfused,
        ols_hc0_reference,
        ols_hc0_fused,
        5,
        "econometrics: heteroskedasticity-robust scalar OLS inference",
    ),
    (
        "normalized_policy_loss",
        "Sample-standardized advantage policy loss",
        normalized_policy_loss_unfused,
        normalized_policy_loss_reference,
        normalized_policy_loss_fused,
        5,
        "reinforcement learning: normalized-advantage policy gradients",
    ),
):
    register(
        Operator(
            name, "closure", doc, unfused, reference, hand, states=states, showcase=Showcase(domain)
        )
    )
