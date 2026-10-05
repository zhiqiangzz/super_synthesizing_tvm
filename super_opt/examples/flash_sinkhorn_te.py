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

"""One Sinkhorn iteration, before and after FlashSinkhorn, in TVM TE.

Entropic optimal transport between point clouds ``x`` (n points) and ``y``
(m points) with weights ``a`` and ``b``, cost ``C_ij = |x_i - y_j|²`` and
regularisation ``eps``. One stabilised log-domain Sinkhorn iteration updates
the dual potentials (FlashSinkhorn, eq. 2-3)::

    f_i  <- -eps LSE_j[(g_j - C_ij) / eps + log b_j]
    g'_j <- -eps LSE_i[(f_i - C_ij) / eps + log a_i]

Both variants compute exactly this map ``(x, y, a, b, g) -> (f, g')``; the
optimal plan is ``P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps)`` and is never
needed inside the iteration. ``sinkhorn_logdomain`` is the program a
superoptimizer starts from; ``flash_sinkhorn`` is the form the paper derives
from it.

``sinkhorn_logdomain``
    The tensorised log-domain implementation, written the way the update
    reads: materialise ``C``, the logits ``S = (g - C) / eps + log b`` and
    ``S2 = (f - C) / eps + log a``, and take each log-sum-exp in two passes
    (row max, then sum of shifted exponentials) -- two ``n x m``
    intermediates per half-step.
``flash_sinkhorn``
    FlashSinkhorn (Proposition 1 and Algorithms 1, 3). With
    ``alpha_i = |x_i|²`` and ``beta_j = |y_j|²`` the cost splits as
    ``C_ij = alpha_i + beta_j - 2 x_i·y_j``, so on the shifted potentials
    ``f^ = f - alpha``, ``g^ = g - beta`` each half-step is a row-wise LSE of a
    biased dot-product score, the normalisation of scaled dot-product
    attention (``Q = sqrt(2) X``, ``K = sqrt(2) Y``)::

        f^ <- -eps LSE_row((Q Kᵀ + 1 (g^ + eps log b)ᵀ) / eps)
        g^ <- -eps LSE_row((K Qᵀ + 1 (f^ + eps log a)ᵀ) / eps)

    Each LSE is one ``te.comm_reducer`` carrying the online ``(running max,
    rescaled sum of exp)`` of Algorithm 1, so a half-step is a single
    streaming pass that writes only a potential vector; ``C``, ``K`` and the
    logits are never formed. The shift by ``alpha``/``beta`` is applied on the
    way in and out so both variants share one signature (in a full solve it
    happens once, outside the loop).

As ``S = Q Kᵀ`` in ``flashattn_te.py``, the dot products ``X Yᵀ`` are one
materialised reduction here: a TE reduction owns the whole body of its block,
so the GEMM cannot sit inside the reducer's inputs. Computing it tile by tile
inside each pass, in on-chip memory, is the schedule's job, not the
algorithm's.

By default this prints the shape-generic s_tir of every variant and checks an
LLVM build of each against a float64 numpy iteration, for a comfortable and a
small ``eps``. See ``--help``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import numpy as np
from harness import Accuracy, accuracy_columns, label, render_source, render_table
from rich.console import Console

import tvm
from tvm import te
from tvm import tirx as tir  # tvm.tir is absent in this fork; intrinsics live in tirx


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SinkhornShape:
    """A fully specialised problem size."""

    batch: int
    n: int
    m: int
    dim: int

    def __str__(self) -> str:
        return f"batch={self.batch} n={self.n} m={self.m} dim={self.dim}"


def _inputs(batch, n, m, dim, dtype):
    """Points, weights and the current column potential ``g``.

    Any extent left as ``None`` becomes a ``te.var``.
    """

    def extent(value, name):
        return te.var(name) if value is None else value

    n_b, n_x, n_y, d = (
        extent(v, k) for v, k in ((batch, "batch"), (n, "n"), (m, "m"), (dim, "dim"))
    )
    x = te.placeholder((n_b, n_x, d), name="x", dtype=dtype)
    y = te.placeholder((n_b, n_y, d), name="y", dtype=dtype)
    a = te.placeholder((n_b, n_x), name="a", dtype=dtype)
    b = te.placeholder((n_b, n_y), name="b", dtype=dtype)
    g = te.placeholder((n_b, n_y), name="g", dtype=dtype)
    return x, y, a, b, g


def _cost(x, y):
    """``C[b, i, j] = Σ_k (x[b, i, k] - y[b, j, k])²``, materialised."""
    n_b, n_x, d = x.shape
    n_y = y.shape[1]
    k = te.reduce_axis((0, d), name="k")
    return te.compute(
        (n_b, n_x, n_y),
        lambda bb, i, j: te.sum((x[bb, i, k] - y[bb, j, k]) * (x[bb, i, k] - y[bb, j, k]), axis=k),
        name="C",
    )


def _module(x, y, a, b, g, f, g_new) -> tvm.IRModule:
    return tvm.IRModule({"main": te.create_prim_func([x, y, a, b, g, f, g_new])})


def sinkhorn_logdomain(*, batch=None, n=None, m=None, dim=None, eps=0.5, dtype="float32"):
    """Tensorised log domain: ``C``, ``S`` and ``S2`` materialised, two-pass LSE."""
    x, y, a, b, g = _inputs(batch, n, m, dim, dtype)
    n_b, n_x, n_y = x.shape[0], x.shape[1], y.shape[1]
    eps_c = tir.const(eps, dtype)
    inv_eps = tir.const(1.0 / eps, dtype)

    C = _cost(x, y)

    # ---- f-update: S[i, j] = (g_j - C_ij) / eps + log b_j, row-wise LSE --------
    S = te.compute(
        C.shape,
        lambda bb, i, j: (g[bb, j] - C[bb, i, j]) * inv_eps + tir.log(b[bb, j]),
        name="S",
    )
    j1 = te.reduce_axis((0, n_y), name="j")
    s_max = te.compute((n_b, n_x), lambda bb, i: te.max(S[bb, i, j1], axis=j1), name="S_max")
    j2 = te.reduce_axis((0, n_y), name="j")
    s_sum = te.compute(
        (n_b, n_x),
        lambda bb, i: te.sum(tir.exp(S[bb, i, j2] - s_max[bb, i]), axis=j2),
        name="S_sumexp",
    )
    f = te.compute(
        (n_b, n_x),
        lambda bb, i: -eps_c * (s_max[bb, i] + tir.log(s_sum[bb, i])),
        name="f",
    )

    # ---- g-update: S2[j, i] = (f_i - C_ij) / eps + log a_i, row-wise LSE -------
    S2 = te.compute(
        (n_b, n_y, n_x),
        lambda bb, j, i: (f[bb, i] - C[bb, i, j]) * inv_eps + tir.log(a[bb, i]),
        name="S2",
    )
    i1 = te.reduce_axis((0, n_x), name="i")
    s2_max = te.compute((n_b, n_y), lambda bb, j: te.max(S2[bb, j, i1], axis=i1), name="S2_max")
    i2 = te.reduce_axis((0, n_x), name="i")
    s2_sum = te.compute(
        (n_b, n_y),
        lambda bb, j: te.sum(tir.exp(S2[bb, j, i2] - s2_max[bb, j]), axis=i2),
        name="S2_sumexp",
    )
    g_new = te.compute(
        (n_b, n_y),
        lambda bb, j: -eps_c * (s2_max[bb, j] + tir.log(s2_sum[bb, j])),
        name="g_new",
    )
    return _module(x, y, a, b, g, f, g_new)


def flash_sinkhorn(*, batch=None, n=None, m=None, dim=None, eps=0.5, dtype="float32"):
    """FlashSinkhorn: biased dot-product scores, one online-LSE reduction per half-step."""
    x, y, a, b, g = _inputs(batch, n, m, dim, dtype)
    n_b, n_x, d = x.shape
    n_y = y.shape[1]
    eps_c = tir.const(eps, dtype)
    inv_eps = tir.const(1.0 / eps, dtype)
    two = tir.const(2.0, dtype)
    one = tir.const(1.0, dtype)

    # ---- precomputable per-point terms (Proposition 1) -------------------------
    k1 = te.reduce_axis((0, d), name="k")
    alpha = te.compute(
        (n_b, n_x), lambda bb, i: te.sum(x[bb, i, k1] * x[bb, i, k1], axis=k1), name="alpha"
    )
    k2 = te.reduce_axis((0, d), name="k")
    beta = te.compute(
        (n_b, n_y), lambda bb, j: te.sum(y[bb, j, k2] * y[bb, j, k2], axis=k2), name="beta"
    )
    delta = te.compute((n_b, n_y), lambda bb, j: eps_c * tir.log(b[bb, j]), name="delta")
    gamma = te.compute((n_b, n_x), lambda bb, i: eps_c * tir.log(a[bb, i]), name="gamma")
    g_hat = te.compute((n_b, n_y), lambda bb, j: g[bb, j] - beta[bb, j], name="g_hat")

    # ---- dot products X Yᵀ: Q Kᵀ = 2 X Yᵀ with Q = sqrt(2) X, K = sqrt(2) Y ---------
    k3 = te.reduce_axis((0, d), name="k")
    XY = te.compute(
        (n_b, n_x, n_y),
        lambda bb, i, j: te.sum(x[bb, i, k3] * y[bb, j, k3], axis=k3),
        name="XY",
    )

    # ---- online LSE (Algorithm 1, lines 10-13), as a commutative reducer --------
    def merge(state_a, state_b):
        """Merge two partial ``(max, Σ exp(score - max))`` states."""
        max_a, sum_a = state_a
        max_b, sum_b = state_b
        new_max = tir.max(max_a, max_b)
        return new_max, sum_a * tir.exp(max_a - new_max) + sum_b * tir.exp(max_b - new_max)

    def empty_state(max_dtype, sum_dtype):
        return tir.min_value(max_dtype), tir.const(0.0, sum_dtype)  # m <- -inf, s <- 0

    online_lse = te.comm_reducer(merge, empty_state, name="online_lse")

    # ---- f^-update: f^_i = -eps LSE_j((2 X Yᵀ + g^ + delta)_ij / eps) ------------
    j = te.reduce_axis((0, n_y), name="j")
    row_max, row_sum = te.compute(
        (n_b, n_x),
        lambda bb, i: online_lse(
            ((two * XY[bb, i, j] + g_hat[bb, j] + delta[bb, j]) * inv_eps, one), axis=j
        ),
        name="lse_f",
    )
    f_hat = te.compute(
        (n_b, n_x), lambda bb, i: -eps_c * (row_max[bb, i] + tir.log(row_sum[bb, i])), name="f_hat"
    )

    # ---- g^-update: g^_j = -eps LSE_i((2 X Yᵀ + f^ + gamma)_ij / eps) ------------
    i = te.reduce_axis((0, n_x), name="i")
    col_max, col_sum = te.compute(
        (n_b, n_y),
        lambda bb, j: online_lse(
            ((two * XY[bb, i, j] + f_hat[bb, i] + gamma[bb, i]) * inv_eps, one), axis=i
        ),
        name="lse_g",
    )
    g_hat_new = te.compute(
        (n_b, n_y),
        lambda bb, j: -eps_c * (col_max[bb, j] + tir.log(col_sum[bb, j])),
        name="g_hat_new",
    )

    # ---- back to the unshifted potentials ---------------------------------------
    f = te.compute((n_b, n_x), lambda bb, i: f_hat[bb, i] + alpha[bb, i], name="f")
    g_new = te.compute((n_b, n_y), lambda bb, j: g_hat_new[bb, j] + beta[bb, j], name="g_new")
    return _module(x, y, a, b, g, f, g_new)


# The two ways of writing the same Sinkhorn iteration, behind one name each.
# They take the same keyword arguments and produce the same signature
# (x, y, a, b, g -> f, g').
VARIANTS = {
    "logdomain": sinkhorn_logdomain,
    "flash": flash_sinkhorn,
}

VARIANT_SUBTITLES = {
    "logdomain": "C, S and S2 materialised; each LSE is max -> exp -> sum -> log",
    "flash": "X Yᵀ materialised; each half-step one online-LSE reduce over biased scores",
}


# ---------------------------------------------------------------------------
# Numerical check against a float64 reference
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AccuracyReport:
    variant: str
    eps: float
    accuracy: Accuracy


def make_inputs(shape: SinkhornShape, *, dtype: str, seed: int):
    """Points in the unit cube, uniform weights and a random current ``g``."""
    rng = np.random.default_rng(seed)
    x = rng.random((shape.batch, shape.n, shape.dim)).astype(dtype)
    y = rng.random((shape.batch, shape.m, shape.dim)).astype(dtype)
    a = np.full((shape.batch, shape.n), 1.0 / shape.n, dtype=dtype)
    b = np.full((shape.batch, shape.m), 1.0 / shape.m, dtype=dtype)
    g = rng.uniform(-0.1, 0.1, (shape.batch, shape.m)).astype(dtype)
    return x, y, a, b, g


def reference_iteration(x, y, a, b, g, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (2)-(3) in float64 with a max-shifted LSE."""
    x, y, a, b, g = (v.astype(np.float64) for v in (x, y, a, b, g))
    C = ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(-1)

    def lse(z, axis):
        mx = z.max(axis=axis, keepdims=True)
        return (mx + np.log(np.exp(z - mx).sum(axis=axis, keepdims=True))).squeeze(axis)

    f = -eps * lse((g[:, None, :] - C) / eps + np.log(b)[:, None, :], axis=2)
    g_new = -eps * lse((f[:, :, None] - C) / eps + np.log(a)[:, :, None], axis=1)
    return f, g_new


def check(variant, shape, *, eps, dtype, seed, rtol, atol) -> Accuracy:
    mod = VARIANTS[variant](**dataclasses.asdict(shape), eps=eps, dtype=dtype)
    lib = tvm.compile(mod, target="llvm")
    args = make_inputs(shape, dtype=dtype, seed=seed)
    f_out = np.zeros((shape.batch, shape.n), dtype=dtype)
    g_out = np.zeros((shape.batch, shape.m), dtype=dtype)
    tensors = [tvm.runtime.tensor(v) for v in (*args, f_out, g_out)]
    lib["main"](*tensors)
    got = np.concatenate([t.numpy().astype(np.float64).ravel() for t in tensors[-2:]])
    want = np.concatenate([v.ravel() for v in reference_iteration(*args, eps=eps)])
    with np.errstate(invalid="ignore"):
        finite = bool(np.all(np.isfinite(got)))
        max_abs = float(np.abs(got - want).max()) if finite else float("inf")
    return Accuracy(
        max_abs_err=max_abs,
        rel_err=max_abs / float(np.abs(want).max()),
        rtol=rtol,
        atol=atol,
        passed=finite and bool(np.allclose(got, want, rtol=rtol, atol=atol)),
    )


def render_accuracy(console: Console, reports: list[AccuracyReport], caption: str) -> None:
    render_table(
        console,
        title="TVM LLVM vs one float64 log-domain Sinkhorn iteration (f and g')",
        caption=caption,
        columns=[
            label("variant", lambda report: report.variant),
            label("eps", lambda report: f"{report.eps:g}"),
            *accuracy_columns(),
        ],
        rows=reports,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    shape = parser.add_argument_group("problem")
    shape.add_argument("--batch", type=int, default=2, help="(default: %(default)s)")
    shape.add_argument("--n", type=int, default=64, help="source points (default: %(default)s)")
    shape.add_argument("--m", type=int, default=48, help="target points (default: %(default)s)")
    shape.add_argument("--dim", type=int, default=4, help="point dimension (default: %(default)s)")
    shape.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.5, 0.001],
        help="entropic regularisation(s) to check (default: 0.5 0.001)",
    )
    numerics = parser.add_argument_group("numerics")
    numerics.add_argument("--dtype", default="float32", help="(default: %(default)s)")
    numerics.add_argument("--rtol", type=float, default=1e-3, help="(default: %(default)s)")
    numerics.add_argument("--atol", type=float, default=1e-5, help="(default: %(default)s)")
    numerics.add_argument("--seed", type=int, default=0, help="(default: %(default)s)")
    output = parser.add_argument_group("output")
    output.add_argument(
        "--variant",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
        help="formulation(s) to build (default: all)",
    )
    output.add_argument("--no-ir", dest="show_ir", action="store_false", help="skip the s_tir dump")
    output.add_argument(
        "--no-check", dest="check", action="store_false", help="skip the numerical check"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    if args.show_ir:
        for variant in args.variant:
            symbolic = VARIANTS[variant](eps=args.eps[0], dtype=args.dtype)
            render_source(
                console,
                symbolic["main"].script(),
                lexer="python",
                title=f"[bold]shape-generic s_tir[/] — {variant}",
                subtitle=VARIANT_SUBTITLES[variant],
            )
    if not args.check:
        return 0
    shape = SinkhornShape(batch=args.batch, n=args.n, m=args.m, dim=args.dim)
    reports = [
        AccuracyReport(
            variant,
            eps,
            check(
                variant,
                shape,
                eps=eps,
                dtype=args.dtype,
                seed=args.seed,
                rtol=args.rtol,
                atol=args.atol,
            ),
        )
        for eps in args.eps
        for variant in args.variant
    ]
    render_accuracy(console, reports, f"{shape}, one iteration, {args.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
