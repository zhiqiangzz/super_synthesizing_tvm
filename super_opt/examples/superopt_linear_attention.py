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

"""Superoptimize (non-causal) linear attention into its O(N d²) order.

The target is kernelised attention written the quadratic way, from feature
maps ``Qf = φ(Q)``, ``Kf = φ(K)`` given as inputs, with symbolic shapes::

    A[i, j] = Σ_d Qf[i, d] Kf[j, d]              Nq x N scores, materialised
    O[i, e] = Σ_j A[i, j] V[j, e]                (--normalized: / Σ_j A[i, j])

That costs ``Nq N d + Nq N e`` multiply-adds and an ``Nq x N`` intermediate.
Contracting the sequence first, ``KV[d, e] = Σ_j Kf[j, d] V[j, e]`` and
``O = Qf KV``, costs ``N d e + Nq d e`` and keeps only ``d x e``: linear in
the sequence length. As with the matrix chain, nothing in the search knows
this; the reordered program is proven equal by the sum-product normal form.

Replacing *softmax* attention by a kernelised one changes the model and is
not an equivalence, so it is not something this search can (or should)
find; the rewrite here starts from the kernelised definition.

The normalised form (``--normalized``) is expressible the same way --
``KV = Kfᵀ V``, ``ks = Σ_j Kf``, ``Qf KV``, ``Qf ks``, one division -- but
that is five ops, and the exhaustive enumeration does not finish budget 5
within an hour for this target.

    python super_opt/examples/superopt_linear_attention.py
    python super_opt/examples/superopt_linear_attention.py --check --te --no-ir
    python super_opt/examples/superopt_linear_attention.py --normalized --budget 5   # slow

See ``--help``.
"""

from __future__ import annotations

import sys

import numpy as np
from harness.superopt import make_parser, run_llvm, search, shape_of, show
from rich.console import Console

from tvm import te


def linear_attention_te(dtype: str, normalized: bool):
    """Quadratic-order kernelised attention as ``(inputs, output)`` te.Tensors."""
    Nq, N, D, E = (te.var(n) for n in ("Nq", "N", "D", "E"))
    Qf = te.placeholder((Nq, D), name="Qf", dtype=dtype)
    Kf = te.placeholder((N, D), name="Kf", dtype=dtype)
    V = te.placeholder((N, E), name="V", dtype=dtype)
    d = te.reduce_axis((0, D), name="d")
    A = te.compute((Nq, N), lambda i, j: te.sum(Qf[i, d] * Kf[j, d], axis=d), name="A")
    j1 = te.reduce_axis((0, N), name="j")
    num = te.compute((Nq, E), lambda i, e: te.sum(A[i, j1] * V[j1, e], axis=j1), name="num")
    if not normalized:
        return [Qf, Kf, V], num
    j2 = te.reduce_axis((0, N), name="j")
    den = te.compute((Nq,), lambda i: te.sum(A[i, j2], axis=j2), name="den")
    O = te.compute((Nq, E), lambda i, e: num[i, e] / den[i], name="O")  # noqa: E741
    return [Qf, Kf, V], O


def materialises_scores(result) -> bool:
    """Does some op write an ``Nq x N`` tensor (the quadratic intermediate)?"""
    ctx = result.ctx
    nq, n = ctx.dims.key_of_name("Nq"), ctx.dims.key_of_name("N")
    for rec in result.snapshot.ops:
        for o in rec.outputs:
            shape = result.snapshot.pool[o].shape
            if nq in shape and n in shape:
                return True
    return False


def check(result, inputs, sizes, seed: int, normalized: bool) -> float:
    """Max relative error of the LLVM build against numpy (positive feature maps)."""
    rng = np.random.default_rng(seed)
    dtype = str(inputs[0].dtype)
    qf, kf = (rng.random(shape_of(t, sizes)).astype(dtype) for t in inputs[:2])
    v = rng.standard_normal(shape_of(inputs[2], sizes)).astype(dtype)
    got = run_llvm(result, inputs, [qf, kf, v], sizes).astype("float64")
    scores = qf.astype("float64") @ kf.astype("float64").T
    want = scores @ v.astype("float64")
    if normalized:
        want = want / scores.sum(axis=1, keepdims=True)
    return float(np.abs(got - want).max() / np.abs(want).max())


def main(argv: list[str] | None = None) -> int:
    parser = make_parser(__doc__, budget=2, sizes="Nq=512,N=512,D=16,E=16")
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="divide by Σ_j A[i, j] (needs --budget 5, which takes over an hour)",
    )
    args = parser.parse_args(argv)
    console = Console()
    inputs, out = linear_attention_te(args.dtype, args.normalized)
    results, elapsed = search(console, out, inputs, args)
    found = [r for r in results if not materialises_scores(r)]
    console.print(
        f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s, "
        f"{len(found)} of them without the Nq x N score matrix"
    )
    for n, r in enumerate(found if args.only_found else results):
        tag = "" if materialises_scores(r) else "  [green]linear in N[/]"
        show(console, n, r, args, tag)
        if args.check:
            err = check(r, inputs, args.sizes, args.seed, args.normalized)
            console.print(f"LLVM vs numpy at {args.sizes}: max relative error {err:.3g}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
