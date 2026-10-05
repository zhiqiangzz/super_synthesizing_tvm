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

"""Superoptimize pairwise squared distances -- and watch the precision gate reject the result.

The target is the definition, with symbolic shapes::

    D[i, j] = Σ_k (X[i, k] - Y[j, k])²          X: M x d,  Y: N x d

Expanding the square gives ``|x_i|² + |y_j|² - 2 x_i·y_j``: two row norms
and a GEMM, the form every BLAS-backed library uses. Over the reals the two
are the same polynomial (they even share a canonical form, so the proof is
by hash). The search finds the expansion; with ``--budget 4`` as one
three-state reducer ``(Σx², Σy², Σxy)`` plus the combination, the pure GEMM
form (norms, matmul, broadcast adds) takes ``--budget 6``.

In floating point the expansion cancels catastrophically when the points
are far from the origin compared with their distances: ``|x|²`` and ``2x·y``
are ~1e8 for coordinates ~1e4 while ``D`` is ~1. The precision gate measures
exactly that (relative error ~10 on the "offset" family against ~1e-6 for
the original), so under the default ``--accuracy filter`` nothing is kept.
This example therefore defaults to ``--accuracy report``: every equivalent
program is shown with the gate's verdict.

    python super_opt/examples/superopt_pairwise_distance.py
    python super_opt/examples/superopt_pairwise_distance.py --accuracy filter

See ``--help``.
"""

from __future__ import annotations

import sys

import numpy as np
from harness.superopt import make_parser, run_llvm, search, shape_of, show
from rich.console import Console

from tvm import te


def pairwise_te(dtype: str):
    """``Σ_k (X[i, k] - Y[j, k])²`` as ``(inputs, output)`` te.Tensors."""
    M, N, d = te.var("M"), te.var("N"), te.var("d")
    X = te.placeholder((M, d), name="X", dtype=dtype)
    Y = te.placeholder((N, d), name="Y", dtype=dtype)
    k = te.reduce_axis((0, d), name="k")
    D = te.compute(
        (M, N), lambda i, j: te.sum((X[i, k] - Y[j, k]) * (X[i, k] - Y[j, k]), axis=k), name="D"
    )
    return [X, Y], D


def expanded(result) -> bool:
    """Does the program use the expansion (a matmul, or a reducer of several moments)?"""
    for rec in result.snapshot.ops:
        if rec.spec.name == "matmul":
            return True
        if rec.spec.name == "comm_reduce" and rec.params.spec.arity > 1:
            return True
    return False


def check(result, inputs, sizes, seed: int, offset: float) -> float:
    """Max relative error of the LLVM build against float64 numpy, points around ``offset``."""
    rng = np.random.default_rng(seed)
    dtype = str(inputs[0].dtype)
    x, y = ((offset + rng.standard_normal(shape_of(t, sizes))).astype(dtype) for t in inputs)
    got = run_llvm(result, inputs, [x, y], sizes).astype("float64")
    diff = x.astype("float64")[:, None, :] - y.astype("float64")[None, :, :]
    want = (diff * diff).sum(axis=2)
    return float(np.abs(got - want).max() / np.abs(want).max())


def main(argv: list[str] | None = None) -> int:
    parser = make_parser(__doc__, budget=4, sizes="M=64,N=64,d=32")
    parser.set_defaults(accuracy="report")
    args = parser.parse_args(argv)
    console = Console()
    inputs, out = pairwise_te(args.dtype)
    results, elapsed = search(console, out, inputs, args)
    found = [r for r in results if expanded(r)]
    kept = [r for r in found if r.accuracy is None or r.accuracy.ok]
    console.print(
        f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s, "
        f"{len(found)} of them expanding the square, {len(kept)} of those accepted by the "
        "precision gate"
    )
    for n, r in enumerate(found if args.only_found else results):
        show(console, n, r, args, "  [yellow]expanded[/]" if expanded(r) else "")
        if args.check:
            for offset in (0.0, 1.0e4):
                err = check(r, inputs, args.sizes, args.seed, offset)
                console.print(
                    f"LLVM vs numpy at {args.sizes}, points around {offset:g}: "
                    f"max relative error {err:.3g}"
                )
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
