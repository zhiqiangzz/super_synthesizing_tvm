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

"""Superoptimize a Gram matrix of a projection: (A X)(A X)ᵀ -> A (X Xᵀ) Aᵀ.

The target projects first and correlates the long axis afterwards::

    Y[i, n]  = Σ_k A[i, k] X[k, n]            A: M x K,  X: K x N
    C[i, i'] = Σ_n Y[i, n] Y[i', n]           C: M x M

That is ``M K N + M² N`` multiply-adds and an ``M x N`` intermediate. When
the long axis ``N`` dominates, correlating the inputs first is far cheaper:
``G = X Xᵀ`` (``K² N``), then ``A G Aᵀ`` (``M K² + M² K``), with only
``K x K`` and ``M x K`` intermediates.

Two things the search needs here beyond the matrix chain. ``X Xᵀ`` and
``C`` have two axes of the same extent that must *not* be shared (an outer
product, not a batch): matmul offers that for extents the target itself uses
twice (``C[i, i']``; the two sums over ``k``). And ``A (X Xᵀ)`` is not a
canonical sub-term of the target -- the sums nest the other way -- so it
survives pruning through its sum-product form, whose tensors and summed axes
fit inside the target's single prenex term ``Σ_n Σ_k Σ_k' A A X X``.

    python super_opt/examples/superopt_gram.py
    python super_opt/examples/superopt_gram.py --check --te --no-ir

See ``--help``.
"""

from __future__ import annotations

import sys

import numpy as np
from harness.superopt import make_parser, run_llvm, search, shape_of, show
from rich.console import Console

from tvm import te


def gram_te(dtype: str):
    """``(A X)(A X)ᵀ`` as ``(inputs, output)`` te.Tensors with symbolic shapes."""
    M, K, N = te.var("M"), te.var("K"), te.var("N")
    A = te.placeholder((M, K), name="A", dtype=dtype)
    X = te.placeholder((K, N), name="X", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    Y = te.compute((M, N), lambda i, n: te.sum(A[i, k] * X[k, n], axis=k), name="Y")
    n = te.reduce_axis((0, N), name="n")
    C = te.compute((M, M), lambda i, i2: te.sum(Y[i, n] * Y[i2, n], axis=n), name="C")
    return [A, X], C


def correlates_inputs_first(result) -> bool:
    """Does the program start with ``X Xᵀ`` (an op reading the input X twice)?"""
    x = result.inputs.index(next(t for t in result.inputs if t.op.name == "X"))
    return any(tuple(rec.operands) == (x, x) for rec in result.snapshot.ops)


def check(result, inputs, sizes, seed: int) -> float:
    rng = np.random.default_rng(seed)
    dtype = str(inputs[0].dtype)
    arrays = [rng.standard_normal(shape_of(t, sizes)).astype(dtype) for t in inputs]
    got = run_llvm(result, inputs, arrays, sizes).astype("float64")
    a, x = (v.astype("float64") for v in arrays)
    want = (a @ x) @ (a @ x).T
    return float(np.abs(got - want).max() / np.abs(want).max())


def main(argv: list[str] | None = None) -> int:
    parser = make_parser(__doc__, budget=3, sizes="M=64,K=16,N=4096")
    args = parser.parse_args(argv)
    console = Console()
    inputs, out = gram_te(args.dtype)
    results, elapsed = search(console, out, inputs, args)
    found = [r for r in results if correlates_inputs_first(r)]
    console.print(
        f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s, "
        f"{len(found)} of them computing A (X Xᵀ) Aᵀ"
    )
    for n, r in enumerate(found if args.only_found else results):
        tag = "  [green]A (X Xᵀ) Aᵀ[/]" if correlates_inputs_first(r) else ""
        show(console, n, r, args, tag)
        if args.check:
            err = check(r, inputs, args.sizes, args.seed)
            console.print(f"LLVM vs numpy at {args.sizes}: max relative error {err:.3g}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
