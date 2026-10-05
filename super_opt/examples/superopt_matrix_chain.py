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

"""Superoptimize a matrix chain and rediscover the other association order.

The target multiplies left to right, with symbolic shapes::

    AB[m, n] = Σ_k A[m, k] B[k, n]          A: M x K,  B: K x N,  C: N x P
    D[m, p]  = Σ_n AB[m, n] C[n, p]

``(A B) C`` costs ``M K N + M N P`` multiply-adds and materialises an
``M x N`` intermediate; ``A (B C)`` costs ``K N P + M K P`` and materialises
``K x P``. For ``M = N = 10000, K = P = 10`` that is 2e9 against 2e6.

The search knows no associativity rule. Both orders are plain two-matmul
programs of the grammar; what makes ``A (B C)`` an answer is the proof. The
canonical form keeps sums where a program nests them (``Σ_n (Σ_k A B) C``
against ``Σ_k A (Σ_n B C)``), so the two are proven equal by the sum-product
normal form instead: both expand to ``Σ_k Σ_n A B C``. The superoptimizer
has no cost model; the multiply-add counts printed next to each program are
for ``--sizes`` only.

    python super_opt/examples/superopt_matrix_chain.py
    python super_opt/examples/superopt_matrix_chain.py --sizes M=10000,K=10,N=10000,P=10
    python super_opt/examples/superopt_matrix_chain.py --check --te --no-ir

See ``--help``.
"""

from __future__ import annotations

import sys

import numpy as np
from harness.superopt import make_parser, run_llvm, search, shape_of, show
from rich.console import Console

from tvm import te


def chain_te(dtype: str):
    """``(A @ B) @ C`` as ``(inputs, output)`` te.Tensors with symbolic shapes."""
    M, K, N, P = (te.var(n) for n in "MKNP")
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    C = te.placeholder((N, P), name="C", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    AB = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="AB")
    n = te.reduce_axis((0, N), name="n")
    D = te.compute((M, P), lambda m, p: te.sum(AB[m, n] * C[n, p], axis=n), name="D")
    return [A, B, C], D


def association(result) -> str:
    """``"(AB)C"`` or ``"A(BC)"``: which inputs the first matmul multiplies."""
    first = result.snapshot.ops[0]
    names = {result.inputs[i].op.name for i in first.operands if i < len(result.inputs)}
    if names == {"A", "B"}:
        return "(AB)C"
    return "A(BC)" if names == {"B", "C"} else "?"


def check(result, inputs, sizes, seed: int) -> float:
    """Max relative error of the LLVM build against ``numpy``'s ``A @ B @ C``."""
    rng = np.random.default_rng(seed)
    dtype = str(inputs[0].dtype)
    arrays = [rng.standard_normal(shape_of(t, sizes)).astype(dtype) for t in inputs]
    got = run_llvm(result, inputs, arrays, sizes).astype("float64")
    a, b, c = (x.astype("float64") for x in arrays)
    want = a @ b @ c
    return float(np.abs(got - want).max() / np.abs(want).max())


def main(argv: list[str] | None = None) -> int:
    parser = make_parser(__doc__, budget=2, sizes="M=1000,K=10,N=1000,P=10")
    args = parser.parse_args(argv)
    console = Console()
    inputs, out = chain_te(args.dtype)
    results, elapsed = search(console, out, inputs, args)
    found = [r for r in results if association(r) == "A(BC)"]
    console.print(
        f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s, "
        f"{len(found)} of them associating A (B C)"
    )
    for n, r in enumerate(found if args.only_found else results):
        show(console, n, r, args, f"  [green]{association(r)}[/]")
        if args.check:
            err = check(r, inputs, args.sizes, args.seed)
            console.print(f"LLVM vs numpy at {args.sizes}: max relative error {err:.3g}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
