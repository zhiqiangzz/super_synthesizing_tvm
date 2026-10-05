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

"""Superoptimize an HMM likelihood written as a sum over all paths into dynamic programming.

For a batch of observation sequences of fixed length ``T`` (emission
likelihoods ``B_t[b, s]`` given per step), the target is the definition --
a sum over all ``S^T`` hidden paths::

    P[b] = Σ_{s1..sT} π[s1] B1[b, s1] A[s1, s2] B2[b, s2] ... A[s(T-1), sT] BT[b, sT]

which costs ``O(S^T)`` per sequence. The search finds the backward/forward
algorithm, i.e. *variable elimination*: sum one hidden state out at a time,
each step a matrix-vector product and an elementwise scaling,

    β[b, s2] = Σ_{s3} A[s2, s3] B3[b, s3];   β'[b, s1] = Σ_{s2} A[s1, s2] (B2 β)[b, s2];  ...

``O(T S²)`` per sequence. The rewrite theory has no dynamic-programming rule:
elimination orders are ordinary matmul / mul programs, proven equal to the
path sum by the sum-product normal form.

The eliminated program has ``2T - 1`` ops: ``T = 2`` (three ops, ``S²``
either way) is found in seconds; ``T = 3`` (``S³`` against ``S²``) needs a
five-op search, which takes far longer.

What stays out of reach is a recurrence over a *symbolic* sequence length --
``f_t = (f_{t-1} A) ⊙ B_t`` for ``t = 1..T`` is a scan along an ordered axis,
not a fixed program, and not a commutative reduction.

    python super_opt/examples/superopt_hmm_chain.py --check --te --no-ir   # T = 2
    python super_opt/examples/superopt_hmm_chain.py --steps 3                # slow

See ``--help``.
"""

from __future__ import annotations

import sys

import numpy as np
from harness.superopt import make_parser, run_llvm, search, shape_of, show
from rich.console import Console

from tvm import te


def hmm_chain_te(dtype: str, steps: int):
    """The path-sum likelihood of ``steps`` observations as ``(inputs, output)``."""
    S, batch = te.var("S"), te.var("batch")
    pi = te.placeholder((S,), name="pi", dtype=dtype)
    A = te.placeholder((S, S), name="A", dtype=dtype)
    Bs = [te.placeholder((batch, S), name=f"B{t + 1}", dtype=dtype) for t in range(steps)]
    ss = [te.reduce_axis((0, S), name=f"s{t + 1}") for t in range(steps)]

    def path(b):
        e = pi[ss[0]] * Bs[0][b, ss[0]]
        for t in range(1, steps):
            e = e * A[ss[t - 1], ss[t]] * Bs[t][b, ss[t]]
        return te.sum(e, axis=ss)

    P = te.compute((batch,), path, name="P")
    return [pi, A, *Bs], P


def eliminates_one_at_a_time(result) -> bool:
    """No intermediate holds two hidden-state axes for a sequence (``[b, s, s']``)."""
    ctx, snap = result.ctx, result.snapshot
    s, b = ctx.dims.key_of_name("S"), ctx.dims.key_of_name("batch")
    for rec in snap.ops:
        for o in rec.outputs:
            shape = snap.pool[o].shape
            if b in shape and shape.count(s) >= 2:
                return False
    return True


def check(result, inputs, sizes, seed: int) -> float:
    """Max relative error of the LLVM build against the forward algorithm in float64."""
    rng = np.random.default_rng(seed)
    dtype = str(inputs[0].dtype)
    arrays = [rng.random(shape_of(t, sizes)).astype(dtype) for t in inputs]
    got = run_llvm(result, inputs, arrays, sizes).astype("float64")
    pi, A, *Bs = (x.astype("float64") for x in arrays)
    alpha = pi[None, :] * Bs[0]
    for B in Bs[1:]:
        alpha = (alpha @ A) * B
    want = alpha.sum(axis=1)
    return float(np.abs(got - want).max() / np.abs(want).max())


def main(argv: list[str] | None = None) -> int:
    parser = make_parser(__doc__, budget=0, sizes="S=64,batch=32")
    parser.add_argument("--steps", type=int, default=2, help="sequence length T (default: 2)")
    args = parser.parse_args(argv)
    if args.budget <= 0:
        args.budget = 2 * args.steps - 1
    console = Console()
    inputs, out = hmm_chain_te(args.dtype, args.steps)
    results, elapsed = search(console, out, inputs, args)
    found = [r for r in results if eliminates_one_at_a_time(r)]
    console.print(
        f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s, "
        f"{len(found)} of them eliminating one hidden state at a time"
    )
    for n, r in enumerate(found if args.only_found else results):
        tag = "  [green]variable elimination[/]" if eliminates_one_at_a_time(r) else ""
        show(console, n, r, args, tag)
        if args.check:
            err = check(r, inputs, args.sizes, args.seed)
            console.print(f"LLVM vs numpy at {args.sizes}: max relative error {err:.3g}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
