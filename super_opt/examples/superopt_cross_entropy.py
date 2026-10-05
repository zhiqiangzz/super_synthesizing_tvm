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

"""Superoptimize a linear + cross-entropy head into one streaming pass (Cut Cross-Entropy).

The target is the textbook LM head: logits, a stable log-sum-exp in three
passes over the vocabulary, and the target logit, with symbolic shapes::

    L[i, v] = Σ_d X[i, d] W[v, d]                 N x V logits
    m[i]    = max_v L[i, v]
    s[i]    = Σ_v exp(L[i, v] - m[i])
    t[i]    = Σ_v Y[i, v] L[i, v]                 Y: one-hot labels
    loss[i] = m[i] + log s[i] - t[i]

Every one of the three reductions re-reads the ``N x V`` logits. The search
finds a single ``comm_reducer`` over the vocabulary carrying
``(running max, rescaled sum of exp, target logit)`` -- the online
log-sum-exp of Cut Cross-Entropy -- so the logits are read once, by one
reduction, and a later s_TIR pass can compute them inside it instead of
materialising them (as for ``S`` in FlashAttention). Its states are read off
the original's ``m``, ``s`` and ``t`` (partialisation), and the merge keeps
the max shift: ``s = exp(m_a - m) s_a + exp(m_b - m) s_b``. The loss
``m + log s - t`` is the reducer's epilogue, derived with it and fused into
the same search op, so the whole program is two ops (``--budget 2``); with
separate elementwise ops it takes five and a half-hour search.

The labels are a one-hot matrix: selecting ``L[i, y_i]`` with integer labels
is a data-dependent index, which the symbolic semantics cannot express.

    python super_opt/examples/superopt_cross_entropy.py
    python super_opt/examples/superopt_cross_entropy.py --check --te --no-ir

See ``--help``.
"""

from __future__ import annotations

import sys

import numpy as np
from harness.superopt import make_parser, run_llvm, search, shape_of, show
from rich.console import Console

from tvm import te
from tvm import tirx as tir


def cross_entropy_te(dtype: str):
    """Per-row cross entropy of ``X Wᵀ`` against one-hot ``Y`` as ``(inputs, output)``."""
    N, D, V = te.var("N"), te.var("D"), te.var("V")
    X = te.placeholder((N, D), name="X", dtype=dtype)
    W = te.placeholder((V, D), name="W", dtype=dtype)
    Y = te.placeholder((N, V), name="Y", dtype=dtype)
    d = te.reduce_axis((0, D), name="d")
    L = te.compute((N, V), lambda i, v: te.sum(X[i, d] * W[v, d], axis=d), name="logits")
    v1 = te.reduce_axis((0, V), name="v")
    m = te.compute((N,), lambda i: te.max(L[i, v1], axis=v1), name="m")
    e = te.compute((N, V), lambda i, v: tir.exp(L[i, v] - m[i]), name="e")
    v2 = te.reduce_axis((0, V), name="v")
    s = te.compute((N,), lambda i: te.sum(e[i, v2], axis=v2), name="s")
    v3 = te.reduce_axis((0, V), name="v")
    t = te.compute((N,), lambda i: te.sum(Y[i, v3] * L[i, v3], axis=v3), name="t")
    loss = te.compute((N,), lambda i: m[i] + tir.log(s[i]) - t[i], name="loss")
    return [X, W, Y], loss


def single_pass(result) -> bool:
    """Is the ``N x V`` logits tensor read by exactly one op, a streaming reducer?"""
    ctx, snap = result.ctx, result.snapshot
    nv = {ctx.dims.key_of_name("N"), ctx.dims.key_of_name("V")}
    wide = {i for i, e in enumerate(snap.pool) if set(e.shape) == nv and i >= snap.n_inputs}
    readers = [rec for rec in snap.ops if wide & set(rec.operands)]
    return len(readers) == 1 and readers[0].spec.name == "comm_reduce"


def check(result, inputs, sizes, seed: int) -> float:
    """Max relative error of the LLVM build against a float64 numpy cross entropy."""
    rng = np.random.default_rng(seed)
    dtype = str(inputs[0].dtype)
    x, w = (rng.standard_normal(shape_of(t, sizes)).astype(dtype) for t in inputs[:2])
    n, v = shape_of(inputs[2], sizes)
    y = np.zeros((n, v), dtype=dtype)
    y[np.arange(n), rng.integers(0, v, n)] = 1
    got = run_llvm(result, inputs, [x, w, y], sizes).astype("float64")
    logits = x.astype("float64") @ w.astype("float64").T
    mx = logits.max(axis=1)
    want = mx + np.log(np.exp(logits - mx[:, None]).sum(axis=1)) - (y * logits).sum(axis=1)
    return float(np.abs(got - want).max() / np.abs(want).max())


def main(argv: list[str] | None = None) -> int:
    parser = make_parser(__doc__, budget=2, sizes="N=64,D=32,V=4096")
    args = parser.parse_args(argv)
    console = Console()
    inputs, out = cross_entropy_te(args.dtype)
    results, elapsed = search(console, out, inputs, args)
    found = [r for r in results if single_pass(r)]
    console.print(
        f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s, "
        f"{len(found)} of them reading the logits in a single streaming pass"
    )
    for n, r in enumerate(found if args.only_found else results):
        show(console, n, r, args, "  [green]single pass[/]" if single_pass(r) else "")
        if args.check:
            err = check(r, inputs, args.sizes, args.seed)
            console.print(f"LLVM vs numpy at {args.sizes}: max relative error {err:.3g}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
