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

"""Superoptimize the textbook TE attention and print every equivalent program found.

Starts from ``attention()`` in ``flashattn_te.py`` (matmul -> max -> exp -> sum ->
divide -> matmul, seven blocks, symbolic shapes) and searches the bounded TE
grammar -- including automatically synthesised tuple ``te.comm_reducer``
reductions -- for equivalent programs. Two tensor ops already suffice for the
online-softmax (FlashAttention) formulation: ``S = Q Kᵀ`` and one reducer over
the keys, either with normalised output states (FA1-style merge, read off
the original program) or with ``(m, l, o)`` and the division ``o / l`` as
its fused epilogue (FA2-style).

By default only programs at least as accurate as the original are kept (the
precision gate of ``tvm.te.superopt.accuracy``): the unshifted ``exp`` of an
``(L, O)`` reducer is rejected, the max-shifted online softmax passes.
``--accuracy report`` keeps every equivalent program and prints its verdict.

    python super_opt/examples/superopt_attention.py
    python super_opt/examples/superopt_attention.py --budget 6 --leaf-exp --no-ir

See ``--help``.
"""

from __future__ import annotations

import argparse
import sys
import time

from harness import render_source
from rich.console import Console

from tvm import te
from tvm import tirx as tir
from tvm.te.superopt import Bounds, superoptimize


def attention_te(dtype: str):
    """The unfused attention of ``flashattn_te.attention`` as ``(inputs, output)`` tensors.

    The superoptimizer works on ``te.Tensor`` and on a single dtype (no
    ``astype`` anywhere), so the same graph is rebuilt here without the
    accumulation casts of the original.
    """
    n_b, n_h, n_q, n_k, dim = (
        te.var(n) for n in ("batch", "num_heads", "seqlen_q", "seqlen_k", "head_dim")
    )
    Q = te.placeholder((n_b, n_h, n_q, dim), name="Q", dtype=dtype)
    K = te.placeholder((n_b, n_h, n_k, dim), name="K", dtype=dtype)
    V = te.placeholder((n_b, n_h, n_k, dim), name="V", dtype=dtype)
    scale = tir.const(1.0, dtype) / tir.sqrt(dim.astype(dtype))
    d = te.reduce_axis((0, dim), name="d")
    S = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: te.sum(Q[b, h, i, d] * K[b, h, j, d], axis=d),
        name="S",
    )
    j_max = te.reduce_axis((0, n_k), name="j")
    row_max = te.compute(
        (n_b, n_h, n_q), lambda b, h, i: te.max(S[b, h, i, j_max], axis=j_max), name="row_max"
    )
    exp_S = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: tir.exp((S[b, h, i, j] - row_max[b, h, i]) * scale),
        name="exp_S",
    )
    j_sum = te.reduce_axis((0, n_k), name="j")
    denominator = te.compute(
        (n_b, n_h, n_q), lambda b, h, i: te.sum(exp_S[b, h, i, j_sum], axis=j_sum), name="den"
    )
    P = te.compute(
        (n_b, n_h, n_q, n_k), lambda b, h, i, j: exp_S[b, h, i, j] / denominator[b, h, i], name="P"
    )
    j_pv = te.reduce_axis((0, n_k), name="j")
    PV = te.compute(
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: te.sum(P[b, h, i, j_pv] * V[b, h, j_pv, e], axis=j_pv),
        name="PV",
    )
    return [Q, K, V], PV


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    search = parser.add_argument_group("search bounds")
    search.add_argument(
        "--budget", type=int, default=2, help="max tensor ops (default: %(default)s)"
    )
    search.add_argument(
        "--states", type=int, default=3, help="max reducer arity (default: %(default)s)"
    )
    search.add_argument(
        "--min-states",
        type=int,
        default=2,
        help="min reducer arity; 1 also synthesises plain sum/max reducers (default: %(default)s)",
    )
    search.add_argument(
        "--leaf-exp", action="store_true", help="allow exp inside reducer input tuples"
    )
    search.add_argument(
        "--leaf-nodes",
        type=int,
        default=4,
        help="node budget per reducer input expression (default: %(default)s)",
    )
    numerics = parser.add_argument_group("numerics")
    numerics.add_argument("--dtype", default="float16", help="(default: %(default)s)")
    numerics.add_argument(
        "--accuracy",
        choices=("filter", "report", "off"),
        default="filter",
        help="precision gate: drop programs less accurate than the original (filter), "
        "only annotate them (report), or skip the gate (default: %(default)s)",
    )
    output = parser.add_argument_group("output")
    output.add_argument("--no-ir", dest="show_ir", action="store_false", help="skip the s_tir dump")
    output.add_argument(
        "--te", dest="show_te", action="store_true", help="print each program as TE Python source"
    )
    output.add_argument("--max-results", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    inputs, out = attention_te(args.dtype)
    bounds = Bounds(
        max_tensor_ops=args.budget,
        max_states=args.states,
        min_states=args.min_states,
        leaf_exp=args.leaf_exp,
        max_leaf_nodes=args.leaf_nodes,
    )
    t0 = time.time()
    results = superoptimize(
        out, inputs, bounds, max_results=args.max_results, accuracy=args.accuracy
    )
    elapsed = time.time() - t0
    console.print(f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s")
    for n, r in enumerate(results):
        console.rule(f"program {n}: {r.n_ops} ops, verified by {r.verdict.method}")
        console.print(r.snapshot.describe())
        if r.accuracy is not None:
            console.print(r.accuracy.summary())
        if args.show_te:
            render_source(
                console,
                r.te_source(),
                lexer="python",
                title=f"[bold]TE source[/] — program {n}",
                subtitle="runnable te.compute / te.comm_reducer definition",
            )
        if args.show_ir:
            render_source(
                console,
                r.prim_func().script(),
                lexer="python",
                title=f"[bold]s_tir[/] — program {n}",
                subtitle="te.create_prim_func of the discovered TE",
            )
    if results:
        stats = results[0].ctx.stats
        console.print({k: v for k, v in sorted(stats.items()) if not k.startswith("prune:order")})
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
