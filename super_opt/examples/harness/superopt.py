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
"""Shared pieces of the ``superopt_*`` examples: CLI, result display, op counts.

The superoptimizer itself has no cost model; :func:`op_counts` only puts
numbers next to the programs it finds (multiply-adds and materialised
elements for concrete sizes) so the effect of a rewrite is visible.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
from rich.console import Console

import tvm
from tvm import tirx as tir
from tvm.te.superopt import Bounds, superoptimize

from .render import render_source


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def parse_sizes(text: str) -> dict[str, int]:
    """``"M=1000,K=10"`` -> ``{"M": 1000, "K": 10}``."""
    out = {}
    for part in filter(None, text.split(",")):
        name, value = part.split("=")
        out[name.strip()] = int(value)
    return out


def make_parser(doc: str, *, budget: int, states: int = 3, min_states: int = 2, sizes: str = ""):
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    search = parser.add_argument_group("search bounds")
    search.add_argument(
        "--budget", type=int, default=budget, help="max tensor ops (default: %(default)s)"
    )
    search.add_argument(
        "--states", type=int, default=states, help="max reducer arity (default: %(default)s)"
    )
    search.add_argument(
        "--min-states",
        type=int,
        default=min_states,
        help="min reducer arity; 1 also synthesises plain sum/max reducers (default: %(default)s)",
    )
    search.add_argument(
        "--leaf-nodes",
        type=int,
        default=4,
        help="node budget per reducer input expression (default: %(default)s)",
    )
    numerics = parser.add_argument_group("numerics")
    numerics.add_argument("--dtype", default="float32", help="(default: %(default)s)")
    numerics.add_argument(
        "--accuracy",
        choices=("filter", "report", "off"),
        default="filter",
        help="precision gate: drop programs less accurate than the original (filter), "
        "only annotate them (report), or skip the gate (default: %(default)s)",
    )
    numerics.add_argument(
        "--check",
        action="store_true",
        help="compile each shown program for LLVM and compare it with a numpy reference",
    )
    numerics.add_argument(
        "--sizes",
        type=parse_sizes,
        default=parse_sizes(sizes),
        help="symbolic extents for --check and the op counts, e.g. M=64,K=8 (default: %(default)s)",
    )
    numerics.add_argument("--seed", type=int, default=0, help="(default: %(default)s)")
    output = parser.add_argument_group("output")
    output.add_argument(
        "--all",
        dest="only_found",
        action="store_false",
        help="show every program, not only those with the transformation this example is about",
    )
    output.add_argument("--no-ir", dest="show_ir", action="store_false", help="skip the s_tir dump")
    output.add_argument(
        "--te", dest="show_te", action="store_true", help="print each program as TE Python source"
    )
    output.add_argument("--max-results", type=int, default=None)
    return parser


def bounds_of(args, **extra) -> Bounds:
    return Bounds(
        max_tensor_ops=args.budget,
        max_states=args.states,
        min_states=args.min_states,
        max_leaf_nodes=args.leaf_nodes,
        **extra,
    )


# ---------------------------------------------------------------------------
# op counts
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class OpCount:
    op: str
    elements: int  # elements written (the op's outputs)
    madds: int  # multiply-adds (or elementwise operations) performed


def _extent(ctx, key, sizes: dict[str, int]) -> int:
    ext = ctx.dims.extent(key)
    if isinstance(ext, tir.IntImm):
        return int(ext.value)
    return sizes[ext.name]


def op_counts(result, sizes: dict[str, int]) -> list[OpCount]:
    """Per op of ``result``: elements written and multiply-adds, for concrete ``sizes``."""
    ctx = result.ctx
    snap = result.snapshot
    out = []
    for rec in snap.ops:
        outs = [snap.pool[i] for i in rec.outputs]
        elements = sum(int(np.prod([_extent(ctx, k, sizes) for k in e.shape])) for e in outs)
        per_out = int(np.prod([_extent(ctx, k, sizes) for k in outs[0].shape]))
        name = rec.spec.name
        if name == "matmul":
            a = snap.pool[rec.operands[0]]
            madds = per_out * _extent(ctx, a.shape[rec.params[0]], sizes)
        elif name in ("sum", "max"):
            a = snap.pool[rec.operands[0]]
            madds = int(np.prod([_extent(ctx, k, sizes) for k in a.shape]))
        elif name == "comm_reduce":
            spec = rec.params.spec
            madds = per_out * _extent(ctx, spec.axis, sizes) * spec.arity
        else:
            madds = per_out
        out.append(OpCount(name, elements, madds))
    return out


def count_summary(result, sizes: dict[str, int]) -> str:
    counts = op_counts(result, sizes)
    madds = sum(c.madds for c in counts)
    inner = sum(c.elements for c in counts[:-1])
    return f"{madds:.3g} multiply-adds, {inner:.3g} intermediate elements materialised"


# ---------------------------------------------------------------------------
# running and display
# ---------------------------------------------------------------------------
def shape_of(tensor, sizes: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(s.value) if isinstance(s, tir.IntImm) else sizes[s.name] for s in tensor.shape)


def run_llvm(result, inputs, arrays, sizes: dict[str, int]) -> np.ndarray:
    """Compile a found program for LLVM and run it on ``arrays``."""
    out = result.materialize()
    lib = tvm.compile(tvm.IRModule({"main": result.prim_func()}), target="llvm")
    args = [tvm.runtime.tensor(a) for a in arrays]
    args.append(tvm.runtime.tensor(np.zeros(shape_of(out, sizes), dtype=str(out.dtype))))
    lib["main"](*args)
    return args[-1].numpy()


def search(console: Console, out, inputs, args, **extra):
    t0 = time.time()
    results = superoptimize(
        out,
        inputs,
        bounds_of(args, **extra),
        max_results=args.max_results,
        accuracy=args.accuracy,
    )
    return results, time.time() - t0


def show(console: Console, n: int, result, args, tag: str = "") -> None:
    console.rule(f"program {n}: {result.n_ops} ops, verified by {result.verdict.method}{tag}")
    console.print(result.snapshot.describe())
    if result.accuracy is not None:
        console.print(result.accuracy.summary())
    if args.sizes:
        try:
            console.print(f"at {args.sizes}: {count_summary(result, args.sizes)}")
        except KeyError:
            pass
    if args.show_te:
        render_source(
            console,
            result.te_source(),
            lexer="python",
            title=f"[bold]TE source[/] — program {n}",
            subtitle="runnable te.compute / te.comm_reducer definition",
        )
    if args.show_ir:
        render_source(
            console,
            result.prim_func().script(),
            lexer="python",
            title=f"[bold]s_tir[/] — program {n}",
            subtitle="te.create_prim_func of the discovered TE",
        )


__all__ = [
    "OpCount",
    "bounds_of",
    "count_summary",
    "make_parser",
    "op_counts",
    "parse_sizes",
    "run_llvm",
    "search",
    "shape_of",
    "show",
]
