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

"""Superoptimize a two-pass row variance and rediscover Welford's online algorithm.

The target is the textbook definition, one pass for the mean and a second for
the squared deviations, with symbolic shapes::

    mean[i] = Σ_j x[i, j] / cols
    var[i]  = Σ_j (x[i, j] - mean[i])² / cols

The search is told nothing about Welford. For the ``cols`` axis the reducer
synthesiser enumerates state definitions over the target's own partial sums
(Σx, Σx²) plus at most one latent partial reduction (a count, a max, ...), and
derives each reducer's input tuple, identity and merge from the definition.
Welford's ``(count, mean, M2)`` is among the reducers it keeps: the count is the
one extra quantity that turns the per-element input of the second moment into
``x² - x²/1 = 0``.

The same states also come out of *partialisation* -- running the original's
``total``, ``mean`` and ``ss`` on a sub-range of the axis -- which additionally
prints the merge the way the original computes, around each side's own mean:
``M2 = M2_a + M2_b + n_a (mean - mean_a)² + n_b (mean - mean_b)²``. The
canonical merge (``Σx² - (Σx)²/n`` expanded) is only the equivalence proof;
the precision gate rejects it on data with a large mean, and keeps the
re-based one.

    python super_opt/examples/superopt_welford.py
    python super_opt/examples/superopt_welford.py --welford-only --te --no-ir --check
    python super_opt/examples/superopt_welford.py --welford-only --explain --no-ir

See ``--help``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
from harness import render_source
from rich.console import Console

import tvm
from tvm import te
from tvm.te.superopt import Bounds, superoptimize
from tvm.te.superopt.reducer import grammar
from tvm.te.superopt.reducer.synth import ReducerSpec, collect_goals
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import (
    mk_card,
    mk_div,
    mk_pow,
    mk_reduce,
    mk_sub,
    subst_domain,
)


def variance_te(dtype: str):
    """Two-pass row variance as ``(inputs, output)`` te.Tensors with symbolic shapes."""
    rows, cols = te.var("rows"), te.var("cols")
    X = te.placeholder((rows, cols), name="X", dtype=dtype)
    cols_f = cols.astype(dtype)
    j1 = te.reduce_axis((0, cols), name="j")
    total = te.compute((rows,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((rows,), lambda i: total[i] / cols_f, name="mean")
    dev2 = te.compute(
        (rows, cols), lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), name="dev2"
    )
    j2 = te.reduce_axis((0, cols), name="j")
    ss = te.compute((rows,), lambda i: te.sum(dev2[i, j2], axis=j2), name="ss")
    var = te.compute((rows,), lambda i: ss[i] / cols_f, name="var")
    return [X], var


# ---------------------------------------------------------------------------
# Recognising Welford among the synthesised reducers
# ---------------------------------------------------------------------------
def welford_states(spec: ReducerSpec) -> tuple[ir.SymExpr, ...] | None:
    """Welford's state definitions ``(n, mean, M2)`` over the spec's own partial domain.

    Built from the canonical constructors, so comparing them with the
    synthesised states by identity is an exact algebraic check:

        n(R)    = |R|
        mean(R) = Σ_R x / |R|
        M2(R)   = Σ_R x² - (Σ_R x)² / |R|
    """
    x = next((leaf for leaf in spec.leaves if isinstance(leaf, ir.Elem)), None)
    if x is None:
        return None
    R = ir.dsym(spec.axis, "R")
    n = mk_card(R)
    s1 = mk_reduce("sum", R, 0, x)
    s2 = mk_reduce("sum", R, 0, mk_pow(x, 2))
    return n, mk_div(s1, n), mk_sub(s2, mk_div(mk_pow(s1, 2), n))


def is_welford(spec: ReducerSpec) -> bool:
    expected = welford_states(spec)
    return expected is not None and set(spec.states) == set(expected)


def result_is_welford(result) -> bool:
    return any(is_welford(rec.spec) for rec in result.reducers())


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def readable(e, ctx) -> str:
    """Symbolic expressions with placeholder / axis names instead of internal ids."""
    names = {ctx.lower.tensor_id(t): t.op.name.lower() for t in ctx.lower.placeholders}
    s = str(e)
    s = re.sub(r"T(\d+)\[[^\]]*\]", lambda m: names.get(int(m.group(1)), m.group(0)), s)
    s = re.sub(r"(sum|max)_\{j\d in ([A-Z])@ax\d+\}", r"\1_\2", s)
    s = re.sub(
        r"(sum|max)_\{j\d in ax(\d+)\}",
        lambda m: f"{m.group(1)}_{ctx.dims.name(int(m.group(2)))}",
        s,
    )
    s = re.sub(r"([A-Z])@ax\d+", r"\1", s)
    return s


def explain(console: Console, results, ctx, axis) -> None:
    """How the reducer for ``axis`` was synthesised: goals, atoms, derived Welford reducer."""
    full, R = ir.dfull(axis), ir.dsym(axis, "R")
    goals = collect_goals(ctx.target.body, axis)[0]
    goal_atoms = [subst_domain(g, {full: R}) for g in goals]
    sigs = grammar.signals(goals)
    atoms = grammar.atom_candidates(sigs, R)
    console.rule("[bold]reducer synthesis over the cols axis")
    console.print("target  :", readable(ctx.target.body, ctx))
    console.print("goals   :", [readable(g, ctx) for g in goals])
    console.print("signals :", [readable(s, ctx) for s in sigs])
    console.print(
        "atoms   :",
        [readable(a, ctx) + ("  (goal)" if a in goal_atoms else "  (latent)") for a in atoms],
    )
    stats = {k[len("reducer:") :]: v for k, v in ctx.stats.items() if k.startswith("reducer:")}
    console.print("counts  :", stats)
    shown = set()
    for r in results:
        for rec in r.reducers():
            spec = rec.spec
            if not is_welford(spec) or id(spec) in shown:
                continue
            shown.add(id(spec))
            console.print(f"\n[bold]Welford reducer[/] (monoid laws: {spec.proof})")
            for k in range(spec.arity):
                console.print(f"  slot {k}: I(R)        = {readable(spec.states[k], ctx)}")
                console.print(f"          leaf  I({{j}}) = {readable(spec.leaves[k], ctx)}")
                console.print(f"          identity I(∅)= {readable(spec.identity[k], ctx)}")
                console.print(f"          merge I(AuB) = {readable(spec.merge[k], ctx)}")


def check(result, inputs, rows: int, cols: int, seed: int) -> float:
    """Compile for LLVM and return the max abs error against ``numpy.var``."""
    dtype = str(inputs[0].dtype)
    x = np.random.default_rng(seed).standard_normal((rows, cols)).astype(dtype)
    lib = tvm.compile(tvm.IRModule({"main": result.prim_func()}), target="llvm")
    args = [tvm.runtime.tensor(x), tvm.runtime.tensor(np.zeros(rows, dtype=dtype))]
    lib["main"](*args)
    return float(np.abs(args[1].numpy().astype("float64") - x.astype("float64").var(axis=1)).max())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
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
        help="compile each shown program for LLVM and compare against numpy.var",
    )
    numerics.add_argument("--rows", type=int, default=5, help="check shape (default: %(default)s)")
    numerics.add_argument("--cols", type=int, default=33, help="check shape (default: %(default)s)")
    numerics.add_argument("--seed", type=int, default=0, help="(default: %(default)s)")
    output = parser.add_argument_group("output")
    output.add_argument(
        "--welford-only",
        action="store_true",
        help="show only programs whose reducer is Welford's (count, mean, M2)",
    )
    output.add_argument(
        "--explain", action="store_true", help="print how the Welford reducer was synthesised"
    )
    output.add_argument("--no-ir", dest="show_ir", action="store_false", help="skip the s_tir dump")
    output.add_argument(
        "--te", dest="show_te", action="store_true", help="print each program as TE Python source"
    )
    output.add_argument("--max-results", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    inputs, out = variance_te(args.dtype)
    bounds = Bounds(
        max_tensor_ops=args.budget,
        max_states=args.states,
        min_states=args.min_states,
        max_leaf_nodes=args.leaf_nodes,
    )
    t0 = time.time()
    results = superoptimize(
        out, inputs, bounds, max_results=args.max_results, accuracy=args.accuracy
    )
    elapsed = time.time() - t0
    welford = [r for r in results if result_is_welford(r)]
    console.print(
        f"[bold]{len(results)} equivalent program(s)[/] in {elapsed:.1f}s, "
        f"{len(welford)} of them with Welford's (count, mean, M2) reducer"
    )
    if results and args.explain:
        ctx = results[0].ctx
        explain(console, results, ctx, ctx.dims.key_of_name("cols"))
    shown = welford if args.welford_only else results
    for n, r in enumerate(shown):
        tag = "  [green]Welford[/]" if result_is_welford(r) else ""
        console.rule(f"program {n}: {r.n_ops} ops, verified by {r.verdict.method}{tag}")
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
        if args.check:
            err = check(r, inputs, args.rows, args.cols, args.seed)
            console.print(f"LLVM vs numpy.var ({args.rows}x{args.cols}): max abs error {err:.3g}")
    return 0 if welford else 1


if __name__ == "__main__":
    sys.exit(main())
