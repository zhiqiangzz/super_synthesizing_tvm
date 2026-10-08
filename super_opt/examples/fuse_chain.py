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
"""Fuse reduction chains: unfused TE in, single-pass TE out, checked against a hand-written one.

Every operator of ``chain_ops`` is a program in which reductions over one
axis feed each other -- a maximum and the sums shifted by it, a mean and
the moments about it. ``tvm.te.superopt.fuse`` turns each such chain into
one tuple ``te.comm_reducer`` reduction. For an operator this script shows

* the chain as the program writes it;
* the reducer that was synthesised, as TE source, with the states the
  program never computes and where they came from;
* the unfused program, the hand-written single-pass one and the
  synthesised one side by side: passes over the axis and error against a
  float64 reference, on centred data and on data with a large offset.

    python super_opt/examples/fuse_chain.py                # every operator
    python super_opt/examples/fuse_chain.py attention moment3
    python super_opt/examples/fuse_chain.py --list
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

from chain_ops import OPERATORS, Operator
from chain_ops.run import extents, passes, rel_err, run_llvm, sample
from harness import Column, label, render_source, render_table
from rich.console import Console
from rich.rule import Rule

from tvm.te.superopt import fuse

OFFSET = 1.0e4  # every input shifted by this much: where expanded formulas cancel


@dataclasses.dataclass(frozen=True)
class Form:
    """One way of writing the operator, measured."""

    name: str
    passes: str
    centred: float
    offset: float


@dataclasses.dataclass(frozen=True)
class Outcome:
    op: Operator
    fused: bool
    states: str
    derived: str
    seconds: float
    as_expected: bool


def _measure(name, ins, outs, op: Operator, values, axes, seed: int) -> Form:
    errs = []
    for shift in (0.0, OFFSET):
        arrays = sample(ins, outs, values, seed, shift=shift)
        errs.append(rel_err(run_llvm(ins, outs, arrays, values), op.reference(*arrays)))
    walks = " + ".join(str(passes(outs, extent)) for extent in axes)
    return Form(name, walks, *errs)


def _show(console: Console, op: Operator, args) -> Outcome:
    console.print(Rule(f"[bold]{op.name}[/]  [dim]({op.group})[/]  {op.doc}"))
    ins, outs = op.unfused()
    start = time.perf_counter()
    res = fuse(outs, ins, max_states=args.max_states or op.max_states, accuracy=args.accuracy)
    seconds = time.perf_counter() - start

    written = "\n\n".join("\n".join(c.original()) for c in res.chains)
    if written:
        render_source(console, written, "text", "the chain as the program writes it", "unfused")
    if res.fused:
        render_source(console, res.source(), "python", "synthesised reducer", f"{seconds:.2f} s")
    for c in res.chains:
        if not c.fused:
            console.print(f"[yellow]not fused[/] -- {c.reason}")
        if args.verbose:
            for r in c.rejected:
                console.print(f"  [dim]rejected ({', '.join(r.states)}): {r.why}[/]")
    for s in res.skipped:
        console.print(f"[yellow]not fused[/] ({', '.join(s.members)}) -- {s.reason}")
    if args.tir and res.fused:
        render_source(console, res.prim_func().script(), "python", "fused program", "TIR")

    if res.fused:
        values = extents(ins, outs, reduce=args.extent)
        axes = list({str(c.chain.extent): c.chain.extent for c in res.chains}.values())
        forms = [_measure("unfused", ins, outs, op, values, axes, args.seed)]
        if op.fused is not None:
            forms.append(_measure("fused by hand", *op.fused(), op, values, axes, args.seed))
        forms.append(_measure("synthesised", res.inputs, res.outputs, op, values, axes, args.seed))
        render_table(
            console,
            title="the same function, three ways",
            caption=f"float32 on LLVM against float64 numpy, reduction extent {args.extent}",
            columns=[
                label("form", lambda f: f.name),
                Column("passes over the axis", lambda f: f.passes),
                Column("rel. error", lambda f: f"{f.centred:.1e}"),
                Column(f"rel. error, inputs + {OFFSET:g}", lambda f: f"{f.offset:.1e}"),
            ],
            rows=forms,
        )
    states = [len(c.states) for c in res.chains if c.fused]
    expected = res.fused == op.fusible
    if op.fusible and args.max_states is None:
        expected = expected and all(n == op.states for n in states)
    if not op.fusible:
        expected = expected and op.reason in res.summary()
    derived = sorted({name for c in res.chains for name in c.auxiliaries})
    return Outcome(
        op, res.fused, " + ".join(map(str, states)), ", ".join(derived), seconds, expected
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("operators", nargs="*", help="operators to fuse (default: all)")
    parser.add_argument("--list", action="store_true", help="list the operators and exit")
    parser.add_argument("--max-states", type=int, default=None, help="largest reducer to consider")
    parser.add_argument(
        "--accuracy",
        choices=("filter", "report", "off"),
        default="filter",
        help="only accept a reducer as accurate as the original (default), or do not compare",
    )
    parser.add_argument("--extent", type=int, default=256, help="reduction extent of the runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tir", action="store_true", help="also print the fused program as TIR")
    parser.add_argument("--verbose", action="store_true", help="list the reducers not taken")
    args = parser.parse_args(argv)
    unknown = [n for n in args.operators if n not in OPERATORS]
    if unknown:
        parser.error(f"unknown operator(s) {', '.join(unknown)}; see --list")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    if args.list:
        render_table(
            console,
            title="reduction-chain operators",
            caption="group: how the chain is fused (negative: it is not)",
            columns=[
                label("operator", lambda op: op.name),
                label("group", lambda op: op.group),
                label("", lambda op: op.doc),
            ],
            rows=OPERATORS.values(),
        )
        return 0
    chosen = [OPERATORS[n] for n in (args.operators or OPERATORS)]
    outcomes = [_show(console, op, args) for op in chosen]
    if len(outcomes) > 1:
        console.print(Rule("summary"))
        render_table(
            console,
            title="reduction chains fused",
            caption="states: per chain; derived: states the program does not compute",
            columns=[
                label("operator", lambda o: o.op.name),
                label("group", lambda o: o.op.group),
                label("fused", lambda o: "[green]yes[/]" if o.fused else "[yellow]no[/]"),
                Column("states", lambda o: o.states),
                label("derived", lambda o: o.derived),
                Column("seconds", lambda o: f"{o.seconds:.2f}"),
                label("", lambda o: "" if o.as_expected else "[red]unexpected[/]"),
            ],
            rows=outcomes,
        )
    return 0 if all(o.as_expected for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
