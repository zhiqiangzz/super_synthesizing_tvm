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
"""The precision gate: is a TIRx kernel as accurate as torch.compile?

For every size and random seed, both the TIRx kernel and torch.compile of the
operator's unfused program (``bench.ops.SPECS[op].unfused``) run on the same
inputs, and each output's error is measured against the float64 reference as
``max|out - ref| / max|ref|`` (``harness.compare``). An output passes when

    err_TIRx <= max(2 * err_compile, FLOOR[dtype])

The floor is where both errors are rounding of the output dtype itself (one
bf16 rounding is up to 2^-9 = 2e-3; float32 results agree to ~1e-7), and a
ratio of two such numbers says nothing. A NaN or infinity fails; so does an
output of the wrong shape or dtype (they must be those of ``unfused``).

The sizes are ``small`` -- odd ones, chosen so the tails of every loop and
the unaligned (non-16-byte) paths run -- and ``bench``, every size the
performance runs use (``bench.shapes``). Seeds 0, 1, 2 by default.

    python -m kernels.check --op softmax --sizes all
    python -m kernels.check --all --sizes all            # the gate
    python -m kernels.check --op common.example_logsumexp

Results are written to ``bench/results/check/<op>.json``; the exit status is
non-zero if anything failed or did not run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import traceback

import torch
from bench import RESULTS, shapes
from harness import Column, compare, label, render_table, verdict
from rich.console import Console

from . import OPS, load, spec_of

FLOOR = {torch.bfloat16: 4e-3, torch.float16: 4e-3, torch.float32: 1e-6}
SEEDS = (0, 1, 2)

_ROWS = [{"n": 7, "m": 1000}, {"n": 5, "m": 1001}, {"n": 3, "m": 4099}, {"n": 1, "m": 77777}]
SMALL: dict[str, list[dict]] = {
    "softmax": _ROWS,
    "cross_entropy": [
        {"n": 7, "d": 64, "v": 1001},
        {"n": 33, "d": 128, "v": 32003},
        {"n": 5, "d": 256, "v": 50000},
    ],
    "softmax_entropy": _ROWS,
    "attention": [
        {"b": 1, "h": 2, "s": 77, "d": 64},
        {"b": 2, "h": 3, "s": 130, "d": 128},
        {"b": 1, "h": 1, "s": 1000, "d": 64},
    ],
    "layernorm": _ROWS,
    "layernorm_backward": _ROWS,
    "contrastive": [{"n": 37}, {"n": 1000}, {"n": 4099}],
    "sinkhorn_iteration": [
        {"n": 30, "m": 20, "d": 3},
        {"n": 1000, "m": 777, "d": 3},
        {"n": 513, "m": 1025, "d": 64},
    ],
    "pearson": _ROWS,
    "kurtosis": _ROWS,
    "ess": _ROWS,
    "papr": _ROWS,
    "householder": _ROWS,
    "heat_capacity": _ROWS,
    "ols_hc0": _ROWS,
    "normalized_policy_loss": _ROWS,
    "ledoit_wolf": [
        {"b": 3, "n": 50, "d": 7},
        {"b": 2, "n": 1001, "d": 33},
        {"b": 1, "n": 4099, "d": 64},
    ],
    "common.example_logsumexp": _ROWS,
}


@dataclasses.dataclass
class Row:
    op: str
    sizes: str  # "small" or "bench"
    cfg: dict
    seed: int | None
    output: int | None
    dtype: str = ""
    err_tirx: float | None = None
    err_compile: float | None = None
    bound: float | None = None
    passed: bool = False
    error: str = ""


def sizes(op: str, which: str) -> list[tuple[str, dict]]:
    small = [("small", cfg) for cfg in SMALL[op]]
    if op in shapes.SHAPES:
        bench = [("bench", cfg) for _, cfg in shapes.configs(op)]
    else:
        bench = [("bench", cfg) for cfgs in load(op).SHAPES.values() for cfg in cfgs]
    return {"small": small, "bench": bench, "all": small + bench}[which]


def _first_line(e: BaseException) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0] if text else ''}"[:200]


def _compare(op, which, cfg, seed, i, out, comp, ref) -> Row:
    row = Row(op, which, cfg, seed, i, dtype=str(comp.dtype).replace("torch.", ""))
    if out.shape != comp.shape or out.dtype != comp.dtype:
        got, want = f"{tuple(out.shape)} {out.dtype}", f"{tuple(comp.shape)} {comp.dtype}"
        row.error = f"output {i} is {got}, expected {want}"
        return row
    row.err_tirx = compare(out.double(), ref, rtol=0.0, atol=0.0).rel_err
    row.err_compile = compare(comp.double(), ref, rtol=0.0, atol=0.0).rel_err
    row.bound = max(2.0 * row.err_compile, FLOOR[comp.dtype])
    row.passed = math.isfinite(row.err_tirx) and row.err_tirx <= row.bound
    return row


def _results(op: str, variant: str):
    name = op if variant == "fused" else f"{op}.{variant}"
    return RESULTS / "check" / f"{name}.json"


def check_op(op: str, which: str, seeds, console: Console, variant: str = "fused") -> list[Row]:
    rows: list[Row] = []
    module = load(op)
    if module is None:
        return [Row(op, which, {}, None, None, error=f"no module kernels/{op}.py")]
    spec = spec_of(op)
    for size_set, cfg in sizes(op, which):
        t0 = time.time()
        try:
            run = module.build(cfg, variant=variant)
        except Exception as e:
            console.log(
                f"[red]{op} {shapes.label(cfg)}: build failed[/]\n{traceback.format_exc(limit=4)}"
            )
            rows.append(Row(op, size_set, cfg, None, None, error="build: " + _first_line(e)))
            continue
        torch._dynamo.reset()
        compiled = torch.compile(spec.unfused, dynamic=False)
        for seed in seeds:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            inputs = spec.inputs(cfg, gen)
            refs = spec.reference(*inputs)
            comps = compiled(*inputs)
            try:
                outs = run(*inputs)
                torch.cuda.synchronize()
            except Exception as e:
                console.log(f"[red]{op} {shapes.label(cfg)} seed {seed}: run failed[/]")
                rows.append(Row(op, size_set, cfg, seed, None, error="run: " + _first_line(e)))
                continue
            if len(outs) != len(refs):
                rows.append(
                    Row(
                        op,
                        size_set,
                        cfg,
                        seed,
                        None,
                        error=f"{len(outs)} outputs, expected {len(refs)}",
                    )
                )
                continue
            for i, (out, comp, ref) in enumerate(zip(outs, comps, refs)):
                rows.append(_compare(op, size_set, cfg, seed, i, out, comp, ref))
            del inputs, refs, comps, outs
            torch.cuda.empty_cache()
        console.log(f"{op} {shapes.label(cfg)}: {time.time() - t0:.1f} s")
    return rows


def _num(x) -> str:
    return "" if x is None else f"{x:.2e}"


def print_rows(console: Console, op: str, rows: list[Row]) -> None:
    columns = [
        label("sizes", lambda r: r.sizes),
        label("cfg", lambda r: shapes.label(r.cfg)),
        Column("seed", lambda r: "" if r.seed is None else str(r.seed)),
        Column("out", lambda r: "" if r.output is None else str(r.output)),
        label("dtype", lambda r: r.dtype),
        Column("err TIRx", lambda r: _num(r.err_tirx)),
        Column("err compile", lambda r: _num(r.err_compile)),
        Column("bound", lambda r: _num(r.bound)),
        Column("", lambda r: verdict(r.passed), justify="center"),
        label("error", lambda r: r.error),
    ]
    passed = sum(r.passed for r in rows)
    render_table(
        console, title=op, caption=f"{passed} / {len(rows)} passed", columns=columns, rows=rows
    )


def summarize(console: Console, ops, variant: str = "fused") -> bool:
    """One line per operator from its saved results: the worst errors and how close to the bound."""
    lines, ok = [], True
    for op in ops:
        path = _results(op, variant)
        rows = [Row(**r) for r in json.loads(path.read_text())] if path.exists() else []
        timed = [r for r in rows if r.err_tirx is not None]
        passed = bool(rows) and all(r.passed for r in rows)
        ok &= passed
        worst = max(timed, key=lambda r: r.err_tirx / r.bound, default=None)
        lines.append(
            {
                "op": op,
                "checks": f"{sum(r.passed for r in rows)} / {len(rows)}",
                "sizes": len({json.dumps(r.cfg) for r in rows}),
                "dtype": ", ".join(sorted({r.dtype for r in timed})),
                "tirx": max((r.err_tirx for r in timed), default=None),
                "compile": max((r.err_compile for r in timed), default=None),
                "worst": worst,
                "passed": passed,
            }
        )
    columns = [
        label("operator", lambda s: s["op"]),
        Column("checks", lambda s: s["checks"]),
        Column("sizes", lambda s: str(s["sizes"])),
        label("output dtype", lambda s: s["dtype"]),
        Column("max err TIRx", lambda s: _num(s["tirx"])),
        Column("max err compile", lambda s: _num(s["compile"])),
        Column(
            "worst err / bound",
            lambda s: "" if s["worst"] is None else f"{s['worst'].err_tirx / s['worst'].bound:.2f}",
        ),
        Column("", lambda s: verdict(s["passed"]), justify="center"),
    ]
    render_table(
        console,
        title="precision gate",
        caption="err = max|out - ref| / max|ref| vs float64; bound = max(2 err_compile, floor)",
        columns=columns,
        rows=lines,
    )
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--op", nargs="+")
    group.add_argument("--all", action="store_true", help="every showcase operator")
    parser.add_argument("--sizes", default="small", choices=("small", "bench", "all"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--variant",
        default="fused",
        choices=("fused", "unfused"),
        help="the one-pass kernel, or the k-pass ablation",
    )
    parser.add_argument(
        "--summary", action="store_true", help="summarise the saved results instead of running"
    )
    args = parser.parse_args(argv)

    console = Console(width=160)
    ops = OPS if args.all else args.op
    if args.summary:
        return 0 if summarize(console, ops, args.variant) else 1
    failed = []
    for op in ops:
        rows = check_op(op, args.sizes, args.seeds, console, args.variant)
        print_rows(console, f"{op} ({args.variant})", rows)
        out = _results(op, args.variant)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([dataclasses.asdict(r) for r in rows], indent=1))
        if not rows or not all(r.passed for r in rows):
            failed.append(op)
    console.print(
        f"[bold]{len(ops) - len(failed)} / {len(ops)} operators pass[/]"
        + (f"; failing: {', '.join(failed)}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
