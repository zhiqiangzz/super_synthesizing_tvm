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
"""Check and time every implementation of the showcase operators.

For each operator and size: build the inputs, compute the float64 reference,
then for every implementation (the baselines of ``baselines``, and the TIRx
kernels once there are some) build it, check its outputs against the
reference, and time it. An implementation is timed only once it has passed.

Times come from ``tvm_tirx.testing.bench``, the timer tirx-kernels reports
with: ``proton`` is the GPU time of the kernels a call launches, with L2
flushed before every call (host and launch overhead excluded, which favours
the multi-kernel unfused baselines); ``event`` is the wall time of a call
between two CUDA events, launch gaps included. Both are recorded.

    python -m bench.run --roofline                       # measure the card first
    python -m bench.run --op softmax --regime split
    python -m bench.run --out phase0.json                # every operator, every size
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
import traceback

import chain_ops
import torch
from harness import Column, compare, label, render_table
from rich.console import Console

from . import RESULTS, baselines, roofline, shapes
from .ops import SPECS

TIMERS = ("proton", "event")


@dataclasses.dataclass
class Row:
    op: str
    regime: str
    cfg: dict
    impl: str
    status: str  # "ok", "inaccurate", "OOM", or the error
    rel_err: float | None = None  # max over outputs of max|out - ref| / max|ref|
    us: dict = dataclasses.field(default_factory=dict)  # timer -> microseconds per call
    bound: str = ""
    of_floor: float | None = None  # floor time / proton time
    note: str = ""


def _first_line(e: BaseException) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0] if text else ''}"[:160]


def _check(spec, outs, refs) -> float:
    assert len(outs) == len(refs), (len(outs), len(refs))
    return max(compare(o.double(), r, rtol=0.0, atol=0.0).rel_err for o, r in zip(outs, refs))


def _time(fn, timer: str, rounds: int) -> float:
    from tvm_tirx.testing.bench import bench

    return bench({"impl": fn}, timer=timer, rounds=rounds)["impls"]["impl"]


def run_config(op, regime, cfg, *, peaks, timers, rounds, skip, only, console) -> list[Row]:
    spec = SPECS[op]
    gen = torch.Generator(device="cuda").manual_seed(0)
    inputs = spec.inputs(cfg, gen)
    t0 = time.time()
    refs = spec.reference(*inputs)
    console.log(f"{op} {shapes.label(cfg)}: reference in {time.time() - t0:.1f} s")

    rows = []
    for impl, build in baselines.builders(op, cfg).items():
        if impl in skip or (only and impl not in only):
            continue
        row = Row(op, regime, cfg, impl, "ok", note=baselines.NOTES.get((op, impl), ""))
        rows.append(row)
        try:
            fn = build(*inputs)
            outs = fn()
            torch.cuda.synchronize()
            row.rel_err = _check(spec, outs, refs)
            del outs
            if row.rel_err > spec.tol:
                row.status = "inaccurate"
                continue
            for timer in timers:
                row.us[timer] = _time(fn, timer, rounds)
            if "proton" in row.us:
                row.bound, row.of_floor = roofline.fraction(peaks, spec, cfg, row.us["proton"])
        except torch.OutOfMemoryError:
            row.status = "OOM"
        except ModuleNotFoundError as e:
            row.status = f"not installed ({e.name})"
        except Exception as e:  # report and go on: one baseline must not stop the sweep
            row.status = _first_line(e)
            console.log(f"[red]{op} {impl}[/]: {traceback.format_exc(limit=3)}")
        finally:
            fn = None
            torch.cuda.empty_cache()
    del inputs, refs
    torch.cuda.empty_cache()
    return rows


def _fmt_us(row: Row, timer: str) -> str:
    return f"{row.us[timer]:,.1f}" if timer in row.us else ""


def print_rows(console: Console, op: str, cfg: dict, rows: list[Row]) -> None:
    unfused = next((r.us.get("proton") for r in rows if r.impl == "torch-unfused"), None)
    columns = [
        label("implementation", lambda r: r.impl),
        label("status", lambda r: r.status if r.status == "ok" else f"[yellow]{r.status}[/]"),
        Column("err / ‖ref‖∞", lambda r: "" if r.rel_err is None else f"{r.rel_err:.1e}"),
        Column("proton µs", lambda r: _fmt_us(r, "proton")),
        Column("event µs", lambda r: _fmt_us(r, "event")),
        Column(
            "× unfused",  # noqa: RUF001
            lambda r: f"{unfused / r.us['proton']:.2f}" if unfused and "proton" in r.us else "",
        ),
        Column(
            "of floor",
            lambda r: "" if r.of_floor is None else f"{100 * r.of_floor:.0f}% {r.bound}",
        ),
    ]
    render_table(
        console, title=f"{op}  {shapes.label(cfg)}", caption="", columns=columns, rows=rows
    )


def _meta(peaks: dict) -> dict:
    import triton

    import tvm_tirx

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "tvm_tirx": getattr(tvm_tirx, "__version__", "?"),
        "roofline": peaks,
        "showcase": [op.name for op in chain_ops.showcase()],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--op", nargs="*", default=None, help="operators (default: the showcase)")
    parser.add_argument("--regime", default="all")
    parser.add_argument("--timer", nargs="*", default=list(TIMERS), choices=TIMERS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--skip", nargs="*", default=[], help="implementations to leave out")
    parser.add_argument("--only", nargs="*", default=[], help="time only these implementations")
    parser.add_argument(
        "--out", default=None, help="file name under results/ (default: <date>.json)"
    )
    parser.add_argument("--roofline", action="store_true", help="(re)measure the card's peaks")
    args = parser.parse_args(argv)

    console = Console(width=150)
    peaks = roofline.measure() if args.roofline else roofline.load()
    roofline.warm_up()
    console.log(
        f"{peaks['device']}: HBM {peaks['hbm_bytes_per_s'] / 1e12:.2f} TB/s, "
        f"bf16 {peaks['bf16_flops_per_s'] / 1e12:.0f} TFLOPS (measured), "
        f"fp32 {peaks['fp32_flops_per_s'] / 1e12:.0f} TFLOPS, "
        f"ex2 {peaks['ex2_per_s'] / 1e12:.2f} T/s (computed)"
    )
    showcase = [op.name for op in chain_ops.showcase()]
    assert set(showcase) == set(SPECS), set(showcase) ^ set(SPECS)
    ops = args.op if args.op is not None else showcase
    if not ops:
        return 0

    out = RESULTS / (args.out or f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for op in ops:
        for regime, cfg in shapes.configs(op, args.regime):
            new = run_config(
                op,
                regime,
                cfg,
                peaks=peaks,
                timers=args.timer,
                rounds=args.rounds,
                skip=set(args.skip),
                only=set(args.only),
                console=console,
            )
            print_rows(console, op, cfg, new)
            rows += new
            out.write_text(
                json.dumps(
                    {"meta": _meta(peaks), "rows": [dataclasses.asdict(r) for r in rows]}, indent=1
                )
            )
    console.log(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
