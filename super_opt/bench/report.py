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
"""Markdown tables from one or more ``run`` results files.

One row per operator and size: the floor the roofline allows, then every
implementation's Proton time with the fraction of the floor it reaches, the
fastest one in bold. Implementations that did not run say why underneath.

    python -m bench.report results/phase0.json > results/phase0.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import roofline, shapes
from .ops import SPECS


def _cell(row: dict) -> str:
    us = row["us"].get("proton")
    if us is None:
        return "—"
    frac = row["of_floor"]
    return f"{us:,.0f} ({100 * frac:.0f}%)" if frac is not None else f"{us:,.0f}"


def render(results: dict) -> str:
    peaks = results["meta"]["roofline"]
    rows = results["rows"]
    out = [
        f"Device: {results['meta']['device']}, torch {results['meta']['torch']}, "
        f"CUDA {results['meta']['cuda']}, triton {results['meta']['triton']}, "
        f"measured {results['meta']['date']}.",
        "",
        f"Floors: HBM {peaks['hbm_bytes_per_s'] / 1e12:.2f} TB/s and bf16 tensor cores "
        f"{peaks['bf16_flops_per_s'] / 1e12:.0f} TFLOPS (measured); float32 FMA "
        f"{peaks['fp32_flops_per_s'] / 1e12:.0f} TFLOPS and ex2 {peaks['ex2_per_s'] / 1e12:.2f} "
        "T/s (from the clock). A cell is Proton µs per call (L2 flushed) and, in "
        "parentheses, the floor over that time. Repeated measurements of the peaks "
        "differ by up to 2%, so a cell slightly above 100% is at the floor.",
        "",
    ]
    for op in dict.fromkeys(r["op"] for r in rows):
        op_rows = [r for r in rows if r["op"] == op]
        impls = list(dict.fromkeys(r["impl"] for r in op_rows))
        out += [f"### {op}", "", "| size | regime | floor µs (bound) | " + " | ".join(impls) + " |"]
        out.append("|" + "---|" * (3 + len(impls)))
        problems = {}
        for key in dict.fromkeys(json.dumps(r["cfg"]) for r in op_rows):
            cfg = json.loads(key)
            here = {r["impl"]: r for r in op_rows if r["cfg"] == cfg}
            regime = next(iter(here.values()))["regime"]
            bound, seconds = max(
                roofline.floors(peaks, SPECS[op], cfg).items(), key=lambda kv: kv[1]
            )
            timed = {k: r["us"]["proton"] for k, r in here.items() if "proton" in r["us"]}
            best = min(timed, key=timed.get) if timed else None
            cells = []
            for impl in impls:
                row = here.get(impl)
                if row is None:
                    cells.append("")
                    continue
                if row["status"] != "ok":
                    problems.setdefault(impl, row["status"])
                cell = _cell(row)
                cells.append(f"**{cell}**" if impl == best else cell)
            out.append(
                f"| {shapes.label(cfg)} | {regime} | {seconds * 1e6:,.0f} ({bound}) | "
                + " | ".join(cells)
                + " |"
            )
        if problems:
            out.append("")
            out += [f"- {impl}: {status}" for impl, status in problems.items()]
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)
    merged, rows = None, {}
    for path in args.results:  # a later file's row replaces an earlier one's
        data = json.loads(path.read_text())
        merged = merged or data
        for row in data["rows"]:
            rows[(row["op"], json.dumps(row["cfg"]), row["impl"])] = row
    merged["rows"] = list(rows.values())
    print(render(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
