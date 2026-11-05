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
"""The TIRx kernels against torch.compile and the libraries, from recorded ``run`` results.

One row per operator and size: the fused kernel's Proton time and the fraction of
the roofline floor it reaches, then how many times longer torch.compile, the
fastest library and the unfused TIRx ablation take. A library is any baseline
other than the unfused torch program and torch.compile (``torch`` is torch's own
kernel). ``--all`` lists every implementation's time, or why it did not run.

    python -m bench.summary                  # the recorded results
    python -m bench.summary FILE ... [--all]

The recorded results are every ``results/final-<op>.json``, and the
``results/verify-<op>*.json`` of operators without a final file. When files
overlap, the later file (by name) wins.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import RESULTS, shapes

OURS = ("TIRx-fused", "TIRx-unfused")
NOT_LIBRARIES = {*OURS, "torch-unfused", "torch.compile"}


def recorded() -> list:
    finals = sorted(RESULTS.glob("final-*.json"))
    have = {p.stem.removeprefix("final-") for p in finals}
    verify = [
        p for p in sorted(RESULTS.glob("verify-*.json")) if not any(op in have for op in _ops(p))
    ]
    return verify + finals


def _ops(path) -> set:
    return {r["op"] for r in json.loads(path.read_text())["rows"]}


def load(paths) -> dict:
    """``(op, cfg) -> {impl: row}``, by operator name and then in ``shapes`` order."""
    table: dict = {}
    for path in paths:
        for r in json.loads(path.read_text())["rows"]:
            table.setdefault((r["op"], json.dumps(r["cfg"])), {})[r["impl"]] = r

    def order(key):
        op, cfg = key
        sizes = [json.dumps(c) for _, c in shapes.configs(op)] if op in shapes.SHAPES else []
        return op, sizes.index(cfg) if cfg in sizes else len(sizes)

    return {key: table[key] for key in sorted(table, key=order)}


def _size(cfg: str) -> str:
    return shapes.label(json.loads(cfg))


def _us(row) -> float | None:
    return row["us"].get("proton") if row and row["status"] == "ok" else None


def comparison(table: dict) -> str:
    out = [
        "| op | size | TIRx fused µs (of floor) | torch.compile | fastest library | TIRx unfused |",
        "|---|---|---|---|---|---|",
    ]
    for (op, cfg), impls in table.items():
        fused = _us(impls.get("TIRx-fused"))
        if fused is None:
            continue

        def times(us, fused=fused):
            return "—" if us is None else f"{us:,.1f} ({us / fused:.2f}×)"  # noqa: RUF001

        libs = {
            k: _us(r) for k, r in impls.items() if k not in NOT_LIBRARIES and _us(r) is not None
        }
        best = min(libs, key=libs.get) if libs else None
        frac = impls["TIRx-fused"]["of_floor"]
        out.append(
            f"| {op} | {_size(cfg)} | {fused:,.1f} ({100 * frac:.0f}%) "
            f"| {times(_us(impls.get('torch.compile')))} "
            f"| {f'{best} {times(libs[best])}' if best else '—'} "
            f"| {times(_us(impls.get('TIRx-unfused')))} |"
        )
    out.append("")
    out.append(
        "In parentheses after a baseline: its time over the fused kernel's (above 1: slower)."
    )
    return "\n".join(out)


def every_impl(table: dict) -> str:
    out = []
    for (op, cfg), impls in table.items():
        out.append(f"{op} {_size(cfg)}")
        for impl, r in impls.items():
            us = _us(r)
            out.append(f"    {impl:24s} {f'{us:,.1f} µs' if us is not None else r['status']}")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", help="results files (default: the recorded results)")
    parser.add_argument("--all", action="store_true", help="every implementation's time")
    args = parser.parse_args(argv)
    paths = [pathlib.Path(f) if "/" in f else RESULTS / f for f in args.files] or recorded()
    table = load(paths)
    print(f"from: {', '.join(p.name for p in paths)}\n")
    print(every_impl(table) if args.all else comparison(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
