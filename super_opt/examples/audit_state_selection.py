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
"""Audit the showcase selection routes with the default numerical gate.

Run with the worktree's TVM PYTHONPATH:
    python super_opt/examples/audit_state_selection.py [--json]
"""

import argparse
import dataclasses
import json
from pathlib import Path

from chain_ops import showcase

from tvm.te.superopt import AccuracyConfig, fuse


def audit():
    rows = []
    for op in showcase():
        inputs, outputs = op.unfused()
        fused = fuse(
            outputs,
            inputs,
            max_states=op.max_states,
            accuracy_config=AccuracyConfig(positive=op.positive),
        )
        chains = []
        for report in fused.chains:
            selection = report.selection
            chains.append(
                {
                    "route": selection.route if selection else "",
                    "states": len(selection.states) if selection else 0,
                    "repairs": [s.summary() for s in selection.steps if s.action == "hoist"]
                    if selection
                    else [],
                    "search": [dataclasses.asdict(s) for s in selection.search]
                    if selection
                    else [],
                    "failure": report.reason,
                    "supplements": report.derived(),
                }
            )
        rows.append(
            {
                "operator": op.name,
                "fused": fused.fused,
                "accuracy": fused.accuracy.ok if fused.accuracy else None,
                "chains": chains,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "production",
            "common",
            "reference_only",
            "enumeration",
            "single_term",
            "fixed_join",
            "retry",
        ],
        default=[
            "production",
            "common",
            "reference_only",
            "enumeration",
            "single_term",
            "fixed_join",
            "retry",
        ],
    )
    parser.add_argument("--operators", nargs="*")
    parser.add_argument("--pytest-log", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        from numerical_audit.selection import collect

        return collect(args)
    rows = audit()
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print("| Operator | Route(s) | Source repair | Fused / accuracy |")
        print("|---|---|---|---|")
        for r in rows:
            routes = ", ".join(c["route"] for c in r["chains"])
            repair = any(c["repairs"] for c in r["chains"])
            print(f"| {r['operator']} | {routes} | {repair} | {r['fused']} / {r['accuracy']} |")
    return 0 if all(r["fused"] and r["accuracy"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
