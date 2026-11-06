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
"""Controlled state-search, fixed-I emission, and execution-retry experiments.

All runs use the production implementation with explicit internal controls.
The frozen tuples are actual IR objects from the baseline selection. Numeric
rejections are retained, including every warmup and timed retry repetition.
"""

import csv
import dataclasses
import json
import shutil
import statistics
import time

from chain_ops import OPERATORS, showcase

from numerical_audit.prior import PRIORS
from numerical_audit.programs import TARGET, Program, compile_strict
from numerical_audit.records import Ledger, dump, json_safe
from tvm.te.superopt import AccuracyConfig
from tvm.te.superopt.diagnostics import Experiment, solution_record
from tvm.te.superopt.fusion import _fuse_chain, chain_key
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.domain import TeSigns


def workloads():
    for op in [*showcase(), OPERATORS["normalised_projection"], OPERATORS["scaled_norm"]]:
        yield "registry/" + op.name, op.unfused, op.positive, op.max_states
    for op in PRIORS:
        if op.key in ("stable_l2", "performer_g3", "fp8_quant_gemm"):
            yield "prior/" + op.key, op.build, op.positive, None


def append(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(json_safe(data), ensure_ascii=False, allow_nan=False) + "\n")


def experiment(**kwargs):
    return Experiment(compiler=compile_strict, compiler_key=str(TARGET), **kwargs)


def coverage(result, targets):
    accepted = {c.chain_id for c in result.chains if c.fused}
    return {
        "target_chains": len(targets),
        "accepted_chains": len(accepted & set(targets)),
        "complete_fusion": bool(targets) and set(targets) <= accepted,
        "partial_fusion": bool(accepted) and not set(targets) <= accepted,
        "missing_targets": sorted(set(targets) - {c.chain_id for c in result.chains}),
        "skipped": [dataclasses.asdict(s) for s in result.skipped],
    }


def record_run(ledger, workload, mode, repetition, result, phases, exp, targets):
    directory = ledger.directory
    common = {
        "workload": workload,
        "mode": mode,
        "repetition": repetition,
        "warmup": repetition == 0,
    }
    case_links = {}
    for gi, gate in enumerate(exp.gates):
        for family in gate.cases:
            case = f"gate/{workload}/{mode}/{repetition}/{gi}/{family[0]}"
            hashes = ledger.store_inputs(case, family[1])
            case_links[family[0]] = {
                "case_id": case,
                "input_hashes": hashes,
                "extents": gate.var_values,
                "precision": family[7],
                "sensitivity_samples": family[8],
            }
    ledger.reports[f"{workload}/{mode}/{repetition}"] = {
        "config": [dataclasses.asdict(g.config) for g in exp.gates],
        "outputs": [
            {"name": name, "dtype": dtype}
            for g in exp.gates
            for name, dtype in zip(g.output_names, g.dtypes)
        ],
        "checks": [dataclasses.asdict(c) for c in exp.checks],
        "actual_extents": [g.var_values for g in exp.gates],
        "cases": case_links,
    }
    summary = {
        **common,
        **coverage(result, targets),
        "timings": phases,
        "proposals": sum(
            s.candidates
            for c in result.chains
            if c.selection
            for s in c.selection.search
            if s.stage == "sygus"
        ),
        "checker_calls": sum(len(c.selection.checks) for c in result.chains if c.selection),
        "unique_programs": len(
            {a["fingerprint"] for c in result.chains for a in c.attempts if "fingerprint" in a}
        ),
        "numerical_checks": sum(
            a.get("numerical_executed", False) and a["status"] != "duplicate"
            for c in result.chains
            for a in c.attempts
        ),
        "first_rejections": [
            c.attempts[0]["status"] if c.attempts else "not_attempted" for c in result.chains
        ],
        "accepted_ordinals": [
            next((a["attempt_id"] for a in c.attempts if a["status"] == "accepted"), None)
            for c in result.chains
        ],
    }
    for report in result.chains:
        key = report.chain_id
        sel = report.selection
        row = {
            **common,
            "chain_id": key,
            "members": report.members,
            "original": report.original(),
            "required": report.required,
            "theory": report.theory.summary(),
            "fused": report.fused,
            "reason": report.reason,
            "accuracy": dataclasses.asdict(report.accuracy) if report.accuracy else None,
            "selection": None
            if sel is None
            else {
                "route": sel.route,
                "steps": [s.summary() for s in sel.steps],
                "search": [dataclasses.asdict(s) for s in sel.search],
                "checks": sel.checks,
                "diagnostics": sel.diagnostics,
                "no_auxiliary_search_needed": bool(
                    sel.checks and sel.checks[0].get("algebra") == "passed"
                ),
            },
            "solution": solution_record(report.solution) if report.solution else None,
            "cases": case_links,
        }
        append(directory / "chains.jsonl", row)
        for attempt in report.attempts:
            saved = dict(attempt)
            code = saved.pop("program", None)
            if code is not None:
                dest = directory / "expressions" / workload / mode / str(repetition) / key
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"attempt{attempt['attempt_id']}.tir"
                path.write_text(code)
                saved["program_path"] = str(path.relative_to(directory))
            append(
                directory / "attempts.jsonl",
                {**common, "chain_id": key, **saved, "cases": case_links},
            )
    for event in exp.events:
        append(directory / "search.jsonl", {**common, **event})
    append(directory / "runs.jsonl", summary)
    return summary


def fixed_join(ledger, workload, inputs, outputs, config, baseline, targets):
    """Both forms start from the same certified selection and committed prefix."""
    from tvm.te.superopt.fusion import Fused

    source = Program(inputs, outputs).compile(ledger.directory / "programs" / workload / "source")
    rewritten_outputs = baseline.graph[0]
    for key, sol in baseline.solutions.items():
        chain = next(c for c in baseline.graph[1] if chain_key(c) == key)
        programs = {"source": source}
        unavailable = {}
        raw = dataclasses.replace(sol, spec=dataclasses.replace(sol.spec, merge_print=None))
        identical = raw.spec.merge_code == sol.spec.merge_code
        for label, fixed in (("printed", sol), ("no_rebase_printing", raw)):
            exp = experiment(frozen=baseline.frozen, fixed={key: fixed}, prefixes=baseline.prefixes)
            prefix = baseline.prefixes[key].fork()
            start = time.perf_counter()
            report, rewritten = _fuse_chain(
                chain,
                rewritten_outputs,
                prefix,
                baseline.gates[0],
                baseline.agreements[0],
                None,
                "filter",
                TeSigns(outputs, config.positive),
                False,
                exp,
            )
            exp.gates = baseline.gates
            result = Fused([rewritten.tensor(t) for t in rewritten_outputs], inputs, [report], [])
            mode = "fixed_join/" + label
            record_run(
                ledger,
                workload,
                mode,
                1,
                result,
                {**exp.timings, "total": time.perf_counter() - start},
                exp,
                [key],
            )
            append(
                ledger.directory / "fixed_join.jsonl",
                {
                    "workload": workload,
                    "chain_id": key,
                    "variant": label,
                    "identical_emission": identical,
                    "accepted": report.fused,
                    "solution": solution_record(fixed),
                    "reason": report.reason,
                },
            )
            if report.fused:
                programs[label] = Program(inputs, result.outputs).compile(
                    ledger.directory / "programs" / workload / key / label
                )
            else:
                programs[label] = None
                unavailable[label] = report.reason
        # Same quantized gate cases, single source MPFR evaluation per measurement.
        gate = baseline.gates[0]
        for family in gate.cases:
            ledger.measure(
                f"fixed_join/{workload}/{key}/{family[0]}",
                source,
                programs,
                family[1],
                gate.var_values,
                {
                    "section": "fixed_join",
                    "workload": workload,
                    "chain_id": key,
                    "identical_emission": identical,
                },
                unavailable,
            )


def collect(args):
    from audit_numerical_accuracy import manifest, timed_fusion

    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("use a fresh state experiment directory")
    selected = [
        item
        for item in workloads()
        if not args.operators or item[0].split("/")[-1] in args.operators
    ]
    if not selected:
        raise ValueError("operator filter selected no workloads")
    info = manifest(argparse_manifest(args))
    info.update(
        modes=args.modes,
        scope="17 showcase + 5 predeclared rejection workloads",
        synthesis_limits={"seconds": 20, "proposals": 256, "terms": 3},
        controlled_repair="frozen baseline IR",
        retry_default=False,
    )
    dump(directory / "manifest.json", info)
    ledger, rows = Ledger(directory), []
    if args.pytest_log:
        shutil.copyfile(args.pytest_log, directory / "pytest.log")
    try:
        for workload, build, positive, budget in selected:
            inputs, outputs = build()
            config = AccuracyConfig(positive=positive)
            chains, skipped = discover_chains(outputs)
            if any("two different indices" in s.reason for s in skipped):
                from tvm.te.superopt.reducer.views import specialize

                # Enumerate targets after the same authorized view specialization
                # used by production. Keep any remaining skipped targets.
                chains, skipped = discover_chains(specialize(outputs))
            targets = [chain_key(c) for c in chains] + [
                f"skipped:{i}:{s.members}" for i, s in enumerate(skipped)
            ]
            dump(directory / (workload.replace("/", "_") + "_targets.json"), targets)
            baseline = None
            modes = list(dict.fromkeys(["production", *args.modes]))
            for mode in modes:
                if mode == "fixed_join":
                    fixed_join(ledger, workload, inputs, outputs, config, baseline, targets)
                    ledger.checkpoint()
                    continue
                # Retry on/off each get a warmup and five full calls. Other
                # controlled modes are also repeated; no successful-run filtering.
                for rep in range(6):
                    exp = experiment(mode="production" if mode == "retry" else mode)
                    if mode not in ("production", "retry"):
                        exp.frozen, exp.graph = baseline.frozen, baseline.graph
                    start = time.perf_counter()
                    try:
                        result, phases, _ = timed_fusion(
                            inputs, outputs, config, budget, retry=mode == "retry", experiment=exp
                        )
                        row = record_run(ledger, workload, mode, rep, result, phases, exp, targets)
                        if mode == "production":
                            baseline = exp
                    except Exception as err:
                        row = {
                            "workload": workload,
                            "mode": mode,
                            "repetition": rep,
                            "warmup": rep == 0,
                            "complete_fusion": False,
                            "failure": f"{type(err).__name__}: {err}",
                            "target_chains": len(targets),
                            "accepted_chains": 0,
                            "timings": {"total": time.perf_counter() - start},
                        }
                        append(directory / "runs.jsonl", row)
                        # Keep diagnostics even if an orchestration failure occurred.
                        append(directory / "search.jsonl", {**row, "events": exp.events})
                        if mode == "production" and baseline is None:
                            raise
                    rows.append(row)
                    print(
                        workload,
                        mode,
                        rep,
                        row.get("complete_fusion"),
                        f"{row['timings']['total']:.3f}s",
                        flush=True,
                    )
                ledger.checkpoint()
        info["completed"] = True
    except BaseException as err:
        info.update(completed=False, failure=f"{type(err).__name__}: {err}")
        raise
    finally:
        ledger.checkpoint()
        info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        info["run_count"] = len(rows)
        dump(directory / "manifest.json", info)
        fields = [
            "workload",
            "mode",
            "repetition",
            "warmup",
            "target_chains",
            "accepted_chains",
            "complete_fusion",
            "partial_fusion",
            "proposals",
            "checker_calls",
            "unique_programs",
            "numerical_checks",
            "seconds",
            "failure",
        ]
        with (directory / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {k: r["timings"]["total"] if k == "seconds" else r.get(k) for k in fields}
                )
        medians = []
        for workload, mode in dict.fromkeys((r["workload"], r["mode"]) for r in rows):
            group = [
                r
                for r in rows
                if r["workload"] == workload and r["mode"] == mode and not r["warmup"]
            ]
            medians.append(
                {
                    "workload": workload,
                    "mode": mode,
                    "median_seconds": statistics.median(r["timings"]["total"] for r in group),
                    "complete_runs": sum(r["complete_fusion"] for r in group),
                    "runs": len(group),
                }
            )
        dump(directory / "medians.json", medians)
        (directory / "README.md").write_text("""# State selection experiments

Reproduce using the manifest command in the activated development checkout.
The default scope is 17 showcase operators plus normalised_projection, scaled_norm,
stable_l2, performer_g3 and fp8_quant_gemm. Every mode retains complete recovery.
Production and retry each execute one warmup and five measured full fusion calls.
Controlled modes reuse the baseline's repaired IR and reference tuple and skip the
whole-closure shortcut; native semantic unknown does not switch to enumeration.
The reference-only mode still fixes source states. These are bounded comparisons,
not completeness, minimality, or direct-join-synthesis comparisons.

chains.jsonl preserves source/repair/reference/candidates/checks and final expressions.
search.jsonl contains grammar, query limits, proposals, recipes and wall times.
attempts.jsonl preserves every execution verdict, full numerical reports and input
hashes. Missing stages are not_attempted. failed means an observed rejection;
unknown includes a missing constructive inverse, not an impossibility proof.
Checker counts are actual recover calls; proposals include duplicate proposals.
Timings are exclusive of nested instrumented phases; checker seconds separately
include callback time and record its execution/proof split. The 20-second search
clock includes execution callbacks, but excludes the initial/whole-closure attempts.
Running attempts may overrun; this is not an end-to-end hard deadline.

Fixed-join comparisons retain the exact selected tuple, leaves, Printer.kept effects
and epilogue printing. Only merge_print is cleared. Both use the same committed
prefix and numerical gate; rejection leaves later stages unexecuted. They never
retry or select another representation. Identical emissions are explicitly marked.
Measurements and input_cases link to inputs.npz; numerical nonfinite values are
encoded as strings. The fixed-join numeric long table is measurements.jsonl;
summary.csv instead reports state experiment coverage and timings.

Retry changes only execution rejection feedback. Numerical failures are replayed on
the same gate cases, not encoded into the real SyGuS recovery constraint. This is
not a new CEGIS implementation or a formal floating-point correctness proof.
""")
    return 0


def argparse_manifest(args):
    # Share the numerical experiment's dependency/compiler/source manifest.
    import argparse

    return argparse.Namespace(sections=["state_selection"], operators=args.operators)
