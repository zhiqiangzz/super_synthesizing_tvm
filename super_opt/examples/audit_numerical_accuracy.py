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
"""Collect the fixed MPFR experiment protocol, independent of the paper checkout.

python super_opt/examples/audit_numerical_accuracy.py --output-dir reports/numerical
"""

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.metadata
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import gmpy2
import numpy as np
from chain_ops import OPERATORS
from chain_ops.run import extents, sample
from numerical_audit.cases import HAZARDS, LENGTHS, SCANS, SEEDS, representative_cases
from numerical_audit.prior import PRIORS
from numerical_audit.programs import TARGET, Program, compile_strict, reference_programs
from numerical_audit.records import Ledger, dump, output_record

import tvm
import tvm_tirx
from tvm import support
from tvm.te.superopt import AccuracyConfig, fuse
from tvm.te.superopt import accuracy as acc
from tvm_tirx.codegen import llvm

ROOT = Path(__file__).resolve().parents[2]
FUSION = importlib.import_module("tvm.te.superopt.fusion")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(args):
    paths = [Path(__file__), Path(__file__).with_name("numerical_audit") / "PROTOCOL.md"]
    for folder in (
        "python/tvm/te/superopt",
        "super_opt/examples/numerical_audit",
        "super_opt/examples/chain_ops",
        "tests/python/te/superopt",
    ):
        paths.extend(sorted((ROOT / folder).rglob("*.py")))
    versions = {}
    for name in ("gmpy2", "hypothesis", "numpy", "sympy", "z3-solver", "cvc5", "apache-tvm-ffi"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed under this distribution name"
    versions.update(
        mpfr=gmpy2.mpfr_version(),
        llvm_major=llvm.version_major(),
        python=sys.version,
        tvm=tvm.__version__,
        tvm_tirx=tvm_tirx.__version__,
    )
    libraries = {
        Path(line.split()[-1])
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and "libtvm" in line and ".so" in line
    }
    return {
        "schema_version": 1,
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "worktree_status": git("status", "--porcelain=v1"),
        "submodules": git("submodule", "status"),
        "fingerprints": {str(p.relative_to(ROOT)): fingerprint(p) for p in paths},
        "versions": versions,
        "compiler_build": support.libinfo(),
        "loaded_compiler_libraries": {str(p): fingerprint(p) for p in sorted(libraries)},
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "target": TARGET,
        "schedule": "default TE lowering; serial increasing reduction indices; no transforms",
        "pass_context": "default PassContext; strict LLVM target for all compiles including gate",
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": {
            k: os.environ.get(k)
            for k in ("PYTHONPATH", "TVM_LIBRARY_PATH", "TVM_TIRX_LIBRARY_PATH", "OMP_NUM_THREADS")
        },
        "protocol": {
            "lengths": LENGTHS,
            "seeds": SEEDS,
            "scans": SCANS,
            "warmup_runs": 1,
            "timed_runs": 5,
            "mpfr_bits": [128, 2048],
        },
        "accuracy_config_default": dataclasses.asdict(AccuracyConfig()),
        "sections": args.sections,
        "operator_filter": args.operators,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def timed_fusion(inputs, outputs, config, max_states):
    """Observe real fuse() calls; no changed policy and no subtraction of separate runs."""
    phases = defaultdict(float)
    observations = {"gates": [], "checks": []}

    def timed(fn, phase, capture=None):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                phases[phase] += time.perf_counter() - start
            if capture == "gate":
                observations["gates"].append(args[0])
            elif capture == "check":
                observations["checks"].append(result)
            return result

        return wrapped

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(acc, "_compile", compile_strict))
        for owner, method, phase, capture in (
            (FUSION, "_judge", "derivation_proof", None),
            (FUSION, "select", "derivation_proof", None),
            (acc.AccuracyGate, "__init__", "accuracy_setup", "gate"),
            (acc.AccuracyGate, "check", "accuracy_check", "check"),
            (FUSION._Agreement, "__init__", "agreement", None),
            (FUSION._Agreement, "__call__", "agreement", None),
        ):
            stack.enter_context(
                patch.object(owner, method, timed(getattr(owner, method), phase, capture))
            )
        start = time.perf_counter()
        try:
            result = fuse(outputs, inputs, max_states=max_states, accuracy_config=config)
        finally:
            phases["total"] = time.perf_counter() - start
    phases["other"] = phases["total"] - sum(v for k, v in phases.items() if k != "total")
    return result, dict(phases), observations


def fusion_report(ledger, key, inputs, outputs, config, max_states=None):
    timing = []
    result, observed = None, None
    attempts = []
    for repetition in range(6):
        start = time.perf_counter()
        try:
            result, phases, observed = timed_fusion(inputs, outputs, config, max_states)
            attempts.append({"fused": result.fused, "summary": result.summary()})
        except Exception as err:
            phases = {"total": time.perf_counter() - start}
            attempts.append({"failure": f"{type(err).__name__}: {err}"})
            result, observed = None, None
        if repetition:
            timing.append(phases)
        else:
            warmup = phases
    entry = {
        "config": dataclasses.asdict(config),
        "max_states": max_states,
        "outputs": [{"name": t.op.name, "dtype": str(t.dtype)} for t in outputs],
        "warmup": warmup,
        "timings": timing,
        "median_seconds": {
            k: statistics.median(t[k] for t in timing if k in t)
            for k in {k for t in timing for k in t}
        },
        "attempts": attempts,
        "fused": result.fused if result is not None else False,
        "status": "failed" if result is None else ("accepted" if result.fused else "not_fused"),
    }
    if observed is not None:
        entry["checks"] = [dataclasses.asdict(r) for r in observed["checks"]]
        entry["actual_extents"] = [g.var_values for g in observed["gates"]]
        entry["gate_inputs"] = {}
        entry["gate_source_diagnostics"] = {}
        for index, gate in enumerate(observed["gates"]):
            for family in gate.cases:
                case = f"gate/{key}/{index}/{family[0]}"
                entry["gate_inputs"][case] = ledger.store_inputs(case, family[1])
                entry["gate_source_diagnostics"][case] = [
                    {
                        "output_name": t.op.name,
                        "dtype": dtype,
                        **output_record(got, want, dtype, family[7]),
                    }
                    for t, got, want, dtype in zip(outputs, family[3], family[2], gate.dtypes)
                ]
    if result is not None:
        entry["accuracy"] = dataclasses.asdict(result.accuracy) if result.accuracy else None
        entry["chains"] = [
            {
                "members": c.members,
                "fused": c.fused,
                "reason": c.reason,
                "states": c.states,
                "auxiliaries": c.auxiliaries,
                "rejected": [dataclasses.asdict(r) for r in c.rejected],
                "accuracy": dataclasses.asdict(c.accuracy) if c.accuracy else None,
                "route": c.selection.route if c.selection else None,
            }
            for c in result.chains
        ]
        folder = ledger.directory / "programs" / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "selection.txt").write_text(result.summary() + "\n\n" + result.source())
        dump(
            folder / "proofs.json",
            [
                {
                    "states": [str(s) for s in c.solution.spec.states],
                    "merge_proof": c.solution.spec.proof,
                    "reconstruction": [str(e) for e in c.solution.spec.epilogue_code],
                    "selection": [s.summary() for s in c.selection.steps],
                }
                for c in result.chains
                if c.solution is not None
            ],
        )
    ledger.reports[key] = entry
    print(key, entry["status"], f"{entry['median_seconds']['total']:.3f}s", flush=True)
    return result


def compile_programs(ledger, key, programs):
    unavailable = {}
    for label, program in programs.items():
        if program is None:
            unavailable[label] = (
                "no implementation supplied or fusion not accepted (see accuracy_reports)"
            )
            continue
        try:
            program.compile(ledger.directory / "programs" / key / label)
        except Exception as err:
            unavailable[label] = f"{type(err).__name__}: {err}"
            programs[label] = None
    return unavailable


def representatives(ledger):
    for name in SCANS:
        programs, definitions = reference_programs(name)
        source = programs["source"]
        key = f"representative/{name}"
        unavailable = compile_programs(ledger, key, programs)
        dump(ledger.directory / "programs" / key / "states.json", definitions)
        for case, array, metadata in representative_cases(name):
            ledger.measure(
                f"representative/{case}",
                source,
                programs,
                [array],
                {"n": 3, "m": array.shape[1]},
                {"section": "representative", "operator": name, **metadata},
                unavailable,
            )
        ledger.checkpoint()
        print(key, "finished", flush=True)


def registry(ledger, names):
    for name, op in OPERATORS.items():
        if names and name not in names:
            continue
        inputs, outputs = op.unfused()
        source = Program(inputs, outputs)
        config = AccuracyConfig(positive=op.positive)
        key = f"registry/{name}"
        result = fusion_report(ledger, key, inputs, outputs, config, op.max_states)
        programs = {
            "source": source,
            "hand": Program(*op.fused()) if op.fused is not None else None,
            "maintained": Program(inputs, result.outputs) if result and result.fused else None,
        }
        unavailable = compile_programs(ledger, key, programs)
        values = acc._var_values(outputs, inputs, config)
        for family, shift in (("centred", 0), ("offset", 10000)):
            arrays = sample(inputs, outputs, values, 0, shift=shift, positive=op.positive)
            ledger.measure(
                f"{key}/{family}",
                source,
                programs,
                arrays,
                values,
                {
                    "section": "registry",
                    "operator": name,
                    "family": family,
                    "seed": 0,
                    "offset": shift,
                },
                unavailable,
            )
        ledger.checkpoint()


def hazards(ledger):
    cache = {}
    for index, (name, tweak, description) in enumerate(HAZARDS):
        if name not in cache:
            op = OPERATORS[name]
            inputs, outputs = op.unfused()
            source = Program(inputs, outputs)
            key = f"hazards/{name}"
            result = fusion_report(
                ledger, key, inputs, outputs, AccuracyConfig(positive=op.positive), op.max_states
            )
            programs = {
                "source": source,
                "hand": Program(*op.fused()) if op.fused is not None else None,
                "maintained": Program(inputs, result.outputs) if result and result.fused else None,
            }
            unavailable = compile_programs(ledger, key, programs)
            cache[name] = source, programs, unavailable
        source, programs, unavailable = cache[name]
        values = extents(source.inputs, source.outputs, reduce=64)
        arrays = sample(source.inputs, source.outputs, values, 1, positive=OPERATORS[name].positive)
        arrays = [a.astype(np.float32) for a in tweak(arrays)]
        ledger.measure(
            f"hazards/{index}-{name}",
            source,
            programs,
            arrays,
            values,
            {"section": "hazards", "operator": name, "description": description, "seed": 1},
            unavailable,
        )
        ledger.checkpoint()


def prior_work(ledger, names):
    for prior in PRIORS:
        if names and prior.key not in names:
            continue
        inputs, outputs = prior.build()
        source = Program(inputs, outputs)
        key = f"prior/{prior.key}"
        config = AccuracyConfig(positive=prior.positive)
        result = fusion_report(ledger, key, inputs, outputs, config)
        programs = {
            "source": source,
            "maintained": Program(inputs, result.outputs) if result and result.fused else None,
        }
        unavailable = compile_programs(ledger, key, programs)
        values = acc._var_values(outputs, inputs, config)
        # The grouped-query spelling requires a nonzero integer number of KV heads.
        if prior.key == "gqa":
            values["N"] = 4
        arrays = sample(inputs, outputs, values, 0, positive=prior.positive)
        # A causal mask represented as data must actually be a mask, not Gaussian noise.
        if prior.key == "causal_data":
            mask = arrays[-1]
            mask[:] = np.where(
                np.arange(mask.shape[1])[None, :] <= np.arange(mask.shape[0])[:, None], 0, -np.inf
            )
        if prior.key == "a1_program":
            arrays = [
                np.array([[1, -1, 3, 0]], dtype=np.float32),
                np.array([[1, 1, 1, 0]], dtype=np.float32),
            ]
            values = {"n": 1, "m": 4}
        ledger.measure(
            key,
            source,
            programs,
            arrays,
            values,
            {
                "section": "prior",
                "operator": prior.key,
                "paper": prior.paper,
                "where": prior.where,
                "spelling": prior.spelling,
                "seed": 0,
            },
            unavailable,
        )
        ledger.checkpoint()


def readme(ledger):
    statuses = Counter(r["status"] for r in ledger.rows)
    worse = sum(
        r.get("larger_absolute_error_than_source", False)
        for r in ledger.rows
        if r["implementation"] == "maintained"
    )
    text = f"""# Numerical evaluation artifact

This directory was generated by `super_opt/examples/audit_numerical_accuracy.py`.
The exact command, environment, commit, dirty state, code hashes, dependency
versions and LLVM target are in `manifest.json`. From an activated checkout:

```bash
export PYTHONPATH=python:3rdparty/tvm-tirx/python:super_opt/examples
python super_opt/examples/audit_numerical_accuracy.py --output-dir reports/numerical-reproduction
```

The protocol and definitions are in `PROTOCOL.md`. `measurements.jsonl` has one
row per case/implementation/output, including unavailable implementations;
`summary.csv` contains the same rows. Status counts (no failures removed):
`{dict(statuses)}`. Maintained outputs with larger absolute error than the source:
{worse}. This count is descriptive, not the gate's acceptance decision.

Inputs are content-addressed in `inputs.npz` (SHA256 of dtype, shape and bytes).
`input_cases.json` and each measurement map case IDs to those keys. All compared
programs receive the same arrays. Gate cases are also archived. Error fields
absolute/relative/normwise/ulps come from `numeric.errors` on raw MPFR results.
ULPs are diagnostic. Nonfinite numbers are explicit strings `Infinity`,
`-Infinity`, `NaN`; null means unavailable, never numerical zero. `nonfinite`
counts actual output elements; `reference_nonfinite` counts MPFR elements.
`finite_metrics` reports only finite pairs, with its denominator separately;
the main metrics and status retain all failures. `exact_zero` means an entirely
zero reference and exactly matching output. `zero_comparison` preserves both-zero
versus candidate-nonzero when source error is zero. `absolute_error_mpfr` retains
the MPFR error as a decimal string; `absolute_metric_underflow` marks nonzero
errors rounded to zero in float diagnostics. Exact-zero and error-order decisions
use MPFR, including for such tiny errors.
`accuracy_reports.json` also includes `gate_source_diagnostics` for the exact
source errors in gate families; the original AccuracyReport remains unmodified.

`accuracy_reports.json` retains dataclasses.asdict of each actual AccuracyReport,
including failed checks, full config, output names/dtypes and actual dimensions.
Its `ok` fields are the genuine elementwise gate verdicts. No threshold ratio is
inferred from normwise errors. Sensitivity is sampled adjacent-float response,
not a condition-number bound. The fixed scans are independent measurements,
not extra gate families or a change to state selection.

Timings use one warmup then five full calls to fuse(), with raw times and median.
`derivation_proof` measures _judge + select (including recovery/domain/merge
proofs); `accuracy_setup` includes source compilation, MPFR and sensitivity;
`accuracy_check` includes static checks, candidate compilation and execution;
`agreement` covers the existing small-shape agreement checks; `other` includes
discovery, static spec checks, rewriting and orchestration. These are observed
within the same calls; total is not inferred by subtraction of separate runs.
Warm caches and Python overhead are included; these are CPU wall times, not GPU
kernel benchmarks. Every repetition's decision/failure is retained.

`programs/` contains emitted TIR and LLVM IR, selection and state/proof details.
The actual B runs bypass numerical/static acceptance; its diagnostics are saved.
The source, B and maintained tuple use serial increasing reduction indices;
both generated tuples initialize from the first element. All LLVM fast-math
options, contraction and reassociation are disabled. MPFR starts at 128 bits,
doubles up to 2048 and uses the existing agreement test; this is not an interval
certificate. `pytest.log` records validation (or an explicit not-run marker).

The prior-program definitions were copied from the paper's scripts/programs/prior.py
without changing their modelling choices (e.g. FP8 rounding and MoE TopK omitted).
They measure this compiler on transcribed workloads, not prior systems' binaries.
Causal-as-data uses an actual causal mask; gqa uses four query heads to respect
its head-group division. The A.1 case uses the published handoff's concrete u,v.
"""
    (ledger.directory / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["representative", "registry", "hazards", "prior"],
        default=["representative", "registry", "hazards", "prior"],
    )
    parser.add_argument("--operators", nargs="*", help="optional registry/prior smoke-test filter")
    parser.add_argument("--pytest-log", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists():
        parser.error("output directory already contains a run; use a fresh directory")
    info = manifest(args)
    dump(args.output_dir / "manifest.json", info)
    shutil.copyfile(
        Path(__file__).with_name("numerical_audit") / "PROTOCOL.md", args.output_dir / "PROTOCOL.md"
    )
    if args.pytest_log:
        shutil.copyfile(args.pytest_log, args.output_dir / "pytest.log")
    else:
        (args.output_dir / "pytest.log").write_text("Tests not run by this command.\n")
    ledger = Ledger(args.output_dir)
    try:
        for section in args.sections:
            if section == "representative":
                representatives(ledger)
            elif section == "registry":
                registry(ledger, args.operators)
            elif section == "hazards":
                hazards(ledger)
            else:
                prior_work(ledger, args.operators)
        info["completed"] = True
    except BaseException as err:
        info["completed"] = False
        info["failure"] = f"{type(err).__name__}: {err}"
        raise
    finally:
        ledger.checkpoint()
        readme(ledger)
        info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        info["measurement_count"] = len(ledger.rows)
        dump(args.output_dir / "manifest.json", info)


if __name__ == "__main__":
    main()
