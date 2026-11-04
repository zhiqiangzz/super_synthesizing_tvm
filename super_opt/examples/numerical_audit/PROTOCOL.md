<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# MPFR numerical evaluation protocol

This implements the numerical-evaluation-handoff.md protocol from the Hapax
paper worktree, received 2026-11-04, against compiler baseline d2b0653238.
The experiment does not change selection, proofs, accuracy thresholds or scans.
Results may favor any implementation. There is no post-hoc sample exclusion.

Run after activating this worktree (MPFR, Hypothesis and solver extras installed):

```bash
export PYTHONPATH="$PWD/python:$PWD/3rdparty/tvm-tirx/python:$PWD/super_opt/examples"
python -m pytest tests/python/te/superopt -q > /tmp/numerical-pytest.log 2>&1
python super_opt/examples/audit_numerical_accuracy.py --output-dir reports/numerical --pytest-log /tmp/numerical-pytest.log
```

Use an unused output directory. `--sections representative registry hazards prior`
is the default. `--operators` restricts registry/prior only, for development;
the manifest records such restrictions. No code imports or executes paper scripts.

## Actual reference states

For variance and softmax, discover the unique chain and run `_Select` once.
Collect `atoms_in(state, R)` from its repaired base and required output definitions.
Derive these atoms with `preserve=True`, independently build B and the selected
maintained representation with `build_chain` and separate `Rewriter` instances.
Save the complete tuples, merge proofs and independently checked reconstruction
equalities. These proofs establish real semantics, not floating-point equivalence.

- Variance B: count, sum(x), sum(x*x); reconstruct ss = sum(x*x) - sum(x)^2/m,
  then the original var = ss/m. The count remains in the tuple even though final
  reconstruction reads m. Backend dead-code elimination may remove an unused
  count store; both pre-lowering TIR and final LLVM are available for inspection.
- Softmax B: max(x), sum(exp(x)); reconstruct den = sum(exp(x))*exp(-max(x)),
  then keep the original exp(x-max(x))/den. Do not cancel these factors.

B bypasses the gate; its static issues are recorded. Compilation uses the same
explicit LLVM target for every implementation, all fast-math flags false,
including contraction/reassociation. There are no schedule transformations.
The tuple reducers initialize from element zero and merge increasing indices.
`programs/` retains TIR and compiled LLVM IR. The registry hand-written programs
keep their existing identities, distinct from the generated nonempty reducers.

## Fixed representative inputs

For each m in 16, 256, 4096 and seed in 0, 1, 2, create a (3,m) integer array
uniformly in [-16,16], divide by eight, replace each row's first three entries
by -2,0,2, then shuffle each row with the same RNG. Reuse this base across scans.
Transform in float64 and quantize once to float32 before any program runs.

| Operator | Offset scan, scale 1 | Scale scan, offset 0 |
| --- | --- | --- |
| Variance | 0,1,16,256,4096,10000,65536,1048576 | 2^(-80,-70,-60,-30,0,30,60,65,70) |
| Softmax | -128,-112,-104,-96,-88,-64,0,64,80,88,96,128,10000 | 1/16,1,4,16,32,64,128 |

Additionally, for each length, use all-zero and all-3 arrays; for singleton
inputs use three rows [-2], [0], [2]. There are 347 representative cases and
1041 implementation/output measurements. Repeated parameter combinations across
offset/scale scans are intentionally retained as predeclared cases.

The shared reference is the original TE evaluated on the actual float32 inputs
by high_precision_references(), doubling from 128 up to 2048 bits under the
existing agreement criterion. Never convert it to float64 before numeric.errors().

## Registry, hazards, prior workloads and timing

All 51 registry entries are included, including negatives. For centred/offset
data use seed 0, shifts 0 and 10000, with extents from the default AccuracyConfig:
reduction extent 256, other extent 3, per-operation iteration cap 16384.
Generate inputs once per case and evaluate one common MPFR reference for source,
hand-written and accepted synthesized programs. An unaccepted implementation has
explicit not_executed rows, not a source copy labelled synthesized.

Nine hazard cases reproduce the prior paper collector's definitions: zero leading
weights, zero q/x columns, nearly parallel projection, three 1e25 scaling cases,
constant variance and constant zscore_max. These use extent 64, other extent 3,
seed 1. Source-domain failures remain in the ledger.

The 25 prior-program definitions are copied from Hapax/scripts/programs/prior.py,
including its stated modelling limitations. They are transcriptions, not original
prior-system binaries. Use default capped extents and seed 0. The causal-data
mask is 0/-inf according to j<=i; gqa has four query heads to keep N//4 nonzero.
A.1 uses u=[1,-1,3,0], v=[1,1,1,0]. Unsupported evaluations and rejected fusion
remain distinct from numeric pass/fail. No old float64 collectors are run.

One warmup plus five timed full fuse() calls is used for each registry/prior
program and each distinct hazard operator. Instrument existing functions to
measure disjoint derivation/proof, accuracy setup, accuracy check, small-shape
agreement and residual orchestration time in each call. Save all timings and
verdicts; do not report sums of phase medians as the median total. Source and
candidate compile paths inside the gate use the identical strict LLVM target.

AccuracyReport is serialized using dataclasses.asdict, with complete configuration,
output names/dtypes, extents, sampled sensitivity, reference precision and valid
perturbation counts. Actual `ok` values are retained. No threshold-use ratio is
derived from scalar normwise errors. Default empirical elementwise allowance is
100*max(source absolute error, sampled sensitivity) + 64*eps(dtype)*abs(reference).
Absolute tolerance is zero. Sensitivity is not a condition-number bound.

## Artifact semantics

`manifest.json` records revision, dirty status, source fingerprints, versions,
target, commands and settings. `measurements.jsonl` and `summary.csv` retain all
case/implementation/output rows, including nonfinite references/results and
execution failures. `inputs.npz` stores dtype-preserving, content-addressed arrays;
case mappings are in `input_cases.json`. Actual gate input arrays are included.
`accuracy_reports.json` stores full reports and timings. `programs/` stores source,
B and maintained TIR/LLVM plus state definitions, selection and proof results.
`pytest.log` accompanies the artifact; `README.md` describes columns and reproduction.

JSON Infinity/-Infinity/NaN are strings, not nonstandard JSON numbers; null is
missing. Finite-pair errors have their own counts and never replace aggregate
failure metrics. Zero reference uses absolute error with exact-zero indicators.
Source error zero preserves both-zero versus candidate-nonzero comparisons.
`absolute_error_mpfr` retains the MPFR magnitude as a decimal string;
`absolute_metric_underflow` marks a nonzero error rounded to zero in the standard
float diagnostics. Exact-zero and pairwise error comparisons use MPFR, not that
rounded float. This corrects a collector bug found in the first complete run:
extremely small errors could otherwise be labelled both-zero. No rows in that
run's measurements.jsonl actually had a nonzero error rounded to zero, but the
gate's Performer g3 offset family exposed this risk. The unmodified AccuracyReport
float fields are supplemented by `gate_source_diagnostics`, including MPFR strings.
ULPs are diagnostics, not universal acceptance limits. Unknown or unsupported
references are reported as reference_failed, with no invented numerical result.
