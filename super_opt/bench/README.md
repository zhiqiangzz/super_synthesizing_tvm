<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Showcase operators on H800: TIRx kernels against torch.compile and the libraries

This directory measures the 17 operators of `chain_ops.showcase()` on an NVIDIA H800
(sm_90a, 132 SMs, 1980 MHz). Each operator has a hand-written TIRx kernel,
`../kernels/<op>.py`, in two variants built from the same parts:

- **fused:** the one-pass algorithm;
- **unfused:** an ablation that runs the unfused program's passes.

Both variants are compared with:

- the unfused program written in torch;
- that program through `torch.compile`;
- the libraries that implement the operator, where any exist.

Before any kernel is timed, it must pass the precision gate, `../kernels/check.py`.

| file | contents |
|---|---|
| `ops.py` | per operator: inputs, the unfused torch program, a float64 reference, traffic and work |
| `shapes.py` | the sizes, grouped by regime (short, long and split rows; GEMM; ...) |
| `baselines.py` | every implementation, and adapters to the libraries |
| `roofline.py` | the card's peaks, and the floor each size allows |
| `run.py` | checks each implementation against the reference, then times it |
| `report.py`, `summary.py` | tables built from the results |
| `reproduce.sh` | runs the gate, the timing and the tests on chosen GPUs |
| `build_fa3.sh` | builds FlashAttention-3, the strongest attention baseline |
| `environment.txt` | the software that produced the results |
| `results/` | the recorded results, described below |

## Method

- **Precision gate.** Command: `python -m kernels.check`.
  - Per output, the TIRx error against the float64 reference must satisfy
    `err_TIRx <= max(2 * err_torch.compile, floor)`.
  - The floor is 4e-3 for bf16 outputs and 1e-6 for float32 outputs.
  - Inputs: random data with seeds 0, 1 and 2, at small odd sizes and at every benchmark size.
- **Timing.** Command: `python -m bench.run`.
  - Times come from `tvm_tirx.testing.bench`, which follows Triton's `do_bench`. Each time
    is the mean of 3 rounds.
  - `proton` is the GPU time of the kernels a call launches, with L2 flushed before every call.
  - `event` is the wall time between two CUDA events.
  - Every process first runs 5 s of SM work (`roofline.warm_up`), because the clock takes
    about 6 s to ramp up.
  - An implementation is timed only after its outputs meet the operator's tolerance.
- **Floor.** The floor is the largest of four lower bounds:
  - HBM bytes / 3.04 TB/s;
  - tensor-core FLOPs / 746 TFLOPS;
  - float32 FLOPs / 66.9 TFLOPS;
  - exp2 calls / 4.18 T/s.

  The first two peaks were measured on this card and are stored in `results/roofline.json`.
  The other two follow from the clock. A time's "% of floor" is floor / time. It can exceed
  100%, because the measured peaks sit below the datasheet's 3.35 TB/s and 989 TFLOPS.
- **Cards.** Use idle, cool cards, with one job per card at a time (`reproduce.sh` holds a
  lock on each card for every job).
  - A kernel's time repeats within about 2%.
  - A card that idled at 83 °C and throttled moved times by about 10%.
  - That card later fell off the bus under load. From then on, CUDA failed to initialize
    on every card of the machine.

## Reproducing

**Set up the environment:**

1. Build TVM and tvm_tirx from this commit (`../../AGENTS.md`). The `3rdparty/tvm-tirx`
   submodule pins tvm_tirx.
2. Install the packages in `environment.txt`: CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1,
   and the libraries.
3. Build FlashAttention-3 with `bash super_opt/bench/build_fa3.sh`. It uses flash-attention
   v2.8.3 (`060c9188`), and its header explains the one-line patch it applies.

**Run** these commands from the repository root. GPU indices follow `nvidia-smi`.

```bash
bash super_opt/bench/reproduce.sh gate 0,1,2,3     # every results/check file
bash super_opt/bench/reproduce.sh pytest 0         # tests/python/te/superopt/kernels
bash super_opt/bench/reproduce.sh time 1 final-attention --op attention   # one timing file
PYTHONPATH=super_opt python -m bench.summary       # the summary below, from results/
```

### Recorded timing results

**How each file was produced.** In the table below, `time GPU NAME ARGS` stands for
`reproduce.sh time GPU NAME ARGS`. Dates are in the machine's time zone (PST). The GPU
numbers use the card numbering at the time, when all 8 cards were present.

| file | command | produced |
|---|---|---|
| `final-attention.json`, `final-layernorm.json`, `final-kurtosis.json`, `final-contrastive.json`, `final-sinkhorn_iteration.json` | `time GPU final-<op> --op <op>` (every implementation) | 2026-11-04 04:09–04:22, GPUs 1, 4 and 7, by the coordinator, one operator at a time per card |
| `verify-cross_entropy.json` | `time GPU verify-cross_entropy --op cross_entropy --only TIRx-fused TIRx-unfused torch.compile torch "liger FLCE" CCE "cuBLAS + QuACK CE"` | 2026-11-04 16:06, GPU 1, by an independent verifier |
| `verify-softmax.json` | `time GPU verify-softmax --op softmax --only TIRx-fused TIRx-unfused torch.compile torch quack liger` | 2026-11-04 16:06, GPU 4, verifier. A watchdog stopped it after five sizes: QuACK's compile at 1×67108864 never finishes |
| `verify-softmax-r3-split-a.json` | `time GPU verify-softmax-r3-split-a --op softmax --regime split --only TIRx-fused TIRx-unfused torch.compile torch liger` | 2026-11-04 12:45, GPU 4, verifier |
| `verify-ols_hc0.json` | `time GPU verify-ols_hc0 --op ols_hc0 --only TIRx-fused TIRx-unfused torch.compile` | 2026-11-04 16:19, GPU 3, verifier |
| `verify-normalized_policy_loss.json` | the same `--only` list, for `normalized_policy_loss` | 2026-11-05 00:25, GPU 5, verifier |
| `verify-ledoit_wolf.json` | the same `--only` list, for `ledoit_wolf` | 2026-11-05 07:38, GPU 4, verifier |
| `roofline.json` | `python -m bench.run --roofline --op` (measures the peaks, times no operator) | 2026-10-29 04:54 |

**Why the committed code is the code that ran:**
- No kernel module changed after its results were written.
- For softmax, cross_entropy, ols_hc0 and normalized_policy_loss, the verifiers recorded the
  module's md5 before and after their runs. These md5s equal the committed files'.
- Since then, the shared code has changed only by additions and by formatting:
  - `common.reduce.fmax` / `fmin` are new;
  - `common.build.compile_sm90(ftz=...)` is a new parameter, and its default compiles as before;
  - the formatting was checked to leave every module's AST unchanged, apart from import
    order and one docstring.

The final timing of the five second-round operators by the coordinator
(`final-<op>.json`, every implementation) is still to come. Until then, `summary.py`
reads their `verify-*` files.

### Recorded precision results

**Command.** Each `results/check/<op>.json` (fused) and `<op>.unfused.json` was produced by
`python -m kernels.check --op <op> --sizes all --seeds 0 1 2 [--variant unfused]`.
Every recorded file passes.

**Missing files, to be rerun with `reproduce.sh gate` once the GPUs are back:**
- the fused gate of cross_entropy, kurtosis and layernorm;
- the unfused gate of attention, contrastive, cross_entropy, householder, kurtosis,
  layernorm, layernorm_backward, papr, pearson and softmax_entropy.

**Why they are missing.** On 2026-11-05 a run lost a GPU partway through, and from then on
CUDA failed on every card. That run overwrote eight of these files with CUDA errors. Two
others (pearson and layernorm_backward, unfused) had been written by an earlier version of
the harness. Both kinds are kept, but not recorded, in `results/local/`. The unfused gate of
householder, papr and softmax_entropy has not been run yet.

### Summary

Generated by `python -m bench.summary`. In parentheses after each baseline: its time divided
by the fused kernel's (above 1 means the baseline is slower).

| op | size | TIRx fused µs (of floor) | torch.compile | fastest library | TIRx unfused |
|---|---|---|---|---|---|
| attention | b=8 h=16 s=1024 d=128 | 114.1 (81%) | 617.3 (5.41×) | FlashAttention-3 112.7 (0.99×) | 232.4 (2.04×) |
| attention | b=2 h=16 s=4096 d=128 | 410.5 (90%) | 2,513.2 (6.12×) | FlashAttention-3 387.5 (0.94×) | 798.0 (1.94×) |
| attention | b=1 h=16 s=8192 d=128 | 817.9 (90%) | 4,665.9 (5.70×) | FlashAttention-3 759.6 (0.93×) | 1,565.9 (1.91×) |
| attention | b=2 h=16 s=4096 d=64 | 287.3 (64%) | 2,452.8 (8.54×) | FlashAttention-3 287.4 (1.00×) | 631.7 (2.20×) |
| contrastive | n=8192 | 59.9 (74%) | 172.6 (2.88×) | open_clip ClipLoss 986.7 (16.46×) | 190.9 (3.19×) |
| contrastive | n=16384 | 196.1 (90%) | 726.7 (3.71×) | open_clip ClipLoss 2,861.0 (14.59×) | 704.9 (3.59×) |
| contrastive | n=32768 | 744.5 (95%) | 2,919.7 (3.92×) | open_clip ClipLoss 11,493.1 (15.44×) | 2,753.9 (3.70×) |
| cross_entropy | n=8192 d=4096 v=32000 | 2,844.2 (101%) | 3,188.4 (1.12×) | cuBLAS + QuACK CE 3,054.1 (1.07×) | 5,789.4 (2.04×) |
| cross_entropy | n=8192 d=4096 v=128256 | 11,381.8 (101%) | 12,541.1 (1.10×) | cuBLAS + QuACK CE 12,025.1 (1.06×) | 22,453.7 (1.97×) |
| cross_entropy | n=8192 d=4096 v=256000 | 22,355.7 (103%) | 25,164.4 (1.13×) | cuBLAS + QuACK CE 24,264.5 (1.09×) | 45,305.0 (2.03×) |
| kurtosis | n=16384 m=4096 | 96.4 (92%) | 97.1 (1.01×) | — | 268.8 (2.79×) |
| kurtosis | n=4096 m=65536 | 346.9 (102%) | 939.0 (2.71×) | — | 1,018.5 (2.94×) |
| kurtosis | n=16 m=4194304 | 98.7 (89%) | 279.9 (2.83×) | — | 271.7 (2.75×) |
| kurtosis | n=1 m=67108864 | 98.8 (89%) | 280.5 (2.84×) | — | 272.0 (2.75×) |
| layernorm | n=32768 m=4096 | 178.3 (99%) | 190.9 (1.07×) | quack 178.4 (1.00×) | 352.1 (1.97×) |
| layernorm | n=16384 m=8192 | 178.4 (99%) | 176.8 (0.99×) | quack 178.4 (1.00×) | 352.0 (1.97×) |
| layernorm | n=4096 m=65536 | 357.8 (99%) | 588.5 (1.64×) | quack 364.3 (1.02×) | 702.1 (1.96×) |
| layernorm | n=4 m=16777216 | 111.4 (79%) | 187.5 (1.68×) | torch 31,269.7 (280.63×) | 188.2 (1.69×) |
| ledoit_wolf | b=256 n=1024 d=32 | 39.1 (41%) | 143.0 (3.65×) | — | 101.4 (2.59×) |
| ledoit_wolf | b=64 n=4096 d=64 | 114.6 (56%) | 386.7 (3.38×) | — | 149.0 (1.30×) |
| ledoit_wolf | b=8 n=65536 d=128 | 692.6 (74%) | 4,410.9 (6.37×) | — | 670.2 (0.97×) |
| normalized_policy_loss | n=65536 m=256 | 48.3 (92%) | 49.5 (1.03×) | — | 97.8 (2.03×) |
| normalized_policy_loss | n=256 m=262144 | 181.9 (97%) | 426.7 (2.35×) | — | 351.8 (1.93×) |
| normalized_policy_loss | n=1 m=33554432 | 100.0 (88%) | 194.8 (1.95×) | — | 187.9 (1.88×) |
| ols_hc0 | n=16384 m=4096 | 180.5 (98%) | 178.1 (0.99×) | — | 349.2 (1.93×) |
| ols_hc0 | n=4096 m=65536 | 680.0 (104%) | 1,210.5 (1.78×) | — | 1,344.2 (1.98×) |
| ols_hc0 | n=16 m=4194304 | 186.8 (95%) | 368.4 (1.97×) | — | 353.5 (1.89×) |
| ols_hc0 | n=1 m=67108864 | 185.9 (95%) | 369.2 (1.99×) | — | 354.1 (1.90×) |
| sinkhorn_iteration | n=8192 m=8192 d=3 | 58.9 (54%) | 590.3 (10.02×) | FlashSinkhorn 408.9 (6.94×) | 86.2 (1.46×) |
| sinkhorn_iteration | n=32768 m=32768 d=3 | 706.5 (73%) | 9,368.3 (13.26×) | KeOps 3,408.0 (4.82×) | 1,069.0 (1.51×) |
| sinkhorn_iteration | n=8192 m=8192 d=64 | 528.1 (49%) | 778.3 (1.47×) | FlashSinkhorn 1,169.0 (2.21×) | 974.0 (1.84×) |
| sinkhorn_iteration | n=32768 m=32768 d=64 | 7,707.9 (53%) | 14,026.2 (1.82×) | FlashSinkhorn 13,574.3 (1.76×) | 14,744.5 (1.91×) |
| softmax | n=32768 m=4096 | 178.3 (99%) | 315.7 (1.77×) | liger 178.2 (1.00×) | 354.6 (1.99×) |
| softmax | n=16384 m=8192 | 177.9 (99%) | 309.6 (1.74×) | quack 178.4 (1.00×) | 354.3 (1.99×) |
| softmax | n=4096 m=32768 | 179.9 (98%) | 277.6 (1.54×) | quack 180.2 (1.00×) | 357.0 (1.98×) |
| softmax | n=2048 m=131072 | 358.9 (98%) | 595.4 (1.66×) | quack 356.6 (0.99×) | 712.6 (1.99×) |
| softmax | n=8 m=8388608 | 124.6 (71%) | 185.7 (1.49×) | torch 2,145.1 (17.21×) | 187.7 (1.51×) |
| softmax | n=1 m=67108864 | 138.0 (64%) | 185.7 (1.35×) | torch 16,925.5 (122.69×) | 188.2 (1.36×) |

`torch` is torch's own kernel. Where a "fastest library" cell names it at a split size, the
other libraries failed or do not support rows that long.

## Status, 2026-11-05

**Open precision issues.** Adversarial audits found problems in two kernels. Both still pass
the random-input gate, and both are timed above.
- **normalized_policy_loss:** rows that are nearly constant at float32 resolution. Two cases fail:
  - the fused streaming paths drop a term of the variance correction;
  - the unfused ablation's variance misses `m * mean_l^2`.
- **ledoit_wolf:** three problems:
  - for b >= 65537 with 64 < d <= 128, the index into the split states overflows int32;
  - the float32 fold of the split states loses accuracy on heavy-tailed data;
  - offsets placed exactly on the reference point's sampled rows defeat it.

**Untimed kernels.** Seven kernels have so far only passed the precision gate, are untuned,
and have no timing here: ess, heat_capacity, householder, layernorm_backward, papr, pearson,
softmax_entropy.

**Workspace sharing.**
- The second-round kernels key their cached workspaces by CUDA stream, and allocate fresh
  ones under graph capture.
- The first-round kernels cache workspaces together with the compiled build. These are
  attention (its unfused variant), contrastive, kurtosis, layernorm and sinkhorn_iteration.
  Two streams, or two captured graphs, running the same size at the same time would share
  the workspaces.

**Subnormals.** Every kernel except ledoit_wolf compiles with `--use_fast_math`. That flushes
float32 values below 1.2e-38 to zero, where torch keeps them.

**pytest.** On 2026-11-04, before the second round, `reproduce.sh pytest` passed 282 tests.
Since then, each changed kernel's tests have passed on their own; the whole suite is still to
be rerun.
