#!/usr/bin/env bash
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

# Reproduce the recorded results of super_opt/bench on H800 GPUs (see README.md).
#
#   bash super_opt/bench/reproduce.sh env > super_opt/bench/environment.txt
#   bash super_opt/bench/reproduce.sh gate GPUS [OP ...]      # precision gate, both variants
#   bash super_opt/bench/reproduce.sh time GPU NAME ARGS...   # python -m bench.run ARGS --out NAME.json
#   bash super_opt/bench/reproduce.sh pytest GPU              # the kernel tests
#
# Run it with the environment active: `python` must import torch, triton, tvm_tirx and the
# baseline libraries of environment.txt. GPUS is a comma-separated list of CUDA device
# indices (nvidia-smi order). Use cool, idle cards only: a card that idles hot or
# throttles moves a kernel's time by ~10%, and one that idled at 83 C fell off the bus
# under load. Every job holds a lock on its card for its whole run, so no two jobs this
# script starts share a card; time by hand with the same lock (SUPER_OPT_GPU_LOCKS).

set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPER_OPT="$(dirname "$HERE")"
REPO="$(dirname "$SUPER_OPT")"
LOCKS="${SUPER_OPT_GPU_LOCKS:-${TMPDIR:-/tmp}/super_opt_gpu_locks}"
LOGS="${SUPER_OPT_LOGS:-$HERE/results/logs}"
FUSED_ONLY=" ess heat_capacity " # the kernels without an unfused variant

# on_gpu GPU ARGS...: python ARGS on one card, from the repository root, holding its lock.
on_gpu() {
  local gpu=$1
  shift
  mkdir -p "$LOCKS"
  (
    cd "$REPO"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$SUPER_OPT${PYTHONPATH:+:$PYTHONPATH}"
    export TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
    exec flock "$LOCKS/gpu$gpu.lock" python "$@"
  )
}

all_ops() {
  (cd "$REPO" && PYTHONPATH="$SUPER_OPT" python -c "from bench.ops import SPECS; print(*SPECS)")
}

# next_job QUEUE: pop the queue's first line, under the queue's lock.
next_job() {
  local job
  exec 9> "$1.lock"
  flock 9
  job=$(head -n 1 "$1")
  if [ -n "$job" ]; then sed -i 1d "$1"; fi
  printf '%s' "$job"
}

gate_worker() {
  local gpu=$1 queue=$2 job op variant log
  while job=$(next_job "$queue") && [ -n "$job" ]; do
    op=${job% *}
    variant=${job#* }
    log="$LOGS/gate-$op.$variant.log"
    if on_gpu "$gpu" -m kernels.check --op "$op" --sizes all --seeds 0 1 2 --variant "$variant" \
      > "$log" 2>&1; then
      echo "pass $op ($variant) on GPU $gpu"
    else
      echo "FAIL $op ($variant) on GPU $gpu: $log"
    fi
  done
}

# gate GPUS [OP ...]: every operator (or the given ones), all sizes, seeds 0 1 2, the fused
# kernel and the unfused ablation, as one queue that one worker per card drains.
gate() {
  local gpus=$1 ops op variant gpu queue
  shift
  if [ "$#" -gt 0 ]; then ops=("$@"); else read -r -a ops <<< "$(all_ops)"; fi
  mkdir -p "$LOGS"
  queue="$LOGS/gate.queue"
  : > "$queue"
  for variant in fused unfused; do
    for op in "${ops[@]}"; do
      if [ "$variant" = unfused ] && [[ $FUSED_ONLY == *" $op "* ]]; then continue; fi
      echo "$op $variant" >> "$queue"
    done
  done
  for gpu in ${gpus//,/ }; do
    gate_worker "$gpu" "$queue" &
  done
  wait
  (
    cd "$REPO"
    export PYTHONPATH="$SUPER_OPT"
    python -m kernels.check --summary --op "${ops[@]}" || true
    python -m kernels.check --summary --variant unfused --op "${ops[@]}" || true
  )
}

# time GPU NAME ARGS...: bench.run ARGS, written to results/NAME.json and logged.
time_run() {
  local gpu=$1 name=$2
  shift 2
  mkdir -p "$LOGS"
  on_gpu "$gpu" -m bench.run "$@" --out "$name.json" 2>&1 | tee "$LOGS/time-$name.log"
}

env_report() {
  echo "# The environment of super_opt/bench, written by 'reproduce.sh env' on $(date '+%F %T %Z')."
  echo
  echo "## repository (HEAD when this was written; tvm and tvm_tirx are built from it)"
  git -C "$REPO" log -1 --format='%H %s'
  git -C "$REPO" submodule status 3rdparty/tvm-tirx
  echo
  echo "## GPU"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader | sort -u
  echo
  echo "## toolchain"
  python --version
  if command -v nvcc > /dev/null; then nvcc --version | tail -n 2; fi
  echo
  echo "## python distributions importable from this python (name==version [source])"
  python - << 'EOF'
import importlib.metadata as md
import json

seen = {}
for dist in md.distributions():
    name = dist.metadata["Name"]
    if not name or name.lower() in seen:
        continue
    source = ""
    direct = dist.read_text("direct_url.json")
    if direct:
        url = json.loads(direct)
        vcs = url.get("vcs_info")
        if vcs:
            source = f" [{url['url']}@{vcs['commit_id']}]"
        elif url.get("url", "").startswith("file:"):
            source = " [built locally]"
    seen[name.lower()] = f"{name}=={dist.version}{source}"
for key in sorted(seen):
    print(seen[key])
EOF
}

case "${1:-}" in
  env) env_report ;;
  gate) shift && gate "$@" ;;
  time) shift && time_run "$@" ;;
  pytest) on_gpu "$2" -m pytest -p no:cacheprovider -q tests/python/te/superopt/kernels ;;
  *)
    sed -n '/^# Reproduce/,/^$/p' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac
