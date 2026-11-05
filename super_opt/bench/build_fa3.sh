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

# Build FlashAttention-3, attention's "FlashAttention-3" baseline, the way the recorded
# results used it, and install it into the active python's environment.
#
#   bash super_opt/bench/build_fa3.sh [WORKDIR]    # default WORKDIR: a new temporary directory
#
# There is no FA3 wheel on PyPI, so this builds flash-attention v2.8.3's hopper/ (the
# version of the installed FA2) for sm_90a, with only what the benchmark calls: bf16
# forward, head dims 64 and 128. hopper/setup.py downloads nvcc 12.6 unless CUDA is
# 12.8, and that nvcc cannot read CUDA 13 headers ("CUDA compiler and CUDA toolkit headers
# are incompatible"); the one-line change below makes FA3_USE_SYSTEM_NVCC=1 keep the
# environment's nvcc. MAX_JOBS defaults to a third of the hardware threads.

set -eo pipefail
WORK="${1:-$(mktemp -d)}"
SRC="$WORK/flash-attention"
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch v2.8.3 --recurse-submodules --shallow-submodules \
    https://github.com/Dao-AILab/flash-attention.git "$SRC"
fi
cd "$SRC/hopper"
python - << 'EOF'
import pathlib

setup = pathlib.Path("setup.py")
old = 'if bare_metal_version != Version("12.8"):'
new = 'if bare_metal_version != Version("12.8") and os.getenv("FA3_USE_SYSTEM_NVCC") != "1":'
text = setup.read_text()
if new not in text:
    assert text.count(old) == 1, "hopper/setup.py changed: find its nvcc download condition"
    setup.write_text(text.replace(old, new))
EOF

export MAX_JOBS="${MAX_JOBS:-$(($(nproc) / 3))}"
export NVCC_THREADS=1 FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=9.0a
export FA3_USE_SYSTEM_NVCC=1
for flag in BACKWARD SPLIT PAGEDKV APPENDKV LOCAL SOFTCAP PACKGQA FP16 FP8 VARLEN CLUSTER \
  HDIM96 HDIM192 HDIM256 SM80; do
  export "FLASH_ATTENTION_DISABLE_$flag=TRUE"
done
echo "nvcc: $(command -v nvcc) ($(nvcc --version | tail -n 1)), MAX_JOBS=$MAX_JOBS"
python setup.py bdist_wheel
if command -v uv > /dev/null; then
  uv pip install --python "$(command -v python)" --no-deps dist/*.whl
else
  python -m pip install --no-deps dist/*.whl
fi
