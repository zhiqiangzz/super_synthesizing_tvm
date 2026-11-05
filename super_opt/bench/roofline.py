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
"""The card's peaks, and how close a time is to them.

An implementation cannot be faster than any of four floors, so its floor is the
largest of them:

* HBM: the least traffic it must move (``ops.OpSpec.traffic``) over the
  bandwidth a plain copy reaches;
* tensor cores: its bf16 matrix work (``tc_flops``) over what cuBLAS reaches on
  a large bf16 GEMM;
* float32 FMA: the matrix work it does in float32 (``fp32_flops``,
  sinkhorn_iteration's distances) over 128 lanes x 2 FLOPs per cycle per SM;
* exponentials: ``exps`` over 16 ``ex2`` per cycle per SM, the special
  function units' issue rate.

The first two are measured, not taken from the data sheet (H800 SXM: 3.35 TB/s,
~989 dense bf16 TFLOPS), because the data sheet is not reachable; the last two
are computed from the clock (``FP32_FLOPS_PER_CLOCK``, ``EX2_PER_CLOCK``).
"""

from __future__ import annotations

import json
import time

import torch

from . import RESULTS

EX2_PER_CLOCK = 16  # MUFU.EX2 per cycle per SM on sm_90
FP32_FLOPS_PER_CLOCK = 128 * 2  # FMA lanes per SM, two FLOPs each


def _tirx_copy(n: int):
    """A grid-stride copy with 16-byte loads and stores: the HBM ceiling a TIRx kernel sees."""
    import tvm_tirx
    from tvm_tirx.script import tirx as T

    vec, threads, unroll = 4, 512, 4
    blocks = torch.cuda.get_device_properties(0).multi_processor_count * 4
    chunk = threads * vec * unroll
    assert n % chunk == 0
    iters = n // chunk

    @T.function
    def copy(A: T.Tensor((n,), "float32", align=16), B: T.Tensor((n,), "float32", align=16)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(blocks,), block=(threads,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        it: T.int32 = bx
        while it < iters:
            base: T.let[T.int64] = T.Cast("int64", it) * (chunk // vec)
            for u in T.unroll(unroll):
                v: T.let[T.int64] = base + (u * threads + tx)  # index in 16-byte vectors
                B[T.Ramp(v * vec, 1, vec)] = A[T.Ramp(v * vec, 1, vec)]
            it += blocks

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    return tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": copy}), target=target, pipeline="tirx")


def warm_up(seconds: float = 5.0) -> None:
    """Keep the GPU streaming memory for ``seconds`` before anything is timed.

    Measured on this H800: for the first ~6 s of a process a 1 GiB copy kernel
    runs at 1.2-1.3 TB/s, after that at a steady 2.9 TB/s, whatever the timer.
    Without this every first size of a sweep would be timed on a cold card. The
    load has to run on the SMs: ``cudaMemcpy`` keeps only the copy engines busy
    and leaves the SM clock where it was.
    """
    from tvm_tirx.testing.bench import bench

    a = torch.empty(1 << 28, device="cuda")
    b = torch.empty_like(a)
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            torch.add(a, 0.0, out=b)
            x @ x
        torch.cuda.synchronize()
    # The first Proton session of a process also comes out slow (CUPTI starting
    # up): spend it here, on nothing that is reported.
    bench({"warm-up": lambda: torch.add(a, 0.0, out=b)}, timer="proton", cooldown_s=0.0)


def measure(path=None) -> dict:
    """Measure the HBM and bf16 tensor-core peaks of device 0 and save them."""
    from tvm_tirx.testing.bench import bench

    props = torch.cuda.get_device_properties(0)
    n = 1 << 28  # 1 GiB of float32 per side: far past the 50 MB L2
    a = torch.randn(n, device="cuda")
    b, c = torch.empty_like(a), torch.empty_like(a)
    copy = _tirx_copy(n)
    copy(a, b)
    assert torch.equal(a, b)
    warm_up()
    moved = 2 * n * 4
    kernels = bench(
        {"tirx copy": lambda: copy(a, b)},  # noqa: F821  (a, b, c are deleted only below)
        references={"torch add(x, 0)": lambda: (lambda: torch.add(a, 0.0, out=c))},  # noqa: F821
        timer="proton",
        rounds=3,
    )["impls"]
    # cudaMemcpy is no kernel, so Proton does not see it: time it with events
    memcpy = bench({"cudaMemcpy": lambda: c.copy_(a)}, timer="event", rounds=3)["impls"]  # noqa: F821
    bandwidth = {k: moved / (us * 1e-6) for k, us in {**kernels, **memcpy}.items()}
    # A reduction only reads, and reads stream faster than a copy's mixed read
    # and write (papr through torch.compile beat the copy peak by 2%).
    read = bench({"torch sum": lambda: a.sum()}, timer="proton", rounds=3)  # noqa: F821
    bandwidth["torch sum (read only)"] = n * 4 / (read["impls"]["torch sum"] * 1e-6)
    del a, b, c

    s = 8192
    x = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
    gemm_us = bench({"cuBLAS bf16 8192³": lambda: x @ y}, timer="proton", rounds=3)["impls"]
    tflops = {k: 2 * s**3 / (us * 1e-6) for k, us in gemm_us.items()}

    clock_hz = _max_sm_clock_hz()
    peaks = {
        "device": props.name,
        "sms": props.multi_processor_count,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hbm_bytes_per_s": max(bandwidth.values()),
        "hbm_by_method": bandwidth,
        "bf16_flops_per_s": max(tflops.values()),
        "bf16_by_method": tflops,
        "sm_clock_hz": clock_hz,
        "ex2_per_s": props.multi_processor_count * EX2_PER_CLOCK * clock_hz,
        "fp32_flops_per_s": props.multi_processor_count * FP32_FLOPS_PER_CLOCK * clock_hz,
    }
    path = path or RESULTS / "roofline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(peaks, indent=2))
    return peaks


def load(path=None) -> dict:
    """The saved peaks, measuring them first if there are none."""
    path = path or RESULTS / "roofline.json"
    if not path.exists():
        return measure(path)
    return json.loads(path.read_text())


def _max_sm_clock_hz() -> float:
    """The SM boost clock of torch's device 0, from nvidia-smi (torch does not report it)."""
    import subprocess

    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"  # CUDA_VISIBLE_DEVICES-proof
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits", "-i", uuid],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.split()[0]) * 1e6


def floors(peaks: dict, spec, cfg: dict) -> dict[str, float]:
    """Seconds each of the four floors allows ``spec`` at ``cfg``."""
    return {
        "hbm": spec.traffic(cfg) / peaks["hbm_bytes_per_s"],
        "tensor": spec.tc_flops(cfg) / peaks["bf16_flops_per_s"],
        "fp32": spec.fp32_flops(cfg) / peaks["fp32_flops_per_s"],
        "ex2": spec.exps(cfg) / peaks["ex2_per_s"],
    }


def fraction(peaks: dict, spec, cfg: dict, us: float) -> tuple[str, float]:
    """The binding floor and the fraction of it ``us`` reaches (1.0: at the floor)."""
    bound, seconds = max(floors(peaks, spec, cfg).items(), key=lambda kv: kv[1])
    return bound, seconds / (us * 1e-6)
