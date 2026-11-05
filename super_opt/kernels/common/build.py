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
"""Compiling TIRx functions for the H800 and caching them by size.

``compile_sm90(func)`` runs the TIRx pipeline for sm_90a and returns the
runtime module; call it on torch tensors directly (``mod(a, b)``): TVM FFI
takes CUDA torch tensors as they are. The pipeline compiles with NVRTC and
``--use_fast_math`` (``T.exp`` becomes ``__expf``, as Triton's ``tl.exp``
does). It passes ptxas ``-v``, but NVRTC's log is dropped, so registers and
spills are not reported; ``ptxas_report(mod)`` compiles the source once more
with the same settings and returns them.

``--use_fast_math`` also flushes subnormal float32 values to zero (FTZ): a
square of 1e-20 becomes 0, where torch and torch.compile keep 1e-40. A kernel
whose intermediates can be that small for plausible inputs (sums of squares of
data in units of 1e-10, say) compiles with ``compile_sm90(func, ftz=False)``:
``--use_fast_math --ftz=false`` keeps every other fast-math choice.

``source(mod)`` is the generated CUDA C of the kernel(s), for reading what a
TIRx function turned into; ``SUPER_OPT_DUMP_DIR=<dir>`` writes every compiled
kernel's source there as ``<name>.cu``.
"""

from __future__ import annotations

import functools
import os
import pathlib

import tvm_tirx

TARGET = {"kind": "cuda", "arch": "sm_90a"}
SMS = 132  # H800 SXM


def compile_sm90(func, name: str | None = None, ftz: bool = True):
    """Compile a ``@T.function`` for sm_90a; the result is callable on torch tensors.

    ``ftz=False`` keeps subnormals (``--use_fast_math --ftz=false``).
    """
    spec = dict(TARGET)
    if not ftz:
        spec["backend_config"] = {"cuda": {"nvrtc": ["--use_fast_math", "--ftz=false"]}}
    target = tvm_tirx.target.Target(spec)
    mod = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    dump = os.environ.get("SUPER_OPT_DUMP_DIR")
    if dump:
        path = pathlib.Path(dump) / f"{name or 'kernel'}.cu"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source(mod))
    return mod


def source(mod) -> str:
    """The CUDA C the module's device code was generated as."""
    return mod.mod.imports[0].inspect_source()


def ptxas_report(mod_or_source) -> str:
    """ptxas's lines for every kernel: registers, shared memory, spills, stack frame.

    Takes a compiled module or CUDA source. Compiles it again through NVRTC with
    the pipeline's options, as the agents' scratch ``ptxas.py`` scripts did.
    """
    from tvm_tirx.support.nvcc import _compile_cuda_nvrtc

    code = mod_or_source if isinstance(mod_or_source, str) else source(mod_or_source)
    _, log = _compile_cuda_nvrtc(
        code,
        target_format="cubin",
        arch=TARGET["arch"],
        options=[
            "--use_fast_math",
            "--ptxas-options=-v",
            "--ptxas-options=--warn-on-local-memory-usage",
        ],
    )
    keep = ("Compiling entry", "Used", "spill", "stack frame")
    return "\n".join(line for line in log.splitlines() if any(k in line for k in keep))


def ctas_per_sm(mod_or_source, threads: int, dynamic_smem: int = 0) -> dict[str, int]:
    """How many CTAs of ``threads`` threads fit on one SM, per kernel, from ptxas's report.

    The lowest of sm_90's limits:
    * registers: 4 sub-partitions of 16384, allocated per warp in units of 256;
    * threads: 2048;
    * shared memory: 228 KB, 1 KB of it reserved per CTA;
    * CTAs: 32.

    ptxas only reports static shared memory, so pass the dynamic part. The kurtosis
    kernel lost 10% to a grid sized one CTA per SM larger than this.
    """
    import re

    out, name = {}, None
    for line in ptxas_report(mod_or_source).splitlines():
        entry = re.search(r"Compiling entry function '([^']+)'", line)
        if entry:
            name = entry.group(1)
            continue
        used = re.search(r"Used (\d+) registers", line)
        if used and name:
            regs_per_warp = -(-int(used.group(1)) * 32 // 256) * 256
            static = re.search(r"(\d+) bytes smem", line)
            smem = (int(static.group(1)) if static else 0) + dynamic_smem + 1024
            warps = -(-threads // 32)
            by_regs = 4 * (16384 // regs_per_warp) // warps
            out[name] = min(by_regs, 2048 // (warps * 32), 233472 // smem, 32)
    return out


def key(cfg: dict) -> tuple:
    """A hashable form of a size dict."""
    return tuple(sorted(cfg.items()))


def cached(build):
    """Memoise ``build(cfg, variant=...)`` by the sizes and the variant."""

    @functools.cache
    def by_key(cfg_key, variant):
        return build(dict(cfg_key), variant=variant)

    @functools.wraps(build)
    def wrapper(cfg: dict, variant: str = "fused"):
        return by_key(key(cfg), variant)

    return wrapper
