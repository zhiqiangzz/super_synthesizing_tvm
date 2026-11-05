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
"""Hand-written TIRx kernels for the showcase operators, for sm_90a (H800).

One module per operator, ``kernels/<op>.py``, named exactly as in
``bench.ops.SPECS``. A module provides

``KERNEL_META``
    a dict: ``op``, ``dtype`` (of its main tensors), ``algorithm`` (one line:
    the single pass and its states), ``structure`` (one line: how the work is
    laid out on the GPU);
``build(cfg, variant="fused")``
    compiles for the static sizes in ``cfg`` (the same dict ``bench.shapes``
    and ``bench.ops.SPECS[op].inputs`` take) and returns ``run``. ``run(*inputs)``
    takes exactly the tensors ``SPECS[op].inputs(cfg, gen)`` makes, allocates
    its outputs, launches, and returns the outputs as a tuple, in the order and
    dtype of ``SPECS[op].unfused``. ``build`` caches by ``cfg``
    (``common.build.cached``).

``check`` holds every kernel to torch.compile's accuracy on random inputs;
``bench.run`` times them next to the baselines. ``common`` is what the
kernels share: compiling, warp and CTA reductions, 16-byte loads and stores,
and ``common.example_logsumexp``, a complete kernel to start from.
"""

from __future__ import annotations

import importlib
import importlib.util

from bench.ops import SPECS

OPS = tuple(SPECS)


def available(op: str) -> bool:
    return importlib.util.find_spec(f"kernels.{op}") is not None


def load(op: str):
    """The kernel module of ``op``, or ``None`` while there is none."""
    return importlib.import_module(f"kernels.{op}") if available(op) else None


def spec_of(op: str):
    """The operator's ``OpSpec``: the showcase's, or the ``SPEC`` a non-showcase module carries."""
    if op in SPECS:
        return SPECS[op]
    return importlib.import_module(f"kernels.{op}").SPEC
