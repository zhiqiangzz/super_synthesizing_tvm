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
"""Scratch buffers a kernel keeps between calls, safe across CUDA streams and graphs.

Kernels that split a row (or a matrix) over many CTAs keep partial states and
last-CTA-done counters in device buffers, and cache them so a call allocates
only its output. A buffer cached by size alone is shared by every caller: two
calls of one size on two CUDA streams then race on it (the verifiers of
cross_entropy, softmax, ols_hc0 and normalized_policy_loss produced wrong rows
and counters left non-zero that way), and two CUDA graphs captured from one
``run`` replay on the same memory.

``workspace(key, make)`` returns the buffer(s) for ``key`` on the current device
and stream, calling ``make()`` the first time; ``make`` must return them in the
state a call expects to find (counters zeroed, ``torch.zeros``). Under CUDA-graph
capture it returns a fresh ``make()`` every time, so each graph owns its buffers
(their initialisation is captured with it). Calls on one stream are ordered, so
they can share one buffer -- provided every call leaves it as it found it (the
last CTA resets its counter). TVM FFI launches on torch's current stream.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

import torch

_CACHE: dict[tuple, Any] = {}


def workspace(key: Hashable, make: Callable[[], Any]):
    """The buffers for ``key`` on the current device and CUDA stream (fresh under graph capture)."""
    if torch.cuda.is_current_stream_capturing():
        return make()
    stream = torch.cuda.current_stream()
    full = (key, stream.device.index, stream.cuda_stream)
    found = _CACHE.get(full)
    if found is None:
        found = _CACHE[full] = make()
    return found
