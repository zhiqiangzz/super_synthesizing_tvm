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
"""Reduction-chain operators: each one unfused, fused by hand, and as numpy.

Every operator comes in three forms over the same inputs:

``unfused()``
    The textbook program: one pass over the reduction axis per link of the
    chain (``max``, then ``Σ exp(x - max)``, then ...). This is what
    ``tvm.te.superopt.fuse`` is given.
``fused()``
    The single-pass form written by hand as a tuple ``te.comm_reducer``: the
    reference answer the synthesised program is compared with.
``reference(*arrays)``
    The same function in float64 numpy.

``unfused`` and ``fused`` return ``(inputs, outputs)`` with symbolic extents.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

DTYPE = "float32"


@dataclasses.dataclass(frozen=True)
class Operator:
    name: str
    group: str  # "shift/scale", "closure", "hoist" or "negative"
    doc: str
    unfused: Callable[[], tuple[list, list]]
    reference: Callable[..., tuple]
    fused: Callable[[], tuple[list, list]] | None = None
    # what the synthesis is expected to do with it
    fusible: bool = True
    states: int | None = None  # number of reducer states
    reason: str = ""  # part of the reported reason when it is not fusible
    max_states: int = 4


OPERATORS: dict[str, Operator] = {}


def register(op: Operator) -> Operator:
    assert op.name not in OPERATORS, op.name
    OPERATORS[op.name] = op
    return op


# importing the operator modules is what fills the registry
from . import attention, moments, sinkhorn, softmax

__all__ = ["DTYPE", "OPERATORS", "Operator", "register"]
