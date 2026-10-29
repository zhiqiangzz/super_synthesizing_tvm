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
"""Tensor reads in TE bodies.

A read ``A[i, j]`` is a ``te.tensor_load`` call: its first argument is the
tensor, the others are the indices, one per axis. Being a call, a read must be
recognised before the intrinsics are. A reduction axis indexes as the
``IterVar`` itself; :func:`load_indices` gives its variable instead, like the
loop variables of the compute axes.
"""

from __future__ import annotations

from tvm.s_tir import IterVar
from tvm_tirx.ir import Call, Op

TENSOR_LOAD = Op.get("te.tensor_load")


def is_load(e) -> bool:
    return isinstance(e, Call) and e.op.same_as(TENSOR_LOAD)


def load_tensor(e):
    """The tensor a read reads."""
    return e.args[0]


def load_indices(e) -> list:
    """The indices of a read, one per axis."""
    return [i.var if isinstance(i, IterVar) else i for i in e.args[1:]]


__all__ = ["TENSOR_LOAD", "is_load", "load_indices", "load_tensor"]
