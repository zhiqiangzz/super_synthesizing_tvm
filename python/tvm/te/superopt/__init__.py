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
"""Reduction-chain fusion for TE: several passes over an axis become one.

A *reduction chain* is a set of reductions over the same axis in which a
later one reads the final value of an earlier one: the maximum and the
shifted sum of a softmax, the mean and the moments about it. Written that
way a program walks the axis once per link. :func:`fuse` rewrites each
chain into a single tuple ``te.comm_reducer`` reduction -- the online
softmax, FlashAttention's reducer, Welford's variance -- together with
whatever has to happen before and after it.

The reducer is derived from the program, not looked up: its states are the
program's own tensors restricted to a part of the axis, plus what merging
two such parts turns out to need (see :mod:`reducer.states`). It performs
no scheduling and makes no performance claim; that is the job of later
s_TIR passes.

Merge formulas accept two non-empty parts. State selection and proofs do
not evaluate an empty state. The generated serial reduction starts from
the first leaf; parallel execution must initialize each active partition
from its own first leaf and skip partitions with no elements. The full
reduction axis is assumed non-empty.

:func:`judge` answers the question that comes before any derivation: does a
single pass with finitely many states exist for a chain, and how many are
enough? The answer is algebra on the program (see :mod:`reducer.theory`):
one exists where every reduction can be freed of the earlier results it
reads, a sum by linearity, a maximum or minimum by the functions that grow
with their argument. :func:`fuse` asks it first and only then looks for a
reducer.
"""

from .accuracy import AccuracyConfig, AccuracyReport
from .dims import DimKey, DimTable
from .fusion import ChainReport, Fused, fuse, judge
from .reducer.theory import Judgement
from .symbolic import LowerCtx, TensorSem, Unsupported

__all__ = [
    "AccuracyConfig",
    "AccuracyReport",
    "ChainReport",
    "DimKey",
    "DimTable",
    "Fused",
    "Judgement",
    "LowerCtx",
    "TensorSem",
    "Unsupported",
    "fuse",
    "judge",
]
