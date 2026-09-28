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
"""TE algorithm superoptimizer: enumerate and verify equivalent TE programs.

Given a TE tensor with symbolic shapes, :func:`superoptimize` searches a
bounded space of real TE programs built from a small operator grammar --
including automatically synthesised tuple ``te.comm_reducer`` reductions --
and returns those proven equivalent under real-number semantics. It performs
no performance analysis; that is the job of later s_TIR passes.
"""

from .api import Result, iter_superoptimize, superoptimize
from .config import Bounds
from .dims import DimKey, DimTable
from .symbolic import LowerCtx, TensorSem, Unsupported, lower_tensor

__all__ = [
    "Bounds",
    "DimKey",
    "DimTable",
    "LowerCtx",
    "Result",
    "TensorSem",
    "Unsupported",
    "iter_superoptimize",
    "lower_tensor",
    "superoptimize",
]
