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
"""16-byte loads and stores of a row, and when a row allows them.

A 16-byte access is one instruction (``uint4``) instead of four to eight, and
it is what a memory-bound kernel needs to approach HBM bandwidth. It is legal
when the address is a multiple of 16 bytes: a row of an ``(n, m)`` tensor
starts at ``row * m`` elements, so every row is aligned exactly when ``m`` is
a multiple of the vector width (``vectorisable``). For other ``m`` take the
scalar path; the sizes are static, so a Python ``if`` picks the path when the
kernel is built and only one of them reaches the CUDA source. The parser still
builds and type-checks *both* arms, though: an arm that would not compile at
these sizes (an index of the wrong rank, say) fails the build. Choose such code
in Python instead, e.g. by passing one of two ``@T.inline`` functions in.

The compiler must be able to *prove* the alignment, which decides how the
index is written: ``T.Ramp(v * vec, 1, vec)`` with ``v`` a vector index works;
an element offset bound with ``T.let`` first does not ("Cannot prove alignment
for CUDA vector store").

``T.cuda.tile.apply`` of ``cp.async`` likewise proves its source region in
bounds and ignores an enclosing ``if``: clamp the indices with ``T.min`` /
``T.max`` (the layernorm kernel does) rather than relying on the guard.

``load_row(buf, A, row, v, vec)``
    ``buf[0:vec] = A[row, v*vec : v*vec+vec]``, into a local array of ``A``'s dtype;
``store_row(A, row, v, buf, vec)``
    the reverse. Widen to float32 with ``T.Cast("float32", buf[i])`` and narrow
    back with ``T.Cast("bfloat16", x)``, one element at a time.
"""

from __future__ import annotations

from tvm_tirx.script import tirx as T

VEC = {"bfloat16": 8, "float16": 8, "float32": 4}  # elements per 16 bytes


def vector_width(dtype: str) -> int:
    return VEC[dtype]


def vectorisable(m: int, dtype: str) -> bool:
    """Is every row of an ``(n, m)`` tensor of ``dtype`` 16-byte aligned?"""
    return m % VEC[dtype] == 0


@T.inline
def load_row(buf, A, row, v, vec):
    buf[T.Ramp(0, 1, vec)] = A[row, T.Ramp(v * vec, 1, vec)]


@T.inline
def store_row(A, row, v, buf, vec):
    A[row, T.Ramp(v * vec, 1, vec)] = buf[T.Ramp(0, 1, vec)]
