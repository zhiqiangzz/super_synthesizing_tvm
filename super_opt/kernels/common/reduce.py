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
"""All-reductions of a tuple state across a warp and across a CTA.

A one-pass reduction carries a *state* per thread: ``k`` float32 values in a
local array (``T.alloc_local((k,), "float32")``), e.g. ``(m, l)`` for
log-sum-exp. Merging two states is the operator's *combine*, an ``@T.inline``
``combine(st, ot)`` that overwrites ``st`` with ``st ⊕ ot``; it is the
operator's own (``chain_ops/<file>.py``'s ``<op>_fused()`` writes it as a
tuple ``te.comm_reducer``). These helpers only move states around and call it:

``warp_allreduce(st, ot, k, combine)``
    butterfly over 32 lanes (offsets 16, 8, 4, 2, 1): afterwards every lane
    holds the warp's total. ``ot`` is scratch of the same shape.
``cta_allreduce(st, ot, k, combine, smem, warp, lane, warps)``
    the warp totals through shared memory ``smem`` (``T.alloc_shared((warps,
    k), "float32")``), merged in warp order by every thread: afterwards every
    thread of the CTA holds the CTA's total. Ends with a barrier, so ``smem``
    can be reused at once.

The merge order is fixed (butterfly, then warp 0, 1, 2, ...), so a kernel's
result is the same from run to run. Write the combine for any pair of states
a lane can hold, including a lane that saw no element: start such a lane at
the combine's identity (for a maximum use a finite ``-3e38`` rather than
``-inf``, so that ``exp(m - m)`` of two empty states is 1 and not NaN).

``fmax(a, b)`` / ``fmin(a, b)``
    ``fmaxf`` / ``fminf``: one instruction. ``T.max`` / ``T.min`` lower to a
    NaN-propagating compare-and-select instead, which cost the attention,
    contrastive and sinkhorn kernels several percent each in their inner loops.
"""

from __future__ import annotations

from tvm_tirx.script import tirx as T

FULL_MASK = 0xFFFFFFFF


def fmax(a, b):
    return T.call_pure_extern("fmaxf", a, b, ty="float32")


def fmin(a, b):
    return T.call_pure_extern("fminf", a, b, ty="float32")


@T.inline
def warp_allreduce(st, ot, k, combine):
    for i in T.unroll(5):
        for j in T.unroll(k):
            ot[j] = T.gpu_warp_shuffle_xor(FULL_MASK, st[j], 16 >> i, 32, 32)
        combine(st, ot)


@T.inline
def cta_allreduce(st, ot, k, combine, smem, warp, lane, warps):
    warp_allreduce(st, ot, k, combine)
    if lane == 0:
        for j in T.unroll(k):
            smem[warp, j] = st[j]
    T.cuda.cta_sync()
    for j in T.unroll(k):
        st[j] = smem[0, j]
    if warps > 1:
        for w in T.unroll(warps - 1):
            for j in T.unroll(k):
                ot[j] = smem[w + 1, j]
            combine(st, ot)
    T.cuda.cta_sync()
