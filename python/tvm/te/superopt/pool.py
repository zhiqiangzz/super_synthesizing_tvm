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
"""The search state: a pool of tensors and the op sequence that produced them.

During the search no ``te.Tensor`` is built: each pool entry carries only its
symbolic semantics, axis keys and dtype. A finished :class:`Program` is
*materialised* into real TE by replaying its op records on the input
placeholders.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tvm import te

from .dims import DimKey
from .symbolic.lower import TensorSem


@dataclasses.dataclass(frozen=True)
class PoolEntry:
    sem: TensorSem
    op_index: int | None  # None for inputs
    slot: int  # output slot within the op
    index: int  # position in the pool

    @property
    def shape(self) -> tuple[DimKey, ...]:
        return self.sem.axis_keys

    @property
    def dtype(self) -> str:
        return self.sem.dtype

    @property
    def rank(self) -> int:
        return self.sem.rank

    @property
    def key(self) -> tuple:
        return (self.sem.body, self.sem.axis_keys, self.sem.dtype)


@dataclasses.dataclass(frozen=True)
class OpRecord:
    spec: Any  # OpSpec
    operands: tuple[int, ...]  # pool indices
    params: Any
    outputs: tuple[int, ...]  # pool indices of the outputs
    order_key: tuple


class Program:
    """Mutable stack of op records over a pool; supports push/pop for DFS."""

    def __init__(self, inputs: list[TensorSem]) -> None:
        self.pool: list[PoolEntry] = []
        self.seen: dict[tuple, int] = {}
        self.ops: list[OpRecord] = []
        self.consumers: list[int] = []  # per pool index: number of consuming ops
        self.n_inputs = len(inputs)
        for sem in inputs:
            self._append(sem, None, 0)

    def _append(self, sem: TensorSem, op_index: int | None, slot: int) -> PoolEntry:
        entry = PoolEntry(sem, op_index, slot, len(self.pool))
        self.pool.append(entry)
        self.consumers.append(0)
        self.seen[entry.key] = entry.index
        return entry

    def has(self, sem: TensorSem) -> bool:
        return (sem.body, sem.axis_keys, sem.dtype) in self.seen

    def push(self, spec, operands: tuple[int, ...], params, outputs: list[TensorSem], order_key):
        op_index = len(self.ops)
        out_idx = []
        for slot, sem in enumerate(outputs):
            out_idx.append(self._append(sem, op_index, slot).index)
        for o in operands:
            self.consumers[o] += 1
        self.ops.append(OpRecord(spec, tuple(operands), params, tuple(out_idx), order_key))

    def pop(self) -> None:
        rec = self.ops.pop()
        for o in rec.operands:
            self.consumers[o] -= 1
        for _ in rec.outputs:
            entry = self.pool.pop()
            self.consumers.pop()
            self.seen.pop(entry.key, None)

    @property
    def last_op_index(self) -> int | None:
        return len(self.ops) - 1 if self.ops else None

    def unconsumed(self) -> int:
        return sum(1 for i in range(self.n_inputs, len(self.pool)) if self.consumers[i] == 0)

    def fan_in_ops(self, index: int) -> set[int]:
        """Indices of the ops in the transitive fan-in of pool entry ``index``."""
        out: set[int] = set()
        stack = [index]
        while stack:
            i = stack.pop()
            e = self.pool[i]
            if e.op_index is None or e.op_index in out:
                continue
            out.add(e.op_index)
            stack.extend(self.ops[e.op_index].operands)
        return out

    def snapshot(self, output: int) -> ProgramSnapshot:
        return ProgramSnapshot(
            n_inputs=self.n_inputs,
            ops=tuple(self.ops),
            output=output,
            pool=tuple(self.pool),
        )


@dataclasses.dataclass(frozen=True)
class ProgramSnapshot:
    n_inputs: int
    ops: tuple[OpRecord, ...]
    output: int
    pool: tuple[PoolEntry, ...]

    def materialize(self, inputs: list[te.Tensor], ctx) -> te.Tensor:
        """Replay the op records on real placeholders and return the output tensor."""
        tensors: list[te.Tensor] = list(inputs)
        assert len(tensors) == self.n_inputs
        for rec in self.ops:
            outs = rec.spec.build([tensors[i] for i in rec.operands], rec.params, ctx)
            assert len(outs) == len(rec.outputs)
            tensors.extend(outs)
        return tensors[self.output]

    def describe(self) -> str:
        lines = []
        for i, rec in enumerate(self.ops):
            outs = ",".join(f"t{o}" for o in rec.outputs)
            args = ",".join(f"t{o}" for o in rec.operands)
            lines.append(
                f"op{i}: {outs} = {rec.spec.name}({args}; {rec.spec.describe(rec.params)})"
            )
        lines.append(f"output: t{self.output}")
        return "\n".join(lines)
