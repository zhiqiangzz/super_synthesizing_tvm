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
"""Bounded-exhaustive enumeration of TE programs (iterative deepening DFS).

Canonical generation: an op that does not consume the previous op's output
must have a larger order key than the previous op, which admits exactly one
topological order per program (the greedy smallest-key order). Pruning is
purely semantic: legality, semantic dedup, size, axis feasibility and
abstract-expression containment.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable, Iterator

from .abstract_expr import contains
from .equivalence import NumericOracle, Verdict, equivalent
from .pool import PoolEntry, Program, ProgramSnapshot
from .symbolic.canonicalize import Unsupported
from .symbolic.lower import TensorSem
from .target import SearchCtx
from .tensor_ops import OpSpec


@dataclasses.dataclass(frozen=True)
class Found:
    snapshot: ProgramSnapshot
    verdict: Verdict
    sem: TensorSem


def accept_entry(sem: TensorSem, ctx: SearchCtx, result_directed: bool = False) -> bool:
    """Semantic pruning of a freshly produced tensor.

    ``result_directed`` outputs (synthesised reducer states) were derived from
    the target's goal closure and are exempt from the containment check.
    """
    if sem.body.size > ctx.bounds.max_sem_nodes:
        ctx.count("prune:size")
        return False
    if not result_directed and not contains(ctx.target.body, sem.body):
        ctx.count("prune:abstract")
        return False
    return True


def choose_operands(prog: Program, spec: OpSpec) -> Iterable[tuple[int, ...]]:
    n = len(prog.pool)
    if spec.arity == 1:
        for i in range(n):
            yield (i,)
    elif spec.arity == 2:
        if spec.commutative:
            for i in range(n):
                for j in range(i, n):
                    if i != j or not spec.distinct:
                        yield (i, j)
        else:
            for i in range(n):
                for j in range(n):
                    if i != j or not spec.distinct:
                        yield (i, j)
    else:
        for r in spec.arities(prog):
            for combo in itertools.combinations(range(n), r):
                yield combo


class Enumerator:
    def __init__(self, ctx: SearchCtx, specs: list[OpSpec], inputs: list[TensorSem]):
        self.ctx = ctx
        self.specs = sorted(specs, key=lambda s: s.rank)
        self.inputs = inputs
        self.oracle = NumericOracle(ctx)
        self.max_arity = max(max(s.arity, getattr(s, "max_arity", 0)) for s in specs)
        self.found_keys: set = set()

    def run(self, max_ops: int | None = None) -> Iterator[Found]:
        max_ops = self.ctx.bounds.max_tensor_ops if max_ops is None else max_ops
        prog = Program(self.inputs)
        for budget in range(0, max_ops + 1):
            self.ctx.count("budget")
            yield from self._dfs(prog, budget, budget)

    # -- DFS -----------------------------------------------------------------
    def _dfs(self, prog: Program, remaining: int, budget: int) -> Iterator[Found]:
        ctx = self.ctx
        last = prog.last_op_index
        if last is not None or budget == 0:
            yield from self._check_outputs(prog, remaining)
        if remaining == 0:
            return
        # Dead-code bound: every intermediate must eventually feed the output.
        dead = self._unconsumed_ops(prog)
        if dead - 1 > remaining * (self.max_arity - 1):
            ctx.count("prune:dead")
            return
        last_key = prog.ops[last].order_key if last is not None else None
        last_outputs = set(prog.ops[last].outputs) if last is not None else set()
        for spec in self.specs:
            for operands in list(choose_operands(prog, spec)):
                entries = [prog.pool[i] for i in operands]
                consumes_last = bool(last_outputs.intersection(operands))
                base_key = (spec.rank, tuple(prog.pool[i].sem.body.uid for i in operands))
                try:
                    params_list = list(spec.params(entries, ctx))
                except Unsupported:
                    continue
                for params in params_list:
                    order_key = (*base_key, spec.param_key(params))
                    if not consumes_last and last_key is not None and order_key <= last_key:
                        ctx.count(f"prune:order:{spec.name}")
                        continue
                    try:
                        outputs = spec.apply_sem(entries, params, ctx)
                    except Unsupported:
                        ctx.count("prune:unsupported")
                        continue
                    ctx.count("candidates")
                    if any(prog.has(o) for o in outputs):
                        ctx.count("prune:dedup")
                        continue
                    rd = getattr(spec, "result_directed", False)
                    if not all(accept_entry(o, ctx, rd) for o in outputs):
                        continue
                    ctx.count("expanded")
                    ctx.count(f"expand:{spec.name}")
                    prog.push(spec, operands, params, outputs, order_key)
                    try:
                        yield from self._dfs(prog, remaining - 1, budget)
                    finally:
                        prog.pop()

    def _unconsumed_ops(self, prog: Program) -> int:
        n = 0
        for rec in prog.ops:
            if all(prog.consumers[o] == 0 for o in rec.outputs):
                n += 1
        return n

    def _check_outputs(self, prog: Program, remaining: int) -> Iterator[Found]:
        if remaining != 0:
            return  # iterative deepening: only report at the exact budget
        ctx = self.ctx
        target = ctx.target.sem
        candidates: list[PoolEntry]
        if prog.ops:
            candidates = [prog.pool[i] for i in prog.ops[-1].outputs]
        else:
            candidates = list(prog.pool)
        n_ops = len(prog.ops)
        for entry in candidates:
            sem = entry.sem
            if sem.axis_keys != target.axis_keys or sem.dtype != target.dtype:
                continue
            if len(prog.fan_in_ops(entry.index)) != n_ops:
                ctx.count("prune:deadcode")
                continue
            ctx.count("interface")
            verdict = equivalent(sem, target, ctx, self.oracle)
            if not verdict.equivalent:
                ctx.count("interface:rejected")
                continue
            snap = prog.snapshot(entry.index)
            key = tuple((r.spec.name, r.operands, r.spec.param_key(r.params)) for r in snap.ops)
            if key in self.found_keys:
                continue
            self.found_keys.add(key)
            yield Found(snap, verdict, sem)
