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
"""Public entry point: ``superoptimize(output, inputs, bounds)``."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from tvm import te

from .config import Bounds
from .enumerate import Enumerator, Found
from .equivalence import Verdict
from .pool import ProgramSnapshot
from .symbolic.lower import LowerCtx, TensorSem
from .target import SearchCtx, analyze_target
from .tensor_ops import OpSpec, default_specs


@dataclasses.dataclass
class Result:
    """One equivalent program: its op sequence, proof status and (lazily) real TE."""

    snapshot: ProgramSnapshot
    verdict: Verdict
    sem: TensorSem
    ctx: SearchCtx
    inputs: list[te.Tensor]
    _tensor: te.Tensor | None = None

    @property
    def n_ops(self) -> int:
        return len(self.snapshot.ops)

    def materialize(self) -> te.Tensor:
        if self._tensor is None:
            self._tensor = self.snapshot.materialize(self.inputs, self.ctx)
        return self._tensor

    def prim_func(self):
        return te.create_prim_func([*self.inputs, self.materialize()])

    def describe(self) -> str:
        return f"[{self.verdict.method}] {len(self.snapshot.ops)} ops\n{self.snapshot.describe()}"

    def te_source(self) -> str:
        """The program as runnable TE Python source (placeholders, computes, reducers)."""
        from .emit import emit_program

        return emit_program(self.snapshot, self.inputs, self.ctx)

    def reducers(self) -> list:
        return [
            rec.params
            for rec in self.snapshot.ops
            if getattr(rec.spec, "name", "") == "comm_reduce"
        ]


def make_context(
    output: te.Tensor, inputs: list[te.Tensor], bounds: Bounds | None = None
) -> tuple[SearchCtx, list[TensorSem]]:
    bounds = bounds or Bounds()
    lower = LowerCtx()
    for t in inputs:
        lower.tensor_id(t)
    target_sem = lower.lower(output)
    ctx = SearchCtx(
        lower=lower, bounds=bounds, target=analyze_target(target_sem), dtype=target_sem.dtype
    )
    input_sems = [lower.lower(t) for t in inputs]
    return ctx, input_sems


def iter_superoptimize(
    output: te.Tensor,
    inputs: list[te.Tensor],
    bounds: Bounds | None = None,
    specs: list[OpSpec] | None = None,
    with_reducers: bool = True,
) -> Iterator[Result]:
    ctx, input_sems = make_context(output, inputs, bounds)
    if specs is None:
        specs = default_specs()
        if with_reducers:
            from .reducer.op import CommReduce

            specs = [*specs, CommReduce()]
    enum = Enumerator(ctx, specs, input_sems)
    for found in enum.run():
        yield Result(found.snapshot, found.verdict, found.sem, ctx, list(inputs))


def superoptimize(
    output: te.Tensor,
    inputs: list[te.Tensor],
    bounds: Bounds | None = None,
    *,
    specs: list[OpSpec] | None = None,
    with_reducers: bool = True,
    max_results: int | None = None,
) -> list[Result]:
    """Enumerate TE programs equivalent to ``output`` over the given ``inputs``."""
    out: list[Result] = []
    for r in iter_superoptimize(output, inputs, bounds, specs, with_reducers):
        out.append(r)
        if max_results is not None and len(out) >= max_results:
            break
    return out


__all__ = ["Found", "Result", "iter_superoptimize", "make_context", "superoptimize"]
