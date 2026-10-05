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
"""Static analysis of the target program and the shared search context."""

from __future__ import annotations

import dataclasses

from .config import Bounds
from .dims import DimKey, DimTable
from .symbolic import ir
from .symbolic.canonicalize import elems, reductions
from .symbolic.lower import LowerCtx, TensorSem


def is_constant(e: ir.Node) -> bool:
    """True if ``e`` reads no tensor data and no bound index (an extent-only value)."""
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Elem | ir.BIdx | ir.Idx | ir.StateVar | ir.Atom | ir.Reduce):
            return False
        if isinstance(n, ir.MonoidReduce):
            return False
        stack.extend(n.children())
    return True


def constant_subterms(body: ir.SymExpr) -> list[ir.SymExpr]:
    """Maximal data-free scalar sub-terms of ``body`` (the search's constant pool)."""
    out: dict[ir.SymExpr, None] = {}

    def visit(n: ir.Node) -> None:
        if isinstance(n, ir.SymExpr) and is_constant(n):
            if not isinstance(n, ir.Const) or n.value not in (0, 1):
                out[n] = None
            return
        if isinstance(n, ir.Mul | ir.Add):
            # A constant factor / term of a compound is itself a constant.
            const_parts = [a for a in n.args if is_constant(a)]
            if len(const_parts) > 1:
                from .symbolic.canonicalize import mk_add, mk_mul

                combined = (mk_mul if isinstance(n, ir.Mul) else mk_add)(*const_parts)
                if not (isinstance(combined, ir.Const) and combined.value in (0, 1)):
                    out[combined] = None
        for c in n.children():
            visit(c)

    visit(body)
    return list(out)


@dataclasses.dataclass(frozen=True)
class Target:
    sem: TensorSem
    reduce_axes: frozenset[DimKey]
    tensors: frozenset[int]
    consts: tuple[ir.SymExpr, ...]
    # extents the target uses for two independent indices at once (two output
    # axes, or two distinct reductions): the only ones worth an outer product
    repeated_keys: frozenset[DimKey] = frozenset()

    @property
    def body(self) -> ir.SymExpr:
        return self.sem.body

    @property
    def interface(self) -> tuple:
        return (self.sem.axis_keys, self.sem.dtype)


def analyze_target(sem: TensorSem) -> Target:
    axes = frozenset(r.domain.axis for r in reductions(sem.body))
    monoid_axes = set()
    stack: list[ir.Node] = [sem.body]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.MonoidReduce):
            monoid_axes.add(n.domain.axis)
        stack.extend(n.children())
    uses = list(sem.axis_keys) + [r.domain.axis for r in set(reductions(sem.body))]
    return Target(
        sem=sem,
        reduce_axes=axes | frozenset(monoid_axes),
        tensors=frozenset(elems(sem.body)),
        consts=tuple(constant_subterms(sem.body)),
        repeated_keys=frozenset(k for k in uses if uses.count(k) > 1),
    )


@dataclasses.dataclass
class SearchCtx:
    """Everything an operator builder or pruning hook may need."""

    lower: LowerCtx
    bounds: Bounds
    target: Target
    dtype: str  # the single working dtype of the search (that of the target)
    stats: dict = dataclasses.field(default_factory=dict)
    # The program the search started from (partialisation reads its TE graph).
    output: object = None
    inputs: tuple = ()
    # Static precision properties of the original that every result must keep
    # ("exp", "domain"; see ``accuracy``): violating ops are pruned at once.
    static_required: tuple[str, ...] = ()

    @property
    def dims(self) -> DimTable:
        return self.lower.dims

    def count(self, what: str, n: int = 1) -> None:
        self.stats[what] = self.stats.get(what, 0) + n
