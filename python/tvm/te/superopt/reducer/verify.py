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
"""Equalities between derived expressions, decided by proof.

:func:`prove_equal` decides that two expressions are equal for every value of
their variables, by proof and nothing else:

1. the canonical form is the same;
2. otherwise a ``max`` that does not move with a reduced index is split into
   cases, one per argument: in the region where that argument is the
   largest the ``max`` *is* that argument, the regions cover everything, so
   equality in every case is equality. Cases are split one ``max`` at a
   time, re-canonicalising in between;
3. in each case the two sides are compared as rational functions by SymPy
   (:mod:`..symbolic.sympy_bridge`), which is where a common denominator is
   formed.

Anything else is *not proven* (``None``), never "probably equal".

Associativity and commutativity on non-empty parts follow from the derived
merge invariant (see :mod:`derive`). Empty parts are skipped by execution.
"""

from __future__ import annotations

from ..symbolic import ir, sympy_bridge
from ..symbolic.canonicalize import Unsupported, recanonicalize, subst, walk

MAX_CASES = 64  # cases a proof may split into
_RANK = {"canonical": 0, "case-split": 1, "sympy": 2}


def worst(*methods: str) -> str:
    """The weakest of several proof methods (the one a combined proof rests on)."""
    return max(methods, key=_RANK.__getitem__)


def _open_max(e: ir.Node) -> ir.Max | None:
    """A ``max`` that does not move with an index bound by a reduction around it."""
    for n in walk(e):
        if isinstance(n, ir.Max) and not n.levels:
            return n
    return None


def prove_equal(lhs: ir.SymExpr, rhs: ir.SymExpr) -> str | None:
    """How ``lhs == rhs`` is proven (``"canonical"``, ``"case-split"`` or
    ``"sympy"``), or ``None`` when it is not."""
    budget = [MAX_CASES]
    return _prove(lhs, rhs, budget)


def _prove(lhs: ir.SymExpr, rhs: ir.SymExpr, budget: list[int]) -> str | None:
    if lhs is rhs:
        return "canonical"
    split = _open_max(lhs) or _open_max(rhs)
    if split is None:
        return "sympy" if sympy_bridge.equal(lhs, rhs) else None
    budget[0] -= len(split.args)
    if budget[0] < 0:
        return None
    method = "case-split"
    for arg in split.args:
        try:
            l2, r2 = subst(lhs, {split: arg}), subst(rhs, {split: arg})
        except Unsupported:
            return None
        got = _prove(l2, r2, budget)
        if got is None:
            return None
        method = worst(method, got)
    return method


def proves(printed: ir.SymExpr, canonical: ir.SymExpr) -> str | None:
    """How an expression built with the raw constructors equals a canonical one."""
    try:
        got = recanonicalize(printed)
    except Unsupported:
        return None
    return prove_equal(got, canonical)


__all__ = ["prove_equal", "proves", "worst"]
