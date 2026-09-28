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
"""Result-directed pruning by abstract-expression containment (after Mirage).

A candidate intermediate whose (constant-abstracted) semantics is not a
"sub-expression" of the target's cannot be completed into the target, so the
prefix is discarded. The relation deliberately errs on the permissive side:
it never rejects a prefix of a program the rewrite theory can prove
equivalent to the target.
"""

from __future__ import annotations

from .symbolic import ir
from .symbolic.canonicalize import shift
from .target import is_constant

_MEMO: dict[tuple[ir.Node, ir.Node], bool] = {}


def contains(sup: ir.SymExpr, sub: ir.SymExpr, depth: int = 0) -> bool:
    """Is ``sub`` (at binder depth ``depth``) an abstract sub-expression of ``sup``?"""
    if sup is sub or is_constant(sub):
        return True
    if isinstance(sub, ir.Elem):
        return _reads_tensor(sup, sub.tensor)
    key = (sup, sub)
    hit = _MEMO.get(key)
    if hit is not None:
        return hit
    r = _contains(sup, sub, depth)
    _MEMO[key] = r
    return r


def _reads_tensor(e: ir.Node, tensor: int) -> bool:
    key = ("reads", tensor)
    hit = e.cache.get(key)
    if hit is not None:
        return hit
    r = isinstance(e, ir.Elem) and e.tensor == tensor
    if not r:
        r = any(_reads_tensor(c, tensor) for c in e.children())
    e.cache[key] = r
    return r


def _same_kind(a: ir.Node, b: ir.Node) -> bool:
    return type(a) is type(b)


def _contains(sup: ir.SymExpr, sub: ir.SymExpr, depth: int) -> bool:
    if isinstance(sub, ir.Add | ir.Mul | ir.Max) and _same_kind(sup, sub):
        if _match_args(sup.args, tuple(a for a in sub.args if not is_constant(a)), depth):
            return True
    if isinstance(sub, ir.Exp) and isinstance(sup, ir.Exp):
        if contains(sup.arg, sub.arg, depth):
            return True
    if isinstance(sub, ir.Pow) and isinstance(sup, ir.Pow) and sup.exponent == sub.exponent:
        if contains(sup.base, sub.base, depth):
            return True
    if isinstance(sub, ir.Reduce) and isinstance(sup, ir.Reduce):
        if sup.kind == sub.kind and sup.domain.axis == sub.domain.axis:
            if contains(sup.body, sub.body, depth + 1):
                return True
    if isinstance(sub, ir.MonoidReduce) and isinstance(sup, ir.MonoidReduce):
        if sup.domain.axis == sub.domain.axis:
            return True
    # Plain sub-term: descend into sup.
    if isinstance(sup, ir.Reduce):
        return contains(sup.body, shift(sub, depth, 1), depth + 1)
    if isinstance(sup, ir.MonoidReduce):
        shifted = shift(sub, depth, 1)
        return any(contains(x, shifted, depth + 1) for x in sup.leaf)
    return any(contains(c, sub, depth) for c in sup.children() if isinstance(c, ir.SymExpr))


def _match_args(sup_args: tuple, sub_args: tuple, depth: int) -> bool:
    """Each sub arg must be contained in a distinct sup arg (small bipartite matching)."""
    if len(sub_args) > len(sup_args):
        return False
    if not sub_args:
        return True
    used: list[bool] = [False] * len(sup_args)

    def rec(i: int) -> bool:
        if i == len(sub_args):
            return True
        for j, s in enumerate(sup_args):
            if not used[j] and contains(s, sub_args[i], depth):
                used[j] = True
                if rec(i + 1):
                    return True
                used[j] = False
        return False

    return rec(0)
