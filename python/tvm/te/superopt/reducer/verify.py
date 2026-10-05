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
"""Monoid laws for synthesised reducers: identity, commutativity, associativity.

Equalities are decided in the canonical IR, with a case split over the total
orders of the state variables that appear inside ``max`` (``max`` is the only
piecewise operation in the grammar). When a ``max`` has non-variable
arguments the law falls back to random numeric testing.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..symbolic import ir
from ..symbolic.canonicalize import Unsupported, subst, transform


def _state_vars(e: ir.Node) -> set[ir.StateVar]:
    out = set()
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.StateVar):
            out.add(n)
        stack.extend(n.children())
    return out


def _shape_syms(e: ir.Node) -> set[ir.ShapeSym]:
    out = set()
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.ShapeSym):
            out.add(n)
        stack.extend(n.children())
    return out


def _max_nodes(e: ir.Node) -> list[ir.Max]:
    out = []
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Max):
            out.append(n)
        stack.extend(n.children())
    return out


def _resolve_max(e: ir.SymExpr, order: dict[ir.StateVar, int]) -> ir.SymExpr:
    """Replace ``max`` of state variables by the largest one under ``order``."""

    def fn(n: ir.Node):
        if isinstance(n, ir.Max) and all(isinstance(a, ir.StateVar) for a in n.args):
            return max(n.args, key=lambda a: order[a])
        return None

    return transform(e, fn)


def equal_modulo_max(lhs: ir.SymExpr, rhs: ir.SymExpr) -> bool:
    """Canonical equality after case-splitting on orders of the ``max``-ed state variables."""
    if lhs is rhs:
        return True
    maxes = _max_nodes(lhs) + _max_nodes(rhs)
    if not maxes:
        return False
    vars_in_max: set[ir.StateVar] = set()
    for m in maxes:
        if not all(isinstance(a, ir.StateVar) for a in m.args):
            return _numeric_equal(lhs, rhs)
        vars_in_max.update(m.args)
    vs = sorted(vars_in_max, key=lambda v: (v.side, v.k))
    if len(vs) > 6:
        return _numeric_equal(lhs, rhs)
    for perm in itertools.permutations(range(len(vs))):
        order = {v: p for v, p in zip(vs, perm)}
        if _resolve_max(lhs, order) is not _resolve_max(rhs, order):
            return _numeric_equal(lhs, rhs) and False
    return True


def _eval(e: ir.SymExpr, env: dict[ir.Node, float]) -> float:
    if isinstance(e, ir.StateVar | ir.ShapeSym):
        return env[e]
    if isinstance(e, ir.Const):
        return float(e.value)
    if isinstance(e, ir.Add):
        return sum(_eval(a, env) for a in e.args)
    if isinstance(e, ir.Mul):
        out = 1.0
        for a in e.args:
            out *= _eval(a, env)
        return out
    if isinstance(e, ir.Pow):
        return float(np.power(_eval(e.base, env), float(e.exponent)))
    if isinstance(e, ir.Exp):
        return float(np.exp(_eval(e.arg, env)))
    if isinstance(e, ir.Log):
        return float(np.log(_eval(e.arg, env)))
    if isinstance(e, ir.Max):
        return max(_eval(a, env) for a in e.args)
    raise Unsupported(f"cannot evaluate {type(e).__name__}")


def _numeric_equal(lhs: ir.SymExpr, rhs: ir.SymExpr, n: int = 64, seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    vs = sorted(_state_vars(lhs) | _state_vars(rhs), key=lambda v: (v.side, v.k))
    shapes = sorted(_shape_syms(lhs) | _shape_syms(rhs), key=lambda v: v.name)
    try:
        for _ in range(n):
            env: dict[ir.Node, float] = {v: float(rng.uniform(-2, 2)) for v in vs}
            env.update({v: float(rng.integers(1, 9)) for v in shapes})
            with np.errstate(all="ignore"):
                a, b = _eval(lhs, env), _eval(rhs, env)
            if not np.isclose(a, b, rtol=1e-8, atol=1e-10):
                return False
    except Unsupported:
        return False
    return True


def check_laws(merge: tuple[ir.SymExpr, ...], identity: tuple[ir.Const, ...]) -> str | None:
    """Return the proof method (``"canonical"``/``"case-split"``/``"numeric"``) or ``None``."""
    n = len(merge)
    a = [ir.state_var("a", k) for k in range(n)]
    b = [ir.state_var("b", k) for k in range(n)]
    c = [ir.state_var("c", k) for k in range(n)]
    method = "canonical"
    # identity: M(a, e) == a
    for k in range(n):
        got = subst(merge[k], {b[i]: identity[i] for i in range(n)})
        if got is not a[k]:
            if not equal_modulo_max(got, a[k]):
                return None
            method = "case-split"
    # commutativity: M(a, b) == M(b, a)
    swap = {}
    for i in range(n):
        swap[a[i]] = b[i]
        swap[b[i]] = a[i]
    for k in range(n):
        got = subst(merge[k], swap)
        if got is not merge[k]:
            if not equal_modulo_max(got, merge[k]):
                return None
            method = "case-split"
    # associativity: M(M(a, b), c) == M(a, M(b, c))
    to_c = {b[i]: c[i] for i in range(n)}
    m_bc = [
        subst(subst(merge[i], {a[j]: b[j] for j in range(n)} | {b[j]: c[j] for j in range(n)}), {})
        for i in range(n)
    ]
    for k in range(n):
        lhs = subst(merge[k], {a[i]: merge[i] for i in range(n)} | to_c)
        rhs = subst(merge[k], {b[i]: m_bc[i] for i in range(n)})
        if lhs is not rhs:
            if not equal_modulo_max(lhs, rhs):
                if _numeric_equal(lhs, rhs):
                    method = "numeric"
                    continue
                return None
            if method == "canonical":
                method = "case-split"
    return method
