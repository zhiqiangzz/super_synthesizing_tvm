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
"""Prior-free building blocks for reducer state definitions.

A state of a tuple reducer is a function ``I(R)`` of *atoms*: partial
reductions ``Σ_{j∈R} b_j`` / ``max_{j∈R} b_j`` whose bodies are *signals*,
i.e. sub-terms of the target's own reduction bodies that vary with the
reduction index (plus the constant ``1``, whose sum is the count). The
state itself is the atom combined with a small expression over the atoms
introduced before it. Nothing in this module encodes a particular online
algorithm; the search is bounded by expression sizes only.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..symbolic import ir
from ..symbolic.canonicalize import (
    Unsupported,
    mk_add,
    mk_div,
    mk_exp,
    mk_log,
    mk_mul,
    mk_pow,
    mk_reduce,
    mk_sub,
    positive,
)

FORMS = ("mul", "add", "sub")


def signals(goals: Iterable[ir.Reduce]) -> list[ir.SymExpr]:
    """Sub-terms of the goal bodies that depend on the reduction index (depth 1)."""
    out: dict[ir.SymExpr, None] = {}
    for g in goals:
        stack: list[ir.Node] = [g.body]
        while stack:
            n = stack.pop()
            if (
                isinstance(n, ir.SymExpr)
                and not isinstance(n, ir.Const)
                and n.uses_level(0)
                and n.levels <= {0}
            ):
                out[n] = None
            stack.extend(n.children())
    return list(out)


def atoms_in(e: ir.Node, domain: ir.Domain) -> list[ir.SymExpr]:
    """Reductions (and cardinalities) over ``domain`` in ``e``: the unknowns of a state."""
    out: dict[ir.SymExpr, None] = {}
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Reduce | ir.Card) and n.domain is domain:
            out[n] = None
            continue
        stack.extend(n.children())
    return list(out)


def atom_candidates(sigs: Iterable[ir.SymExpr], R: ir.DSym) -> list[ir.SymExpr]:
    """``Σ_R b`` and ``max_R b`` for every signal ``b`` and for ``b = 1`` (the count)."""
    out: dict[ir.SymExpr, None] = {}
    for body in (ir.ONE, *sigs):
        for kind in ("sum", "max"):
            e = mk_reduce(kind, R, 0, body)  # canonicalisation may pull factors out
            for a in atoms_in(e, R):
                out[a] = None
    return list(out)


def expr_table(
    leaves: Iterable[ir.SymExpr], consts: Iterable[ir.SymExpr], max_size: int
) -> dict[ir.SymExpr, int]:
    """Expressions over ``leaves`` and ``consts`` with at most ``max_size`` productions.

    Productions: ``x²``, ``exp x``, ``log x`` (positive ``x``), ``x + y``,
    ``x - y``, ``x * y``, ``x / y``; leaves cost one, constants nothing.
    Expressions are kept in canonical form, so algebraically equal ones
    appear once. Pure constants are only admitted as leaves.
    """
    table: dict[ir.SymExpr, int] = {}
    by_size: dict[int, list[ir.SymExpr]] = {}

    def put(e: ir.SymExpr, size: int) -> None:
        if e in table:
            return
        if size > 0 and (_is_data_free(e) or _rationally_scaled(e)):
            return
        table[e] = size
        by_size.setdefault(size, []).append(e)

    _POOL.clear()
    _POOL.update(consts)
    for c in consts:
        put(c, 0)
    for x in leaves:
        put(x, 1)
    for size in range(2, max_size + 1):
        for e in list(by_size.get(size - 1, [])):
            for f in (lambda x: mk_pow(x, 2), mk_exp, _safe_log):
                try:
                    r = f(e)
                except Unsupported:
                    continue
                if r is not None:
                    put(r, size)
        for sa in range(0, size - 1):
            sb = size - 1 - sa
            if sb < sa:
                continue
            for ea in list(by_size.get(sa, [])):
                for eb in list(by_size.get(sb, [])):
                    for op in (mk_add, mk_sub, mk_mul, mk_div):
                        for x, y in ((ea, eb), (eb, ea)):
                            try:
                                put(op(x, y), size)
                            except Unsupported:
                                pass
    return table


def _rationally_scaled(e: ir.SymExpr) -> bool:
    """A coefficient or offset that is arithmetic on constants (``2x``, ``x + 1``,
    ``x/m²``): never a new structure. A single constant from the target's pool
    (``-1/sqrt(d)``) is a legitimate coefficient and passes."""
    if isinstance(e, ir.Mul):
        coeff = [a for a in e.args if _is_data_free(a)]
        if not coeff:
            return False
        if len(coeff) > 1:
            return True
        c = coeff[0]
        return c not in _POOL and not (isinstance(c, ir.Const) and c.value in (1, -1))
    if isinstance(e, ir.Add):
        return any(isinstance(a, ir.Const) or _rationally_scaled(a) for a in e.args)
    return False


_POOL: set[ir.SymExpr] = set()


def _safe_log(e: ir.SymExpr) -> ir.SymExpr | None:
    return mk_log(e) if positive(e) else None


def _is_data_free(e: ir.Node) -> bool:
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Reduce | ir.Card | ir.Elem | ir.Atom | ir.StateVar | ir.MonoidReduce):
            return False
        stack.extend(n.children())
    return True


def primitive_signals(sigs: Iterable[ir.SymExpr]) -> set[ir.Node]:
    """Signals that contain no other signal: what a leaf must be built from."""
    sigs = list(sigs)
    prims: set[ir.Node] = set()
    for s in sigs:
        if not any(t is not s and _descends(s, t) for t in sigs):
            prims.add(s)
    return prims


def _descends(e: ir.Node, t: ir.Node) -> bool:
    stack = list(e.children())
    while stack:
        n = stack.pop()
        if n is t:
            return True
        stack.extend(n.children())
    return False


def is_data_free(e: ir.Node) -> bool:
    return _is_data_free(e)


def complexity(e: ir.Node, opaque: set[ir.Node]) -> int:
    """Node count of ``e`` with every node in ``opaque`` (a primitive signal) counting as one."""
    if e in opaque or not e.children():
        return 1
    return 1 + sum(complexity(c, opaque) for c in e.children() if isinstance(c, ir.SymExpr))
