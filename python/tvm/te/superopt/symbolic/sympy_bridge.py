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
"""Equalities the canonical rules do not normalise, decided by SymPy.

The canonical form has no common denominators: ``(c_a m_a + c_b m_b) / (c_a +
c_b)`` squared and expanded is not recognised as the same rational function
written another way. SymPy's ``cancel`` is a decision procedure for that: two
rational functions are equal exactly when the numerator of their difference,
over a common denominator, cancels to zero.

Everything that is not arithmetic becomes a symbol of its own: a tensor
element, a reduction, a state variable. Symbols are real; only a count, an
extent, and a reduction of something positive are declared positive.
``exp`` and ``log`` stay as functions, so they act as further unknowns of the
rational function (``exp(a) exp(-a)`` still cancels: SymPy combines powers of
the same base when it builds a product). ``simplify`` is not used: it is not
a decision procedure and its running time is unbounded. Every call is under a
time limit.
"""

from __future__ import annotations

import contextlib
import signal
import threading

import sympy

from . import ir
from .canonicalize import Unsupported, positive

TIME_LIMIT = 5  # seconds for one equality


def to_sympy(e: ir.SymExpr, symbols: dict[ir.Node, sympy.Symbol]):
    """``e`` as a SymPy expression; ``symbols`` maps the leaves (shared between calls)."""

    def leaf(n: ir.Node) -> sympy.Symbol:
        s = symbols.get(n)
        if s is None:
            pos = isinstance(n, ir.ShapeSym | ir.Card) or (isinstance(n, ir.Reduce) and positive(n))
            s = sympy.Symbol(f"v{len(symbols)}", real=True, positive=True if pos else None)
            symbols[n] = s
        return s

    def rec(n: ir.SymExpr):
        if isinstance(n, ir.Const):
            v = n.value
            if isinstance(v, float):
                return sympy.oo if v > 0 else -sympy.oo
            return sympy.Rational(v.numerator, v.denominator)
        if isinstance(n, ir.Add):
            return sympy.Add(*[rec(a) for a in n.args])
        if isinstance(n, ir.Mul):
            return sympy.Mul(*[rec(a) for a in n.args])
        if isinstance(n, ir.Pow):
            p = n.exponent
            return sympy.Pow(rec(n.base), sympy.Rational(p.numerator, p.denominator))
        if isinstance(n, ir.Exp):
            return sympy.exp(rec(n.arg))
        if isinstance(n, ir.Log):
            return sympy.log(rec(n.arg))
        if isinstance(n, ir.Max):
            return sympy.Max(*[rec(a) for a in n.args])
        if isinstance(
            n, ir.Elem | ir.Reduce | ir.MonoidReduce | ir.Atom | ir.StateVar | ir.Card | ir.ShapeSym
        ):
            return leaf(n)
        raise Unsupported(f"no SymPy form for {type(n).__name__}")

    return rec(e)


class _TimeUp(Exception):
    pass


@contextlib.contextmanager
def _limit(seconds: int):
    """Raise :class:`_TimeUp` after ``seconds`` (only on the main thread, where
    signals are delivered; elsewhere there is no limit)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(signum, frame):
        raise _TimeUp

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def equal(lhs: ir.SymExpr, rhs: ir.SymExpr, seconds: int = TIME_LIMIT) -> bool:
    """Is ``lhs - rhs`` zero as a rational function of its leaves (and of the
    ``exp`` and ``log`` in it)? ``False`` also when it cannot be decided in time."""
    symbols: dict[ir.Node, sympy.Symbol] = {}
    try:
        a = to_sympy(lhs, symbols)
        b = to_sympy(rhs, symbols)
    except Unsupported:
        return False
    try:
        with _limit(seconds):
            diff = sympy.cancel(sympy.together(a - b))
    except _TimeUp:
        return False
    return diff == 0


def positive_ratio(lhs: ir.SymExpr, rhs: ir.SymExpr, seconds: int = TIME_LIMIT) -> bool:
    """Is ``lhs / rhs``, cancelled to lowest terms, positive for every value of its
    leaves (by SymPy's assumptions: counts and extents are positive)?

    ``Σ x² - (Σ x)² / n`` over ``(Σ x² - (Σ x)² / n) / n`` is ``n``: the two are
    zero together.
    """
    symbols: dict[ir.Node, sympy.Symbol] = {}
    try:
        a = to_sympy(lhs, symbols)
        b = to_sympy(rhs, symbols)
    except Unsupported:
        return False
    try:
        with _limit(seconds):
            q = sympy.cancel(sympy.together(a / b))
            return bool(q.is_positive)
    except _TimeUp:
        return False


__all__ = ["equal", "positive_ratio", "to_sympy"]
