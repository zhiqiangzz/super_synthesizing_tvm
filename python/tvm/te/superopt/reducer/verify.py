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
"""Monoid laws for derived reducers: identity, commutativity, associativity.

Equalities are decided in the canonical IR, with a case split over the total
orders of the state variables that appear inside ``max`` (``max`` is the only
piecewise operation). What the canonical form cannot decide is tested on
random points: exactly, in rational arithmetic, when both sides are rational
functions of the state variables, and in float64 on the expression's own
domain otherwise.

The laws are statements over the reals. :func:`identity_safe` adds the one
thing they cannot see: whether the merge, as it will be evaluated in floating
point, survives meeting the identity element (``inf * 0``, ``inf - inf``).
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

from ..symbolic import ir
from ..symbolic.canonicalize import Unsupported, subst, transform, walk


def _collect(e: ir.Node, cls) -> set:
    return {n for n in walk(e) if isinstance(n, cls)}


def _max_nodes(e: ir.Node) -> list[ir.Max]:
    return [n for n in walk(e) if isinstance(n, ir.Max)]


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
            return numeric_equal(lhs, rhs)
        vars_in_max.update(m.args)
    vs = sorted(vars_in_max, key=lambda v: (v.side, v.k))
    if len(vs) > 6:
        return numeric_equal(lhs, rhs)
    for perm in itertools.permutations(range(len(vs))):
        order = {v: p for v, p in zip(vs, perm)}
        if _resolve_max(lhs, order) is not _resolve_max(rhs, order):
            return False
    return True


# ---------------------------------------------------------------------------
# evaluation on sample points
# ---------------------------------------------------------------------------
def _rational(e: ir.Node) -> bool:
    """A rational function of its variables with rational coefficients."""
    for n in walk(e):
        if isinstance(n, ir.Exp | ir.Log):
            return False
        if isinstance(n, ir.Pow) and n.exponent.denominator != 1:
            return False
        if isinstance(n, ir.Const) and isinstance(n.value, float):
            return False
    return True


def _eval_exact(e: ir.SymExpr, env: dict[ir.Node, Fraction]) -> Fraction:
    """Exact value; ``env`` holds the variables and doubles as the memo of shared sub-terms."""
    hit = env.get(e)
    if hit is not None:
        return hit
    if isinstance(e, ir.Const):
        out = e.value
    elif isinstance(e, ir.Add):
        out = sum((_eval_exact(a, env) for a in e.args), Fraction(0))
    elif isinstance(e, ir.Mul):
        out = Fraction(1)
        for a in e.args:
            out *= _eval_exact(a, env)
    elif isinstance(e, ir.Pow):
        out = _eval_exact(e.base, env) ** int(e.exponent)
    elif isinstance(e, ir.Max):
        out = max(_eval_exact(a, env) for a in e.args)
    else:
        raise Unsupported(f"cannot evaluate {type(e).__name__}")
    env[e] = out
    return out


def _eval(e: ir.SymExpr, env: dict[ir.Node, object], limits=None):
    """Evaluate with numpy scalars or arrays (one entry per sample point).

    ``env`` holds the variables and doubles as the memo: merges substituted
    into each other share most of their sub-terms. ``limits`` =
    ``(lowest, highest)`` stands for ``-inf`` / ``+inf``.
    """
    if e in env:
        return env[e]
    if isinstance(e, ir.Const):
        v = e.value
        if limits is None:
            out = float(v)
        elif isinstance(v, float):
            out = limits[0] if v < 0 else limits[1]
        else:
            out = type(limits[0])(float(v))
    elif isinstance(e, ir.Add):
        out = _eval(e.args[0], env, limits)
        for a in e.args[1:]:
            out = out + _eval(a, env, limits)
    elif isinstance(e, ir.Mul):
        out = _eval(e.args[0], env, limits)
        for a in e.args[1:]:
            out = out * _eval(a, env, limits)
    elif isinstance(e, ir.Pow):
        base = _eval(e.base, env, limits)
        p = e.exponent
        if p.denominator == 1 and abs(p.numerator) <= 4:  # as generated: repeated products
            out = base
            for _ in range(abs(p.numerator) - 1):
                out = out * base
            if p < 0:
                out = np.reciprocal(out)
        else:
            out = np.power(base, np.asarray(float(p), dtype=np.asarray(base).dtype))
    elif isinstance(e, ir.Exp):
        out = np.exp(_eval(e.arg, env, limits))
    elif isinstance(e, ir.Log):
        out = np.log(_eval(e.arg, env, limits))
    elif isinstance(e, ir.Max):
        out = _eval(e.args[0], env, limits)
        for a in e.args[1:]:
            out = np.maximum(out, _eval(a, env, limits))
    else:
        raise Unsupported(f"cannot evaluate {type(e).__name__}")
    env[e] = out
    return out


def _positive_vars(*exprs: ir.Node) -> set[ir.StateVar]:
    """State variables under a ``log`` or a fractional power: sampled positive."""
    out: set[ir.StateVar] = set()
    for e in exprs:
        for n in walk(e):
            if isinstance(n, ir.Log):
                out |= _collect(n.arg, ir.StateVar)
            elif isinstance(n, ir.Pow) and n.exponent.denominator != 1:
                out |= _collect(n.base, ir.StateVar)
    return out


def numeric_equal(lhs: ir.SymExpr, rhs: ir.SymExpr, n: int = 64, seed: int = 0) -> bool:
    """``lhs == rhs`` on ``n`` random points of their common domain."""
    rng = np.random.default_rng(seed)
    order = lambda v: (v.side, v.k)  # noqa: E731
    vs = sorted(_collect(lhs, ir.StateVar) | _collect(rhs, ir.StateVar), key=order)
    shapes = sorted(_collect(lhs, ir.ShapeSym) | _collect(rhs, ir.ShapeSym), key=lambda v: v.name)
    try:
        if _rational(lhs) and _rational(rhs):
            valid = 0
            for _ in range(n):
                env = {v: Fraction(int(rng.integers(-9, 10)), int(rng.integers(1, 8))) for v in vs}
                env.update({v: Fraction(int(rng.integers(1, 9))) for v in shapes})
                try:
                    if _eval_exact(lhs, env) != _eval_exact(rhs, env):
                        return False
                except ZeroDivisionError:
                    continue
                valid += 1
            return valid >= n // 4
        positive = _positive_vars(lhs, rhs)
        env = {v: rng.uniform(0.25, 2, n) if v in positive else rng.uniform(-2, 2, n) for v in vs}
        env.update({v: rng.integers(1, 9, n).astype(np.float64) for v in shapes})
        with np.errstate(all="ignore"):
            a = np.broadcast_to(_eval(lhs, env), (n,))
            b = np.broadcast_to(_eval(rhs, env), (n,))
    except Unsupported:
        return False
    valid = np.isfinite(a) & np.isfinite(b)  # the other points are outside one side's domain
    if not np.all(np.isclose(a[valid], b[valid], rtol=1e-8, atol=1e-10)):
        return False
    return int(valid.sum()) >= n // 4


def identity_safe(
    merge: tuple[ir.SymExpr, ...],
    identity: tuple[ir.Const, ...],
    dtype: str = "float32",
    n: int = 16,
    seed: int = 0,
) -> bool:
    """``merge(a, e) = merge(e, a) = a`` in ``dtype`` arithmetic, as the code evaluates it.

    ``merge`` is taken as written (a printed merge is not re-canonicalised) and
    ``-inf``/``+inf`` are the dtype's lowest/highest finite values, as in the
    generated ``comm_reducer``. A merge that subtracts two infinities or scales
    one by zero when a side is still empty turns the whole reduction into NaN.
    """
    k = len(merge)
    ftype = np.dtype(dtype).type
    info = np.finfo(dtype)
    limits = (ftype(info.min), ftype(info.max))
    rng = np.random.default_rng(seed)
    ident = [
        (limits[0] if c.value < 0 else limits[1]) if ir.is_inf(c) else ftype(float(c.value))
        for c in identity
    ]
    shapes = sorted(set().union(*[_collect(m, ir.ShapeSym) for m in merge]), key=lambda v: v.name)
    try:
        for _ in range(n):
            sample = [ftype(rng.uniform(0.5, 2)) for _ in range(k)]
            extents = {v: ftype(rng.integers(1, 9)) for v in shapes}
            for mine, other in (("a", "b"), ("b", "a")):
                env: dict[ir.Node, object] = dict(extents)
                for i in range(k):
                    env[ir.state_var(mine, i)] = sample[i]
                    env[ir.state_var(other, i)] = ident[i]
                with np.errstate(all="ignore"):
                    got = [_eval(m, env, limits) for m in merge]
                for i in range(k):
                    if not np.isfinite(got[i]):
                        return False
                    if not np.isclose(got[i], sample[i], rtol=1e-4, atol=1e-6):
                        return False
    except Unsupported:
        return False
    return True


_RANK = {"canonical": 0, "case-split": 1, "numeric": 2}
_SYMMETRY: dict[tuple, str | None] = {}  # merge -> how commutativity and associativity hold


def identity_law(merge: tuple[ir.SymExpr, ...], identity: tuple[ir.Const, ...]) -> str | None:
    """``M(a, e) == a``: the proof method, or ``None``."""
    n = len(merge)
    method = "canonical"
    for k in range(n):
        got = subst(merge[k], {ir.state_var("b", i): identity[i] for i in range(n)})
        if got is not ir.state_var("a", k):
            if not equal_modulo_max(got, ir.state_var("a", k)):
                return None
            method = "case-split"
    return method


def symmetry_laws(merge: tuple[ir.SymExpr, ...]) -> str | None:
    """Commutativity and associativity of ``merge``: the proof method, or ``None``.

    Neither depends on the identity element, and associativity is by far the
    most expensive of the three laws: the answer is kept per merge.
    """
    merge = tuple(merge)
    if merge not in _SYMMETRY:
        try:
            _SYMMETRY[merge] = _symmetry_laws(merge)
        except Unsupported:
            _SYMMETRY[merge] = None
    return _SYMMETRY[merge]


def _symmetry_laws(merge: tuple[ir.SymExpr, ...]) -> str | None:
    n = len(merge)
    a = [ir.state_var("a", k) for k in range(n)]
    b = [ir.state_var("b", k) for k in range(n)]
    c = [ir.state_var("c", k) for k in range(n)]
    method = "canonical"
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
        subst(merge[i], {a[j]: b[j] for j in range(n)} | {b[j]: c[j] for j in range(n)})
        for i in range(n)
    ]
    for k in range(n):
        lhs = subst(merge[k], {a[i]: merge[i] for i in range(n)} | to_c)
        rhs = subst(merge[k], {b[i]: m_bc[i] for i in range(n)})
        if lhs is not rhs:
            if not equal_modulo_max(lhs, rhs):
                if numeric_equal(lhs, rhs):
                    method = "numeric"
                    continue
                return None
            if method == "canonical":
                method = "case-split"
    return method


def check_laws(merge: tuple[ir.SymExpr, ...], identity: tuple[ir.Const, ...]) -> str | None:
    """Return the proof method (``"canonical"``/``"case-split"``/``"numeric"``) or ``None``."""
    first = identity_law(merge, identity)
    if first is None:
        return None
    second = symmetry_laws(merge)
    if second is None:
        return None
    return max(first, second, key=_RANK.__getitem__)
