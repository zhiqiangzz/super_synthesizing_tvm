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
"""Equalities between derived expressions, and the floating-point check of a merge.

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

The monoid laws of a derived reducer are not checked here one by one: they
follow from how the merge is derived (see :mod:`derive`). What the laws over
the reals cannot see is checked by :func:`identity_safe`: whether the merge,
evaluated in floating point as the generated code evaluates it, survives
meeting the identity element (``inf * 0``, ``inf - inf``). That is a check of
the final program and the one place here that evaluates numbers.
"""

from __future__ import annotations

import numpy as np

from ..symbolic import ir, sympy_bridge
from ..symbolic.canonicalize import Unsupported, recanonicalize, subst, walk

MAX_CASES = 64  # cases a proof may split into
_RANK = {"canonical": 0, "case-split": 1, "sympy": 2}


def worst(*methods: str) -> str:
    """The weakest of several proof methods (the one a combined proof rests on)."""
    return max(methods, key=_RANK.__getitem__)


def _collect(e: ir.Node, cls) -> set:
    return {n for n in walk(e) if isinstance(n, cls)}


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


# ---------------------------------------------------------------------------
# the merge in floating point, against the identity element
# ---------------------------------------------------------------------------
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


__all__ = ["identity_safe", "prove_equal", "proves", "worst"]
