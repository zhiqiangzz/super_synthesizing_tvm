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
"""Second-line algebraic prover: sympy over atom-abstracted expressions.

Tensor elements, reductions and shape symbols become opaque sympy symbols
(reductions are ``positive`` when their bodies are); the scalar algebra in
between is handed to ``sympy.simplify``. This catches identities the
canonicaliser's fixed rule set misses (e.g. ``(x + y)^2`` expansions).
"""

from __future__ import annotations

from . import ir
from .canonicalize import Unsupported, positive


def to_sympy(e: ir.SymExpr, atoms: dict[ir.Node, object] | None = None):
    import sympy

    if atoms is None:
        atoms = {}

    def sym_for(n: ir.Node, pos: bool):
        s = atoms.get(n)
        if s is None:
            s = sympy.Symbol(f"v{len(atoms)}", real=True, positive=pos or None)
            atoms[n] = s
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
            return sympy.Pow(
                rec(n.base), sympy.Rational(n.exponent.numerator, n.exponent.denominator)
            )
        if isinstance(n, ir.Exp):
            return sympy.exp(rec(n.arg))
        if isinstance(n, ir.Log):
            return sympy.log(rec(n.arg))
        if isinstance(n, ir.Max):
            return sympy.Max(*[rec(a) for a in n.args])
        if isinstance(n, ir.ShapeSym | ir.Card):
            return sym_for(n, True)
        if isinstance(n, ir.Elem | ir.Reduce | ir.MonoidReduce | ir.Atom | ir.StateVar):
            return sym_for(n, isinstance(n, ir.Reduce | ir.Card) and positive(n))
        raise Unsupported(f"sympy: {type(n).__name__}")

    return rec(e)


def prove_equal(lhs: ir.SymExpr, rhs: ir.SymExpr) -> bool:
    """True if sympy can show ``lhs - rhs == 0`` (atoms shared between both sides)."""
    if lhs is rhs:
        return True
    try:
        import sympy
    except ImportError:  # pragma: no cover
        return False
    atoms: dict[ir.Node, object] = {}
    try:
        a = to_sympy(lhs, atoms)
        b = to_sympy(rhs, atoms)
    except Unsupported:
        return False
    diff = sympy.simplify(a - b)
    return diff == 0
