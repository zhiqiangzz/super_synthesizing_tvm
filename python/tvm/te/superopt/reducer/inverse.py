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
"""Bounded algebraic inverse synthesis with solver-certified branches.

SymPy proposes roots after opening scalar maxima. Z3 checks each proposed
inverse on reachable reduction states, including its domain. A final whole
inverse proof checks that the final branch is valid on every uncovered state.
Neither symbolic solve failure nor a solver timeout proves impossibility.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import product

import sympy

from ..symbolic import ir, sympy_bridge
from ..symbolic.canonicalize import (
    Unsupported,
    mk_add,
    mk_max,
    mk_mul,
    mk_pow,
    mk_sub,
    subst,
)
from . import smt


def _from_sympy(e, symbols):
    if e in symbols:
        return symbols[e]
    if isinstance(e, sympy.Rational):
        return ir.const(Fraction(int(e.p), int(e.q)))
    if isinstance(e, sympy.Add):
        return mk_add(*(_from_sympy(a, symbols) for a in e.args))
    if isinstance(e, sympy.Mul):
        return mk_mul(*(_from_sympy(a, symbols) for a in e.args))
    if isinstance(e, sympy.Pow) and isinstance(e.exp, sympy.Rational):
        if e.exp.q not in (1, 2):
            raise Unsupported("inverse synthesis supports rational functions and square roots")
        return mk_pow(_from_sympy(e.base, symbols), Fraction(int(e.exp.p), int(e.exp.q)))
    raise Unsupported("inverse contains an unsolved atom or unsupported function")


@lru_cache(maxsize=1024)
def solve(states, domain):
    from .derive import atoms_in

    if not smt.available():
        return None
    atoms = sorted({a for s in states for a in atoms_in(s, domain)})
    # This search handles fully determined algebraic systems, not parameterized
    # families or dependent coordinates needing additional reachable equalities.
    if len(states) < len(atoms):
        return None
    maxima = set()
    pending = list(states)
    while pending:
        e = pending.pop()
        if isinstance(e, ir.Reduce | ir.Card):
            continue
        if isinstance(e, ir.Max):
            maxima.add(e)
        pending.extend(e.children())
    if not maxima:
        return None
    maxima = sorted(maxima)
    cases = 1
    for m in maxima:
        cases *= len(m.args)
    if cases > 16:
        return None
    symbols = {}
    unknowns = [sympy_bridge.to_sympy(a, symbols) for a in atoms]
    slots = [sympy_bridge.to_sympy(ir.state_var("a", k), symbols) for k in range(len(states))]
    reverse = {v: k for k, v in symbols.items() if k not in atoms}
    accepted = []
    try:
        with sympy_bridge._limit(5):
            for choices in product(*(m.args for m in maxima)):
                branch = dict(zip(maxima, choices))
                equations = [
                    sympy_bridge.to_sympy(subst(s, branch), symbols) - v
                    for s, v in zip(states, slots)
                ]
                guards = [mk_sub(a, b) for m, a in branch.items() for b in m.args if a is not b]
                for answer in sympy.solve(equations, unknowns, dict=True):
                    if any(a not in answer for a in unknowns):
                        continue
                    try:
                        sol = {a: _from_sympy(answer[v], reverse) for a, v in zip(atoms, unknowns)}
                        tests = tuple(subst(g, sol) for g in guards)
                    except Unsupported:
                        continue
                    if smt.guarded_inverse(states, sol, tests):
                        accepted.append((sol, tests))
    except (sympy_bridge._TimeUp, Unsupported, NotImplementedError):
        return None
    if not accepted:
        return None
    # IfZero(max(-g, 0), yes, no) is a lazy test of g >= 0, reusing the
    # existing conditional IR and code generation, including definedness.
    result = dict(accepted[-1][0])
    for sol, tests in reversed(accepted[:-1]):
        for a in atoms:
            value = sol[a]
            for g in reversed(tests):
                value = ir.if_zero(mk_max(mk_mul(ir.const(-1), g), ir.ZERO), value, result[a])
            result[a] = value
    env = {ir.state_var("a", k): s for k, s in enumerate(states)}
    if all(smt.equal(ir.raw_subst(value, env), atom) for atom, value in result.items()):
        return result
    return None
