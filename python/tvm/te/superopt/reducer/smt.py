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
"""Real-domain proof obligations for state recovery, discharged by Z3.

Reductions are abstract values, not a bounded unrolling of the input. Generic
finite-reduction laws lift solver-proven pointwise facts to every nonempty segment:
nonnegative summands have a nonnegative sum; if a nonnegative sum is zero,
all its summands are zero; a pointwise upper bound bounds the sum by the count
times its maximum. The solver, rather than an operator template, decides
which body inequalities and zero implications hold.

The abstraction overapproximates reachable states. UNSAT proves an obligation;
SAT is only a possible counterexample, and UNKNOWN is never a disproof.
Roots use polynomial constraints. Transcendentals are conservative opaque
values (exp is positive); their identities remain the symbolic/CAS verifier's
job. This module proves real identities, not IEEE floating-point accuracy.
"""

from __future__ import annotations

from functools import lru_cache

from ..symbolic import ir
from ..symbolic.canonicalize import Unsupported, mk_card, mk_mul, mk_sub, walk

try:
    import z3
except ImportError:
    z3 = None

TIMEOUT_MS = 1500


def available() -> bool:
    return z3 is not None


class Encoding:
    """Expressions and their domains; division is never implicitly assumed valid."""

    def __init__(self):
        if z3 is None:
            raise Unsupported("state proofs require the superopt extra (z3-solver)")
        self.memo = {}
        self.constraints = []

    def value(self, e):
        if e in self.memo:
            return self.memo[e]
        out = self._value(e)
        self.memo[e] = out
        return out

    def _value(self, e):
        true = z3.BoolVal(True)
        if isinstance(e, ir.Const):
            if isinstance(e.value, float):
                raise Unsupported("infinity is not a real")
            return z3.RealVal(str(e.value)), true
        if isinstance(e, ir.Elem | ir.StateVar | ir.Atom | ir.ShapeSym | ir.Card | ir.Reduce):
            v = z3.Real(f"n{e.uid}")
            if isinstance(e, ir.Card | ir.ShapeSym):
                self.constraints.append(v > 0)
            return v, true
        if isinstance(e, ir.IfZero):
            t, dt = self.value(e.test)
            a, da = self.value(e.zero)
            b, db = self.value(e.nonzero)
            return z3.If(t == 0, a, b), z3.And(dt, z3.If(t == 0, da, db))
        if isinstance(e, ir.Add | ir.Mul | ir.Max):
            pairs = [self.value(a) for a in e.args]
            vals = [v for v, _ in pairs]
            if isinstance(e, ir.Add):
                v = z3.Sum(vals)
            elif isinstance(e, ir.Mul):
                v = z3.RealVal(1)
                for a in vals:
                    v = v * a
            else:
                v = vals[0]
                for a in vals[1:]:
                    v = z3.If(v >= a, v, a)
            return v, z3.And(*[d for _, d in pairs])
        if isinstance(e, ir.Pow):
            b, defined = self.value(e.base)
            p = e.exponent
            if p.denominator != 1:
                if p.denominator != 2:
                    raise Unsupported("SMT encoding supports integer powers and square roots")
                root = z3.Real(f"root{e.base.uid}")
                self.constraints.append(z3.Implies(b >= 0, z3.And(root >= 0, root * root == b)))
                defined = z3.And(defined, b >= 0)
                b = root
            power = abs(p.numerator)
            v = z3.RealVal(1)
            for _ in range(power):
                v = v * b
            if p < 0:
                return 1 / v, z3.And(defined, b != 0)
            return v, defined
        if isinstance(e, ir.Exp | ir.Log):
            a, defined = self.value(e.arg)
            # Opaque functions remain congruent for equal arguments.
            fn = z3.Function(type(e).__name__, z3.RealSort(), z3.RealSort())
            v = fn(a)
            if isinstance(e, ir.Exp):
                self.constraints.append(v > 0)
            else:
                defined = z3.And(defined, a > 0)
            return v, defined
        raise Unsupported(f"SMT encoding of {type(e).__name__}")

    def check(self, condition):
        # General SMT also accepts conservative opaque exp/log functions.
        solver = z3.Solver()
        solver.set(timeout=TIMEOUT_MS)
        solver.add(*self.constraints, condition)
        return solver.check()

    def facts(self, exprs):
        reds = {n for e in exprs for n in walk(e) if isinstance(n, ir.Reduce)}
        for r in reds:
            v, _ = self.value(r)
            if body_fact(r.body, "positive"):
                self.constraints.append(v > 0)
            elif body_fact(r.body, "nonnegative"):
                self.constraints.append(v >= 0)
        sums = [r for r in reds if r.kind == "sum"]
        # Pointwise f <= g implies sum(f) <= |R| max(g). In particular,
        # applying this also to -f proves both bounds on a segment's mean.
        # These are finite-reduction laws; the body comparison is solver-proved.
        for m in (r for r in reds if r.kind == "max"):
            mv, _ = self.value(m)
            count, _ = self.value(mk_card(m.domain))
            for s in sums:
                if s.domain is not m.domain or s.level != m.level:
                    continue
                sv, _ = self.value(s)
                for sign in (1, -1):
                    if body_fact(mk_sub(m.body, mk_mul(ir.const(sign), s.body)), "nonnegative"):
                        self.constraints.append(sign * sv <= count * mv)
        for a in sums:
            if not body_fact(a.body, "nonnegative"):
                continue
            av, _ = self.value(a)
            for b in sums:
                if a is b or a.domain is not b.domain or a.level != b.level:
                    continue
                if body_zero_implies(a.body, b.body):
                    bv, _ = self.value(b)
                    self.constraints.append(z3.Implies(av == 0, bv == 0))


@lru_cache(maxsize=4096)
def body_fact(body, kind):
    if z3 is None:
        return False
    try:
        enc = Encoding()
        v, defined = enc.value(body)
        # Do not lift a fact whose pointwise domain is itself unproven.
        claim = z3.And(defined, v > 0 if kind == "positive" else v >= 0)
        return enc.check(z3.Not(claim)) == z3.unsat
    except Unsupported:
        return False


@lru_cache(maxsize=4096)
def body_zero_implies(a, b):
    if z3 is None:
        return False
    try:
        enc = Encoding()
        av, da = enc.value(a)
        bv, db = enc.value(b)
        return enc.check(z3.And(da, av == 0, z3.Not(z3.And(db, bv == 0)))) == z3.unsat
    except Unsupported:
        return False


@lru_cache(maxsize=4096)
def zero_implies(test, value):
    """Prove test == 0 => value == 0 on reachable reduction states."""
    if z3 is None:
        return False
    try:
        enc = Encoding()
        t, dt = enc.value(test)
        v, dv = enc.value(value)
        enc.facts((test, value))
        return enc.check(z3.And(dt, t == 0, z3.Not(z3.And(dv, v == 0)))) == z3.unsat
    except Unsupported:
        return False


@lru_cache(maxsize=4096)
def defined(e):
    if z3 is None:
        return False
    try:
        enc = Encoding()
        _, d = enc.value(e)
        enc.facts((e,))
        return enc.check(z3.Not(d)) == z3.unsat
    except Unsupported:
        return False


def source_defined(e):
    """Prove source domains, including bodies hidden by the reduction abstraction.

    ``defined`` treats reduction results as scalar coordinates. That is suitable
    for inverse proofs, but source validation must also establish every summand's
    domain. Checking all bodies is conservative for branch-local reductions.
    """
    return defined(e) and all(defined(n.body) for n in walk(e) if isinstance(n, ir.Reduce))


@lru_cache(maxsize=4096)
def condition_holds(operand, kind, signs=()):
    if z3 is None:
        return False
    try:
        enc = Encoding()
        v, valid = enc.value(operand)
        for node, sign in signs:
            x, _ = enc.value(node)
            if sign:
                enc.constraints.append(x > 0 if sign == 2 else x >= 0)
        enc.facts((operand,))
        claim = {"nonzero": v != 0, "positive": v > 0, "nonnegative": v >= 0}[kind]
        return enc.check(z3.Not(z3.And(valid, claim))) == z3.unsat
    except Unsupported:
        return False


@lru_cache(maxsize=4096)
def equal(lhs, rhs):
    """Equality AND domain validity; unknown and abstract counterexamples both refuse."""
    if z3 is None:
        return False
    try:
        enc = Encoding()
        a, da = enc.value(lhs)
        b, db = enc.value(rhs)
        enc.facts((lhs, rhs))
        return enc.check(z3.Not(z3.And(da, db, a == b))) == z3.unsat
    except Unsupported:
        return False


def guarded_inverse(states, solution, guards):
    """Prove an inverse on the region where all its nonnegative guards hold."""
    if z3 is None:
        return False
    env = {ir.state_var("a", k): s for k, s in enumerate(states)}
    try:
        enc = Encoding()
        conditions, claims, exprs = [], [], list(states)
        for guard in guards:
            e = ir.raw_subst(guard, env)
            g, dg = enc.value(e)
            conditions.append(z3.And(dg, g >= 0))
            exprs.append(e)
        for atom, inverse in solution.items():
            e = ir.raw_subst(inverse, env)
            v, dv = enc.value(e)
            a, da = enc.value(atom)
            claims.append(z3.And(dv, da, v == a))
            exprs.extend((e, atom))
        enc.facts(exprs)
        return enc.check(z3.And(*conditions, z3.Not(z3.And(*claims)))) == z3.unsat
    except Unsupported:
        return False


def guard_inverse(atom, expr, divisors, states):
    """Repair only zero branches certified by the solver, never by convention."""
    env = {ir.state_var("a", k): s for k, s in enumerate(states)}
    if any(isinstance(n, ir.Reduce | ir.Card) for n in walk(expr)):
        return expr
    out = expr
    for d in divisors:
        over = ir.raw_subst(d, env)
        if defined(ir.raw_pow(over, -1)):
            continue
        if zero_implies(over, atom):
            out = ir.if_zero(d, ir.ZERO, out)
    if out is expr or not equal(ir.raw_subst(out, env), atom):
        return expr
    return out


def singleton_counterexample(operand, kind, domain, tensor_sign):
    """A SAT witness on an actual one-element segment authorizes a domain repair.

    Unlike an abstract reduction-state model, these variables are input elements.
    Respect the caller's declared signs. Unsupported/unknown does not authorize
    replacing an original state.
    """
    if z3 is None:
        return False
    from .derive import singleton

    try:
        one = singleton(operand, domain)
        if any(isinstance(n, ir.Reduce | ir.MonoidReduce) for n in walk(one)):
            return False
        enc = Encoding()
        value, valid = enc.value(one)
        for n in walk(one):
            if isinstance(n, ir.Elem):
                sign = tensor_sign(n.tensor)
                v, _ = enc.value(n)
                if sign:
                    enc.constraints.append(v > 0 if sign == 2 else v >= 0)
        bad = {"nonzero": value == 0, "positive": value <= 0, "nonnegative": value < 0}[kind]
        return enc.check(z3.And(valid, bad)) == z3.sat
    except Unsupported:
        return False


def _recovery_query(states, targets, domain, state_signs=(), basis=()):
    """Build the same overapproximation for uniqueness and counterexample queries."""
    from ..symbolic.canonicalize import subst_domain

    other = ir.dsym(domain.axis, "recovery_other")

    def copy(e):
        return subst_domain(e, {domain: other})

    enc = Encoding()
    equal_states, different, valid = [], [], []
    exprs = (*states, *targets)
    for i, e in enumerate(exprs):
        a, da = enc.value(e)
        b, db = enc.value(copy(e))
        valid.extend((da, db))
        if i < len(state_signs) and state_signs[i]:
            sign = state_signs[i]
            valid.extend((a > 0, b > 0) if sign == 2 else (a >= 0, b >= 0))
        (equal_states if i < len(states) else different).append(
            a == b if i < len(states) else a != b
        )
    differences = []
    for e in basis:
        a, da = enc.value(e)
        b, db = enc.value(copy(e))
        valid.extend((da, db))
        differences.append(a - b)
    enc.facts((*exprs, *basis, *map(copy, (*exprs, *basis))))
    return enc, z3.And(*valid, *equal_states, z3.Or(*different)), differences


@lru_cache(maxsize=1024)
def recovery_status(states, targets, domain, state_signs=()):
    """Two copies: equal A and unequal B. Only UNSAT proves unique recovery.

    SAT is a collision in the abstraction, not a license to discard a state.
    Constructive inverses can still prove recovery when this abstraction is
    too weak, notably for transcendental identities.
    """
    if z3 is None:
        return "unknown"
    try:
        enc, condition, _ = _recovery_query(states, targets, domain, state_signs)
        return str(enc.check(condition))
    except Unsupported:
        return "unknown"


def recovery_collision(states, targets, domain, basis):
    """Return exact basis differences at an abstract collision for CEGIS.

    A SAT model is a search counterexample, not proof of an actual input
    collision. No approximate decimal conversion is used for algebraic values.
    The independent recovery/domain/merge checker still decides acceptance.
    """
    if z3 is None:
        return "unknown", ()
    try:
        enc, condition, differences = _recovery_query(states, targets, domain, basis=basis)
        solver = z3.Solver()
        solver.set(timeout=TIMEOUT_MS)
        solver.add(*enc.constraints, condition)
        status = solver.check()
        if status != z3.sat:
            return str(status), ()
        model = solver.model()
        return "sat", tuple(model.eval(d, model_completion=True) for d in differences)
    except Unsupported:
        return "unknown", ()
