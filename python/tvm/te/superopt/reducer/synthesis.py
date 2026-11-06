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
"""Native cvc5 SyGuS search over rebase features and verification coordinates.

The grammar combines features with addition/subtraction. Preferred features may
be nonlinear functions of B, but may not introduce new verification atoms or
undefined expressions. Search first uses preferred features alone, then mixes
them with B. cvc5 owns enumeration and counterexample refinement; this module
only translates the recovery specification and independently checks solutions.
An injective coordinate map is not accepted without a constructive safe inverse
and a proved merge. Exhaustion/unknown never proves global impossibility.
"""

from dataclasses import dataclass
from fractions import Fraction
from time import monotonic

from ..symbolic import ir
from ..symbolic.canonicalize import Unsupported, mk_add, mk_mul, subst_domain, walk
from . import smt

try:
    import cvc5
except ImportError:
    cvc5 = None


@dataclass(frozen=True)
class Proposal:
    """Keep feature recipes so the caller can retain their centered structure."""

    features: tuple
    weights: tuple

    @property
    def definitions(self):
        return tuple(
            mk_add(*(mk_mul(ir.const(c), e) for c, e in zip(row, self.features) if c))
            for row in self.weights
        )


@dataclass(frozen=True)
class Result:
    status: str
    definitions: tuple = ()
    payload: object = None
    candidates: int = 0
    phase: str = ""
    detail: str = ""


class _Translator:
    """Translate the existing real-state abstraction, not a second set of laws.

    Opaque function applications become universally quantified values, with
    congruence constraints. This is conservative and keeps SyGuS first-order.
    """

    def __init__(self, solver):
        self.solver = solver
        self.tm = solver.getTermManager()
        self.memo = {}
        self.opaque = {}
        self.constraints = []

    def value(self, e):
        if e in self.memo:
            return self.memo[e]
        z3, K = smt.z3, cvc5.Kind
        tm = self.tm
        if z3.is_true(e) or z3.is_false(e):
            out = tm.mkBoolean(z3.is_true(e))
        elif z3.is_rational_value(e):
            out = tm.mkReal(e.numerator_as_long(), e.denominator_as_long())
        elif z3.is_int_value(e):
            out = tm.mkReal(e.as_long())
        elif e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            out = self.solver.declareSygusVar(f"v{len(self.memo)}", tm.getRealSort())
            if e.num_args():
                args = tuple(self.value(a) for a in e.children())
                for prev, val in self.opaque.get(e.decl(), []):
                    equal = self.all(tm.mkTerm(K.EQUAL, a, b) for a, b in zip(args, prev))
                    self.constraints.append(
                        tm.mkTerm(K.IMPLIES, equal, tm.mkTerm(K.EQUAL, out, val))
                    )
                self.opaque.setdefault(e.decl(), []).append((args, out))
        else:
            kinds = {
                z3.Z3_OP_ADD: K.ADD,
                z3.Z3_OP_MUL: K.MULT,
                z3.Z3_OP_SUB: K.SUB,
                z3.Z3_OP_UMINUS: K.NEG,
                z3.Z3_OP_DIV: K.DIVISION,
                z3.Z3_OP_EQ: K.EQUAL,
                z3.Z3_OP_DISTINCT: K.DISTINCT,
                z3.Z3_OP_LE: K.LEQ,
                z3.Z3_OP_GE: K.GEQ,
                z3.Z3_OP_LT: K.LT,
                z3.Z3_OP_GT: K.GT,
                z3.Z3_OP_AND: K.AND,
                z3.Z3_OP_OR: K.OR,
                z3.Z3_OP_NOT: K.NOT,
                z3.Z3_OP_IMPLIES: K.IMPLIES,
                z3.Z3_OP_ITE: K.ITE,
            }
            kind = kinds.get(e.decl().kind())
            if kind is None:
                raise Unsupported(f"SyGuS translation of {e.decl()}")
            args = [self.value(a) for a in e.children()]
            if len(args) == 1 and kind in (K.ADD, K.MULT, K.AND, K.OR):
                out = args[0]
            elif not args and kind in (K.AND, K.OR):
                out = tm.mkBoolean(kind == K.AND)
            else:
                out = tm.mkTerm(kind, *args)
        self.memo[e] = out
        return out

    def all(self, values):
        values = list(values)
        if not values:
            return self.tm.mkBoolean(True)
        return values[0] if len(values) == 1 else self.tm.mkTerm(cvc5.Kind.AND, *values)


def _weights(term, variables):
    """Read cvc5's linear feature expression, including its simplifications."""
    K = cvc5.Kind
    if term in variables:
        return tuple(Fraction(int(term == v)) for v in variables)
    if term.isRealValue():
        if term.getRealValue() != 0:
            raise Unsupported("SyGuS returned a nonzero constant coordinate")
        return (Fraction(0),) * len(variables)
    kind = term.getKind()
    if kind == K.NEG:
        return tuple(-c for c in _weights(term[0], variables))
    if kind in (K.ADD, K.SUB):
        rows = [_weights(t, variables) for t in term]
        return tuple(
            sum(row[i] * (-1 if kind == K.SUB and j else 1) for j, row in enumerate(rows))
            for i in range(len(variables))
        )
    if kind == K.MULT:
        scalar = Fraction(1)
        row = None
        for t in term:
            if t.isRealValue():
                scalar *= t.getRealValue()
            elif row is None:
                row = _weights(t, variables)
            else:
                raise Unsupported("SyGuS returned a nonlinear feature combination")
        if row is not None:
            return tuple(scalar * c for c in row)
    raise Unsupported(f"SyGuS solution outside the feature grammar: {term}")


def _grammar(solver, features, max_terms, tuple_width=None):
    K = cvc5.Kind
    tm = solver.getTermManager()
    real = tm.getRealSort()
    variables = tuple(tm.mkVar(real, f"e{i}") for i in range(len(features)))
    # A finite grammar limits expression size, not polynomial degree in B.
    # A nonlinear rebase expression is a single feature terminal.
    levels = [tm.mkVar(real, f"E{i}") for i in range(max_terms)]
    nonterminals = list(reversed(levels))
    if tuple_width is not None:
        root = tm.mkVar(tm.mkTupleSort(*([real] * tuple_width)), "Coordinates")
        nonterminals.insert(0, root)
    grammar = solver.mkGrammar(list(variables), nonterminals)
    if tuple_width is not None:
        grammar.addRule(root, tm.mkTuple([levels[-1]] * tuple_width))
    grammar.addRules(levels[0], list(variables))
    for i in range(1, max_terms):
        grammar.addRules(
            levels[i],
            [
                levels[i - 1],
                tm.mkTerm(K.ADD, levels[i - 1], levels[0]),
                tm.mkTerm(K.SUB, levels[i - 1], levels[0]),
            ],
        )
    return grammar, variables


def _problem(original, targets, features, domain, width, max_terms):
    K = cvc5.Kind
    solver = cvc5.Solver()
    solver.setOption("sygus", "true")
    solver.setOption("incremental", "true")
    solver.setOption("full-sygus-verify", "true")
    solver.setLogic("ALL")
    tm = solver.getTermManager()
    real = tm.getRealSort()
    grammar, variables = _grammar(solver, features, max_terms)
    functions = [solver.synthFun(f"C{i}", list(variables), real, grammar) for i in range(width)]
    enc, collision = smt.recovery_query(original, targets, domain, features=features)
    other = ir.dsym(domain.axis, "recovery_other")
    values = [
        [enc.value(subst_domain(e, {domain: side}))[0] for e in features]
        for side in (domain, other)
    ]
    tr = _Translator(solver)
    collision = tr.value(collision)
    args = [[tr.value(v) for v in side] for side in values]
    facts = [tr.value(c) for c in enc.constraints]
    same = [
        tm.mkTerm(K.EQUAL, tm.mkTerm(K.APPLY_UF, f, *args[0]), tm.mkTerm(K.APPLY_UF, f, *args[1]))
        for f in functions
    ]
    # No two legal abstract B states may collide in both A0 and all C slots.
    solver.addSygusConstraint(tm.mkTerm(K.NOT, tr.all([*facts, *tr.constraints, collision, *same])))
    return solver, functions, variables


def _enumerator(features, width, max_terms):
    """cvc5's native grammar enumeration when its quantified verifier is unknown.

    This supplies proposals only; the existing independent checker must still
    prove every accepted inverse/domain/merge. No sampled acceptance or custom
    counterexample refinement is introduced by this fallback.
    """
    solver = cvc5.Solver()
    solver.setOption("sygus", "true")
    solver.setOption("incremental", "true")
    solver.setLogic("ALL")
    grammar, variables = _grammar(solver, features, max_terms, tuple_width=width)
    return solver, grammar, variables


def _over_basis(expr, basis):
    if expr in basis or isinstance(expr, ir.Const):
        return True
    children = tuple(expr.children())
    return bool(children) and all(_over_basis(c, basis) for c in children)


def parameters_of(original):
    """Problem extents already read by A0, shared by both copies of a segment."""
    return {n for e in original for n in walk(e) if isinstance(n, ir.ShapeSym)}


def valid_feature(expr, basis, parameters=()):
    """A defined function of B and the program's fixed, shared parameters."""
    return _over_basis(expr, set(basis) | set(parameters)) and smt.defined(expr)


def synthesize(
    original,
    basis,
    domain,
    max_extra,
    verify,
    *,
    preferred=(),
    max_candidates=256,
    timeout=20,
    max_terms=3,
    mode="production",
    observe=None,
):
    """Ask native SyGuS for coordinates; ``verify(Proposal)`` proves the reducer.

    Preferred-only and mixed grammars give rebase features explicit priority.
    ``checkSynthNext`` advances cvc5's search after a solution lacks an independent
    constructive proof. There is no application-owned counterexample/learner loop.
    """
    if cvc5 is None or not smt.available():
        return Result("unavailable", detail="state synthesis requires cvc5 and z3-solver")
    if max_extra <= 0 or not basis:
        return Result("budget", detail="no supplementary slots or verification atoms")
    if max_terms < 1:
        return Result("budget", detail="the feature grammar needs at least one term")
    if mode not in ("production", "common", "reference_only", "enumeration", "single_term"):
        raise ValueError(f"unknown synthesis mode: {mode}")
    if mode == "single_term":
        max_terms = 1
    started = monotonic()

    def emit(kind, **data):
        if observe is not None:
            observe(kind, **data)

    basis = tuple(basis)
    preferred = tuple(dict.fromkeys(preferred))
    if any(not valid_feature(e, basis, parameters_of(original)) for e in preferred):
        return Result("unsupported", detail="a preferred feature is not a defined function of B")
    mixed = tuple(dict.fromkeys((*preferred, *basis)))
    phases = [("rebase", preferred)] if preferred else []
    if mixed != preferred:
        phases.append(("mixed", mixed))
    if mode != "production":
        phases = [("mixed", basis if mode == "reference_only" else mixed)]
    widths = [max_extra, *range(1, max_extra)]
    attempted = 0
    unknown = False
    last_phase = ""
    # Reserve time for later grammars/widths instead of starving them behind an
    # impossible preferred grammar. This is scheduling, not a minimality claim.
    slice_seconds = timeout / max(1, len(phases) * len(widths))
    emit(
        "grammar",
        mode=mode,
        max_terms=max_terms,
        widths=widths,
        phases=[(name, [str(e) for e in fs]) for name, fs in phases],
        timeout=timeout,
        max_candidates=max_candidates,
        slice_seconds=slice_seconds,
    )

    def check(proposal, phase, width):
        start = monotonic()
        emit(
            "proposal",
            proposal_id=attempted,
            phase=phase,
            width=width,
            features=[str(e) for e in proposal.features],
            weights=[[str(c) for c in row] for row in proposal.weights],
        )
        if start >= started + timeout:
            emit(
                "proposal_result",
                proposal_id=attempted,
                status="resource_limit",
                checker_called=False,
                seconds=0.0,
            )
            return None
        payload = verify(proposal)
        end = monotonic()
        emit(
            "proposal_result",
            proposal_id=attempted,
            status="passed" if payload is not None else "not_accepted",
            checker_called=True,
            seconds=end - start,
            deadline_overrun=max(0, end - started - timeout),
        )
        return payload

    try:
        for phase, features in phases:
            last_phase = phase
            for width in widths:
                if attempted >= max_candidates or monotonic() - started >= timeout:
                    return Result(
                        "limit", candidates=attempted, phase=phase, detail="SyGuS resource limit"
                    )
                if mode != "enumeration":
                    solver, functions, variables = _problem(
                        original, basis, features, domain, width, max_terms
                    )
                end = min(started + timeout, monotonic() + slice_seconds)
                first = True
                fallback = mode == "enumeration"
                emit("width", phase=phase, width=width, seconds=max(0, end - monotonic()))
                while mode != "enumeration" and attempted < max_candidates and monotonic() < end:
                    query_ms = max(1, min(1500, int((end - monotonic()) * 1000)))
                    solver.setOption("tlimit-per", str(query_ms))
                    query_start = monotonic()
                    result = solver.checkSynth() if first else solver.checkSynthNext()
                    emit(
                        "solver_query",
                        phase=phase,
                        width=width,
                        query="semantic",
                        next=not first,
                        limit_ms=query_ms,
                        seconds=monotonic() - query_start,
                        status="passed"
                        if result.hasSolution()
                        else "unknown"
                        if result.isUnknown()
                        else "exhausted",
                    )
                    first = False
                    if not result.hasSolution():
                        fallback = result.isUnknown()
                        unknown |= fallback
                        break
                    attempted += 1
                    rows = []
                    for function in functions:
                        solution = solver.getSynthSolution(function)
                        rows.append(_weights(solution[1], variables))
                    proposal = Proposal(features, tuple(rows))
                    payload = check(proposal, phase, width)
                    if payload is not None:
                        return Result(
                            "proved",
                            proposal.definitions,
                            payload,
                            attempted,
                            phase,
                            "cvc5 SyGuS solution passed independent inverse/domain/merge checks",
                        )
                else:
                    unknown |= mode != "enumeration"
                if (
                    fallback
                    and mode in ("production", "enumeration")
                    and attempted < max_candidates
                    and monotonic() < end
                ):
                    solver, grammar, variables = _enumerator(features, width, max_terms)
                    first = True
                    while attempted < max_candidates and monotonic() < end:
                        query_ms = max(1, int((end - monotonic()) * 1000))
                        if mode != "production":
                            query_ms = min(1500, query_ms)
                        solver.setOption("tlimit-per", str(query_ms))
                        query_start = monotonic()
                        term = (
                            solver.findSynth(cvc5.FindSynthTarget.ENUM, grammar)
                            if first
                            else solver.findSynthNext()
                        )
                        emit(
                            "solver_query",
                            phase=phase,
                            width=width,
                            query="enumeration",
                            next=not first,
                            limit_ms=query_ms,
                            seconds=monotonic() - query_start,
                            status="unknown" if term.isNull() else "passed",
                        )
                        first = False
                        if term.isNull():
                            unknown = True
                            break
                        attempted += 1
                        proposal = Proposal(
                            features, tuple(_weights(t, variables) for t in list(term)[1:])
                        )
                        payload = check(proposal, phase + "-enumeration", width)
                        if payload is not None:
                            return Result(
                                "proved",
                                proposal.definitions,
                                payload,
                                attempted,
                                phase + "-enumeration",
                                "cvc5 proposal passed independent inverse/domain/merge checks",
                            )
    except Unsupported as err:
        return Result("unsupported", candidates=attempted, phase=last_phase, detail=str(err))
    except RuntimeError as err:
        return Result("unknown", candidates=attempted, phase=last_phase, detail=f"cvc5: {err}")
    return Result(
        "limit"
        if attempted >= max_candidates or monotonic() - started >= timeout
        else "unknown"
        if unknown
        else "exhausted",
        candidates=attempted,
        phase=last_phase,
        detail="no independently verified SyGuS solution; not an impossibility proof",
    )
