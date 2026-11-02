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
"""Counterexample-guided synthesis of supplementary state coordinates.

The grammar is C_j = sum_i c_ji * B_i, with bounded integer coefficients.
B_i are the verification atoms; there is no extrema-pair or operator-specific
candidate list. Z3 solves the coefficient sketch. Recovery collisions refine
that sketch; a separate constructive inverse/domain/merge checker accepts it.
The finite grammar, timeout and rejected proof attempts never prove global
impossibility or authorize replacing an original state.
"""

from dataclasses import dataclass
from itertools import pairwise
from time import monotonic

from ..symbolic import ir
from ..symbolic.canonicalize import mk_add, mk_mul, walk
from . import smt


@dataclass(frozen=True)
class Result:
    status: str
    definitions: tuple = ()
    payload: object = None
    candidates: int = 0
    counterexamples: int = 0
    detail: str = ""


def synthesize(
    original,
    basis,
    domain,
    max_extra,
    verify,
    *,
    coefficient_bound=2,
    max_candidates=64,
    timeout=20,
):
    """Synthesize C and obtain a checked inverse D from ``verify(C)``.

    ``verify`` must return a payload only after complete recovery, source and
    inverse definedness, and merge proof. In particular, injectivity alone does
    not suffice. Failed proofs block just that coefficient assignment.
    """
    if not smt.available():
        return Result("unavailable", detail="Z3 is required for CEGIS")
    if max_extra <= 0 or not basis:
        return Result("budget", detail="no supplementary slots or verification atoms")
    z3 = smt.z3
    basis = tuple(basis)
    started = monotonic()
    attempted = collisions = 0
    samples = []
    unproved = False
    incomplete = False
    unknown = False
    transcendental = any(isinstance(n, ir.Exp | ir.Log) for e in original for n in walk(e))

    def result(status, detail, definitions=(), payload=None):
        return Result(status, definitions, payload, attempted, collisions, detail)

    # The theoretical budget is an upper bound, not a minimum-state claim.
    # Try that width first instead of spending the budget on underdetermined
    # smaller representations; still give the other widths a finite turn.
    widths = [max_extra, *range(1, max_extra)]
    for width_index, size in enumerate(widths):
        learner = z3.Solver()
        learner.set(timeout=smt.TIMEOUT_MS)
        coefficients = [[z3.Int(f"c_{j}_{i}") for i in range(len(basis))] for j in range(size)]
        weights = []
        codes = []
        for row in coefficients:
            learner.add(*[z3.And(c >= -coefficient_bound, c <= coefficient_bound) for c in row])
            learner.add(z3.Or(*[c != 0 for c in row]))
            # Global sign and slot permutations do not affect recovery.
            for i, c in enumerate(row):
                learner.add(z3.Implies(z3.And(*[a == 0 for a in row[:i]]), c >= 0))
            weights.extend(z3.If(c >= 0, c, -c) for c in row)
            codes.append(z3.Sum(*[c * (2 * coefficient_bound + 1) ** i for i, c in enumerate(row)]))
        learner.add(*[a < b for a, b in pairwise(codes)])
        cost = z3.Sum(*weights)
        complexity = size
        max_complexity = size * len(basis) * coefficient_bound

        def distinguish(delta):
            return z3.Or(
                *[z3.Sum(*[c * d for c, d in zip(row, delta)]) != 0 for row in coefficients]
            )

        for delta in samples:
            learner.add(distinguish(delta))
        size_attempts = 0
        while True:
            # Give every admissible state count a share of the finite budget;
            # difficult smaller tuples must not starve a larger valid one.
            if width_index < len(widths) - 1 and size_attempts >= max(
                1, max_candidates // max_extra
            ):
                incomplete = True
                break
            if attempted >= max_candidates or monotonic() - started >= timeout:
                return result("limit", "CEGIS resource limit; preservation is not disproven")
            status = learner.check(cost <= complexity)
            if status == z3.unknown:
                unknown = True
                break
            if status == z3.unsat:
                if complexity < max_complexity:
                    complexity += 1
                    continue
                break
            model = learner.model()
            values = [[model.eval(c).as_long() for c in row] for row in coefficients]
            definitions = tuple(
                mk_add(*[mk_mul(ir.const(c), b) for c, b in zip(row, basis)]) for row in values
            )
            attempted += 1
            size_attempts += 1
            if transcendental:
                # Opaque SMT functions can produce spurious collisions despite
                # an inverse proved using exp/log identities. Check that proof
                # before using the overapproximation to constrain the learner.
                payload = verify(definitions)
                if payload is not None:
                    return result(
                        "proved",
                        "C and a safe inverse passed independent checks",
                        definitions,
                        payload,
                    )
            status, delta = smt.recovery_collision((*original, *definitions), basis, domain, basis)
            if status == "sat":
                collisions += 1
                samples.append(delta)
                learner.add(distinguish(delta))
            else:
                # UNKNOWN is never accepted as a proof. The independent checker
                # may still prove recovery constructively (e.g. exp identities).
                payload = verify(definitions)
                if payload is not None:
                    return result(
                        "proved",
                        "C and a safe inverse passed independent checks",
                        definitions,
                        payload,
                    )
                unproved = True
            # Also block unknown/unsupported/nonconstructive candidates. Never
            # infer that their representation is impossible from proof failure.
            learner.add(
                z3.Or(
                    *[c != v for row, vals in zip(coefficients, values) for c, v in zip(row, vals)]
                )
            )
    return result(
        "unknown" if unknown else "limit" if incomplete else "exhausted",
        "bounded search ended under abstract collisions"
        + (" and unproved candidates" if unproved else "")
        + "; not an impossibility proof",
    )
