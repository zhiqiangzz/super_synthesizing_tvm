<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Preserving reduction states with verified recovery

The verification coordinates are **B**: context-free reductions identified by
algebraic expansion. The maintained tuple is **A**: original reductions plus
supplements. The default state budget is the theoretical construction's count;
`max_states` can tighten it. This is a budget, not a theorem that every original
coordinate system admits an equally small representation.

## Selection and proof

1. Start with the original reductions. Check context-dependent domains, including
   divisors still inside a reduction. Only a solver-demonstrated invalid domain
   on a legal singleton authorizes replacing an original. Declared input signs
   are included in that counterexample query.
2. Enumerate supplement sets by increasing cardinality, within the budget.
   The finite pool contains basic verification reductions, counts, and rebase
   coefficients/components. There is a limit of 256 candidate sets. Candidate
   ordering is a search heuristic, never a substitute for a proof.
3. Construct a complete inverse A -> B, including every verification coordinate,
   not just combinations that happen to cancel from a merge. Prove its identity
   and domains. In the polynomial fragment, independently ask Z3 whether two
   different B states can produce the same A. Only UNSAT establishes uniqueness.
4. Derive and verify the tuple merge invariant
   `merge(A(L), A(R)) = A(L union R)`. Different correction rates can use other
   tuple coordinates; individual states need not rebase in isolation.
5. Generate code and run the existing static and numerical accuracy checks.
   Outputs retained as original states read their own slots. Selection does not
   delete original or verification coordinates afterward.

A failed inverse, a SAT abstract collision, a timeout, and exhausted candidate
search do **not** prove that retaining an original is impossible. If no candidate
passes, the source program remains unfused. There is currently no global
impossibility prover authorizing replacements of otherwise defined originals.

## Tools and trusted boundary

Install the `superopt` optional dependencies, or install `sympy` and `z3-solver`
into the worktree environment. TVM itself must still be imported using PYTHONPATH,
not an editable install. Missing Z3 refuses selection with an explicit reason.

* Symbolic normalization and SymPy construct/check algebraic identities.
* Z3 checks real domains, zero implications, and recovery uniqueness. Each query
  has a 1500 ms timeout; UNKNOWN never counts as a proof.
* Arbitrary segment lengths are handled by generic finite-sum laws: a sum of
  nonnegative terms is nonnegative; if that sum is zero, every term is zero.
  Z3 proves nonnegativity and implications between the actual summand
  expressions. There is no Householder-name or expression-template dispatch.
* Square roots are encoded by a nonnegative root and its polynomial equation.
  Exp/log are conservative opaque functions for SMT (exp is positive); their
  algebraic identities use the symbolic/CAS verifier. The abstraction may have
  spurious counterexamples.

For a candidate G: B -> A, the uniqueness query is:

```
reachable_overapprox(B1) and reachable_overapprox(B2)
and G(B1) == G(B2) and B1 != B2
```

UNSAT proves injectivity on reachable states. A concrete, domain-checked inverse
is still required for code generation. SAT alone does not demonstrate a real
input collision unless the model can be realized as input data.

## Householder

For each nonempty segment:

```
q = sum(x*x)
k = sqrt(q)
v = sum((x + k*e)**2)
U = sum(e*e)
C = sum(x*e)
```

Keep `A = (q, v, U)` and use `B = (q, U, C)` for verification:

```
A -> B: C = 0 if q == 0 else (v - q - q*U) / (2*sqrt(q))
B -> A: v = q + 2*sqrt(q)*C + q*U
```

The solver proves the pointwise implications `x*x >= 0` and
`x*x == 0 => x*e == 0`. The generic sum laws give `q == 0 => C == 0`.
This justifies a zero branch without evaluating division by zero. Supplementing
C instead of U fails: `(x,e)=([0],[1])` and `([0],[3])` have the same `(q,v,C)`
but different U.

Generated guards use lazy `tir.if_then_else`; source-code common-expression
hoisting does not move branch-local divisions outside their guard. The merge
and any printed rebase formula are verified on reachable states. The output
reads the retained v slot directly.

## Different correction rates

If `T = P + Q` and rebasing gives `T' = rP*P + rQ*Q`, retain `(T,P)` and recover
`Q = T-P`. The rebase formula is `T' = rQ*T + (rP-rQ)*P`. Expression decomposition
provides supplement candidates; it does not replace T. The test combining a
centered squared sum with a shifted exponential sum retains `energy` and adds
only one component (plus the needed count).

## Current limits and behavior changes

Complete recovery is stricter than merge/output closure. Label smoothing now
uses six states, within its six-state theoretical budget, instead of the old
four-state representation that only retained the needed combinations.

The present inverse constructor does not certify preserving representations for
`max_abs_deviation`, `peak_power`, `papr`, `peak_to_geomean`, or `monotone_sum`
within their budgets. These are left unfused, rather than silently replacing
their original maxima. This is a search limitation, not an impossibility result.
Kurtosis still permits a repair because the source standardized moment divides
by zero on singleton segments.

The inverse constructor remains deliberately limited: it primarily isolates
linear reduction atoms, and proposes a zero-valued exceptional branch only when
Z3 proves that value. SMT validates candidates; it does not synthesize arbitrary
inverse functions or search every possible supplementary expression.

These are exact-real proofs, **not universal IEEE-754 proofs**. In particular,
a floating-point squared norm can underflow to zero without all input elements
being zero, and supplementary moments can overflow. The numerical gate and
regression cases provide bounded empirical validation, not an all-input error
bound. Disabling the accuracy gate does not disable recovery or preservation
proofs. Empty partitions continue to be skipped by the execution layer.

Tests: `tests/python/te/superopt/test_preserving.py` covers solver obligations,
unknown/missing-solver behavior, state budgets, guard emission and LLVM execution
on zero-norm segments. The remaining superopt tests cover ordinary chains,
source-domain repairs, differing context rates and accuracy filtering.
