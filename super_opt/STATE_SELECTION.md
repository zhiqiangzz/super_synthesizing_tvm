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
2. Try the complete rebase closure within the budget, then subsets of the
   algebraically extracted coefficients/components (at most 256 subset attempts).
   If these cannot establish all proof obligations, start Z3 CEGIS with only
   the protected original states fixed. The solver constructs supplementary
   expressions from verification atoms; failed rebase supplements are not locked
   into this search. Candidate generation and search scheduling never replace
   proof obligations. See the two-stage audit below for the precise grammar.
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
* When linear isolation fails, SymPy proposes simultaneous algebraic inverses
  after enumerating scalar-max branches (at most 16 cases, under a five-second
  search limit on the main thread). Only rational functions and square roots
  are translated back. Z3 proves each guarded inverse, then proves the assembled
  inverse on every reachable state, including ties and exceptional values.
* Arbitrary segment lengths are handled by generic finite-sum laws: a sum of
  nonnegative terms is nonnegative; if that sum is zero, every term is zero.
  Also, a pointwise proof `f <= g` implies `sum(f) <= count * max(g)`.
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

## PAPR: five states retaining the original reductions

The source program has the following reductions (the last two can share a
traversal after the mean is known):

```python
total = 0
for x in segment:
    total += x
mean = total / n
ss = peak = 0
for x in segment:
    d = x - mean
    ss += d * d
    peak = max(peak, d * d)
papr = peak / (ss / n)
```

For nonempty segments, let `hi = max(x)`, `lo = min(x)`, and `q = sum(x*x)`.
The verification coordinates are `B = (n, total, q, hi, lo)`. The selected
maintained coordinates are **`A = (total, peak, ss, n, w)`**, where `w = hi + lo`.
All three original reductions remain slots. The CEGIS coefficient solver constructs w as `max(x) - max(-x)`.
There is no hand-enumerated extrema-pair rule or operator-name dispatch.

The forward definition B -> A is:

```text
mu   = total / n
ss   = q - total**2 / n
peak = max(hi - mu, mu - lo)**2
w    = hi + lo
```

The inverse A -> B, in a compact equivalent form, is:

```text
mu = total / n
r  = sqrt(peak)
d  = w - 2*mu
hi = mu + r + min(d, 0)
lo = mu - r + max(d, 0)
q  = ss + total**2 / n
```

SymPy derives the two algebraic branches; the emitted code uses a lazy
conditional, not this min/max spelling. Z3 needs the reachable inequalities
`n*lo <= total <= n*hi`. The generic pointwise/sum/max law above supplies them
from x and -x. Together with `n > 0`, they establish nonnegative endpoint
distances, the correct square-root branch, and full recovery. No division by
peak or sqrt(peak) occurs: a constant segment, including a zero singleton,
has `peak = ss = 0` and a defined inverse.

The leaf is `(x, 0, 0, 1, 2*x)`. To merge two nonempty segments:

```text
decode hi_L, lo_L and hi_R, lo_R using the inverse
n     = n_L + n_R
total = total_L + total_R
mu    = total / n
ss    = ss_L + ss_R + n_L*(mu - mu_L)**2 + n_R*(mu - mu_R)**2
hi    = max(hi_L, hi_R)
lo    = min(lo_L, lo_R)
peak  = max(hi - mu, mu - lo)**2
w     = hi + lo
return (total, peak, ss, n, w)
```

The decoded extrema are temporary values, not additional persistent states.
Verified inverse expressions are used as local equality lemmas when proving
the complete merge invariant; a wrong merge still fails that proof. The printer
keeps the stable centered ss update even when the new joint coordinate needs
the general derived spelling. Expanding ss into differences of large raw
second moments fails the existing offset accuracy test.

Why add w? `[-3,1,1,1]` and `[-1,-1,-1,3]` both have
`(total,peak,ss,n) = (0,9,12,4)`, but w is -2 versus 2. Appending the same
`[4]` produces peaks `361/25` and `256/25`. The original tuple plus count
does not contain that orientation information. This is a concrete collision,
not a claim that five is a universal lower bound over arbitrary representations.

The default accuracy gate passes PAPR's unit, offset, and wide families.
Regression tests also compile the merge in float32/float64 and exercise
singletons, constant segments, ties, translated data, swapped operands, and
different merge trees. Source PAPR itself remains NaN on zero-variance input
because its final output is 0/0; the retained reduction states and their
inverse remain defined. This does not authorize replacing peak or ss.

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

The joint-coordinate and branch-inverse extension certifies `papr` in five
states and `max_abs_deviation`/`peak_power` in four, retaining their originals.
`peak_to_geomean` and `monotone_sum` still have no certified preserving
representation within the current search. This is a search limitation, not an
impossibility result. Scaled norm now has an algebraically certified candidate
after an authorized repair of its undefined zero-segment source state, but
the default numerical gate rejects its overflow/underflow behavior.
Kurtosis still permits a repair because the source standardized moment divides
by zero on singleton segments.

The inverse constructor remains limited: linear isolation and bounded algebraic
max-branch solving are not a complete decision procedure. The nonlinear search
requires at least as many coordinates as reduction atoms; it does not exploit
arbitrary reachable equalities to solve underdetermined systems. A zero-valued
exceptional branch is proposed only when Z3 proves that value. SMT validates
candidates; it does not synthesize arbitrary inverse functions or search every
possible supplementary expression.

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

## Additional applications: non-count supplements

`chain_ops.advanced` contains full unfused TE programs, independent handwritten
nonempty reducers, and float64 NumPy references. The synthesizer consumes only
the unfused programs. None of the synthesis or proof rules dispatch on these
operator names. All three use the default accuracy gate.

### Ledoit–Wolf covariance and shrinkage

Input is `X[batch, samples, features]`; outputs are the entire shrunk covariance
matrix and a shrinkage coefficient per batch. This follows the centered
[scikit-learn Ledoit–Wolf definition](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html).
For a feature pair, write its inputs as x and y. The source loops compute:

```text
sx = sum(x); sy = sum(y)
mx = sx/n; my = sy/n
C11 = sum((x-mx)*(y-my))
V22 = sum((x-mx)**2*(y-my)**2)
```

The maintained tuple has nine coordinates **per feature pair**:
`(sx, sy, V22, C11, n, C20, C02, C12, C21)`, where
`Cpq = sum((x-mx)**p * (y-my)**q)`. The four extra moments, unlike count,
contain data-dependent information. The current code duplicates means and
marginal moments across pairs; nine is not a claim about the minimal storage
of the full matrix.

The verification tuple is the nine raw moments `Mpq = sum(x**p*y**q)` at
`(p,q) = (0,0),(1,0),(0,1),(2,0),(0,2),(1,1),(1,2),(2,1),(2,2)`.
Both directions are finite triangular binomial transforms:

```text
B -> A: Cpq = sum_{i<=p,j<=q} binom(p,i)*binom(q,j)
                             * (-mx)**(p-i)*(-my)**(q-j)*Mij
A -> B: Mpq = sum_{i<=p,j<=q} binom(p,i)*binom(q,j)
                             * mx**(p-i)*my**(q-j)*Cij
C00=n, C10=C01=0, M10=sx, M01=sy
```

Only n is a divisor, and n is positive on every legal segment. To rebase a
segment to the union means, substitute `dx=mx_union-mx_segment` and similarly
dy into the same binomial transform. For example:

```text
C21' = C21 - dy*C20 - 2*dx*C11 - n*dx**2*dy
C12' = C12 - dx*C02 - 2*dy*C11 - n*dx*dy**2
V22' = V22 - 2*dy*C21 - 2*dx*C12 + dy**2*C20 + dx**2*C02
             + 4*dx*dy*C11 + n*dx**2*dy**2
```

Add the rebased left/right moments. Singleton leaves are
`(x,y,0,0,1,0,0,0,0)`. Thus neither zero variance nor singleton partitions
require replacing the original C11 or V22.

After the sample pass, let `S=C11/n`, `mu=trace(S)/d`,
`delta=sum(S*S)/d-mu**2`, and
`beta=min((sum(V22)/n-sum(S*S))/(d*n), delta)`.
The shrinkage coefficient is zero for one feature or beta=0, otherwise
`rho=beta/delta`. The complete covariance is `(1-rho)*S+rho*mu*I`.
The remaining trace and matrix sums traverse feature coordinates, not samples.

Shared `mean[i]` and `mean[j]` producers are specialized into independent read
views before chain discovery. Ordinary covariance matrices now benefit from
this same capability. The original outputs remain the agreement/accuracy
reference, and are returned unchanged when no candidate is accepted.

### Scalar OLS HC0 robust covariance

This is **one regressor without an intercept**, not ridge regression or general
multivariate HC0. It implements the scalar case of
[statsmodels HC0](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.HC0_se.html):

```text
a = sum(x*x); b = sum(x*y)
theta = 0 if a == 0 else b/a
H = sum(x*x*(y-theta*x)**2)
output = 0 if a == 0 else H/(a*a)
```

Keep **A=(a,b,H,D,F)** with `D=sum(x**4)` and
`F=sum(x**3*(y-theta*x))`. Verification coordinates are
**B=(a,b,D,E,K)**, `E=sum(x**3*y)`, `K=sum(x*x*y*y)`:

```text
B -> A: F=E-theta*D; H=K-2*theta*E+theta**2*D
A -> B: E=F+theta*D; K=H+2*theta*F+theta**2*D
```

The zero-a branch follows the scalar pseudoinverse convention; it introduces
no unguarded division. With `dt=theta_union-theta_segment`:
`F'=F-dt*D`, `H'=H-2*dt*F+dt**2*D`. Add the two rebased tuples.
Singleton leaves are `(x*x,x*y,0,x**4,0)`, including x=0.
Z3 proves these guarded leaf residuals zero; the emitted code does not form
large cancelling residual polynomials at every leaf. H remains an original
state. Source-domain proofs check lazy branches and reduction bodies, rather
than treating every syntactic divisor as active.

### Normalized-advantage policy loss

This follows the policy-loss expression in
[SB3 A2C with advantage normalization enabled](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/a2c/a2c.html).
Inputs are advantages a and log probabilities g; this operator does not compute
the policy network, GAE, gradients, entropy, or value loss.

```text
S=sum(a); mu=S/n
ss=sum((a-mu)**2)
scale=sqrt(ss/(n-1))+1e-8
loss_sum=sum((a-mu)*g/scale)
loss=-loss_sum/n
```

Use sample standard deviation, with epsilon **outside** the square root.
Unlike an epsilon-regularized population variance, the source normalized sum
is undefined on singleton segments. That real domain failure authorizes
moving its normalization to the epilogue. It does not authorize dropping S or ss.

The selected tuple is **A=(S,ss,C,n,G)**, where `C=sum((a-mu)*g)` and `G=sum(g)`.
For **B=(S,Q,P,n,G)**, with `Q=sum(a*a)`, `P=sum(a*g)`:

```text
B -> A: ss=Q-S*S/n; C=P-S*G/n
A -> B: Q=ss+S*S/n; P=C+S*G/n
```

The additional G cannot be replaced by count: rebasing C by a shift dt needs
`C'=C-dt*G`. Equivalently, the merged C is
`C_L+C_R+(mu_a_R-mu_a_L)*(mu_g_R-mu_g_L)*n_L*n_R/(n_L+n_R)`.
The code printer proposes mean-coordinate factorization only for bilinear
corrections and proves the resulting expression equal to the derived merge.
It leaves higher central moments in their stable shift form.
The final output is `-C/(n*(sqrt(ss/(n-1))+1e-8))`.
A singleton final batch still returns NaN; constant batches with n>1 return zero.

### Validation and scope

`test_advanced.py` checks complete LLVM outputs, external-library comparisons,
singletons, constant/zero inputs, single-feature covariance, large offsets,
nonempty partitions, swapped operands, and different merge trees in float32
and float64. External comparisons are optional when their packages are absent;
the NumPy, handwritten TE, source TE, proof and numerical-gate tests remain
mandatory. These are correctness checks, not GPU performance measurements.
Exact-real proof does not establish a universal floating-point error bound.

For HC0 on large-offset data, the residual cross moment F can have a much
larger relative error than H: perturbing theta by one floating-point ulp
perturbs F by approximately `D*ulp(theta)`. The merge-tree test therefore
checks F against this explicit conditioning scale, while H and the complete
covariance keep their ordinary tolerances. The default accuracy gate is
unchanged. Current synthesis retains sums a,b, rather than storing a separate
higher-precision theta; this sensitivity remains a limitation.

The showcase now contains 17 operators. The three additions have no claimed
specialized GPU baseline; their unfused TE programs and handwritten reducers
are the execution comparisons. Independent correctness checks were also run
with scikit-learn 1.9.1, statsmodels 0.15.0 and PyTorch (the A2C expression),
without treating these checks as performance comparisons.

## Two-stage selection and the 17-operator audit

Selection now follows this order:

```text
preserve the source reductions (repair only a demonstrated source-domain defect)
  -> try the originals alone
  -> extract rebase coefficients/components; try them within the state budget
  -> prove complete recovery, source/inverse domains, the full merge and its domains
       success -> generate code -> existing numerical gate
       failure -> Z3 CEGIS -> independent recovery/domain/merge proof -> numerical gate
```

The numerical gate still decides final acceptance; a proof does not bypass it.
A numerical rejection does not silently authorize replacing a source state.
Only A0 is fixed on entry to CEGIS; unsuccessful rebase supplements are discarded.
This matters when a different representation must fit the same state budget.
The old manually enumerated pairwise sums/differences of extrema are removed.

The `parts()` candidate source and its repair-time extraction of inner reductions
are also removed. Algebraic candidates still come from rebase coefficients and
additive `pieces()`; CEGIS can synthesize joint extrema coordinates. The non-core
`max_j ((x_j - mean) / sigma)^3` fusion test is removed: supporting that case again
requires extending rebase decomposition and safe inversion or repair-coordinate
selection. Its former extraction of the inner maximum is no longer provided.

The implementation uses **Z3-driven CEGIS**, not a cvc5 SyGuS invocation. The
current explicit grammar is:

```text
C_j(B) = sum_i c_ji * B_i,  c_ji in {-2,-1,0,1,2}
```

B_i range over the complete verification atoms. Thus the solver can construct
joint coordinates, not only select a subset of an operator-specific list.
This first implementation does not synthesize arbitrary products, conditional
state definitions or transcendental expressions for C. Those remain supported
where algebraic rebase extraction already provides them. The inverse D is
constructed with the existing algebraic/SymPy machinery and independently
checked; this is not unrestricted joint synthesis of arbitrary C and D programs.

For each coefficient proposal the verifier looks for equal maintained states
with unequal B. An exact model supplies basis differences `d_i=B_i(left)-B_i(right)`;
the next coefficient matrix must satisfy `OR_j(sum_i c_ji*d_i != 0)`. These
counterexamples eliminate entire families of coefficients. No floating-point
rounding is used when feeding models back. A proposed tuple is accepted only
when the separate checker constructs a full, defined inverse and proves the
actual reducer. Injectivity by itself is insufficient.

Counterexamples belong to an overapproximation of reachable states. They are
not automatically realizable input segments, and failure after using them is
not an impossibility theorem. Automatic refinement of the reachable-state
abstraction is not implemented. For exp/log, the constructive checker is also
tried before learning a collision, because opaque-function SMT models can miss
identities known to the algebraic verifier.

The coefficient learner uses Z3 with increasing L1 expression-cost bounds.
It tries the available state-budget width first, then smaller widths, with fair
shares of a 64-proposal budget and a 20-second search deadline (an in-flight
proof may finish after that deadline). This scheduling does not establish
minimality. Solver calls have timeouts; missing Z3, unknown, exhausted bounded
search and resource limits are reported separately and never permit a hoist.
Independent proof failure blocks that coefficient assignment and continues.

`Selection.route` records `original`, `rebase`, or `cegis`; `Selection.search`
records stage outcomes, candidate counts and collision counts. A proved selection
may still be rejected by the numerical gate, so the audit records both.
Reproduce the audit, including the default numerical gate, with:

```bash
python super_opt/examples/audit_state_selection.py
python super_opt/examples/audit_state_selection.py --json
```

The measured result for the current showcase is:

| Operator | Selection route | Source-domain repair first | Maintained slots per chain |
|---|---|---|---|
| Ledoit–Wolf covariance | rebase | No | 9 per feature pair |
| Scalar OLS HC0 | rebase | No | 5 |
| Normalized policy loss | rebase | Yes: singleton sample std is undefined | 5 |
| Softmax | original | No | 2 |
| Cross entropy | original | No | 3 |
| Softmax entropy | original | No | 3 |
| Attention | original | No | 3 |
| Layer normalization | rebase | No | 3 |
| Layer normalization backward | rebase | No | 5 |
| PAPR | **CEGIS** | No | 5 |
| Householder | rebase | No | 3 |
| Sinkhorn iteration | original | No | 2 + 2 |
| Kurtosis | rebase | Yes: singleton standardized moment is undefined | 5 |
| Pearson correlation | rebase | No | 6 |
| Effective sample size | original | No | 3 |
| Contrastive loss | original | No | 2 + 2 |
| Heat capacity | original | No | 4 |

All 17 pass the default numerical gate. **Only PAPR needs the CEGIS fallback**
under the current budget and rebase candidate language. Its original peak is
rebased through the two extrema; storing both separately alongside the original
three states and count would take six slots. The five-slot solution instead
uses `count` and the joint coordinate `max(x)+min(x)`. Z3 constructs that
coordinate from the verification atoms; SymPy and Z3 certify its guarded inverse.
This is an observed limitation of the current rebase candidate language, not a
proof that no stronger algebraic analysis could discover the same coordinate.

Kurtosis and normalized policy loss are a different category: the source
normalized reduction is genuinely undefined on singleton partitions. After the
existing solver-authorized repair, rebase coefficients suffice. Householder's
zero-norm branch is proved safe, retains vv, and does not enter CEGIS.
