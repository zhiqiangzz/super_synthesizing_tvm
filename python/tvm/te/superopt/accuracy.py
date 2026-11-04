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
"""Empirical precision gate for a fused program relative to its source.

The original and candidate execute in their working dtype. MPFR evaluates the
source expressions at increasing precision on the same quantized inputs. A
precision-stability check is evidence for a reference, not an error certificate.

The configurable regression policy permits a factor of the source error or
sampled input sensitivity, plus a relative epsilon floor and an absolute floor,
per output element. Sensitivity samples are not condition-number bounds or a
backward-stability proof. Static domain/exp checks remain separate. Hypothesis
property tests exercise wider input and merge-tree spaces outside this fast gate.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np

import tvm
import tvm_tirx
from tvm import te
from tvm_tirx import tirx as tir
from tvm_tirx.ir import Call, FloatImm, IntImm

from . import numeric
from .loads import is_load, load_indices, load_tensor
from .symbolic import ir
from .symbolic.canonicalize import (
    Unsupported,
    contains_node,
    instantiate,
    is_constant,
    map_elems,
    mk_mul,
    mk_neg,
    positive,
    recanonicalize,
    term_view,
)
from .symbolic.lower import LowerCtx, classify_combiner, compute_ops

# exp overflows float32 above ~88.7 and float16 above ~11; a provably bounded
# argument is one bounded by a constant no larger than this.
EXP_ARG_LIMIT = Fraction(8)


@dataclasses.dataclass(frozen=True)
class AccuracyConfig:
    """How candidates are exercised numerically.

    reduce_extent
        Value given to every symbolic extent that some reduction of the
        original runs over (long reductions are where rounding accumulates).
    other_extent
        Value given to the remaining symbolic extents.
    max_reference_elements
        Cap each operation's sampled iteration volume. Symbolic dimensions are
        reduced deterministically when a multidimensional graph exceeds this
        budget; a fixed shape exceeding it is reported as unsupported.
    families
        ``(name, shift, scale)``: every input is ``shift + scale * N(0, 1)``.
    factor, relative_eps, absolute_tolerance
        Per element, allow ``factor * max(source_abs_error, sampled_sensitivity)
        + relative_eps * eps(dtype) * abs(reference) + absolute_tolerance``.
        This is an empirical regression budget, not a numerical error theorem.
    perturbations
        Adjacent-float input perturbations used to sample sensitivity. Invalid
        perturbed references do not increase the budget.
    reference_precision, reference_max_precision
        MPFR precision starts here and doubles until successive references agree
        to 80 relative bits, or the maximum precision rejects the reference.
    positive
        Names of inputs that are only meaningful when positive (weights,
        masses), besides those the program itself puts under a ``log``, a
        ``sqrt`` or a division: they are sampled that way.
    magnitudes, magnitude_bar
        Two more families, scaled to the dtype: every input beyond the square
        root of the largest (``"huge"``) and of the smallest (``"tiny"``)
        normal number, where a sum of squares overflows or vanishes. They
        only count where the original is itself accurate to
        ``magnitude_bar``: a program written to survive such inputs (a norm
        computed with scaling) has to keep doing so, one that does not is
        not held to it.
    """

    reduce_extent: int = 256
    other_extent: int = 3
    max_reference_elements: int = 16384
    families: tuple[tuple[str, float, float], ...] = (
        ("unit", 0.0, 1.0),
        ("offset", 1.0e4, 1.0),
        ("wide", 0.0, 10.0),
    )
    factor: float = 100.0
    relative_eps: float = 64.0
    absolute_tolerance: float = 0.0
    reference_precision: int = 128
    reference_max_precision: int = 2048
    perturbations: int = 2
    positive: tuple[str, ...] = ()
    magnitudes: bool = True
    magnitude_bar: float = 1.0e-3
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class FamilyError:
    family: str
    original: float  # normwise error against the MPFR reference
    sensitivity: float  # sampled sensitivity, not a condition-number bound
    candidate: float
    ok: bool
    original_metrics: tuple[numeric.ErrorMetrics, ...] = ()
    candidate_metrics: tuple[numeric.ErrorMetrics, ...] = ()
    reference_precision: int = 0
    sensitivity_samples: int = 0


@dataclasses.dataclass(frozen=True)
class AccuracyReport:
    ok: bool
    # (kind, op name) for each static issue the original does not have; kind is
    # "exp" (argument not provably bounded above) or "domain" (a divisor or log
    # argument not provably positive)
    static: tuple[tuple[str, str], ...]
    families: tuple[FamilyError, ...]
    note: str = ""  # why no numeric comparison was possible
    extents: tuple[tuple[str, int], ...] = ()

    @property
    def static_ok(self) -> bool:
        return not self.static

    def summary(self) -> str:
        head = "passed sampled precision checks" if self.ok else "REJECTED"
        what = {"exp": "unbounded exp", "domain": "divisor/log of unknown sign"}
        parts = [f"{what[kind]} in {op}" for kind, op in self.static]
        if self.note:
            parts.append(self.note)
        for f in self.families:
            mark = "" if f.ok else " (!)"
            details = ""
            if f.candidate_metrics:
                details = (
                    f", abs {max(e.absolute for e in f.candidate_metrics):.2g}"
                    f", rel {max(e.relative for e in f.candidate_metrics):.2g}"
                    f", ULP {max(e.ulps for e in f.candidate_metrics):.2g}"
                    f", nonfinite {sum(e.nonfinite for e in f.candidate_metrics)}"
                )
            parts.append(
                f"{f.family}: {f.candidate:.2g} (original {f.original:.2g}, "
                f"sampled sensitivity {f.sensitivity:.2g}, MPFR {f.reference_precision} bits"
                f"{details}){mark}"
            )
        return f"{head}: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# static: bounded exp arguments
# ---------------------------------------------------------------------------
PSEUDO_BASE = -1000  # tensor ids of computed tensors read opaquely: -1000, -1001, ...


class _OpLower(LowerCtx):
    """Lower one op body at a time; reads of computed tensors stay opaque elements."""

    def __init__(self) -> None:
        # reductions as written: ``Σ exp(s - m)`` must not become ``exp(-m) Σ exp(s)``
        super().__init__(raw_reductions=True)
        self.computed: list[te.Tensor] = []
        self._defs: dict[int, ir.SymExpr | None] = {}

    def pseudo_id(self, t: te.Tensor) -> int:
        for n, known in enumerate(self.computed):
            if known.op.same_as(t.op) and known.value_index == t.value_index:
                return PSEUDO_BASE - n
        self.computed.append(t)
        return PSEUDO_BASE - (len(self.computed) - 1)

    def lower_load(self, e, env, level):
        if isinstance(load_tensor(e).op, te.ComputeOp):
            tid = self.pseudo_id(load_tensor(e))
            return ir.elem(tid, tuple(self.lower_index(i, env) for i in load_indices(e)))
        return super().lower_load(e, env, level)

    def body(self, op, k: int, branch: int = 2) -> ir.SymExpr:
        env = {iv.var.name: ir.idx(f"i{n}") for n, iv in enumerate(op.axis)}
        if (
            op.tag == "superopt.singleton"
            and isinstance(op.body[k], Call)
            and op.body[k].op.name == "prim.if_then_else"
        ):
            return self.lower_expr(op.body[k].args[branch], env, 0)
        if op.tag == "superopt.nonempty":
            # For static analysis, restore the full axis including the first
            # leaf used as init. Both merge operands then have reachable-state
            # signs. No empty-state value participates in this analysis.
            body = op.body[k]
            ax = body.axis[0]
            end = tvm_tirx.sym.Analyzer().simplify(ax.dom.begin + ax.dom.extent)
            axis = te.reduce_axis((0, end), ax.var.name)
            full = te.Reduce(body.combiner, body.source, [axis], body.condition, k, [])
            lowered = self.lower_expr(full, env, 0)
            if isinstance(lowered, ir.MonoidReduce):
                return ir.raw_monoid(
                    lowered.domain,
                    lowered.level,
                    lowered.leaf,
                    lowered.merge,
                    lowered.leaf,
                    lowered.slot,
                )
            return lowered
        return self.lower_expr(op.body[k], env, 0)

    def definition(self, tid: int) -> ir.SymExpr | None:
        """Body of the computed tensor behind a pseudo id (over ``i0 ..``)."""
        if tid not in self._defs:
            t = self.computed[PSEUDO_BASE - tid]
            try:
                self._defs[tid] = self.body(t.op, int(t.value_index))
            except Unsupported:
                self._defs[tid] = None
        return self._defs[tid]


@dataclasses.dataclass
class _Scope:
    lower: _OpLower
    out_keys: tuple
    binders: dict[int, int]  # binder level -> axis key
    depth: int

    def index_key(self, e: ir.IndexExpr):
        if isinstance(e, ir.Idx) and e.name.startswith("i"):
            k = int(e.name[1:])
            return self.out_keys[k] if k < len(self.out_keys) else None
        if isinstance(e, ir.BIdx):
            return self.binders.get(e.level)
        return None


def _split_scale(e: ir.SymExpr) -> tuple[ir.SymExpr, ir.SymExpr]:
    """``e = w * rest`` with ``w`` the product of its positive data-free factors."""
    if not isinstance(e, ir.Mul):
        return ir.ONE, e
    w = [a for a in e.args if is_constant(a) and positive(a)]
    if not w:
        return ir.ONE, e
    rest = [a for a in e.args if not (is_constant(a) and positive(a))]
    return mk_mul(*w), (mk_mul(*rest) if rest else ir.ONE)


def _match(p: ir.Node, t: ir.Node, var: ir.BIdx, bind: dict) -> bool:
    """Structural match of ``p`` against ``t`` with ``var`` as the only pattern variable."""
    if p is var:
        if var in bind:
            return bind[var] is t
        bind[var] = t
        return True
    if not p.uses_level(var.level):
        return p is t
    if type(p) is not type(t) or len(p._fields) != len(t._fields):
        return False
    for fp, ft in zip(p._fields, t._fields):
        if isinstance(fp, ir.Node):
            if not isinstance(ft, ir.Node) or not _match(fp, ft, var, bind):
                return False
        elif isinstance(fp, tuple):
            if not isinstance(ft, tuple) or len(fp) != len(ft):
                return False
            for a, b in zip(fp, ft):
                if isinstance(a, ir.Node):
                    if not isinstance(b, ir.Node) or not _match(a, b, var, bind):
                        return False
                elif a != b:
                    return False
        elif fp != ft:
            return False
    return True


def _running_max(m: ir.MonoidReduce) -> bool:
    k = m.slot
    a, b = ir.state_var("a", k), ir.state_var("b", k)
    return isinstance(m.merge[k], ir.Max) and set(m.merge[k].args) == {a, b}


def _looked_through(e: ir.SymExpr, scope: _Scope, fuel: int = 6) -> ir.SymExpr:
    """``e`` with every elementwise computed tensor replaced by its definition.

    Two programs may read the same quantity through a stored tensor and by
    recomputing it (``Z[i, j]`` against ``X[i, j] / T[i]``): written down to
    the tensors that are not elementwise, they are the same expression.
    """

    def inline(n: ir.Elem, depth: int):
        if n.tensor > PSEUDO_BASE:
            return None
        d = scope.lower.definition(n.tensor)
        if d is None or contains_node(d, lambda x: isinstance(x, ir.Reduce | ir.MonoidReduce)):
            return None
        return instantiate(d, {ir.idx(f"i{k}"): i for k, i in enumerate(n.indices)}, depth)

    for _ in range(fuel):
        new = map_elems(e, inline)
        if new is e:
            break
        e = new
    return e


def _over_axis(body: ir.SymExpr, red, x: ir.SymExpr, scope: _Scope) -> bool:
    """``x`` is ``body`` at some point of the axis ``red`` takes the maximum over."""
    var = ir.bidx(red.level)
    for pattern, value in ((body, x), (_looked_through(body, scope), _looked_through(x, scope))):
        bind: dict = {}
        if _match(pattern, value, var, bind):
            point = bind.get(var)
            return point is None or scope.index_key(point) == red.domain.axis
    return False


def _dominates(m: ir.SymExpr, x: ir.SymExpr, scope: _Scope, fuel: int = 8) -> bool:
    """Syntactic proof that ``m >= x`` everywhere."""
    if m is x:
        return True
    if fuel == 0:
        return False
    if isinstance(m, ir.Max):
        return any(_dominates(a, x, scope, fuel - 1) for a in m.args)
    if isinstance(m, ir.MonoidReduce) and _running_max(m):
        # a slot merged by ``max(a_k, b_k)`` is the maximum of its leaf over the axis
        return _over_axis(m.leaf[m.slot], m, x, scope)
    wm, rm = _split_scale(m)
    wx, rx = _split_scale(x)
    if wm is not wx:
        return False
    if wm is not ir.ONE:
        return _dominates(rm, rx, scope, fuel - 1)
    if isinstance(m, ir.Reduce) and m.kind == "max":
        return _over_axis(m.body, m, x, scope)
    if (
        isinstance(m, ir.Mul)
        and len(m.args) == 2
        and m.args[0] is ir.MINUS_ONE
        and isinstance(m.args[1], ir.Elem)
        and m.args[1].tensor <= PSEUDO_BASE
    ):  # the negation of a stored minimum is a maximum
        t = m.args[1]
        d = scope.lower.definition(t.tensor)
        if d is None:
            return False
        imap = {ir.idx(f"i{k}"): i for k, i in enumerate(t.indices)}
        return _dominates(mk_neg(instantiate(d, imap, scope.depth)), x, scope, fuel - 1)
    if isinstance(m, ir.Elem) and m.tensor <= PSEUDO_BASE:
        d = scope.lower.definition(m.tensor)
        if d is None:
            return False
        imap = {ir.idx(f"i{k}"): i for k, i in enumerate(m.indices)}
        return _dominates(instantiate(d, imap, scope.depth), x, scope, fuel - 1)
    return False


def _nonneg(e: ir.SymExpr) -> bool:
    if positive(e) or isinstance(e, ir.Exp | ir.Card):
        return True
    if isinstance(e, ir.Pow):
        p = e.exponent
        return p.denominator == 1 and p.numerator % 2 == 0
    if isinstance(e, ir.Mul):
        return all(_nonneg(a) for a in e.args)
    if isinstance(e, ir.Max):
        return any(_nonneg(a) for a in e.args)
    return False


def bounded_above(arg: ir.SymExpr, scope: _Scope) -> bool:
    """``arg <= C`` for a small constant ``C``: shifted by a dominating max, or constant."""
    if _bounded_as_written(arg, scope):
        return True
    through = _looked_through(arg, scope)
    return through is not arg and _bounded_as_written(through, scope)


def _bounded_as_written(arg: ir.SymExpr, scope: _Scope) -> bool:
    """``arg <= C``: every term that can grow is held down by one that dominates it.

    ``Σ c_i x_i - C max(.., x_i, ..)`` with ``Σ c_i <= C`` is at most zero, and so
    is its mirror image ``C min(.., x_i, ..) - Σ c_i x_i``, which the IR writes
    with ``min u = -max(-u)`` as subtracted terms only. A dominating term is
    used up by what it holds down, coefficient for coefficient; whatever is
    left of a subtracted term has to be non-negative itself.
    """
    c, terms = term_view(arg)
    if isinstance(c, float) or c > EXP_ARG_LIMIT:
        return c == float("-inf")
    neg = [(core, -coeff) for core, coeff in terms.items() if coeff < 0]
    left = [coeff for _, coeff in neg]  # of each subtracted term: what is not used up yet

    def absorb(need, holds, skip: int | None = None) -> bool:
        """Take ``need`` from the subtracted terms that ``holds`` accepts, if they have it."""
        takers = [n for n in range(len(neg)) if n != skip and left[n] > 0 and holds(neg[n][0])]
        if sum(left[n] for n in takers) < need:
            return False
        for n in takers:
            take = min(need, left[n])
            left[n] -= take
            need -= take
        return True

    for core, coeff in terms.items():
        if coeff <= 0:
            continue
        # x - max(.., x, ..) <= 0, and so is min(.., x, ..) - x

        def holds(m, core=core) -> bool:
            return _dominates(m, core, scope) or _dominates(mk_neg(core), mk_neg(m), scope)

        if not absorb(coeff, holds):
            return False
    progress = True
    while progress:  # -u - max(.., -u, ..) <= 0: min(.., u, ..) - u as the IR writes it
        progress = False
        for n, (core, _) in enumerate(neg):
            if left[n] > 0 and not _nonneg(core):
                x = mk_neg(core)
                if absorb(left[n], lambda m, x=x: _dominates(m, x, scope), skip=n):
                    left[n] = Fraction(0)
                    progress = True
    return all(left[n] == 0 or _nonneg(core) for n, (core, _) in enumerate(neg))


POS, NONNEG, ANY = 2, 1, 0  # sign lattice: > 0, >= 0, unknown


class _Signs:
    """Sign inference over op bodies, computed tensors and reducer states.

    A slot's sign is the greatest fixpoint of the leaf and merged-state signs.
    Generic TE folds also include their initializer on the left side.
    Derived non-empty folds use a leaf as that initializer, so both merge
    operands have reachable-state signs, without an empty-state value.
    """

    def __init__(self, lower: _OpLower) -> None:
        self.lower = lower
        self.tensors: dict[int, int] = {}
        self.monoids: dict[ir.MonoidReduce, dict] = {}

    def tensor(self, tid: int) -> int:
        if tid not in self.tensors:
            self.tensors[tid] = ANY  # guards recursion
            t = self.lower.computed[PSEUDO_BASE - tid]
            if t.op.tag == "superopt.singleton":
                try:
                    self.tensors[tid] = min(
                        self.sign(self.lower.body(t.op, int(t.value_index), branch), {})
                        for branch in (1, 2)
                    )
                except Unsupported:
                    self.tensors[tid] = ANY
            else:
                d = self.lower.definition(tid)
                self.tensors[tid] = ANY if d is None else self.sign(d, {})
        return self.tensors[tid]

    def sign(self, e: ir.SymExpr, env: dict) -> int:
        if e in env:
            return env[e]
        if isinstance(e, ir.Const):
            v = e.value
            return POS if v > 0 else (NONNEG if v == 0 else ANY)
        if isinstance(e, ir.ShapeSym | ir.Card | ir.Exp):
            return POS
        if isinstance(e, ir.StateVar):
            return env.get(e, ANY)
        if isinstance(e, ir.Elem):
            return self.tensor(e.tensor) if e.tensor <= PSEUDO_BASE else ANY
        if isinstance(e, ir.Add):
            signs = [self.sign(a, env) for a in e.args]
            if min(signs) == ANY:
                return ANY
            return POS if max(signs) == POS else NONNEG
        if isinstance(e, ir.Mul):
            return min(self.sign(a, env) for a in e.args)
        if isinstance(e, ir.Pow):
            p, b = e.exponent, self.sign(e.base, env)
            if p < 0:
                return POS if b == POS else ANY
            if p.denominator == 1 and p.numerator % 2 == 0:
                return POS if b == POS else NONNEG
            return b
        if isinstance(e, ir.Max):
            return max(self.sign(a, env) for a in e.args)
        if isinstance(e, ir.IfZero):
            return min(self.sign(e.zero, env), self.sign(e.nonzero, env))
        if isinstance(e, ir.Reduce):
            return self.sign(e.body, env)
        if isinstance(e, ir.MonoidReduce):
            return self.fold(e)["slots"][e.slot]
        return ANY

    def fold(self, m: ir.MonoidReduce) -> dict:
        """``{"slots": sign per slot, "env": state-variable signs inside the merge}``."""
        hit = self.monoids.get(m)
        if hit is not None:
            return hit
        n = len(m.leaf)
        ident = [self.sign(c, {}) for c in m.identity]
        leaf = [self.sign(x, {}) for x in m.leaf]
        slots = [POS] * n
        while True:
            env = {}
            for k in range(n):
                env[ir.state_var("a", k)] = min(slots[k], ident[k])
                env[ir.state_var("b", k)] = slots[k]
            new = [min(slots[k], leaf[k], self.sign(m.merge[k], env)) for k in range(n)]
            if new == slots:
                break
            slots = new
        hit = {"slots": slots, "env": env}
        self.monoids[m] = hit
        return hit


def _sites(e: ir.Node, depth: int, binders: dict, env: dict, signs: _Signs, out: list) -> None:
    """Collect ``("exp", arg, depth, binders)`` and ``("domain", operand, env)`` sites."""
    if isinstance(e, ir.IfZero):
        _sites(e.test, depth, binders, env, signs, out)
        _sites(e.zero, depth, binders, env, signs, out)
        nonzero = dict(env)
        if signs.sign(e.test, env) >= NONNEG:
            nonzero[e.test] = POS
            if isinstance(e.test, ir.Pow) and e.test.exponent > 0:
                if signs.sign(e.test.base, env) >= NONNEG:
                    nonzero[e.test.base] = POS
        _sites(e.nonzero, depth, binders, nonzero, signs, out)
        return
    if isinstance(e, ir.Exp):
        out.append(("exp", e.arg, depth, binders))
    elif isinstance(e, ir.Pow) and e.exponent < 0:
        out.append(("domain", e.base, env))
    elif isinstance(e, ir.Log):
        out.append(("domain", e.arg, env))
    if isinstance(e, ir.Reduce):
        _sites(e.body, e.level + 1, {**binders, e.level: e.domain.axis}, env, signs, out)
        return
    if isinstance(e, ir.MonoidReduce):
        inner = {**binders, e.level: e.domain.axis}
        for x in e.leaf:
            _sites(x, e.level + 1, inner, env, signs, out)
        merge_env = signs.fold(e)["env"]
        for x in e.merge:
            _sites(x, 0, {}, merge_env, signs, out)
        return
    for c in e.children():
        _sites(c, depth, binders, env, signs, out)


def _tensors(outputs) -> list[te.Tensor]:
    return [outputs] if isinstance(outputs, te.Tensor) else list(outputs)


def static_issues(outputs) -> dict[str, list[str]]:
    """Ops (by name) with an unbounded ``exp`` argument (``"exp"``) or a divisor /
    ``log`` argument not provably positive (``"domain"``)."""
    lower = _OpLower()
    signs = _Signs(lower)
    issues: dict[str, list[str]] = {"exp": [], "domain": []}
    for op in compute_ops(outputs):
        bodies = [0] if isinstance(op.body[0], te.Reduce) else range(len(op.body))
        out_keys = tuple(lower.dims.key(iv.dom.extent) for iv in op.axis)
        bad: set[str] = set()
        for k in bodies:
            try:
                sites: list = []
                for branch in (1, 2) if op.tag == "superopt.singleton" else (2,):
                    _sites(lower.body(op, k, branch), 0, {}, {}, signs, sites)
            except Unsupported:
                bad |= {"exp", "domain"}
                break
            for site in sites:
                if site[0] == "exp":
                    _, arg, depth, binders = site
                    if not bounded_above(arg, _Scope(lower, out_keys, binders, depth)):
                        bad.add("exp")
                elif not _positive_domain(site[1], site[2], signs):
                    bad.add("domain")
        for kind in sorted(bad):
            issues[kind].append(op.name)
    return issues


def _positive_domain(operand, env, signs):
    if signs.sign(operand, env) == POS:
        return True
    from .reducer import smt
    from .symbolic.canonicalize import walk

    known = dict(env)
    for n in walk(operand):
        if isinstance(n, ir.StateVar | ir.Elem):
            known[n] = signs.sign(n, env)
    return smt.condition_holds(operand, "positive", tuple(known.items()))


# ---------------------------------------------------------------------------
# differential: MPFR reference by interpreting the TE expressions
# ---------------------------------------------------------------------------
# Intrinsics the reference can evaluate. The ops around a chain are not analysed,
# only run, so this covers more than the lowering understands.
def _elementwise(name, *args):
    mp = numeric.require_mpfr()
    if name == "prim.if_then_else":
        return np.where(*args)
    functions = {
        "prim.exp": mp.exp,
        "prim.log": mp.log,
        "prim.sqrt": mp.sqrt,
        "prim.rsqrt": lambda x: 1 / mp.sqrt(x),
        "prim.pow": lambda x, y: x**y,
        "prim.tanh": mp.tanh,
        "prim.sigmoid": lambda x: 1 / (1 + mp.exp(-x)),
        "prim.fabs": abs,
        "prim.sin": mp.sin,
        "prim.cos": mp.cos,
        "prim.floor": mp.floor,
        "prim.ceil": mp.ceil,
    }
    if name not in functions:
        raise Unsupported(f"intrinsic {name}")
    return np.frompyfunc(functions[name], len(args), 1)(*args)


class _Reference:
    """Evaluate source expressions in the active MPFR context, without float casts.

    This models real arithmetic approximately; it does not emulate intermediate
    storage rounding. Input floats and literal constants retain their exact values.
    """

    def __init__(self, inputs: list[te.Tensor], arrays: list[np.ndarray], var_values):
        self.values: list[tuple[object, list[np.ndarray]]] = [
            (t.op, [numeric.mp_array(a)]) for t, a in zip(inputs, arrays)
        ]
        self.vars = var_values

    def tensor(self, t: te.Tensor) -> np.ndarray:
        for op, arrs in self.values:
            if op.same_as(t.op):
                return arrs[int(t.value_index)]
        if not isinstance(t.op, te.ComputeOp):
            raise Unsupported(f"operation {type(t.op).__name__}")
        arrs = self.compute(t.op)
        self.values.append((t.op, arrs))
        return arrs[int(t.value_index)]

    def extent(self, e) -> int:
        return int(self.index(e, {}))

    def index(self, e, env):
        if isinstance(e, IntImm):
            return int(e.value)
        if isinstance(e, tir.Var):
            return env[e.name] if e.name in env else int(self.vars[e.name])
        if isinstance(e, tir.Add):
            return self.index(e.a, env) + self.index(e.b, env)
        if isinstance(e, tir.Sub):
            return self.index(e.a, env) - self.index(e.b, env)
        if isinstance(e, tir.Mul):
            return self.index(e.a, env) * self.index(e.b, env)
        if isinstance(e, tir.FloorDiv):
            return self.index(e.a, env) // self.index(e.b, env)
        if isinstance(e, tir.FloorMod):
            return self.index(e.a, env) % self.index(e.b, env)
        raise Unsupported(f"index expression {type(e).__name__}")

    def ev(self, e, env):
        if isinstance(e, FloatImm | IntImm):
            # as written: a reducer identity is the dtype's lowest finite value, and
            # ``0 * lowest`` is 0 where ``0 * -inf`` would be NaN
            return numeric.require_mpfr().mpfr(e.value)
        if isinstance(e, tir.Var):
            if e.name in env:
                return env[e.name]
            return numeric.require_mpfr().mpfr(self.vars[e.name])
        if isinstance(e, tir.Cast):
            value = self.ev(e.value, env)
            if str(e.ty.dtype).startswith(("int", "uint")):
                return np.frompyfunc(int, 1, 1)(value)
            if str(e.ty.dtype) == "bool":
                return np.not_equal(value, 0)
            return numeric.mp_array(value)
        binary = {
            tir.EQ: np.equal,
            tir.NE: np.not_equal,
            tir.LT: np.less,
            tir.LE: np.less_equal,
            tir.GT: np.greater,
            tir.GE: np.greater_equal,
            tir.Add: np.add,
            tir.Sub: np.subtract,
            tir.Mul: np.multiply,
            tir.Div: np.divide,
            tir.Max: np.maximum,
            tir.Min: np.minimum,
        }
        for cls, fn in binary.items():
            if isinstance(e, cls):
                return fn(self.ev(e.a, env), self.ev(e.b, env))
        if is_load(e):
            arr = self.tensor(load_tensor(e))
            return arr[tuple(self.index(i, env) for i in load_indices(e))]
        if isinstance(e, Call):
            name = e.op.name
            return _elementwise(name, *(self.ev(a, env) for a in e.args))
        raise Unsupported(f"expression {type(e).__name__}")

    def compute(self, op) -> list[np.ndarray]:
        shape = [self.extent(iv.dom.extent) for iv in op.axis]
        first = op.body[0]
        raxes = list(first.axis) if isinstance(first, te.Reduce) else []
        full = shape + [self.extent(iv.dom.extent) for iv in raxes]
        nd = len(full)
        env = {}
        for pos, iv in enumerate([*op.axis, *raxes]):
            env[iv.var.name] = (self.extent(iv.dom.begin) + np.arange(full[pos])).reshape(
                [1] * pos + [full[pos]] + [1] * (nd - pos - 1)
            )
        with np.errstate(all="ignore"):
            if not raxes:
                return [np.broadcast_to(self.ev(b, env), shape).copy() for b in op.body]
            cond = first.condition
            if not (isinstance(cond, IntImm) and int(cond.value) == 1):
                raise Unsupported("conditional reduction")
            srcs = [np.broadcast_to(self.ev(s, env), full) for s in first.source]
            dims = tuple(range(len(shape), nd))
            kind = (
                None
                if len(first.init) or op.tag == "superopt.nonempty"
                else classify_combiner(first.combiner)
            )
            if kind == "sum":
                states = [srcs[0].sum(axis=dims)]
            elif kind == "max":
                states = [srcs[0].max(axis=dims)]
            elif kind == "min":
                states = [srcs[0].min(axis=dims)]
            else:
                init_env = {
                    iv.var.name: env[iv.var.name].reshape(
                        [1] * p + [shape[p]] + [1] * (len(shape) - p - 1)
                    )
                    for p, iv in enumerate(op.axis)
                }
                initial = first.init if len(first.init) else first.combiner.identity_element
                init = [self.ev(x, init_env) for x in initial]
                states = self._fold(first.combiner, srcs, shape, init)
        return [np.asarray(states[int(b.value_index)]).copy() for b in op.body]

    def _fold(self, comb, srcs, shape, init=None) -> list[np.ndarray]:
        flat = [s.reshape([*shape, -1]) for s in srcs]
        initial = [self.ev(x, {}) for x in comb.identity_element] if init is None else init
        state = [np.broadcast_to(x, shape) for x in initial]
        for p in range(flat[0].shape[-1]):
            env = {}
            for k, (lhs, rhs) in enumerate(zip(comb.lhs, comb.rhs)):
                env[lhs.name] = state[k]
                env[rhs.name] = flat[k][..., p]
            state = [np.broadcast_to(self.ev(r, env), shape) for r in comb.result]
        return state


def high_precision_references(
    outputs, inputs, arrays, var_values, *, precision=128, max_precision=2048
):
    """Return MPFR arrays and the precision at which two evaluations agree.

    Agreement to 80 relative bits (with a floor 2**-16 times the target's
    smallest subnormal) is an empirical check, not a rigorous enclosure.
    Nonfinite results must agree in classification/sign.
    Failure to stabilize rejects the reference instead of weakening a tolerance.
    """
    if precision < 96 or max_precision < 2 * precision:
        raise ValueError("reference precision must be >= 96, with room for a doubled evaluation")

    def evaluate(bits):
        with numeric.context(bits):
            ref = _Reference(list(inputs), list(arrays), var_values)
            return [ref.tensor(t) for t in _tensors(outputs)]

    previous = evaluate(precision)
    while precision < max_precision:
        precision = min(2 * precision, max_precision)
        current = evaluate(precision)
        if numeric.references_agree(
            previous, current, precision, [str(t.dtype) for t in _tensors(outputs)]
        ):
            return current, precision
        previous = current
    raise Unsupported(f"MPFR reference did not stabilize by {max_precision} bits")


def references(outputs, inputs, arrays, var_values) -> list[np.ndarray]:
    """MPFR source evaluation, converted to float64 for NumPy-facing callers.

    The precision gate uses high_precision_references directly, so its error
    calculations do not discard reference bits by rounding to float64 first.
    """
    values, _ = high_precision_references(outputs, inputs, arrays, var_values)
    return [np.asarray(a, dtype=np.float64) for a in values]


# ---------------------------------------------------------------------------
# inputs: extents and the domain the program is defined on
# ---------------------------------------------------------------------------
def extent_names(outputs, inputs) -> tuple[set[str], set[str]]:
    """``(all, reduced)``: names of the symbolic extents, and of those some reduction runs over."""
    names: set[str] = set()
    reduced: set[str] = set()

    def note(e, reduce: bool) -> None:
        if isinstance(e, tir.Var):
            names.add(e.name)
            if reduce:
                reduced.add(e.name)

    for op in compute_ops(outputs):
        for iv in op.axis:
            note(iv.dom.extent, False)
        for iv in op.reduce_axis:
            note(iv.dom.extent, True)
    for t in [*inputs, *_tensors(outputs)]:
        for s in t.shape:
            note(s, False)
    return names, reduced


def _var_values(outputs, inputs, config: AccuracyConfig) -> dict[str, int]:
    names, reduced = extent_names(outputs, inputs)
    values = {n: (config.reduce_extent if n in reduced else config.other_extent) for n in names}
    if config.max_reference_elements < 1 or any(v < 1 for v in values.values()):
        raise ValueError("sampled dimensions and reference budget must be positive")
    shapes = [list(t.shape) for t in inputs]
    shapes.extend(
        [iv.dom.extent for iv in (*op.axis, *op.reduce_axis)] for op in compute_ops(outputs)
    )
    while True:
        ref = _Reference([], [], values)
        work = [int(np.prod([ref.extent(s) for s in shape], dtype=object)) for shape in shapes]
        largest = shapes[int(np.argmax(work))]
        if max(work) <= config.max_reference_elements:
            return values
        adjustable = [s.name for s in largest if isinstance(s, tir.Var) and values[s.name] > 1]
        if not adjustable:
            raise Unsupported("fixed reference shape exceeds max_reference_elements")
        name = max(adjustable, key=lambda n: (values[n], n))
        values[name] = max(1, values[name] // 2)


def positive_inputs(outputs, inputs, declared=()) -> list[bool]:
    """Per input: is it read under a ``log``, a ``sqrt`` or as a divisor?

    Such an input is only meaningful when positive (a weight, a variance); it
    is sampled that way so that the comparison happens on the program's domain.
    ``declared`` names inputs the caller knows to be positive where the
    program does not show it (a weight that is only ever summed).
    """
    hits: list = []

    def scan(e, inside: bool) -> None:
        if is_load(e):
            if inside and isinstance(load_tensor(e).op, te.PlaceholderOp):
                hits.append(load_tensor(e).op)
        elif isinstance(e, tir.Div):
            scan(e.a, inside)
            scan(e.b, True)
        elif isinstance(e, tir.Add | tir.Sub | tir.Mul | tir.Max | tir.Min):
            scan(e.a, inside)
            scan(e.b, inside)
        elif isinstance(e, tir.Cast):
            scan(e.value, inside)
        elif isinstance(e, Call):
            partial = e.op.name in ("prim.log", "prim.sqrt", "prim.rsqrt")
            if e.op.name == "prim.pow":  # a negative power divides by its base
                k = e.args[1]
                partial = isinstance(k, IntImm | FloatImm) and k.value < 0
            for a in e.args:  # exp maps every real to a positive value
                scan(a, partial or (inside and e.op.name != "prim.exp"))
        elif isinstance(e, te.Reduce):
            for x in e.source:
                scan(x, inside)

    for op in compute_ops(outputs):
        for body in op.body:
            scan(body, False)
    return [t.op.name in declared or any(t.op.same_as(h) for h in hits) for t in inputs]


def concrete_shape(shape, var_values: dict[str, int]) -> tuple[int, ...]:
    """A shape with its symbolic extents replaced by ``var_values`` (by name)."""
    ref = _Reference([], [], var_values)
    return tuple(ref.extent(s) for s in shape)


def sample_inputs(inputs, positive, extent, rng, shift: float = 0.0, scale: float = 1.0):
    """One array per input, ``shift + scale * N(0, 1)`` in the input's own dtype."""
    arrays = []
    for t, pos in zip(inputs, positive):
        a = shift + scale * rng.standard_normal(tuple(extent(s) for s in t.shape))
        if pos:
            a = np.abs(a) + 0.1
        arrays.append(a.astype(str(t.dtype)))
    return arrays


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def _compile(inputs, outputs):
    func = te.create_function([*inputs, *_tensors(outputs)])
    return tvm.compile(tvm_tirx.IRModule({"main": func}), target="llvm")


def _run(lib, arrays, out_shapes, dtypes) -> list[np.ndarray]:
    args = [tvm_tirx.runtime.tensor(a) for a in arrays]
    outs = [tvm_tirx.runtime.tensor(np.zeros(s, dtype=d)) for s, d in zip(out_shapes, dtypes)]
    lib["main"](*args, *outs)
    return [o.numpy().astype(np.float64) for o in outs]


class AccuracyGate:
    """Judges candidate programs against ``outputs`` over ``inputs``."""

    def __init__(self, outputs, inputs, config: AccuracyConfig | None = None) -> None:
        self.config = config = config or AccuracyConfig()
        outputs = _tensors(outputs)
        self.inputs = list(inputs)
        self.dtypes = [str(o.dtype) for o in outputs]
        # a static property is required of candidates only if the original has it
        self.required = [k for k, ops in static_issues(outputs).items() if not ops]
        self.var_values = _var_values(outputs, inputs, config)
        ref = _Reference(self.inputs, [], self.var_values)
        self.out_shapes = [tuple(ref.extent(s) for s in o.shape) for o in outputs]
        if (
            any(
                not np.isfinite(x) or x < 0
                for x in (config.factor, config.relative_eps, config.absolute_tolerance)
            )
            or config.perturbations < 0
        ):
            raise ValueError(
                "accuracy budgets and perturbation count must be finite and nonnegative"
            )
        positive = positive_inputs(outputs, self.inputs, config.positive)
        rng = np.random.default_rng(config.seed)
        signs = np.random.default_rng(config.seed + 1)  # perturbations: data stays fixed
        lib = _compile(self.inputs, outputs)
        families = [(name, shift, scale, False) for name, shift, scale in config.families]
        if config.magnitudes:
            info = np.finfo(self.dtypes[0])
            families.append(("huge", 0.0, 8.0 * float(np.sqrt(info.max)), True))
            families.append(("tiny", 0.0, 1.0e-4 * float(np.sqrt(info.tiny)), True))
        self.cases = []
        for name, shift, scale, magnitude in families:
            arrays = sample_inputs(self.inputs, positive, ref.extent, rng, shift, scale)
            want, bits = self.reference(outputs, arrays)
            if not all(np.all(numeric.finite(w)) for w in want):
                continue
            got = _run(lib, arrays, self.out_shapes, self.dtypes)
            original_metrics = tuple(
                numeric.errors(g, w, d, bits) for g, w, d in zip(got, want, self.dtypes)
            )
            original_error = max(e.normwise for e in original_metrics)
            if magnitude and original_error > config.magnitude_bar:
                continue  # This extra family is only required if the source handles it.
            sensitivity = [np.zeros(w.shape, dtype=object) for w in want]
            samples = 0
            for _ in range(config.perturbations):
                with np.errstate(all="ignore"):
                    moved = [
                        np.nextafter(
                            a, signs.choice((-np.inf, np.inf), size=a.shape).astype(a.dtype)
                        )
                        for a in arrays
                    ]
                if not all(np.all(np.isfinite(a)) for a in moved):
                    continue
                shifted, shifted_bits = self.reference(outputs, moved)
                bits = max(bits, shifted_bits)
                if not all(np.all(numeric.finite(w)) for w in shifted):
                    continue
                with numeric.context(bits + 64):
                    sensitivity = [
                        np.maximum(s, abs(a - b)) for s, a, b in zip(sensitivity, shifted, want)
                    ]
                samples += 1
            with numeric.context(bits + 64):
                sensitivity_error = max(
                    float(max(s.flat) / max(abs(w).flat)) if np.any(w) else float(max(s.flat))
                    for s, w in zip(sensitivity, want)
                )
            self.cases.append(
                (
                    name,
                    arrays,
                    want,
                    got,
                    original_metrics,
                    sensitivity,
                    sensitivity_error,
                    bits,
                    samples,
                )
            )

    def reference(self, outputs, arrays):
        return high_precision_references(
            outputs,
            self.inputs,
            arrays,
            self.var_values,
            precision=self.config.reference_precision,
            max_precision=self.config.reference_max_precision,
        )

    def check(self, candidates) -> AccuracyReport:
        candidates = _tensors(candidates)
        found = static_issues(candidates)
        bad = tuple((k, op) for k in self.required for op in found[k])
        if bad:
            return AccuracyReport(False, bad, ())
        if not self.cases:  # nothing to compare on: the original itself is not finite
            return AccuracyReport(False, (), (), "no input family with a finite reference")
        lib = _compile(self.inputs, candidates)
        fams = []
        for (
            name,
            arrays,
            want,
            orig,
            original_metrics,
            sensitivity,
            sampled,
            bits,
            samples,
        ) in self.cases:
            got = _run(lib, arrays, self.out_shapes, self.dtypes)
            metrics = tuple(
                numeric.errors(g, w, d, bits) for g, w, d in zip(got, want, self.dtypes)
            )
            ok = all(
                numeric.regression_check(
                    g,
                    o,
                    w,
                    s,
                    d,
                    factor=self.config.factor,
                    relative_eps=self.config.relative_eps,
                    absolute_tolerance=self.config.absolute_tolerance,
                    precision=bits,
                )
                for g, o, w, s, d in zip(got, orig, want, sensitivity, self.dtypes)
            )
            fams.append(
                FamilyError(
                    name,
                    max(e.normwise for e in original_metrics),
                    sampled,
                    max(e.normwise for e in metrics),
                    ok,
                    original_metrics,
                    metrics,
                    bits,
                    samples,
                )
            )
        return AccuracyReport(
            all(f.ok for f in fams), (), tuple(fams), extents=tuple(sorted(self.var_values.items()))
        )


# ---------------------------------------------------------------------------
# the static checks on a derived reducer, before any code is generated
# ---------------------------------------------------------------------------
def spec_issues(spec) -> set[str]:
    """Static issues of a synthesised reducer: its inputs and its (printed) merge.

    The merge is analysed in canonical form, as the gate sees it after lowering
    the generated ``comm_reducer``: a printed ``exp(-(max(a, b) - a) c)`` is
    ``exp(c a - c max(a, b))`` there. Both sides represent non-empty parts;
    the sign analysis starts from the leaf, not from an empty state.
    """
    merge = tuple(recanonicalize(m) for m in spec.merge_code)
    monoid = ir.raw_monoid(ir.dfull(spec.axis), 0, spec.leaves, merge, spec.leaves, 0)
    lower = _OpLower()
    signs = _Signs(lower)
    sites: list = []
    _sites(monoid, 0, {}, {}, signs, sites)
    out = set()
    for site in sites:
        if site[0] == "exp":
            _, arg, depth, binders = site
            if not bounded_above(arg, _Scope(lower, (), binders, depth)):
                out.add("exp")
        elif not _positive_domain(site[1], site[2], signs):
            out.add("domain")
    return out


__all__ = [
    "AccuracyConfig",
    "AccuracyGate",
    "AccuracyReport",
    "FamilyError",
    "bounded_above",
    "concrete_shape",
    "extent_names",
    "positive_inputs",
    "references",
    "sample_inputs",
    "spec_issues",
    "static_issues",
]
