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
"""Precision gate: a fused program may not be less accurate than the original.

A reducer is derived over the reals, where ``exp(s - max s) / Σ exp(s - max s)``
and ``exp(s) / Σ exp(s)`` are the same function and so are the two-pass and
the one-pass variance. In floating point they are not: the first overflows
where the second does not, the second cancels catastrophically. The gate
compares every candidate with the program it was derived from, never with
an absolute standard, so a program is only rejected for trouble the
original does not already have.

static
    Two syntactic properties, each required of a candidate only when the
    original has it. *Bounded exp*: every ``exp`` argument is bounded above
    (``c*x - c*max(.., x, ..)``, a max reduction over the axis ``x`` ranges
    over, or a constant), in elementwise ops, reducer inputs and merge
    functions alike. *Positive domains*: every divisor and ``log`` argument
    is provably positive (extents, sums of exponentials, counts), where the
    sign of a reducer state follows its serial fold. A merge dividing by a
    running sum of data of either sign fails the latter.
differential
    Original and candidate are compiled for LLVM in the working dtype and
    run on the same inputs from a few adversarial families (unit, large
    offset, wide range); both are measured against a float64 evaluation of
    the original's own expressions. The candidate fails if it produces a
    non-finite value where the original does not, or if it loses more than
    ``log10(factor)`` digits against the better of two yardsticks: the
    original's actual error, and the *inherent* error of the problem at the
    working precision -- how much the exact result moves when every input is
    perturbed by one rounding (``|dx| = eps |x|``), i.e. its condition number
    times ``eps``. The second yardstick admits the one-pass algorithms whose
    error is the backward-stable ``κ eps`` (Welford, online softmax) even when
    the original happens to be more accurate, while an expanded moment
    formula, whose error grows like ``κ² eps``, is still orders beyond it.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np

import tvm
from tvm import te
from tvm import tirx as tir
from tvm.ir import Call

from .symbolic import ir
from .symbolic.canonicalize import (
    Unsupported,
    contains_node,
    instantiate,
    is_constant,
    map_elems,
    mk_mul,
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
    families
        ``(name, shift, scale)``: every input is ``shift + scale * N(0, 1)``.
    factor, floor_ulps
        A candidate passes a family when ``err <= factor * max(err(original),
        err(inherent)) + floor_ulps * eps(dtype)``.
    perturbations
        Random one-ulp input perturbations used to estimate the inherent error.
    """

    reduce_extent: int = 256
    other_extent: int = 3
    families: tuple[tuple[str, float, float], ...] = (
        ("unit", 0.0, 1.0),
        ("offset", 1.0e4, 1.0),
        ("wide", 0.0, 10.0),
    )
    factor: float = 100.0
    floor_ulps: float = 64.0
    perturbations: int = 2
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class FamilyError:
    family: str
    original: float  # normwise relative error of the original vs the float64 reference
    inherent: float  # change of the exact result under one-ulp input perturbations
    candidate: float
    ok: bool


@dataclasses.dataclass(frozen=True)
class AccuracyReport:
    ok: bool
    # (kind, op name) for each static issue the original does not have; kind is
    # "exp" (argument not provably bounded above) or "domain" (a divisor or log
    # argument not provably positive)
    static: tuple[tuple[str, str], ...]
    families: tuple[FamilyError, ...]
    note: str = ""  # why no numeric comparison was possible

    @property
    def static_ok(self) -> bool:
        return not self.static

    def summary(self) -> str:
        head = "accurate" if self.ok else "REJECTED"
        what = {"exp": "unbounded exp", "domain": "divisor/log of unknown sign"}
        parts = [f"{what[kind]} in {op}" for kind, op in self.static]
        if self.note:
            parts.append(self.note)
        for f in self.families:
            mark = "" if f.ok else " (!)"
            parts.append(
                f"{f.family}: {f.candidate:.2g} (original {f.original:.2g}, "
                f"inherent {f.inherent:.2g}){mark}"
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
        if isinstance(e.producer.op, te.ComputeOp):
            tid = self.pseudo_id(e.producer)
            return ir.elem(tid, tuple(self.lower_index(i, env) for i in e.indices))
        return super().lower_load(e, env, level)

    def body(self, op, k: int) -> ir.SymExpr:
        env = {iv.var.name: ir.idx(f"i{n}") for n, iv in enumerate(op.axis)}
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
    c, terms = term_view(arg)
    if isinstance(c, float) or c > EXP_ARG_LIMIT:
        return c == float("-inf")
    neg = [(core, -coeff) for core, coeff in terms.items() if coeff < 0]
    used: set[int] = set()
    for core, coeff in terms.items():
        if coeff <= 0:
            continue
        for n, (ncore, ncoeff) in enumerate(neg):
            if n not in used and ncoeff == coeff and _dominates(ncore, core, scope):
                used.add(n)
                break
        else:
            return False
    return all(n in used or _nonneg(core) for n, (core, _) in enumerate(neg))


POS, NONNEG, ANY = 2, 1, 0  # sign lattice: > 0, >= 0, unknown


class _Signs:
    """Sign inference over op bodies, computed tensors and reducer states.

    The state of a ``comm_reducer`` is analysed along the serial fold that
    TE generates, ``s = identity; s = merge(s, leaf_j)``: the left operand of
    a merge is the running state (possibly still the identity), the right
    one a single leaf. A slot's sign is the greatest fixpoint of
    ``sign(leaf) /\\ sign(merge)`` under those assumptions.
    """

    def __init__(self, lower: _OpLower) -> None:
        self.lower = lower
        self.tensors: dict[int, int] = {}
        self.monoids: dict[ir.MonoidReduce, dict] = {}

    def tensor(self, tid: int) -> int:
        if tid not in self.tensors:
            self.tensors[tid] = ANY  # guards recursion
            d = self.lower.definition(tid)
            self.tensors[tid] = ANY if d is None else self.sign(d, {})
        return self.tensors[tid]

    def sign(self, e: ir.SymExpr, env: dict) -> int:
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
        bodies = [0] if isinstance(op.body[0], tir.Reduce) else range(len(op.body))
        out_keys = tuple(lower.dims.key(iv.dom.extent) for iv in op.axis)
        bad: set[str] = set()
        for k in bodies:
            try:
                sites: list = []
                _sites(lower.body(op, k), 0, {}, {}, signs, sites)
            except Unsupported:
                bad |= {"exp", "domain"}
                break
            for site in sites:
                if site[0] == "exp":
                    _, arg, depth, binders = site
                    if not bounded_above(arg, _Scope(lower, out_keys, binders, depth)):
                        bad.add("exp")
                elif signs.sign(site[1], site[2]) != POS:
                    bad.add("domain")
        for kind in sorted(bad):
            issues[kind].append(op.name)
    return issues


# ---------------------------------------------------------------------------
# differential: float64 reference by interpreting the TE expressions
# ---------------------------------------------------------------------------
class _Reference:
    """Evaluate a TE graph in float64 exactly as written (no rewriting)."""

    def __init__(self, inputs: list[te.Tensor], arrays: list[np.ndarray], var_values):
        self.values: list[tuple[object, list[np.ndarray]]] = [
            (t.op, [np.asarray(a, dtype=np.float64)]) for t, a in zip(inputs, arrays)
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
        if isinstance(e, tir.IntImm):
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
        if isinstance(e, tir.FloatImm | tir.IntImm):
            # as written: a reducer identity is the dtype's lowest finite value, and
            # ``0 * lowest`` is 0 where ``0 * -inf`` would be NaN
            return float(e.value)
        if isinstance(e, tir.Var):
            if e.name in env:
                return env[e.name]
            return float(self.vars[e.name])
        if isinstance(e, tir.Cast):
            return np.asarray(self.ev(e.value, env), dtype=np.float64)
        binary = {
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
        if isinstance(e, Call):
            unary = {"tirx.exp": np.exp, "tirx.sqrt": np.sqrt, "tirx.log": np.log}
            name = e.op.name
            if name in unary:
                return unary[name](self.ev(e.args[0], env))
            if name == "tirx.rsqrt":
                return 1.0 / np.sqrt(self.ev(e.args[0], env))
            raise Unsupported(f"intrinsic {name}")
        if isinstance(e, tir.ProducerLoad):
            arr = self.tensor(e.producer)
            return arr[tuple(self.index(i, env) for i in e.indices)]
        raise Unsupported(f"expression {type(e).__name__}")

    def compute(self, op) -> list[np.ndarray]:
        shape = [self.extent(iv.dom.extent) for iv in op.axis]
        first = op.body[0]
        raxes = list(first.axis) if isinstance(first, tir.Reduce) else []
        full = shape + [self.extent(iv.dom.extent) for iv in raxes]
        nd = len(full)
        env = {}
        for pos, iv in enumerate([*op.axis, *raxes]):
            env[iv.var.name] = np.arange(full[pos]).reshape(
                [1] * pos + [full[pos]] + [1] * (nd - pos - 1)
            )
        with np.errstate(all="ignore"):
            if not raxes:
                return [np.broadcast_to(self.ev(b, env), shape).copy() for b in op.body]
            cond = first.condition
            if not (isinstance(cond, tir.IntImm) and int(cond.value) == 1):
                raise Unsupported("conditional reduction")
            srcs = [np.broadcast_to(self.ev(s, env), full) for s in first.source]
            dims = tuple(range(len(shape), nd))
            kind = classify_combiner(first.combiner)
            if kind == "sum":
                states = [srcs[0].sum(axis=dims)]
            elif kind == "max":
                states = [srcs[0].max(axis=dims)]
            else:
                states = self._fold(first.combiner, srcs, shape)
        return [np.asarray(states[int(b.value_index)]).copy() for b in op.body]

    def _fold(self, comb, srcs, shape) -> list[np.ndarray]:
        flat = [s.reshape([*shape, -1]) for s in srcs]
        state = [np.broadcast_to(self.ev(x, {}), shape) for x in comb.identity_element]
        for p in range(flat[0].shape[-1]):
            env = {}
            for k, (lhs, rhs) in enumerate(zip(comb.lhs, comb.rhs)):
                env[lhs.name] = state[k]
                env[rhs.name] = flat[k][..., p]
            state = [np.broadcast_to(self.ev(r, env), shape) for r in comb.result]
        return state


def references(outputs, inputs, arrays, var_values) -> list[np.ndarray]:
    """float64 values of ``outputs`` computed with the program's own expressions."""
    ref = _Reference(list(inputs), list(arrays), var_values)
    return [ref.tensor(t) for t in _tensors(outputs)]


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
    return {n: (config.reduce_extent if n in reduced else config.other_extent) for n in names}


def positive_inputs(outputs, inputs) -> list[bool]:
    """Per input: is it read under a ``log``, a ``sqrt`` or as a divisor?

    Such an input is only meaningful when positive (a weight, a variance); it
    is sampled that way so that the comparison happens on the program's domain.
    """
    hits: list = []

    def scan(e, inside: bool) -> None:
        if isinstance(e, tir.ProducerLoad):
            if inside and isinstance(e.producer.op, te.PlaceholderOp):
                hits.append(e.producer.op)
        elif isinstance(e, tir.Div):
            scan(e.a, inside)
            scan(e.b, True)
        elif isinstance(e, tir.Add | tir.Sub | tir.Mul | tir.Max | tir.Min):
            scan(e.a, inside)
            scan(e.b, inside)
        elif isinstance(e, tir.Cast):
            scan(e.value, inside)
        elif isinstance(e, Call):
            partial = e.op.name in ("tirx.log", "tirx.sqrt", "tirx.rsqrt")
            for a in e.args:  # exp maps every real to a positive value
                scan(a, partial or (inside and e.op.name != "tirx.exp"))
        elif isinstance(e, tir.Reduce):
            for x in e.source:
                scan(x, inside)

    for op in compute_ops(outputs):
        for body in op.body:
            scan(body, False)
    return [any(t.op.same_as(h) for h in hits) for t in inputs]


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
    func = te.create_prim_func([*inputs, *_tensors(outputs)])
    return tvm.compile(tvm.IRModule({"main": func}), target="llvm")


def _run(lib, arrays, out_shapes, dtypes) -> list[np.ndarray]:
    args = [tvm.runtime.tensor(a) for a in arrays]
    outs = [tvm.runtime.tensor(np.zeros(s, dtype=d)) for s, d in zip(out_shapes, dtypes)]
    lib["main"](*args, *outs)
    return [o.numpy().astype(np.float64) for o in outs]


def _rel_err(got: list[np.ndarray], ref: list[np.ndarray]) -> float:
    """Largest normwise relative error over the outputs."""
    worst = 0.0
    for g, r in zip(got, ref):
        if not np.all(np.isfinite(g)):
            return float("inf")
        scale = float(np.max(np.abs(r))) if r.size else 0.0
        err = float(np.max(np.abs(g - r))) if r.size else 0.0
        worst = max(worst, err / scale if scale > 0 else err)
    return worst


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
        eps = max(float(np.finfo(d).eps) for d in self.dtypes)
        self.floor = config.floor_ulps * eps
        positive = positive_inputs(outputs, self.inputs)
        rng = np.random.default_rng(config.seed)
        signs = np.random.default_rng(config.seed + 1)  # perturbations: data stays fixed
        lib = _compile(self.inputs, outputs)
        # per family: (name, input arrays, float64 reference, original output, original error)
        self.cases = []
        for name, shift, scale in config.families:
            arrays = sample_inputs(self.inputs, positive, ref.extent, rng, shift, scale)
            want = references(outputs, self.inputs, arrays, self.var_values)
            if not all(np.all(np.isfinite(w)) for w in want):
                continue
            inherent = 0.0
            for _ in range(config.perturbations):
                moved = [
                    a.astype(np.float64)
                    * (1.0 + float(np.finfo(a.dtype).eps) * signs.choice((-1.0, 1.0), size=a.shape))
                    for a in arrays
                ]
                shifted = references(outputs, self.inputs, moved, self.var_values)
                inherent = max(inherent, _rel_err(shifted, want))
            got = _run(lib, arrays, self.out_shapes, self.dtypes)
            self.cases.append((name, arrays, want, got, _rel_err(got, want), inherent))

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
        for name, arrays, want, orig, orig_err, inherent in self.cases:
            got = _run(lib, arrays, self.out_shapes, self.dtypes)
            err = _rel_err(got, want)
            new_nonfinite = any(
                bool(np.any(~np.isfinite(g) & np.isfinite(o))) for g, o in zip(got, orig)
            )
            bar = self.config.factor * max(orig_err, inherent) + self.floor
            ok = not new_nonfinite and err <= bar
            fams.append(FamilyError(name, orig_err, inherent, err, ok))
        return AccuracyReport(all(f.ok for f in fams), (), tuple(fams))


# ---------------------------------------------------------------------------
# the static checks on a derived reducer, before any code is generated
# ---------------------------------------------------------------------------
def spec_issues(spec) -> set[str]:
    """Static issues of a synthesised reducer: its inputs and its (printed) merge.

    The merge is analysed in canonical form, as the gate sees it after lowering
    the generated ``comm_reducer``: a printed ``exp(-(max(a, b) - a) c)`` is
    ``exp(c a - c max(a, b))`` there.
    """
    merge = tuple(recanonicalize(m) for m in spec.merge_code)
    monoid = ir.raw_monoid(ir.dfull(spec.axis), 0, spec.leaves, merge, spec.identity, 0)
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
        elif signs.sign(site[1], site[2]) != POS:
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
