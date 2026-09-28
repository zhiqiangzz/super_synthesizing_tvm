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
"""Canonicalising constructors for the symbolic IR.

Every ``mk_*`` function takes *canonical* children and returns the canonical
node for the operation, applying (to a fixpoint) the rewrite theory:

* exact constant folding, ``+-inf`` absorption;
* n-ary flattening of ``+``, ``*``, ``max`` with like-term / like-factor
  collection and a fixed argument order;
* ``exp(x) * exp(y) = exp(x + y)``, ``exp(x)^p = exp(p x)``,
  ``exp(log x) = x``, ``log(exp x) = x``, ``log(x y) = log x + log y`` and
  ``log(x^p) = p log x`` for provably positive ``x``, ``y``;
* ``max(f + c, g + c) = max(f, g) + c`` and ``max(c f, c g) = c max(f, g)``
  for ``c > 0``;
* for ``sum`` reductions: ``Σ(f + g) = Σf + Σg``, ``Σ c f = c Σ f`` and
  ``Σ exp(x_j + y) = exp(y) Σ exp(x_j)`` whenever ``c``/``y`` do not depend
  on the bound index, ``Σ_D c = c |D|``;
* for ``max`` reductions: ``max_j (f + c) = max_j f + c``,
  ``max_j c f = c max_j f`` (``c > 0``), ``max_j max(f, g) = max(max_j f,
  max_j g)``, ``max_j exp(f) = exp(max_j f)``;
* domain algebra: ``Σ_{A u B} = Σ_A + Σ_B``, ``max_{A u B} = max(max_A,
  max_B)``, ``Σ_∅ = 0``, ``max_∅ = -inf``, ``Σ_{{p}} f = f[p]``.

Because the constructors are the only way to build nodes, ``a is b`` decides
equality modulo this theory.
"""

from __future__ import annotations

from fractions import Fraction

from . import ir
from .ir import (
    NEG_INF,
    NEG_INF_C,
    ONE,
    POS_INF,
    ZERO,
    Add,
    Atom,
    BIdx,
    Card,
    Const,
    DEmpty,
    Domain,
    DSingleton,
    DUnion,
    Elem,
    Exp,
    Idx,
    IndexExpr,
    Log,
    Max,
    MonoidReduce,
    Mul,
    Node,
    Pow,
    Reduce,
    ShapeSym,
    StateVar,
    SymExpr,
    const,
)


class Unsupported(Exception):
    """Raised when a TE construct has no symbolic semantics in this IR."""


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def term_view(e: SymExpr) -> tuple[ir.Number, dict[SymExpr, Fraction]]:
    """View ``e`` as ``const + Σ coeff_i * core_i`` (cores are non-Add, coefficient-free)."""
    c: ir.Number = Fraction(0)
    terms: dict[SymExpr, Fraction] = {}
    args = e.args if isinstance(e, Add) else (e,)
    for a in args:
        if isinstance(a, Const):
            c = _add_num(c, a.value)
        else:
            coeff, core = _split_coeff(a)
            terms[core] = terms.get(core, Fraction(0)) + coeff
    return c, terms


def _split_coeff(a: SymExpr) -> tuple[Fraction, SymExpr]:
    if isinstance(a, Mul) and isinstance(a.args[0], Const) and not ir.is_inf(a.args[0]):
        rest = a.args[1:]
        core = rest[0] if len(rest) == 1 else ir.raw_mul(rest)
        return a.args[0].value, core
    return Fraction(1), a


def factor_view(e: SymExpr) -> tuple[ir.Number, dict[SymExpr, Fraction], SymExpr | None]:
    """View ``e`` as ``coeff * Π base_i^{p_i} * exp(x)`` (``x`` may be None)."""
    coeff: ir.Number = Fraction(1)
    factors: dict[SymExpr, Fraction] = {}
    exp_arg: SymExpr | None = None
    args = e.args if isinstance(e, Mul) else (e,)
    for a in args:
        if isinstance(a, Const):
            coeff = _mul_num(coeff, a.value)
        elif isinstance(a, Pow):
            factors[a.base] = factors.get(a.base, Fraction(0)) + a.exponent
        elif isinstance(a, Exp):
            exp_arg = a.arg if exp_arg is None else mk_add(exp_arg, a.arg)
        else:
            factors[a] = factors.get(a, Fraction(0)) + 1
    return coeff, factors, exp_arg


def _add_num(a: ir.Number, b: ir.Number) -> ir.Number:
    if isinstance(a, float) or isinstance(b, float):
        r = float(a) + float(b)
        if r != r:  # inf - inf
            raise Unsupported("inf - inf")
        return r
    return a + b


def _mul_num(a: ir.Number, b: ir.Number) -> ir.Number:
    if isinstance(a, float) or isinstance(b, float):
        if a == 0 or b == 0:
            return Fraction(0)
        return float(a) * float(b)
    return a * b


def _num_const(v: ir.Number) -> Const:
    return const(v)


# ---------------------------------------------------------------------------
# Scalar constructors
# ---------------------------------------------------------------------------
def mk_add(*args: SymExpr) -> SymExpr:
    c: ir.Number = Fraction(0)
    terms: dict[SymExpr, Fraction] = {}
    for a in args:
        if isinstance(a, Add):
            ac, at = term_view(a)
            c = _add_num(c, ac)
            for core, k in at.items():
                terms[core] = terms.get(core, Fraction(0)) + k
        elif isinstance(a, Const):
            c = _add_num(c, a.value)
        else:
            coeff, core = _split_coeff(a)
            terms[core] = terms.get(core, Fraction(0)) + coeff
    if isinstance(c, float):
        return _num_const(c)  # +-inf absorbs every finite term
    out = []
    for core, k in terms.items():
        if k == 0:
            continue
        out.append(mk_mul(const(k), core) if k != 1 else core)
    if not out:
        return _num_const(c)
    if c != 0:
        out.append(_num_const(c))
    if len(out) == 1:
        return out[0]
    out.sort()
    return ir.raw_add(tuple(out))


def mk_neg(a: SymExpr) -> SymExpr:
    return mk_mul(ir.MINUS_ONE, a)


def mk_sub(a: SymExpr, b: SymExpr) -> SymExpr:
    return mk_add(a, mk_neg(b))


def mk_mul(*args: SymExpr) -> SymExpr:
    coeff: ir.Number = Fraction(1)
    factors: dict[SymExpr, Fraction] = {}
    exp_terms: list[SymExpr] = []

    def absorb(a: SymExpr) -> None:
        nonlocal coeff
        if isinstance(a, Const):
            coeff = _mul_num(coeff, a.value)
        elif isinstance(a, Mul):
            for x in a.args:
                absorb(x)
        elif isinstance(a, Pow):
            factors[a.base] = factors.get(a.base, Fraction(0)) + a.exponent
        elif isinstance(a, Exp):
            exp_terms.append(a.arg)
        else:
            factors[a] = factors.get(a, Fraction(0)) + 1

    for a in args:
        absorb(a)
    if coeff == 0:
        return ZERO
    out: list[SymExpr] = []
    for base, p in factors.items():
        if p == 0:
            continue
        f = mk_pow(base, p)
        if isinstance(f, Const):
            coeff = _mul_num(coeff, f.value)
        elif isinstance(f, Mul):  # pow distributed over a product (should not happen)
            for x in f.args:
                absorb(x)
        elif isinstance(f, Exp):
            exp_terms.append(f.arg)
        else:
            out.append(f)
    if exp_terms:
        e = mk_exp(mk_add(*exp_terms))
        if isinstance(e, Const):
            coeff = _mul_num(coeff, e.value)
        else:
            out.append(e)
    if coeff == 0:
        return ZERO
    if isinstance(coeff, float) and all(positive(f) for f in out):
        return _num_const(coeff)  # +-inf times a positive quantity
    # Distribute over a sum factor: x * (a + b) = x a + x b (polynomial normal form).
    for i, f in enumerate(out):
        if isinstance(f, Add):
            others = out[:i] + out[i + 1 :] + [_num_const(coeff)]
            return mk_add(*[mk_mul(t, *others) for t in f.args])
    if not out:
        return _num_const(coeff)
    if coeff != 1:
        out.append(_num_const(coeff))
    if len(out) == 1:
        return out[0]
    out.sort()
    return ir.raw_mul(tuple(out))


def mk_div(a: SymExpr, b: SymExpr) -> SymExpr:
    return mk_mul(a, mk_pow(b, Fraction(-1)))


def mk_pow(base: SymExpr, exponent) -> SymExpr:
    p = Fraction(exponent)
    if p == 0:
        return ONE
    if p == 1:
        return base
    if isinstance(base, Const):
        v = base.value
        if isinstance(v, float):
            if p > 0:
                return base if v > 0 or p.numerator % 2 == 1 else ir.POS_INF_C
            return ZERO
        if p.denominator == 1:
            if v == 0:
                if p < 0:
                    raise Unsupported("division by zero")
                return ZERO
            return const(v**p.numerator)
        if v == 0:
            return ZERO
        if v == 1:
            return ONE
        root = _exact_root(v, p.denominator)
        if root is not None:
            return const(root**p.numerator)
        return ir.raw_pow(base, p)
    if isinstance(base, Pow):
        return mk_pow(base.base, base.exponent * p)
    if isinstance(base, Mul):
        return mk_mul(*[mk_pow(a, p) for a in base.args])
    if isinstance(base, Add) and p.denominator == 1 and 2 <= p <= 4:
        # Polynomial normal form: expand (a + b)^n term by term (never through
        # mk_mul(base, base), which would collect the factor back into a Pow).
        out: SymExpr = base
        for _ in range(p.numerator - 1):
            terms = out.args if isinstance(out, Add) else (out,)
            out = mk_add(*[mk_mul(x, y) for x in terms for y in base.args])
        return out
    if isinstance(base, Exp):
        return mk_exp(mk_mul(const(p), base.arg))
    return ir.raw_pow(base, p)


def _exact_root(v: Fraction, n: int) -> Fraction | None:
    if v < 0:
        return None

    def iroot(x: int) -> int | None:
        r = round(x ** (1.0 / n))
        for cand in (r - 1, r, r + 1):
            if cand >= 0 and cand**n == x:
                return cand
        return None

    num, den = iroot(v.numerator), iroot(v.denominator)
    if num is None or den is None:
        return None
    return Fraction(num, den)


def mk_sqrt(x: SymExpr) -> SymExpr:
    return mk_pow(x, Fraction(1, 2))


def mk_exp(x: SymExpr) -> SymExpr:
    if isinstance(x, Const):
        v = x.value
        if v == 0:
            return ONE
        if isinstance(v, float):
            return ZERO if v < 0 else ir.POS_INF_C
        return ir.raw_exp(x)
    if isinstance(x, Log):
        return x.arg
    if isinstance(x, Add):
        # exp(k log u + rest) = u^k exp(rest)
        c, terms = term_view(x)
        logs = [(core, k) for core, k in terms.items() if isinstance(core, Log)]
        if logs:
            rest = [
                mk_mul(const(k), core) for core, k in terms.items() if not isinstance(core, Log)
            ]
            factors = [mk_pow(core.arg, k) for core, k in logs]
            return mk_mul(*factors, mk_exp(mk_add(*rest, const(c))))
    return ir.raw_exp(x)


def mk_log(x: SymExpr) -> SymExpr:
    if isinstance(x, Const):
        v = x.value
        if v == 1:
            return ZERO
        if v == 0:
            return NEG_INF_C
        if isinstance(v, float) and v > 0:
            return ir.POS_INF_C
        return ir.raw_log(x)
    if isinstance(x, Exp):
        return x.arg
    if isinstance(x, Pow) and positive(x.base):
        return mk_mul(const(x.exponent), mk_log(x.base))
    if isinstance(x, Mul) and all(positive(a) for a in x.args):
        return mk_add(*[mk_log(a) for a in x.args])
    return ir.raw_log(x)


def mk_max(*args: SymExpr) -> SymExpr:
    flat: list[SymExpr] = []
    for a in args:
        if isinstance(a, Max):
            flat.extend(a.args)
        else:
            flat.append(a)
    best: ir.Number | None = None
    items: dict[SymExpr, None] = {}
    for a in flat:
        if isinstance(a, Const):
            if best is None or a.value > best:
                best = a.value
        else:
            items[a] = None
    if best is not None and isinstance(best, float):
        if best > 0:
            return ir.POS_INF_C
        best = None  # -inf is the identity of max
    if best is not None:
        items[_num_const(best)] = None
    if not items:
        return NEG_INF_C
    if len(items) == 1:
        return next(iter(items))
    xs = list(items)
    # Common additive part: max(f + c, g + c) = max(f, g) + c.
    views = [term_view(x) for x in xs]
    common: dict[SymExpr, Fraction] = {}
    first_c, first_terms = views[0]
    for core, k in first_terms.items():
        if all(t.get(core) == k for _, t in views[1:]):
            common[core] = k
    c0 = min(c for c, _ in views)
    if common or c0 != 0:
        shifted = []
        sub = mk_add(*([mk_mul(const(k), core) for core, k in common.items()] + [const(c0)]))
        for x in xs:
            shifted.append(mk_sub(x, sub))
        return mk_add(mk_max(*shifted), sub)
    # Common positive multiplicative part: max(c f, c g) = c max(f, g), c > 0.
    fviews = [factor_view(x) for x in xs]
    coeffs = [fv[0] for fv in fviews]
    pull: list[SymExpr] = []
    if all(isinstance(c, Fraction) and c == coeffs[0] and c > 0 for c in coeffs) and coeffs[0] != 1:
        pull.append(const(coeffs[0]))
    for base, p in fviews[0][1].items():
        if positive(base) and all(fv[1].get(base) == p for fv in fviews[1:]):
            pull.append(mk_pow(base, p))
    if fviews[0][2] is not None and all(fv[2] is fviews[0][2] for fv in fviews[1:]):
        pull.append(mk_exp(fviews[0][2]))
    if pull:
        factor = mk_mul(*pull)
        inv = mk_pow(factor, Fraction(-1))
        return mk_mul(factor, mk_max(*[mk_mul(x, inv) for x in xs]))
    xs.sort()
    return ir.raw_max(tuple(xs))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
_REDUCE_CACHE: dict[tuple, SymExpr] = {}


def mk_reduce(kind: str, domain: Domain, level: int, body: SymExpr) -> SymExpr:
    key = (kind, domain, level, body)
    hit = _REDUCE_CACHE.get(key)
    if hit is not None:
        return hit
    out = _mk_reduce(kind, domain, level, body)
    _REDUCE_CACHE[key] = out
    return out


def _mk_reduce(kind: str, domain: Domain, level: int, body: SymExpr) -> SymExpr:
    if isinstance(domain, DUnion):
        a = mk_reduce(kind, domain.a, level, body)
        b = mk_reduce(kind, domain.b, level, body)
        return mk_add(a, b) if kind == "sum" else mk_max(a, b)
    if isinstance(domain, DEmpty):
        return ZERO if kind == "sum" else NEG_INF_C
    if isinstance(domain, DSingleton):
        inst = subst(body, {ir.bidx(level): domain.point})
        return shift(inst, level + 1, -1)
    if not body.uses_level(level):
        # Body does not depend on the bound index.
        out = shift(body, level + 1, -1)
        if kind == "sum":
            return mk_mul(out, mk_card(domain))
        return out  # domains are non-empty
    if kind == "sum":
        return _mk_sum(domain, level, body)
    return _mk_rmax(domain, level, body)


def _partition_terms(body: SymExpr, level: int) -> tuple[list[SymExpr], list[SymExpr]]:
    dep, free = [], []
    for a in body.args if isinstance(body, Add) else (body,):
        (dep if a.uses_level(level) else free).append(a)
    return dep, free


def _mk_sum(domain: Domain, level: int, body: SymExpr) -> SymExpr:
    if isinstance(body, Add):
        return mk_add(*[mk_reduce("sum", domain, level, t) for t in body.args])
    coeff, factors, exp_arg = factor_view(body)
    inner: list[SymExpr] = []
    outer: list[SymExpr] = [_num_const(coeff)]
    for base, p in factors.items():
        f = mk_pow(base, p)
        (inner if f.uses_level(level) else outer).append(f)
    if exp_arg is not None:
        dep, free = _partition_terms(exp_arg, level)
        if free:
            outer.append(mk_exp(mk_add(*free)))
        if dep:
            inner.append(mk_exp(mk_add(*dep)))
    if not inner:  # cannot happen (body uses the level) but keep total
        return mk_mul(shift(body, level + 1, -1), ir.card(domain))
    outer_shifted = [shift(x, level + 1, -1) for x in outer]
    red = ir.raw_reduce("sum", domain, level, mk_mul(*inner))
    return mk_mul(*outer_shifted, red)


def _mk_rmax(domain: Domain, level: int, body: SymExpr) -> SymExpr:
    if isinstance(body, Max):
        return mk_max(*[mk_reduce("max", domain, level, a) for a in body.args])
    if isinstance(body, Exp):
        return mk_exp(mk_reduce("max", domain, level, body.arg))
    if isinstance(body, Add):
        dep, free = _partition_terms(body, level)
        if free:
            out = shift(mk_add(*free), level + 1, -1)
            return mk_add(out, mk_reduce("max", domain, level, mk_add(*dep)))
    coeff, factors, exp_arg = factor_view(body)
    inner: list[SymExpr] = []
    outer: list[SymExpr] = []
    if isinstance(coeff, Fraction) and coeff > 0:
        if coeff != 1:
            outer.append(_num_const(coeff))
    else:
        inner.append(_num_const(coeff))
    for base, p in factors.items():
        f = mk_pow(base, p)
        if not f.uses_level(level) and positive(f):
            outer.append(f)
        else:
            inner.append(f)
    if exp_arg is not None:
        dep, free = _partition_terms(exp_arg, level)
        if free:
            outer.append(mk_exp(mk_add(*free)))
        if dep:
            inner.append(mk_exp(mk_add(*dep)))
    if outer:
        outer_shifted = [shift(x, level + 1, -1) for x in outer]
        return mk_mul(*outer_shifted, mk_reduce("max", domain, level, mk_mul(*inner)))
    return ir.raw_reduce("max", domain, level, body)


def mk_card(domain: Domain) -> SymExpr:
    """``|D|``: the extent for a full axis, additive over disjoint unions."""
    if isinstance(domain, DUnion):
        return mk_add(mk_card(domain.a), mk_card(domain.b))
    if isinstance(domain, DEmpty):
        return ZERO
    if isinstance(domain, DSingleton):
        return ONE
    if isinstance(domain, ir.DFull):
        ext = ir.extent_of(domain.axis)
        if ext is not None:
            return ext
    return ir.card(domain)


def mk_monoid(domain, level, leaf, merge, identity, slot) -> SymExpr:
    if isinstance(domain, DEmpty):
        return identity[slot]
    return ir.raw_monoid(domain, level, leaf, merge, identity, slot)


# ---------------------------------------------------------------------------
# Structural transformations (all rebuild through the canonical constructors)
# ---------------------------------------------------------------------------
def rebuild(e: Node, children: dict[Node, Node]) -> Node:
    """Rebuild ``e`` with some direct children replaced (canonicalising)."""
    if isinstance(e, Add):
        return mk_add(*[children.get(a, a) for a in e.args])
    if isinstance(e, Mul):
        return mk_mul(*[children.get(a, a) for a in e.args])
    if isinstance(e, Max):
        return mk_max(*[children.get(a, a) for a in e.args])
    if isinstance(e, Pow):
        return mk_pow(children.get(e.base, e.base), e.exponent)
    if isinstance(e, Exp):
        return mk_exp(children.get(e.arg, e.arg))
    if isinstance(e, Log):
        return mk_log(children.get(e.arg, e.arg))
    if isinstance(e, Reduce):
        return mk_reduce(
            e.kind, children.get(e.domain, e.domain), e.level, children.get(e.body, e.body)
        )
    if isinstance(e, MonoidReduce):
        return mk_monoid(
            children.get(e.domain, e.domain),
            e.level,
            tuple(children.get(x, x) for x in e.leaf),
            tuple(children.get(x, x) for x in e.merge),
            tuple(children.get(x, x) for x in e.identity),
            e.slot,
        )
    if isinstance(e, Elem):
        return ir.elem(e.tensor, tuple(children.get(i, i) for i in e.indices))
    if isinstance(e, Card):
        return mk_card(children.get(e.domain, e.domain))
    if isinstance(e, DUnion):
        return ir.dunion(children.get(e.a, e.a), children.get(e.b, e.b))
    if isinstance(e, DSingleton):
        return ir.dsingleton(e.axis, children.get(e.point, e.point))
    return e


def transform(e: Node, fn, memo: dict | None = None) -> Node:
    """Bottom-up rewrite: ``fn(node)`` returns a replacement or ``None`` to recurse."""
    if memo is None:
        memo = {}
    hit = memo.get(e)
    if hit is not None:
        return hit
    out = fn(e)
    if out is None:
        kids = e.children()
        if kids:
            new = {}
            for k in kids:
                nk = transform(k, fn, memo)
                if nk is not k:
                    new[k] = nk
            out = rebuild(e, new) if new else e
        else:
            out = e
    memo[e] = out
    return out


def subst(e: Node, mapping: dict[Node, Node]) -> Node:
    """Replace whole sub-nodes (typically leaves: Idx, BIdx, Atom, StateVar, Elem)."""
    if not mapping:
        return e
    return transform(e, lambda n: mapping.get(n))


def shift(e: Node, cutoff: int, delta: int) -> Node:
    """Shift every bound level ``>= cutoff`` by ``delta`` (binders included)."""
    if delta == 0:
        return e
    key = ("shift", cutoff, delta)
    hit = e.cache.get(key)
    if hit is not None:
        return hit

    def fn(n: Node):
        if isinstance(n, BIdx):
            return ir.bidx(n.level + delta) if n.level >= cutoff else n
        if isinstance(n, Reduce) and n.level >= cutoff:
            return mk_reduce(
                n.kind,
                shift(n.domain, cutoff, delta),
                n.level + delta,
                shift(n.body, cutoff, delta),
            )
        if isinstance(n, MonoidReduce) and n.level >= cutoff:
            return mk_monoid(
                shift(n.domain, cutoff, delta),
                n.level + delta,
                tuple(shift(x, cutoff, delta) for x in n.leaf),
                n.merge,
                n.identity,
                n.slot,
            )
        if not n.children():
            return n
        return None

    out = transform(e, fn)
    e.cache[key] = out
    return out


def max_binder_level(e: Node) -> int:
    """Largest ``Reduce.level`` inside ``e`` (``-1`` if none)."""
    best = -1
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, Reduce | MonoidReduce):
            best = max(best, n.level)
        stack.extend(n.children())
    return best


def subst_domain(e: Node, mapping: dict[Domain, Domain]) -> Node:
    return transform(e, lambda n: mapping.get(n) if isinstance(n, Domain) else None)


def instantiate(body: SymExpr, index_map: dict[Idx, IndexExpr], level: int) -> SymExpr:
    """Inline a tensor body (levels from 0) at binder depth ``level`` with new indices."""
    shifted = shift(body, 0, level) if level else body
    return subst(shifted, dict(index_map))


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------
def positive(e: SymExpr) -> bool:
    """Syntactic proof that ``e > 0`` for all inputs (extents are >= 1)."""
    key = "positive"
    hit = e.cache.get(key)
    if hit is not None:
        return hit
    if isinstance(e, Const):
        r = e.value > 0
    elif isinstance(e, ShapeSym | Card | Exp):
        r = True
    elif isinstance(e, Pow):
        r = positive(e.base)
    elif isinstance(e, Mul | Add):
        r = all(positive(a) for a in e.args)
    elif isinstance(e, Max):
        r = any(positive(a) for a in e.args)
    elif isinstance(e, Reduce):
        r = positive(e.body)
    else:
        r = False
    e.cache[key] = r
    return r


def contains_node(e: Node, pred) -> bool:
    stack = [e]
    while stack:
        n = stack.pop()
        if pred(n):
            return True
        stack.extend(n.children())
    return False


def reductions(e: Node) -> list[Reduce]:
    out = []
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, Reduce):
            out.append(n)
        stack.extend(n.children())
    return out


def leaves(e: Node, cls=(Elem, ShapeSym, Atom, StateVar, Idx, BIdx)) -> set[Node]:
    out = set()
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, cls):
            out.add(n)
        stack.extend(n.children())
    return out


def elems(e: Node) -> set[int]:
    return {n.tensor for n in leaves(e, (Elem,))}


__all__ = [
    "NEG_INF",
    "POS_INF",
    "Unsupported",
    "contains_node",
    "elems",
    "factor_view",
    "instantiate",
    "leaves",
    "max_binder_level",
    "mk_add",
    "mk_card",
    "mk_div",
    "mk_exp",
    "mk_log",
    "mk_max",
    "mk_monoid",
    "mk_mul",
    "mk_neg",
    "mk_pow",
    "mk_reduce",
    "mk_sqrt",
    "mk_sub",
    "positive",
    "rebuild",
    "reductions",
    "shift",
    "subst",
    "subst_domain",
    "term_view",
    "transform",
]
