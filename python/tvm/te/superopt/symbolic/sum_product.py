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
"""Sum-product (einsum) normal form: equality modulo the order of contractions.

The canonical form keeps every sum where the program put it and pulls the
factors that do not depend on its index out of it. Two contraction orders of
one product are then different trees: ``Σ_n (Σ_k A B) C`` for ``(A B) C``
against ``Σ_k A (Σ_n B C)`` for ``A (B C)``, or a sum over all paths of a
chain against its variable-eliminated form. Here all sums are brought to the
front instead -- products of sums are multiplied out,
``(Σ_a f)(Σ_b g) = Σ_a Σ_b f g`` -- and every resulting term
``c · Σ_{b1..bk} Π factors`` is compared up to a renaming of its bound
indices. Only ``sum`` reductions over whole axes are opened; everything else
(``max``, ``exp``, divisions, reducer outputs) is an opaque factor.
"""

from __future__ import annotations

import itertools
from collections import Counter
from fractions import Fraction

from . import ir
from .canonicalize import shift, subst

MAX_TERMS = 4096
MAX_BINDER_PERMUTATIONS = 720

_Term = tuple[Fraction, tuple[tuple[ir.Idx, int], ...], tuple[ir.SymExpr, ...]]


class _TooBig(Exception):
    pass


def _expand(e: ir.SymExpr, fresh) -> list[_Term]:
    if isinstance(e, ir.Const):
        if isinstance(e.value, float):  # +-inf: opaque
            return [(Fraction(1), (), (e,))]
        return [] if e.value == 0 else [(e.value, (), ())]
    if isinstance(e, ir.Add):
        out: list[_Term] = []
        for a in e.args:
            out.extend(_expand(a, fresh))
            if len(out) > MAX_TERMS:
                raise _TooBig
        return out
    if isinstance(e, ir.Mul):
        return _product([_expand(a, fresh) for a in e.args])
    if isinstance(e, ir.Pow) and e.exponent.denominator == 1 and 2 <= e.exponent <= 4:
        # every copy of a sum gets its own bound index
        return _product([_expand(e.base, fresh) for _ in range(int(e.exponent))])
    if isinstance(e, ir.Reduce) and e.kind == "sum" and isinstance(e.domain, ir.DFull):
        name = fresh()
        body = subst(e.body, {ir.bidx(e.level): name})
        return [(c, ((name, e.domain.axis), *bs), fs) for c, bs, fs in _expand(body, fresh)]
    return [(Fraction(1), (), (_closed(e),))]


def _product(parts: list[list[_Term]]) -> list[_Term]:
    out: list[_Term] = [(Fraction(1), (), ())]
    for terms in parts:
        out = [(c1 * c2, b1 + b2, f1 + f2) for c1, b1, f1 in out for c2, b2, f2 in terms]
        if len(out) > MAX_TERMS:
            raise _TooBig
    return out


def _closed(f: ir.SymExpr) -> ir.SymExpr:
    """An opaque factor with its own binders renumbered from 0 (depth-independent)."""
    levels = [n.level for n in _walk(f) if isinstance(n, ir.Reduce | ir.MonoidReduce)]
    lo = min(levels, default=0)
    return shift(f, lo, -lo) if lo else f


def _walk(e: ir.Node):
    stack = [e]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children())


def _term_key(binders, factors) -> tuple:
    """Key of ``Σ_binders Π factors`` invariant under renaming the bound indices."""
    order = sorted(range(len(binders)), key=lambda i: binders[i][1])
    axes = tuple(binders[i][1] for i in order)
    groups = [list(g) for _, g in itertools.groupby(order, key=lambda i: binders[i][1])]
    best = None
    count = 0
    for perms in itertools.product(*[itertools.permutations(g) for g in groups]):
        count += 1
        if count > MAX_BINDER_PERMUTATIONS:
            raise _TooBig
        flat = [i for p in perms for i in p]
        ren = {binders[i][0]: ir.idx(f"_s{pos}") for pos, i in enumerate(flat)}
        key = tuple(sorted(subst(f, ren).uid for f in factors))
        if best is None or key < best:
            best = key
    return (axes, best if best is not None else ())


def normal_form(e: ir.SymExpr) -> dict[tuple, Fraction] | None:
    """``{term key: coefficient}``, or None when the expansion is too large."""
    counter = itertools.count()

    def fresh() -> ir.Idx:
        return ir.idx(f"_p{next(counter)}")

    try:
        terms = _expand(e, fresh)
        out: dict[tuple, Fraction] = {}
        for c, bs, fs in terms:
            key = _term_key(bs, fs)
            out[key] = out.get(key, Fraction(0)) + c
    except _TooBig:
        return None
    return {k: v for k, v in out.items() if v != 0}


def sum_product_equal(a: ir.SymExpr, b: ir.SymExpr) -> bool:
    """Proof of ``a == b`` by the sum-product normal form (False means "unknown")."""
    na = normal_form(a)
    return na is not None and na == normal_form(b)


# ---------------------------------------------------------------------------
# containment (for pruning): a contraction inside one of the target's terms
# ---------------------------------------------------------------------------
def _profile(e: ir.SymExpr):
    """Per prenex term: (tensors read by plain element factors, summed axes, opaque?)."""
    counter = itertools.count()

    def fresh() -> ir.Idx:
        return ir.idx(f"_p{next(counter)}")

    try:
        terms = _expand(e, fresh)
    except _TooBig:
        return None
    out = []
    for _, bs, fs in terms:
        tensors = Counter(f.tensor for f in fs if isinstance(f, ir.Elem))
        opaque = any(not isinstance(f, ir.Elem) and not _data_free(f) for f in fs)
        out.append((tensors, Counter(axis for _, axis in bs), opaque))
    return out


def _data_free(e: ir.Node) -> bool:
    return not any(isinstance(n, ir.Elem | ir.StateVar | ir.BIdx) for n in _walk(e))


def sum_product_contains(target: ir.SymExpr, sub: ir.SymExpr) -> bool:
    """Is every prenex term of the pure contraction ``sub`` part of a term of ``target``?

    ``Σ_k Σ_n A[i, k] X[k, n] X[k', n]`` (``A (X Xᵀ)``) is not a canonical
    sub-term of ``Σ_n (Σ_k A X)(Σ_k' A X)`` but reads a sub-multiset of the
    tensors of its single term and sums over a sub-multiset of its axes: it
    can still be completed into the target by further contractions.
    """
    mine = _profile(sub)
    if not mine or any(opaque for _, _, opaque in mine):
        return False
    key = "sum_product_profile"
    theirs = target.cache.get(key)
    if theirs is None:
        theirs = _profile(target) or []
        target.cache[key] = theirs
    return all(any(t <= tt and b <= bb for tt, bb, _ in theirs) for t, b, _ in mine)


__all__ = ["normal_form", "sum_product_contains", "sum_product_equal"]
