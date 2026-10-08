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
"""Canonical symbolic semantics and lowering."""

from fractions import Fraction

import te_programs

import tvm.testing
from tvm import te
from tvm.te.superopt.dims import DimTable
from tvm.te.superopt.symbolic import LowerCtx, ir
from tvm.te.superopt.symbolic.canonicalize import (
    instantiate,
    mk_add,
    mk_card,
    mk_div,
    mk_exp,
    mk_max,
    mk_mul,
    mk_neg,
    mk_pow,
    mk_reduce,
    mk_sub,
    positive,
    subst_domain,
)

x, y, z = ir.atom("x"), ir.atom("y"), ir.atom("z")
c2, c3 = ir.const(2), ir.const(3)


def test_canon_flatten_fold_sort():
    assert mk_add(x, mk_add(y, c2), c3) is mk_add(c3, y, c2, x)
    assert mk_add(x, x) is mk_mul(c2, x)
    assert mk_add(x, mk_neg(x)) is ir.ZERO
    assert mk_mul(x, mk_mul(y, c2), ir.ONE) is mk_mul(c2, y, x)
    assert mk_mul(x, ir.ZERO) is ir.ZERO
    assert mk_mul(x, x) is mk_pow(x, 2)
    assert mk_div(x, x) is ir.ONE
    assert mk_pow(mk_pow(x, Fraction(1, 2)), 2) is x
    assert mk_pow(ir.const(4), Fraction(1, 2)) is c2


def test_canon_distributes_products_over_sums():
    assert mk_mul(c2, mk_add(x, y)) is mk_add(mk_mul(c2, x), mk_mul(c2, y))
    assert mk_mul(z, mk_sub(x, y)) is mk_sub(mk_mul(z, x), mk_mul(z, y))


def test_canon_exp_merge():
    assert mk_mul(mk_exp(x), mk_exp(y)) is mk_exp(mk_add(x, y))
    assert mk_pow(mk_exp(x), -1) is mk_exp(mk_neg(x))
    assert mk_mul(mk_exp(x), mk_exp(mk_neg(x))) is ir.ONE
    assert mk_exp(ir.ZERO) is ir.ONE
    assert mk_exp(ir.NEG_INF_C) is ir.ZERO


def test_canon_infinities():
    assert mk_max(x, ir.NEG_INF_C) is x
    assert mk_add(x, ir.NEG_INF_C) is ir.NEG_INF_C
    assert mk_mul(ir.POS_INF_C, ir.ZERO) is ir.ZERO
    sym = ir.shape_sym("n")
    assert positive(mk_pow(sym, Fraction(-1, 2)))
    assert mk_mul(ir.NEG_INF_C, mk_pow(sym, Fraction(-1, 2))) is ir.NEG_INF_C


def test_canon_max_common_parts():
    assert mk_max(mk_add(x, z), mk_add(y, z)) is mk_add(mk_max(x, y), z)
    assert mk_max(mk_mul(c2, x), mk_mul(c2, y)) is mk_mul(c2, mk_max(x, y))
    assert mk_max(x, y) is mk_max(y, x, x)


def _elem(t, *idx):
    return ir.elem(t, tuple(idx))


def test_reduce_rules():
    D = ir.dfull(0)
    j = ir.bidx(0)
    f = _elem(0, ir.idx("i0"), j)
    g = _elem(1, ir.idx("i0"), j)
    c = _elem(2, ir.idx("i0"))  # j-free
    # Σ(f + g) = Σf + Σg ; Σ c f = c Σ f
    assert mk_reduce("sum", D, 0, mk_add(f, g)) is mk_add(
        mk_reduce("sum", D, 0, f), mk_reduce("sum", D, 0, g)
    )
    assert mk_reduce("sum", D, 0, mk_mul(c, f)) is mk_mul(c, mk_reduce("sum", D, 0, f))
    # shift extraction: Σ exp(f - c) = exp(-c) Σ exp(f)
    assert mk_reduce("sum", D, 0, mk_exp(mk_sub(f, c))) is mk_mul(
        mk_exp(mk_neg(c)), mk_reduce("sum", D, 0, mk_exp(f))
    )
    # max rules
    assert mk_reduce("max", D, 0, mk_add(f, c)) is mk_add(mk_reduce("max", D, 0, f), c)
    assert mk_reduce("max", D, 0, mk_mul(c2, f)) is mk_mul(c2, mk_reduce("max", D, 0, f))
    assert mk_reduce("max", D, 0, mk_exp(f)) is mk_exp(mk_reduce("max", D, 0, f))
    assert mk_reduce("max", D, 0, mk_max(f, g)) is mk_max(
        mk_reduce("max", D, 0, f), mk_reduce("max", D, 0, g)
    )
    # constant body
    assert mk_reduce("sum", D, 0, c) is mk_mul(c, mk_card(D))
    assert mk_reduce("max", D, 0, c) is c


def test_domain_algebra():
    axis = 0
    j = ir.bidx(0)
    f = _elem(0, ir.idx("i0"), j)
    A, B = ir.dsym(axis, "A"), ir.dsym(axis, "B")
    s = mk_reduce("sum", ir.dsym(axis, "R"), 0, f)
    split = subst_domain(s, {ir.dsym(axis, "R"): ir.dunion(A, B)})
    assert split is mk_add(mk_reduce("sum", A, 0, f), mk_reduce("sum", B, 0, f))
    m = mk_reduce("max", ir.dsym(axis, "R"), 0, f)
    assert subst_domain(m, {ir.dsym(axis, "R"): ir.dunion(A, B)}) is mk_max(
        mk_reduce("max", A, 0, f), mk_reduce("max", B, 0, f)
    )
    assert subst_domain(s, {ir.dsym(axis, "R"): ir.dempty(axis)}) is ir.ZERO
    assert subst_domain(m, {ir.dsym(axis, "R"): ir.dempty(axis)}) is ir.NEG_INF_C


def test_alpha_equivalence_of_nested_reductions():
    D0, D1 = ir.dfull(0), ir.dfull(1)
    inner = mk_reduce("sum", D1, 1, mk_mul(_elem(0, ir.bidx(0), ir.bidx(1)), _elem(1, ir.bidx(1))))
    outer = mk_reduce("sum", D0, 0, mk_exp(inner))
    # rebuilding the same structure yields the same node
    inner2 = mk_reduce("sum", D1, 1, mk_mul(_elem(1, ir.bidx(1)), _elem(0, ir.bidx(0), ir.bidx(1))))
    assert mk_reduce("sum", D0, 0, mk_exp(inner2)) is outer
    # inlining a body under a binder shifts its levels
    body = mk_reduce("sum", D1, 0, _elem(0, ir.idx("i0"), ir.bidx(0)))
    inst = instantiate(body, {ir.idx("i0"): ir.bidx(0)}, 1)
    assert isinstance(inst, ir.Reduce) and inst.level == 1
    assert inst.body is _elem(0, ir.bidx(0), ir.bidx(1))


def test_dimtable_symbolic_vs_concrete():
    n, m = te.var("n"), te.var("m")
    dims = DimTable()
    assert dims.key(n) == dims.key(n)
    assert dims.key(n) != dims.key(m)
    assert dims.key(4) == dims.key(4)
    assert dims.key(4) != dims.key(n)
    assert dims.name(dims.key(n)) == "n" and dims.key_of_name("n") == dims.key(n)


def test_lower_attention_naive_equals_flash_closed_form():
    """The textbook chain canonicalises to the unshifted softmax; the max shift cancels."""
    ins, out = te_programs.attention("naive")
    ctx = LowerCtx()
    sem = ctx.lower(out)
    body = sem.body
    assert sem.rank == 4 and sem.dtype == "float32"
    # exactly two top-level reductions over the key axis and no max anywhere
    reds = [r for r in _all_nodes(body) if isinstance(r, ir.Reduce) and r.level == 0]
    assert sorted(r.kind for r in reds) == ["sum", "sum"]
    assert not any(isinstance(n, ir.Max) for n in _all_nodes(body))
    assert not any(isinstance(n, ir.Reduce) and n.kind == "max" for n in _all_nodes(body))


def test_lower_flash_is_opaque_monoid():
    ins, out = te_programs.attention("flash")
    sem = LowerCtx().lower(out)
    monoids = [n for n in _all_nodes(sem.body) if isinstance(n, ir.MonoidReduce)]
    assert monoids and all(len(m.leaf) == 3 and len(m.merge) == 3 for m in monoids)
    assert monoids[0].identity[0] is ir.NEG_INF_C


def _all_nodes(e):
    out, stack = [], [e]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children())
    return out


if __name__ == "__main__":
    tvm.testing.main()
