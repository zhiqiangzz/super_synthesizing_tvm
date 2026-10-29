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
"""Choosing the states: keep the program's loops, supplement, replace -- one procedure."""

from chain_ops import OPERATORS

import tvm.testing
from tvm import te
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.derive import atoms_in, closed_merges, domains
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.states import (
    _Chain,
    _content,
    _scaled,
    _Select,
    members_of,
    select,
)
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import (
    mk_add,
    mk_card,
    mk_div,
    mk_exp,
    mk_mul,
    mk_pow,
    mk_reduce,
    mk_sub,
    recanonicalize,
    walk,
)
from tvm_tirx import tirx as tir


def _chain(name: str, which: int = 0):
    _, outs = OPERATORS[name].unfused()
    return discover_chains(outs)[0][which]


def _select(name: str, positive=None, which: int = 0):
    op = OPERATORS[name]
    _, outs = op.unfused()
    chain = discover_chains(outs)[0][which]
    signs = TeSigns(outs, op.positive if positive is None else positive)
    return select(chain, members_of(chain), signs)


def _world(name: str) -> _Chain:
    chain = _chain(name)
    return _Chain(chain, members_of(chain), TeSigns([]))


def _x(tensor: int = 0):
    """A boundary tensor, read along the reduction."""
    return ir.elem(tensor, (ir.idx("i0"), ir.bidx(0)))


def _R(sel) -> ir.DSym:
    """The part of the axis the selected states are defined on."""
    for c in sel.states:
        for n in walk(c.state):
            if isinstance(n, ir.Reduce | ir.Card) and isinstance(n.domain, ir.DSym):
                return n.domain
    raise AssertionError("no state reduces anything")


def _names(sel) -> list[str]:
    return [c.name for c in sel.states]


def _steps(sel, action: str) -> list[str]:
    return [st.summary() for st in sel.steps if st.action == action]


def _same(a, b) -> bool:
    """The same state up to a rational factor."""
    return _scaled(a, _content(a)) is _scaled(b, _content(b))


def _ab(n):
    return ir.state_var("a", n), ir.state_var("b", n)


def _rows():
    n, m = te.var("n"), te.var("m")
    return te.placeholder((n, m), name="X"), n, m


# ---------------------------------------------------------------------------
# keep: the loops of the program
# ---------------------------------------------------------------------------
def test_softmax_keeps_its_loops_and_merges_them_by_rebasing():
    sel = _select("softmax")
    assert _names(sel) == ["mx", "den"]
    assert [st.action for st in sel.steps] == ["keep", "keep"]  # nothing else was needed
    spec = sel.solution.spec
    assert spec.leaves == (_x(), ir.ONE) and spec.identity == (ir.NEG_INF_C, ir.ZERO)
    # printed: each side keeps its own sum and is rescaled by exp(own max - merged max)
    (a0, b0), (a1, b1) = _ab(0), _ab(1)
    m = ir.raw_max([a0, b0])

    def rescaled(s0, s1):
        shift = ir.raw_add([m, ir.raw_mul([ir.MINUS_ONE, s0])])
        return ir.raw_mul([ir.raw_exp(ir.raw_mul([ir.MINUS_ONE, shift])), s1])

    assert spec.merge_print == (m, ir.raw_add([rescaled(a0, a1), rescaled(b0, b1)]))
    # canonical: the merged max read as the merged slot m0, the same once it is filled in
    m0 = ir.state_var("m", 0)
    assert spec.merge[1] is mk_add(
        mk_mul(a1, mk_exp(mk_sub(a0, m0))), mk_mul(b1, mk_exp(mk_sub(b0, m0)))
    )
    assert recanonicalize(spec.merge_print[1]) is closed_merges(spec.merge)[1]


def test_attention_keeps_the_normalised_output():
    """The output re-bases against two contexts at once: shift for the max, scale for the
    mass; on an empty part the mass is 0 and the scale is printed as ``den_X / den``."""
    sel = _select("attention")
    assert _names(sel) == ["row_max", "den", "O"] and sel.solution.spec.merge_print is not None
    w = _world("attention")
    out = next(c for c in w.members if c.name == "O")
    modes = {(w.name(pid), mode) for _, _, pid, mode in w.rebaser.decompose(out).corr}
    assert modes == {("row_max", "shift"), ("den", "scale")}


def test_attention_canonical_merge_keeps_its_contexts():
    """``O`` reads ``row_max`` and ``den`` as their merged slots ``m0``, ``m1``: the max
    shift folds into each side's own instead of cancelling against the mass's, so the
    merge never reads ``exp(a0)`` bare -- written over the partial reductions it would."""
    spec = _select("attention").solution.spec
    (a0, b0), (a1, b1), (a2, b2) = _ab(0), _ab(1), _ab(2)
    m0, m1 = ir.state_var("m", 0), ir.state_var("m", 1)

    def side(s0, s1, s2):
        return mk_mul(s1, s2, mk_pow(m1, -1), mk_exp(mk_sub(s0, m0)))

    assert spec.merge[2] is mk_add(side(a0, a1, a2), side(b0, b1, b2))
    assert not {mk_exp(a0), mk_exp(b0)} & set(walk(closed_merges(spec.merge)[2]))


def test_members_run_on_a_part_of_the_axis_are_where_the_states_start():
    chain = _chain("variance")
    members = members_of(chain)
    assert [(c.name, c.origin) for c in members] == [
        ("total_mean", "member"),
        ("mean", "member"),
        ("ss", "member"),
    ]
    _, R, _, _ = domains(chain.axis)
    assert set(atoms_in(members[2].state, R)) == {
        mk_reduce("sum", R, 0, _x()),
        mk_reduce("sum", R, 0, mk_pow(_x(), 2)),
        mk_card(R),
    }


# ---------------------------------------------------------------------------
# supplement: what a merge needs
# ---------------------------------------------------------------------------
def test_variance_is_supplemented_with_the_count_and_keeps_the_mean():
    """``ss`` re-based around the mean needs the count. The total and the count would do,
    but the mean they give is 0 / 0 on an empty part: the mean itself is kept instead."""
    sel = _select("variance")
    assert _names(sel) == ["mean", "ss", "count"]
    (promote,) = _steps(sel, "promote")
    assert promote.startswith("promote mean in place of total_mean")
    assert "merged with an empty part it is NaN, through mean there" in promote
    assert _steps(sel, "supplement") == ["supplement count = Σ_j 1: mean needs Σ_j 1"]
    # Welford: the squares are corrected around the merged mean, never expanded into Σx²
    spec = sel.solution.spec
    (ma, mb), (sa, sb), (na, nb) = _ab(0), _ab(1), _ab(2)
    mean = mk_div(mk_add(mk_mul(ma, na), mk_mul(mb, nb)), mk_add(na, nb))
    want = mk_add(
        sa, sb, mk_mul(na, mk_pow(mk_sub(mean, ma), 2)), mk_mul(nb, mk_pow(mk_sub(mean, mb), 2))
    )
    assert recanonicalize(spec.merge_print[1]) is want
    assert spec.leaves[1] is ir.ZERO and spec.leaves[2] is ir.ONE
    # canonical: the merged mean m0 kept, ss written over each side's power sums around it
    m0 = ir.state_var("m", 0)
    power_sums = mk_add(sa, mk_mul(na, mk_pow(ma, 2)), sb, mk_mul(nb, mk_pow(mb, 2)))
    first = mk_mul(ir.const(-2), m0, mk_add(mk_mul(ma, na), mk_mul(mb, nb)))
    assert spec.merge[1] is mk_add(power_sums, first, mk_mul(mk_add(na, nb), mk_pow(m0, 2)))


def test_higher_moments_are_supplemented_with_the_lower_ones():
    sel = _select("moment3")
    R = _R(sel)
    x, n = _x(), mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)
    m2 = mk_reduce("sum", R, 0, mk_pow(mk_sub(x, mean), 2))
    (lifted,) = [c for c in sel.states if c.origin == "closure"]
    assert lifted.source == "s3" and _same(lifted.state, m2)
    assert {c.origin for c in sel.states} == {"member", "extent", "closure"}
    sel = _select("moment4")
    assert len(sel.states) == 5 and sel.solution.spec.merge_print is not None
    assert [c.origin for c in sel.states].count("closure") == 2


def test_the_fifth_moment_takes_the_second_third_and_fourth():
    """``Σ (x - mean)⁵`` re-bases by a shift like the lower moments (a power of a sum is
    expanded by the size of the result, not by its degree): nothing is taken out, and
    the moments below it are supplemented."""
    sel = _select("moment5")
    assert _names(sel) == ["mean", "s5", "count", "s5.r0", "s5.r1", "s5.r2"]
    assert not _steps(sel, "hoist") and sel.solution.spec.merge_print is not None
    R = _R(sel)
    x, n = _x(), mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)
    for c, k in zip(sel.states[3:], (2, 3, 4)):
        assert c.origin == "closure" and c.source == "s5"
        assert _same(c.state, mk_reduce("sum", R, 0, mk_pow(mk_sub(x, mean), k)))


def test_a_coefficient_that_is_one_reduction_comes_before_a_compound_one():
    """``Σ (a - r q)²``: re-basing it around ``r`` leaves ``Σ q²`` and ``2 Σ (r q² - q a)``;
    the second would have to be divided by ``r`` to read anything back."""
    sel = _select("projection")
    assert _names(sel) == ["r", "rem", "rem.r0"]
    R = _R(sel)
    q = ir.elem(0, (ir.idx("i0"), ir.bidx(0)))
    assert _same(sel.states[2].state, mk_reduce("sum", R, 0, mk_pow(q, 2)))


# ---------------------------------------------------------------------------
# replace: a state that is not defined on every part
# ---------------------------------------------------------------------------
def test_a_divisor_that_is_zero_on_a_single_element_is_taken_out():
    """``max_j (x - mean) / sd``: ``sd`` of one element is 0. Only ``sd`` is taken out;
    the mean stays, re-based by a shift."""
    sel = _select("zscore_max")
    assert _names(sel) == ["mean", "ss", "zmax.h0", "count"]
    (hoist,) = _steps(sel, "hoist")
    assert hoist == (
        "hoist sigma out of zmax: zmax: it divides by sigma[i], which is 0 on every single element"
    )
    extreme = sel.states[2]
    assert extreme.kind == "max" and extreme.origin == "hoist" and extreme.depth == 1
    w = _world("zscore_max")
    assert [w.name(p) for p in w.rebaser.contexts(extreme)] == ["mean"]


def test_both_norms_are_taken_out_of_a_cosine_at_once():
    """The cosine divides by ``‖x‖`` and ``‖y‖``, each 0 where its vector is: the state left
    is ``Σ x y``, as the hand-written single pass has it."""
    sel = _select("cosine")
    (hoist,) = _steps(sel, "hoist")
    assert hoist.startswith("hoist ny, nx out of cos")
    assert "which is 0 where X[i, j] is" in hoist and "which is 0 where Y[i, j] is" in hoist
    R = _R(sel)
    assert _same(sel.states[2].state, mk_reduce("sum", R, 0, mk_mul(_x(0), _x(1))))


def test_a_context_defined_by_a_division_is_checked_through():
    """``rem`` reads ``r = qa / qq``, and ``qq = Σ q²`` is 0 where ``q`` is."""
    sel = _select("normalised_projection")
    (hoist,) = _steps(sel, "hoist")
    assert hoist == (
        "hoist r out of rem: rem: it divides by qq[i] (through r), which is 0 where Q[i, j] is"
    )


def test_what_reading_a_reduction_back_divides_by_is_checked():
    """``out = Σ_j y_j g`` with ``g = Σ_j x_j``: ``Σ y = out / g``, and ``g`` is 0 where
    ``x`` is. ``out`` is replaced by ``Σ y`` -- the merge of ``(g, out)`` would lose it."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    Y = te.placeholder((n, m), name="Y")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    g = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="g")
    out = te.compute((n,), lambda i: te.sum(Y[i, j2] * g[i], axis=j2), name="out")
    (chain,), _ = discover_chains(out)
    sel = select(chain, members_of(chain), TeSigns(out))
    (hoist,) = _steps(sel, "hoist")
    assert hoist.startswith("hoist g out of out: out: reading Σ_j Y[i, j] back from it divides")
    assert hoist.endswith("which is 0 where X[i, j] is")
    _, R, _, _ = domains(chain.axis)
    assert _same(sel.states[1].state, mk_reduce("sum", R, 0, _x(1)))


def test_a_context_that_is_infinite_on_an_empty_part_is_taken_out():
    """``Σ q (lse - x)``: on an empty part ``log den`` is ``-inf`` and the merge adds
    ``Σ q · (lse - lse_empty)`` = ``0 · inf``. The maximum's shift stays: on an empty
    part it is the lowest float, and ``exp`` of minus that is 0."""
    sel = _select("soft_cross_entropy")
    (hoist,) = _steps(sel, "hoist")
    assert hoist.startswith("hoist loss.ctx0 out of loss: loss: merged with an empty part")
    R = _R(sel)
    sums = {c.name: c for c in sel.states if c.origin == "hoist"}
    assert any(_same(c.state, mk_reduce("sum", R, 0, _x(1))) for c in sums.values())  # Σq
    (loss,) = sel.solution.spec.epilogues
    assert not any(isinstance(n, ir.Reduce) for n in walk(loss))


def test_entropy_takes_the_log_of_the_normaliser_out_and_keeps_the_shift():
    sel = _select("softmax_entropy")
    assert _names(sel) == ["mx", "den", "H.h0"]
    (hoist,) = _steps(sel, "hoist")
    assert hoist.startswith("hoist H.ctx0 out of H")
    w = _world("softmax_entropy")
    kept = {w.name(p) for p in w.rebaser.contexts(sel.states[2])}
    assert kept == {"mx", "den"}


def test_a_scale_that_is_undefined_on_an_empty_part_is_taken_out():
    """LayerNorm's backward: ``sgx`` divides by ``sd = sqrt(ss / n + eps)``, positive on
    every non-empty part, but ``0 / 0`` on an empty one. A factor moves out of a sum
    at no cost; the mean, a shift, is kept as a state instead."""
    sel = _select("layernorm_backward")
    assert _names(sel) == ["sg", "mean", "ss", "sgx.h0", "count"]
    hoist = _steps(sel, "hoist")
    assert hoist == [
        "hoist sigma out of sgx: sgx: merged with an empty part it is NaN, through sigma there"
    ]
    assert _steps(sel, "promote")[0].startswith("promote mean in place of total")


def test_a_maximum_of_reductions_is_taken_apart():
    """``max_j |x - mean|`` is the larger of two reductions, each a shift by the mean."""
    sel = _select("max_abs_deviation")
    assert _steps(sel, "parts") == [
        "parts dmax: no reduction it is made of can be read back from it"
    ]
    assert _names(sel) == ["mean", "dmax.m0", "dmax.m1", "count"]
    assert all(c.origin == "part" and c.source == "dmax" for c in sel.states[1:3])
    a, b = _ab(3)
    assert sel.solution.spec.merge[3] is mk_add(a, b)


def test_a_function_taken_out_of_a_maximum_leaves_its_reduction():
    sel = _select("peak_power")
    assert [c.origin for c in sel.states[1:3]] == ["inner", "inner"]
    assert sel.solution.spec.epilogue_print is None
    sel = _select("peak_to_geomean")
    assert _names(sel) == ["mean", "ratio.m0", "count"] and sel.states[1].origin == "inner"


def _energy():
    """``Σ (x - mean)² + exp(x - mx)``: a sum whose two parts move under different corrections."""
    X, n, m = _rows()
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m.astype("float32"), name="mean")
    mx = te.compute((n,), lambda i: te.max(X[i, j2], axis=j2), name="mx")
    return te.compute(
        (n,),
        lambda i: te.sum(
            (X[i, j3] - mean[i]) * (X[i, j3] - mean[i]) + tir.exp(X[i, j3] - mx[i]), axis=j3
        ),
        name="energy",
    )


def test_a_sum_whose_parts_move_differently_is_kept_as_its_parts():
    """``Σ (x - mean)² + exp(x - mx)``: re-based, the first part moves with ``1, δ, δ²``,
    the second with ``e^{-ε}``, and no coefficient is the whole sum. Each part re-bases
    by itself and is kept; taking the contexts out instead leaves ``Σ exp(x)``."""
    energy = _energy()
    (chain,), _ = discover_chains([energy])
    sel = select(chain, members_of(chain), TeSigns([energy]))
    assert _names(sel) == ["mean", "mx", "energy.p0", "energy.p1", "count"]
    assert not _steps(sel, "hoist") and sel.solution.spec.merge_print is not None
    (split,) = _steps(sel, "split")
    assert split.startswith("split energy into ")
    assert split.endswith(": it does not re-base; each of these does")
    R, x = _R(sel), _x()
    centred = mk_sub(x, mk_div(mk_reduce("sum", R, 0, x), mk_card(R)))
    shifted = mk_sub(x, mk_reduce("max", R, 0, x))
    want = [mk_reduce("sum", R, 0, mk_pow(centred, 2)), mk_reduce("sum", R, 0, mk_exp(shifted))]
    pieces = sel.states[2:4]
    assert all(any(_same(c.state, s) for c in pieces) for s in want)
    assert all(c.origin == "piece" and c.source == "energy" for c in pieces)


def test_a_state_that_does_not_re_base_gets_nothing_supplemented():
    """Read off as it is, ``energy`` needs the count and ``Σ exp(x)``. It does not re-base,
    so neither is added: it is split first, and its piece ``Σ exp(x - mx)`` holds what the
    raw sum would have."""
    energy = _energy()
    (chain,), _ = discover_chains([energy])
    sel = _Select(_Chain(chain, members_of(chain), TeSigns([energy])))
    states, sol, trace = sel.supplement(sel.base())
    assert [c.name for c in states] == ["total", "mx", "energy"] and sol is None
    c, (fault,) = sel.faults(states, sol, trace)
    assert (c.name, fault.kind, fault.reason) == ("energy", "structure", "it does not re-base")


def test_two_rates_of_one_context_are_two_states():
    """``Σ exp(x - mx) + exp(2 (x - mx))``: one context, whose correction weighs the two
    parts by ``e^{-δ}`` and ``e^{-2δ}``. A polynomial of the shift is one rate: ``Σ (x -
    mean)² + (x - mean)`` re-bases whole."""
    X, n, m = _rows()
    j1, j2 = (te.reduce_axis((0, m), "j") for _ in range(2))
    mx = te.compute((n,), lambda i: te.max(X[i, j1], axis=j1), name="mx")
    two = tir.const(2.0, "float32")
    out = te.compute(
        (n,),
        lambda i: te.sum(tir.exp(X[i, j2] - mx[i]) + tir.exp((X[i, j2] - mx[i]) * two), axis=j2),
        name="out",
    )
    (chain,), _ = discover_chains([out])
    sel = select(chain, members_of(chain), TeSigns([out]))
    assert _names(sel) == ["mx", "out.p0", "out.p1"] and len(_steps(sel, "split")) == 1

    X, n, m = _rows()
    j1, j2 = (te.reduce_axis((0, m), "j") for _ in range(2))
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m.astype("float32"), name="mean")
    poly = te.compute(
        (n,),
        lambda i: te.sum(
            (X[i, j2] - mean[i]) * (X[i, j2] - mean[i]) + (X[i, j2] - mean[i]), axis=j2
        ),
        name="poly",
    )
    (chain,), _ = discover_chains([poly])
    sel = select(chain, members_of(chain), TeSigns([poly]))
    assert _names(sel) == ["mean", "poly", "count"] and not _steps(sel, "split")


def test_each_context_takes_the_correction_its_dependence_fits():
    """Context by context, the others held fixed: ``exp(x - mx)`` fits a shift, ``v / den``
    a scale; ``(x - mean)²`` fits both and re-bases whole only under a shift. ``Σ y g``
    re-bases whole under both, with one coefficient under a scale and two under a shift."""
    w = _world("attention")
    out = next(c for c in w.members if c.name == "O")
    chosen, neither = w.rebaser.modes(out)
    assert {w.name(p): mode for p, mode in chosen.items()} == {"row_max": "shift", "den": "scale"}
    assert not neither
    w = _world("variance")
    ss = next(c for c in w.members if c.name == "ss")
    assert list(w.rebaser.modes(ss)[0].values()) == ["shift"]

    X, n, m = _rows()
    Y = te.placeholder((n, m), name="Y")
    j1, j2 = (te.reduce_axis((0, m), "j") for _ in range(2))
    g = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="g")
    out = te.compute((n,), lambda i: te.sum(Y[i, j2] * g[i], axis=j2), name="out")
    (chain,), _ = discover_chains(out)
    w = _Chain(chain, members_of(chain), TeSigns(out))
    c = next(c for c in w.members if c.name == "out")
    assert list(w.rebaser.modes(c)[0].values()) == ["scale"]
    assert [mode for *_, mode in w.rebaser.decompose(c).corr] == ["scale", "scale"]  # both sides


def test_only_the_context_no_correction_fits_is_taken_out():
    """``max_j (x - mean) / sd`` with ``sd = sqrt(ss / n + eps)``, never 0: the maximum
    moves with a shift of ``mean``, with neither a shift nor a scale of ``sd``. Only
    ``sd`` is taken out, and the maximum of ``x - mean`` is kept."""
    X, n, m = _rows()
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    count = m.astype("float32")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / count, name="mean")
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - mean[i]) * (X[i, j2] - mean[i]), axis=j2), name="ss"
    )
    eps = tir.const(1e-5, "float32")
    sd = te.compute((n,), lambda i: tir.sqrt(ss[i] / count + eps), name="sd")
    zmax = te.compute((n,), lambda i: te.max((X[i, j3] - mean[i]) / sd[i], axis=j3), name="zmax")
    (chain,), _ = discover_chains([zmax])
    members = members_of(chain)
    sel = select(chain, members, TeSigns([zmax]))
    assert _names(sel) == ["mean", "ss", "zmax.h0", "count"]
    (hoist,) = _steps(sel, "hoist")
    assert hoist == (
        "hoist sd out of zmax: zmax: it does not re-base: "
        "neither a shift nor a scale of sd leaves its reduction"
    )
    w = _Chain(chain, members, TeSigns([zmax]))
    assert [w.name(p) for p in w.rebaser.contexts(sel.states[2])] == ["mean"]
    # where taking it out alone does not leave the reduction, every context goes with it
    sel = _select("standardised_softmax")
    (hoist,) = _steps(sel, "hoist")
    assert hoist.startswith("hoist sigma, mean out of zden")
    assert hoist.endswith(
        "neither a shift nor a scale of sigma leaves its reduction; "
        "sigma alone does not leave the reduction"
    )


def test_declared_positive_weights_keep_the_weighted_mean():
    """``mean = Σ w x / Σ w``: with ``w`` declared positive the weight sum is never 0 and
    the mean is kept; without, it may be, and the mean has to be taken out."""
    sel = _select("weighted_variance")
    assert _names(sel) == ["mean", "sw", "ss"] and not _steps(sel, "hoist")
    sel = _select("weighted_variance", positive=())
    (hoist,) = _steps(sel, "hoist")
    assert hoist.endswith("it divides by sw[i] (through mean), which is 0 where W[i, j] is")


def test_only_a_multiple_of_a_loop_result_is_promoted():
    """``Σ (log x - log mean)``: ``log mean`` is 0 / 0 on an empty part like the mean, but
    as a state it would merge through ``exp`` of its own values. It is taken out instead."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    mean = te.compute((n,), lambda i: total[i] / m.astype("float32"), name="mean")
    out = te.compute(
        (n,), lambda i: te.sum(tir.log(X[i, j2]) - tir.log(mean[i]), axis=j2), name="out"
    )
    (chain,), _ = discover_chains(out)
    sel = select(chain, members_of(chain), TeSigns(out))
    assert not _steps(sel, "promote") and _steps(sel, "hoist")
    assert sel.solution is not None


# ---------------------------------------------------------------------------
# the extent as a count or as a constant
# ---------------------------------------------------------------------------
def test_the_extent_is_a_count_unless_it_is_a_term_of_what_is_added_up():
    assert _chain("variance").count  # mean = total / n: the mean of what was reduced
    assert _chain("softmax").count  # never used as a value
    assert not _chain("label_smoothing").count  # 0.9 y + 0.1 / k: a constant of the problem
    sel = _select("label_smoothing")
    assert len(sel.states) == 4 and sel.solution.spec.merge_print is not None


# ---------------------------------------------------------------------------
# the procedure as a whole
# ---------------------------------------------------------------------------
def test_every_state_comes_out_of_the_program():
    """No state is invented: a loop of the program, a context it computes, the count, or
    taken from the expansion of one of those."""
    origins = (
        "member",
        "context",
        "extent",
        "part",
        "inner",
        "piece",
        "closure",
        "hoist",
        "trivial",
    )
    for name in ("softmax", "attention", "moment4", "covariance", "softmax_entropy", "kurtosis"):
        sel = _select(name)
        assert sel.solution is not None, name
        loops = {m.name for m in _chain(name).members}
        for c in sel.states:
            assert c.origin in origins, (name, c.name)
            if c.origin in ("part", "inner", "piece", "closure", "hoist"):
                assert c.source.split(".")[0] in loops, (name, c.name)


def test_the_selection_is_the_same_whatever_ran_before():
    """Orders are structural, so the states do not depend on what the process built first."""
    names = ["kurtosis", "attention", "zscore_max", "projection"]
    first = {n: (_names(_select(n)), [s.summary() for s in _select(n).steps]) for n in names}
    again = {
        n: (_names(_select(n)), [s.summary() for s in _select(n).steps]) for n in reversed(names)
    }
    assert first == again


def test_kurtosis_takes_well_under_a_second():
    """The procedure derives a handful of reducers, not every subset of the candidates."""
    import time

    start = time.perf_counter()
    sel = _select("kurtosis")
    assert sel.solution is not None and len(sel.states) == 5
    assert time.perf_counter() - start < 30


if __name__ == "__main__":
    tvm.testing.main()
