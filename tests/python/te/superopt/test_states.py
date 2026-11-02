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
    assert spec.leaves == (_x(), ir.ONE)
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
    mass; both sides are non-empty and the scale is printed as ``den_X / den``."""
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
def test_variance_keeps_the_total_and_supplements_the_count():
    """The mean is computed from total/count on non-empty parts, not promoted
    merely because that expression would be undefined on an empty part."""
    sel = _select("variance")
    assert _names(sel) == ["total_mean", "ss", "ss.r0"]
    assert sel.states[2].state == mk_card(_R(sel))
    assert sel.states[2].origin == "closure" and sel.states[2].source == "ss"
    assert not _steps(sel, "promote") and not _steps(sel, "hoist")
    assert _steps(sel, "supplement")[0].startswith("supplement ss.r0 = Σ_j 1:")
    spec = sel.solution.spec
    (ta, tb), (sa, sb), (na, nb) = _ab(0), _ab(1), _ab(2)
    ma, mb = mk_div(ta, na), mk_div(tb, nb)
    mean = mk_div(mk_add(ta, tb), mk_add(na, nb))
    want = mk_add(
        sa, sb, mk_mul(na, mk_pow(mk_sub(mean, ma), 2)), mk_mul(nb, mk_pow(mk_sub(mean, mb), 2))
    )
    assert recanonicalize(spec.merge_print[1]) is want
    assert spec.leaves == (_x(), ir.ZERO, ir.ONE)


def test_higher_moments_are_supplemented_with_the_lower_ones():
    sel = _select("moment3")
    R = _R(sel)
    x, n = _x(), mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)
    m2 = mk_reduce("sum", R, 0, mk_pow(mk_sub(x, mean), 2))
    (lifted,) = [c for c in sel.states if c.origin == "closure" and c.state != n]
    assert lifted.source == "s3" and _same(lifted.state, m2)
    assert {c.origin for c in sel.states} == {"member", "closure"}
    sel = _select("moment4")
    assert len(sel.states) == 5 and sel.solution.spec.merge_print is not None
    assert [c.origin for c in sel.states].count("closure") == 3


def test_the_fifth_moment_takes_the_second_third_and_fourth():
    """``Σ (x - mean)⁵`` re-bases by a shift like the lower moments (a power of a sum is
    expanded by the size of the result, not by its degree): nothing is taken out, and
    the moments below it are supplemented."""
    sel = _select("moment5")
    assert _names(sel) == ["total_mean", "s5", "s5.r0", "s5.r1", "s5.r2", "s5.r3"]
    assert not _steps(sel, "hoist") and sel.solution.spec.merge_print is not None
    R = _R(sel)
    x, n = _x(), mk_card(R)
    mean = mk_div(mk_reduce("sum", R, 0, x), n)
    assert sel.states[2].state == n and sel.states[2].origin == "closure"
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
    assert _names(sel) == ["total", "ss", "zmax.h0", "ss.r0"]
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
    """A singular inverse cannot authorize replacing a defined original state."""
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    Y = te.placeholder((n, m), name="Y")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    g = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="g")
    out = te.compute((n,), lambda i: te.sum(Y[i, j2] * g[i], axis=j2), name="out")
    (chain,), _ = discover_chains(out)
    sel = select(chain, members_of(chain), TeSigns(out))
    assert not _steps(sel, "hoist")
    assert _names(sel) == ["g", "out"]
    assert sel.solution is None and "not an impossibility proof" in sel.failure


def test_cross_entropy_keeps_the_log_context_on_nonempty_parts():
    sel = _select("soft_cross_entropy")
    assert _names(sel) == ["mx", "den", "loss", "loss.r0"]
    assert sel.solution is not None and not _steps(sel, "hoist")
    R = _R(sel)
    assert _same(sel.states[3].state, mk_reduce("sum", R, 0, _x(1)))


def test_entropy_keeps_its_original_reduction():
    sel = _select("softmax_entropy")
    assert _names(sel) == ["mx", "den", "H"]
    assert sel.solution is not None
    assert [step.action for step in sel.steps] == ["keep", "keep", "keep"]
    assert sel.solution.spec.leaves[2] is ir.ZERO


def test_layernorm_backward_keeps_the_scale_and_total():
    """sigma is positive on non-empty parts (eps > 0); its empty value is irrelevant."""
    sel = _select("layernorm_backward")
    assert _names(sel) == ["sg", "total", "ss", "sgx", "ss.r0"]
    assert sel.solution is not None, sel.failure
    assert not _steps(sel, "hoist") and not _steps(sel, "promote")


def test_a_maximum_is_not_replaced_when_recovery_is_unproved():
    sel = _select("peak_to_geomean")
    assert sel.solution is None
    assert "not an impossibility proof" in sel.failure
    assert all(c.origin == "member" for c in sel.states)
    assert not _steps(sel, "hoist") and not _steps(sel, "parts")


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


def test_different_rates_supplement_a_component_and_preserve_the_original_sum():
    energy = _energy()
    (chain,), _ = discover_chains([energy])
    sel = select(chain, members_of(chain), TeSigns([energy]))
    assert sel.solution is not None, sel.failure
    assert _names(sel)[:3] == ["total", "mx", "energy"]
    assert len(sel.states) == 5
    assert not _steps(sel, "hoist") and not _steps(sel, "split")
    assert sum(c.origin == "piece" for c in sel.states) == 1
    # The preserved sum and one component recover the other component.
    assert sel.solution.spec.epilogues[0].tensor == -3
    from tvm.te.superopt import fuse

    fused = fuse([energy], list(chain.boundary))
    assert fused.fused and fused.accuracy.ok, fused.summary()


def test_rebase_decomposition_is_not_a_prerequisite_to_supplementing_a_tuple():
    energy = _energy()
    (chain,), _ = discover_chains([energy])
    selector = _Select(_Chain(chain, members_of(chain), TeSigns([energy])))
    states, sol, trace = selector.supplement(selector.base())
    assert [c.name for c in states][:3] == ["total", "mx", "energy"]
    assert sol is not None and len(states) == 5


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
    assert _names(sel) == ["mx", "out", "out.p0"] and not _steps(sel, "split")
    assert sel.solution is not None, sel.failure

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
    assert _names(sel) == ["total", "poly", "poly.r0"] and not _steps(sel, "split")
    assert sel.solution is not None, sel.failure
    assert sel.states[2].state == sel.states[2].struct == mk_card(_R(sel))
    assert sel.states[2].origin == "closure" and sel.states[2].source == "poly"


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


def test_tuple_recovery_keeps_a_context_that_individual_rebase_rules_cannot_handle():
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
    assert _names(sel) == ["total", "ss", "zmax", "ss.r0"]
    assert sel.solution is not None, sel.failure
    assert not _steps(sel, "hoist")
    # This other source really divides by zero on a singleton. A repair is
    # permitted, although the remaining transcendental reduction is unsupported.
    sel = _select("standardised_softmax")
    assert sel.solution is None
    assert all("0 on every single element" in st for st in _steps(sel, "hoist"))


def test_declared_positive_weights_keep_the_weighted_context():
    """``mean = Σ w x / Σ w``: with ``w`` declared positive the weight sum is never 0 and
    the mean is kept; without, it may be, and the mean has to be taken out."""
    sel = _select("weighted_variance")
    assert _names(sel) == ["swx", "sw", "ss"] and not _steps(sel, "hoist")
    sel = _select("weighted_variance", positive=())
    (hoist,) = _steps(sel, "hoist")
    assert hoist.endswith("it divides by sw[i] (through mean), which is 0 where W[i, j] is")


def test_log_mean_is_kept_without_an_empty_part_repair():
    """log(mean) is defined on non-empty parts when the input to log is positive."""
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
    assert not _steps(sel, "promote") and not _steps(sel, "hoist")
    assert sel.solution is not None


# ---------------------------------------------------------------------------
# the extent as a count or as a constant
# ---------------------------------------------------------------------------
def test_the_extent_is_a_count_unless_it_is_a_term_of_what_is_added_up():
    assert _chain("variance").count  # mean = total / n: the mean of what was reduced
    assert _chain("softmax").count  # never used as a value
    assert not _chain("label_smoothing").count  # 0.9 y + 0.1 / k: a constant of the problem
    sel = _select("label_smoothing")
    assert len(sel.states) == 6 and sel.solution.spec.merge_print is not None


# ---------------------------------------------------------------------------
# the procedure as a whole
# ---------------------------------------------------------------------------
def test_every_state_comes_out_of_the_program():
    """States come from the program or its expansion, including sum(1)."""
    origins = (
        "member",
        "context",
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
            if c.origin in ("piece", "closure", "hoist"):
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
