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
"""Reading reduction chains off a TE graph: members, boundary, stages, coordinates."""

from chain_ops import OPERATORS

import tvm.testing
from tvm import te
from tvm import tirx as tir
from tvm.te.superopt.reducer.chain import PSEUDO, discover_chains, extent_symbol
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import contains_node, walk


def _chains(name: str):
    _, outs = OPERATORS[name].unfused()
    return discover_chains(outs)


def _names(tensors) -> list[str]:
    return sorted(t.op.name for t in tensors)


def _pinned(struct) -> list[ir.Elem]:
    """Reads of other members that sit inside a reduction (they carry its index)."""
    return [n for n in walk(struct) if isinstance(n, ir.Elem) and n.tensor <= PSEUDO and n.levels]


# ---------------------------------------------------------------------------
# members, required, boundary
# ---------------------------------------------------------------------------
def test_attention_chain():
    (chain,), skipped = _chains("attention")
    assert not skipped
    assert [m.name for m in chain.members] == ["row_max", "den", "O"]
    assert [m.kind for m in chain.members] == ["max", "sum", "sum"]
    # the attention matrix and the exponentials are absorbed: only the output is read
    assert [m.name for m in chain.required] == ["O"]
    # the scaled scores are read once per element and looked through to the matmul
    assert _names(chain.boundary) == ["QK", "V"]
    # row maximum and normaliser live on (batch, head, query); the output adds its own axis
    assert [m.coords for m in chain.members] == [(0, 1, 2), (0, 1, 2), (0, 1, 2, 3)]
    assert len(chain.coord_extents) == 4 and chain.jname == "j"
    # the output reads both contexts from inside its reduction
    out = chain.members[2]
    assert sorted(n.tensor for n in _pinned(out.body)) == [PSEUDO - 1, PSEUDO]


def test_required_members_follow_what_the_program_still_reads():
    (chain,), _ = _chains("softmax")
    assert [m.name for m in chain.required] == ["mx", "den"]  # P needs both
    (chain,), _ = _chains("layernorm")
    assert [(m.name, m.kind) for m in chain.members] == [
        ("total_mean", "sum"),
        ("mean", "value"),
        ("ss", "sum"),
    ]
    # the normalised output reads the mean itself, not the sum behind it
    assert [m.name for m in chain.required] == ["mean", "ss"]
    (chain,), _ = _chains("variance")
    assert [m.name for m in chain.required] == ["ss"]


def test_a_reduction_without_context_joins_the_chain_it_is_read_with():
    """Cut cross-entropy: ``Σ y l`` reads no maximum but must share the pass over the logits."""
    (chain,), _ = _chains("cross_entropy")
    assert [m.name for m in chain.members] == ["mx", "den", "t"]
    assert not _pinned(chain.members[2].body)
    assert _names(chain.boundary) == ["Y", "logits"]


def test_unrelated_reductions_are_no_chain():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    s1 = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="s1")
    s2 = te.compute((n,), lambda i: te.sum(X[i, j2] * X[i, j2], axis=j2), name="s2")
    out = te.compute((n,), lambda i: s2[i] - s1[i] * s1[i], name="out")
    assert discover_chains(out) == ([], [])


def _variance(m, mean_inline: bool):
    n = te.var("n")
    X = te.placeholder((n, m), name="X")
    count = tir.const(float(m), "float32") if isinstance(m, int) else m.astype("float32")
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total")
    if mean_inline:
        centre = lambda i: total[i] / count  # noqa: E731
    else:
        mean = te.compute((n,), lambda i: total[i] / count, name="mean")
        centre = lambda i: mean[i]  # noqa: E731
    ss = te.compute(
        (n,), lambda i: te.sum((X[i, j2] - centre(i)) * (X[i, j2] - centre(i)), axis=j2), name="ss"
    )
    return X, ss


def test_value_computed_inside_the_reduction_is_a_member():
    """``total[i] / m`` written in place is the mean all the same: a context without a tensor."""
    _, ss = _variance(te.var("m"), mean_inline=True)
    (chain,), _ = discover_chains(ss)
    assert [(m.name, m.kind, m.op is None) for m in chain.members] == [
        ("total", "sum", False),
        ("ss.ctx0", "value", True),
        ("ss", "sum", False),
    ]
    centre, squares = chain.members[1], chain.members[2]
    assert {n.tensor for n in _pinned(squares.body)} == {centre.pid}  # read once, as a unit
    assert not _pinned(centre.body) and centre.coords == (0,)
    # it is the same definition as the named mean of the other spelling
    _, named = _variance(te.var("m"), mean_inline=False)
    (other,), _ = discover_chains(named)
    assert [m.name for m in other.members] == ["total", "mean", "ss"]


def test_constant_extent_written_as_a_number_is_still_the_extent():
    _, ss = _variance(16, mean_inline=False)
    (chain,), _ = discover_chains(ss)
    symbol = extent_symbol(chain.axis)
    mean = chain.members[1]
    assert contains_node(mean.body, lambda x: x is symbol)
    assert not contains_node(mean.body, lambda x: x is ir.const(16))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def test_sinkhorn_iteration_is_two_chains():
    chains, skipped = _chains("sinkhorn_iteration")
    assert not skipped and len(chains) == 2
    first, second = chains
    assert [m.name for m in first.members] == ["S_max", "S_sumexp"]
    assert [m.name for m in second.members] == ["S2_max", "S2_sumexp"]
    assert first.axis != second.axis and (first.jname, second.jname) == ("j", "i")
    # the first potential is shared by every column: it stays a stored tensor
    assert _names(first.boundary) == ["C", "b", "g"]
    assert _names(second.boundary) == ["C", "a", "f"]
    assert [m.name for m in first.required] == ["S_max", "S_sumexp"]


def test_equal_extents_do_not_merge_the_two_half_steps():
    chains, _ = _chains("sinkhorn_square")
    assert len(chains) == 2
    first, second = chains
    assert first.axis == second.axis  # the same extent ...
    assert (first.stage, second.stage) == (0, 1)  # ... but the second reads the first along it
    assert _names(second.boundary) == ["C", "a", "f"]


def test_square_softmax_pins_by_index_not_by_extent():
    n = te.var("n")
    X = te.placeholder((n, n), name="X")
    j1, j2 = te.reduce_axis((0, n), "j"), te.reduce_axis((0, n), "j")
    mx = te.compute((n,), lambda i: te.max(X[i, j1], axis=j1), name="mx")
    den = te.compute((n,), lambda i: te.sum(tir.exp(X[i, j2] - mx[i]), axis=j2), name="den")
    P = te.compute((n, n), lambda i, j: tir.exp(X[i, j] - mx[i]) / den[i], name="P")
    (chain,), _ = discover_chains(P)
    assert [m.name for m in chain.members] == ["mx", "den"]
    (read,) = _pinned(chain.members[1].body)
    assert read.indices == (ir.idx("i0"), ir.bidx(0))


# ---------------------------------------------------------------------------
# outside the supported shape of a chain
# ---------------------------------------------------------------------------
def test_one_tensor_at_two_indices_is_reported():
    chains, (skip,) = _chains("cov_matrix")
    assert not chains and skip.members == ["total_mean", "C"]
    assert "two different indices of C" in skip.reason


def test_incomparable_index_spaces_are_reported():
    """Two outputs sharing one softmax, each with an axis of its own."""
    n, m, k, l = (te.var(x) for x in "nmkl")  # noqa: E741
    S = te.placeholder((n, m), name="S")
    V = te.placeholder((m, k), name="V")
    W = te.placeholder((m, l), name="W")
    j1, j2, j3, j4 = (te.reduce_axis((0, m), "j") for _ in range(4))
    mx = te.compute((n,), lambda i: te.max(S[i, j1], axis=j1), name="mx")
    den = te.compute((n,), lambda i: te.sum(tir.exp(S[i, j2] - mx[i]), axis=j2), name="den")
    a = te.compute(
        (n, k),
        lambda i, c: te.sum(tir.exp(S[i, j3] - mx[i]) / den[i] * V[j3, c], axis=j3),
        name="a",
    )
    b = te.compute(
        (n, l),
        lambda i, c: te.sum(tir.exp(S[i, j4] - mx[i]) / den[i] * W[j4, c], axis=j4),
        name="b",
    )
    chains, (skip,) = discover_chains([a, b])
    assert not chains and skip.members == ["mx", "den", "a", "b"]
    assert "no reduction spans" in skip.reason


def test_shadowed_loop_variable_is_reported():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X")
    r1, r2 = te.reduce_axis((0, m), "i"), te.reduce_axis((0, m), "i")
    mx = te.compute((n,), lambda i: te.max(X[i, r1], axis=r1), name="mx")
    den = te.compute((n,), lambda i: te.sum(tir.exp(X[i, r2] - mx[i]), axis=r2), name="den")
    chains, (skip,) = discover_chains([mx, den])
    assert not chains and "shadows" in skip.reason


def test_minimum_is_a_reduction_of_the_chain():
    """``te.min`` is a link like ``te.max``; in the IR it is ``-max`` of the negation."""
    (chain,), skipped = _chains("minmax_mean")
    assert not skipped
    assert [(m.name, m.kind) for m in chain.members] == [
        ("lo", "min"),
        ("hi", "max"),
        ("scaled.ctx0", "value"),
        ("scaled", "sum"),
    ]
    lo = chain.members[0]
    assert ir.min_args(lo.body) is None and lo.body.args[0] is ir.MINUS_ONE
    (red,) = [n for n in walk(lo.body) if isinstance(n, ir.Reduce)]
    assert red.kind == "max" and [m.name for m in chain.reductions] == ["lo", "hi", "scaled"]
    # the weights of a heat capacity wait for the lowest energy the way a softmax waits
    # for the highest score
    (chain,), _ = _chains("heat_capacity")
    assert [m.name for m in chain.members] == ["lo", "e1s", "Z", "e1", "cs"]
    assert {n.tensor for n in _pinned(chain.members[2].body)} == {chain.members[0].pid}


if __name__ == "__main__":
    tvm.testing.main()
