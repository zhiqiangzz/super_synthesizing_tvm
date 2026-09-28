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
"""M5-M9: reducer synthesis, from the sum pair to full attention."""

import itertools
import time

import numpy as np
import te_programs

import tvm.testing
from tvm import te
from tvm.te.superopt.api import make_context, superoptimize
from tvm.te.superopt.config import Bounds
from tvm.te.superopt.reducer import check_laws, synthesize
from tvm.te.superopt.reducer.synth import SynthesisProblem, extract_goals
from tvm.te.superopt.symbolic import LowerCtx, ir
from tvm.te.superopt.symbolic.canonicalize import mk_add, mk_exp, mk_max, mk_mul, mk_sub, subst


def _attention_problem():
    ins, out = te_programs.attention("naive")
    ctx, sems = make_context(out, ins, Bounds())
    axis = ctx.dims.key(ins[1].shape[2])  # seqlen_k
    return ctx, axis


# ---------------------------------------------------------------------------
# M5: laws and the simplest tuple reducer
# ---------------------------------------------------------------------------
def _sv(side, k):
    return ir.state_var(side, k)


def test_laws_sum_pair_and_rejections():
    a0, a1, b0, b1 = _sv("a", 0), _sv("a", 1), _sv("b", 0), _sv("b", 1)
    assert check_laws((mk_add(a0, b0), mk_add(a1, b1)), (ir.ZERO, ir.ZERO)) == "canonical"
    assert check_laws((mk_sub(a0, b0),), (ir.ZERO,)) is None  # not commutative
    assert check_laws((mk_add(a0, b0),), (ir.ONE,)) is None  # wrong identity
    assert check_laws((mk_max(a0, b0),), (ir.NEG_INF_C,)) == "canonical"
    assert check_laws((mk_max(a0, b0),), (ir.ZERO,)) is None


def test_laws_online_softmax_reference_merge():
    a0, a1, a2 = (_sv("a", k) for k in range(3))
    b0, b1, b2 = (_sv("b", k) for k in range(3))
    m = mk_max(a0, b0)
    ra, rb = mk_exp(mk_sub(a0, m)), mk_exp(mk_sub(b0, m))
    merge = (m, mk_add(mk_mul(a1, ra), mk_mul(b1, rb)), mk_add(mk_mul(a2, ra), mk_mul(b2, rb)))
    assert check_laws(merge, (ir.NEG_INF_C, ir.ZERO, ir.ZERO)) in ("canonical", "case-split")
    # breaking the rescale breaks associativity
    bad = (m, mk_add(mk_mul(a1, ra), b1), mk_add(mk_mul(a2, ra), mk_mul(b2, rb)))
    assert check_laws(bad, (ir.NEG_INF_C, ir.ZERO, ir.ZERO)) is None


def test_e2e_sum_pair():
    ins, T = te_programs.sum_pair()
    res = superoptimize(T, ins, Bounds(max_tensor_ops=2, max_states=2))
    reducers = [r for r in res if r.reducers()]
    assert reducers, "expected a 2-state (Σx, Σy) reducer followed by add"
    spec = reducers[0].reducers()[0].spec
    assert spec.arity == 2
    assert spec.identity == (ir.ZERO, ir.ZERO)
    assert all(m is mk_add(_sv("a", k), _sv("b", k)) for k, m in enumerate(spec.merge))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 4)).astype("float32")
    Y = rng.standard_normal((3, 4)).astype("float32")
    got = te_programs.run_llvm(ins, reducers[0].materialize(), [X, Y], "float32")
    np.testing.assert_allclose(got, X.sum(1) + Y.sum(1), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# M6: goal analysis
# ---------------------------------------------------------------------------
def test_goal_signals_and_latent_atoms_attention():
    """The grammar's raw material comes from the target only: no algorithm is encoded."""
    from tvm.te.superopt.reducer import grammar
    from tvm.te.superopt.symbolic.canonicalize import subst_domain

    ctx, axis = _attention_problem()
    goals = extract_goals(ctx.target.body, axis)
    assert len(goals) == 2 and all(g.kind == "sum" for g in goals)
    sigs = grammar.signals(goals)
    # s_j, c s_j, exp(c s_j), v_j, v_j exp(c s_j)
    assert len(sigs) == 5
    R = ir.dsym(axis, "R")
    goal_atoms = [subst_domain(g, {ir.dfull(axis): R}) for g in goals]
    latent = [a for a in grammar.atom_candidates(sigs, R) if a not in goal_atoms]
    kinds = sorted(("card" if isinstance(a, ir.Card) else a.kind) for a in latent)
    assert "card" in kinds and kinds.count("max") >= 2 and kinds.count("sum") >= 1


def test_solve_atoms_and_state_relevance():
    ctx, axis = _attention_problem()
    specs = synthesize(
        SynthesisProblem(
            ctx.target.body,
            axis,
            3,
            24,
            ctx.target.sem.axis_keys,
            1,
            ctx.target.consts,
            max_leaf_nodes=6,  # room for the leaf v exp(c s) of the plain (L, O) pair
        )
    )
    two = [s for s in specs if s.arity == 2]
    # (L, O) with plain sums, plus the log-domain / normalised-output variants
    assert any(all(not _has_exp(m) and not isinstance(m, ir.Log) for m in s.merge) for s in two)
    assert any(any(isinstance(m, ir.Log) for m in s.merge) for s in two)
    assert all(s.proof in ("canonical", "case-split", "numeric") for s in specs)
    stable = []
    for s in specs:
        if s.arity == 3 and sum(isinstance(m, ir.Max) for m in s.merge) == 1:
            ids = sorted(str(i) for i in s.identity)
            assert ids[0] == "-inf" and set(ids) <= {"-inf", "0"}
            if all(isinstance(m, ir.Max) or _has_exp(m) for m in s.merge):
                stable.append(s)
    # the classic online-softmax variant: leaf tuple (s, 1, v)
    classic = [s for s in stable if ir.ONE in s.leaves]
    assert len(classic) >= 1, [s.pretty() for s in specs]


def _has_exp(e):
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Exp):
            return True
        stack.extend(n.children())
    return False


# ---------------------------------------------------------------------------
# M7/M8: softmax * V with explicit scores
# ---------------------------------------------------------------------------
def _spec_signature(spec):
    return sorted(str(leaf) for leaf in spec.leaves)


def test_e2e_softmax_value_unstable_and_stable():
    ins, out = te_programs.softmax_value(stable=True)
    res = superoptimize(out, ins, Bounds(max_tensor_ops=3, max_states=3, leaf_exp=True))
    specs = [rec.spec for r in res for rec in r.reducers()]
    assert specs
    arities = {s.arity for s in specs}
    assert arities == {2, 3}  # (L, out) with exp leaves and (m, l, o)
    stable = [s for s in specs if s.arity == 3 and ir.NEG_INF_C in s.identity]
    assert stable, [s.pretty() for s in specs]
    rng = np.random.default_rng(1)
    S = rng.standard_normal((3, 5)).astype("float32")
    V = rng.standard_normal((5, 2)).astype("float32")
    P = np.exp(S - S.max(1, keepdims=True))
    ref = (P @ V) / P.sum(1, keepdims=True)
    for r in res:
        got = te_programs.run_llvm(ins, r.materialize(), [S, V], "float32")
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_logspace_normalised_reducer_is_synthesised():
    """The (log-sum-exp, normalised output) reducer is derived and matches the reference."""
    ins, out = te_programs.softmax_value(stable=True)
    res = superoptimize(out, ins, Bounds(max_tensor_ops=1, max_states=2))
    specs = [rec.spec for r in res for rec in r.reducers()]
    assert specs, "expected one-op reducer programs"
    ref_merge, ref_identity = _reference_merge_of(te_programs.softmax_value_logspace())
    logspace = [s for s in specs if _same_reducer(s, ref_merge, ref_identity)]
    assert logspace, [s.pretty() for s in specs]
    spec = logspace[0]
    assert any(isinstance(m, ir.Log) for m in spec.merge)
    assert sorted(str(leaf) for leaf in spec.leaves) == sorted(["T0[i0,j0]", "T1[j0,i1]"])
    # runs and matches numpy
    rng = np.random.default_rng(2)
    S = rng.standard_normal((3, 5)).astype("float32")
    V = rng.standard_normal((5, 2)).astype("float32")
    P = np.exp(S - S.max(1, keepdims=True))
    ref = (P @ V) / P.sum(1, keepdims=True)
    for r in res:
        if r.reducers() and _same_reducer(r.reducers()[0].spec, ref_merge, ref_identity):
            got = te_programs.run_llvm(ins, r.materialize(), [S, V], "float32")
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
            src = r.te_source()
            assert "tir.log(" in src


def test_reducer_goal_nested_under_projection():
    """A goal inside a later reduction (``Σ_e O W``) is still synthesised."""
    ins, out = te_programs.attention_projected("naive")
    res = superoptimize(out, ins, Bounds(max_tensor_ops=3, max_states=2))
    progs = [[o.spec.name for o in r.snapshot.ops] for r in res]
    assert ["matmul", "comm_reduce", "matmul"] in progs, progs
    rng = np.random.default_rng(3)
    b, h, q, k, d, f = 1, 2, 3, 5, 4, 3
    Q = rng.standard_normal((b, h, q, d)).astype("float32")
    K = rng.standard_normal((b, h, k, d)).astype("float32")
    V = rng.standard_normal((b, h, k, d)).astype("float32")
    W = rng.standard_normal((d, f)).astype("float32")
    ref = te_programs.attention_reference(Q, K, V) @ W.astype("float64")
    for r in res:
        got = te_programs.run_llvm(ins, r.materialize(), [Q, K, V, W], "float32")
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_min_states_skips_single_state_reducers():
    ins, out = te_programs.attention("naive")
    default = superoptimize(out, ins, Bounds(max_tensor_ops=2, max_states=2))
    assert all(rec.spec.arity >= 2 for r in default for rec in r.reducers())
    assert all(r.snapshot.ops[0].spec.name == "matmul" for r in default)
    with_single = superoptimize(out, ins, Bounds(max_tensor_ops=2, max_states=2, min_states=1))
    firsts = {r.snapshot.ops[0].spec.name for r in with_single}
    assert firsts == {"matmul", "comm_reduce"}  # S as a synthesised sum reducer too
    single = [rec.spec for r in with_single for rec in r.reducers() if rec.spec.arity == 1]
    assert single and all(str(s.merge[0]) == "(a0 + b0)" for s in single)


def _divides_by_count(merge, count: int) -> bool:
    a, b = ir.state_var("a", count), ir.state_var("b", count)
    stack = [merge]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Pow) and n.exponent < 0 and isinstance(n.base, ir.Add):
            if set(n.base.args) == {a, b}:
                return True
        stack.extend(n.children())
    return False


def _reference_merge_of(program):
    ins, out = program
    sem = LowerCtx().lower(out)
    stack = [sem.body]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.MonoidReduce):
            return n.merge, n.identity
        stack.extend(n.children())
    raise AssertionError("no reducer in reference")


def test_welford_variance_reducer_is_synthesised():
    """Row variance: a (count, mean, M2) reducer is derived from the target alone."""
    ins, out = te_programs.variance_rows()
    res = superoptimize(out, ins, Bounds(max_tensor_ops=2, max_states=3))
    specs = [rec.spec for r in res for rec in r.reducers()]
    welford = []
    for s in specs:
        if s.arity != 3 or sorted(str(leaf) for leaf in s.leaves) != ["0", "1", "T0[i0,j0]"]:
            continue
        count = s.leaves.index(ir.ONE)
        mean = next(k for k, leaf in enumerate(s.leaves) if isinstance(leaf, ir.Elem))
        m2 = s.leaves.index(ir.ZERO)
        if str(s.merge[count]) != f"(a{count} + b{count})":
            continue
        # mean merge is (n_a m_a + n_b m_b) / (n_a + n_b); M2 merge divides by the
        # merged count too, and stays finite against the identity (0, 0, 0)
        if not _divides_by_count(s.merge[mean], count) or not _divides_by_count(s.merge[m2], count):
            continue
        welford.append(s)
    assert welford, [s.pretty() for s in specs if s.arity == 3]
    spec = welford[0]
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 7)).astype("float32")
    ref = X.astype("float64").var(axis=1)
    for r in res:
        if r.reducers() and r.reducers()[0].spec is spec:
            got = te_programs.run_llvm(ins, r.materialize(), [X], "float32")
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
            break
    else:
        raise AssertionError("welford program not materialised")


# ---------------------------------------------------------------------------
# M9: attention end to end
# ---------------------------------------------------------------------------
def _is_online_softmax(spec) -> bool:
    """Three states (running max, rescaled denominator, rescaled output): leaf tuple
    (score, 1, value), a max merge, identity (-inf, 0, 0) and exp rescaling."""
    if spec.arity != 3 or ir.ONE not in spec.leaves:
        return False
    maxes = [k for k, m in enumerate(spec.merge) if isinstance(m, ir.Max)]
    if len(maxes) != 1 or spec.identity[maxes[0]] is not ir.NEG_INF_C:
        return False
    return all(_has_exp(m) for k, m in enumerate(spec.merge) if k != maxes[0])


def _reference_merge():
    """Canonical merge tuple of the hand-written online_softmax reducer."""
    ins, out = te_programs.attention("flash")
    sem = LowerCtx().lower(out)
    stack = [sem.body]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.MonoidReduce):
            return n.merge, n.identity
        stack.extend(n.children())
    raise AssertionError("no reducer in reference")


def _same_reducer(spec, ref_merge, ref_identity):
    """Match modulo slot permutation."""
    n = spec.arity
    for perm in itertools.permutations(range(n)):
        ren = {}
        for k in range(n):
            ren[ir.state_var("a", k)] = ir.state_var("a", perm[k])
            ren[ir.state_var("b", k)] = ir.state_var("b", perm[k])
        merge = [subst(spec.merge[k], ren) for k in range(n)]
        if all(merge[k] is ref_merge[perm[k]] for k in range(n)) and all(
            spec.identity[k] is ref_identity[perm[k]] for k in range(n)
        ):
            return True
    return False


def test_e2e_attention_discovers_flash_attention():
    ins, out = te_programs.attention("naive")
    t0 = time.time()
    res = superoptimize(out, ins, Bounds(max_tensor_ops=3))
    elapsed = time.time() - t0
    assert res, "no equivalent program found"
    assert elapsed < 300
    assert all(r.verdict.method == "hash" for r in res)
    flash = [r for r in res if r.reducers() and _is_online_softmax(r.reducers()[0].spec)]
    assert flash, "no online-softmax reducer among the results"
    assert [rec.spec.name for rec in flash[0].snapshot.ops] == ["matmul", "comm_reduce", "div"]
    # the discovered kernel runs and matches the reference numerically
    rng = np.random.default_rng(0)
    b, h, q, k, d = 2, 2, 3, 5, 4
    Q = rng.standard_normal((b, h, q, d)).astype("float32")
    K = rng.standard_normal((b, h, k, d)).astype("float32")
    V = rng.standard_normal((b, h, k, d)).astype("float32")
    ref = te_programs.attention_reference(Q, K, V)
    naive = te_programs.run_llvm(ins, out, [Q, K, V], "float32").astype("float64")
    for r in res:
        got = te_programs.run_llvm(ins, r.materialize(), [Q, K, V], "float32").astype("float64")
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got, naive, rtol=1e-4, atol=1e-5)


def test_te_source_rebuilds_the_same_program():
    """Executing the printed TE source yields the same PrimFunc as materialize()."""
    import tvm

    ins, out = te_programs.attention("naive")
    res = superoptimize(out, ins, Bounds(max_tensor_ops=3), max_results=2)
    assert res
    for r in res:
        src = r.te_source()
        assert "te.comm_reducer(" in src and "te.compute(" in src
        ns: dict = {}
        exec(src, ns)
        names = [t.op.name for t in ins]
        rebuilt_inputs = [ns[name] for name in names]
        pf_a = te.create_prim_func([*rebuilt_inputs, ns["output"]])
        pf_b = r.prim_func()
        tvm.ir.assert_structural_equal(pf_a, pf_b, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
