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
"""M1: tensor-level builders -- symbolic semantics must agree with the real TE."""

import te_programs

import tvm.testing
from tvm import te
from tvm.te.superopt.api import make_context
from tvm.te.superopt.config import Bounds
from tvm.te.superopt.pool import Program
from tvm.te.superopt.tensor_ops import default_specs, subseq_matchings


def _setup():
    ins, out = te_programs.attention("naive")
    ctx, sems = make_context(out, ins, Bounds())
    prog = Program(sems)
    return ins, ctx, prog, {s.name: s for s in default_specs()}


def _check(spec, entries, tensors, params, ctx):
    """apply_sem and build agree, and the built TE lowers to a PrimFunc."""
    sems = spec.apply_sem(entries, params, ctx)
    outs = spec.build(tensors, params, ctx)
    assert len(sems) == len(outs)
    for sem, t in zip(sems, outs):
        lowered = ctx.lower.lower(t)
        assert lowered.body is sem.body, f"{spec.name}: {lowered.body} != {sem.body}"
        assert lowered.axis_keys == sem.axis_keys
        assert lowered.dtype == sem.dtype
    te.create_prim_func([*ctx.lower.placeholders, *outs])
    return sems, outs


def test_matmul_symbolic_axes():
    ins, ctx, prog, specs = _setup()
    Q, K, V = prog.pool
    mm = specs["matmul"]
    params = list(mm.params([Q, K], ctx))
    # only head_dim can be contracted (the target never reduces over batch/heads)
    assert params == [(3, 3, ((0, 0), (1, 1)))]
    sems, outs = _check(mm, [Q, K], ins[:2], params[0], ctx)
    assert sems[0].axis_keys == (Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2])
    assert sems[0].dtype == "float32"


def test_ewise_broadcast_by_dropping():
    assert list(subseq_matchings((0, 1), (0, 1, 2))) == [(0, 1)]
    assert list(subseq_matchings((0,), (0, 1, 0))) == [(0,), (2,)]
    ins, ctx, prog, specs = _setup()
    Q, K, V = prog.pool
    mm, red, div = specs["matmul"], specs["sum"], specs["div"]
    p = next(iter(mm.params([Q, K], ctx)))
    S = mm.apply_sem([Q, K], p, ctx)[0]
    S_t = mm.build(ins[:2], p, ctx)[0]
    prog.push(mm, (0, 1), p, [S], ())
    s_entry = prog.pool[3]
    rp = [q for q in red.params([s_entry], ctx)]
    assert rp == [3]  # only the key axis is a target reduction axis of S
    den = red.apply_sem([s_entry], 3, ctx)[0]
    den_t = red.build([S_t], 3, ctx)[0]
    prog.push(red, (3,), 3, [den], ())
    d_entry = prog.pool[4]
    dp = list(div.params([s_entry, d_entry], ctx))
    assert dp == [(0, (0, 1, 2))]
    _check(div, [s_entry, d_entry], [S_t, den_t], dp[0], ctx)


def test_unary_scale_cast_reduce_agree_with_te():
    ins, ctx, prog, specs = _setup()
    Q = prog.pool[0]
    for name in ("exp", "sqrt", "scale", "sum", "max"):
        spec = specs[name]
        for params in spec.params([Q], ctx):
            _check(spec, [Q], [ins[0]], params, ctx)


def test_comm_reduce_build_matches_reference_structure():
    """A synthesised online-softmax reducer builds a 3-output compute like the reference."""
    from tvm.te.superopt.reducer.op import CommReduce

    ins, ctx, prog, specs = _setup()
    Q, K, V = prog.pool
    mm = specs["matmul"]
    p = next(iter(mm.params([Q, K], ctx)))
    S = mm.apply_sem([Q, K], p, ctx)[0]
    S_t = mm.build(ins[:2], p, ctx)[0]
    prog.push(mm, (0, 1), p, [S], ())
    cr = CommReduce()
    realized = list(cr.params([prog.pool[3], V], ctx))
    assert len(realized) >= 2
    for r in realized:
        assert r.spec.arity in (2, 3)
        sems, outs = _check(cr, [prog.pool[3], V], [S_t, ins[2]], r, ctx)
        if r.fused:  # the epilogue (e.g. o / l) reads the reducer's outputs
            assert len(outs) == 1 and outs[0].op.name == "fin"
            continue
        assert len(outs) == r.spec.arity
        assert all(o.op.same_as(outs[0].op) for o in outs)
    # one of them is exactly the reference: leaves (c*S, 1, V) with identity (-inf, 0, 0)
    leaf_kinds = {tuple(sorted(t[0] for t in r.leaf_trees)) for r in realized}
    assert ("const", "elem", "mul") in leaf_kinds


if __name__ == "__main__":
    tvm.testing.main()
