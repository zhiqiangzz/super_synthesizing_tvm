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
"""Partialisation: reducer states read off the original program, re-based merges."""

import numpy as np
import te_programs

import tvm.testing
from tvm.te.superopt.api import make_context, superoptimize
from tvm.te.superopt.config import Bounds
from tvm.te.superopt.reducer.partial import _proves, candidates, synthesize_partial
from tvm.te.superopt.reducer.synth import SynthesisProblem
from tvm.te.superopt.symbolic import ir


def _problem(ins, out, axis_name):
    ctx, _ = make_context(out, ins, Bounds())
    b = ctx.bounds
    problem = SynthesisProblem(
        ctx.target.body,
        ctx.dims.key_of_name(axis_name),
        b.max_states,
        b.max_merge_nodes,
        ctx.target.sem.axis_keys,
        b.min_states,
        ctx.target.consts,
        b.max_leaf_nodes,
        b.max_state_expr_nodes,
        b.max_latent_atoms,
    )
    return ctx, problem


def _nodes(e):
    stack, out = [e], []
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children())
    return out


def test_candidates_are_the_originals_tensors():
    ctx, problem = _problem(*te_programs.variance_rows(), "m")
    got = {(c.name, c.kind) for c in candidates(problem, ctx)}
    assert got == {("total", "sum"), ("mean", "value"), ("ss", "sum"), ("count", "count")}
    ctx, problem = _problem(*te_programs.attention("naive"), "seqlen_k")
    got = {(c.name, c.kind) for c in candidates(problem, ctx)}
    assert got == {("row_max", "max"), ("den", "sum"), ("PV", "sum")}


def test_variance_merge_is_rebased_on_the_means():
    ctx, problem = _problem(*te_programs.variance_rows(), "m")
    (spec,) = synthesize_partial(problem, ctx)
    assert spec.origin == "partial" and spec.merge_print is not None
    # proven equal to the canonical merge (numerically where the rational forms differ)
    for printed, canonical in zip(spec.merge_print, spec.merge):
        assert _proves(printed, canonical)
    # the canonical M2 merge squares the running means themselves ...
    m2 = next(k for k, leaf in enumerate(spec.leaves) if leaf is ir.ZERO)
    mean = next(k for k, leaf in enumerate(spec.leaves) if isinstance(leaf, ir.Elem))
    a_mean = ir.state_var("a", mean)

    def squares(e):
        return [n for n in _nodes(e) if isinstance(n, ir.Pow) and n.exponent == 2]

    assert any(p.base is a_mean for p in squares(spec.merge[m2]))
    # ... the printed one only squares a difference of means
    printed = squares(spec.merge_print[m2])
    assert printed and all(isinstance(p.base, ir.Add) for p in printed)


def test_attention_merge_keeps_the_max_shift():
    ctx, problem = _problem(*te_programs.attention("naive"), "seqlen_k")
    (spec,) = synthesize_partial(problem, ctx)
    assert spec.merge_print is not None
    # every exp of the printed merge subtracts the merged max
    for m in spec.merge_print:
        for n in _nodes(m):
            if isinstance(n, ir.Exp):
                assert any(isinstance(x, ir.Max) for x in _nodes(n.arg))


def test_e2e_partial_programs_pass_the_gate():
    ins, out = te_programs.attention("naive")
    res = superoptimize(out, ins, Bounds(max_tensor_ops=2))
    assert res and all(r.accuracy.ok for r in res)
    partial = [r for r in res if any(p.spec.origin == "partial" for p in r.reducers())]
    assert [rec.spec.name for rec in partial[0].snapshot.ops] == ["matmul", "comm_reduce"]
    # large logits: the re-based merge stays finite and accurate
    rng = np.random.default_rng(0)
    b, h, q, k, d = 1, 2, 3, 40, 4
    Q = (30 * rng.standard_normal((b, h, q, d))).astype("float32")
    K = (30 * rng.standard_normal((b, h, k, d))).astype("float32")
    V = rng.standard_normal((b, h, k, d)).astype("float32")
    got = te_programs.run_llvm(ins, partial[0].materialize(), [Q, K, V], "float32")
    np.testing.assert_allclose(got, te_programs.attention_reference(Q, K, V), atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
