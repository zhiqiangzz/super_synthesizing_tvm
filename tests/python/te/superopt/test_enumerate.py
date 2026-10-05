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
"""M3/M4: pruning, canonical enumeration and algebraic rediscovery."""

import numpy as np
import te_programs

import tvm.testing
from tvm.te.superopt.abstract_expr import contains
from tvm.te.superopt.api import make_context, superoptimize
from tvm.te.superopt.config import Bounds
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import mk_add, mk_exp, mk_mul, mk_reduce


def test_contains_axioms():
    ins, out = te_programs.attention("naive")
    ctx, sems = make_context(out, ins, Bounds())
    target = ctx.target.body
    Q, K, V = (s.body for s in sems)
    dQ = ctx.dims.key(ins[0].shape[3])
    dB = ctx.dims.key(ins[0].shape[0])
    i = [ir.idx(f"i{k}") for k in range(4)]
    j = ir.bidx(0)
    s = mk_reduce(
        "sum",
        ir.dfull(dQ),
        0,
        mk_mul(ir.elem(0, (i[0], i[1], i[2], j)), ir.elem(1, (i[0], i[1], i[3], j))),
    )
    assert contains(target, s)  # Q Kᵀ
    assert contains(target, mk_exp(s))  # exp(s) ⊑ exp(c s)
    assert contains(target, mk_mul(ctx.target.consts[0], s))  # scale
    qv = mk_reduce(
        "sum",
        ir.dfull(dQ),
        0,
        mk_mul(ir.elem(0, (i[0], i[1], i[2], j)), ir.elem(2, (i[0], i[1], i[3], j))),
    )
    assert not contains(target, qv)  # Q V never appears
    assert not contains(
        target, mk_reduce("sum", ir.dfull(dB), 0, ir.elem(0, (j, i[1], i[2], i[3])))
    )
    assert not contains(target, mk_add(Q, V))
    assert contains(target, mk_mul(ir.const(2), s))  # pure constants are abstracted away


def test_rediscovers_scaled_matmul_both_orders():
    ins, out = te_programs.scaled_matmul()
    res = superoptimize(out, ins, Bounds(max_tensor_ops=2), with_reducers=False, accuracy="off")
    programs = {tuple(r.spec.name for r in x.snapshot.ops) for x in res}
    assert ("matmul", "scale") in programs  # alpha (A B)
    assert ("scale", "matmul") in programs  # (alpha A) B
    assert all(r.verdict.method == "hash" for r in res)


def test_softmax_chain_rediscovered_without_shift():
    ins, P = te_programs.softmax_rows()
    res = superoptimize(P, ins, Bounds(max_tensor_ops=3), with_reducers=False, accuracy="off")
    assert res, "expected exp -> sum -> div"
    names = [tuple(r.spec.name for r in x.snapshot.ops) for x in res]
    assert ("exp", "sum", "div") in names
    assert all(r.verdict.proved for r in res)


def test_canonical_order_yields_each_program_once():
    ins, out = te_programs.scaled_matmul()
    res = superoptimize(out, ins, Bounds(max_tensor_ops=3), with_reducers=False, accuracy="off")
    keys = [tuple((r.spec.name, r.operands) for r in x.snapshot.ops) for x in res]
    assert len(keys) == len(set(keys))


def test_branching_factor_guard():
    """Regression guard on pruning power for the attention target."""
    ins, out = te_programs.attention("naive")
    res = superoptimize(out, ins, Bounds(max_tensor_ops=2), with_reducers=False, accuracy="off")
    stats = res[0].ctx.stats if res else None
    if stats is None:
        ctx, sems = make_context(out, ins, Bounds(max_tensor_ops=2))
        from tvm.te.superopt.enumerate import Enumerator
        from tvm.te.superopt.tensor_ops import default_specs

        list(Enumerator(ctx, default_specs(), sems).run())
        stats = ctx.stats
    assert stats["expanded"] < 400
    assert stats["prune:abstract"] > stats["expanded"]


def test_materialized_program_runs_on_llvm():
    ins, out = te_programs.scaled_matmul()
    res = superoptimize(out, ins, Bounds(max_tensor_ops=2), with_reducers=False, accuracy="off")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 4)).astype("float32")
    B = rng.standard_normal((4, 2)).astype("float32")
    ref = (A @ B) / np.sqrt(4.0)
    for r in res:
        got = te_programs.run_llvm(ins, r.materialize(), [A, B], "float32")
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    tvm.testing.main()
