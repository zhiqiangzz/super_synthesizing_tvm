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
"""Write a fused chain back as TE and rebuild the rest of the program around it.

The chain becomes one ``te.comm_reducer`` compute over the chain coordinates:
its inputs are the reducer's leaves evaluated on the boundary tensors, its
merge is defined for two non-empty parts. The first leaf initializes the
reduction; subsequent leaves are merged into it. Parallel execution must
likewise skip empty partitions, rather than merge padding values. Every
member the program still reads gets an *epilogue* compute over its own axes
that forms it from the state outputs. The ops downstream are the original
ones with those reads replaced; nothing else in the program is touched.
"""

from __future__ import annotations

import tvm_ffi

import tvm
from tvm import te
from tvm_tirx import tirx as tir
from tvm_tirx.ir import Call

from ..loads import is_load, load_tensor
from ..symbolic import ir
from ..symbolic.realize import RealizeEnv, to_prim
from .chain import Chain
from .derive import ReducerSpec


def same_tensor(a: te.Tensor, b: te.Tensor) -> bool:
    return a.op.same_as(b.op) and int(a.value_index) == int(b.value_index)


def _swap(expr, lookup):
    """``expr`` with its tensor reads redirected; everything else is kept."""

    def visit(e):
        if not is_load(e):
            return e
        new = lookup(load_tensor(e))
        return e if new is None else new(*list(e.args)[1:])

    return tvm_ffi.structural_map(expr, (Call, visit))


class Rewriter:
    """The program with some tensors replaced: every op that read them is rebuilt."""

    def __init__(self, replaced: list[tuple[te.Tensor, te.Tensor]] | None = None) -> None:
        self.replaced = list(replaced or [])  # (old tensor, new tensor), as decided
        self._ops: list[tuple[object, object]] = []  # op -> its rebuilt op (or itself)

    def fork(self) -> Rewriter:
        """A copy to try one more replacement on."""
        return Rewriter(self.replaced)

    def replace(self, old: te.Tensor, new: te.Tensor) -> None:
        self.replaced.append((old, new))

    def tensor(self, t: te.Tensor) -> te.Tensor:
        """``t`` as it is in the rewritten program."""
        for old, new in self.replaced:
            if same_tensor(old, t):
                return new
        op = t.op
        if not isinstance(op, te.ComputeOp):
            return t
        new_op = next((new for known, new in self._ops if known.same_as(op)), None)
        if new_op is None:
            swaps = [(x, self.tensor(x)) for x in op.input_tensors]
            swaps = [(old, new) for old, new in swaps if not same_tensor(old, new)]
            new_op = _rebuild(op, swaps) if swaps else op
            self._ops.append((op, new_op))
        # an op nothing changed for keeps its tensors: the same objects as before
        return t if new_op.same_as(op) else new_op.output(int(t.value_index))


def _rebuild(op, swaps: list[tuple[te.Tensor, te.Tensor]]):
    """``op`` reading the new tensors instead of the old ones."""

    def lookup(t: te.Tensor):
        return next((new for old, new in swaps if same_tensor(old, t)), None)

    first = op.body[0]
    if isinstance(first, te.Reduce):
        # the outputs of a tuple reduction must share one source array: rewrite the
        # pieces by hand
        source = _swap(first.source, lookup)
        cond = _swap(first.condition, lookup)
        init = _swap(first.init, lookup)
        comb = first.combiner
        comb = te.CommReducer(
            comb.lhs, comb.rhs, _swap(comb.result, lookup), _swap(comb.identity_element, lookup)
        )
        bodies = [
            te.Reduce(comb, source, first.axis, cond, int(b.value_index), init) for b in op.body
        ]
    else:
        bodies = [_swap(b, lookup) for b in op.body]
    return tvm.te._ffi_api.ComputeOp(op.name, op.tag, op.attrs, op.axis, bodies)


def build_chain(chain: Chain, spec: ReducerSpec, rewriter: Rewriter) -> list[te.Tensor]:
    """Emit the fused reducer of ``chain`` and register its required members with
    ``rewriter``. Returns the state tensors of the reducer."""
    lower = chain.lower
    dtype = str(chain.reductions[0].tensor.dtype)
    n = spec.arity
    rank = len(chain.coord_extents)
    j = te.reduce_axis((1, chain.extent), name=chain.jname)

    def env_for(index: dict) -> RealizeEnv:
        return RealizeEnv(dtype=dtype, dims=lower.dims, index=index, scalars=lower.scalars)

    def load(tid: int, idx: tuple):
        return rewriter.tensor(lower.placeholder(tid))(*idx)

    def fcombine(a, b):
        env = env_for({})
        for k in range(n):
            env.state[ir.state_var("a", k)] = a[k]
            env.state[ir.state_var("b", k)] = b[k]
        memo: dict = {}
        return tuple(to_prim(spec.merge_code[k], env, memo) for k in range(n))

    def fcompute(*idx):
        env = env_for({ir.idx(f"i{c}"): idx[c] for c in range(rank)} | {ir.bidx(0): j})
        env.elem = load
        memo: dict = {}
        leaves = tuple(to_prim(leaf, env, memo) for leaf in spec.leaves)
        env.index[ir.bidx(0)] = 0
        memo = {}
        init = tuple(to_prim(leaf, env, memo) for leaf in spec.leaves)

        def fidentity(*dtypes):
            # TE's initialization hook holds the first leaf, not an empty
            # state. Execution must not inject it into other partial states.
            return init

        reducer = te.comm_reducer(fcombine, fidentity, name=f"fused_{chain.jname}")
        return reducer(leaves, axis=j)

    name = "_".join(m.name for m in chain.reductions)
    states = te.compute(
        chain.coord_extents,
        fcompute,
        name=f"fused_{name}",
        tag="superopt.nonempty",
        varargs_names=list(chain.coord_names),
    )
    states = list(states) if isinstance(states, list | tuple) else [states]

    def finish(*idx):
        env = env_for({ir.idx(f"i{c}"): idx[c] for c in range(rank)} | {ir.bidx(0): 0})
        env.elem = load
        memo: dict = {}
        # A singleton is already a complete non-empty state. Its tail loop has
        # no iterations, so no reduction-block initialization runs in TE.
        return tuple(
            tir.if_then_else(chain.extent == 1, to_prim(leaf, env, memo), state(*idx))
            for leaf, state in zip(spec.leaves, states)
        )

    finished = te.compute(
        chain.coord_extents,
        finish,
        name=f"nonempty_{name}",
        tag="superopt.singleton",
        varargs_names=list(chain.coord_names),
    )
    states = list(finished) if isinstance(finished, list | tuple) else [finished]

    def epilogue(m, expr: ir.SymExpr) -> te.Tensor:
        whole = m.coords == tuple(range(rank))
        if whole and isinstance(expr, ir.Elem) and expr.tensor < 0:
            return states[-expr.tensor - 1]  # the member is a state as it is

        def fepilogue(*idx):
            at = dict(zip(m.coords, idx))
            env = env_for({ir.idx(f"i{c}"): i for c, i in at.items()})

            def elem(tid: int, eidx: tuple):
                if tid >= 0:
                    return load(tid, eidx)
                # a state is the same along the coordinates it does not depend on
                return states[-tid - 1](*[at.get(c, 0) for c in range(rank)])

            env.elem = elem
            return to_prim(expr, env, {})

        names = [iv.var.name for iv in m.op.axis]
        return te.compute(m.tensor.shape, fepilogue, name=m.name, varargs_names=names)

    for m, expr in zip(chain.required, spec.epilogue_code):
        rewriter.replace(m.tensor, epilogue(m, expr))
    return states


__all__ = ["Rewriter", "build_chain", "same_tensor"]
