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
"""A numpy interpreter for symbolic semantics.

Used as the cheap dev-time oracle: extents are instantiated to tiny integers
and inputs to random float64 arrays. Arrays are kept in *broadcast form*:
at binder depth ``d`` every value has ``rank + d`` dimensions, one per output
index followed by one per enclosing reduction binder.
"""

from __future__ import annotations

import numpy as np

from ..dims import DimKey
from . import ir
from .canonicalize import Unsupported
from .lower import TensorSem


class Interp:
    def __init__(self, data: dict[int, np.ndarray], extents: dict[DimKey, int], rank: int):
        self.data = data
        self.extents = extents
        self.rank = rank

    def index(self, e: ir.IndexExpr, depth: int) -> np.ndarray | int:
        nd = self.rank + depth
        if isinstance(e, ir.IConst):
            return int(e.value)
        if isinstance(e, ir.Idx):
            pos = int(e.name[1:])
            ext = self.extents[("out", pos)]
        elif isinstance(e, ir.BIdx):
            pos = self.rank + e.level
            ext = self.extents[("lvl", e.level)]
        else:
            raise Unsupported(f"index {e}")
        shape = [1] * nd
        shape[pos] = ext
        return np.arange(ext).reshape(shape)

    def ev(self, e: ir.SymExpr, depth: int):
        if isinstance(e, ir.Const):
            return float(e.value)
        if isinstance(e, ir.ShapeSym):
            return float(self.extents[("name", e.name)])
        if isinstance(e, ir.Card):
            if isinstance(e.domain, ir.DFull):
                return float(self.extents[("axis", e.domain.axis)])
            raise Unsupported("cardinality of an abstract domain")
        if isinstance(e, ir.Elem):
            ix = tuple(self.index(i, depth) for i in e.indices)
            return self.data[e.tensor][ix]
        if isinstance(e, ir.Add):
            out = 0.0
            for a in e.args:
                out = out + self.ev(a, depth)
            return out
        if isinstance(e, ir.Mul):
            out = 1.0
            for a in e.args:
                out = out * self.ev(a, depth)
            return out
        if isinstance(e, ir.Pow):
            return np.power(self.ev(e.base, depth), float(e.exponent))
        if isinstance(e, ir.Exp):
            return np.exp(self.ev(e.arg, depth))
        if isinstance(e, ir.Log):
            return np.log(self.ev(e.arg, depth))
        if isinstance(e, ir.Max):
            out = self.ev(e.args[0], depth)
            for a in e.args[1:]:
                out = np.maximum(out, self.ev(a, depth))
            return out
        if isinstance(e, ir.Reduce):
            return self.reduce(e, depth)
        if isinstance(e, ir.MonoidReduce):
            return self.monoid(e, depth)
        raise Unsupported(f"cannot evaluate {type(e).__name__}")

    def _extent_of(self, domain: ir.Domain) -> int:
        if isinstance(domain, ir.DFull):
            return self.extents[("axis", domain.axis)]
        raise Unsupported("evaluation over an abstract domain")

    def _full(self, value, depth: int, axis_pos: int, n: int) -> np.ndarray:
        nd = self.rank + depth
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim < nd:
            arr = arr.reshape(arr.shape + (1,) * (nd - arr.ndim))
        shape = list(arr.shape)
        shape[axis_pos] = n
        return np.broadcast_to(arr, shape)

    def reduce(self, e: ir.Reduce, depth: int):
        assert e.level == depth
        n = self._extent_of(e.domain)
        self.extents[("lvl", e.level)] = n
        body = self.ev(e.body, depth + 1)
        pos = self.rank + e.level
        body = self._full(body, depth + 1, pos, n)
        return body.sum(axis=pos) if e.kind == "sum" else body.max(axis=pos)

    def monoid(self, e: ir.MonoidReduce, depth: int):
        assert e.level == depth
        n = self._extent_of(e.domain)
        self.extents[("lvl", e.level)] = n
        pos = self.rank + e.level
        leaves = [self._full(self.ev(x, depth + 1), depth + 1, pos, n) for x in e.leaf]
        state = [np.asarray(float(c.value)) for c in e.identity]
        for j in range(n):
            cur = [np.take(x, j, axis=pos) for x in leaves]
            env = {}
            for k in range(len(state)):
                env[ir.state_var("a", k)] = state[k]
                env[ir.state_var("b", k)] = cur[k]
            state = [self.ev_merge(m, env) for m in e.merge]
        return state[e.slot]

    def ev_merge(self, e: ir.SymExpr, env):
        if isinstance(e, ir.StateVar):
            return env[e]
        if isinstance(e, ir.Const):
            return float(e.value)
        if isinstance(e, ir.ShapeSym):
            return float(self.extents[("name", e.name)])
        if isinstance(e, ir.Add):
            out = 0.0
            for a in e.args:
                out = out + self.ev_merge(a, env)
            return out
        if isinstance(e, ir.Mul):
            out = 1.0
            for a in e.args:
                out = out * self.ev_merge(a, env)
            return out
        if isinstance(e, ir.Pow):
            return np.power(self.ev_merge(e.base, env), float(e.exponent))
        if isinstance(e, ir.Exp):
            return np.exp(self.ev_merge(e.arg, env))
        if isinstance(e, ir.Log):
            return np.log(self.ev_merge(e.arg, env))
        if isinstance(e, ir.Max):
            out = self.ev_merge(e.args[0], env)
            for a in e.args[1:]:
                out = np.maximum(out, self.ev_merge(a, env))
            return out
        raise Unsupported(f"cannot evaluate merge {type(e).__name__}")


def evaluate(
    sem: TensorSem,
    data: dict[int, np.ndarray],
    axis_extents: dict[DimKey, int],
    shape_names: dict[str, int],
) -> np.ndarray:
    """Evaluate ``sem`` on placeholder ``data`` (keyed by tensor id) as float64."""
    extents: dict = {("axis", k): v for k, v in axis_extents.items()}
    for pos, k in enumerate(sem.axis_keys):
        extents[("out", pos)] = axis_extents[k]
    for name, v in shape_names.items():
        extents[("name", name)] = v
    interp = Interp(data, extents, sem.rank)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = interp.ev(sem.body, 0)
    shape = tuple(axis_extents[k] for k in sem.axis_keys)
    arr = np.asarray(out, dtype=np.float64)
    if arr.ndim < len(shape):
        arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    return np.broadcast_to(arr, shape).copy()
