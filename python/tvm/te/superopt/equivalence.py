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
"""Tiered whole-program equivalence checking.

1. canonical-form identity (``a is b``) -- a proof under the rewrite theory;
2. numeric fingerprint on tiny random instances -- rejects, never proves;
3. algebraic proof via the sympy bridge (case-splitting on ``max``);
4. otherwise the verdict is ``numeric-only`` (dev tier evidence).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .dims import DimKey
from .symbolic import ir
from .symbolic.canonicalize import Unsupported
from .symbolic.interp import evaluate
from .symbolic.lower import TensorSem
from .target import SearchCtx


@dataclasses.dataclass(frozen=True)
class Verdict:
    equivalent: bool
    method: str  # "hash" | "algebraic" | "numeric-only" | "rejected:<why>"

    @property
    def proved(self) -> bool:
        return self.equivalent and self.method in ("hash", "algebraic")


class NumericOracle:
    """Random tiny instances of the placeholders for interpreter-based checks."""

    def __init__(self, ctx: SearchCtx, seed: int = 0, n_instances: int = 3):
        self.ctx = ctx
        self.instances: list[tuple[dict, dict, dict]] = []
        rng = np.random.default_rng(seed)
        dims = ctx.dims
        extents_choices = ctx.bounds.test_extents
        for n in range(n_instances):
            ext: dict[DimKey, int] = {}
            for k in dims.all_keys():
                if dims.is_symbolic(k):
                    ext[k] = extents_choices[(k + n) % len(extents_choices)]
                else:
                    ext[k] = int(dims.extent(k).value)
            names = {dims.name(k): v for k, v in ext.items() if dims.is_symbolic(k)}
            data = {}
            for t in ctx.lower.placeholders:
                tid = ctx.lower.tensor_id(t)
                shape = tuple(ext[k] for k in dims.keys(t.shape))
                data[tid] = 0.7 * rng.standard_normal(shape)
            self.instances.append((data, ext, names))

    def refresh(self) -> None:
        """Extents may have grown (new dims registered): rebuild lazily."""
        if len(self.instances[0][1]) != len(self.ctx.dims):
            self.__init__(self.ctx)

    def evaluate(self, sem: TensorSem) -> list[np.ndarray]:
        self.refresh()
        out = []
        for data, ext, names in self.instances:
            out.append(evaluate(sem, data, ext, names))
        return out

    def same(self, a: TensorSem, b: TensorSem, rtol=1e-7, atol=1e-9) -> bool:
        try:
            va = self.evaluate(a)
            vb = self.evaluate(b)
        except Unsupported:
            return False
        for x, y in zip(va, vb):
            if x.shape != y.shape:
                return False
            if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                return False
        return True


def equivalent(
    cand: TensorSem, target: TensorSem, ctx: SearchCtx, oracle: NumericOracle
) -> Verdict:
    if cand.axis_keys != target.axis_keys or cand.dtype != target.dtype:
        return Verdict(False, "rejected:interface")
    if cand.body is target.body:
        return Verdict(True, "hash")
    if not oracle.same(cand, target):
        return Verdict(False, "rejected:numeric")
    try:
        from .symbolic.sympy_bridge import prove_equal

        if prove_equal(cand.body, target.body):
            return Verdict(True, "algebraic")
    except Exception:  # pragma: no cover - sympy is best effort
        pass
    return Verdict(True, "numeric-only")


def fingerprint(sem: TensorSem, oracle: NumericOracle) -> tuple:
    vals = oracle.evaluate(sem)
    return tuple(np.round(v, 6).tobytes() for v in vals)


__all__ = ["NumericOracle", "Verdict", "equivalent", "fingerprint", "ir"]
