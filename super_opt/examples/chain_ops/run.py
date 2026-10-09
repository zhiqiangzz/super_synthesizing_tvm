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
"""Run a TE program with symbolic extents on concrete data."""

from __future__ import annotations

import numpy as np

import tvm
from tvm import te
from tvm import tirx as tir
from tvm.te.superopt.accuracy import (
    concrete_shape,
    extent_names,
    positive_inputs,
    sample_inputs,
)
from tvm.te.superopt.symbolic.lower import compute_ops


def extents(inputs, outputs, reduce: int = 64, other: int = 3) -> dict[str, int]:
    """A value for every symbolic extent: ``reduce`` for those a reduction runs over."""
    names, reduced = extent_names(outputs, inputs)
    return {n: (reduce if n in reduced else other) for n in sorted(names)}


def sample(
    inputs, outputs, values: dict[str, int], seed: int = 0, shift=0.0, scale=1.0, positive=()
):
    """Random inputs on the domain of the program: positive where it takes a log or
    divides, and for the inputs named in ``positive`` (weights)."""
    rng = np.random.default_rng(seed)
    positive = positive_inputs(outputs, inputs, positive)
    return sample_inputs(
        inputs, positive, lambda s: concrete_shape([s], values)[0], rng, shift, scale
    )


def run_llvm(inputs, outputs, arrays, values: dict[str, int]) -> list[np.ndarray]:
    """Compile for LLVM in the program's own dtype and run; outputs as float64."""
    func = te.create_prim_func([*inputs, *outputs])
    lib = tvm.compile(tvm.IRModule({"main": func}), target="llvm")
    outs = [
        tvm.runtime.tensor(np.zeros(concrete_shape(o.shape, values), dtype=str(o.dtype)))
        for o in outputs
    ]
    lib["main"](*[tvm.runtime.tensor(a) for a in arrays], *outs)
    return [o.numpy().astype(np.float64) for o in outs]


def rel_err(got, want) -> float:
    """Largest normwise relative error over the outputs."""
    worst = 0.0
    for g, w in zip(got, want):
        if g.shape != np.shape(w) or not np.all(np.isfinite(g)):
            return float("inf")
        scale = float(np.max(np.abs(w))) if np.size(w) else 0.0
        err = float(np.max(np.abs(g - w))) if np.size(w) else 0.0
        worst = max(worst, err / scale if scale > 0 else err)
    return worst


def passes(outputs, extent) -> int:
    """Number of reductions in the program that walk an axis of extent ``extent``.

    A symbolic extent is matched by name, so that two builds of one operator
    (each with its own ``te.var``) can be compared.
    """

    def key(e):
        e = tir.convert(e)
        return e.name if isinstance(e, tir.Var) else int(e.value)

    want = key(extent)
    count = 0
    for op in compute_ops(outputs):
        if isinstance(op.body[0], tir.Reduce):
            count += sum(key(iv.dom.extent) == want for iv in op.reduce_axis)
    return count


__all__ = ["extents", "passes", "rel_err", "run_llvm", "sample"]
