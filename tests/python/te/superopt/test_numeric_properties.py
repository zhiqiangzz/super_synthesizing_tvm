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
"""Generated inputs and nonempty merge trees; Hypothesis owns shrinking/replay."""

import gmpy2
import numpy as np
import pytest
from chain_ops import showcase
from chain_ops.run import extents
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tvm import te
from tvm.te.superopt import fuse, numeric
from tvm.te.superopt.accuracy import (
    _compile,
    _run,
    concrete_shape,
    high_precision_references,
    positive_inputs,
)
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.realize import RealizeEnv, to_prim


@pytest.fixture(scope="module", params=showcase(), ids=lambda op: op.name)
def compiled_operator(request):
    op = request.param
    inputs, outputs = op.unfused()
    result = fuse(outputs, inputs, accuracy="off")
    assert result.fused, result.summary()
    return op, inputs, outputs, result, _compile(inputs, outputs), _compile(inputs, result.outputs)


@settings(max_examples=12, deadline=None)
@given(data=st.data())
def test_showcase_generated_inputs_against_mpfr(compiled_operator, data):
    op, inputs, outputs, result, source, fused = compiled_operator
    length = data.draw(st.integers(2, 9), label="reduction length")
    values = extents(inputs, outputs, reduce=length, other=2)
    positive = positive_inputs(outputs, inputs, op.positive)
    arrays = []
    for t, pos in zip(inputs, positive):
        # Dyadic values include exact zeros and repeated values, and shrink well.
        # Extreme magnitudes and offsets also have fixed accuracy-gate regressions.
        elements = st.integers(1 if pos else -32, 32).map(lambda x: x / 8)
        arrays.append(
            data.draw(
                hnp.arrays(
                    np.dtype(str(t.dtype)),
                    tuple(concrete_shape(t.shape, values)),
                    elements=elements,
                    fill=st.nothing(),
                ),
                label=t.op.name,
            )
        )
    expected, _ = high_precision_references(outputs, inputs, arrays, values)
    # The generator must respect the source's domain (e.g. nonzero variance).
    # Undefined-source cases are exercised by the fixed domain/degenerate tests.
    assume(all(np.all(numeric.finite(w)) for w in expected))
    shapes = [concrete_shape(t.shape, values) for t in outputs]
    dtypes = [str(t.dtype) for t in outputs]
    note(op.name)
    for label, lib in (("original", source), ("fused", fused)):
        for got, want in zip(_run(lib, arrays, shapes, dtypes), expected):
            np.testing.assert_allclose(
                got,
                np.asarray(want, dtype="float64"),
                rtol=1e-4,
                atol=1e-5,
                equal_nan=False,
                err_msg=f"{op.name}: {label}",
            )


@pytest.fixture(scope="module", params=["float32", "float64"])
def moment_merger(request):
    dtype = request.param
    m = te.var("m")
    x = te.placeholder((1, m), dtype=dtype, name="X")
    j = te.reduce_axis((0, m), "j")
    total = te.compute((1,), lambda i: te.sum(x[i, j], axis=j), name="s1")
    mean = te.compute((1,), lambda i: total[i] / m.astype(dtype), name="mu")
    outputs = [total]

    def moment(power):
        axis = te.reduce_axis((0, m), "j")

        def body(i):
            d = x[i, axis] - mean[i]
            term = d
            for _ in range(power - 1):
                term = term * d
            return te.sum(term, axis=axis)

        return te.compute((1,), body, name=f"s{power}")

    outputs.extend(moment(p) for p in (2, 3, 4))
    result = fuse(outputs, [x], accuracy="off")
    assert result.fused, result.summary()
    selection = result.chains[0].selection
    spec = selection.solution.spec
    size = len(spec.states)
    a, b = [te.placeholder((size,), dtype=dtype, name=n) for n in ("a", "b")]
    env = RealizeEnv(
        dtype,
        state={
            ir.state_var(side, k): v[k] for side, v in (("a", a), ("b", b)) for k in range(size)
        },
    )

    def scalar(expr, bindings, name):
        return te.compute((), lambda: to_prim(expr, bindings), name=name)

    merged = [scalar(e, env, f"merge{k}") for k, e in enumerate(spec.merge_code)]
    lib = _compile([a, b], merged)
    point = te.placeholder((1,), dtype=dtype, name="point")
    leaf_env = RealizeEnv(
        dtype, index={ir.idx("i0"): 0, ir.bidx(0): 0}, elem=lambda tid, indices: point[0]
    )
    leaves = [scalar(e, leaf_env, f"leaf{k}") for k, e in enumerate(spec.leaves)]
    leaf_lib = _compile([point], leaves)

    def leaf(value):
        return np.array(
            _run(leaf_lib, [np.array([value], dtype=dtype)], [()] * size, [dtype] * size),
            dtype=dtype,
        )

    def merge(left, right):
        return np.array(_run(lib, [left, right], [()] * size, [dtype] * size), dtype=dtype)

    return dtype, selection.states, leaf, merge


@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_generated_nonempty_partitions_and_merge_trees(moment_merger, data):
    dtype, states, leaf, merge = moment_merger
    values = data.draw(st.lists(st.integers(-32, 32), min_size=1, max_size=16), label="values")
    offset = data.draw(st.sampled_from([0, 16, 1024]), label="offset")
    values = np.asarray([offset + v / 8 for v in values], dtype=dtype)
    n = len(values)
    cuts = sorted(data.draw(st.sets(st.integers(1, n - 1)), label="cuts")) if n > 1 else []
    chunks = np.split(values, cuts)
    partials = []
    for chunk in chunks:
        state = leaf(chunk[0])
        for value in chunk[1:]:
            state = merge(state, leaf(value))
        partials.append(state)
    while len(partials) > 1:
        k = data.draw(st.integers(0, len(partials) - 2), label="merge position")
        a, b = partials[k : k + 2]
        if data.draw(st.booleans(), label="swap operands"):
            a, b = b, a
        partials[k : k + 2] = [merge(a, b)]
    with numeric.context(256):
        xs = numeric.mp_array(values)
        total = gmpy2.fsum(xs)
        centered = xs - total / n
        direct = {"s1": total, **{f"s{p}": gmpy2.fsum(v**p for v in centered) for p in (2, 3, 4)}}
        expected = [
            gmpy2.mpfr(n) if isinstance(s.state, ir.Card) else direct[s.name] for s in states
        ]
        # Domain-specific conservative budgets scale with the accumulated inputs
        # and central powers; they are test tolerances, not certified error bounds.
        eps = np.finfo(dtype).eps
        budget = []
        for s in states:
            if isinstance(s.state, ir.Card):
                budget.append(0.0)
            elif s.name == "s1":
                budget.append(float(64 * n * eps * gmpy2.fsum(abs(v) for v in xs)))
            else:
                p = int(s.name[1:])
                scale = gmpy2.fsum(abs(v) ** (p - 1) for v in centered)
                budget.append(float(64 * n * eps * max(abs(v) for v in xs) * scale))
    error = np.abs(partials[0] - np.asarray(expected, dtype="float64"))
    np.testing.assert_array_less(error, np.nextafter(np.array(budget), np.inf))
