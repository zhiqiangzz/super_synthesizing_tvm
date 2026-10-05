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
"""Sum-product normal form: reassociated contractions, proofs and pruning."""

import pytest

import tvm.testing
from tvm import te
from tvm.te.superopt.api import superoptimize
from tvm.te.superopt.config import Bounds
from tvm.te.superopt.symbolic import LowerCtx, Unsupported
from tvm.te.superopt.symbolic.sum_product import sum_product_contains, sum_product_equal


def _mm(A, B, name):
    k = te.reduce_axis((0, A.shape[1]), "k")
    return te.compute(
        (A.shape[0], B.shape[1]), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name=name
    )


def _chain():
    M, K, N, P = (te.var(n) for n in "MKNP")
    A = te.placeholder((M, K), name="A", dtype="float32")
    B = te.placeholder((K, N), name="B", dtype="float32")
    C = te.placeholder((N, P), name="C", dtype="float32")
    return A, B, C


def test_association_orders_are_equal():
    A, B, C = _chain()
    lower = LowerCtx()
    left = lower.lower(_mm(_mm(A, B, "AB"), C, "L")).body
    right = lower.lower(_mm(A, _mm(B, C, "BC"), "R")).body
    assert left is not right  # the canonical form keeps the nesting
    assert sum_product_equal(left, right)
    # a different product is not
    D = te.placeholder((C.shape[0], C.shape[1]), name="D", dtype="float32")
    other = lower.lower(_mm(A, _mm(B, D, "BD"), "O")).body
    assert not sum_product_equal(left, other)


def test_products_of_sums_get_their_own_indices():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype="float32")
    k1, k2, k3 = (te.reduce_axis((0, m), f"k{t}") for t in (1, 2, 3))
    s1 = te.compute((n,), lambda i: te.sum(X[i, k1], axis=k1), name="s1")
    square = te.compute((n,), lambda i: s1[i] * s1[i], name="sq")
    double = te.compute((n,), lambda i: te.sum(X[i, k2] * X[i, k3], axis=[k2, k3]), name="dd")
    k4 = te.reduce_axis((0, m), "k4")
    diag = te.compute((n,), lambda i: te.sum(X[i, k4] * X[i, k4], axis=k4), name="diag")
    lower = LowerCtx()
    sq, dd, dg = (lower.lower(t).body for t in (square, double, diag))
    assert sum_product_equal(sq, dd)  # (Σ_k x)^2 = Σ_k Σ_k' x x'
    assert not sum_product_equal(sq, dg)  # but not Σ_k x^2


def test_shadowed_loop_variable_is_rejected():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype="float32")
    k1, k2 = te.reduce_axis((0, m), "k"), te.reduce_axis((0, m), "k")
    dd = te.compute((n,), lambda i: te.sum(X[i, k1] * X[i, k2], axis=[k1, k2]), name="dd")
    with pytest.raises(Unsupported):
        LowerCtx().lower(dd)


def test_gram_prefix_is_contained():
    M, K, N = te.var("M"), te.var("K"), te.var("N")
    A = te.placeholder((M, K), name="A", dtype="float32")
    X = te.placeholder((K, N), name="X", dtype="float32")
    k = te.reduce_axis((0, K), "k")
    Y = te.compute((M, N), lambda i, n: te.sum(A[i, k] * X[k, n], axis=k), name="Y")
    n = te.reduce_axis((0, N), "n")
    C = te.compute((M, M), lambda i, i2: te.sum(Y[i, n] * Y[i2, n], axis=n), name="C")
    n2 = te.reduce_axis((0, N), "n")
    G = te.compute((K, K), lambda a, b: te.sum(X[a, n2] * X[b, n2], axis=n2), name="G")
    AG = _mm(A, G, "AG")
    lower = LowerCtx()
    target = lower.lower(C).body
    assert sum_product_contains(target, lower.lower(AG).body)
    # reading a tensor twice more than the target does is not contained
    AGG = _mm(AG, G, "AGG")
    assert not sum_product_contains(target, lower.lower(AGG).body)


def test_e2e_matrix_chain_reassociates():
    A, B, C = _chain()
    out = _mm(_mm(A, B, "AB"), C, "D")
    res = superoptimize(out, [A, B, C], Bounds(max_tensor_ops=2), with_reducers=False)
    methods = {r.verdict.method for r in res}
    assert methods == {"hash", "sum-product"}
    right = [r for r in res if r.verdict.method == "sum-product"]
    first = right[0].snapshot.ops[0]
    assert sorted(first.operands) == [1, 2]  # B C first
    assert all(r.accuracy.ok for r in res)


if __name__ == "__main__":
    tvm.testing.main()
