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
"""Specialize shared compute graphs for distinct direct-index read views."""

import tvm_ffi

from tvm import te
from tvm.te import _ffi_api
from tvm_tirx import tirx as tir
from tvm_tirx.ir import Call

from ..loads import is_load, load_indices, load_tensor
from .build import _rebuild


def specialize(outputs):
    cloned = {}
    rebuilt = {}

    def clone(t, key):
        if not isinstance(t.op, te.ComputeOp):
            return t
        cache = (t.op, int(t.value_index), key)
        if cache not in cloned:
            swaps = [(a, clone(a, key)) for a in t.op.input_tensors]
            op = _rebuild(t.op, swaps)
            # Reports and executable source must distinguish the two views too.
            name = op.name + "_view_" + "_".join(key)
            op = _ffi_api.ComputeOp(name, op.tag, op.attrs, op.axis, op.body)
            cloned[cache] = op.output(int(t.value_index))
        return cloned[cache]

    def rec(t):
        if not isinstance(t.op, te.ComputeOp):
            return t
        if t.op in rebuilt:
            return rebuilt[t.op].output(int(t.value_index))
        op = _rebuild(t.op, [(a, rec(a)) for a in t.op.input_tensors])
        seen = {}
        replacements = {}
        axes = {iv.var.name: f"i{k}" for k, iv in enumerate(op.axis)}

        def find(e):
            if not is_load(e):
                return
            src = load_tensor(e)
            if not isinstance(src.op, te.ComputeOp):
                return
            idx = load_indices(e)
            if not all(isinstance(i, tir.Var) for i in idx):
                return
            key = tuple(axes.get(i.name, "reduction") for i in idx)
            first = seen.setdefault(src.op, key)
            if key != first:
                replacements[e] = clone(src, key)(*idx)

        for body in op.body:
            tvm_ffi.structural_walk(body, find)
        if replacements:
            body = tvm_ffi.structural_map(op.body, (Call, lambda e: replacements.get(e, e)))
            op = _ffi_api.ComputeOp(op.name, op.tag, op.attrs, op.axis, body)
        rebuilt[t.op] = op
        return op.output(int(t.value_index))

    return [rec(t) for t in outputs]
