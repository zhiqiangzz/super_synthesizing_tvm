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
"""Emit a discovered program as runnable TE Python source.

The text declares the symbolic extents and placeholders, then one
``te.compute`` per op (a synthesised ``comm_reduce`` becomes a
``te.comm_reducer`` with its merge and identity spelled out), and finally the
output. Executing it with ``te``/``tir`` in scope rebuilds the same TE graph
that :meth:`Result.materialize` produces.
"""

from __future__ import annotations

from tvm import te

from .pool import ProgramSnapshot
from .target import SearchCtx


def emit_program(snapshot: ProgramSnapshot, inputs: list[te.Tensor], ctx: SearchCtx) -> str:
    dims = ctx.dims
    lines: list[str] = [
        "from tvm import te",
        "from tvm import tirx as tir",
        "",
    ]
    # symbolic extents
    sym = [dims.name(k) for k in dims.all_keys() if dims.is_symbolic(k)]
    for name in sym:
        lines.append(f'{name} = te.var("{name}")')
    if sym:
        lines.append("")
    # placeholders
    names: list[str] = []
    for t in inputs:
        name = t.op.name
        names.append(name)
        shape = shape_source(dims.keys(t.shape), ctx)
        lines.append(f'{name} = te.placeholder({shape}, name="{name}", dtype="{t.dtype}")')
    lines.append("")
    for n, rec in enumerate(snapshot.ops):
        out_names = [f"t{o}" for o in rec.outputs]
        operand_names = [names[i] for i in rec.operands]
        entries = [snapshot.pool[i] for i in rec.operands]
        lines.extend(rec.spec.emit(operand_names, entries, out_names, rec.params, ctx, n))
        lines.append("")
        names.extend(out_names)
    lines.append(f"output = t{snapshot.output}")
    return "\n".join(lines) + "\n"


def shape_source(keys, ctx: SearchCtx) -> str:
    parts = [ctx.dims.name(k) for k in keys]
    return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"


def index_names(rank: int) -> list[str]:
    return [f"i{k}" for k in range(rank)]


def compute_source(out: str, shape: str, args: list[str], body: str, name: str) -> str:
    return f'{out} = te.compute({shape}, lambda {", ".join(args)}: {body}, name="{name}")'


def load_source(name: str, indices: list[str]) -> str:
    return f"{name}[{', '.join(indices)}]"
