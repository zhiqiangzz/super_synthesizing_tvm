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
"""Compile the actual verification tuple, without applying the numerical gate."""

import dataclasses
import re

import numpy as np
from chain_ops import OPERATORS

import tvm
import tvm_tirx
from tvm import te
from tvm.te.superopt.accuracy import concrete_shape, spec_issues
from tvm.te.superopt.reducer.build import Rewriter, build_chain
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.derive import atoms_in, derive
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.source import chain_source, render
from tvm.te.superopt.reducer.states import _Chain, _Select, members_of
from tvm.te.superopt.reducer.verify import prove_equal
from tvm.te.superopt.symbolic import ir
from tvm.te.superopt.symbolic.canonicalize import subst, subst_domain

TARGET = {
    "kind": "llvm",
    "fast-math": False,
    "fast-math-nnan": False,
    "fast-math-ninf": False,
    "fast-math-nsz": False,
    "fast-math-arcp": False,
    "fast-math-contract": False,
    "fast-math-reassoc": False,
}


def compile_strict(inputs, outputs):
    func = te.create_function([*inputs, *outputs])
    return tvm.compile(tvm_tirx.IRModule({"main": func}), target=TARGET)


@dataclasses.dataclass
class Program:
    inputs: list
    outputs: list
    lib: object = None

    def compile(self, directory=None):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "program.tir").write_text(
                te.create_function([*self.inputs, *self.outputs]).script()
            )
        self.lib = compile_strict(self.inputs, self.outputs)
        if directory is not None:
            llvm = self.lib.mod.inspect_source("ll")
            # Fail loudly if the compiled experiment violates the FP contract.
            if re.search(r"\b(?:fadd|fsub|fmul|fdiv|call) (?:fast|reassoc|contract)\b", llvm):
                raise RuntimeError("unexpected LLVM fast-math flags")
            (directory / "program.ll").write_text(llvm)
        return self

    def run(self, arrays, values):
        if self.lib is None:
            self.compile()
        args = [tvm_tirx.runtime.tensor(a) for a in arrays]
        out = [
            tvm_tirx.runtime.tensor(np.zeros(concrete_shape(t.shape, values), dtype=str(t.dtype)))
            for t in self.outputs
        ]
        self.lib["main"](*args, *out)
        return [t.numpy() for t in out]


def reference_programs(name):
    """Run selection once; derive B from its repaired base and required definitions."""
    inputs, outputs = OPERATORS[name].unfused()
    chains, skipped = discover_chains(outputs)
    if len(chains) != 1 or skipped:
        raise ValueError("reference-state experiment requires exactly one complete chain")
    chain = chains[0]
    world = _Chain(chain, members_of(chain), TeSigns(outputs))
    selector = _Select(world)
    selection = selector.run()
    if selection.solution is None:
        raise ValueError(selection.failure)
    base = selector.base()
    if isinstance(base, str):
        raise ValueError(base)
    targets = {atom for candidate in base for atom in atoms_in(candidate.state, world.R)}
    for member in chain.required:
        targets.update(atoms_in(world.by_pid[member.pid].state, world.R))
    required = tuple(
        subst_domain(world.by_pid[m.pid].state, {world.R: world.full}) for m in chain.required
    )
    reference = derive(tuple(sorted(targets)), required, chain.axis, preserve=True)
    if reference is None:
        raise ValueError("cannot derive the actual verification tuple")
    programs = {"source": Program(inputs, outputs)}
    definitions = {"selection_route": selector.route}
    for label, spec in (("reference_B", reference), ("maintained", selection.solution.spec)):
        rewriter = Rewriter()
        build_chain(chain, spec, rewriter)
        programs[label] = Program(inputs, [rewriter.tensor(t) for t in outputs])
        slots = {
            ir.elem(-k - 1, tuple(ir.idx(f"i{c}") for c in range(len(chain.coord_extents)))): s
            for k, s in enumerate(spec.closed)
        }
        proofs = [
            prove_equal(subst(expr, slots), want)
            for expr, want in zip(spec.epilogue_code, required)
        ]
        if not all(proofs):
            raise ValueError("result reconstruction not proved")
        definitions[label] = {
            "states": [render(s, chain) for s in spec.states],
            "states_ir": [str(s) for s in spec.states],
            "required": [str(s) for s in required],
            "reconstruction": [str(s) for s in spec.epilogue_code],
            "reconstruction_proofs": proofs,
            "merge_invariant_proof": spec.proof,
            "static_issues": sorted(spec_issues(spec)),
            "gate_applied": False,
            "reducer": chain_source(chain, spec, [f"s{k}" for k in range(spec.arity)]),
            "initialization": "first leaf; tail indices 1..m-1; singleton uses its leaf",
        }
    return programs, definitions
