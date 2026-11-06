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
"""Explicit, private controls and observations for reproducible fusion experiments.

These controls do not change the public defaults. Timings are exclusive of
nested instrumented stages; search wall time also includes candidate execution.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic

MODES = ("production", "common", "reference_only", "enumeration", "single_term")


@dataclass
class Experiment:
    mode: str = "production"
    compiler: object = None
    compiler_key: str = "llvm/default"
    events: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    agreements: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    # Shared only among controlled runs of the same input graph. The values
    # retain the repaired IR itself, not a reparsed display string.
    frozen: dict = field(default_factory=dict)
    graph: object = None
    fixed: dict = field(default_factory=dict)
    prefixes: dict = field(default_factory=dict)
    solutions: dict = field(default_factory=dict)
    gate_observer: object = None
    _stack: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown synthesis experiment: {self.mode}")

    def emit(self, kind, **data):
        self.events.append({"kind": kind, **data})

    @contextmanager
    def span(self, phase):
        frame = [monotonic(), 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = monotonic() - frame[0]
            self.timings[phase] = self.timings.get(phase, 0.0) + elapsed - frame[1]
            self._stack.pop()
            if self._stack:
                self._stack[-1][1] += elapsed


def candidate_record(c):
    return {
        "name": c.name,
        "origin": c.origin,
        "source": c.source,
        "definition": str(c.state),
        "structure": str(c.struct),
    }


def solution_record(sol):
    from .reducer.derive import closed_merges

    spec = sol.spec
    return {
        "states": [candidate_record(c) for c in sol.states],
        "initialization": [str(e) for e in spec.leaves],
        "algebraic_join": [str(e) for e in spec.merge],
        "printed_join": [str(e) for e in spec.merge_code],
        "changed_slots": [a != b for a, b in zip(closed_merges(spec.merge), spec.merge_code)],
        "specialized_slots": [
            bool(entry["used"] and spec.merge_print is not None) for entry in sol.printing
        ],
        "printing_proofs": sol.printing,
        "rebase_printing_enabled": spec.merge_print is not None,
        "outputs": [str(e) for e in spec.epilogue_code],
        "merge_invariant": spec.proof,
    }
