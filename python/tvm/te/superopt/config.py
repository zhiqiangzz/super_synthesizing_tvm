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
"""Search bounds. All bounds are syntactic; there is no cost model."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Bounds:
    """Bounded-exhaustive search limits.

    max_tensor_ops
        Number of tensor-level operators in a candidate program (a
        multi-output ``comm_reduce`` counts once).
    max_states
        Reducer state arity ``K``.
    min_states
        Smallest reducer arity to synthesise. A single-state reducer is
        always ``elementwise -> sum/max -> elementwise`` in disguise, which the
        plain operators already cover, so the default skips it.
    max_leaf_nodes
        Node budget for each scalar expression in a reducer's input tuple.
    max_state_expr_nodes
        Production budget for the expression that combines a new reducer
        atom with the atoms introduced before it (``A * F``, ``A + G``).
    max_latent_atoms
        How many partial reductions a reducer may maintain beyond those the
        target itself needs (a running max, a count, ...).
    max_merge_nodes
        SSA node budget for the whole merge function (shared sub-expressions
        count once).
    leaf_exp
        Whether ``exp`` is allowed inside reducer leaves (otherwise an
        explicit ``exp`` tensor op has to feed the reducer).
    test_extents
        Extents substituted for symbolic dimensions in numeric checks.
    max_sem_nodes
        Candidates whose symbolic semantics exceed this size are dropped.
    """

    max_tensor_ops: int = 5
    max_states: int = 3
    min_states: int = 2
    max_leaf_nodes: int = 4
    max_state_expr_nodes: int = 4
    max_latent_atoms: int = 1
    max_merge_nodes: int = 24
    leaf_exp: bool = False
    test_extents: tuple[int, ...] = (1, 2, 3)
    max_sem_nodes: int = 400
