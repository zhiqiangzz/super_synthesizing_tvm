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
"""From a reduction chain to a tuple ``te.comm_reducer``.

``chain``   read the chains off a TE graph
``theory``  whether a single pass exists and how many states it takes
``states``  choosing the states of a chain: keep, supplement, replace
``domain``  where a state is defined, by proof
``rebase``  merging partial results whose context has moved
``derive``  leaf, merge, identity and epilogues from a set of states
``verify``  equalities by proof, and the identity element in floating point
``build``   write the reducer back as TE and rebuild the program around it
``source``  the same as text
"""

from .build import Rewriter, build_chain
from .chain import Chain, Member, Skipped, discover_chains
from .derive import ReducerSpec, derive
from .rebase import Candidate
from .states import Selection, Solution, Step, members_of, select
from .theory import Judgement, judge
from .verify import identity_safe, prove_equal

__all__ = [
    "Candidate",
    "Chain",
    "Judgement",
    "Member",
    "ReducerSpec",
    "Rewriter",
    "Selection",
    "Skipped",
    "Solution",
    "Step",
    "build_chain",
    "derive",
    "discover_chains",
    "identity_safe",
    "judge",
    "members_of",
    "prove_equal",
    "select",
]
