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
``states``  the candidate states of a chain and the reducers they give
``rebase``  merging partial results whose context has moved
``derive``  leaf, merge, identity and epilogues from a set of states
``verify``  the monoid laws, and the identity element in floating point
``build``   write the reducer back as TE and rebuild the program around it
``source``  the same as text
"""

from .build import Rewriter, build_chain
from .chain import Chain, Member, Skipped, discover_chains
from .derive import ReducerSpec, derive
from .rebase import Candidate
from .states import Pool, Solution, synthesize
from .verify import check_laws, identity_safe

__all__ = [
    "Candidate",
    "Chain",
    "Member",
    "Pool",
    "ReducerSpec",
    "Rewriter",
    "Skipped",
    "Solution",
    "build_chain",
    "check_laws",
    "derive",
    "discover_chains",
    "identity_safe",
    "synthesize",
]
