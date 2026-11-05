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
"""GPU performance of the showcase operators against their baselines.

The showcase (``chain_ops.showcase()``) is the fourteen reduction-chain
operators measured for performance and reported. For each of them this
package holds, on the GPU and at production sizes,

* the inputs, in the precision the operator is used at (``ops``);
* a float64 reference (``ops``);
* the baselines: the unfused program as torch writes it, the same program
  through ``torch.compile``, the idiomatic torch call where torch has one, and
  the fused library implementations that are installed (``baselines``);
* the sizes, grouped by regime (``shapes``);
* the least traffic and work any implementation must do, and the measured
  peaks of the card (``roofline``).

``run`` checks every implementation against the reference and times it;
``report`` turns the results into tables.

    python -m bench.run --op softmax --regime all
    python -m bench.run --roofline
"""

from __future__ import annotations

import pathlib
import sys

# chain_ops and harness live next to this package under super_opt/examples.
_EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
