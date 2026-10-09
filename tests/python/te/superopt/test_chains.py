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
"""Every operator of the registry: unfused, fused by hand, and fused by synthesis."""

import os

import pytest
from chain_ops import OPERATORS
from chain_ops.run import extents, passes, rel_err, run_llvm, sample

import tvm.testing
from tvm.te.superopt import AccuracyConfig, fuse, judge

TOLERANCE = 1.0e-4  # float32 programs against a float64 reference, unit-scale data
SLOW = pytest.mark.skipif(
    not os.environ.get("TVM_SUPEROPT_SLOW"), reason="the search takes minutes"
)


def _operators(fusible: bool) -> list:
    return [
        pytest.param(name, marks=SLOW) if op.slow else name
        for name, op in OPERATORS.items()
        if op.fusible == fusible
    ]


@pytest.mark.parametrize("name", _operators(True))
def test_fusible_operator(name):
    op = OPERATORS[name]
    ins, outs = op.unfused()
    config = AccuracyConfig(positive=op.positive)
    res = fuse(outs, ins, max_states=op.max_states, accuracy_config=config)
    assert res.fused and not res.skipped, res.summary()
    assert all(c.fused and len(c.states) == op.states for c in res.chains), res.summary()
    assert res.accuracy is not None and res.accuracy.ok

    # the three forms agree with the float64 reference on the same data
    values = extents(ins, outs)
    arrays = sample(ins, outs, values, seed=1, positive=op.positive)
    want = op.reference(*arrays)
    assert rel_err(run_llvm(ins, outs, arrays, values), want) < TOLERANCE
    assert rel_err(run_llvm(res.inputs, res.outputs, arrays, values), want) < TOLERANCE
    hand_ins, hand_outs = op.fused()
    assert rel_err(run_llvm(hand_ins, hand_outs, arrays, values), want) < TOLERANCE

    # one pass over each chain's axis where the original took several
    for extent in {str(c.chain.extent): c.chain.extent for c in res.chains}.values():
        fused_chains = sum(str(c.chain.extent) == str(extent) for c in res.chains)
        assert passes(res.outputs, extent) == fused_chains == passes(hand_outs, extent)
        assert passes(outs, extent) > fused_chains


@pytest.mark.parametrize("name", _operators(False))
def test_operator_outside_the_method(name):
    op = OPERATORS[name]
    ins, outs = op.unfused()
    res = fuse(outs, ins, max_states=op.max_states)
    assert not res.fused
    assert op.reason in res.summary(), res.summary()
    # the program comes back untouched
    assert all(new.same_as(old) for new, old in zip(res.outputs, outs))


@pytest.mark.parametrize("name", list(OPERATORS))
def test_probes_say_whether_a_single_pass_exists(name):
    """Without deriving anything: every fused operator has one, and so do some that are
    not fused (a fifth moment, a norm with scaling); a mean absolute deviation has none."""
    op = OPERATORS[name]
    _, outs = op.unfused()
    theories = judge(outs)
    if op.exists is None:
        assert theories == []  # no chain was read off the program
        return
    assert theories and all(t.fusible is op.exists for t in theories), [
        t.summary() for t in theories
    ]
    assert not op.fusible or op.exists


def test_registry_covers_every_mechanism():
    groups = {op.group for op in OPERATORS.values()}
    assert groups == {"shift/scale", "closure", "hoist", "extreme", "negative"}
    assert all(op.fused is not None for op in OPERATORS.values() if op.fusible)


if __name__ == "__main__":
    tvm.testing.main()
