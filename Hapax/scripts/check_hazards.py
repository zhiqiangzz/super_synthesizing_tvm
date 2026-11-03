"""Inputs the accuracy gate does not sample, tried on fused operators.

Adapted from the method documentation's ``check_hazards.py``. Each case runs the
unfused program, the hand-written fused one and the synthesised one on an input
chosen to be awkward, and records the normwise relative error of each against the
float64 reference (``inf`` when a result is not finite).

    Hapax/scripts/py Hapax/scripts/check_hazards.py Hapax/data/hazards.json
"""

import sys

import common
import numpy as np
from chain_ops import OPERATORS
from chain_ops.run import extents, rel_err, run_llvm, sample

from tvm.te.superopt import AccuracyConfig, fuse


def run(name: str, tweak, what: str) -> dict:
    op = OPERATORS[name]
    ins, outs = op.unfused()
    res = fuse(
        outs, ins, max_states=op.max_states, accuracy_config=AccuracyConfig(positive=op.positive)
    )
    values = extents(ins, outs, reduce=64)
    arrays = sample(ins, outs, values, 1, positive=op.positive)
    arrays = [a.astype("float32") for a in tweak([a.copy() for a in arrays])]
    with np.errstate(all="ignore"):
        want = op.reference(*arrays)
        row = {
            "operator": name,
            "input": what,
            "fused": res.fused,
            "unfused": rel_err(run_llvm(ins, outs, arrays, values), want),
            # not fused: the program comes back as it was
            "synthesised": rel_err(run_llvm(res.inputs, res.outputs, arrays, values), want),
        }
        if op.fused is not None:
            hand_ins, hand_outs = op.fused()
            row["hand"] = rel_err(run_llvm(hand_ins, hand_outs, arrays, values), want)
    print(row, flush=True)
    return row


def zero_leading_weights(arrays):
    x, w = arrays
    w[:, :3] = 0.0
    return [x, w]


def one_zero_in_q(arrays):
    q, a = arrays
    q[:, 5] = 0.0
    return [q, a]


def one_zero_in_x(arrays):
    x, y = arrays
    x[:, 5] = 0.0
    return [x, y]


def nearly_parallel(arrays):
    q, a = arrays
    q = q / np.sqrt((q.astype(np.float64) ** 2).sum(axis=1, keepdims=True))
    return [q, 2.0 * q + 1.0e-3 * a]


def scaled(factor):
    return lambda arrays: [a * factor for a in arrays]


def constant_rows(arrays):
    return [np.full_like(a, 3.0) for a in arrays]


CASES = [
    ("weighted_variance", zero_leading_weights, "the first 3 weights of every row are exactly 0"),
    ("normalised_projection", one_zero_in_q, "one column of q is exactly 0"),
    ("cosine", one_zero_in_x, "one column of x is exactly 0"),
    ("projection", nearly_parallel, "a = 2q + 1e-3 noise, q a unit vector"),
    ("variance", scaled(1.0e25), "every input multiplied by 1e25"),
    ("softmax", scaled(1.0e25), "every input multiplied by 1e25"),
    ("minmax_mean", scaled(1.0e25), "every input multiplied by 1e25"),
    ("variance", constant_rows, "every row constant (3)"),
    ("zscore_max", constant_rows, "every row constant (3): the standard deviation is 0"),
]

if __name__ == "__main__":
    rows = [run(*case) for case in CASES]
    common.dump(sys.argv[1], {"meta": common.meta("check_hazards.py", extent=64, seed=1), "rows": rows})
