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
"""Predeclared scans and the paper's existing exceptional-input cases."""

import numpy as np

SCANS = {
    "variance": {
        "offset": [0, 1, 16, 256, 4096, 10000, 65536, 1048576],
        "scale": [2.0**p for p in (-80, -70, -60, -30, 0, 30, 60, 65, 70)],
    },
    "softmax": {
        "offset": [-128, -112, -104, -96, -88, -64, 0, 64, 80, 88, 96, 128, 10000],
        "scale": [1 / 16, 1, 4, 16, 32, 64, 128],
    },
}
LENGTHS = (16, 256, 4096)
SEEDS = (0, 1, 2)


def representative_cases(name):
    for m in LENGTHS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            base = rng.integers(-16, 17, size=(3, m)).astype(np.float32) / 8
            base[:, :3] = [-2, 0, 2]
            for row in base:
                rng.shuffle(row)
            for scan, parameters in SCANS[name].items():
                for index, parameter in enumerate(parameters):
                    shift, scale = (parameter, 1) if scan == "offset" else (0, parameter)
                    x = (shift + scale * base.astype(np.float64)).astype(np.float32)
                    yield (
                        f"{name}-{scan}-{m}-{seed}-{index}",
                        x,
                        {
                            "scan": scan,
                            "offset": shift,
                            "scale": scale,
                            "seed": seed,
                        },
                    )
        for label, value in (("zeros", 0), ("constant", 3)):
            yield f"{name}-{label}-{m}", np.full((3, m), value, dtype=np.float32), {"scan": label}
    yield f"{name}-singleton", np.array([[-2], [0], [2]], dtype=np.float32), {"scan": "singleton"}


def zero_leading_weights(arrays):
    arrays[1][:, :3] = 0
    return arrays


def zero_column(arrays):
    arrays[0][:, 5] = 0
    return arrays


def nearly_parallel(arrays):
    q, a = arrays
    q = q / np.sqrt((q.astype(np.float64) ** 2).sum(axis=1, keepdims=True))
    return [q, 2 * q + 1e-3 * a]


def scaled(arrays):
    return [a * 1e25 for a in arrays]


def constant(arrays):
    return [np.full_like(a, 3) for a in arrays]


HAZARDS = [
    ("weighted_variance", zero_leading_weights, "first three weights exactly zero"),
    ("normalised_projection", zero_column, "one column of q exactly zero"),
    ("cosine", zero_column, "one column of x exactly zero"),
    ("projection", nearly_parallel, "a = 2q + 1e-3 noise, q unit vector"),
    ("variance", scaled, "inputs multiplied by 1e25"),
    ("softmax", scaled, "inputs multiplied by 1e25"),
    ("minmax_mean", scaled, "inputs multiplied by 1e25"),
    ("variance", constant, "constant rows (3)"),
    ("zscore_max", constant, "constant rows (3), zero standard deviation"),
]
