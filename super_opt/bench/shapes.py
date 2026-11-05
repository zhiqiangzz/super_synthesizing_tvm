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
"""The sizes each showcase operator is measured at, grouped by regime.

For the row reductions the regime is where a row lives while it is reduced,
which is what decides whether fusing the chain saves HBM traffic at all:

``short``
    a row of at most 8K elements fits in one CTA's registers: every library
    already reads it from HBM once and makes its passes on chip;
``long``
    32K to 256K elements: past one CTA's 227 KB of shared memory in bfloat16 at
    the top end, within a cluster's distributed shared memory;
``split``
    a handful of rows of millions of elements: one row has to be spread over
    many CTAs, and a chain of k reductions costs k sweeps of HBM unless it is
    fused into one.

The products (attention, cross_entropy, sinkhorn_iteration) are sized like the
workloads they come from. Every size keeps the unfused torch program (its
float32 intermediates included) within one 80 GB H800.
"""

from __future__ import annotations

_ROWS_F32 = {
    "short": [{"n": 16384, "m": 4096}],
    "long": [{"n": 4096, "m": 65536}],
    "split": [{"n": 16, "m": 4194304}, {"n": 1, "m": 67108864}],
}

SHAPES: dict[str, dict[str, list[dict]]] = {
    "softmax": {
        "short": [{"n": 32768, "m": 4096}, {"n": 16384, "m": 8192}],
        "long": [{"n": 4096, "m": 32768}, {"n": 2048, "m": 131072}],
        "split": [{"n": 8, "m": 8388608}, {"n": 1, "m": 67108864}],
    },
    "cross_entropy": {
        # tokens x hidden x vocabulary: Llama-2 32000, Llama-3 128256, Gemma 256000
        "gemm": [
            {"n": 8192, "d": 4096, "v": 32000},
            {"n": 8192, "d": 4096, "v": 128256},
            {"n": 8192, "d": 4096, "v": 256000},
        ],
    },
    "softmax_entropy": {
        # tokens x vocabulary: Llama-2 32000, Qwen2.5 151936
        "long": [{"n": 16384, "m": 32000}, {"n": 4096, "m": 151936}],
    },
    "attention": {
        # batch x heads x sequence x head dim, not causal
        "gemm": [
            {"b": 8, "h": 16, "s": 1024, "d": 128},
            {"b": 2, "h": 16, "s": 4096, "d": 128},
            {"b": 1, "h": 16, "s": 8192, "d": 128},
            {"b": 2, "h": 16, "s": 4096, "d": 64},
        ],
    },
    "layernorm": {
        "short": [{"n": 32768, "m": 4096}, {"n": 16384, "m": 8192}],
        "long": [{"n": 4096, "m": 65536}],
        "split": [{"n": 4, "m": 16777216}],
    },
    "layernorm_backward": {
        "short": [{"n": 32768, "m": 4096}, {"n": 16384, "m": 8192}],
        "long": [{"n": 4096, "m": 65536}],
        "split": [{"n": 4, "m": 16777216}],
    },
    "contrastive": {
        # global batch of image-text pairs
        "square": [{"n": 8192}, {"n": 16384}, {"n": 32768}],
    },
    "sinkhorn_iteration": {
        # point clouds: 3-D shapes, and 64-D embeddings
        "points": [
            {"n": 8192, "m": 8192, "d": 3},
            {"n": 32768, "m": 32768, "d": 3},
            {"n": 8192, "m": 8192, "d": 64},
            {"n": 32768, "m": 32768, "d": 64},
        ],
    },
    "pearson": _ROWS_F32,
    "kurtosis": _ROWS_F32,
    "ess": {
        # particle systems x particles
        "short": [{"n": 16384, "m": 4096}],
        "long": [{"n": 1024, "m": 262144}],
        "split": [{"n": 4, "m": 16777216}, {"n": 1, "m": 67108864}],
    },
    "papr": {
        # OFDM symbols x subcarriers
        "short": [{"n": 262144, "m": 1024}, {"n": 65536, "m": 4096}],
    },
    "householder": {
        # columns of a batch of QR panels x their height
        "short": [{"n": 65536, "m": 256}, {"n": 16384, "m": 4096}],
        "long": [{"n": 1024, "m": 65536}],
    },
    "heat_capacity": {
        # replicas x samples
        "short": [{"n": 16384, "m": 4096}],
        "long": [{"n": 1024, "m": 262144}],
        "split": [{"n": 4, "m": 16777216}, {"n": 1, "m": 67108864}],
    },
    # regressions (bootstrap replicates, assets) x observations
    "ols_hc0": _ROWS_F32,
    "normalized_policy_loss": {
        # groups x samples per group; one batch-wide normalisation over every token
        "short": [{"n": 65536, "m": 256}],
        "long": [{"n": 256, "m": 262144}],
        "split": [{"n": 1, "m": 33554432}],
    },
    "ledoit_wolf": {
        # covariance problems x samples x features
        "batch": [
            {"b": 256, "n": 1024, "d": 32},
            {"b": 64, "n": 4096, "d": 64},
            {"b": 8, "n": 65536, "d": 128},
        ],
    },
}


def configs(op: str, regime: str = "all") -> list[tuple[str, dict]]:
    """``(regime, cfg)`` pairs of ``op``, all regimes or one."""
    return [
        (name, cfg) for name, cfgs in SHAPES[op].items() if regime in ("all", name) for cfg in cfgs
    ]


def label(cfg: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in cfg.items())
