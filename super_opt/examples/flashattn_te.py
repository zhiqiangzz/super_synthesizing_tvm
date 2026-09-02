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

"""Flash-attention style scaled dot-product attention, expressed in TVM TE.

    O = softmax(Q @ Kᵀ · softmax_scale) @ V

The softmax is never spelled out as a max → exp → sum → divide chain. Instead
the online-softmax running state ``(m, l, acc)`` — row max, denominator and
weighted value sum — is handed to ``te.comm_reducer`` as a single commutative
merge over the key axis, so TE emits exactly one reduction block for it.

``attention`` is the unfused counterpart in the same file: same math, same
layouts, but the textbook matmul -> softmax -> matmul chain with the scores and
the probabilities both materialised.

By default this prints the shape-generic s_tir for both variants and checks a
CUDA build of each against ``torch.nn.functional.scaled_dot_product_attention``;
``--variant`` picks one. See ``--help``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from harness import (
    Accuracy,
    accuracy_columns,
    check_against_torch,
    generator,
    label,
    randn,
    render_source,
    render_table,
    torch_dtype,
)
from rich.console import Console

import tvm
import tvm.s_tir as s_tir
from tvm import te
from tvm import tirx as tir  # tvm.tir is absent in this fork; intrinsics live in tirx

SUPPORTED_DTYPES = ("float16", "bfloat16")

# (rtol, atol) per input dtype, against an fp32 reference. Our kernel and the
# reference both accumulate in fp32, so what is being measured is the rounding
# of the inputs and of the final narrowing store; bfloat16 carries 8 fewer
# mantissa bits than float16 and needs correspondingly more slack.
DEFAULT_TOLERANCE = {
    "float16": (1e-2, 4e-3),
    "bfloat16": (2e-2, 3e-2),
}


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AttentionShape:
    """A fully specialised attention problem size."""

    batch: int
    num_heads: int
    seqlen_q: int
    seqlen_k: int
    head_dim: int

    def __str__(self) -> str:
        return (
            f"batch={self.batch} num_heads={self.num_heads} seqlen_q={self.seqlen_q} "
            f"seqlen_k={self.seqlen_k} head_dim={self.head_dim}"
        )


def flash_attention(
    *,
    batch: int | None = None,
    num_heads: int | None = None,
    seqlen_q: int | None = None,
    seqlen_k: int | None = None,
    head_dim: int | None = None,
    dtype: str = "float16",
    accum_dtype: str = "float32",
) -> tvm.IRModule:
    """Build ``O = softmax(Q @ Kᵀ · softmax_scale) @ V`` as a TE PrimFunc.

    Batch and head are plain leading dimensions -- attention is independent
    along both, so they only widen the spatial iteration space. Layout matches
    ``torch.nn.functional.scaled_dot_product_attention``::

        Q : (batch, num_heads, seqlen_q, head_dim)
        K : (batch, num_heads, seqlen_k, head_dim)
        V : (batch, num_heads, seqlen_k, head_dim)
        O : (batch, num_heads, seqlen_q, head_dim)

    Any dimension left as ``None`` becomes a ``te.var``, yielding a shape-generic
    kernel; passing an int specialises the kernel to that extent.
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")

    def extent(value: int | None, name: str):
        return te.var(name) if value is None else value

    n_b = extent(batch, "batch")
    n_h = extent(num_heads, "num_heads")
    n_q = extent(seqlen_q, "seqlen_q")
    n_k = extent(seqlen_k, "seqlen_k")
    dim = extent(head_dim, "head_dim")

    Q = te.placeholder((n_b, n_h, n_q, dim), name="Q", dtype=dtype)
    K = te.placeholder((n_b, n_h, n_k, dim), name="K", dtype=dtype)
    V = te.placeholder((n_b, n_h, n_k, dim), name="V", dtype=dtype)

    # ---- scores S = Q @ Kᵀ, reduced over the head dimension ----------------
    d = te.reduce_axis((0, dim), name="d")
    S = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: te.sum(
            Q[b, h, i, d].astype(accum_dtype) * K[b, h, j, d].astype(accum_dtype), axis=d
        ),
        name="S",
    )

    # ---- online softmax, as a commutative reducer over the key axis --------
    def merge(state_a, state_b):
        """Merge two partial ``(max, denominator, weighted values)`` states."""
        max_a, denom_a, acc_a = state_a
        max_b, denom_b, acc_b = state_b
        row_max = tir.max(max_a, max_b)
        rescale_a = tir.exp(max_a - row_max)
        rescale_b = tir.exp(max_b - row_max)
        return (
            row_max,
            denom_a * rescale_a + denom_b * rescale_b,
            acc_a * rescale_a + acc_b * rescale_b,
        )

    def empty_state(max_dtype, denom_dtype, acc_dtype):
        return (
            tir.min_value(max_dtype),  # no key seen yet
            tir.const(0.0, denom_dtype),
            tir.const(0.0, acc_dtype),
        )

    online_softmax = te.comm_reducer(merge, empty_state, name="online_softmax")

    # ``dim`` is a Python int when specialised and a te.var when symbolic.
    head_dim_f = (
        dim.astype(accum_dtype) if hasattr(dim, "astype") else tir.const(float(dim), accum_dtype)
    )
    softmax_scale = tir.const(1.0, accum_dtype) / tir.sqrt(head_dim_f)

    j = te.reduce_axis((0, n_k), name="j")
    _row_max, denominator, weighted_values = te.compute(
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: online_softmax(
            (
                S[b, h, i, j] * softmax_scale,  # this key's score, i.e. a one-element max
                tir.const(1.0, accum_dtype),  # its weight relative to that max
                V[b, h, j, e].astype(accum_dtype),  # its contribution to the output
            ),
            axis=j,
        ),
        name="softmax_state",
    )

    O = te.compute(  # noqa: E741
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: (weighted_values[b, h, i, e] / denominator[b, h, i, e]).astype(dtype),
        name="O",
    )

    return tvm.IRModule({"main": te.create_prim_func([Q, K, V, O])})


def attention(
    *,
    batch: int | None = None,
    num_heads: int | None = None,
    seqlen_q: int | None = None,
    seqlen_k: int | None = None,
    head_dim: int | None = None,
    dtype: str = "float16",
    accum_dtype: str = "float32",
) -> tvm.IRModule:
    """Build the same attention as a textbook matmul -> softmax -> matmul chain.

    Identical mathematics and identical layouts to :func:`flash_attention`, but
    written the way the formula reads: the scores are materialised, the softmax
    is spelled out as max -> exp -> sum -> divide over the key axis, and the
    normalised probabilities are then contracted with ``V``.

    That costs seven blocks instead of three, and both ``S`` and ``P`` are full
    ``(batch, num_heads, seqlen_q, seqlen_k)`` tensors -- which is precisely the
    traffic the online-softmax formulation exists to avoid. It is here as the
    unfused baseline: the thing a fusion schedule has to beat, and a second
    opinion on the numerics of the fused kernel.
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")

    def extent(value: int | None, name: str):
        return te.var(name) if value is None else value

    n_b = extent(batch, "batch")
    n_h = extent(num_heads, "num_heads")
    n_q = extent(seqlen_q, "seqlen_q")
    n_k = extent(seqlen_k, "seqlen_k")
    dim = extent(head_dim, "head_dim")

    Q = te.placeholder((n_b, n_h, n_q, dim), name="Q", dtype=dtype)
    K = te.placeholder((n_b, n_h, n_k, dim), name="K", dtype=dtype)
    V = te.placeholder((n_b, n_h, n_k, dim), name="V", dtype=dtype)

    # ``dim`` is a Python int when specialised and a te.var when symbolic.
    head_dim_f = (
        dim.astype(accum_dtype) if hasattr(dim, "astype") else tir.const(float(dim), accum_dtype)
    )
    softmax_scale = tir.const(1.0, accum_dtype) / tir.sqrt(head_dim_f)

    # ---- first matmul: S = Q @ Kᵀ, reduced over the head dimension ---------
    d = te.reduce_axis((0, dim), name="d")
    S = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: te.sum(
            Q[b, h, i, d].astype(accum_dtype) * K[b, h, j, d].astype(accum_dtype), axis=d
        ),
        name="S",
    )

    # ---- softmax over the key axis, in the usual four steps ----------------
    # A TE reduction has to be the whole body of its block, so the scale cannot
    # be folded into S. It is applied here instead: max(c·S) = c·max(S) for
    # c > 0, so subtracting the *unscaled* row max and scaling the difference
    # gives exactly exp(c·S - max(c·S)) -- the same shift-for-stability trick.
    j_max = te.reduce_axis((0, n_k), name="j")
    row_max = te.compute(
        (n_b, n_h, n_q),
        lambda b, h, i: te.max(S[b, h, i, j_max], axis=j_max),
        name="row_max",
    )

    exp_S = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: tir.exp((S[b, h, i, j] - row_max[b, h, i]) * softmax_scale),
        name="exp_S",
    )

    j_sum = te.reduce_axis((0, n_k), name="j")
    denominator = te.compute(
        (n_b, n_h, n_q),
        lambda b, h, i: te.sum(exp_S[b, h, i, j_sum], axis=j_sum),
        name="denominator",
    )

    P = te.compute(
        (n_b, n_h, n_q, n_k),
        lambda b, h, i, j: exp_S[b, h, i, j] / denominator[b, h, i],
        name="P",
    )

    # ---- second matmul: O = P @ V, reduced over the key axis ---------------
    j_pv = te.reduce_axis((0, n_k), name="j")
    PV = te.compute(
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: te.sum(
            P[b, h, i, j_pv] * V[b, h, j_pv, e].astype(accum_dtype), axis=j_pv
        ),
        name="PV",
    )
    # A reduction owns the whole body of its block, so narrowing back to
    # ``dtype`` has to be a block of its own.
    O = te.compute(  # noqa: E741
        (n_b, n_h, n_q, dim),
        lambda b, h, i, e: PV[b, h, i, e].astype(dtype),
        name="O",
    )

    return tvm.IRModule({"main": te.create_prim_func([Q, K, V, O])})


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def schedule_naive_cuda(mod: tvm.IRModule, threads_per_block: int = 64) -> s_tir.Schedule:
    """Bind one CUDA thread per output element, leaving every reduction serial.

    Deliberately unoptimised: no tiling, no shared memory, and every intermediate
    fully materialised in global memory. This exists to give the TE definitions
    above a runnable CUDA backend to check numerics against, not to be a fast
    kernel.

    It walks whatever blocks the module turned out to have rather than a fixed
    list of names, so the three-block fused formulation and the seven-block
    unfused one both go through it unchanged.
    """
    sch = s_tir.Schedule(mod)
    for block in sch.get_child_blocks(sch.get_sblock("root")):
        n_spatial = sum(
            iter_var.iter_type == tir.IterVar.DataPar for iter_var in sch.get(block).iter_vars
        )
        loops = sch.get_loops(block)
        fused = sch.fuse(*loops[:n_spatial])
        block_idx, thread_idx = sch.split(fused, factors=[None, threads_per_block])
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(thread_idx, "threadIdx.x")
    return sch


# The two ways of writing the same attention, behind one name each. Both take
# the same keyword arguments and produce the same signature, so everything
# downstream -- schedule, build, check -- is variant-agnostic.
VARIANTS = {
    "flash": flash_attention,
    "naive": attention,
}

VARIANT_SUBTITLES = {
    "flash": "S materialised; softmax is one commutative reduce over the key axis",
    "naive": "S and P both materialised; softmax is max -> exp -> sum -> divide",
}


def build_cuda(
    shape: AttentionShape,
    *,
    variant: str,
    dtype: str,
    accum_dtype: str,
    threads_per_block: int,
):
    """Compile the shape-specialised kernel for the CUDA target.

    Returns the schedule alongside the executable so callers can dump the
    scheduled TIR without re-deriving it.
    """
    mod = VARIANTS[variant](**dataclasses.asdict(shape), dtype=dtype, accum_dtype=accum_dtype)
    sch = schedule_naive_cuda(mod, threads_per_block)
    return sch, tvm.compile(sch.mod, target="cuda")


# ---------------------------------------------------------------------------
# Numerical check against torch
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AccuracyReport:
    variant: str
    dtype: str
    accuracy: Accuracy


def make_inputs(shape: AttentionShape, *, dtype: str, seed: int):
    """Allocate the kernel's arguments on the GPU, in PrimFunc order."""
    import torch  # local: printing the IR should not require torch

    rng = generator(seed)
    lead = (shape.batch, shape.num_heads)
    return (
        randn(*lead, shape.seqlen_q, shape.head_dim, generator=rng, dtype=dtype),
        randn(*lead, shape.seqlen_k, shape.head_dim, generator=rng, dtype=dtype),
        randn(*lead, shape.seqlen_k, shape.head_dim, generator=rng, dtype=dtype),
        torch.empty(*lead, shape.seqlen_q, shape.head_dim, device="cuda", dtype=torch_dtype(dtype)),
    )


def torch_reference(q, k, v, out):
    """``F.scaled_dot_product_attention`` in fp32, on upcast inputs.

    Comparing against a half-precision SDPA would fold the reference's own
    rounding into the error, which is the same order of magnitude as what we are
    trying to measure.
    """
    import torch  # local: printing the IR should not require torch

    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def render_accuracy(console: Console, reports: list[AccuracyReport], shape: str) -> None:
    render_table(
        console,
        title="TVM CUDA vs torch SDPA (fp32 reference)",
        caption=str(shape),
        columns=[
            label("variant", lambda report: report.variant),
            label("dtype", lambda report: report.dtype),
            *accuracy_columns(),
        ],
        rows=reports,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    shape = parser.add_argument_group("attention shape")
    shape.add_argument("--batch", type=int, default=2, help="batch size (default: %(default)s)")
    shape.add_argument(
        "--num-heads", type=int, default=8, help="attention heads (default: %(default)s)"
    )
    shape.add_argument(
        "--seqlen-q",
        type=int,
        default=1024,
        help="query positions (default: %(default)s)",
    )
    shape.add_argument(
        "--seqlen-k",
        type=int,
        default=1024,
        help="key/value positions (default: %(default)s)",
    )
    shape.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="per-head dimension (default: %(default)s)",
    )

    numerics = parser.add_argument_group("numerics")
    numerics.add_argument(
        "--dtype",
        nargs="+",
        choices=SUPPORTED_DTYPES,
        # A bare string would be iterated character-by-character by main's loop,
        # since argparse passes defaults through untouched under nargs="+".
        default=["bfloat16"],
        help="input/output dtypes to check (default: bfloat16)",
    )
    numerics.add_argument(
        "--accum-dtype",
        default="float32",
        help="reduction dtype (default: %(default)s)",
    )
    numerics.add_argument("--rtol", type=float, default=None, help="override the per-dtype default")
    numerics.add_argument("--atol", type=float, default=None, help="override the per-dtype default")
    numerics.add_argument(
        "--seed", type=int, default=0, help="torch RNG seed (default: %(default)s)"
    )

    schedule = parser.add_argument_group("schedule")
    schedule.add_argument(
        "--variant",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
        help="attention formulation(s) to build (default: all)",
    )
    schedule.add_argument(
        "--threads-per-block", type=int, default=64, help="(default: %(default)s)"
    )

    output = parser.add_argument_group("output")
    output.add_argument("--no-ir", dest="show_ir", action="store_false", help="skip the s_tir dump")
    output.add_argument(
        "--dump_schedule_tir",
        action="store_true",
        help="dump the scheduled TIR, after scheduling and before codegen",
    )
    output.add_argument("--dump_cuda", action="store_true", help="dump the generated CUDA source")
    output.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="skip the torch comparison",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    if args.show_ir:
        for variant in args.variant:
            symbolic = VARIANTS[variant](dtype=args.dtype[0], accum_dtype=args.accum_dtype)
            render_source(
                console,
                symbolic["main"].script(),
                lexer="python",
                title=f"[bold]shape-generic s_tir[/] — {variant}, unscheduled te output",
                subtitle=VARIANT_SUBTITLES[variant],
            )

    if not (args.dump_schedule_tir or args.dump_cuda or args.check):
        return 0

    shape = AttentionShape(
        batch=args.batch,
        num_heads=args.num_heads,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        head_dim=args.head_dim,
    )
    reports: list[AccuracyReport] = []
    for variant in args.variant:
        for dtype in args.dtype:
            sch, kernel = build_cuda(
                shape,
                variant=variant,
                dtype=dtype,
                accum_dtype=args.accum_dtype,
                threads_per_block=args.threads_per_block,
            )

            if args.dump_schedule_tir:
                render_source(
                    console,
                    sch.mod["main"].script(),
                    lexer="python",
                    title=f"[bold]scheduled s_tir[/] — {variant}, {dtype}",
                    subtitle=f"schedule_naive_cuda, {args.threads_per_block} threads/block",
                )

            if args.dump_cuda:
                render_source(
                    console,
                    kernel.mod.imports[0].inspect_source(),
                    lexer="cuda",
                    title=f"[bold]generated CUDA[/] — {variant}, {dtype}",
                    subtitle=(
                        f"one thread per output element, {args.threads_per_block} threads/block"
                    ),
                )

            if args.check:
                rtol, atol = DEFAULT_TOLERANCE[dtype]
                reports.append(
                    AccuracyReport(
                        variant=variant,
                        dtype=dtype,
                        accuracy=check_against_torch(
                            kernel,
                            make_inputs(shape, dtype=dtype, seed=args.seed),
                            reference=torch_reference,
                            rtol=args.rtol if args.rtol is not None else rtol,
                            atol=args.atol if args.atol is not None else atol,
                        ),
                    )
                )

    if reports:
        render_accuracy(console, reports, shape)
    return 0 if all(report.accuracy.passed for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
