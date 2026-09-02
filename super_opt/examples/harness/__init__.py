"""Shared reporting harness for the TE examples.

Every example in this directory builds a kernel, checks it against a torch
reference and (sometimes) times it. Only the kernel is interesting; the
measuring and the tables around it are the same everywhere, so they live here.

The seam is a callback in both directions. ``benchmark`` takes the kernel and
its baseline as two no-argument callables, so an example decides for itself what
it is racing against. ``render_table`` takes :class:`Column` objects that each
carry a lambda pulling their cell out of the example's own row type, so no
example has to bend its report into a shape this package prescribes.
``check_against_torch`` is the same idea for the check itself: it owns the
launch and the measurement, and takes the reference as a callable over the
kernel's own arguments.
"""

from .accuracy import (
    Accuracy,
    accuracy_columns,
    compare,
    error_columns,
    tolerance_column,
    verdict_column,
)
from .execution import check_against_torch, generator, launch, randn, torch_dtype
from .render import Column, label, render_source, render_table, verdict
from .timing import MAX_BENCH_ITERS, MIN_BENCH_ITERS, Timing, benchmark, time_ms, timing_columns

__all__ = [
    "MAX_BENCH_ITERS",
    "MIN_BENCH_ITERS",
    "Accuracy",
    "Column",
    "Timing",
    "accuracy_columns",
    "benchmark",
    "check_against_torch",
    "compare",
    "error_columns",
    "generator",
    "label",
    "launch",
    "randn",
    "render_source",
    "render_table",
    "time_ms",
    "timing_columns",
    "tolerance_column",
    "torch_dtype",
    "verdict",
    "verdict_column",
]
