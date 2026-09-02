"""Timing a kernel against a baseline, both supplied as callables."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from .render import Column

# Bounds on the auto-chosen iteration count when ``iters`` is not given.
MIN_BENCH_ITERS = 3
MAX_BENCH_ITERS = 200


@dataclasses.dataclass(frozen=True)
class Timing:
    """A kernel's wall time next to a baseline's, over the same work."""

    tvm_ms: float
    baseline_ms: float
    flops: int

    def _tflops(self, milliseconds: float) -> float:
        return self.flops / (milliseconds * 1e-3) / 1e12

    @property
    def tvm_tflops(self) -> float:
        return self._tflops(self.tvm_ms)

    @property
    def baseline_tflops(self) -> float:
        return self._tflops(self.baseline_ms)

    @property
    def slowdown(self) -> float:
        return self.tvm_ms / self.baseline_ms


def time_ms(
    run: Callable[[], Any], *, warmup: int = 3, iters: int | None = None, target_ms: float = 200.0
) -> float:
    """Milliseconds per call of ``run``, measured with CUDA events.

    With ``iters=None`` the count comes from a single probe. Schedules in these
    examples span four orders of magnitude, so a fixed count either takes
    minutes on the slow one or measures nothing but launch noise on the fast one.
    """
    import torch  # local: printing the IR should not require torch

    def elapsed_ms(count: int) -> float:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            run()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / count

    for _ in range(warmup):
        run()
    count = iters
    if count is None:
        probe = elapsed_ms(1)
        count = int(min(MAX_BENCH_ITERS, max(MIN_BENCH_ITERS, target_ms / max(probe, 1e-6))))
    return elapsed_ms(count)


def benchmark(
    *,
    run: Callable[[], Any],
    baseline: Callable[[], Any],
    flops: int,
    warmup: int = 3,
    iters: int | None = None,
    target_ms: float = 200.0,
) -> Timing:
    """Time ``run`` and ``baseline`` under identical conditions.

    Both arrive as callables so the caller keeps ownership of the tensors and of
    what "the baseline" means -- cuBLAS, SDPA, a torch one-liner. Neither should
    allocate, or the measurement includes the caching allocator.
    """
    return Timing(
        tvm_ms=time_ms(run, warmup=warmup, iters=iters, target_ms=target_ms),
        baseline_ms=time_ms(baseline, warmup=warmup, iters=iters, target_ms=target_ms),
        flops=flops,
    )


def timing_columns(
    baseline_name: str = "torch",
    get: Callable[[Any], Timing] = lambda row: row.timing,
) -> list[Column]:
    """The five columns every timing table ends with."""
    return [
        Column("TVM ms", lambda row: f"{get(row).tvm_ms:.3f}"),
        Column("TVM TF/s", lambda row: f"{get(row).tvm_tflops:.1f}"),
        Column(f"{baseline_name} ms", lambda row: f"{get(row).baseline_ms:.3f}"),
        Column(f"{baseline_name} TF/s", lambda row: f"{get(row).baseline_tflops:.1f}"),
        Column("slower", lambda row: f"[yellow]{get(row).slowdown:.0f}x[/]"),
    ]
