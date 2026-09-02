"""The comparison every example performs against its torch reference."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from .render import Column, verdict


@dataclasses.dataclass(frozen=True)
class Accuracy:
    """One kernel output measured against a higher-precision reference."""

    max_abs_err: float
    rel_err: float
    rtol: float
    atol: float
    passed: bool


def compare(actual, reference, *, rtol: float, atol: float) -> Accuracy:
    """Measure ``actual`` against ``reference``. Both are torch tensors.

    ``rel_err`` normalises by the reference's own peak rather than element-wise.
    A per-element ratio is dominated by entries where the reference is near zero
    -- after a ReLU half of them are exactly zero, and in a distance matrix the
    smallest entries sit on the diagonal -- which says nothing about the kernel.
    """
    import torch  # local: printing the IR should not require torch

    max_abs_err = (actual - reference).abs().max()
    return Accuracy(
        max_abs_err=max_abs_err.item(),
        rel_err=(max_abs_err / reference.abs().max()).item(),
        rtol=rtol,
        atol=atol,
        passed=bool(torch.allclose(actual, reference, rtol=rtol, atol=atol)),
    )


def _accuracy_of(row: Any) -> Accuracy:
    """Default accessor: by convention an example names the field ``accuracy``."""
    return row.accuracy


def error_columns(get: Callable[[Any], Accuracy] = _accuracy_of) -> list[Column]:
    """Absolute and normalised error."""
    return [
        Column("max abs err", lambda row: f"{get(row).max_abs_err:.3e}"),
        Column("err / ‖ref‖∞", lambda row: f"{get(row).rel_err:.3e}"),
    ]


def tolerance_column(get: Callable[[Any], Accuracy] = _accuracy_of) -> Column:
    return Column("rtol / atol", lambda row: f"{get(row).rtol:g} / {get(row).atol:g}")


def verdict_column(get: Callable[[Any], Accuracy] = _accuracy_of) -> Column:
    return Column("allclose", lambda row: verdict(get(row).passed), justify="center")


def accuracy_columns(get: Callable[[Any], Accuracy] = _accuracy_of) -> list[Column]:
    """The four columns an accuracy table ends with, in their usual order.

    ``get`` pulls the :class:`Accuracy` out of the example's own row type, so an
    example is free to carry labels and extra diagnostics alongside it. An
    example that needs its own columns *between* these can compose the pieces
    instead -- see ``pairwise_euclidean_distance_te.py``.
    """
    return [*error_columns(get), tolerance_column(get), verdict_column(get)]
