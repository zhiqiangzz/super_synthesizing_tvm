"""Running a compiled kernel against a torch reference.

The examples differ in what they compute and in how they build their inputs.
They do not differ in the mechanics around that: seed a generator, allocate,
hand the tensors to the kernel through DLPack, wait for the GPU, then recompute
the same thing in a precision the kernel is not trying to match. That part is
here, and the parts that do differ arrive as callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import tvm

from .accuracy import Accuracy, compare


def torch_dtype(name: str):
    """``"bfloat16"`` -> ``torch.bfloat16``, for any dtype torch exposes."""
    import torch  # local: printing the IR should not require torch

    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name!r} is not a torch dtype")
    return dtype


def generator(seed: int, device: str = "cuda"):
    """A seeded RNG, so rerunning an example reproduces its error figures."""
    import torch  # local: printing the IR should not require torch

    return torch.Generator(device=device).manual_seed(seed)


def randn(
    *size: int,
    generator,
    dtype: str,
    scale: float = 1.0,
    offset: float = 0.0,
    device: str = "cuda",
):
    """Normal samples, scaled and shifted in fp32 and narrowed to ``dtype`` once.

    Sampling wide and narrowing at the end keeps the requested spread exact:
    scaling after the cast would round twice, and a large ``offset`` applied in
    float16 would swallow the spread it is supposed to sit on top of.
    """
    import torch  # local: printing the IR should not require torch

    raw = torch.randn(*size, generator=generator, device=device, dtype=torch.float32)
    return (raw * scale + offset).to(torch_dtype(dtype))


def launch(kernel, *tensors) -> None:
    """Call the compiled PrimFunc on torch tensors and wait for it to finish."""
    import torch  # local: printing the IR should not require torch

    kernel["main"](*(tvm.runtime.from_dlpack(t) for t in tensors))
    torch.cuda.synchronize()


def _last_output(*tensors):
    """Default output accessor: the PrimFunc's last argument, upcast to fp32."""
    return tensors[-1].float()


def check_against_torch(
    kernel,
    tensors: Sequence,
    *,
    reference: Callable[..., Any],
    actual: Callable[..., Any] = _last_output,
    rtol: float,
    atol: float,
) -> Accuracy:
    """Launch ``kernel`` on ``tensors``, then measure the result against torch.

    ``tensors`` is the PrimFunc's whole argument list, inputs and outputs alike,
    in declaration order. Both callbacks are handed it splatted once the kernel
    has finished: ``actual`` pulls out what was computed, ``reference`` recomputes
    it from the very tensors the kernel saw.

    An example whose kernel writes more than one output -- or that wants extra
    diagnostics off the result -- calls :func:`launch` and :func:`compare`
    directly instead; this is only the common single-output case.
    """
    launch(kernel, *tensors)
    return compare(actual(*tensors), reference(*tensors), rtol=rtol, atol=atol)
