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
"""MPFR arithmetic and standard NumPy diagnostics for empirical precision tests."""

import dataclasses

import numpy as np

from .symbolic.canonicalize import Unsupported

try:
    import gmpy2
except ImportError:
    gmpy2 = None


def require_mpfr():
    if gmpy2 is None:
        raise Unsupported("high-precision references require gmpy2 (install the superopt extra)")
    return gmpy2


def context(bits):
    mp = require_mpfr()
    # A fresh context avoids inheriting a caller's IEEE exponent limits/traps.
    return mp.context(precision=bits, round=mp.RoundToNearest)


def mp_array(values):
    mp = require_mpfr()

    def convert(x):
        if isinstance(x, np.generic):
            x = x.item()
        return mp.mpfr(x)

    return np.asarray(np.frompyfunc(convert, 1, 1)(values), dtype=object)


def finite(values):
    return np.asarray(np.frompyfunc(require_mpfr().is_finite, 1, 1)(values), dtype=bool)


def references_agree(left, right, precision, dtypes):
    mp = require_mpfr()
    with context(precision):
        tolerance = mp.mpfr(2) ** -80
        for a, b, dtype in zip(left, right, dtypes):
            info = np.finfo(dtype)
            # Near exact zero, relative agreement is meaningless. Require an
            # absolute difference far below one target subnormal, not a unit-scale
            # atol that could hide a meaningful tiny result.
            floor = mp.mpfr(2) ** (info.minexp - info.nmant - 16)
            if a.shape != b.shape:
                return False
            for x, y in zip(a.flat, b.flat):
                if not mp.is_finite(x) or not mp.is_finite(y):
                    if not (x == y or (mp.is_nan(x) and mp.is_nan(y))):
                        return False
                elif abs(x - y) > max(floor, tolerance * abs(y)):
                    return False
    return len(left) == len(right)


@dataclasses.dataclass(frozen=True)
class ErrorMetrics:
    absolute: float
    relative: float
    normwise: float
    ulps: float
    nonfinite: int = 0


def errors(actual, reference, dtype, precision=256):
    """Errors before rounding the reference; ULPs against a dtype-rounded reference.

    Relative error is infinity for a nonzero error at an exactly zero reference.
    ULP distance is diagnostic, not a universal reduction acceptance criterion.
    """
    mp = require_mpfr()
    actual = np.asarray(actual)
    if actual.shape != reference.shape:
        raise ValueError("numerical comparison requires equal shapes")
    nonfinite = int(np.count_nonzero(~np.isfinite(actual)))
    if nonfinite or not np.all(finite(reference)):
        return ErrorMetrics(*(float("inf"),) * 4, nonfinite=nonfinite)
    if not actual.size:
        return ErrorMetrics(0.0, 0.0, 0.0, 0.0)
    with context(precision + 64):
        difference = abs(mp_array(actual) - reference)
        absolute = max(difference.flat)
        scale = max(abs(reference).flat)
        relative = max(
            d / abs(r) if r else (mp.mpfr("inf") if d else mp.mpfr(0))
            for d, r in zip(difference.flat, reference.flat)
        )
        normwise = absolute / scale if scale else absolute
    # Round directly with the target's IEEE exponent/subnormal rules, avoiding
    # double rounding through float64 in the diagnostic reference.
    with mp.context(mp.ieee(np.dtype(dtype).itemsize * 8)):
        rounded = np.asarray(mp_array(reference), dtype=dtype)
    with np.errstate(all="ignore"):
        distances = np.testing.assert_array_max_ulp(
            actual.astype(dtype), rounded, maxulp=float("inf"), dtype=dtype
        )
    return ErrorMetrics(float(absolute), float(relative), float(normwise), float(np.max(distances)))


def regression_check(
    actual,
    original,
    reference,
    sensitivity,
    dtype,
    *,
    factor,
    relative_eps,
    absolute_tolerance,
    precision,
):
    """Apply an explicitly empirical, elementwise error-regression budget.

    NumPy supplies the assertion and diagnostics; the budget is project policy.
    The sampled sensitivity is not a proven condition number or error bound.
    """
    if np.shape(actual) != np.shape(reference) or np.shape(original) != np.shape(reference):
        return False
    if np.any(~np.isfinite(actual) & np.isfinite(original)):
        return False
    with context(precision + 64):
        error = abs(mp_array(actual) - reference)
        source_error = abs(mp_array(original) - reference)
        allowance = (
            factor * np.maximum(source_error, sensitivity)
            + relative_eps * float(np.finfo(dtype).eps) * abs(reference)
            + absolute_tolerance
        )
        # Normalize in MPFR before passing dimensionless residuals to NumPy.
        # A zero budget permits only exact agreement. NaN never passes.
        ratio = np.asarray(
            [
                e / b if b else (float("inf") if e else 0.0)
                for e, b in zip(error.flat, allowance.flat)
            ],
            dtype=np.float64,
        )
    try:
        np.testing.assert_allclose(ratio, np.zeros_like(ratio), rtol=0, atol=1, equal_nan=False)
    except AssertionError:
        return False
    return True
