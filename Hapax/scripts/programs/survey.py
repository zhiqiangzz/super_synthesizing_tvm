"""The survey candidates: each as an unfused TE program and a float64 numpy reference.

Every candidate is written the way a user would write the multi-pass program:
one named ``te.compute`` per stage, float32, symbolic extents, every reduction
with its own ``te.reduce_axis``. Reduce axes are named ``j`` (``r``/``d``/...
for other axes); the extra axis of an elementwise tensor is named ``k`` so that
no reduce axis shares a name with a compute axis.

``build()`` returns ``(inputs, outputs)``; ``ref(*arrays)`` returns a tuple of
float64 arrays; ``positive`` lists the inputs that have to be positive for the
operator to mean anything (in addition to what the tool detects itself).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from tvm import te
from tvm_tirx import tirx as tir

F = "float32"
LN_EPS = 1.0e-5


@dataclasses.dataclass(frozen=True)
class Cand:
    key: str  # e.g. "07b"
    name: str
    doc: str
    build: Callable[[], tuple[list, list]]
    ref: Callable[..., tuple]
    positive: tuple[str, ...] = ()
    offset_ok: bool = True  # is a +1e4 offset of the inputs a meaningful test?
    note: str = ""


CANDS: dict[str, Cand] = {}


def reg(key, name, doc, build, ref, positive=(), offset_ok=True, note=""):
    assert key not in CANDS, key
    CANDS[key] = Cand(key, name, doc, build, ref, tuple(positive), offset_ok, note)


def c(v: float):
    return tir.const(float(v), F)


def rows(name: str = "X", like=None):
    shape = (te.var("n"), te.var("m")) if like is None else like.shape
    return te.placeholder(shape, name=name, dtype=F)


def rsum(X, f, name: str):
    """``out[i] = Σ_j f(i, j)`` over the second axis of ``X``."""
    n, m = X.shape[0], X.shape[1]
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.sum(f(i, j), axis=j), name=name)


def rmax(X, f, name: str):
    n, m = X.shape[0], X.shape[1]
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.max(f(i, j), axis=j), name=name)


def rmin(X, f, name: str):
    n, m = X.shape[0], X.shape[1]
    j = te.reduce_axis((0, m), "j")
    return te.compute((n,), lambda i: te.min(f(i, j), axis=j), name=name)


def vec(X, f, name: str):
    """A per-row value ``out[i] = f(i)``."""
    return te.compute((X.shape[0],), f, name=name)


def mat(X, f, name: str):
    """An elementwise tensor ``out[i, k] = f(i, k)``."""
    return te.compute((X.shape[0], X.shape[1]), lambda i, k: f(i, k), name=name)


def fm(X):
    return X.shape[1].astype(F)


def mean_of(X, name: str = "mean"):
    total = rsum(X, lambda i, j: X[i, j], f"total_{name}")
    return vec(X, lambda i: total[i] / fm(X), name)


def d64(*arrays):
    return [np.asarray(a, dtype=np.float64) for a in arrays]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def weighted_variance():
    X = rows("X")
    W = rows("W", like=X)
    sw = rsum(X, lambda i, j: W[i, j], "sw")
    swx = rsum(X, lambda i, j: W[i, j] * X[i, j], "swx")
    mean = vec(X, lambda i: swx[i] / sw[i], "mean")
    ss = rsum(X, lambda i, j: W[i, j] * (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    out = vec(X, lambda i: ss[i] / sw[i], "out")
    return [X, W], [out]


def weighted_variance_ref(x, w):
    x, w = d64(x, w)
    sw = w.sum(1)
    mu = (w * x).sum(1) / sw
    return ((w * (x - mu[:, None]) ** 2).sum(1) / sw,)


reg("01", "weighted_variance", "Σ w (x-μ_w)² / Σw", weighted_variance, weighted_variance_ref, ("W",))


def _std_moment(power: int):
    def build():
        X = rows("X")
        mean = mean_of(X)
        ss = rsum(X, lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
        m2 = vec(X, lambda i: ss[i] / fm(X), "m2")
        sp = rsum(
            X,
            lambda i, j: tir.power((X[i, j] - mean[i]) / te.sqrt(m2[i]), c(power)),
            f"s{power}",
        )
        out = vec(X, lambda i: sp[i] / fm(X), "out")
        return [X], [out]

    def ref(x):
        (x,) = d64(x)
        mu = x.mean(1, keepdims=True)
        m2 = ((x - mu) ** 2).mean(1, keepdims=True)
        return ((((x - mu) / np.sqrt(m2)) ** power).mean(1),)

    return build, ref


reg("02", "skewness", "Σ ((x-μ)/sqrt(m2))³ / m  (nested form)", *_std_moment(3))
reg("03", "kurtosis", "Σ ((x-μ)/sqrt(m2))⁴ / m  (nested form)", *_std_moment(4))


def _xy_sums():
    X = rows("X")
    Y = rows("Y", like=X)
    mx, my = mean_of(X, "mean_x"), mean_of(Y, "mean_y")
    sxy = rsum(X, lambda i, j: (X[i, j] - mx[i]) * (Y[i, j] - my[i]), "sxy")
    sxx = rsum(X, lambda i, j: (X[i, j] - mx[i]) * (X[i, j] - mx[i]), "sxx")
    return X, Y, mx, my, sxy, sxx


def pearson():
    X, Y, mx, my, sxy, sxx = _xy_sums()
    syy = rsum(X, lambda i, j: (Y[i, j] - my[i]) * (Y[i, j] - my[i]), "syy")
    out = vec(X, lambda i: sxy[i] / te.sqrt(sxx[i] * syy[i]), "out")
    return [X, Y], [out]


def pearson_ref(x, y):
    x, y = d64(x, y)
    dx, dy = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    return ((dx * dy).sum(1) / np.sqrt((dx * dx).sum(1) * (dy * dy).sum(1)),)


reg("04", "pearson", "sxy / sqrt(sxx·syy)", pearson, pearson_ref)


def linreg_residual():
    X, Y, mx, my, sxy, sxx = _xy_sums()
    beta = vec(X, lambda i: sxy[i] / sxx[i], "beta")

    def res(i, j):
        return (Y[i, j] - my[i]) - beta[i] * (X[i, j] - mx[i])

    out = rsum(X, lambda i, j: res(i, j) * res(i, j), "out")
    return [X, Y], [out]


def linreg_residual_ref(x, y):
    x, y = d64(x, y)
    dx, dy = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    beta = (dx * dy).sum(1, keepdims=True) / (dx * dx).sum(1, keepdims=True)
    return (((dy - beta * dx) ** 2).sum(1),)


reg("05", "linreg_residual", "Σ ((y-μy) - β(x-μx))², β = sxy/sxx", linreg_residual, linreg_residual_ref)


def _mean_sigma(X):
    mean = mean_of(X)
    ss = rsum(X, lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    var = vec(X, lambda i: ss[i] / fm(X), "var")
    sigma = vec(X, lambda i: te.sqrt(var[i]), "sigma")
    return mean, sigma


def zscore_max():
    X = rows("X")
    mean, sigma = _mean_sigma(X)
    out = rmax(X, lambda i, j: (X[i, j] - mean[i]) / sigma[i], "out")
    return [X], [out]


def zscore_max_ref(x):
    (x,) = d64(x)
    return (((x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)).max(1),)


reg("06", "zscore_max", "max_j (x-μ)/σ", zscore_max, zscore_max_ref)


def _max_abs_dev(spell: str):
    def build():
        X = rows("X")
        mean = mean_of(X)
        if spell == "abs":
            out = rmax(X, lambda i, j: te.abs(X[i, j] - mean[i]), "out")
        else:
            out = rmax(X, lambda i, j: tir.max(X[i, j] - mean[i], mean[i] - X[i, j]), "out")
        return [X], [out]

    return build


def max_abs_dev_ref(x):
    (x,) = d64(x)
    return (np.abs(x - x.mean(1, keepdims=True)).max(1),)


reg("07a", "max_abs_dev[te.abs]", "max_j |x-μ| with te.abs", _max_abs_dev("abs"), max_abs_dev_ref)
reg("07b", "max_abs_dev[max(d,-d)]", "max_j max(x-μ, μ-x)", _max_abs_dev("max"), max_abs_dev_ref)


def _minmax_mean(spell: str):
    def build():
        X = rows("X")
        if spell == "min":
            lo = rmin(X, lambda i, j: X[i, j], "lo")
        else:
            nlo = rmax(X, lambda i, j: -X[i, j], "nlo")
            lo = vec(X, lambda i: -nlo[i], "lo")
        hi = rmax(X, lambda i, j: X[i, j], "hi")
        out = rsum(X, lambda i, j: (X[i, j] - lo[i]) / (hi[i] - lo[i]), "out")
        return [X], [out]

    return build


def minmax_mean_ref(x):
    (x,) = d64(x)
    lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
    return (((x - lo) / (hi - lo)).sum(1),)


reg("08a", "minmax_mean[te.min]", "Σ (x-lo)/(hi-lo), lo by te.min", _minmax_mean("min"), minmax_mean_ref)
reg("08b", "minmax_mean[-max(-x)]", "Σ (x-lo)/(hi-lo), lo = -max(-x)", _minmax_mean("negmax"), minmax_mean_ref)


def _sum_abs_from_max(spell: str):
    def build():
        X = rows("X")
        mx = rmax(X, lambda i, j: X[i, j], "mx")
        if spell == "abs":
            out = rsum(X, lambda i, j: te.abs(X[i, j] - mx[i]), "out")
        elif spell == "max":
            out = rsum(X, lambda i, j: tir.max(X[i, j] - mx[i], mx[i] - X[i, j]), "out")
        else:  # the user resolves the sign by hand: |x - max| = max - x
            out = rsum(X, lambda i, j: mx[i] - X[i, j], "out")
        return [X], [out]

    return build


def sum_abs_from_max_ref(x):
    (x,) = d64(x)
    return (np.abs(x - x.max(1, keepdims=True)).sum(1),)


reg("09a", "sum_abs_from_max[te.abs]", "Σ |x - max x| with te.abs", _sum_abs_from_max("abs"), sum_abs_from_max_ref)
reg("09b", "sum_abs_from_max[max(d,-d)]", "Σ max(x-mx, mx-x)", _sum_abs_from_max("max"), sum_abs_from_max_ref)
reg(
    "09c",
    "sum_abs_from_max[mx-x]",
    "Σ (mx - x): the sign resolved by hand (extra)",
    _sum_abs_from_max("signed"),
    sum_abs_from_max_ref,
    note="extra variant, not in the list: |x - max| written as max - x",
)


def sq_from_max():
    X = rows("X")
    mx = rmax(X, lambda i, j: X[i, j], "mx")
    out = rsum(X, lambda i, j: (X[i, j] - mx[i]) * (X[i, j] - mx[i]), "out")
    return [X], [out]


def sq_from_max_ref(x):
    (x,) = d64(x)
    return (((x - x.max(1, keepdims=True)) ** 2).sum(1),)


reg("10", "sq_from_max", "Σ (x - max x)²", sq_from_max, sq_from_max_ref)


def _count_above_mean(spell: str):
    def build():
        X = rows("X")
        mean = mean_of(X)
        if spell == "ite":
            out = rsum(X, lambda i, j: tir.if_then_else(X[i, j] > mean[i], c(1.0), c(0.0)), "out")
        else:
            out = rsum(X, lambda i, j: (X[i, j] > mean[i]).astype(F), "out")
        return [X], [out]

    return build


def count_above_mean_ref(x):
    (x,) = d64(x)
    return ((x > x.mean(1, keepdims=True)).sum(1).astype(np.float64),)


reg("11a", "count_above_mean[if_then_else]", "Σ [x > μ] by if_then_else", _count_above_mean("ite"), count_above_mean_ref)
reg("11b", "count_above_mean[cast]", "Σ [x > μ] by a cast of the comparison", _count_above_mean("cast"), count_above_mean_ref)


def exp_centered():
    X = rows("X")
    mean = mean_of(X)
    out = rsum(X, lambda i, j: te.exp(X[i, j] - mean[i]), "out")
    return [X], [out]


def exp_centered_ref(x):
    (x,) = d64(x)
    return (np.exp(x - x.mean(1, keepdims=True)).sum(1),)


reg("12", "exp_centered", "Σ exp(x - μ)", exp_centered, exp_centered_ref)


def log_ratio():
    X = rows("X")
    mean = mean_of(X)
    out = rsum(X, lambda i, j: te.log(X[i, j] / mean[i]), "out")
    return [X], [out]


def log_ratio_ref(x):
    (x,) = d64(x)
    return (np.log(x / x.mean(1, keepdims=True)).sum(1),)


reg("13", "log_ratio", "Σ log(x/μ), x > 0", log_ratio, log_ratio_ref, ("X",))


def cv_squared():
    X = rows("X")
    mean = mean_of(X)
    out = rsum(X, lambda i, j: (X[i, j] / mean[i] - c(1.0)) * (X[i, j] / mean[i] - c(1.0)), "out")
    return [X], [out]


def cv_squared_ref(x):
    (x,) = d64(x)
    return (((x / x.mean(1, keepdims=True) - 1.0) ** 2).sum(1),)


reg("14", "cv_squared", "Σ (x/μ - 1)²", cv_squared, cv_squared_ref, ("X",))


def autocorr_lag1():
    X = rows("X")
    n, m = X.shape
    mean = mean_of(X)
    j = te.reduce_axis((0, m - 1), "j")
    out = te.compute(
        (n,), lambda i: te.sum((X[i, j] - mean[i]) * (X[i, j + 1] - mean[i]), axis=j), name="out"
    )
    return [X], [out]


def autocorr_lag1_ref(x):
    (x,) = d64(x)
    d = x - x.mean(1, keepdims=True)
    return ((d[:, :-1] * d[:, 1:]).sum(1),)


reg("15", "autocorr_lag1", "Σ_{j<m-1} (x_j-μ)(x_{j+1}-μ)", autocorr_lag1, autocorr_lag1_ref)


# ---------------------------------------------------------------------------
# softmax family
# ---------------------------------------------------------------------------
def _softmax(S, tag: str = ""):
    """``mx, e, den, P`` of the rows of ``S``: one named stage each."""
    mx = rmax(S, lambda i, j: S[i, j], f"mx{tag}")
    e = mat(S, lambda i, k: te.exp(S[i, k] - mx[i]), f"e{tag}")
    den = rsum(S, lambda i, j: e[i, j], f"den{tag}")
    P = mat(S, lambda i, k: e[i, k] / den[i], f"P{tag}")
    return mx, e, den, P


def _np_softmax(s):
    w = np.exp(s - s.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)


def softmax_variance():
    S = rows("S")
    V = rows("V", like=S)
    _, _, _, P = _softmax(S)
    ev = rsum(S, lambda i, j: P[i, j] * V[i, j], "ev")
    out = rsum(S, lambda i, j: P[i, j] * (V[i, j] - ev[i]) * (V[i, j] - ev[i]), "out")
    return [S, V], [out]


def softmax_variance_ref(s, v):
    s, v = d64(s, v)
    p = _np_softmax(s)
    ev = (p * v).sum(1, keepdims=True)
    return ((p * (v - ev) ** 2).sum(1),)


reg("16", "softmax_variance", "Σ p (v - Σ p v)², p = softmax(S)", softmax_variance, softmax_variance_ref)


def renyi2():
    X = rows("X")
    _, _, _, P = _softmax(X)
    out = rsum(X, lambda i, j: P[i, j] * P[i, j], "out")
    return [X], [out]


def renyi2_ref(x):
    (x,) = d64(x)
    return ((_np_softmax(x) ** 2).sum(1),)


reg("17", "renyi2", "Σ p², p = softmax(x)", renyi2, renyi2_ref)


def ess():
    LW = rows("LW")
    mx = rmax(LW, lambda i, j: LW[i, j], "mx")
    w = mat(LW, lambda i, k: te.exp(LW[i, k] - mx[i]), "w")
    sw = rsum(LW, lambda i, j: w[i, j], "sw")
    sw2 = rsum(LW, lambda i, j: w[i, j] * w[i, j], "sw2")
    out = vec(LW, lambda i: sw[i] * sw[i] / sw2[i], "out")
    return [LW], [out]


def ess_ref(lw):
    (lw,) = d64(lw)
    w = np.exp(lw - lw.max(1, keepdims=True))
    return (w.sum(1) ** 2 / (w * w).sum(1),)


reg("18", "ess", "(Σw)² / Σw², w = exp(lw - max lw)", ess, ess_ref)

TEMP = 2.0


def kd_kl():
    T = rows("T")
    S = rows("S", like=T)
    zt = mat(T, lambda i, k: T[i, k] / c(TEMP), "zt")
    zs = mat(T, lambda i, k: S[i, k] / c(TEMP), "zs")
    mxt, _, dent, Pt = _softmax(zt, "_t")
    mxs, _, dens, _ = _softmax(zs, "_s")
    logpt = mat(T, lambda i, k: zt[i, k] - mxt[i] - te.log(dent[i]), "logp_t")
    logps = mat(T, lambda i, k: zs[i, k] - mxs[i] - te.log(dens[i]), "logp_s")
    out = rsum(T, lambda i, j: Pt[i, j] * (logpt[i, j] - logps[i, j]), "out")
    return [T, S], [out]


def kd_kl_ref(t, s):
    t, s = d64(t, s)
    pt, ps = _np_softmax(t / TEMP), _np_softmax(s / TEMP)
    return ((pt * (np.log(pt) - np.log(ps))).sum(1),)


reg(
    "19",
    "kd_kl",
    "Σ p_t (log p_t - log p_s), p = softmax(./2)",
    kd_kl,
    kd_kl_ref,
    offset_ok=False,
    note="offset: softmax is shift invariant, +1e4 only tests the max shift",
)


def softmax_jvp():
    X = rows("X")
    G = rows("G", like=X)
    _, _, _, P = _softmax(X)
    gp = rsum(X, lambda i, j: G[i, j] * P[i, j], "gp")
    out = mat(X, lambda i, k: P[i, k] * (G[i, k] - gp[i]), "out")
    return [X, G], [out]


def softmax_jvp_ref(x, g):
    x, g = d64(x, g)
    p = _np_softmax(x)
    return (p * (g - (g * p).sum(1, keepdims=True)),)


reg("20", "softmax_jvp", "p (g - Σ g p), p = softmax(x)  (elementwise output)", softmax_jvp, softmax_jvp_ref)


def hist_entropy():
    X = rows("X")
    s = rsum(X, lambda i, j: X[i, j], "s")
    P = mat(X, lambda i, k: X[i, k] / s[i], "P")
    H = rsum(X, lambda i, j: P[i, j] * te.log(P[i, j]), "plogp")
    out = vec(X, lambda i: -H[i], "out")
    return [X], [out]


def hist_entropy_ref(x):
    (x,) = d64(x)
    p = x / x.sum(1, keepdims=True)
    return (-(p * np.log(p)).sum(1),)


reg("21", "hist_entropy", "-Σ (x/s) log(x/s), s = Σx, x > 0", hist_entropy, hist_entropy_ref, ("X",))


def hist_kl():
    X = rows("X")
    Y = rows("Y", like=X)
    sx = rsum(X, lambda i, j: X[i, j], "sx")
    sy = rsum(X, lambda i, j: Y[i, j], "sy")
    Px = mat(X, lambda i, k: X[i, k] / sx[i], "Px")
    Py = mat(X, lambda i, k: Y[i, k] / sy[i], "Py")
    out = rsum(X, lambda i, j: Px[i, j] * te.log(Px[i, j] / Py[i, j]), "out")
    return [X, Y], [out]


def hist_kl_ref(x, y):
    x, y = d64(x, y)
    px, py = x / x.sum(1, keepdims=True), y / y.sum(1, keepdims=True)
    return ((px * np.log(px / py)).sum(1),)


reg("22", "hist_kl", "Σ (x/sx) log((x/sx)/(y/sy)), x, y > 0", hist_kl, hist_kl_ref, ("X", "Y"))


def layernorm_backward():
    X = rows("X")
    G = rows("G", like=X)
    mean = mean_of(X)
    ss = rsum(X, lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    var = vec(X, lambda i: ss[i] / fm(X), "var")
    sigma = vec(X, lambda i: te.sqrt(var[i] + c(LN_EPS)), "sigma")
    xh = mat(X, lambda i, k: (X[i, k] - mean[i]) / sigma[i], "xh")
    sg = rsum(X, lambda i, j: G[i, j], "sg")
    a = vec(X, lambda i: sg[i] / fm(X), "a")
    sgx = rsum(X, lambda i, j: G[i, j] * xh[i, j], "sgx")
    b = vec(X, lambda i: sgx[i] / fm(X), "b")
    out = mat(X, lambda i, k: (G[i, k] - a[i] - xh[i, k] * b[i]) / sigma[i], "out")
    return [X, G], [out]


def layernorm_backward_ref(x, g):
    x, g = d64(x, g)
    mu = x.mean(1, keepdims=True)
    sigma = np.sqrt(x.var(1, keepdims=True) + LN_EPS)
    xh = (x - mu) / sigma
    a = g.mean(1, keepdims=True)
    b = (g * xh).mean(1, keepdims=True)
    return ((g - a - xh * b) / sigma,)


reg("23", "layernorm_backward", "(g - mean(g) - xh·mean(g·xh))/σ  (elementwise output)", layernorm_backward, layernorm_backward_ref)


def clip_loss():
    n = te.var("n")
    S = te.placeholder((n, n), name="S", dtype=F)
    j1 = te.reduce_axis((0, n), "j")
    mx_r = te.compute((n,), lambda i: te.max(S[i, j1], axis=j1), name="mx_row")
    j2 = te.reduce_axis((0, n), "j")
    den_r = te.compute((n,), lambda i: te.sum(te.exp(S[i, j2] - mx_r[i]), axis=j2), name="den_row")
    lse_r = te.compute((n,), lambda i: mx_r[i] + te.log(den_r[i]), name="lse_row")
    r1 = te.reduce_axis((0, n), "r")
    mx_c = te.compute((n,), lambda q: te.max(S[r1, q], axis=r1), name="mx_col")
    r2 = te.reduce_axis((0, n), "r")
    den_c = te.compute((n,), lambda q: te.sum(te.exp(S[r2, q] - mx_c[q]), axis=r2), name="den_col")
    lse_c = te.compute((n,), lambda q: mx_c[q] + te.log(den_c[q]), name="lse_col")
    d1 = te.reduce_axis((0, n), "d")
    loss_r = te.compute((1,), lambda z: te.sum(lse_r[d1] - S[d1, d1], axis=d1), name="loss_row")
    d2 = te.reduce_axis((0, n), "d")
    loss_c = te.compute((1,), lambda z: te.sum(lse_c[d2] - S[d2, d2], axis=d2), name="loss_col")
    loss = te.compute((1,), lambda z: loss_r[z] + loss_c[z], name="loss")
    return [S], [loss]


def clip_loss_ref(s):
    (s,) = d64(s)

    def lse(a, axis):
        mx = a.max(axis)
        return mx + np.log(np.exp(a - np.expand_dims(mx, axis)).sum(axis))

    diag = np.diag(s)
    return (np.array([(lse(s, 1) - diag).sum() + (lse(s, 0) - diag).sum()]),)


reg(
    "24",
    "clip_loss",
    "Σ_i (lse_row - S_ii) + Σ_j (lse_col - S_jj), S square",
    clip_loss,
    clip_loss_ref,
    offset_ok=False,
    note="offset: lse(S) - S_ii is shift invariant; with +1e4 the float32 result is cancellation only",
)


def _boltzmann(spell: str):
    def build():
        E = rows("E")
        if spell == "min":
            lo = rmin(E, lambda i, j: E[i, j], "lo")
        else:
            nlo = rmax(E, lambda i, j: -E[i, j], "nlo")
            lo = vec(E, lambda i: -nlo[i], "lo")
        w = mat(E, lambda i, k: te.exp(-(E[i, k] - lo[i])), "w")
        Z = rsum(E, lambda i, j: w[i, j], "Z")
        e1s = rsum(E, lambda i, j: E[i, j] * w[i, j], "e1s")
        e1 = vec(E, lambda i: e1s[i] / Z[i], "e1")
        cs = rsum(E, lambda i, j: (E[i, j] - e1[i]) * (E[i, j] - e1[i]) * w[i, j], "cs")
        out = vec(E, lambda i: cs[i] / Z[i], "out")
        return [E], [out]

    return build


def boltzmann_ref(e):
    (e,) = d64(e)
    w = np.exp(-(e - e.min(1, keepdims=True)))
    z = w.sum(1, keepdims=True)
    e1 = (e * w).sum(1, keepdims=True) / z
    return ((((e - e1) ** 2) * w).sum(1) / z[:, 0],)


reg("25a", "boltzmann_heat_capacity[te.min]", "Σ (E-<E>)² w / Z, w = exp(-(E - min E))", _boltzmann("min"), boltzmann_ref)
reg("25b", "boltzmann_heat_capacity[-max(-E)]", "the same with min E = -max(-E)", _boltzmann("negmax"), boltzmann_ref)


# ---------------------------------------------------------------------------
# expected negatives
# ---------------------------------------------------------------------------
def logitnorm_lse():
    X = rows("X")
    ssq = rsum(X, lambda i, j: X[i, j] * X[i, j], "ssq")
    nrm = vec(X, lambda i: te.sqrt(ssq[i]), "nrm")
    den = rsum(X, lambda i, j: te.exp(X[i, j] / nrm[i]), "den")
    out = vec(X, lambda i: te.log(den[i]), "out")
    return [X], [out]


def logitnorm_lse_ref(x):
    (x,) = d64(x)
    nrm = np.sqrt((x * x).sum(1, keepdims=True))
    return (np.log(np.exp(x / nrm).sum(1)),)


reg("26", "logitnorm_lse", "log Σ exp(x / ‖x‖)", logitnorm_lse, logitnorm_lse_ref)


def std_softmax_den():
    X = rows("X")
    mean, sigma = _mean_sigma(X)
    out = rsum(X, lambda i, j: te.exp((X[i, j] - mean[i]) / sigma[i]), "out")
    return [X], [out]


def std_softmax_den_ref(x):
    (x,) = d64(x)
    return (np.exp((x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)).sum(1),)


reg("27", "std_softmax_den", "Σ exp((x-μ)/σ)", std_softmax_den, std_softmax_den_ref)


def maxtemp_den():
    X = rows("X")
    mx = rmax(X, lambda i, j: X[i, j], "mx")
    out = rsum(X, lambda i, j: te.exp(X[i, j] / mx[i]), "out")
    return [X], [out]


def maxtemp_den_ref(x):
    (x,) = d64(x)
    return (np.exp(x / x.max(1, keepdims=True)).sum(1),)


reg("28", "maxtemp_den", "Σ exp(x / max x), x > 0", maxtemp_den, maxtemp_den_ref, ("X",))


def _log_from_min(spell: str):
    def build():
        X = rows("X")
        if spell == "min":
            lo = rmin(X, lambda i, j: X[i, j], "lo")
        else:
            nlo = rmax(X, lambda i, j: -X[i, j], "nlo")
            lo = vec(X, lambda i: -nlo[i], "lo")
        out = rsum(X, lambda i, j: te.log(X[i, j] - lo[i] + c(1.0)), "out")
        return [X], [out]

    return build


def log_from_min_ref(x):
    (x,) = d64(x)
    return (np.log(x - x.min(1, keepdims=True) + 1.0).sum(1),)


reg("29a", "log_from_min[te.min]", "Σ log(x - min x + 1), te.min", _log_from_min("min"), log_from_min_ref)
reg("29b", "log_from_min[-max(-x)]", "Σ log(x - min x + 1), min = -max(-x)", _log_from_min("negmax"), log_from_min_ref)


# ---------------------------------------------------------------------------
# numerical linear algebra / physics
# ---------------------------------------------------------------------------
def _scaled_norm(spell: str):
    def build():
        X = rows("X")
        if spell == "abs":
            scale = rmax(X, lambda i, j: te.abs(X[i, j]), "scale")
        else:
            scale = rmax(X, lambda i, j: tir.max(X[i, j], -X[i, j]), "scale")
        ssq = rsum(X, lambda i, j: (X[i, j] / scale[i]) * (X[i, j] / scale[i]), "ssq")
        out = vec(X, lambda i: scale[i] * te.sqrt(ssq[i]), "out")
        return [X], [out]

    return build


def scaled_norm_ref(x):
    (x,) = d64(x)
    scale = np.abs(x).max(1, keepdims=True)
    return (scale[:, 0] * np.sqrt(((x / scale) ** 2).sum(1)),)


reg("30a", "scaled_norm[te.abs]", "scale·sqrt(Σ (x/scale)²), scale = max |x| by te.abs", _scaled_norm("abs"), scaled_norm_ref)
reg("30b", "scaled_norm[max(x,-x)]", "the same with |x| = max(x, -x)", _scaled_norm("max"), scaled_norm_ref)


def gs_residual():
    Q = rows("Q")
    A = rows("A", like=Q)
    r = rsum(Q, lambda i, j: Q[i, j] * A[i, j], "r")
    out = rsum(Q, lambda i, j: (A[i, j] - r[i] * Q[i, j]) * (A[i, j] - r[i] * Q[i, j]), "out")
    return [Q, A], [out]


def gs_residual_ref(q, a):
    q, a = d64(q, a)
    r = (q * a).sum(1, keepdims=True)
    return (((a - r * q) ** 2).sum(1),)


reg("31", "gs_residual", "Σ (a - r q)², r = Σ q a", gs_residual, gs_residual_ref)


def gs_residual_normalised():
    Q = rows("Q")
    A = rows("A", like=Q)
    qa = rsum(Q, lambda i, j: Q[i, j] * A[i, j], "qa")
    qq = rsum(Q, lambda i, j: Q[i, j] * Q[i, j], "qq")
    r = vec(Q, lambda i: qa[i] / qq[i], "r")
    out = rsum(Q, lambda i, j: (A[i, j] - r[i] * Q[i, j]) * (A[i, j] - r[i] * Q[i, j]), "out")
    return [Q, A], [out]


def gs_residual_normalised_ref(q, a):
    q, a = d64(q, a)
    r = (q * a).sum(1, keepdims=True) / (q * q).sum(1, keepdims=True)
    return (((a - r * q) ** 2).sum(1),)


reg("32", "gs_residual_normalised", "Σ (a - r q)², r = Σ q a / Σ q²", gs_residual_normalised, gs_residual_normalised_ref)


def cg_residual():
    R = rows("R")
    P = rows("P", like=R)
    Q = rows("Q", like=R)
    rr = rsum(R, lambda i, j: R[i, j] * R[i, j], "rr")
    pq = rsum(R, lambda i, j: P[i, j] * Q[i, j], "pq")
    alpha = vec(R, lambda i: rr[i] / pq[i], "alpha")
    out = rsum(
        R, lambda i, j: (R[i, j] - alpha[i] * Q[i, j]) * (R[i, j] - alpha[i] * Q[i, j]), "out"
    )
    return [R, P, Q], [out]


def cg_residual_ref(r, p, q):
    r, p, q = d64(r, p, q)
    alpha = (r * r).sum(1, keepdims=True) / (p * q).sum(1, keepdims=True)
    return (((r - alpha * q) ** 2).sum(1),)


reg("33", "cg_residual", "Σ (r - α q)², α = Σ r r / Σ p q", cg_residual, cg_residual_ref)


def householder_like():
    X = rows("X")
    E = rows("E", like=X)
    ssq = rsum(X, lambda i, j: X[i, j] * X[i, j], "ssq")
    nrm = vec(X, lambda i: te.sqrt(ssq[i]), "nrm")
    out = rsum(
        X, lambda i, j: (X[i, j] + nrm[i] * E[i, j]) * (X[i, j] + nrm[i] * E[i, j]), "out"
    )
    return [X, E], [out]


def householder_like_ref(x, e):
    x, e = d64(x, e)
    nrm = np.sqrt((x * x).sum(1, keepdims=True))
    return (((x + nrm * e) ** 2).sum(1),)


reg("34", "householder_like", "Σ (x + ‖x‖ e)²", householder_like, householder_like_ref)


def cosine_normalised():
    X = rows("X")
    Y = rows("Y", like=X)
    sxx = rsum(X, lambda i, j: X[i, j] * X[i, j], "sxx")
    syy = rsum(X, lambda i, j: Y[i, j] * Y[i, j], "syy")
    nx = vec(X, lambda i: te.sqrt(sxx[i]), "nx")
    ny = vec(X, lambda i: te.sqrt(syy[i]), "ny")
    out = rsum(X, lambda i, j: (X[i, j] / nx[i]) * (Y[i, j] / ny[i]), "out")
    return [X, Y], [out]


def cosine_normalised_ref(x, y):
    x, y = d64(x, y)
    nx = np.sqrt((x * x).sum(1, keepdims=True))
    ny = np.sqrt((y * y).sum(1, keepdims=True))
    return (((x / nx) * (y / ny)).sum(1),)


reg("35", "cosine_normalised", "Σ (x/‖x‖)(y/‖y‖)", cosine_normalised, cosine_normalised_ref)


def instance_norm_1d():
    nb, ch, ln = te.var("nb"), te.var("ch"), te.var("L")
    X = te.placeholder((nb, ch, ln), name="X", dtype=F)
    cnt = ln.astype(F)
    l1 = te.reduce_axis((0, ln), "l")
    total = te.compute((nb, ch), lambda b, q: te.sum(X[b, q, l1], axis=l1), name="total")
    mean = te.compute((nb, ch), lambda b, q: total[b, q] / cnt, name="mean")
    l2 = te.reduce_axis((0, ln), "l")
    ss = te.compute(
        (nb, ch),
        lambda b, q: te.sum((X[b, q, l2] - mean[b, q]) * (X[b, q, l2] - mean[b, q]), axis=l2),
        name="ss",
    )
    var = te.compute((nb, ch), lambda b, q: ss[b, q] / cnt, name="var")
    out = te.compute(
        (nb, ch, ln),
        lambda b, q, k: (X[b, q, k] - mean[b, q]) / te.sqrt(var[b, q] + c(LN_EPS)),
        name="out",
    )
    return [X], [out]


def instance_norm_1d_ref(x):
    (x,) = d64(x)
    mu = x.mean(2, keepdims=True)
    return ((x - mu) / np.sqrt(x.var(2, keepdims=True) + LN_EPS),)


reg("36", "instance_norm_1d", "(x - mean_L)/sqrt(var_L + eps) per (n, c)", instance_norm_1d, instance_norm_1d_ref)


def _gyration(joint: bool):
    def build():
        n, m, d3 = te.var("n"), te.var("m"), te.var("d3")
        M = te.placeholder((n, m), name="M", dtype=F)
        R = te.placeholder((n, m, d3), name="R", dtype=F)
        j1 = te.reduce_axis((0, m), "j")
        msum = te.compute((n,), lambda i: te.sum(M[i, j1], axis=j1), name="msum")
        j2 = te.reduce_axis((0, m), "j")
        mr = te.compute((n, d3), lambda i, q: te.sum(M[i, j2] * R[i, j2, q], axis=j2), name="mr")
        cm = te.compute((n, d3), lambda i, q: mr[i, q] / msum[i], name="cm")
        if not joint:
            j3 = te.reduce_axis((0, m), "j")
            rg2s = te.compute(
                (n, d3),
                lambda i, q: te.sum(
                    M[i, j3] * (R[i, j3, q] - cm[i, q]) * (R[i, j3, q] - cm[i, q]), axis=j3
                ),
                name="rg2s",
            )
            rg2 = te.compute((n, d3), lambda i, q: rg2s[i, q] / msum[i], name="rg2")
            d1 = te.reduce_axis((0, d3), "d")
            total = te.compute((n,), lambda i: te.sum(rg2[i, d1], axis=d1), name="total")
        else:
            j3 = te.reduce_axis((0, m), "j")
            d1 = te.reduce_axis((0, d3), "d")
            tot_s = te.compute(
                (n,),
                lambda i: te.sum(
                    M[i, j3] * (R[i, j3, d1] - cm[i, d1]) * (R[i, j3, d1] - cm[i, d1]),
                    axis=[j3, d1],
                ),
                name="tot_s",
            )
            total = te.compute((n,), lambda i: tot_s[i] / msum[i], name="total")
        return [M, R], [total]

    return build


def gyration_ref(mass, r):
    mass, r = d64(mass, r)
    ms = mass.sum(1)
    cm = (mass[:, :, None] * r).sum(1) / ms[:, None]
    return ((mass[:, :, None] * (r - cm[:, None, :]) ** 2).sum((1, 2)) / ms,)


reg("37a", "radius_of_gyration[per-d]", "Σ_d (Σ_j M (R - cm_d)² / Σ_j M)", _gyration(False), gyration_ref, ("M",))
reg("37b", "radius_of_gyration[joint j,d]", "Σ_{j,d} M (R - cm_d)² / Σ_j M: one reduction over two axes", _gyration(True), gyration_ref, ("M",))


def batchnorm_stats():
    nb, ch, hh, ww = te.var("nb"), te.var("ch"), te.var("hh"), te.var("ww")
    X = te.placeholder((nb, ch, hh, ww), name="X", dtype=F)
    cnt = (nb * hh * ww).astype(F)
    b1, h1, w1 = te.reduce_axis((0, nb), "b"), te.reduce_axis((0, hh), "h"), te.reduce_axis((0, ww), "w")
    total = te.compute((ch,), lambda q: te.sum(X[b1, q, h1, w1], axis=[b1, h1, w1]), name="total")
    mean = te.compute((ch,), lambda q: total[q] / cnt, name="mean")
    b2, h2, w2 = te.reduce_axis((0, nb), "b"), te.reduce_axis((0, hh), "h"), te.reduce_axis((0, ww), "w")
    ss = te.compute(
        (ch,),
        lambda q: te.sum(
            (X[b2, q, h2, w2] - mean[q]) * (X[b2, q, h2, w2] - mean[q]), axis=[b2, h2, w2]
        ),
        name="ss",
    )
    var = te.compute((ch,), lambda q: ss[q] / cnt, name="var")
    return [X], [mean, var]


def batchnorm_stats_ref(x):
    (x,) = d64(x)
    return (x.mean((0, 2, 3)), x.var((0, 2, 3)))


reg("38", "batchnorm_stats", "mean[c], var[c] over (b, h, w): one reduction over three axes", batchnorm_stats, batchnorm_stats_ref)


# ---------------------------------------------------------------------------
# extra variants (NOT in the task list): the same functions with one step of the
# derivation done by hand, to locate which step the tool is missing
# ---------------------------------------------------------------------------
EXTRA = "extra variant, not in the list: "


def autocorr_lag1_const(m: int = 16):
    """The same with a constant extent, so that `m - 1` is a number and not an expression."""

    def build():
        n = te.var("n")
        X = te.placeholder((n, m), name="X", dtype=F)
        j1 = te.reduce_axis((0, m), "j")
        total = te.compute((n,), lambda i: te.sum(X[i, j1], axis=j1), name="total_mean")
        mean = te.compute((n,), lambda i: total[i] / c(m), name="mean")
        j2 = te.reduce_axis((0, m - 1), "j")
        out = te.compute(
            (n,),
            lambda i: te.sum((X[i, j2] - mean[i]) * (X[i, j2 + 1] - mean[i]), axis=j2),
            name="out",
        )
        return [X], [out]

    return build


reg("15b", "autocorr_lag1[const extent 16]", "the same with X of shape (n, 16): no extent expression", autocorr_lag1_const(), autocorr_lag1_ref, note=EXTRA + "constant extent, to get past the extent expression m - 1")


def zscore_max_outside():
    """max_j (x - μ) taken first, the division by σ outside the reduction."""
    X = rows("X")
    mean, sigma = _mean_sigma(X)
    mxc = rmax(X, lambda i, j: X[i, j] - mean[i], "mxc")
    out = vec(X, lambda i: mxc[i] / sigma[i], "out")
    return [X], [out]


reg("06b", "zscore_max[σ outside]", "max_j (x-μ) / σ: the scale taken out by hand", zscore_max_outside, zscore_max_ref, note=EXTRA + "1/σ > 0 moved out of the max by hand")


def max_abs_dev_two():
    """max(max_j (x - μ), max_j (μ - x)): the two branches as two reductions."""
    X = rows("X")
    mean = mean_of(X)
    up = rmax(X, lambda i, j: X[i, j] - mean[i], "up")
    down = rmax(X, lambda i, j: mean[i] - X[i, j], "down")
    out = vec(X, lambda i: tir.max(up[i], down[i]), "out")
    return [X], [out]


reg("07c", "max_abs_dev[two maxima]", "max(max_j (x-μ), max_j (μ-x)): two reductions", max_abs_dev_two, max_abs_dev_ref, note=EXTRA + "the max over the two branches split by hand")


def log_ratio_split():
    X = rows("X")
    mean = mean_of(X)
    out = rsum(X, lambda i, j: te.log(X[i, j]) - te.log(mean[i]), "out")
    return [X], [out]


reg("13b", "log_ratio[log x - log μ]", "Σ (log x - log μ): the log split by hand", log_ratio_split, log_ratio_ref, ("X",), note=EXTRA + "log(x/μ) written as log x - log μ")


def cv_squared_centred():
    X = rows("X")
    mean = mean_of(X)
    out = rsum(
        X, lambda i, j: ((X[i, j] - mean[i]) / mean[i]) * ((X[i, j] - mean[i]) / mean[i]), "out"
    )
    return [X], [out]


reg("14b", "cv_squared[((x-μ)/μ)²]", "Σ ((x-μ)/μ)²: centred by hand", cv_squared_centred, cv_squared_ref, ("X",), note=EXTRA + "x/μ - 1 written as (x-μ)/μ")


def cv_squared_outside():
    X = rows("X")
    mean = mean_of(X)
    ss = rsum(X, lambda i, j: (X[i, j] - mean[i]) * (X[i, j] - mean[i]), "ss")
    out = vec(X, lambda i: ss[i] / (mean[i] * mean[i]), "out")
    return [X], [out]


reg("14c", "cv_squared[Σ(x-μ)² / μ²]", "Σ (x-μ)² / μ²: the scale taken out by hand", cv_squared_outside, cv_squared_ref, ("X",), note=EXTRA + "1/μ² moved out of the sum by hand")


def hist_entropy_split():
    X = rows("X")
    s = rsum(X, lambda i, j: X[i, j], "s")
    H = rsum(X, lambda i, j: (X[i, j] / s[i]) * (te.log(X[i, j]) - te.log(s[i])), "plogp")
    out = vec(X, lambda i: -H[i], "out")
    return [X], [out]


reg("21b", "hist_entropy[log x - log s]", "-Σ (x/s)(log x - log s): the log split by hand", hist_entropy_split, hist_entropy_ref, ("X",), note=EXTRA + "log(x/s) written as log x - log s")


def scaled_norm_two():
    X = rows("X")
    hi = rmax(X, lambda i, j: X[i, j], "hi")
    nlo = rmax(X, lambda i, j: -X[i, j], "nlo")
    scale = vec(X, lambda i: tir.max(hi[i], nlo[i]), "scale")
    ssq = rsum(X, lambda i, j: (X[i, j] / scale[i]) * (X[i, j] / scale[i]), "ssq")
    out = vec(X, lambda i: scale[i] * te.sqrt(ssq[i]), "out")
    return [X], [out]


reg("30c", "scaled_norm[two maxima]", "scale = max(max x, max -x) as two reductions", scaled_norm_two, scaled_norm_ref, note=EXTRA + "max_j |x| split into two max reductions by hand")
