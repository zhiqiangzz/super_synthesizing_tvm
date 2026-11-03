"""The programs Neptune (PLDI'26) and RedFuser (ASPLOS'26) fuse, written in TE.

Each entry records where in the paper the program comes from. Two spellings are
given where it matters:

* ``as written``: the program as the paper writes it (Neptune Fig. 12 builds every
  attention variant with ``if_then_else`` masks and a ``score_mod`` that may use
  the loop variables, tanh, ...);
* ``as data``: the same computation with the position-dependent part supplied as a
  tensor (a bias or mask tensor), which is how the registry's ``attention_bias``
  is written.

Nothing here is part of the tool or its registry; the programs exist only to
check, program by program, what Hapax does with the prior papers' own workloads.
Every program is float32 with symbolic extents, like the registry.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from tvm import te
from tvm_tirx import tirx as tir

F = "float32"
NEG_INF = tir.const(float("-inf"), F)


@dataclasses.dataclass(frozen=True)
class Prior:
    key: str
    paper: str  # "Neptune" or "RedFuser"
    where: str  # page / equation / figure in the paper
    what: str  # one line
    spelling: str  # "as written" or "as data"
    build: Callable
    positive: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# attention, after Neptune Fig. 12 (create_general_attention, p.26)
# ---------------------------------------------------------------------------
def _qkv(groups: int | None = None):
    b, h, qs, ks, d = (te.var(n) for n in ("B", "N", "QS", "KVS", "H"))
    q = te.placeholder((b, h, qs, d), name="q", dtype=F)
    k = te.placeholder((b, h, ks, d), name="k", dtype=F)
    v = te.placeholder((b, h, ks, d), name="v", dtype=F)
    return q, k, v


def _matmul_qk(q, k):
    b, h, qs, d = q.shape
    ks = k.shape[2]
    r = te.reduce_axis((0, d), "r")
    return te.compute(
        (b, h, qs, ks), lambda bb, n, i, j: te.sum(q[bb, n, i, r] * k[bb, n, j, r], axis=r), name="p"
    )


def _softmax_v(score, v):
    """Fig. 12 from ``softmax_maxelem`` on (the fp16 casts are dropped: one dtype)."""
    b, h, qs, ks = score.shape
    d = v.shape[3]
    j1 = te.reduce_axis((0, ks), "j")
    s_max = te.compute(
        (b, h, qs), lambda bb, n, i: te.max(score[bb, n, i, j1], axis=j1), name="softmax_maxelem"
    )
    s_exp = te.compute(
        (b, h, qs, ks),
        lambda bb, n, i, j: tir.exp(score[bb, n, i, j] - s_max[bb, n, i]),
        name="softmax_exp",
    )
    j2 = te.reduce_axis((0, ks), "j")
    s_expsum = te.compute(
        (b, h, qs), lambda bb, n, i: te.sum(s_exp[bb, n, i, j2], axis=j2), name="softmax_expsum"
    )
    j3 = te.reduce_axis((0, ks), "j")
    sv = te.compute(
        (b, h, qs, d),
        lambda bb, n, i, e: te.sum(s_exp[bb, n, i, j3] * v[bb, n, j3, e], axis=j3),
        name="sv",
    )
    out = te.compute(
        (b, h, qs, d), lambda bb, n, i, e: sv[bb, n, i, e] / s_expsum[bb, n, i], name="softmax_norm"
    )
    return out


def _attention(score_fn, extra=(), qkv=None):
    q, k, v = qkv if qkv is not None else _qkv()
    p = _matmul_qk(q, k)
    score = te.compute(p.shape, lambda *ax: score_fn(p, *ax), name="score_mod")
    return [q, k, v, *extra], [_softmax_v(score, v)]


def global_attention():
    return _attention(lambda p, *ax: p(*ax))


def causal_as_written():
    return _attention(
        lambda p, bb, n, i, j: tir.if_then_else(j <= i, p[bb, n, i, j], NEG_INF)
    )


def causal_as_data():
    q, k, v = _qkv()
    mask = te.placeholder((q.shape[2], k.shape[2]), name="mask", dtype=F)  # 0 or a large negative
    p = _matmul_qk(q, k)
    score = te.compute(p.shape, lambda bb, n, i, j: p[bb, n, i, j] + mask[i, j], name="score_mod")
    return [q, k, v, mask], [_softmax_v(score, v)]


def alibi_as_written():
    q, k, v = _qkv()
    slope = te.placeholder((q.shape[1],), name="slope", dtype=F)
    p = _matmul_qk(q, k)
    score = te.compute(
        p.shape,
        lambda bb, n, i, j: p[bb, n, i, j] - slope[n] * (i - j).astype(F),
        name="score_mod",
    )
    return [q, k, v, slope], [_softmax_v(score, v)]


def alibi_as_data():
    q, k, v = _qkv()
    bias = te.placeholder((q.shape[1], q.shape[2], k.shape[2]), name="alibi", dtype=F)
    p = _matmul_qk(q, k)
    score = te.compute(p.shape, lambda bb, n, i, j: p[bb, n, i, j] + bias[n, i, j], name="score_mod")
    return [q, k, v, bias], [_softmax_v(score, v)]


def softcap_as_written():
    cap = tir.const(50.0, F)
    return _attention(lambda p, *ax: cap * tir.tanh(p(*ax) / cap))


def softcap_as_data():
    """The capped scores supplied as an input (tanh evaluated before the chain)."""
    q, k, v = _qkv()
    capped = te.placeholder((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), name="capped", dtype=F)
    return [q, k, v, capped], [_softmax_v(capped, v)]


def window_as_written():
    w = 64
    return _attention(
        lambda p, bb, n, i, j: tir.if_then_else(
            tir.all(j <= i, i - j < w), p[bb, n, i, j], NEG_INF
        )
    )


def gqa_as_written():
    """Grouped-query attention: query head n reads key/value head n // G."""
    b, h, qs, ks, d = (te.var(n) for n in ("B", "N", "QS", "KVS", "H"))
    g = 4
    q = te.placeholder((b, h, qs, d), name="q", dtype=F)
    k = te.placeholder((b, h // g, ks, d), name="k", dtype=F)
    v = te.placeholder((b, h // g, ks, d), name="v", dtype=F)
    r = te.reduce_axis((0, d), "r")
    p = te.compute(
        (b, h, qs, ks),
        lambda bb, n, i, j: te.sum(q[bb, n, i, r] * k[bb, n // g, j, r], axis=r),
        name="p",
    )
    j1 = te.reduce_axis((0, ks), "j")
    s_max = te.compute((b, h, qs), lambda bb, n, i: te.max(p[bb, n, i, j1], axis=j1), name="smax")
    j2 = te.reduce_axis((0, ks), "j")
    s_sum = te.compute(
        (b, h, qs), lambda bb, n, i: te.sum(tir.exp(p[bb, n, i, j2] - s_max[bb, n, i]), axis=j2),
        name="ssum",
    )
    j3 = te.reduce_axis((0, ks), "j")
    sv = te.compute(
        (b, h, qs, d),
        lambda bb, n, i, e: te.sum(
            tir.exp(p[bb, n, i, j3] - s_max[bb, n, i]) * v[bb, n // g, j3, e], axis=j3
        ),
        name="sv",
    )
    out = te.compute((b, h, qs, d), lambda bb, n, i, e: sv[bb, n, i, e] / s_sum[bb, n, i], name="o")
    return [q, k, v], [out]


def gqa_grouped():
    """The same computation with the query heads laid out as (kv head, group)."""
    b, kh, g, qs, ks, d = (te.var(n) for n in ("B", "KH", "G", "QS", "KVS", "H"))
    q = te.placeholder((b, kh, g, qs, d), name="q", dtype=F)
    k = te.placeholder((b, kh, ks, d), name="k", dtype=F)
    v = te.placeholder((b, kh, ks, d), name="v", dtype=F)
    r = te.reduce_axis((0, d), "r")
    p = te.compute(
        (b, kh, g, qs, ks),
        lambda bb, n, gg, i, j: te.sum(q[bb, n, gg, i, r] * k[bb, n, j, r], axis=r),
        name="p",
    )
    j1 = te.reduce_axis((0, ks), "j")
    s_max = te.compute(
        (b, kh, g, qs), lambda bb, n, gg, i: te.max(p[bb, n, gg, i, j1], axis=j1), name="smax"
    )
    e_p = te.compute(
        p.shape, lambda bb, n, gg, i, j: tir.exp(p[bb, n, gg, i, j] - s_max[bb, n, gg, i]), name="e"
    )
    j2 = te.reduce_axis((0, ks), "j")
    s_sum = te.compute(
        (b, kh, g, qs), lambda bb, n, gg, i: te.sum(e_p[bb, n, gg, i, j2], axis=j2), name="ssum"
    )
    j3 = te.reduce_axis((0, ks), "j")
    sv = te.compute(
        (b, kh, g, qs, d),
        lambda bb, n, gg, i, c: te.sum(e_p[bb, n, gg, i, j3] * v[bb, n, j3, c], axis=j3),
        name="sv",
    )
    out = te.compute(
        sv.shape, lambda bb, n, gg, i, c: sv[bb, n, gg, i, c] / s_sum[bb, n, gg, i], name="o"
    )
    return [q, k, v], [out]


# ---------------------------------------------------------------------------
# Neptune, other operators (Section 4.4, p.13; Appendix C, p.29)
# ---------------------------------------------------------------------------
def _rows(name="x"):
    return te.placeholder((te.var("n"), te.var("m")), name=name, dtype=F)


def _rows2(a="x1", b="x2"):
    """Two inputs of one shape (one pair of extents)."""
    n, m = te.var("n"), te.var("m")
    return (te.placeholder((n, m), name=a, dtype=F), te.placeholder((n, m), name=b, dtype=F))


def stable_l2_norm():
    """||x|| = m sqrt(sum (x/m)^2), m = max |x| (p.13)."""
    x = _rows()
    n, m = x.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(tir.abs(x[i, j1]), axis=j1), name="mx")
    ss = te.compute(
        (n,), lambda i: te.sum((x[i, j2] / mx[i]) * (x[i, j2] / mx[i]), axis=j2), name="ss"
    )
    return [x], [te.compute((n,), lambda i: mx[i] * tir.sqrt(ss[i]), name="nrm")]


def rmsnorm_max():
    """y = x / sqrt(s + eps), s = mean(x^2), then the row max of y (p.13)."""
    x = _rows()
    n, m = x.shape
    eps = tir.const(1e-6, F)
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    ssq = te.compute((n,), lambda i: te.sum(x[i, j1] * x[i, j1], axis=j1), name="ssq")
    s = te.compute((n,), lambda i: ssq[i] / m.astype(F), name="s")
    y = te.compute((n, m), lambda i, j: x[i, j] / tir.sqrt(s[i] + eps), name="y")
    out = te.compute((n,), lambda i: te.max(y[i, j2], axis=j2), name="ymax")
    return [x], [out, y]


ALPHA, BETA = 0.125, 0.5


def _performer_inputs(k: int):
    n, m = te.var("n"), te.var("m")
    return [te.placeholder((n, m), name=f"c{i + 1}", dtype=F) for i in range(k)]


def performer_g1():
    """t = sum alpha exp(c1 - r1 - beta c2), r1 = max c1 (pattern g1, App. C p.29)."""
    c1, c2 = _performer_inputs(2)
    n, m = c1.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    r1 = te.compute((n,), lambda i: te.max(c1[i, j1], axis=j1), name="r1")
    t = te.compute(
        (n,),
        lambda i: te.sum(ALPHA * tir.exp(c1[i, j2] - r1[i] - BETA * c2[i, j2]), axis=j2),
        name="t",
    )
    return [c1, c2], [t]


def performer_g2():
    """t = sum alpha c3 exp(c1 - r1 - beta c2) (pattern g2)."""
    c1, c2, c3 = _performer_inputs(3)
    n, m = c1.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    r1 = te.compute((n,), lambda i: te.max(c1[i, j1], axis=j1), name="r1")
    t = te.compute(
        (n,),
        lambda i: te.sum(
            ALPHA * c3[i, j2] * tir.exp(c1[i, j2] - r1[i] - BETA * c2[i, j2]), axis=j2
        ),
        name="t",
    )
    return [c1, c2, c3], [t]


def performer_g3():
    """t = sum alpha exp(c1 - r1 - beta c2 c3) / r2, two producers (pattern g3)."""
    c1, c2, c3 = _performer_inputs(3)
    n, m = c1.shape
    j1, j2, j3 = (te.reduce_axis((0, m), "j") for _ in range(3))
    r1 = te.compute((n,), lambda i: te.max(c1[i, j1], axis=j1), name="r1")
    r2 = te.compute((n,), lambda i: te.sum(tir.exp(c1[i, j2] - r1[i]), axis=j2), name="r2")
    t = te.compute(
        (n,),
        lambda i: te.sum(
            ALPHA * tir.exp(c1[i, j3] - r1[i] - BETA * c2[i, j3] * c3[i, j3]) / r2[i], axis=j3
        ),
        name="t",
    )
    return [c1, c2, c3], [t]


# ---------------------------------------------------------------------------
# RedFuser (ASPLOS'26; arXiv 2603.10026)
# ---------------------------------------------------------------------------
def safe_softmax():
    """m = max x, t = sum exp(x - m), y = exp(x - m) / t (Section 2.2, p.3)."""
    x = _rows()
    n, m = x.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    mx = te.compute((n,), lambda i: te.max(x[i, j1], axis=j1), name="mx")
    t = te.compute((n,), lambda i: te.sum(tir.exp(x[i, j2] - mx[i]), axis=j2), name="t")
    y = te.compute((n, m), lambda i, j: tir.exp(x[i, j] - mx[i]) / t[i], name="y")
    return [x], [y]


def mha():
    """softmax(Q K^T / sqrt(d)) V (MHA workloads, Section 5.2, Table 2a)."""
    q, k, v = _qkv()
    d = q.shape[3]
    scale = tir.const(1.0, F) / tir.sqrt(d.astype(F))
    return _attention(lambda p, *ax: p(*ax) * scale, qkv=(q, k, v))


def mla_decode():
    """MLA decoding: scores = q_c . c + q_pe . k_pe, values = the latent c (A.5.1)."""
    b, h, qs, ks, dc, dp = (te.var(n) for n in ("bs", "hn", "q", "kv", "hd", "ped"))
    qc = te.placeholder((b, h, qs, dc), name="q_c", dtype=F)
    qpe = te.placeholder((b, h, qs, dp), name="q_pe", dtype=F)
    c = te.placeholder((b, ks, dc), name="c", dtype=F)
    kpe = te.placeholder((b, ks, dp), name="k_pe", dtype=F)
    r1, r2 = te.reduce_axis((0, dc), "r"), te.reduce_axis((0, dp), "s")
    s1 = te.compute(
        (b, h, qs, ks), lambda bb, n, i, j: te.sum(qc[bb, n, i, r1] * c[bb, j, r1], axis=r1),
        name="s_c",
    )
    s2 = te.compute(
        (b, h, qs, ks), lambda bb, n, i, j: te.sum(qpe[bb, n, i, r2] * kpe[bb, j, r2], axis=r2),
        name="s_pe",
    )
    score = te.compute(
        (b, h, qs, ks), lambda bb, n, i, j: s1[bb, n, i, j] + s2[bb, n, i, j], name="score"
    )
    j1, j2, j3 = (te.reduce_axis((0, ks), "j") for _ in range(3))
    mx = te.compute((b, h, qs), lambda bb, n, i: te.max(score[bb, n, i, j1], axis=j1), name="mx")
    den = te.compute(
        (b, h, qs), lambda bb, n, i: te.sum(tir.exp(score[bb, n, i, j2] - mx[bb, n, i]), axis=j2),
        name="den",
    )
    o = te.compute(
        (b, h, qs, dc),
        lambda bb, n, i, e: te.sum(
            tir.exp(score[bb, n, i, j3] - mx[bb, n, i]) / den[bb, n, i] * c[bb, j3, e], axis=j3
        ),
        name="o",
    )
    return [qc, qpe, c, kpe], [o]


def moe_routing_softmax():
    """MoE routing without TopK: logits = x W, then m and t of Eq. 35 (A.2.2).

    TopK has no counterpart among sum, max and min, so it is left out; the paper's
    final s/t needs the TopK.
    """
    s, hd, en = te.var("s"), te.var("hd"), te.var("en")
    x = te.placeholder((s, hd), name="x", dtype=F)
    w = te.placeholder((hd, en), name="w", dtype=F)
    r = te.reduce_axis((0, hd), "r")
    logits = te.compute((s, en), lambda i, e: te.sum(x[i, r] * w[r, e], axis=r), name="logits")
    j1, j2 = te.reduce_axis((0, en), "j"), te.reduce_axis((0, en), "j")
    mx = te.compute((s,), lambda i: te.max(logits[i, j1], axis=j1), name="m")
    t = te.compute((s,), lambda i: te.sum(tir.exp(logits[i, j2] - mx[i]), axis=j2), name="t")
    p = te.compute((s, en), lambda i, e: tir.exp(logits[i, e] - mx[i]) / t[i], name="p")
    return [x, w], [p]


def fp8_quant_gemm():
    """m = max |A[l]|, c = sum (MAX A[l] / m) W[l] (Eq. 17, p.6); rounding is not modelled."""
    mm, kk, nn = te.var("M"), te.var("K"), te.var("N")
    a = te.placeholder((mm, kk), name="A", dtype=F)
    w = te.placeholder((kk, nn), name="W", dtype=F)
    fmax = tir.const(448.0, F)  # the largest FP8 E4M3 value
    l1, l2 = te.reduce_axis((0, kk), "l"), te.reduce_axis((0, kk), "l")
    m = te.compute((mm,), lambda i: te.max(tir.abs(a[i, l1]), axis=l1), name="m")
    c = te.compute(
        (mm, nn), lambda i, n: te.sum(fmax * a[i, l2] / m[i] * w[l2, n], axis=l2), name="c"
    )
    return [a, w], [c]


def sum_sum():
    """m = sum x1^2, s = sum x1 x2 / sqrt(max(m - 10, eps)) (Eq. 39, A.2.3).

    The paper prints ``max(m - 10)`` without its second argument; eps = 1e-6 is our
    choice, the smallest change that makes the expression defined.
    """
    x1, x2 = _rows2("x1", "x2")
    n, m = x1.shape
    eps = tir.const(1e-6, F)
    ten = tir.const(10.0, F)
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    msum = te.compute((n,), lambda i: te.sum(x1[i, j1] * x1[i, j1], axis=j1), name="msum")
    s = te.compute(
        (n,),
        lambda i: te.sum(x1[i, j2] * x2[i, j2] / tir.sqrt(tir.max(msum[i] - ten, eps)), axis=j2),
        name="s",
    )
    return [x1, x2], [s]


def variance():
    """m = (1/L) sum x, var = (1/L) sum (x - m)^2 (Eq. 44, A.6)."""
    x = _rows()
    n, m = x.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    tot = te.compute((n,), lambda i: te.sum(x[i, j1], axis=j1), name="tot")
    mean = te.compute((n,), lambda i: tot[i] / m.astype(F), name="mean")
    ss = te.compute(
        (n,), lambda i: te.sum((x[i, j2] - mean[i]) * (x[i, j2] - mean[i]), axis=j2), name="ss"
    )
    return [x], [te.compute((n,), lambda i: ss[i] / m.astype(F), name="var")]


def inertia_per_coordinate():
    """M = sum m, c = (1/M) sum m x, I = sum m ||x - c||^2 (Eq. 45), coordinates spelled out."""
    bs, npart = te.var("bs"), te.var("n")
    mass = te.placeholder((bs, npart), name="mass", dtype=F)
    xs = [te.placeholder((bs, npart), name=f"x{d}", dtype=F) for d in range(3)]
    j0 = te.reduce_axis((0, npart), "l")
    big_m = te.compute((bs,), lambda i: te.sum(mass[i, j0], axis=j0), name="M")
    def centre(d, x):
        jd = te.reduce_axis((0, npart), "l")
        sd = te.compute((bs,), lambda i: te.sum(mass[i, jd] * x[i, jd], axis=jd), name=f"s{d}")
        return te.compute((bs,), lambda i: sd[i] / big_m[i], name=f"c{d}")

    cs = [centre(d, x) for d, x in enumerate(xs)]
    jl = te.reduce_axis((0, npart), "l")

    def dist2(i, l):
        out = None
        for x, c in zip(xs, cs):
            t = (x[i, l] - c[i]) * (x[i, l] - c[i])
            out = t if out is None else out + t
        return out

    inertia = te.compute((bs,), lambda i: te.sum(mass[i, jl] * dist2(i, jl), axis=jl), name="I")
    return [mass, *xs], [inertia]


def inertia_joint():
    """Eq. 45 with x as a (bs, n, dim) tensor: ||x - c||^2 is a reduction over dim."""
    bs, npart, dim = te.var("bs"), te.var("n"), te.var("dim")
    mass = te.placeholder((bs, npart), name="mass", dtype=F)
    x = te.placeholder((bs, npart, dim), name="x", dtype=F)
    l0 = te.reduce_axis((0, npart), "l")
    big_m = te.compute((bs,), lambda i: te.sum(mass[i, l0], axis=l0), name="M")
    l1 = te.reduce_axis((0, npart), "l")
    s = te.compute((bs, dim), lambda i, d: te.sum(mass[i, l1] * x[i, l1, d], axis=l1), name="s")
    c = te.compute((bs, dim), lambda i, d: s[i, d] / big_m[i], name="c")
    l2, d2 = te.reduce_axis((0, npart), "l"), te.reduce_axis((0, dim), "d")
    inertia = te.compute(
        (bs,),
        lambda i: te.sum(mass[i, l2] * (x[i, l2, d2] - c[i, d2]) * (x[i, l2, d2] - c[i, d2]), axis=[l2, d2]),
        name="I",
    )
    return [mass, x], [inertia]


def a1_program():
    """s = sum u, y = sum v s: the program of our counterexample to Appendix A.1."""
    u, v = _rows2("u", "v")
    n, m = u.shape
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    s = te.compute((n,), lambda i: te.sum(u[i, j1], axis=j1), name="s")
    y = te.compute((n,), lambda i: te.sum(v[i, j2] * s[i], axis=j2), name="y")
    return [u, v], [y]


PRIORS = [
    Prior("global", "Neptune", "Tab. 1 p.16; Fig. 12 p.26", "Global attention", "as written", global_attention),
    Prior("causal", "Neptune", "Tab. 1 p.16; Fig. 12 p.26", "Causal attention", "as written", causal_as_written),
    Prior("causal_data", "Neptune", "Tab. 1 p.16", "Causal attention, mask tensor", "as data", causal_as_data),
    Prior("alibi", "Neptune", "Tab. 1 p.16; Fig. 12 p.26", "ALiBi attention", "as written", alibi_as_written),
    Prior("alibi_data", "Neptune", "Tab. 1 p.16", "ALiBi attention, bias tensor", "as data", alibi_as_data),
    Prior("softcap", "Neptune", "Tab. 1 p.16; Fig. 12 p.26", "Soft-capped attention", "as written", softcap_as_written),
    Prior("softcap_data", "Neptune", "Tab. 1 p.16", "Soft-capped attention, capped scores as input", "as data", softcap_as_data),
    Prior("window", "Neptune", "Tab. 1 p.16; Fig. 12 p.26", "Sliding-window attention", "as written", window_as_written),
    Prior("gqa", "Neptune", "Tab. 1 p.16", "Grouped-query attention, head n // G", "as written", gqa_as_written),
    Prior("gqa_grouped", "Neptune", "Tab. 1 p.16", "Grouped-query attention, (kv head, group) layout", "as data", gqa_grouped),
    Prior("stable_l2", "Neptune", "Sec. 4.4 p.13", "Numerically stable L2 norm", "as written", stable_l2_norm),
    Prior("rmsnorm_max", "Neptune", "Sec. 4.4 p.13", "RMSNorm, then the row max", "as written", rmsnorm_max),
    Prior("performer_g1", "Neptune", "App. C p.29", "Performer reduction pattern g1", "as written", performer_g1),
    Prior("performer_g2", "Neptune", "App. C p.29", "Performer reduction pattern g2", "as written", performer_g2),
    Prior("performer_g3", "Neptune", "App. C p.29", "Performer reduction pattern g3", "as written", performer_g3),
    Prior("safe_softmax", "RedFuser", "Sec. 2.2 p.3", "Safe softmax", "as written", safe_softmax),
    Prior("mha", "RedFuser", "Sec. 5.2; Tab. 2a", "Multi-head attention", "as written", mha),
    Prior("mla", "RedFuser", "A.5.1; Tab. 2b", "Multi-latent attention, decoding", "as written", mla_decode),
    Prior("moe", "RedFuser", "A.2.2 Eq. 35", "MoE routing without TopK", "as written", moe_routing_softmax),
    Prior("fp8_quant_gemm", "RedFuser", "Sec. 3.4 Eq. 17 p.6", "FP8 per-token quantization + GEMM", "as written", fp8_quant_gemm),
    Prior("sum_sum", "RedFuser", "A.2.3 Eq. 39 p.16", "Sum + Sum", "as written", sum_sum),
    Prior("variance", "RedFuser", "A.6 Eq. 44 p.17", "Variance", "as written", variance),
    Prior("inertia", "RedFuser", "A.6 Eq. 45 p.17", "Moment of inertia, per coordinate", "as written", inertia_per_coordinate, positive=("mass",)),
    Prior("inertia_joint", "RedFuser", "A.6 Eq. 45 p.17", "Moment of inertia, joint (n, dim) reduction", "as written", inertia_joint, positive=("mass",)),
    Prior("a1_program", "RedFuser", "A.1 p.15 (our counterexample)", "s = sum u, y = sum v s", "as written", a1_program),
]
