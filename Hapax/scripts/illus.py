"""The numbers of the paper's worked examples, computed rather than typed.

* variance of x = [1, 3 | 2, 6]: two passes, a naive per-part single pass, and
  the derived merge over (mean, ss, count), in exact arithmetic;
* the same data shifted by 1e4 in strict float32 (every operation rounded to
  float32): two passes, the derived merge, and the textbook sum x^2 - (sum x)^2 / n
  (the judgement's tuple of power sums);
* softmax of x = [1, 3 | 2, 6]: the derived online-normaliser merge, and the
  judgement's tuple (max x, sum e^x) against the program's own (mx, den) on
  x = [100, 200] in float32;
* RedFuser Appendix A.1 on u = (1, -1 | 3, 0), v = (1, 1 | 1, 0): the truth, the
  unrepaired merge (Eq. 11) in float32, and the repaired one (Eq. 27-28) in exact
  arithmetic.

    Hapax/scripts/py Hapax/scripts/illus.py Hapax/data/illus.json
"""

import sys
from fractions import Fraction as Q

import common
import numpy as np

f32 = np.float32


def variance_exact():
    a, b = [Q(1), Q(3)], [Q(2), Q(6)]
    x = a + b
    mu = sum(x) / len(x)
    two_pass = sum((v - mu) ** 2 for v in x)

    def part(p):
        m = sum(p) / len(p)
        return m, sum((v - m) ** 2 for v in p), Q(len(p))

    (ma, sa, na), (mb, sb, nb) = part(a), part(b)
    naive = sa + sb
    n = na + nb
    m = (ma * na + mb * nb) / n
    merged = sa + sb + na * (m - ma) ** 2 + nb * (m - mb) ** 2
    return {
        "x": [1, 3, 2, 6],
        "mean": float(mu),
        "two_pass": float(two_pass),
        "part_a": {"mean": float(ma), "ss": float(sa), "count": float(na)},
        "part_b": {"mean": float(mb), "ss": float(sb), "count": float(nb)},
        "naive": float(naive),
        "merged": {"mean": float(m), "ss": float(merged), "count": float(n)},
    }


def variance_float32(offset=1.0e4):
    x = [f32(v + offset) for v in (1.0, 3.0, 2.0, 6.0)]
    n = f32(len(x))
    # two passes
    tot = f32(0)
    for v in x:
        tot = f32(tot + v)
    mu = f32(tot / n)
    ss = f32(0)
    for v in x:
        d = f32(v - mu)
        ss = f32(ss + f32(d * d))
    # derived merge, parts [x0, x1] and [x2, x3], each leaf (x, 0, 1) merged in order
    def merge(s, t):
        cnt = f32(s[2] + t[2])
        m = f32(f32(f32(s[0] * s[2]) + f32(t[0] * t[2])) / cnt)
        da, db = f32(m - s[0]), f32(m - t[0])
        sq = f32(f32(s[1] + t[1]) + f32(f32(da * da) * s[2]) + f32(f32(db * db) * t[2]))
        return (m, sq, cnt)

    leaves = [(v, f32(0), f32(1)) for v in x]
    pa = merge(leaves[0], leaves[1])
    pb = merge(leaves[2], leaves[3])
    whole = merge(pa, pb)
    # the judgement's tuple: sum x, sum x^2, count, then sum x^2 - (sum x)^2 / n
    s1, s2 = f32(0), f32(0)
    for v in x:
        s1 = f32(s1 + v)
        s2 = f32(s2 + f32(v * v))
    textbook = f32(s2 - f32(f32(s1 * s1) / n))
    return {
        "x": [float(v) for v in x],
        "two_pass_ss": float(ss),
        "merged_ss": float(whole[1]),
        "merged_mean": float(whole[0]),
        "power_sums": {"sum": float(s1), "sum_sq": float(s2)},
        "textbook_ss": float(textbook),
        "exact_ss": 14.0,
    }


def softmax_merge():
    a, b = np.array([1.0, 3.0]), np.array([2.0, 6.0])

    def part(p):
        m = float(p.max())
        return m, float(np.exp(p - m).sum())

    (ma, da), (mb, db) = part(a), part(b)
    m = max(ma, mb)
    d = da * np.exp(ma - m) + db * np.exp(mb - m)
    whole = np.concatenate([a, b])
    return {
        "part_a": {"mx": ma, "den": da},
        "part_b": {"mx": mb, "den": db},
        "merged": {"mx": m, "den": float(d)},
        "two_pass": {"mx": float(whole.max()), "den": float(np.exp(whole - whole.max()).sum())},
    }


def softmax_tuples_float32():
    x = np.array([100.0, 200.0], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        judged_sum = np.exp(x).sum(dtype=np.float32)  # sum e^x: the judgement's tuple
        mx = x.max()
        den = np.exp(x - mx).sum(dtype=np.float32)  # the program's own state
        p_judged = np.float32(np.exp(x[1]) / judged_sum)
        p_program = np.float32(np.exp(x[1] - mx) / den)
    return {
        "x": x.tolist(),
        "judged_sum_exp": float(judged_sum),
        "program_mx": float(mx),
        "program_den": float(den),
        "p_last_judged": float(p_judged),
        "p_last_program": float(p_program),
    }


def redfuser_a1():
    """s = sum u, y = sum v s; F = v s = G(v) * H(s), H(s) = s, (x) = *."""
    ua, ub = [Q(1), Q(-1)], [Q(3), Q(0)]
    va, vb = [Q(1), Q(1)], [Q(1), Q(0)]
    s = sum(ua + ub)
    truth = sum(va + vb) * s
    sa, sb = sum(ua), sum(ub)  # partial contexts
    ya, yb = sum(va) * sa, sum(vb) * sb  # leaves, Eq. 6-7
    # Eq. 11: y = ya * H(sa)^-1 * H(s) + yb * H(sb)^-1 * H(s); sa = 0 has no inverse.
    with np.errstate(all="ignore"):
        f = np.float32
        unrepaired = f(f(f(ya) * (f(1) / f(sa))) * f(s)) + f(f(f(yb) * (f(1) / f(sb))) * f(s))
    # Eq. 27-28: H'(sa) = e = 1 when H(sa) is not invertible
    hinv_a = Q(1) if sa == 0 else 1 / sa
    repaired = ya * hinv_a * s + yb * (1 / sb) * s
    return {
        "u": [1, -1, 3, 0],
        "v": [1, 1, 1, 0],
        "s": float(s),
        "truth": float(truth),
        "partials": {"s_a": float(sa), "s_b": float(sb), "y_a": float(ya), "y_b": float(yb)},
        "eq11_float32": float(unrepaired),
        "a1_repaired": float(repaired),
    }


def redfuser_a1_incremental():
    """Eq. 16 step by step on the same input in one segment, with the Appendix A.1
    substitution (identity 1 for a missing inverse); exact fractions."""
    u, v = [Q(1), Q(-1), Q(3), Q(0)], [Q(1), Q(1), Q(1), Q(0)]
    s_prev, y = Q(0), Q(0)  # s_hat[0] is the identity of the sum
    steps = []
    for k in range(4):
        s = s_prev + u[k]
        inv = Q(1) if s_prev == 0 else 1 / s_prev  # A.1: H(s_prev) = 0 has no inverse
        y = y * inv * s + v[k] * s
        truth = sum(v[: k + 1]) * s
        steps.append(
            {"L": k + 1, "s": float(s), "y": float(y), "truth": float(truth), "inverse_missing": s_prev == 0}
        )
        s_prev = s
    return steps


if __name__ == "__main__":
    out = {
        "redfuser_a1_incremental": redfuser_a1_incremental(),
        "meta": common.meta("illus.py"),
        "variance_exact": variance_exact(),
        "variance_float32": variance_float32(),
        "softmax_merge": softmax_merge(),
        "softmax_tuples_float32": softmax_tuples_float32(),
        "redfuser_a1": redfuser_a1(),
    }
    common.dump(sys.argv[1], out)
    import json

    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))
