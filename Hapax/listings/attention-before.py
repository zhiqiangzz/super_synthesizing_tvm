Q = te.placeholder((batch, heads, seq_q, dim), name="Q")
K = te.placeholder((batch, heads, seq_k, dim), name="K")
V = te.placeholder((batch, heads, seq_k, dim), name="V")
d = te.reduce_axis((0, dim), "d")
QK = te.compute(
    (batch, heads, seq_q, seq_k),
    lambda b, h, i, j: te.sum(Q[b, h, i, d] * K[b, h, j, d], axis=d),
    name="QK",
)
S = te.compute(
    (batch, heads, seq_q, seq_k),
    lambda b, h, i, j: QK[b, h, i, j] * (1.0 / tir.sqrt(dim.astype("float32"))),
    name="S",
)
j = te.reduce_axis((0, seq_k), "j")
row_max = te.compute(
    (batch, heads, seq_q),
    lambda b, h, i: te.max(S[b, h, i, j], axis=j),
    name="row_max",
)
exp_S = te.compute(
    (batch, heads, seq_q, seq_k),
    lambda b, h, i, j: tir.exp(S[b, h, i, j] - row_max[b, h, i]),
    name="exp_S",
)
j = te.reduce_axis((0, seq_k), "j")
den = te.compute(
    (batch, heads, seq_q),
    lambda b, h, i: te.sum(exp_S[b, h, i, j], axis=j),
    name="den",
)
P = te.compute(
    (batch, heads, seq_q, seq_k),
    lambda b, h, i, j: exp_S[b, h, i, j] / den[b, h, i],
    name="P",
)
j = te.reduce_axis((0, seq_k), "j")
O = te.compute(
    (batch, heads, seq_q, dim),
    lambda b, h, i, e: te.sum(P[b, h, i, j] * V[b, h, j, e], axis=j),
    name="O",
)
