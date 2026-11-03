# chain over j (row_max, den, O): fused into 3 states (row_max, den, O); selection: original
# Merge operands must represent non-empty parts; execution skips empty partitions.
def merge_j(a, b):
    t0 = tir.max(a[0], b[0])
    t1 = tir.exp((-(t0 - a[0])))
    t2 = tir.exp((-(t0 - b[0])))
    t3 = ((t1 * a[1]) + (t2 * b[1]))
    return (
        t0,  # row_max
        t3,  # den
        ((((a[1] / t3) * t1) * a[2]) + (((b[1] / t3) * t2) * b[2])),  # O
    )

j = te.reduce_axis((1, seq_k), name="j")

def compute_j(b, h, i, e):
    # TE's initialization hook holds the first leaf, not an empty state.
    def identity_j(t0, t1, t2):
        return (
            (QK[b, h, i, 0] / tir.sqrt(dim.astype("float32"))),  # row_max
            tir.const(1.0, "float32"),  # den
            V[b, h, 0, e],  # O
        )
    fused_j = te.comm_reducer(merge_j, identity_j, name="fused_j")
    return fused_j(
        (
            (QK[b, h, i, j] / tir.sqrt(dim.astype("float32"))),  # row_max
            tir.const(1.0, "float32"),  # den
            V[b, h, j, e],  # O
        ),
        axis=j,
    )

row_max, den, O = te.compute(
    (batch, heads, seq_q, dim), compute_j,
    name="fused_row_max_den_O",
    tag="superopt.nonempty",
)
row_max, den, O = te.compute(
    (batch, heads, seq_q, dim),
    lambda b, h, i, e: (
        tir.if_then_else(seq_k == 1, (QK[b, h, i, 0] / tir.sqrt(dim.astype("float32"))), row_max[b, h, i, e]),
        tir.if_then_else(seq_k == 1, tir.const(1.0, "float32"), den[b, h, i, e]),
        tir.if_then_else(seq_k == 1, V[b, h, 0, e], O[b, h, i, e]),
    ),
    name="nonempty_fused_row_max_den_O", tag="superopt.singleton",
)
