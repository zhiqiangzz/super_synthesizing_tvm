# chain over j (mx, den): fused into 2 states (mx, den); selection: original
# Merge operands must represent non-empty parts; execution skips empty partitions.
def merge_j(a, b):
    t0 = tir.max(a[0], b[0])
    return (
        t0,  # mx
        (((tir.exp((-(t0 - a[0])))) * a[1]) + ((tir.exp((-(t0 - b[0])))) * b[1])),  # den
    )

j = te.reduce_axis((1, m), name="j")

def compute_j(i):
    # TE's initialization hook holds the first leaf, not an empty state.
    def identity_j(t0, t1):
        return (
            X[i, 0],  # mx
            tir.const(1.0, "float32"),  # den
        )
    fused_j = te.comm_reducer(merge_j, identity_j, name="fused_j")
    return fused_j(
        (
            X[i, j],  # mx
            tir.const(1.0, "float32"),  # den
        ),
        axis=j,
    )

mx, den = te.compute(
    (n,), compute_j,
    name="fused_mx_den",
    tag="superopt.nonempty",
)
mx, den = te.compute(
    (n,),
    lambda i: (
        tir.if_then_else(m == 1, X[i, 0], mx[i]),
        tir.if_then_else(m == 1, tir.const(1.0, "float32"), den[i]),
    ),
    name="nonempty_fused_mx_den", tag="superopt.singleton",
)
