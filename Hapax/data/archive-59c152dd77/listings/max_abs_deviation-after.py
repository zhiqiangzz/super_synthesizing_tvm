# chain over j (total, dmax): fused into 4 states (total, dmax.m0, dmax.m1, count); derived: dmax.m0, dmax.m1, count
#   dmax.m0 = max_j (mean[i] - X[i, j])    (one of the reductions dmax is the largest of)
#   dmax.m1 = min_j (mean[i] - X[i, j])    (one of the reductions dmax is the largest of)
#   count = sum_j 1    (the extent of the axis, as far as it has been reduced)
# Merge operands must represent non-empty parts; execution skips empty partitions.
def merge_j(a, b):
    t0 = (a[0] + b[0])
    t1 = (a[3] + b[3])
    t2 = (t0 / t1)
    t3 = (t2 - (a[0] / a[3]))
    t4 = (t2 - (b[0] / b[3]))
    return (
        t0,  # total
        tir.max((t3 + a[1]), (t4 + b[1])),  # dmax_m0
        (-(tir.max((tir.const(0.0, "float32") - t3 - a[2]), (tir.const(0.0, "float32") - t4 - b[2])))),  # dmax_m1
        t1,  # count
    )

j = te.reduce_axis((1, m), name="j")

def compute_j(i):
    # TE's initialization hook holds the first leaf, not an empty state.
    def identity_j(t0, t1, t2, t3):
        return (
            X[i, 0],  # total
            tir.const(0.0, "float32"),  # dmax_m0
            tir.const(0.0, "float32"),  # dmax_m1
            tir.const(1.0, "float32"),  # count
        )
    fused_j = te.comm_reducer(merge_j, identity_j, name="fused_j")
    return fused_j(
        (
            X[i, j],  # total
            tir.const(0.0, "float32"),  # dmax_m0
            tir.const(0.0, "float32"),  # dmax_m1
            tir.const(1.0, "float32"),  # count
        ),
        axis=j,
    )

total, dmax_m0, dmax_m1, count = te.compute(
    (n,), compute_j,
    name="fused_total_dmax",
    tag="superopt.nonempty",
)
total, dmax_m0, dmax_m1, count = te.compute(
    (n,),
    lambda i: (
        tir.if_then_else(m == 1, X[i, 0], total[i]),
        tir.if_then_else(m == 1, tir.const(0.0, "float32"), dmax_m0[i]),
        tir.if_then_else(m == 1, tir.const(0.0, "float32"), dmax_m1[i]),
        tir.if_then_else(m == 1, tir.const(1.0, "float32"), count[i]),
    ),
    name="nonempty_fused_total_dmax", tag="superopt.singleton",
)
dmax = te.compute((n,), lambda i: tir.max(dmax_m0[i], (-dmax_m1[i])), name="dmax")
