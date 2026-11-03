# chain over j (total_mean, ss): fused into 3 states (total_mean, ss, ss.r0); derived: ss.r0; selection: rebase
#   ss.r0 = sum_j 1    (left behind when ss is re-based)
# Merge operands must represent non-empty parts; execution skips empty partitions.
def merge_j(a, b):
    t0 = (a[0] + b[0])
    t1 = (a[2] + b[2])
    t2 = (t0 / t1)
    return (
        t0,  # total_mean
        (a[1] + b[1] + (((t2 - (a[0] / a[2])) * (t2 - (a[0] / a[2]))) * a[2]) + (((t2 - (b[0] / b[2])) * (t2 - (b[0] / b[2]))) * b[2])),  # ss
        t1,  # ss_r0
    )

j = te.reduce_axis((1, m), name="j")

def compute_j(i):
    # TE's initialization hook holds the first leaf, not an empty state.
    def identity_j(t0, t1, t2):
        return (
            X[i, 0],  # total_mean
            tir.const(0.0, "float32"),  # ss
            tir.const(1.0, "float32"),  # ss_r0
        )
    fused_j = te.comm_reducer(merge_j, identity_j, name="fused_j")
    return fused_j(
        (
            X[i, j],  # total_mean
            tir.const(0.0, "float32"),  # ss
            tir.const(1.0, "float32"),  # ss_r0
        ),
        axis=j,
    )

total_mean, ss, ss_r0 = te.compute(
    (n,), compute_j,
    name="fused_total_mean_ss",
    tag="superopt.nonempty",
)
total_mean, ss, ss_r0 = te.compute(
    (n,),
    lambda i: (
        tir.if_then_else(m == 1, X[i, 0], total_mean[i]),
        tir.if_then_else(m == 1, tir.const(0.0, "float32"), ss[i]),
        tir.if_then_else(m == 1, tir.const(1.0, "float32"), ss_r0[i]),
    ),
    name="nonempty_fused_total_mean_ss", tag="superopt.singleton",
)
