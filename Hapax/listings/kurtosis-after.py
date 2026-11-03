# chain over j (total_mean, ss, s4): fused into 5 states (total_mean, ss, s4.h0, ss.r0, s4.h0.r0); derived: s4.h0, ss.r0, s4.h0.r0; selection: rebase
#   s4.h0 = 6*sum_j X[i, j]^2*mean[i]^2 + sum_j X[i, j]^4 + sum_j mean[i]^4 - 4*sum_j mean[i]*X[i, j]^3 - 4*sum_j X[i, j]*mean[i]^3    (left when the contexts of s4 are taken out of its reduction)
#   ss.r0 = sum_j 1    (left behind when ss is re-based)
#   s4.h0.r0 = 3*sum_j X[i, j]*mean[i]^2 + sum_j X[i, j]^3 - 3*sum_j mean[i]*X[i, j]^2 - sum_j mean[i]^3    (left behind when s4.h0 is re-based)
# Merge operands must represent non-empty parts; execution skips empty partitions.
def merge_j(a, b):
    t0 = (a[0] + b[0])
    t1 = (a[3] + b[3])
    t2 = (t0 / t1)
    t3 = (t2 - (a[0] / a[3]))
    t4 = (t3 * t3)
    t5 = (t2 - (b[0] / b[3]))
    t6 = (t5 * t5)
    return (
        t0,  # total_mean
        (a[1] + b[1] + (t4 * a[3]) + (t6 * b[3])),  # ss
        (a[2] + b[2] + tir.const(6.0, "float32") * (t4 * a[1]) + tir.const(6.0, "float32") * (t6 * b[1]) + ((t3 * t3 * t3 * t3) * a[3]) + ((t5 * t5 * t5 * t5) * b[3]) - tir.const(4.0, "float32") * (t3 * a[4]) - tir.const(4.0, "float32") * (t5 * b[4])),  # s4_h0
        t1,  # ss_r0
        (a[4] + b[4] - tir.const(3.0, "float32") * (t3 * a[1]) - tir.const(3.0, "float32") * (t5 * b[1]) - ((t3 * t3 * t3) * a[3]) - ((t5 * t5 * t5) * b[3])),  # s4_h0_r0
    )

j = te.reduce_axis((1, m), name="j")

def compute_j(i):
    # TE's initialization hook holds the first leaf, not an empty state.
    def identity_j(t0, t1, t2, t3, t4):
        return (
            X[i, 0],  # total_mean
            tir.const(0.0, "float32"),  # ss
            tir.const(0.0, "float32"),  # s4_h0
            tir.const(1.0, "float32"),  # ss_r0
            tir.const(0.0, "float32"),  # s4_h0_r0
        )
    fused_j = te.comm_reducer(merge_j, identity_j, name="fused_j")
    return fused_j(
        (
            X[i, j],  # total_mean
            tir.const(0.0, "float32"),  # ss
            tir.const(0.0, "float32"),  # s4_h0
            tir.const(1.0, "float32"),  # ss_r0
            tir.const(0.0, "float32"),  # s4_h0_r0
        ),
        axis=j,
    )

total_mean, ss, s4_h0, ss_r0, s4_h0_r0 = te.compute(
    (n,), compute_j,
    name="fused_total_mean_ss_s4",
    tag="superopt.nonempty",
)
total_mean, ss, s4_h0, ss_r0, s4_h0_r0 = te.compute(
    (n,),
    lambda i: (
        tir.if_then_else(m == 1, X[i, 0], total_mean[i]),
        tir.if_then_else(m == 1, tir.const(0.0, "float32"), ss[i]),
        tir.if_then_else(m == 1, tir.const(0.0, "float32"), s4_h0[i]),
        tir.if_then_else(m == 1, tir.const(1.0, "float32"), ss_r0[i]),
        tir.if_then_else(m == 1, tir.const(0.0, "float32"), s4_h0_r0[i]),
    ),
    name="nonempty_fused_total_mean_ss_s4", tag="superopt.singleton",
)
s4 = te.compute((n,), lambda i: (s4_h0[i] * (m.astype("float32") * m.astype("float32")) / (ss[i] * ss[i])), name="s4")
