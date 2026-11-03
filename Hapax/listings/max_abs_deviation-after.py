# chain over j (total, dmax): fused into 4 states (total, dmax, synth0, synth1); derived: synth0, synth1; selection: sygus
#   synth0 = max_j -X[i, j] - max_j X[i, j]    (synthesized by cvc5 SyGuS and independently verified)
#   synth1 = sum_j 1    (synthesized by cvc5 SyGuS and independently verified)
# Merge operands must represent non-empty parts; execution skips empty partitions.
def merge_j(a, b):
    t0 = (a[0] + b[0])
    t1 = (a[3] + b[3])
    t2 = (t0 / t1)
    t3 = (a[0] / a[3])
    t4 = (t2 - t3)
    t5 = (b[0] / b[3])
    t6 = (t2 - t5)
    return (
        t0,  # total
        tir.max(tir.max(tir.max(((tir.if_then_else(tir.max(tir.const(0.0, "float32"), (a[2] + tir.const(2.0, "float32") * t3)) == 0, (a[1] + t3), (a[1] - t3 - a[2])) - t3) - t4), ((tir.if_then_else(tir.max(tir.const(0.0, "float32"), (b[2] + tir.const(2.0, "float32") * t5)) == 0, (b[1] + t5), (b[1] - t5 - b[2])) - t5) - t6)), (t4 + (t3 + tir.if_then_else(tir.max(tir.const(0.0, "float32"), (a[2] + tir.const(2.0, "float32") * t3)) == 0, (a[1] + a[2] + t3), (a[1] - t3))))), (t6 + (t5 + tir.if_then_else(tir.max(tir.const(0.0, "float32"), (b[2] + tir.const(2.0, "float32") * t5)) == 0, (b[1] + b[2] + t5), (b[1] - t5))))),  # dmax
        (tir.max(tir.if_then_else(tir.max(tir.const(0.0, "float32"), (a[2] + tir.const(2.0, "float32") * t3)) == 0, (a[1] + a[2] + t3), (a[1] - t3)), tir.if_then_else(tir.max(tir.const(0.0, "float32"), (b[2] + tir.const(2.0, "float32") * t5)) == 0, (b[1] + b[2] + t5), (b[1] - t5))) - tir.max(tir.if_then_else(tir.max(tir.const(0.0, "float32"), (a[2] + tir.const(2.0, "float32") * t3)) == 0, (a[1] + t3), (a[1] - t3 - a[2])), tir.if_then_else(tir.max(tir.const(0.0, "float32"), (b[2] + tir.const(2.0, "float32") * t5)) == 0, (b[1] + t5), (b[1] - t5 - b[2])))),  # synth0
        t1,  # synth1
    )

j = te.reduce_axis((1, m), name="j")

def compute_j(i):
    # TE's initialization hook holds the first leaf, not an empty state.
    def identity_j(t0, t1, t2, t3):
        return (
            X[i, 0],  # total
            tir.const(0.0, "float32"),  # dmax
            (tir.const(-2.0, "float32") * X[i, 0]),  # synth0
            tir.const(1.0, "float32"),  # synth1
        )
    fused_j = te.comm_reducer(merge_j, identity_j, name="fused_j")
    return fused_j(
        (
            X[i, j],  # total
            tir.const(0.0, "float32"),  # dmax
            (tir.const(-2.0, "float32") * X[i, j]),  # synth0
            tir.const(1.0, "float32"),  # synth1
        ),
        axis=j,
    )

total, dmax, synth0, synth1 = te.compute(
    (n,), compute_j,
    name="fused_total_dmax",
    tag="superopt.nonempty",
)
total, dmax, synth0, synth1 = te.compute(
    (n,),
    lambda i: (
        tir.if_then_else(m == 1, X[i, 0], total[i]),
        tir.if_then_else(m == 1, tir.const(0.0, "float32"), dmax[i]),
        tir.if_then_else(m == 1, (tir.const(-2.0, "float32") * X[i, 0]), synth0[i]),
        tir.if_then_else(m == 1, tir.const(1.0, "float32"), synth1[i]),
    ),
    name="nonempty_fused_total_dmax", tag="superopt.singleton",
)
