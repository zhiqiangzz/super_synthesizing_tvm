from __future__ import annotations

# from tvm.s_tir import script as Ts
# from tvm_tirx.script import tirx as T

@Ts.function
def main[n, m](X: T.Tensor((n, m), T.float32), var: T.Tensor((n,), T.float32)):
    T.func_attr({"script.namespace": "s_tir", "tirx.noalias": True})
    with Ts.sblock("root"):
        Ts.reads()
        Ts.writes()
        fused_total_mean_ss_v0 = Ts.sblock_alloc_tensor((n,), T.float32)
        fused_total_mean_ss_v1 = Ts.sblock_alloc_tensor((n,), T.float32)
        fused_total_mean_ss_v2 = Ts.sblock_alloc_tensor((n,), T.float32)
        nonempty_total_mean_ss_v0 = Ts.sblock_alloc_tensor((n,), T.float32)
        nonempty_total_mean_ss_v1 = Ts.sblock_alloc_tensor((n,), T.float32)
        nonempty_total_mean_ss_v2 = Ts.sblock_alloc_tensor((n,), T.float32)
        for i in range(n):
            for j in range(1, 1 + (m - 1)):
                with Ts.sblock("fused_total_mean_ss"):
                    v_i = Ts.axis.spatial(n, i)
                    v_j = Ts.axis.reduce(T.Range.from_min_extent(1, m - 1), j)
                    Ts.reads(X[v_i, T.min(0, v_j):T.min(0, v_j) + (T.max(0, v_j) + 1 - T.min(0, v_j))])
                    Ts.writes(fused_total_mean_ss_v0[v_i:v_i + 1], fused_total_mean_ss_v1[v_i:v_i + 1], fused_total_mean_ss_v2[v_i:v_i + 1])
                    with Ts.init():
                        fused_total_mean_ss_v0[v_i] = X[v_i, 0]
                        fused_total_mean_ss_v1[v_i] = T.float32(0.0)
                        fused_total_mean_ss_v2[v_i] = T.float32(1.0)
                    v_fused_total_mean_ss_v0: T.let[T.float32] = fused_total_mean_ss_v0[v_i] + X[v_i, v_j]
                    v_fused_total_mean_ss_v1: T.let[T.float32] = T.Add(fused_total_mean_ss_v1[v_i], T.float32(0.0)) + ((fused_total_mean_ss_v0[v_i] + X[v_i, v_j]) / (fused_total_mean_ss_v2[v_i] + T.float32(1.0)) - fused_total_mean_ss_v0[v_i] / fused_total_mean_ss_v2[v_i]) * ((fused_total_mean_ss_v0[v_i] + X[v_i, v_j]) / (fused_total_mean_ss_v2[v_i] + T.float32(1.0)) - fused_total_mean_ss_v0[v_i] / fused_total_mean_ss_v2[v_i]) * fused_total_mean_ss_v2[v_i] + T.Mul(((fused_total_mean_ss_v0[v_i] + X[v_i, v_j]) / (fused_total_mean_ss_v2[v_i] + T.float32(1.0)) - T.Div(X[v_i, v_j], T.float32(1.0))) * ((fused_total_mean_ss_v0[v_i] + X[v_i, v_j]) / (fused_total_mean_ss_v2[v_i] + T.float32(1.0)) - T.Div(X[v_i, v_j], T.float32(1.0))), T.float32(1.0))
                    v_fused_total_mean_ss_v2: T.let[T.float32] = fused_total_mean_ss_v2[v_i] + T.float32(1.0)
                    fused_total_mean_ss_v0[v_i] = v_fused_total_mean_ss_v0
                    fused_total_mean_ss_v1[v_i] = v_fused_total_mean_ss_v1
                    fused_total_mean_ss_v2[v_i] = v_fused_total_mean_ss_v2
        for i_1 in range(n):
            with Ts.sblock("nonempty_total_mean_ss_v0"):
                v_i_1 = Ts.axis.spatial(n, i_1)
                Ts.reads(X[v_i_1, 0:1], fused_total_mean_ss_v0[v_i_1:v_i_1 + 1])
                Ts.writes(nonempty_total_mean_ss_v0[v_i_1:v_i_1 + 1])
                nonempty_total_mean_ss_v0[v_i_1] = T.if_then_else(m == 1, X[v_i_1, 0], fused_total_mean_ss_v0[v_i_1])
            with Ts.sblock("nonempty_total_mean_ss_v1"):
                v_i_2 = Ts.axis.spatial(n, i_1)
                Ts.reads(fused_total_mean_ss_v1[v_i_2:v_i_2 + 1])
                Ts.writes(nonempty_total_mean_ss_v1[v_i_2:v_i_2 + 1])
                nonempty_total_mean_ss_v1[v_i_2] = T.if_then_else(m == 1, T.float32(0.0), fused_total_mean_ss_v1[v_i_2])
            with Ts.sblock("nonempty_total_mean_ss_v2"):
                v_i_3 = Ts.axis.spatial(n, i_1)
                Ts.reads(fused_total_mean_ss_v2[v_i_3:v_i_3 + 1])
                Ts.writes(nonempty_total_mean_ss_v2[v_i_3:v_i_3 + 1])
                nonempty_total_mean_ss_v2[v_i_3] = T.if_then_else(m == 1, T.float32(1.0), fused_total_mean_ss_v2[v_i_3])
        for i_2 in range(n):
            with Ts.sblock("var"):
                v_i_4 = Ts.axis.spatial(n, i_2)
                Ts.reads(nonempty_total_mean_ss_v1[v_i_4:v_i_4 + 1])
                Ts.writes(var[v_i_4:v_i_4 + 1])
                var[v_i_4] = nonempty_total_mean_ss_v1[v_i_4] / T.Cast("float32", m)
