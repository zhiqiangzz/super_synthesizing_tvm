X = te.placeholder((n, m), name="X")
j = te.reduce_axis((0, m), "j")
total_mean = te.compute((n,), lambda i: te.sum(X[i, j], axis=j), name="total_mean")
mean = te.compute((n,), lambda i: total_mean[i] / m.astype("float32"), name="mean")
j = te.reduce_axis((0, m), "j")
ss = te.compute(
    (n,),
    lambda i: te.sum((X[i, j] - mean[i]) * (X[i, j] - mean[i]), axis=j),
    name="ss",
)
m2 = te.compute((n,), lambda i: ss[i] / m.astype("float32"), name="m2")
j = te.reduce_axis((0, m), "j")
s4 = te.compute(
    (n,),
    lambda i: te.sum(tir.power((X[i, j] - mean[i]) / tir.sqrt(m2[i]), 4.0), axis=j),
    name="s4",
)
kurt = te.compute((n,), lambda i: s4[i] / m.astype("float32"), name="kurt")
