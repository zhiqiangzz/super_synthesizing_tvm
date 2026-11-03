X = te.placeholder((n, m), name="X")
j = te.reduce_axis((0, m), "j")
total_mean = te.compute((n,), lambda i: te.sum(X[i, j], axis=j), name="total_mean")
mean = te.compute((n,), lambda i: total_mean[i] / m.astype("float32"), name="mean")
j = te.reduce_axis((0, m), "j")
ss = te.compute(
    (n,),
    lambda i: te.sum(tir.power(X[i, j] - mean[i], 2.0), axis=j),
    name="ss",
)
var = te.compute((n,), lambda i: ss[i] / m.astype("float32"), name="var")
