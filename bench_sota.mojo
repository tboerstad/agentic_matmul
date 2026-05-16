from gemm import matmul_dispatch
from matrix import Matrix
import std.benchmark
from std.time import perf_counter_ns


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn main() raises:
    var t_start = perf_counter_ns()
    print("=== matmul_dispatch SOTA (Qwen 2.5 VL 3B shapes, float64) ===")

    comptime M1 = 1
    comptime N1 = 11008
    comptime K1 = 2048
    var a1 = Matrix(M1, K1)
    var b1 = Matrix(K1, N1)
    var c1 = Matrix(M1, N1)
    fill(a1, 17)
    fill(b1, 13)

    @parameter
    fn d():
        matmul_dispatch(c1, a1, b1)

    var r = std.benchmark.run[d]()
    print("decode  (1x11008x2048):", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")

    comptime M2 = 96
    comptime N2 = 11008
    comptime K2 = 2048
    var a2 = Matrix(M2, K2)
    var b2 = Matrix(K2, N2)
    var c2 = Matrix(M2, N2)
    fill(a2, 17)
    fill(b2, 13)

    @parameter
    fn p():
        matmul_dispatch(c2, a2, b2)

    var r2 = std.benchmark.run[p]()
    print("prefill (96x11008x2048):", r2.mean("ms"), "ms |", gflops(M2, N2, K2, r2.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r2.min("s")), "GFLOPS (peak)")

    var t_end = perf_counter_ns()
    print("\nwall time:", Float64(t_end - t_start) / 1e9, "s")
