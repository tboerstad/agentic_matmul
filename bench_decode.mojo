from decode import matmul_decode
from matrix import Matrix
import std.benchmark
from std.time import perf_counter_ns


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype](i * m.cols + j) * 0.1


fn main() raises:
    var t_start = perf_counter_ns()
    print("=== decode kernel benchmark (1x11008x2048) ===\n")

    comptime M = 1
    comptime N = 11008
    comptime K = 2048

    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    @parameter
    fn bench_decode():
        matmul_decode(c, a, b)

    var r = std.benchmark.run[bench_decode]()
    print("  decode :", r.mean("ms"), "ms |", gflops(M, N, K, r.mean("s")), "GFLOPS (mean) |", gflops(M, N, K, r.min("s")), "GFLOPS (peak)")

    var t_end = perf_counter_ns()
    print("\n=== wall time:", Float64(t_end - t_start) / 1e9, "s ===")
