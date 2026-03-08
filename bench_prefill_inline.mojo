from gemm import matmul_prefill
from matrix import Matrix
import std.benchmark


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn main() raises:
    print("=== prefill-only benchmark (96x11008x2048) ===\n")

    comptime M = 96
    comptime N = 11008
    comptime K = 2048

    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    @parameter
    fn bench_prefill():
        matmul_prefill(c, a, b)

    # Warm up
    matmul_prefill(c, a, b)

    var r = std.benchmark.run[bench_prefill]()
    var s_mean = r.mean("s")
    var s_min = r.min("s")
    print(
        "  prefill :",
        r.mean("ms"),
        "ms (mean) |",
        r.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_min),
        "GFLOPS (min)",
    )

    print("\n--- full report ---\n")
    r.print()
