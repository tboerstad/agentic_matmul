from gemm import matmul_dispatch
from matrix import Matrix
import std.benchmark
from std.collections import List

# Note: linalg.matmul is not available in this Mojo version
# from linalg.matmul import matmul as linalg_matmul


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn bench_decode() raises:
    """Benchmark single-token decode: 1x11008x2048."""
    comptime M = 1
    comptime N = 11008
    comptime K = 2048

    print("--- single-token decode: 1x11008x2048 ---\n")

    # dispatch kernel matrices
    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c_dispatch = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    # Warmup
    matmul_dispatch(c_dispatch, a, b)

    @parameter
    fn bench_dispatch_fn():
        matmul_dispatch(c_dispatch, a, b)

    var r_dispatch = std.benchmark.run[bench_dispatch_fn]()
    var s_d_mean = r_dispatch.mean("s")
    var s_d_min = r_dispatch.min("s")
    print(
        "  dispatch:",
        r_dispatch.mean("ms"),
        "ms (mean) |",
        r_dispatch.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_d_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_d_min),
        "GFLOPS (peak)",
    )
    print("")


fn bench_prefill() raises:
    """Benchmark prefill: 96x11008x2048."""
    comptime M = 96
    comptime N = 11008
    comptime K = 2048

    print("--- prefill: 96x11008x2048 ---\n")

    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c_dispatch = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    # Warmup
    matmul_dispatch(c_dispatch, a, b)

    @parameter
    fn bench_dispatch_fn():
        matmul_dispatch(c_dispatch, a, b)

    var r_dispatch = std.benchmark.run[bench_dispatch_fn]()
    var s_d_mean = r_dispatch.mean("s")
    var s_d_min = r_dispatch.min("s")
    print(
        "  dispatch:",
        r_dispatch.mean("ms"),
        "ms (mean) |",
        r_dispatch.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_d_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_d_min),
        "GFLOPS (peak)",
    )
    print("")


fn main() raises:
    print("=" * 60)
    print("  dispatch (fastest custom) - dispatch only")
    print("=" * 60)
    print("")
    print("Note: linalg.matmul is not available in this Mojo version")
    print("")

    bench_decode()
    bench_prefill()
