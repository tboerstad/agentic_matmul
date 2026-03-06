from gemm import matmul
from matrix import Matrix
import std.benchmark
from time import perf_counter_ns


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    """GFLOPS for an MxNxK matmul: 2*M*N*K FLOPs."""
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn main() raises:
    var t_start = perf_counter_ns()
    print("=== matmul benchmark (Qwen 2.5 VL 3B shapes) ===\n")

    # --- 1x11008x2048 (single-token decode, MLP gate/up projection) ---
    @parameter
    fn bench_decode():
        comptime M = 1
        comptime N = 11008
        comptime K = 2048
        var a = Matrix(M, K)
        var b = Matrix(K, N)
        var c = Matrix(M, N)
        fill(a, 17)
        fill(b, 13)
        matmul(c, a, b)

    var r_dec = std.benchmark.run[bench_decode]()
    var mean_dec = r_dec.mean("s")
    print(
        "1x11008x2048   :",
        r_dec.mean("ms"),
        "ms |",
        gflops(1, 11008, 2048, mean_dec),
        "GFLOPS",
    )

    # --- 96x11008x2048 (prefill batch, MLP gate/up projection) ---
    @parameter
    fn bench_prefill():
        comptime M = 96
        comptime N = 11008
        comptime K = 2048
        var a = Matrix(M, K)
        var b = Matrix(K, N)
        var c = Matrix(M, N)
        fill(a, 17)
        fill(b, 13)
        matmul(c, a, b)

    var r_pre = std.benchmark.run[bench_prefill]()
    var mean_pre = r_pre.mean("s")
    print(
        "96x11008x2048 :",
        r_pre.mean("ms"),
        "ms |",
        gflops(96, 11008, 2048, mean_pre),
        "GFLOPS",
    )

    print("\n--- full reports ---\n")
    print("1x11008x2048 (decode):")
    r_dec.print()
    print("\n96x11008x2048 (prefill):")
    r_pre.print()

    var t_end = perf_counter_ns()
    var elapsed_s = Float64(t_end - t_start) / 1e9
    print("\n=== total benchmark wall time:", elapsed_s, "s ===")
