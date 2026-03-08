"""Focused benchmark: matmul_goto only (decode + prefill shapes)."""
from gemm import matmul_goto
from matrix import Matrix
import std.benchmark
from std.time import perf_counter_ns


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype](((i * m.cols + j) % seed)) * 0.1


fn main() raises:
    var t_start = perf_counter_ns()
    print("=== matmul_goto benchmark (Qwen 2.5 VL 3B shapes) ===\n")

    # ---- Decode: 1x11008x2048 ----
    comptime M1 = 1
    comptime N1 = 11008
    comptime K1 = 2048

    @parameter
    fn bench_decode_goto():
        var a = Matrix(M1, K1)
        var b = Matrix(K1, N1)
        var c = Matrix(M1, N1)
        fill(a, 17)
        fill(b, 13)
        matmul_goto(c, a, b)

    print("--- 1x11008x2048 (decode) ---")
    var r1 = std.benchmark.run[bench_decode_goto]()
    var s1 = r1.mean("s")
    print("  goto:", r1.mean("ms"), "ms |", gflops(M1, N1, K1, s1), "GFLOPS")
    print()
    r1.print()

    # ---- Prefill: 96x11008x2048 ----
    comptime M2 = 96
    comptime N2 = 11008
    comptime K2 = 2048

    @parameter
    fn bench_prefill_goto():
        var a = Matrix(M2, K2)
        var b = Matrix(K2, N2)
        var c = Matrix(M2, N2)
        fill(a, 17)
        fill(b, 13)
        matmul_goto(c, a, b)

    print("\n--- 96x11008x2048 (prefill) ---")
    var r2 = std.benchmark.run[bench_prefill_goto]()
    var s2 = r2.mean("s")
    print("  goto:", r2.mean("ms"), "ms |", gflops(M2, N2, K2, s2), "GFLOPS")
    print()
    r2.print()

    var t_end = perf_counter_ns()
    var elapsed_s = Float64(t_end - t_start) / 1e9
    print("\n=== total wall time:", elapsed_s, "s ===")
