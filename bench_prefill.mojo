from gemm import matmul_goto, matmul_prefill, matmul_prefill_v2
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
    print("=== prefill benchmark: goto vs prefill vs prefill_v2 (96x11008x2048) ===\n")

    comptime M = 96
    comptime N = 11008
    comptime K = 2048

    # Pre-allocate matrices once
    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c_goto = Matrix(M, N)
    var c_prefill = Matrix(M, N)
    var c_v2 = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    @parameter
    fn bench_goto():
        matmul_goto(c_goto, a, b)

    @parameter
    fn bench_prefill():
        matmul_prefill(c_prefill, a, b)

    @parameter
    fn bench_v2():
        matmul_prefill_v2(c_v2, a, b)

    # Warm up
    matmul_goto(c_goto, a, b)
    matmul_prefill(c_prefill, a, b)
    matmul_prefill_v2(c_v2, a, b)

    # Verify correctness
    var max_diff_gp = Float64(0)
    var max_diff_gv = Float64(0)
    for i in range(M):
        for j in range(N):
            var d1 = Float64(c_goto[i, j] - c_prefill[i, j])
            if d1 < 0:
                d1 = -d1
            if d1 > max_diff_gp:
                max_diff_gp = d1
            var d2 = Float64(c_goto[i, j] - c_v2[i, j])
            if d2 < 0:
                d2 = -d2
            if d2 > max_diff_gv:
                max_diff_gv = d2
    print("  max diff (goto vs prefill):", max_diff_gp)
    print("  max diff (goto vs v2):", max_diff_gv)
    if max_diff_gp > 1e-6 or max_diff_gv > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK\n")

    # Benchmark
    var r_goto = std.benchmark.run[bench_goto]()
    var s_goto_mean = r_goto.mean("s")
    var s_goto_min = r_goto.min("s")
    print(
        "  goto       :",
        r_goto.mean("ms"),
        "ms (mean) |",
        r_goto.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_goto_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_goto_min),
        "GFLOPS (min)",
    )

    var r_prefill = std.benchmark.run[bench_prefill]()
    var s_prefill_mean = r_prefill.mean("s")
    var s_prefill_min = r_prefill.min("s")
    print(
        "  prefill    :",
        r_prefill.mean("ms"),
        "ms (mean) |",
        r_prefill.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_prefill_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_prefill_min),
        "GFLOPS (min)",
    )

    var r_v2 = std.benchmark.run[bench_v2]()
    var s_v2_mean = r_v2.mean("s")
    var s_v2_min = r_v2.min("s")
    print(
        "  prefill_v2 :",
        r_v2.mean("ms"),
        "ms (mean) |",
        r_v2.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_v2_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_v2_min),
        "GFLOPS (min)",
    )

    # Speedups
    var speedup_v2_vs_prefill_mean = s_prefill_mean / s_v2_mean
    var speedup_v2_vs_prefill_min = s_prefill_min / s_v2_min
    var speedup_v2_vs_goto_mean = s_goto_mean / s_v2_mean
    var speedup_v2_vs_goto_min = s_goto_min / s_v2_min

    print("\n  v2 vs prefill  speedup mean:", speedup_v2_vs_prefill_mean, "x |", (speedup_v2_vs_prefill_mean - 1.0) * 100.0, "%")
    print("  v2 vs prefill  speedup min :", speedup_v2_vs_prefill_min, "x |", (speedup_v2_vs_prefill_min - 1.0) * 100.0, "%")
    print("  v2 vs goto     speedup mean:", speedup_v2_vs_goto_mean, "x |", (speedup_v2_vs_goto_mean - 1.0) * 100.0, "%")
    print("  v2 vs goto     speedup min :", speedup_v2_vs_goto_min, "x |", (speedup_v2_vs_goto_min - 1.0) * 100.0, "%")
