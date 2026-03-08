from gemm import matmul_goto, matmul_prefill, matmul_adaptive, print_hw_info
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
    print("=== prefill benchmark: goto vs prefill vs adaptive (96x11008x2048) ===\n")
    print_hw_info()

    comptime M = 96
    comptime N = 11008
    comptime K = 2048

    # Pre-allocate matrices once
    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c_goto = Matrix(M, N)
    var c_prefill = Matrix(M, N)
    var c_adaptive = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    @parameter
    fn bench_goto():
        matmul_goto(c_goto, a, b)

    @parameter
    fn bench_prefill():
        matmul_prefill(c_prefill, a, b)

    @parameter
    fn bench_adaptive():
        matmul_adaptive(c_adaptive, a, b)

    # Warm up
    matmul_goto(c_goto, a, b)
    matmul_prefill(c_prefill, a, b)
    matmul_adaptive(c_adaptive, a, b)

    # Verify correctness: compare outputs
    var max_diff = Float64(0)
    for i in range(M):
        for j in range(N):
            var diff = Float64(c_goto[i, j] - c_prefill[i, j])
            if diff < 0:
                diff = -diff
            if diff > max_diff:
                max_diff = diff
    print("  max diff (goto vs prefill):", max_diff)
    if max_diff > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK")

    var max_diff2 = Float64(0)
    for i in range(M):
        for j in range(N):
            var diff = Float64(c_goto[i, j] - c_adaptive[i, j])
            if diff < 0:
                diff = -diff
            if diff > max_diff2:
                max_diff2 = diff
    print("  max diff (goto vs adaptive):", max_diff2)
    if max_diff2 > 1e-6:
        print("  WARNING: adaptive results differ!")
    else:
        print("  correctness (adaptive): OK\n")

    # Benchmark
    var r_goto = std.benchmark.run[bench_goto]()
    var s_goto_mean = r_goto.mean("s")
    var s_goto_min = r_goto.min("s")
    print(
        "  goto    :",
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
        "  prefill  :",
        r_prefill.mean("ms"),
        "ms (mean) |",
        r_prefill.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_prefill_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_prefill_min),
        "GFLOPS (min)",
    )

    var r_adaptive = std.benchmark.run[bench_adaptive]()
    var s_adaptive_mean = r_adaptive.mean("s")
    var s_adaptive_min = r_adaptive.min("s")
    print(
        "  adaptive :",
        r_adaptive.mean("ms"),
        "ms (mean) |",
        r_adaptive.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_adaptive_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_adaptive_min),
        "GFLOPS (min)",
    )

    var speedup_mean = s_goto_mean / s_prefill_mean
    var speedup_min = s_goto_min / s_prefill_min
    print("\n  speedup mean (goto/prefill):", speedup_mean, "x")
    print("  improvement mean:", (speedup_mean - 1.0) * 100.0, "%")
    print("  speedup min  (goto/prefill):", speedup_min, "x")
    print("  improvement min:", (speedup_min - 1.0) * 100.0, "%")

    var speedup_adaptive = s_goto_mean / s_adaptive_mean
    print("\n  speedup mean (goto/adaptive):", speedup_adaptive, "x")
    print("  improvement mean:", (speedup_adaptive - 1.0) * 100.0, "%")

    print("\n--- full reports ---\n")
    print("goto:")
    r_goto.print()
    print("\nprefill:")
    r_prefill.print()
    print("\nadaptive:")
    r_adaptive.print()
