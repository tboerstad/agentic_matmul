from gemm import matmul_goto, matmul_decode, matmul_dispatch
from matrix import Matrix
import std.benchmark
from std.time import perf_counter_ns


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn verify(expected: Matrix, actual: Matrix, label: String) raises:
    var max_diff = Float64(0)
    for i in range(expected.rows):
        for j in range(expected.cols):
            var d = Float64(expected[i, j]) - Float64(actual[i, j])
            if d < 0:
                d = -d
            if d > max_diff:
                max_diff = d
    print("  max diff (" + label + "):", max_diff)
    if max_diff > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK")


fn bench_shape(m: Int, n: Int, k: Int, label: String) raises:
    print("--- " + label + ": " + String(m) + "x" + String(n) + "x" + String(k) + " ---\n")

    var a = Matrix(m, k)
    var b = Matrix(k, n)
    var c_goto = Matrix(m, n)
    var c_decode = Matrix(m, n)
    var c_dispatch = Matrix(m, n)
    fill(a, 17)
    fill(b, 13)

    # Warm up + correctness check
    matmul_goto(c_goto, a, b)
    matmul_decode(c_decode, a, b)
    matmul_dispatch(c_dispatch, a, b)
    verify(c_goto, c_decode, "goto vs decode")
    verify(c_goto, c_dispatch, "goto vs dispatch")
    print("")

    @parameter
    fn bench_goto():
        matmul_goto(c_goto, a, b)

    @parameter
    fn bench_decode():
        matmul_decode(c_decode, a, b)

    @parameter
    fn bench_dispatch():
        matmul_dispatch(c_dispatch, a, b)

    var r_goto = std.benchmark.run[bench_goto]()
    var s_goto_mean = r_goto.mean("s")
    var s_goto_min = r_goto.min("s")
    print(
        "  goto    :",
        r_goto.mean("ms"),
        "ms (mean) |",
        r_goto.min("ms"),
        "ms (min) |",
        gflops(m, n, k, s_goto_mean),
        "GFLOPS (mean) |",
        gflops(m, n, k, s_goto_min),
        "GFLOPS (peak)",
    )

    var r_decode = std.benchmark.run[bench_decode]()
    var s_decode_mean = r_decode.mean("s")
    var s_decode_min = r_decode.min("s")
    print(
        "  decode  :",
        r_decode.mean("ms"),
        "ms (mean) |",
        r_decode.min("ms"),
        "ms (min) |",
        gflops(m, n, k, s_decode_mean),
        "GFLOPS (mean) |",
        gflops(m, n, k, s_decode_min),
        "GFLOPS (peak)",
    )

    var r_dispatch = std.benchmark.run[bench_dispatch]()
    var s_dispatch_mean = r_dispatch.mean("s")
    var s_dispatch_min = r_dispatch.min("s")
    print(
        "  dispatch:",
        r_dispatch.mean("ms"),
        "ms (mean) |",
        r_dispatch.min("ms"),
        "ms (min) |",
        gflops(m, n, k, s_dispatch_mean),
        "GFLOPS (mean) |",
        gflops(m, n, k, s_dispatch_min),
        "GFLOPS (peak)",
    )

    var speedup_mean = s_goto_mean / s_decode_mean
    var speedup_min = s_goto_min / s_decode_min
    print("\n  speedup mean (goto/decode):", speedup_mean, "x")
    print("  improvement mean:", (speedup_mean - 1.0) * 100.0, "%")
    print("  speedup peak (goto/decode):", speedup_min, "x")
    print("  improvement peak:", (speedup_min - 1.0) * 100.0, "%")
    print("")


fn main() raises:
    print("=== decode benchmark: goto vs decode vs dispatch ===\n")

    # M=1: single-token decode (pure bandwidth bound)
    bench_shape(1, 11008, 2048, "single-token decode")

    # M=7: max batch before MR=8 threshold
    bench_shape(7, 11008, 2048, "batch decode (M=7)")
