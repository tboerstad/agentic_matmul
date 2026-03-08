from gemm import matmul_dispatch, matmul_decode, matmul_prefill, matmul_goto
from linalg.matmul import matmul as linalg_matmul
from matrix import Matrix
from buffer import NDBuffer
from std.memory.unsafe_pointer import alloc
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
    print("=== benchmark: best kernels vs linalg.matmul ===\n")

    # ===========================================================================
    #  DECODE: 1 x 11008 x 2048
    # ===========================================================================
    comptime M1 = 1
    comptime N1 = 11008
    comptime K1 = 2048

    print("--- decode: " + String(M1) + "x" + String(N1) + "x" + String(K1) + " ---\n")

    # Matrices for custom kernels
    var a1 = Matrix(M1, K1)
    var b1 = Matrix(K1, N1)
    var c1_dispatch = Matrix(M1, N1)
    var c1_decode = Matrix(M1, N1)
    var c1_goto = Matrix(M1, N1)
    fill(a1, 17)
    fill(b1, 13)

    # Raw buffers for linalg.matmul
    var a1_ptr = alloc[Float64](M1 * K1)
    var b1_ptr = alloc[Float64](K1 * N1)
    var c1_ptr = alloc[Float64](M1 * N1)
    for i in range(M1 * K1):
        a1_ptr[i] = Float64((i // K1 * K1 + i % K1) % 17) * 0.1
    for i in range(K1 * N1):
        b1_ptr[i] = Float64((i // N1 * N1 + i % N1) % 13) * 0.1
    for i in range(M1 * N1):
        c1_ptr[i] = 0.0

    # Warm up + correctness
    matmul_dispatch(c1_dispatch, a1, b1)
    matmul_decode(c1_decode, a1, b1)
    matmul_goto(c1_goto, a1, b1)

    var a1_buf = NDBuffer[DType.float64, 2](a1_ptr, (M1, K1))
    var b1_buf = NDBuffer[DType.float64, 2](b1_ptr, (K1, N1))
    var c1_buf = NDBuffer[DType.float64, 2](c1_ptr, (M1, N1))
    linalg_matmul(c1_buf, a1_buf, b1_buf)

    # Verify correctness
    var max_diff_1 = Float64(0)
    for i in range(M1):
        for j in range(N1):
            var d = Float64(c1_dispatch[i, j]) - c1_ptr[i * N1 + j]
            if d < 0:
                d = -d
            if d > max_diff_1:
                max_diff_1 = d
    print("  max diff (dispatch vs linalg):", max_diff_1)
    if max_diff_1 > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK")
    print("")

    # Benchmark closures
    @parameter
    fn bench_decode_linalg():
        for i in range(M1 * N1):
            c1_ptr[i] = 0.0
        try:
            linalg_matmul(c1_buf, a1_buf, b1_buf)
        except:
            pass

    @parameter
    fn bench_decode_dispatch():
        matmul_dispatch(c1_dispatch, a1, b1)

    @parameter
    fn bench_decode_decode():
        matmul_decode(c1_decode, a1, b1)

    @parameter
    fn bench_decode_goto():
        matmul_goto(c1_goto, a1, b1)

    # Run benchmarks
    var r_linalg_1 = std.benchmark.run[bench_decode_linalg]()
    var s_linalg_1_mean = r_linalg_1.mean("s")
    var s_linalg_1_min = r_linalg_1.min("s")

    var r_dispatch_1 = std.benchmark.run[bench_decode_dispatch]()
    var s_dispatch_1_mean = r_dispatch_1.mean("s")
    var s_dispatch_1_min = r_dispatch_1.min("s")

    var r_decode_1 = std.benchmark.run[bench_decode_decode]()
    var s_decode_1_mean = r_decode_1.mean("s")
    var s_decode_1_min = r_decode_1.min("s")

    var r_goto_1 = std.benchmark.run[bench_decode_goto]()
    var s_goto_1_mean = r_goto_1.mean("s")
    var s_goto_1_min = r_goto_1.min("s")

    # Print results
    print(
        "  linalg  :",
        r_linalg_1.mean("ms"), "ms (mean) |",
        r_linalg_1.min("ms"), "ms (min) |",
        gflops(M1, N1, K1, s_linalg_1_mean), "GFLOPS (mean) |",
        gflops(M1, N1, K1, s_linalg_1_min), "GFLOPS (peak)",
    )
    print(
        "  dispatch:",
        r_dispatch_1.mean("ms"), "ms (mean) |",
        r_dispatch_1.min("ms"), "ms (min) |",
        gflops(M1, N1, K1, s_dispatch_1_mean), "GFLOPS (mean) |",
        gflops(M1, N1, K1, s_dispatch_1_min), "GFLOPS (peak)",
    )
    print(
        "  decode  :",
        r_decode_1.mean("ms"), "ms (mean) |",
        r_decode_1.min("ms"), "ms (min) |",
        gflops(M1, N1, K1, s_decode_1_mean), "GFLOPS (mean) |",
        gflops(M1, N1, K1, s_decode_1_min), "GFLOPS (peak)",
    )
    print(
        "  goto    :",
        r_goto_1.mean("ms"), "ms (mean) |",
        r_goto_1.min("ms"), "ms (min) |",
        gflops(M1, N1, K1, s_goto_1_mean), "GFLOPS (mean) |",
        gflops(M1, N1, K1, s_goto_1_min), "GFLOPS (peak)",
    )

    print("\n  --- speedup vs linalg.matmul (decode) ---")
    print("  dispatch/linalg (mean):", s_linalg_1_mean / s_dispatch_1_mean, "x")
    print("  dispatch/linalg (peak):", s_linalg_1_min / s_dispatch_1_min, "x")
    print("  decode/linalg   (mean):", s_linalg_1_mean / s_decode_1_mean, "x")
    print("  decode/linalg   (peak):", s_linalg_1_min / s_decode_1_min, "x")
    print("  goto/linalg     (mean):", s_linalg_1_mean / s_goto_1_mean, "x")
    print("  goto/linalg     (peak):", s_linalg_1_min / s_goto_1_min, "x")

    # ===========================================================================
    #  PREFILL: 96 x 11008 x 2048
    # ===========================================================================
    comptime M2 = 96
    comptime N2 = 11008
    comptime K2 = 2048

    print("\n--- prefill: " + String(M2) + "x" + String(N2) + "x" + String(K2) + " ---\n")

    # Matrices for custom kernels
    var a2 = Matrix(M2, K2)
    var b2 = Matrix(K2, N2)
    var c2_prefill = Matrix(M2, N2)
    var c2_dispatch = Matrix(M2, N2)
    var c2_goto = Matrix(M2, N2)
    fill(a2, 17)
    fill(b2, 13)

    # Raw buffers for linalg.matmul
    var a2_ptr = alloc[Float64](M2 * K2)
    var b2_ptr = alloc[Float64](K2 * N2)
    var c2_ptr = alloc[Float64](M2 * N2)
    for i in range(M2 * K2):
        a2_ptr[i] = Float64((i // K2 * K2 + i % K2) % 17) * 0.1
    for i in range(K2 * N2):
        b2_ptr[i] = Float64((i // N2 * N2 + i % N2) % 13) * 0.1
    for i in range(M2 * N2):
        c2_ptr[i] = 0.0

    # Warm up + correctness
    matmul_prefill(c2_prefill, a2, b2)
    matmul_dispatch(c2_dispatch, a2, b2)
    matmul_goto(c2_goto, a2, b2)

    var a2_buf = NDBuffer[DType.float64, 2](a2_ptr, (M2, K2))
    var b2_buf = NDBuffer[DType.float64, 2](b2_ptr, (K2, N2))
    var c2_buf = NDBuffer[DType.float64, 2](c2_ptr, (M2, N2))
    linalg_matmul(c2_buf, a2_buf, b2_buf)

    # Verify correctness
    var max_diff_2a = Float64(0)
    for i in range(M2):
        for j in range(N2):
            var d = Float64(c2_prefill[i, j]) - c2_ptr[i * N2 + j]
            if d < 0:
                d = -d
            if d > max_diff_2a:
                max_diff_2a = d
    print("  max diff (prefill vs linalg):", max_diff_2a)
    if max_diff_2a > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK")

    var max_diff_2b = Float64(0)
    for i in range(M2):
        for j in range(N2):
            var d = Float64(c2_dispatch[i, j]) - c2_ptr[i * N2 + j]
            if d < 0:
                d = -d
            if d > max_diff_2b:
                max_diff_2b = d
    print("  max diff (dispatch vs linalg):", max_diff_2b)
    if max_diff_2b > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK")
    print("")

    # Benchmark closures
    @parameter
    fn bench_prefill_linalg():
        for i in range(M2 * N2):
            c2_ptr[i] = 0.0
        try:
            linalg_matmul(c2_buf, a2_buf, b2_buf)
        except:
            pass

    @parameter
    fn bench_prefill_prefill():
        matmul_prefill(c2_prefill, a2, b2)

    @parameter
    fn bench_prefill_dispatch():
        matmul_dispatch(c2_dispatch, a2, b2)

    @parameter
    fn bench_prefill_goto():
        matmul_goto(c2_goto, a2, b2)

    # Run benchmarks
    var r_linalg_2 = std.benchmark.run[bench_prefill_linalg]()
    var s_linalg_2_mean = r_linalg_2.mean("s")
    var s_linalg_2_min = r_linalg_2.min("s")

    var r_prefill_2 = std.benchmark.run[bench_prefill_prefill]()
    var s_prefill_2_mean = r_prefill_2.mean("s")
    var s_prefill_2_min = r_prefill_2.min("s")

    var r_dispatch_2 = std.benchmark.run[bench_prefill_dispatch]()
    var s_dispatch_2_mean = r_dispatch_2.mean("s")
    var s_dispatch_2_min = r_dispatch_2.min("s")

    var r_goto_2 = std.benchmark.run[bench_prefill_goto]()
    var s_goto_2_mean = r_goto_2.mean("s")
    var s_goto_2_min = r_goto_2.min("s")

    # Print results
    print(
        "  linalg  :",
        r_linalg_2.mean("ms"), "ms (mean) |",
        r_linalg_2.min("ms"), "ms (min) |",
        gflops(M2, N2, K2, s_linalg_2_mean), "GFLOPS (mean) |",
        gflops(M2, N2, K2, s_linalg_2_min), "GFLOPS (peak)",
    )
    print(
        "  prefill :",
        r_prefill_2.mean("ms"), "ms (mean) |",
        r_prefill_2.min("ms"), "ms (min) |",
        gflops(M2, N2, K2, s_prefill_2_mean), "GFLOPS (mean) |",
        gflops(M2, N2, K2, s_prefill_2_min), "GFLOPS (peak)",
    )
    print(
        "  dispatch:",
        r_dispatch_2.mean("ms"), "ms (mean) |",
        r_dispatch_2.min("ms"), "ms (min) |",
        gflops(M2, N2, K2, s_dispatch_2_mean), "GFLOPS (mean) |",
        gflops(M2, N2, K2, s_dispatch_2_min), "GFLOPS (peak)",
    )
    print(
        "  goto    :",
        r_goto_2.mean("ms"), "ms (mean) |",
        r_goto_2.min("ms"), "ms (min) |",
        gflops(M2, N2, K2, s_goto_2_mean), "GFLOPS (mean) |",
        gflops(M2, N2, K2, s_goto_2_min), "GFLOPS (peak)",
    )

    print("\n  --- speedup vs linalg.matmul (prefill) ---")
    print("  prefill/linalg  (mean):", s_linalg_2_mean / s_prefill_2_mean, "x")
    print("  prefill/linalg  (peak):", s_linalg_2_min / s_prefill_2_min, "x")
    print("  dispatch/linalg (mean):", s_linalg_2_mean / s_dispatch_2_mean, "x")
    print("  dispatch/linalg (peak):", s_linalg_2_min / s_dispatch_2_min, "x")
    print("  goto/linalg     (mean):", s_linalg_2_mean / s_goto_2_mean, "x")
    print("  goto/linalg     (peak):", s_linalg_2_min / s_goto_2_min, "x")

    # ===========================================================================
    #  Full reports
    # ===========================================================================
    print("\n--- full reports ---\n")
    print("decode linalg:")
    r_linalg_1.print()
    print("\ndecode dispatch:")
    r_dispatch_1.print()
    print("\ndecode decode:")
    r_decode_1.print()
    print("\ndecode goto:")
    r_goto_1.print()
    print("\nprefill linalg:")
    r_linalg_2.print()
    print("\nprefill prefill:")
    r_prefill_2.print()
    print("\nprefill dispatch:")
    r_dispatch_2.print()
    print("\nprefill goto:")
    r_goto_2.print()

    # Clean up
    a1_ptr.free()
    b1_ptr.free()
    c1_ptr.free()
    a2_ptr.free()
    b2_ptr.free()
    c2_ptr.free()

    var t_end = perf_counter_ns()
    var elapsed_s = Float64(t_end - t_start) / 1e9
    print("\n=== total benchmark wall time:", elapsed_s, "s ===")
