from gemm import (
    matmul_naive,
    matmul_tiled,
    matmul_simd,
    matmul_parallel,
    matmul_register_blocked,
    matmul_packed,
    matmul_comptime,
    matmul_goto,
    matmul_prefill,
    matmul_prefill_opt,
    matmul_decode,
    matmul_dispatch,
)
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
    print("=== matmul benchmark: all kernels (Qwen 2.5 VL 3B shapes) ===\n")

    # ---- decode: 1x11008x2048 ------------------------------------------------

    comptime M1 = 1
    comptime N1 = 11008
    comptime K1 = 2048

    var a1 = Matrix(M1, K1)
    var b1 = Matrix(K1, N1)
    var c1 = Matrix(M1, N1)
    fill(a1, 17)
    fill(b1, 13)

    @parameter
    fn d_naive():
        matmul_naive(c1, a1, b1)

    @parameter
    fn d_tiled():
        matmul_tiled(c1, a1, b1)

    @parameter
    fn d_simd():
        matmul_simd(c1, a1, b1)

    @parameter
    fn d_parallel():
        matmul_parallel(c1, a1, b1)

    @parameter
    fn d_regblk():
        matmul_register_blocked(c1, a1, b1)

    @parameter
    fn d_packed():
        matmul_packed(c1, a1, b1)

    @parameter
    fn d_comptime():
        matmul_comptime(c1, a1, b1)

    @parameter
    fn d_goto():
        matmul_goto(c1, a1, b1)

    @parameter
    fn d_prefill():
        matmul_prefill(c1, a1, b1)

    @parameter
    fn d_prefill_opt():
        matmul_prefill_opt(c1, a1, b1)

    @parameter
    fn d_decode():
        matmul_decode(c1, a1, b1)

    @parameter
    fn d_dispatch():
        matmul_dispatch(c1, a1, b1)

    print("--- 1x11008x2048 (decode) ---\n")

    var r = std.benchmark.run[d_naive]()
    var s_naive_1 = r.mean("s")
    print("  naive       :", r.mean("ms"), "ms |", gflops(M1, N1, K1, s_naive_1), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")

    r = std.benchmark.run[d_tiled]()
    print("  tiled       :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_tiled_1 = r.mean("s")

    r = std.benchmark.run[d_simd]()
    print("  simd        :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_simd_1 = r.mean("s")

    r = std.benchmark.run[d_parallel]()
    print("  parallel    :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_parallel_1 = r.mean("s")

    r = std.benchmark.run[d_regblk]()
    print("  regblk      :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_regblk_1 = r.mean("s")

    r = std.benchmark.run[d_packed]()
    print("  packed      :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_packed_1 = r.mean("s")

    r = std.benchmark.run[d_comptime]()
    print("  comptime    :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_comptime_1 = r.mean("s")

    r = std.benchmark.run[d_goto]()
    print("  goto        :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_goto_1 = r.mean("s")

    r = std.benchmark.run[d_prefill]()
    print("  prefill     :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_prefill_1 = r.mean("s")

    r = std.benchmark.run[d_prefill_opt]()
    print("  prefill_opt :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_prefill_opt_1 = r.mean("s")

    r = std.benchmark.run[d_decode]()
    print("  decode      :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_decode_1 = r.mean("s")

    r = std.benchmark.run[d_dispatch]()
    print("  dispatch    :", r.mean("ms"), "ms |", gflops(M1, N1, K1, r.mean("s")), "GFLOPS (mean) |", gflops(M1, N1, K1, r.min("s")), "GFLOPS (peak)")
    var s_dispatch_1 = r.mean("s")

    print("\n  speedup vs naive:")
    print("    tiled       :", s_naive_1 / s_tiled_1, "x")
    print("    simd        :", s_naive_1 / s_simd_1, "x")
    print("    parallel    :", s_naive_1 / s_parallel_1, "x")
    print("    regblk      :", s_naive_1 / s_regblk_1, "x")
    print("    packed      :", s_naive_1 / s_packed_1, "x")
    print("    comptime    :", s_naive_1 / s_comptime_1, "x")
    print("    goto        :", s_naive_1 / s_goto_1, "x")
    print("    prefill     :", s_naive_1 / s_prefill_1, "x")
    print("    prefill_opt :", s_naive_1 / s_prefill_opt_1, "x")
    print("    decode      :", s_naive_1 / s_decode_1, "x")
    print("    dispatch    :", s_naive_1 / s_dispatch_1, "x")
    print("")

    # ---- prefill: 96x11008x2048 ----------------------------------------------

    comptime M2 = 96
    comptime N2 = 11008
    comptime K2 = 2048

    var a2 = Matrix(M2, K2)
    var b2 = Matrix(K2, N2)
    var c2 = Matrix(M2, N2)
    fill(a2, 17)
    fill(b2, 13)

    @parameter
    fn p_naive():
        matmul_naive(c2, a2, b2)

    @parameter
    fn p_tiled():
        matmul_tiled(c2, a2, b2)

    @parameter
    fn p_simd():
        matmul_simd(c2, a2, b2)

    @parameter
    fn p_parallel():
        matmul_parallel(c2, a2, b2)

    @parameter
    fn p_regblk():
        matmul_register_blocked(c2, a2, b2)

    @parameter
    fn p_packed():
        matmul_packed(c2, a2, b2)

    @parameter
    fn p_comptime():
        matmul_comptime(c2, a2, b2)

    @parameter
    fn p_goto():
        matmul_goto(c2, a2, b2)

    @parameter
    fn p_prefill():
        matmul_prefill(c2, a2, b2)

    @parameter
    fn p_prefill_opt():
        matmul_prefill_opt(c2, a2, b2)

    @parameter
    fn p_decode():
        matmul_decode(c2, a2, b2)

    @parameter
    fn p_dispatch():
        matmul_dispatch(c2, a2, b2)

    print("--- 96x11008x2048 (prefill) ---\n")

    r = std.benchmark.run[p_naive]()
    var s_naive_2 = r.mean("s")
    print("  naive       :", r.mean("ms"), "ms |", gflops(M2, N2, K2, s_naive_2), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")

    r = std.benchmark.run[p_tiled]()
    print("  tiled       :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_tiled_2 = r.mean("s")

    r = std.benchmark.run[p_simd]()
    print("  simd        :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_simd_2 = r.mean("s")

    r = std.benchmark.run[p_parallel]()
    print("  parallel    :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_parallel_2 = r.mean("s")

    r = std.benchmark.run[p_regblk]()
    print("  regblk      :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_regblk_2 = r.mean("s")

    r = std.benchmark.run[p_packed]()
    print("  packed      :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_packed_2 = r.mean("s")

    r = std.benchmark.run[p_comptime]()
    print("  comptime    :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_comptime_2 = r.mean("s")

    r = std.benchmark.run[p_goto]()
    print("  goto        :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_goto_2 = r.mean("s")

    r = std.benchmark.run[p_prefill]()
    print("  prefill     :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_prefill_2 = r.mean("s")

    r = std.benchmark.run[p_prefill_opt]()
    print("  prefill_opt :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_prefill_opt_2 = r.mean("s")

    r = std.benchmark.run[p_decode]()
    print("  decode      :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_decode_2 = r.mean("s")

    r = std.benchmark.run[p_dispatch]()
    print("  dispatch    :", r.mean("ms"), "ms |", gflops(M2, N2, K2, r.mean("s")), "GFLOPS (mean) |", gflops(M2, N2, K2, r.min("s")), "GFLOPS (peak)")
    var s_dispatch_2 = r.mean("s")

    print("\n  speedup vs naive:")
    print("    tiled       :", s_naive_2 / s_tiled_2, "x")
    print("    simd        :", s_naive_2 / s_simd_2, "x")
    print("    parallel    :", s_naive_2 / s_parallel_2, "x")
    print("    regblk      :", s_naive_2 / s_regblk_2, "x")
    print("    packed      :", s_naive_2 / s_packed_2, "x")
    print("    comptime    :", s_naive_2 / s_comptime_2, "x")
    print("    goto        :", s_naive_2 / s_goto_2, "x")
    print("    prefill     :", s_naive_2 / s_prefill_2, "x")
    print("    prefill_opt :", s_naive_2 / s_prefill_opt_2, "x")
    print("    decode      :", s_naive_2 / s_decode_2, "x")
    print("    dispatch    :", s_naive_2 / s_dispatch_2, "x")

    var t_end = perf_counter_ns()
    var elapsed_s = Float64(t_end - t_start) / 1e9
    print("\n=== total benchmark wall time:", elapsed_s, "s ===")
