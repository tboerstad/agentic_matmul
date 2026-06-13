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
from std.time import perf_counter_ns


def gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


def fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


def report(label: String, m: Int, n: Int, k: Int, sum_ns: Float64, min_ns: Float64, n_runs: Int) -> Float64:
    var mean_s = sum_ns / Float64(n_runs) / 1e9
    var min_s = min_ns / 1e9
    print(
        "  ", label, ":",
        mean_s * 1e3, "ms |",
        gflops(m, n, k, mean_s), "GFLOPS (mean) |",
        gflops(m, n, k, min_s), "GFLOPS (peak)",
    )
    return mean_s


def main() raises:
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

    print("--- 1x11008x2048 (decode) ---\n")

    var t0: UInt = 0
    var t1: UInt = 0
    var dt: Float64 = 0.0
    var min_ns: Float64 = 0.0
    var sum_ns: Float64 = 0.0

    # naive — one run (very slow)
    t0 = perf_counter_ns()
    matmul_naive(c1, a1, b1)
    t1 = perf_counter_ns()
    sum_ns = Float64(t1 - t0)
    min_ns = sum_ns
    var s_naive_1 = report("naive       ", M1, N1, K1, sum_ns, min_ns, 1)

    # tiled — 3 runs
    matmul_tiled(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_tiled(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_tiled_1 = report("tiled       ", M1, N1, K1, sum_ns, min_ns, 3)

    # simd — 3 runs
    matmul_simd(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_simd(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_simd_1 = report("simd        ", M1, N1, K1, sum_ns, min_ns, 3)

    # parallel — 5 runs
    matmul_parallel(c1, a1, b1)
    matmul_parallel(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_parallel(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_parallel_1 = report("parallel    ", M1, N1, K1, sum_ns, min_ns, 5)

    # register_blocked — 5 runs
    matmul_register_blocked(c1, a1, b1)
    matmul_register_blocked(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_register_blocked(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_regblk_1 = report("regblk      ", M1, N1, K1, sum_ns, min_ns, 5)

    # packed — 5 runs
    matmul_packed(c1, a1, b1)
    matmul_packed(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_packed(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_packed_1 = report("packed      ", M1, N1, K1, sum_ns, min_ns, 5)

    # comptime — 5 runs
    matmul_comptime(c1, a1, b1)
    matmul_comptime(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_comptime(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_comptime_1 = report("comptime    ", M1, N1, K1, sum_ns, min_ns, 5)

    # goto — 5 runs
    matmul_goto(c1, a1, b1)
    matmul_goto(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_goto(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_goto_1 = report("goto        ", M1, N1, K1, sum_ns, min_ns, 5)

    # prefill — 5 runs
    matmul_prefill(c1, a1, b1)
    matmul_prefill(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_prefill(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_prefill_1 = report("prefill     ", M1, N1, K1, sum_ns, min_ns, 5)

    # prefill_opt — 5 runs
    matmul_prefill_opt(c1, a1, b1)
    matmul_prefill_opt(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_prefill_opt(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_prefill_opt_1 = report("prefill_opt ", M1, N1, K1, sum_ns, min_ns, 5)

    # decode — 5 runs
    matmul_decode(c1, a1, b1)
    matmul_decode(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_decode(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_decode_1 = report("decode      ", M1, N1, K1, sum_ns, min_ns, 5)

    # dispatch — 5 runs
    matmul_dispatch(c1, a1, b1)
    matmul_dispatch(c1, a1, b1)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(5):
        t0 = perf_counter_ns(); matmul_dispatch(c1, a1, b1); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_dispatch_1 = report("dispatch    ", M1, N1, K1, sum_ns, min_ns, 5)

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

    print("--- 96x11008x2048 (prefill) ---\n")

    # naive — one run (very slow)
    t0 = perf_counter_ns(); matmul_naive(c2, a2, b2); t1 = perf_counter_ns()
    sum_ns = Float64(t1 - t0); min_ns = sum_ns
    var s_naive_2 = report("naive       ", M2, N2, K2, sum_ns, min_ns, 1)

    # tiled — 1 run
    t0 = perf_counter_ns(); matmul_tiled(c2, a2, b2); t1 = perf_counter_ns()
    sum_ns = Float64(t1 - t0); min_ns = sum_ns
    var s_tiled_2 = report("tiled       ", M2, N2, K2, sum_ns, min_ns, 1)

    # simd — 1 run
    t0 = perf_counter_ns(); matmul_simd(c2, a2, b2); t1 = perf_counter_ns()
    sum_ns = Float64(t1 - t0); min_ns = sum_ns
    var s_simd_2 = report("simd        ", M2, N2, K2, sum_ns, min_ns, 1)

    # parallel — 3 runs
    matmul_parallel(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_parallel(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_parallel_2 = report("parallel    ", M2, N2, K2, sum_ns, min_ns, 3)

    # register_blocked — 3 runs
    matmul_register_blocked(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_register_blocked(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_regblk_2 = report("regblk      ", M2, N2, K2, sum_ns, min_ns, 3)

    # packed — 3 runs
    matmul_packed(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_packed(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_packed_2 = report("packed      ", M2, N2, K2, sum_ns, min_ns, 3)

    # comptime — 3 runs
    matmul_comptime(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_comptime(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_comptime_2 = report("comptime    ", M2, N2, K2, sum_ns, min_ns, 3)

    # goto — 3 runs
    matmul_goto(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_goto(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_goto_2 = report("goto        ", M2, N2, K2, sum_ns, min_ns, 3)

    # prefill — 3 runs
    matmul_prefill(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_prefill(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_prefill_2 = report("prefill     ", M2, N2, K2, sum_ns, min_ns, 3)

    # prefill_opt — 3 runs
    matmul_prefill_opt(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_prefill_opt(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_prefill_opt_2 = report("prefill_opt ", M2, N2, K2, sum_ns, min_ns, 3)

    # decode — 3 runs
    matmul_decode(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_decode(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_decode_2 = report("decode      ", M2, N2, K2, sum_ns, min_ns, 3)

    # dispatch — 3 runs
    matmul_dispatch(c2, a2, b2)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(3):
        t0 = perf_counter_ns(); matmul_dispatch(c2, a2, b2); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    var s_dispatch_2 = report("dispatch    ", M2, N2, K2, sum_ns, min_ns, 3)

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
