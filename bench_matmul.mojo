from gemm import matmul_naive, matmul_tiled, matmul_simd
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
    print("=== matmul benchmark: naive vs tiled vs simd (Qwen 2.5 VL 3B shapes) ===\n")

    # ---- 1x11008x2048 (single-token decode) ---------------------------------

    comptime M1 = 1
    comptime N1 = 11008
    comptime K1 = 2048

    @parameter
    fn bench_decode_naive():
        var a = Matrix(M1, K1)
        var b = Matrix(K1, N1)
        var c = Matrix(M1, N1)
        fill(a, 17)
        fill(b, 13)
        matmul_naive(c, a, b)

    @parameter
    fn bench_decode_tiled():
        var a = Matrix(M1, K1)
        var b = Matrix(K1, N1)
        var c = Matrix(M1, N1)
        fill(a, 17)
        fill(b, 13)
        matmul_tiled(c, a, b)

    @parameter
    fn bench_decode_simd():
        var a = Matrix(M1, K1)
        var b = Matrix(K1, N1)
        var c = Matrix(M1, N1)
        fill(a, 17)
        fill(b, 13)
        matmul_simd(c, a, b)

    print("--- 1x11008x2048 (decode) ---")

    var r_naive_1 = std.benchmark.run[bench_decode_naive]()
    var s_naive_1 = r_naive_1.mean("s")
    print(
        "  naive :",
        r_naive_1.mean("ms"),
        "ms |",
        gflops(M1, N1, K1, s_naive_1),
        "GFLOPS",
    )

    var r_tiled_1 = std.benchmark.run[bench_decode_tiled]()
    var s_tiled_1 = r_tiled_1.mean("s")
    print(
        "  tiled :",
        r_tiled_1.mean("ms"),
        "ms |",
        gflops(M1, N1, K1, s_tiled_1),
        "GFLOPS",
    )

    var r_simd_1 = std.benchmark.run[bench_decode_simd]()
    var s_simd_1 = r_simd_1.mean("s")
    print(
        "  simd  :",
        r_simd_1.mean("ms"),
        "ms |",
        gflops(M1, N1, K1, s_simd_1),
        "GFLOPS",
    )

    print("  speedup (naive/tiled):", s_naive_1 / s_tiled_1, "x")
    print("  speedup (naive/simd) :", s_naive_1 / s_simd_1, "x")
    print("  speedup (tiled/simd) :", s_tiled_1 / s_simd_1, "x\n")

    # ---- 96x11008x2048 (prefill batch) --------------------------------------

    comptime M2 = 96
    comptime N2 = 11008
    comptime K2 = 2048

    @parameter
    fn bench_prefill_naive():
        var a = Matrix(M2, K2)
        var b = Matrix(K2, N2)
        var c = Matrix(M2, N2)
        fill(a, 17)
        fill(b, 13)
        matmul_naive(c, a, b)

    @parameter
    fn bench_prefill_tiled():
        var a = Matrix(M2, K2)
        var b = Matrix(K2, N2)
        var c = Matrix(M2, N2)
        fill(a, 17)
        fill(b, 13)
        matmul_tiled(c, a, b)

    @parameter
    fn bench_prefill_simd():
        var a = Matrix(M2, K2)
        var b = Matrix(K2, N2)
        var c = Matrix(M2, N2)
        fill(a, 17)
        fill(b, 13)
        matmul_simd(c, a, b)

    print("--- 96x11008x2048 (prefill) ---")

    var r_naive_2 = std.benchmark.run[bench_prefill_naive]()
    var s_naive_2 = r_naive_2.mean("s")
    print(
        "  naive :",
        r_naive_2.mean("ms"),
        "ms |",
        gflops(M2, N2, K2, s_naive_2),
        "GFLOPS",
    )

    var r_tiled_2 = std.benchmark.run[bench_prefill_tiled]()
    var s_tiled_2 = r_tiled_2.mean("s")
    print(
        "  tiled :",
        r_tiled_2.mean("ms"),
        "ms |",
        gflops(M2, N2, K2, s_tiled_2),
        "GFLOPS",
    )

    var r_simd_2 = std.benchmark.run[bench_prefill_simd]()
    var s_simd_2 = r_simd_2.mean("s")
    print(
        "  simd  :",
        r_simd_2.mean("ms"),
        "ms |",
        gflops(M2, N2, K2, s_simd_2),
        "GFLOPS",
    )

    print("  speedup (naive/tiled):", s_naive_2 / s_tiled_2, "x")
    print("  speedup (naive/simd) :", s_naive_2 / s_simd_2, "x")
    print("  speedup (tiled/simd) :", s_tiled_2 / s_simd_2, "x\n")

    # ---- full reports --------------------------------------------------------

    print("--- full reports ---\n")
    print("1x11008x2048 naive:")
    r_naive_1.print()
    print("\n1x11008x2048 tiled:")
    r_tiled_1.print()
    print("\n1x11008x2048 simd:")
    r_simd_1.print()
    print("\n96x11008x2048 naive:")
    r_naive_2.print()
    print("\n96x11008x2048 tiled:")
    r_tiled_2.print()
    print("\n96x11008x2048 simd:")
    r_simd_2.print()

    var t_end = perf_counter_ns()
    var elapsed_s = Float64(t_end - t_start) / 1e9
    print("\n=== total benchmark wall time:", elapsed_s, "s ===")
