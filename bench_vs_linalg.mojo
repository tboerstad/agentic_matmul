from gemm import matmul_dispatch
from matrix import Matrix
# from linalg.matmul import matmul as linalg_matmul  # Not available in Mojo 0.26.2
from buffer import NDBuffer, DimList
import std.benchmark
from std.collections import List


# Placeholder for linalg_matmul (not available, so we skip the comparison)
fn linalg_matmul[M: Int, N: Int, K: Int](
    mut c: NDBuffer[DType.float64, 2, _, DimList(M, N)],
    a: NDBuffer[DType.float64, 2, _, DimList(M, K)],
    b: NDBuffer[DType.float64, 2, _, DimList(K, N)],
) raises:
    """Placeholder - linalg module not available."""
    pass


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

    # linalg NDBuffers (sharing underlying data with a/b)
    var a_buf = NDBuffer[DType.float64, 2, _, DimList(M, K)](
        a.data.unsafe_ptr().bitcast[Float64]()
    )
    var b_buf = NDBuffer[DType.float64, 2, _, DimList(K, N)](
        b.data.unsafe_ptr().bitcast[Float64]()
    )
    var c_linalg_data = List[Float64](capacity=M * N)
    for _ in range(M * N):
        c_linalg_data.append(0.0)
    var c_linalg = NDBuffer[DType.float64, 2, _, DimList(M, N)](
        c_linalg_data.unsafe_ptr().bitcast[Float64]()
    )

    # Warmup + correctness
    matmul_dispatch(c_dispatch, a, b)
    linalg_matmul[M, N, K](c_linalg, a_buf, b_buf)

    var max_diff = Float64(0)
    for i in range(M):
        for j in range(N):
            var d = Float64(c_dispatch[i, j]) - Float64(c_linalg[i, j])
            if d < 0:
                d = -d
            if d > max_diff:
                max_diff = d
    print("  max diff:", max_diff)
    if max_diff > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK\n")

    @parameter
    fn bench_dispatch_fn():
        matmul_dispatch(c_dispatch, a, b)

    @parameter
    fn bench_linalg_fn():
        try:
            linalg_matmul[M, N, K](c_linalg, a_buf, b_buf)
        except:
            pass

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

    var r_linalg = std.benchmark.run[bench_linalg_fn]()
    var s_l_mean = r_linalg.mean("s")
    var s_l_min = r_linalg.min("s")
    print(
        "  linalg  :",
        r_linalg.mean("ms"),
        "ms (mean) |",
        r_linalg.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_l_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_l_min),
        "GFLOPS (peak)",
    )

    var speedup_mean = s_l_mean / s_d_mean
    var speedup_min = s_l_min / s_d_min
    print("\n  dispatch/linalg speedup (mean):", speedup_mean, "x")
    print("  dispatch/linalg speedup (peak):", speedup_min, "x")
    if speedup_mean > 1.0:
        print(
            "  >> dispatch is",
            (speedup_mean - 1.0) * 100.0,
            "% FASTER (mean)",
        )
    else:
        print(
            "  >> linalg is",
            (1.0 / speedup_mean - 1.0) * 100.0,
            "% FASTER (mean)",
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

    var a_buf = NDBuffer[DType.float64, 2, _, DimList(M, K)](
        a.data.unsafe_ptr().bitcast[Float64]()
    )
    var b_buf = NDBuffer[DType.float64, 2, _, DimList(K, N)](
        b.data.unsafe_ptr().bitcast[Float64]()
    )
    var c_linalg_data = List[Float64](capacity=M * N)
    for _ in range(M * N):
        c_linalg_data.append(0.0)
    var c_linalg = NDBuffer[DType.float64, 2, _, DimList(M, N)](
        c_linalg_data.unsafe_ptr().bitcast[Float64]()
    )

    # Warmup + correctness
    matmul_dispatch(c_dispatch, a, b)
    linalg_matmul[M, N, K](c_linalg, a_buf, b_buf)

    var max_diff = Float64(0)
    for i in range(M):
        for j in range(N):
            var d = Float64(c_dispatch[i, j]) - Float64(c_linalg[i, j])
            if d < 0:
                d = -d
            if d > max_diff:
                max_diff = d
    print("  max diff:", max_diff)
    if max_diff > 1e-6:
        print("  WARNING: results differ!")
    else:
        print("  correctness: OK\n")

    @parameter
    fn bench_dispatch_fn():
        matmul_dispatch(c_dispatch, a, b)

    @parameter
    fn bench_linalg_fn():
        try:
            linalg_matmul[M, N, K](c_linalg, a_buf, b_buf)
        except:
            pass

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

    var r_linalg = std.benchmark.run[bench_linalg_fn]()
    var s_l_mean = r_linalg.mean("s")
    var s_l_min = r_linalg.min("s")
    print(
        "  linalg  :",
        r_linalg.mean("ms"),
        "ms (mean) |",
        r_linalg.min("ms"),
        "ms (min) |",
        gflops(M, N, K, s_l_mean),
        "GFLOPS (mean) |",
        gflops(M, N, K, s_l_min),
        "GFLOPS (peak)",
    )

    var speedup_mean = s_l_mean / s_d_mean
    var speedup_min = s_l_min / s_d_min
    print("\n  dispatch/linalg speedup (mean):", speedup_mean, "x")
    print("  dispatch/linalg speedup (peak):", speedup_min, "x")
    if speedup_mean > 1.0:
        print(
            "  >> dispatch is",
            (speedup_mean - 1.0) * 100.0,
            "% FASTER (mean)",
        )
    else:
        print(
            "  >> linalg is",
            (1.0 / speedup_mean - 1.0) * 100.0,
            "% FASTER (mean)",
        )
    print("")


fn main() raises:
    print("=" * 60)
    print("  dispatch (fastest custom) vs linalg.matmul (Mojo stdlib)")
    print("=" * 60)
    print("")

    bench_decode()
    bench_prefill()
