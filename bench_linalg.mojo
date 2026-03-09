from matrix import Matrix
from linalg.matmul import matmul as linalg_matmul
from buffer import NDBuffer, DimList
import std.benchmark
from std.collections import List


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn bench_decode() raises:
    """Benchmark linalg.matmul on decode shape: 1x11008x2048."""
    comptime M = 1
    comptime N = 11008
    comptime K = 2048

    print("--- 1x11008x2048 (decode) ---\n")

    var a = Matrix(M, K)
    var b = Matrix(K, N)
    fill(a, 17)
    fill(b, 13)

    var a_buf = NDBuffer[DType.float64, 2, _, DimList(M, K)](
        a.data.unsafe_ptr().bitcast[Float64]()
    )
    var b_buf = NDBuffer[DType.float64, 2, _, DimList(K, N)](
        b.data.unsafe_ptr().bitcast[Float64]()
    )
    var c_data = List[Float64](capacity=M * N)
    for _ in range(M * N):
        c_data.append(0.0)
    var c_buf = NDBuffer[DType.float64, 2, _, DimList(M, N)](
        c_data.unsafe_ptr().bitcast[Float64]()
    )

    linalg_matmul(c_buf, a_buf, b_buf)

    @parameter
    fn bench_fn():
        try:
            linalg_matmul(c_buf, a_buf, b_buf)
        except:
            pass

    var r = std.benchmark.run[bench_fn]()
    var s_mean = r.mean("s")
    var s_min = r.min("s")
    print(
        "  linalg:", r.mean("ms"), "ms |",
        gflops(M, N, K, s_mean), "GFLOPS (mean) |",
        gflops(M, N, K, s_min), "GFLOPS (peak)",
    )
    print("")


fn bench_prefill() raises:
    """Benchmark linalg.matmul on prefill shape: 96x11008x2048."""
    comptime M = 96
    comptime N = 11008
    comptime K = 2048

    print("--- 96x11008x2048 (prefill) ---\n")

    var a = Matrix(M, K)
    var b = Matrix(K, N)
    fill(a, 17)
    fill(b, 13)

    var a_buf = NDBuffer[DType.float64, 2, _, DimList(M, K)](
        a.data.unsafe_ptr().bitcast[Float64]()
    )
    var b_buf = NDBuffer[DType.float64, 2, _, DimList(K, N)](
        b.data.unsafe_ptr().bitcast[Float64]()
    )
    var c_data = List[Float64](capacity=M * N)
    for _ in range(M * N):
        c_data.append(0.0)
    var c_buf = NDBuffer[DType.float64, 2, _, DimList(M, N)](
        c_data.unsafe_ptr().bitcast[Float64]()
    )

    linalg_matmul(c_buf, a_buf, b_buf)

    @parameter
    fn bench_fn():
        try:
            linalg_matmul(c_buf, a_buf, b_buf)
        except:
            pass

    var r = std.benchmark.run[bench_fn]()
    var s_mean = r.mean("s")
    var s_min = r.min("s")
    print(
        "  linalg:", r.mean("ms"), "ms |",
        gflops(M, N, K, s_mean), "GFLOPS (mean) |",
        gflops(M, N, K, s_min), "GFLOPS (peak)",
    )
    print("")


fn main() raises:
    print("=" * 50)
    print("  linalg.matmul benchmark (Mojo stdlib)")
    print("=" * 50)
    print("")

    bench_decode()
    bench_prefill()
