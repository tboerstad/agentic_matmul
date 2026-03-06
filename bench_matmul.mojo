from gemm import matmul
from matrix import Matrix
import std.benchmark
from time import perf_counter_ns


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    """GFLOPS for an MxNxK matmul: 2*M*N*K FLOPs."""
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn gflops(n: Int, secs: Float64) -> Float64:
    """GFLOPS for an NxN matmul: 2*N^3 FLOPs."""
    return gflops(n, n, n, secs)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn main() raises:
    var t_start = perf_counter_ns()
    print("=== matmul benchmark ===\n")

    # --- 128x128 ---
    @parameter
    fn bench_128():
        comptime N = 128
        var a = Matrix(N, N)
        var b = Matrix(N, N)
        var c = Matrix(N, N)
        fill(a, 17)
        fill(b, 13)
        matmul(c, a, b)

    var r128 = std.benchmark.run[bench_128]()
    var mean128 = r128.mean("s")
    print("128x128 :", r128.mean("ms"), "ms |", gflops(128, mean128), "GFLOPS")

    # --- 256x256 ---
    @parameter
    fn bench_256():
        comptime N = 256
        var a = Matrix(N, N)
        var b = Matrix(N, N)
        var c = Matrix(N, N)
        fill(a, 17)
        fill(b, 13)
        matmul(c, a, b)

    var r256 = std.benchmark.run[bench_256]()
    var mean256 = r256.mean("s")
    print("256x256 :", r256.mean("ms"), "ms |", gflops(256, mean256), "GFLOPS")

    # --- 512x512 ---
    @parameter
    fn bench_512():
        comptime N = 512
        var a = Matrix(N, N)
        var b = Matrix(N, N)
        var c = Matrix(N, N)
        fill(a, 17)
        fill(b, 13)
        matmul(c, a, b)

    var r512 = std.benchmark.run[bench_512]()
    var mean512 = r512.mean("s")
    print("512x512 :", r512.mean("ms"), "ms |", gflops(512, mean512), "GFLOPS")

    # --- 512x11008x2048 (Qwen 2.5 VL 3B gate/up projection shape) ---
    @parameter
    fn bench_qwen():
        comptime M = 512
        comptime N = 11008
        comptime K = 2048
        var a = Matrix(M, K)
        var b = Matrix(K, N)
        var c = Matrix(M, N)
        fill(a, 17)
        fill(b, 13)
        matmul(c, a, b)

    var rq = std.benchmark.run[bench_qwen]()
    var meanq = rq.mean("s")
    print(
        "512x11008x2048 :",
        rq.mean("ms"),
        "ms |",
        gflops(512, 11008, 2048, meanq),
        "GFLOPS",
    )

    print("\n--- full reports ---\n")
    print("128x128:")
    r128.print()
    print("\n256x256:")
    r256.print()
    print("\n512x512:")
    r512.print()
    print("\n512x11008x2048:")
    rq.print()

    var t_end = perf_counter_ns()
    var elapsed_s = Float64(t_end - t_start) / 1e9
    print("\n=== total benchmark wall time:", elapsed_s, "s ===")
