from gemm import matmul
import std.benchmark


fn gflops(n: Int, secs: Float64) -> Float64:
    """GFLOPS for an NxN matmul: 2*N^3 FLOPs."""
    return (2.0 * Float64(n) * Float64(n) * Float64(n)) / (secs * 1e9)


fn main() raises:
    print("=== matmul benchmark ===\n")

    # --- 128x128 ---
    @parameter
    fn bench_128():
        comptime N = 128
        var a = List[Float64](capacity=N * N)
        var b = List[Float64](capacity=N * N)
        var c = List[Float64](capacity=N * N)
        for _ in range(N * N):
            a.append(0.0)
            b.append(0.0)
            c.append(0.0)
        for i in range(N * N):
            a[i] = Float64(i % 17) * 0.1
            b[i] = Float64(i % 13) * 0.1
        matmul(c, a, b, m=N, n=N, k=N)

    var r128 = std.benchmark.run[bench_128]()
    var mean128 = r128.mean("s")
    print("128x128 :", r128.mean("ms"), "ms |", gflops(128, mean128), "GFLOPS")

    # --- 256x256 ---
    @parameter
    fn bench_256():
        comptime N = 256
        var a = List[Float64](capacity=N * N)
        var b = List[Float64](capacity=N * N)
        var c = List[Float64](capacity=N * N)
        for _ in range(N * N):
            a.append(0.0)
            b.append(0.0)
            c.append(0.0)
        for i in range(N * N):
            a[i] = Float64(i % 17) * 0.1
            b[i] = Float64(i % 13) * 0.1
        matmul(c, a, b, m=N, n=N, k=N)

    var r256 = std.benchmark.run[bench_256]()
    var mean256 = r256.mean("s")
    print("256x256 :", r256.mean("ms"), "ms |", gflops(256, mean256), "GFLOPS")

    # --- 512x512 ---
    @parameter
    fn bench_512():
        comptime N = 512
        var a = List[Float64](capacity=N * N)
        var b = List[Float64](capacity=N * N)
        var c = List[Float64](capacity=N * N)
        for _ in range(N * N):
            a.append(0.0)
            b.append(0.0)
            c.append(0.0)
        for i in range(N * N):
            a[i] = Float64(i % 17) * 0.1
            b[i] = Float64(i % 13) * 0.1
        matmul(c, a, b, m=N, n=N, k=N)

    var r512 = std.benchmark.run[bench_512]()
    var mean512 = r512.mean("s")
    print("512x512 :", r512.mean("ms"), "ms |", gflops(512, mean512), "GFLOPS")

    print("\n--- full reports ---\n")
    print("128x128:")
    r128.print()
    print("\n256x256:")
    r256.print()
    print("\n512x512:")
    r512.print()
