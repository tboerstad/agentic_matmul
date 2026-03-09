from gemm import _prefill_gemm, _prefill_gemm_v2
from matrix import Matrix
import std.benchmark
from std.sys import simd_width_of


fn gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn main() raises:
    print("=== Fine-tune MR=6 NR=32 for prefill 96x11008x2048 ===\n")

    comptime M = 96
    comptime N = 11008
    comptime K = 2048
    comptime NELTS = simd_width_of[DType.float64]()
    comptime NR = 4 * NELTS  # 32

    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    # KC sweep with TILE_N=64
    @parameter
    fn kc256():
        _prefill_gemm[DType.float64, 6, NR, 256, 8, 64, 256](c, a, b)
    var r1 = std.benchmark.run[kc256]()
    print("KC=256  TN=64  : mean", gflops(M, N, K, r1.mean("s")), "| peak", gflops(M, N, K, r1.min("s")), "GFLOPS")

    @parameter
    fn kc384():
        _prefill_gemm[DType.float64, 6, NR, 384, 8, 64, 256](c, a, b)
    var r2 = std.benchmark.run[kc384]()
    print("KC=384  TN=64  : mean", gflops(M, N, K, r2.mean("s")), "| peak", gflops(M, N, K, r2.min("s")), "GFLOPS")

    @parameter
    fn kc512():
        _prefill_gemm[DType.float64, 6, NR, 512, 8, 64, 256](c, a, b)
    var r3 = std.benchmark.run[kc512]()
    print("KC=512  TN=64  : mean", gflops(M, N, K, r3.mean("s")), "| peak", gflops(M, N, K, r3.min("s")), "GFLOPS")

    # TILE_N sweep with KC=512
    @parameter
    fn tn32():
        _prefill_gemm[DType.float64, 6, NR, 512, 8, 32, 256](c, a, b)
    var r4 = std.benchmark.run[tn32]()
    print("KC=512  TN=32  : mean", gflops(M, N, K, r4.mean("s")), "| peak", gflops(M, N, K, r4.min("s")), "GFLOPS")

    @parameter
    fn tn96():
        _prefill_gemm[DType.float64, 6, NR, 512, 8, 96, 256](c, a, b)
    var r5 = std.benchmark.run[tn96]()
    print("KC=512  TN=96  : mean", gflops(M, N, K, r5.mean("s")), "| peak", gflops(M, N, K, r5.min("s")), "GFLOPS")

    @parameter
    fn tn128():
        _prefill_gemm[DType.float64, 6, NR, 512, 8, 128, 256](c, a, b)
    var r6 = std.benchmark.run[tn128]()
    print("KC=512  TN=128 : mean", gflops(M, N, K, r6.mean("s")), "| peak", gflops(M, N, K, r6.min("s")), "GFLOPS")

    # Best combo candidates
    @parameter
    fn kc384_tn96():
        _prefill_gemm[DType.float64, 6, NR, 384, 8, 96, 256](c, a, b)
    var r7 = std.benchmark.run[kc384_tn96]()
    print("KC=384  TN=96  : mean", gflops(M, N, K, r7.mean("s")), "| peak", gflops(M, N, K, r7.min("s")), "GFLOPS")

    @parameter
    fn kc384_tn128():
        _prefill_gemm[DType.float64, 6, NR, 384, 8, 128, 256](c, a, b)
    var r8 = std.benchmark.run[kc384_tn128]()
    print("KC=384  TN=128 : mean", gflops(M, N, K, r8.mean("s")), "| peak", gflops(M, N, K, r8.min("s")), "GFLOPS")

    # === v2 kernel ===
    print("\n=== v2 prefill kernel (MR=6 NR=24) ===\n")
    comptime NR_V2 = 3 * NELTS  # 24

    @parameter
    fn v2_kc256():
        _prefill_gemm_v2[DType.float64, 6, NR_V2, 256, 8, 64, 256](c, a, b)
    var rv1 = std.benchmark.run[v2_kc256]()
    print("v2 KC=256  TN=64  : mean", gflops(M, N, K, rv1.mean("s")), "| peak", gflops(M, N, K, rv1.min("s")), "GFLOPS")

    @parameter
    fn v2_kc384():
        _prefill_gemm_v2[DType.float64, 6, NR_V2, 384, 8, 64, 256](c, a, b)
    var rv2 = std.benchmark.run[v2_kc384]()
    print("v2 KC=384  TN=64  : mean", gflops(M, N, K, rv2.mean("s")), "| peak", gflops(M, N, K, rv2.min("s")), "GFLOPS")

    @parameter
    fn v2_kc512():
        _prefill_gemm_v2[DType.float64, 6, NR_V2, 512, 8, 64, 256](c, a, b)
    var rv3 = std.benchmark.run[v2_kc512]()
    print("v2 KC=512  TN=64  : mean", gflops(M, N, K, rv3.mean("s")), "| peak", gflops(M, N, K, rv3.min("s")), "GFLOPS")

    @parameter
    fn v2_kc384_tn96():
        _prefill_gemm_v2[DType.float64, 6, NR_V2, 384, 8, 96, 256](c, a, b)
    var rv4 = std.benchmark.run[v2_kc384_tn96]()
    print("v2 KC=384  TN=96  : mean", gflops(M, N, K, rv4.mean("s")), "| peak", gflops(M, N, K, rv4.min("s")), "GFLOPS")

    @parameter
    fn v2_kc512_tn96():
        _prefill_gemm_v2[DType.float64, 6, NR_V2, 512, 8, 96, 256](c, a, b)
    var rv5 = std.benchmark.run[v2_kc512_tn96]()
    print("v2 KC=512  TN=96  : mean", gflops(M, N, K, rv5.mean("s")), "| peak", gflops(M, N, K, rv5.min("s")), "GFLOPS")

    print("\nDone.")
