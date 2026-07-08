# Correctness check for the small-box dispatch band in f32/bf16.
#
# verify_dispatch.mojo runs float64 only, and the small-box gates are
# byte-based, so the f32 routes differ from the f64 ones on the same shapes
# (half the bytes per element, double the NR granularity; see DESIGN.md
# "Small-box f32 routing"). This file exercises the f32/bf16 band those gates
# actually produce: the no-pack keepers, the tail-heavy square-ish evictions,
# and the 2D-grid picks. It also carries the bf16 route coverage, since bf16
# rides the f32-shaped dispatch (it computes in f32; see _compute_dtype in
# gemm.mojo). The reference is a float64 naive matmul over the same
# dtype-rounded inputs, with a relative tolerance scaled for the
# lower-precision storage.
from gemm import matmul_dispatch
from matrix import Matrix
from std.math import abs


def fill[dtype: DType](mut m: Matrix[dtype], seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[dtype]((i * m.cols + j * 7 + 3) % seed) * 0.1 - 0.5


def check[dtype: DType](m: Int, n: Int, k: Int, tol: Float64) raises:
    var a = Matrix[dtype](m, k)
    var b = Matrix[dtype](k, n)
    var c = Matrix[dtype](m, n)
    fill(a, 17)
    fill(b, 13)
    matmul_dispatch(c, a, b)

    # Naive f64 reference over the same (dtype-rounded) inputs.
    var max_rel = Float64(0)
    for i in range(m):
        for j in range(n):
            var dot = Float64(0)
            for p in range(k):
                dot += Float64(a[i, p]) * Float64(b[p, j])
            var e = abs(Float64(c[i, j]) - dot) / max(abs(dot), 1.0)
            if e > max_rel:
                max_rel = e
    var ok = max_rel < tol
    print("  dtype", dtype, "M=", m, "N=", n, "K=", k,
          "| max_rel=", max_rel, "|", "OK" if ok else "FAIL")
    if not ok:
        raise Error("mismatch")


def main() raises:
    print("=== correctness: f32/bf16 small-box routes vs f64 naive ===")
    # The rerouted band: tail-heavy no-pack evictions and 2D-grid picks,
    # plus the shapes that must keep their old route.
    var ms = [256, 288, 300, 320, 352, 512, 256, 300, 306]
    var ns = [256, 288, 300, 320, 352, 128, 128, 300, 300]
    var ks = [256, 288, 300, 320, 352, 512, 512, 999, 306]
    for s in range(len(ms)):
        check[DType.float32](ms[s], ns[s], ks[s], 1e-4)
    # bf16 stores in bf16 and computes in f32 (SOL.md idea 2): the packed
    # panels widen at pack time, the no-pack/GEMV paths widen on load, and C
    # rounds to bf16 once per tile store. The remaining error vs the f64
    # reference is that single 8-mantissa-bit C rounding (~2^-9 relative),
    # so the tolerance is 0.02 rather than f32's 1e-4. Cover every dispatch
    # route: tiny serial, decode GEMV, small batch, thin-N, small box,
    # narrow-N, square-ish (incl. the 2D grid), wide-N, and tall-K, with
    # M/N/K remainders and multi-k-panel accumulation.
    check[DType.bfloat16](33, 47, 51, 0.02)     # tiny serial (odd dims)
    check[DType.bfloat16](1, 2048, 512, 0.02)   # decode GEMV, n % nw chunks
    check[DType.bfloat16](1, 999, 777, 0.02)    # decode GEMV, odd n and k
    check[DType.bfloat16](4, 2048, 512, 0.02)   # small batch (MR = M = 4)
    check[DType.bfloat16](128, 16, 512, 0.02)   # thin-N M-parallel no-pack
    check[DType.bfloat16](128, 50, 512, 0.02)   # thin-N with scalar N tail
    check[DType.bfloat16](256, 128, 512, 0.02)  # small box, no-pack keeper
    check[DType.bfloat16](300, 300, 300, 0.02)  # 2D-grid pick, M/N tails
    check[DType.bfloat16](320, 320, 320, 0.02)  # 2D grid, clean columns
    check[DType.bfloat16](64, 100, 300, 0.02)   # narrow-N NR=32 (n % NR)
    check[DType.bfloat16](512, 480, 512, 0.02)  # square-ish pack-B-only
    check[DType.bfloat16](513, 479, 600, 0.02)  # square-ish, M+N tails
    check[DType.bfloat16](96, 2048, 768, 0.02)  # wide-N packed prefill
    check[DType.bfloat16](257, 2048, 1100, 0.02)  # wide-N SHARED_A, k-panels
    check[DType.bfloat16](257, 512, 2048, 0.02)   # tall-K SHARED_A, M tail
    check[DType.bfloat16](512, 512, 2048, 0.02)   # tall-K big-KC panel
    # AMX tile-kernel gate shapes (m % 32 == n % 16 == k % 32 == 0): on an
    # AMX part these run tdpbf16ps; elsewhere they fall through to the
    # AVX-512 routes above, so the same checks cover both machines.
    check[DType.bfloat16](64, 2064, 64, 0.02)   # AMX, 16-column N tail
    check[DType.bfloat16](32, 16, 1024, 0.02)   # AMX, single 16-column panel
    check[DType.bfloat16](96, 2048, 2048, 0.02)  # AMX, prefill-like
    print("all passed")
