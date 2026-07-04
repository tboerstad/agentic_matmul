# Correctness check for the small-box dispatch band in f32/bf16.
#
# verify_dispatch.mojo runs float64 only, and the small-box gates are
# byte-based, so the f32 routes differ from the f64 ones on the same shapes
# (half the bytes per element, double the NR granularity; see DESIGN.md
# "Small-box f32 routing"). This file exercises the f32/bf16 band those gates
# actually produce: the no-pack keepers, the tail-heavy square-ish evictions,
# and the 2D-grid picks. The reference is a float64 naive matmul over the
# same dtype-rounded inputs, with a relative tolerance scaled for the
# lower-precision accumulation.
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
    # bf16 accumulates in bf16: loose tolerance, just guard against a wrong
    # route producing garbage (wrong offsets, unwritten tiles).
    check[DType.bfloat16](300, 300, 300, 0.15)
    check[DType.bfloat16](320, 320, 320, 0.15)
    print("all passed")
