# Correctness check for matmul_dispatch against a naive reference.
#
# Exercises every dispatch branch and the packed micro-kernel's edge cases
# (full MR panels, NR remainders, and M tails) across both projection
# orientations. test_gemm.mojo only tests a local reference matmul, not the
# real kernels, so this is the actual coverage for matmul_dispatch.
from gemm import matmul_dispatch
from matrix import Matrix
from std.math import abs


def fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j * 7 + 3) % seed) * 0.1 - 0.5


def naive[dtype: DType, //](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    for i in range(a.rows):
        for j in range(c.cols):
            var dot = Scalar[dtype](0)
            for p in range(a.cols):
                dot += a[i, p] * b[p, j]
            c[i, j] = dot


def check(m: Int, n: Int, k: Int) raises:
    var a = Matrix(m, k)
    var b = Matrix(k, n)
    var c = Matrix(m, n)
    var c_ref = Matrix(m, n)
    fill(a, 17)
    fill(b, 13)
    matmul_dispatch(c, a, b)
    naive(c_ref, a, b)
    var max_err = Float64(0)
    for i in range(m):
        for j in range(n):
            var e = abs(c[i, j] - c_ref[i, j])
            if e > max_err:
                max_err = e
    var ok = max_err < 1e-7
    print("  M=", m, "N=", n, "K=", k, "| max_err=", max_err,
          "|", "OK" if ok else "FAIL")
    if not ok:
        raise Error("mismatch")


def main() raises:
    print("=== correctness: dispatch vs naive ===")
    # Exercise full MR panels, NR remainders, and M tails for both orientations.
    # M>256 hits the new wide-N 6x32 branch; 257/300 also stress its M-tail.
    var ms = [2, 3, 4, 5, 6, 7, 13, 64, 65, 70, 96, 128, 130, 256, 257, 300, 384, 512]
    for i in range(len(ms)):
        check(ms[i], 2048, 768)   # wide-N (N>K), incl. new M>256 6x32 branch
    for i in range(len(ms)):
        check(ms[i], 770, 512)    # wide-N, N not a multiple of TILE_N (remainder)
    # M>256 wide-N with K>1024 exercises the KC=1024 branch's multi-k-panel
    # accumulation (2 panels: 1024 + partial) + N remainder + M tail together.
    check(300, 1100, 1100)
    check(384, 1100, 1100)
    # Small-N branch (N <= 192 -> narrow NR=16/TILE_N=16 tile). Cover N values
    # both divisible and not divisible by 16 (NR remainder) across the M tail.
    for i in range(len(ms)):
        check(ms[i], 64, 256)     # N=64, multiple of 16
    for i in range(len(ms)):
        check(ms[i], 96, 512)     # N=96, multiple of 16
    for i in range(len(ms)):
        check(ms[i], 100, 300)    # N=100, NOT a multiple of 16 (NR remainder)
    # Thin-N, tall-M branch (N <= 64, M >= 64 -> M-parallel _thin_n_gemm). Cover
    # the NR_VECS=1 path (N < 2*NELTS), N at/over a multiple of 16, an N that is
    # NOT a multiple of NR (scalar N-remainder strip), and M values that exercise
    # the MR=6 M-tail block — all across a K big enough to clear the wMNK cutoff.
    for i in range(len(ms)):
        check(ms[i], 16, 512)     # N=16, exactly one NR=16 panel
    for i in range(len(ms)):
        check(ms[i], 8, 512)      # N=8 < 2*NELTS -> NR_VECS=1 (NR=8) panel
    for i in range(len(ms)):
        check(ms[i], 50, 512)     # N=50, NOT a multiple of 16 (NR remainder)
    check(64, 13, 700)            # M tail (64%6) + N remainder + NR_VECS=1
    check(128, 40, 1024)          # N=40 (2 panels + 8-col remainder)
    # Square-ish branch (N > 192 AND N <= M -> 6x32 TILE_N=32 KC=512 tile).
    # Cover square M=N=K, a narrow-N-tall-M corner, an N that is NOT a multiple
    # of TILE_N=32 (N-remainder), and M values (257/300/513) that hit the MR=6
    # M-tail, all with K big enough to span multiple KC=512 k-panels.
    check(256, 256, 256)
    check(512, 512, 512)
    check(512, 256, 512)
    check(300, 256, 256)          # M tail (300%6) + square-ish
    check(257, 256, 1100)         # M tail + 3 k-panels (512+512+76)
    check(512, 200, 768)          # N=200, NOT a multiple of 32 (N remainder)
    check(513, 224, 600)          # M tail + N=224 + multi-k-panel
    # SHARED_A large-M gate (m>=192 packs A once) on the wide-N (N>K) and tall-K
    # (N<K) branches. Cover the gate boundary (191 off / 192 on) and large M on
    # both orientations, with K spanning multiple k-panels so the shared single
    # pack-of-A's [i-panel][k][MR] layout + pc offset is exercised across panels.
    check(191, 2048, 1100)        # wide-N, just BELOW gate (per-worker pack)
    check(192, 2048, 1100)        # wide-N, AT gate (SHARED_A on)
    check(256, 2048, 1100)        # wide-N, SHARED_A, multi-k-panel + M tail
    check(512, 4096, 2048)        # wide-N, SHARED_A, cache-aware big-KC panel
    check(191, 512, 2048)         # tall-K (N<K), just BELOW gate
    check(192, 512, 2048)         # tall-K, AT gate (SHARED_A on)
    check(257, 512, 2048)         # tall-K, SHARED_A, M tail + multi-k-panel
    check(512, 512, 2048)         # tall-K, SHARED_A, cache-aware big-KC panel
    # Tiny total work (wMNK < 2^19) -> the serial _matmul_small fast path.
    # Cover its edges: 1xK/Mx1/1x1 degenerate dims, N below NR (=16) so only
    # the N-remainder tail runs, N straddling NR, and fully odd dims that hit
    # the M-remainder, full-panel, and N-tail paths at once.
    check(1, 1, 1)
    check(1, 64, 64)
    check(64, 1, 64)
    check(8, 8, 8)
    check(7, 13, 11)
    check(17, 23, 13)
    check(33, 47, 51)
    check(50, 50, 50)
    check(80, 80, 80)
    print("all passed")
