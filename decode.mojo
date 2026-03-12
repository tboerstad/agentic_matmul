from matrix import Matrix
from std.algorithm.functional import parallelize, vectorize
from std.math import ceildiv, fma
from std.memory import memset_zero
from std.sys import num_physical_cores, simd_width_of


fn _decode_gemv[
    dtype: DType, KU: Int = 4,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # J-parallel GEMV optimized for decode (small M, large K×N).
    #
    # Each worker owns a disjoint column chunk of C and sweeps all K rows.
    # Per-k working set ≈ (N/nw)*8 bytes of B + same for C, which fits L1
    # (e.g. 2752×8 = 21 KB for N=11008, nw=4).  No reduction needed.
    comptime NELTS = simd_width_of[dtype]()

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr().as_noalias_ptr()
    var a_ptr = a.data.unsafe_ptr().as_noalias_ptr()
    var b_ptr = b.data.unsafe_ptr().as_noalias_ptr()
    var nw = num_physical_cores()

    memset_zero(c_ptr, m * n)

    fn worker(wid: Int) capturing:
        var cols_per = ceildiv(n, nw)
        var j0 = wid * cols_per
        var j1 = min(j0 + cols_per, n)
        var chunk = j1 - j0
        if chunk <= 0:
            return

        var b_col = b_ptr + j0  # base pointer into worker's column chunk
        var k_main = (k // KU) * KU

        for i in range(m):
            var ci = c_ptr + i * n + j0
            var ai = a_ptr + i * k
            var p = 0

            while p < k_main:
                fn do_fma[width: Int](j: Int) unified {mut}:
                    var acc = ci.load[width=width](offset=j)
                    comptime for ku in range(KU):
                        acc = fma(
                            SIMD[dtype, width](ai[p + ku]),
                            (b_col + (p + ku) * n).load[width=width](offset=j),
                            acc,
                        )
                    ci.store(offset=j, val=acc)

                vectorize[NELTS, unroll_factor=4](chunk, do_fma)
                p += KU

            while p < k:
                fn do_fma1[width: Int](j: Int) unified {mut}:
                    ci.store(
                        offset=j,
                        val=fma(
                            SIMD[dtype, width](ai[p]),
                            (b_col + p * n).load[width=width](offset=j),
                            ci.load[width=width](offset=j),
                        ),
                    )

                vectorize[NELTS, unroll_factor=4](chunk, do_fma1)
                p += 1

    parallelize[worker](nw, nw)


fn matmul_decode[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  optimized for decode shapes (small M).
    # Uses j-parallel GEMV: each worker owns a column chunk that fits L1.
    _decode_gemv[dtype](c, a, b)
