"""The decode GEMV: the M = 1 kernel, parallelized over columns of C."""

from matmul.matrix import Matrix
from matmul.microkernel import compute_dtype
from std.algorithm.functional import parallelize, vectorize
from std.math import ceildiv
from std.memory import memset_zero
from std.memory.unsafe_pointer import alloc
from std.sys import num_physical_cores, simd_width_of
from std.sys.intrinsics import prefetch, PrefetchOptions


@always_inline
def _decode_fma_chunk[
    dtype: DType, cdtype: DType, KU: Int, NELTS: Int,
    c_origin: MutOrigin, a_origin: ImmutOrigin, b_origin: ImmutOrigin,
](
    ci: UnsafePointer[Scalar[cdtype], c_origin],
    ai: UnsafePointer[Scalar[dtype], a_origin],
    b_col: UnsafePointer[Scalar[dtype], b_origin],
    p: Int,
    n: Int,
    chunk: Int,
):
    """KU consecutive K-steps of the GEMV, vectorized across the worker's
    column chunk; the tail loop reuses it at KU=1. The accumulator row `ci`
    lives in the compute dtype; A and B load in the storage dtype and widen
    (a no-op when the two match).

    A top-level def so the inner closure can capture `p` and the pointers as
    function-scope bindings (a closure nested inside another closure cannot
    capture for/while-loop-body variables in current Mojo)."""
    def do_fma[width: Int](j: Int) {read ci, read ai, read b_col, read p, read n}:
        var acc = ci.load[width=width](offset=j)
        comptime for ku in range(KU):
            # Prefetch the same columns of the next KU-block of B rows: the KU
            # streams are n elements apart, too far for the HW prefetcher. May
            # reach past the end of B on the last block; prefetch is
            # architecturally non-faulting, so that is safe.
            prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                b_col + (p + ku + KU) * n + j
            )
            var a_broadcast = SIMD[cdtype, width](ai[p + ku].cast[cdtype]())
            var b_vec = (b_col + (p + ku) * n).load[width=width, invariant=True](offset=j).cast[cdtype]()
            acc = a_broadcast.fma(b_vec, acc)
        ci.store(offset=j, val=acc)
    vectorize[NELTS, unroll_factor=4](chunk, do_fma)


@always_inline
def _cast_chunk[
    sdtype: DType, cdtype: DType, NELTS: Int,
    d_org: MutOrigin, s_org: MutOrigin,
](
    dst: UnsafePointer[Scalar[sdtype], d_org],
    src: UnsafePointer[Scalar[cdtype], s_org],
    count: Int,
):
    """Narrow `count` elements from the compute-dtype staging row into the
    storage-dtype C row: the bf16 GEMV's single downconvert per finished row.
    A top-level def for the same closure-capture reason as
    `_decode_fma_chunk`."""
    def conv[width: Int](j: Int) {read dst, read src}:
        dst.store(offset=j, val=src.load[width=width](offset=j).cast[sdtype]())
    vectorize[NELTS](count, conv)


def decode_gemv[
    dtype: DType, //, KU: Int = 8,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """J-parallel GEMV optimized for decode (small M, large K x N).

    Each worker owns a disjoint column chunk of C and sweeps all K rows. The
    per-k working set of B plus C columns fits L1, so B streams past exactly
    once and no reduction is needed.

    When the compute dtype differs from storage (bf16), the accumulator rows
    live in a compute-dtype staging buffer: accumulating straight into a bf16
    C would round the partial sum to 8 mantissa bits every KU steps. B still
    streams in bf16 (half the bytes of f32, the whole point for a
    bandwidth-bound decode) and widens in registers; each C row is narrowed
    once when its K-sweep finishes."""
    comptime assert KU > 0, "KU must be positive"
    comptime assert dtype.is_floating_point(), "GEMV requires floating-point dtype"
    comptime CDT = compute_dtype[dtype]()
    comptime NELTS = simd_width_of[CDT]()

    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols
    var nw = num_physical_cores()

    # min size-1 alloc when storage == compute (C accumulated in place).
    var stage_len = 1
    comptime if CDT != dtype:
        stage_len = m * n
    var c_stage = alloc[Scalar[CDT]](stage_len)

    comptime if CDT != dtype:
        memset_zero(c_stage, m * n)
    else:
        memset_zero(c_view.ptr, m * n)

    def worker(wid: Int) {read c_view, read a_view, read b_view, mut c_stage, read m, read n, read k, read nw}:
        var cols_per = ceildiv(n, nw)
        var j0 = wid * cols_per
        var j1 = min(j0 + cols_per, n)
        var chunk = j1 - j0
        if chunk <= 0:
            return

        var b_col = b_view.addr(0, j0)  # base pointer into worker's column chunk
        var k_main = (k // KU) * KU

        for i in range(m):
            var ai = a_view.row(i)
            var p = 0

            comptime if CDT != dtype:
                var ci = c_stage + i * n + j0
                while p < k_main:
                    _decode_fma_chunk[dtype=dtype, KU=KU, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                    p += KU
                while p < k:
                    _decode_fma_chunk[dtype=dtype, KU=1, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                    p += 1
                _cast_chunk[dtype, CDT, NELTS](c_view.addr(i, j0), ci, chunk)
            else:
                var ci = c_view.addr(i, j0)
                while p < k_main:
                    _decode_fma_chunk[dtype=dtype, KU=KU, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                    p += KU
                while p < k:
                    _decode_fma_chunk[dtype=dtype, KU=1, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                    p += 1

    parallelize(worker, nw, nw)
    c_stage.free()
