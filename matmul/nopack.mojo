"""No-pack kernels: serial (tiny) and M-parallel thin-N / small box.

Both read A and B straight from source. For a tiny or cache-resident shape
the data already fits L1/L2, so explicit packing buys nothing and the packed
kernel's packing + thread-launch overhead would dominate. Both are the same
per-block routine, `_nopack_rows`; only the driver differs (a plain loop vs
`parallelize`), so they are bit-identical to each other and to the packed
paths.
"""

from matmul.matrix import Matrix
from matmul.microkernel import (
    RegisterTile,
    compute_dtype,
    load_a_col,
    load_b_row,
)
from matmul.tile import Tile
from std.algorithm.functional import parallelize, vectorize
from std.math import ceildiv
from std.sys import num_physical_cores, simd_width_of


@always_inline
def _nopack_tail_row[
    dtype: DType, cdtype: DType, NELTS: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: ImmutOrigin,
](
    c_row: UnsafePointer[Scalar[dtype], c_org],
    a_row: UnsafePointer[Scalar[dtype], a_org],
    b: Tile[dtype, b_org],
    jr: Int,
):
    """N-remainder columns (n % NR) of one C row: NELTS-wide chunks with an
    automatic scalar tail, accumulating in the compute dtype. A top-level def
    so the inner closure can capture its operands as function-scope bindings
    (see `_decode_fma_chunk` in gemv.mojo)."""
    def tail[width: Int](jj: Int) {read c_row, read a_row, read b, read jr}:
        var acc = SIMD[cdtype, width](0)
        for p in range(b.rows):
            acc = SIMD[cdtype, width](a_row[p].cast[cdtype]()).fma(
                b.addr(p, jr).load[width=width](offset=jj).cast[cdtype](), acc
            )
        c_row.store(offset=jr + jj, val=acc.cast[dtype]())
    vectorize[NELTS](b.cols - jr, tail)


@always_inline
def _nopack_rows[
    dtype: DType, cdtype: DType, MR: Int, NR_VECS: Int, NELTS: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: ImmutOrigin,
](
    c: Tile[dtype, c_org],
    a: Tile[dtype, a_org],
    b: Tile[dtype, b_org],
    i: Int,
    r: Int,
):
    """Rows [i, i+r) of C across the full N, reading A and B straight from
    source (widened on load to the compute dtype). A full MR-row block
    (r == MR) uses the comptime-unrolled MR tile (each B-load reused across
    MR rows); a tail block (m % MR) computes one row at a time with the same
    NR-wide SIMD."""
    comptime NR = NR_VECS * NELTS
    var n = c.cols
    var k = a.cols

    if r == MR:
        var j = 0
        while j + NR <= n:
            var tile = RegisterTile[cdtype, MR, NR_VECS, NELTS]()
            for p in range(k):
                tile.rank1_update(
                    load_a_col[MR, cdtype](a.addr(i, p), a.stride),
                    load_b_row[NR_VECS, NELTS, cdtype](b.addr(p, j)),
                )
            tile.store(c.sub(i, j))
            j += NR
    else:
        for ii in range(i, i + r):
            var j = 0
            while j + NR <= n:
                var tile = RegisterTile[cdtype, 1, NR_VECS, NELTS]()
                for p in range(k):
                    tile.rank1_update(
                        load_a_col[1, cdtype](a.addr(ii, p), a.stride),
                        load_b_row[NR_VECS, NELTS, cdtype](b.addr(p, j)),
                    )
                tile.store(c.sub(ii, j))
                j += NR

    var jr = (n // NR) * NR
    if jr < n:
        for ii in range(i, i + r):
            _nopack_tail_row[dtype, cdtype, NELTS](c.row(ii), a.row(ii), b, jr)


def serial_gemm[
    dtype: DType, MR: Int, NR_VECS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Serial register-tiled GEMM for tiny shapes (no threads, no packing).

    Below the dispatch's tiny cutoff the parallel kernels' fixed cost (thread
    launch + per-worker buffers + packing) dwarfs the compute, running
    10-100x slower than this plain serial loop. MR=6, NR_VECS=2 measured
    best."""
    comptime CDT = compute_dtype[dtype]()
    comptime NELTS = simd_width_of[CDT]()
    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    for i in range(0, m, MR):
        _nopack_rows[dtype, CDT, MR, NR_VECS, NELTS](
            c_view, a_view, b_view, i, min(MR, m - i)
        )


def nopack_gemm[
    dtype: DType, MR: Int, NR_VECS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """M-parallel register-tiled GEMM with NO packing.

    Built for two regimes the N-parallel packed kernel handles badly:

      * thin-N (small N, large M*K): the packed kernel parallelizes only over
        N, so a thin N starves the cores. Here the work is along M.
      * small M-dominant box whose B fits L2: the packed kernel's packing +
        per-worker buffers + launch overhead dwarfs the compute when the whole
        problem is cache-resident.

    Every core owns a band of C's rows and sweeps the full N, reading A/B
    straight from source."""
    comptime CDT = compute_dtype[dtype]()
    comptime NELTS = simd_width_of[CDT]()
    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows

    def worker(blk: Int) {read c_view, read a_view, read b_view, read m}:
        var i = blk * MR
        _nopack_rows[dtype, CDT, MR, NR_VECS, NELTS](
            c_view, a_view, b_view, i, min(MR, m - i)
        )

    parallelize(worker, ceildiv(m, MR), num_physical_cores())
