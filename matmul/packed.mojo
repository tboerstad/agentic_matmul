"""The packed GEMM drivers: the workhorse prefill kernel and the 2D grid.

`packed_gemm` packs A and B into cache-friendly panels and runs the
register-tile micro-kernel over them, parallelized across N (j-tiles).
`pack_b_only_2d` is its single-K-panel sibling for shapes whose column count
splits the cores unevenly: it packs only B and distributes a 2D
(column, MR-row-block) tile grid across workers. The dispatch in
`dispatch.mojo` decides which shapes run which driver.
"""

from matmul.matrix import Matrix
from matmul.microkernel import (
    compute_dtype,
    full_microkernel,
    masked_microkernel,
    partial_n_microkernel,
)
from matmul.tile import Tile
from std.algorithm.functional import parallelize
from std.math import ceildiv
from std.memory.unsafe_pointer import alloc
from std.sys import num_physical_cores, simd_width_of
from std.sys.intrinsics import prefetch, PrefetchOptions


@always_inline
def _pack_a_panel[
    dtype: DType, MR: Int, a_org: ImmutOrigin, ap_org: MutOrigin,
    pdtype: DType,
](
    a: Tile[dtype, a_org],
    dst: UnsafePointer[Scalar[pdtype], ap_org],
    i0: Int,
    pc: Int,
    kc: Int,
):
    """Pack MR rows of A (top-left (i0, pc), kc deep) as [k][MR], so each
    K-step's MR A scalars are contiguous. The pack widens to the compute
    dtype in the same touch (bf16 source, f32 panel; a no-op cast when
    they match)."""
    for pk in range(kc):
        comptime for mr in range(MR):
            dst[pk * MR + mr] = a.row(i0 + mr)[pc + pk].cast[pdtype]()


@always_inline
def _pack_b_slab[
    dtype: DType, NR: Int, NR_VECS: Int, NELTS: Int,
    b_org: ImmutOrigin, bp_org: MutOrigin, pdtype: DType,
](
    b: Tile[dtype, b_org],
    bp: UnsafePointer[Scalar[pdtype], bp_org],
    pc: Int,
    j0: Int,
    kc: Int,
    tile_n: Int,
):
    """Pack a kc x tile_n slab of B (top-left (pc, j0)) into [panel][k][NR],
    software-prefetching the next k-row. The pack widens to the compute dtype
    in the same touch (see `_pack_a_panel`). A partial trailing panel is
    padded out to full NR with zeros, so the micro-kernel can run it as a
    full panel (the zero columns contribute nothing and the masked store
    keeps only the valid ones)."""
    var full_panels = tile_n // NR
    var rem = tile_n % NR
    for pk in range(kc):
        var row = b.addr(pc + pk, j0)
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b.addr(pc + pk + 8, j0)
        )
        for jp in range(full_panels):
            var src = row + jp * NR
            var dst = bp + jp * kc * NR + pk * NR
            comptime for nv in range(NR_VECS):
                dst.store[width=NELTS](
                    offset=nv * NELTS,
                    val=src.load[width=NELTS](offset=nv * NELTS).cast[pdtype](),
                )
        if rem > 0:
            var src = row + full_panels * NR
            var dst = bp + full_panels * kc * NR + pk * NR
            for j in range(rem):
                dst[j] = src[j].cast[pdtype]()
            for j in range(rem, NR):
                dst[j] = Scalar[pdtype](0)


def packed_gemm[
    dtype: DType, MR: Int, NR: Int, KC: Int, KU: Int, TILE_N: Int,
    SHARED_A: Bool = False, PACK_A: Bool = True,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Packed C = A * B: pack A and B into cache-friendly panels, then run the
    `RegisterTile` micro-kernel over them, parallelized across N (j-tiles).

    Pointers are noalias to widen LLVM's hoisting; the NR_VECS B-loads per
    K-step are kept inline (not hidden behind a helper) so the compiler cannot
    re-issue them per accumulator row.

    SHARED_A packs the full A once up front instead of having every worker
    re-pack it per K-panel. Worth it when A is large next to the N-sweep (big
    squares, large M); off the wide/tall shapes A is tiny and the redundant
    per-worker pack is cheap. PACK_A=False skips A packing entirely and the
    micro-kernel gathers A columns straight from the row-major source (see
    docs/DESIGN.md).

    All arithmetic runs in the compute dtype (`compute_dtype`): the packed
    A/B panels are CDT elements (the pack widens bf16 to f32 as it copies),
    the unpacked-A gathers widen on load, and C stays in the storage dtype,
    cast at the register-tile boundary."""
    comptime CDT = compute_dtype[dtype]()
    comptime NELTS = simd_width_of[CDT]()
    comptime NR_VECS = NR // NELTS

    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols

    var num_j_tiles = ceildiv(n, TILE_N)
    var num_full_panels = m // MR
    var num_workers = num_physical_cores()

    var bp_per_worker = ceildiv(TILE_N, NR) * KC * NR + KU * NR
    var bp_buf = alloc[Scalar[CDT]](num_workers * bp_per_worker)

    var ap_per_worker: Int = 0
    var ap_total: Int = 1  # min size-1 alloc when PACK_A is off (A read unpacked)
    comptime if PACK_A:
        if SHARED_A:
            # One shared copy of packed A, laid out [i-panel][k][MR].
            # ap_per_worker = 0 so every worker indexes the same buffer.
            ap_total = ceildiv(m, MR) * MR * k
        else:
            ap_per_worker = ceildiv(m, MR) * MR * KC
            ap_total = num_workers * ap_per_worker
    var ap_buf = alloc[Scalar[CDT]](ap_total)

    comptime if PACK_A:
        if SHARED_A:
            # Pre-pack the full A once, in parallel over full MR-row i-panels
            # (the m % MR remainder rows are read unpacked by the masked tail).
            def pack_shared(ip: Int) {mut ap_buf, read a_view, read k}:
                _pack_a_panel[dtype, MR](a_view, ap_buf + ip * MR * k, ip * MR, 0, k)
            parallelize(pack_shared, num_full_panels, num_workers)

    def worker(worker_id: Int) {read c_view, read a_view, read b_view, mut bp_buf, mut ap_buf, read m, read n, read k, read num_j_tiles, read num_full_panels, read num_workers, read bp_per_worker, read ap_per_worker}:
        var per = ceildiv(num_j_tiles, num_workers)
        var jt0 = worker_id * per
        var jt1 = min(jt0 + per, num_j_tiles)
        if jt0 >= num_j_tiles:
            return

        var bp_worker = bp_buf + worker_id * bp_per_worker
        var ap_worker = ap_buf + worker_id * ap_per_worker

        for pc in range(0, k, KC):
            var kc = min(KC, k - pc)
            var is_first_k = pc == 0

            # Pack this K-panel of A, [i-panel][kc][MR]. Skipped under SHARED_A
            # (packed once up front) and under not-PACK_A (read unpacked).
            comptime if PACK_A:
                if not SHARED_A:
                    for ip in range(num_full_panels):
                        _pack_a_panel[dtype, MR](
                            a_view, ap_worker + ip * MR * kc, ip * MR, pc, kc
                        )

            for jt in range(jt0, jt1):
                var j0 = jt * TILE_N
                var tile_n = min(TILE_N, n - j0)

                _pack_b_slab[dtype, NR, NR_VECS, NELTS](
                    b_view, bp_worker, pc, j0, kc, tile_n
                )

                for jp in range(ceildiv(tile_n, NR)):
                    var jr = jp * NR
                    var cols = min(NR, tile_n - jr)
                    var bp_panel = bp_worker + jp * kc * NR

                    # Full MR-row i-panels. A partial NR-panel (cols < NR) runs
                    # the SAME register-tiled sweep through the zero-padded pack
                    # and masks the store to the valid columns.
                    for ip in range(num_full_panels):
                        var i = ip * MR
                        if cols == NR:
                            comptime if PACK_A:
                                var ap = (
                                    ap_buf + i * k + pc * MR
                                ) if SHARED_A else (ap_worker + i * kc)
                                full_microkernel[
                                    dtype, MR, NR_VECS, NELTS, NR, KU
                                ](
                                    c_view.sub(i, j0 + jr), ap, MR, 1,
                                    bp_panel, kc, is_first_k,
                                )
                            else:
                                full_microkernel[
                                    dtype, MR, NR_VECS, NELTS, NR, KU
                                ](
                                    c_view.sub(i, j0 + jr),
                                    a_view.addr(i, pc), 1, k,
                                    bp_panel, kc, is_first_k,
                                )
                        else:
                            comptime if PACK_A:
                                var ap = (
                                    ap_buf + i * k + pc * MR
                                ) if SHARED_A else (ap_worker + i * kc)
                                partial_n_microkernel[
                                    dtype, MR, NR_VECS, NELTS, NR, KU
                                ](
                                    c_view.sub(i, j0 + jr), ap, MR, 1,
                                    bp_panel, kc, cols, is_first_k,
                                )
                            else:
                                partial_n_microkernel[
                                    dtype, MR, NR_VECS, NELTS, NR, KU
                                ](
                                    c_view.sub(i, j0 + jr),
                                    a_view.addr(i, pc), 1, k,
                                    bp_panel, kc, cols, is_first_k,
                                )

                    # M-remainder (m % MR rows): one masked block reusing the
                    # same packed B panel.
                    var i = num_full_panels * MR
                    if i < m:
                        masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                            c_view.sub(i, j0 + jr), a_view.sub(i, pc),
                            bp_panel, kc, m - i, cols, is_first_k,
                        )

    parallelize(worker, num_workers, num_workers)
    bp_buf.free()
    ap_buf.free()


@always_inline
def pack_b_only_2d[
    dtype: DType, MR: Int, NR: Int, KU: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Pack-B-only GEMM parallelized over a 2D (column, MR-row-block) grid.

    The N-parallel pack-B-only path (dispatch's square-ish route) hands each
    worker whole NR-wide columns; when the column count is not a multiple of
    the worker count the last round leaves cores idle, and no TILE_N choice
    removes that (the unit of work is a whole column). This path makes the
    unit of work one MR x NR C tile instead and distributes the columns x
    row-blocks grid evenly across workers in column-major order.

    Each worker packs into a private [k][NR] buffer the columns its contiguous
    tile range touches, reusing it across that column's row blocks, so B is
    packed once per worker-column with at most one shared boundary column
    repacked per worker pair. Single K-panel only: the caller gates this to
    k <= 2048 so each tile is one sweep over the whole K and C is stored once.
    A reads from source unpacked. docs/DESIGN.md "Small-N square"."""
    comptime CDT = compute_dtype[dtype]()
    comptime NELTS = simd_width_of[CDT]()
    comptime NR_VECS = NR // NELTS

    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols

    var num_cols = ceildiv(n, NR)
    var num_i_panels = ceildiv(m, MR)
    var num_workers = num_physical_cores()
    var total_tiles = num_cols * num_i_panels

    # One private [k][NR] column buffer per worker (B packed lazily as the
    # worker crosses into each column it owns).
    var bp_buf = alloc[Scalar[CDT]](num_workers * k * NR)

    def worker(worker_id: Int) {read c_view, read a_view, read b_view, mut bp_buf, read m, read n, read k, read num_cols, read num_i_panels, read num_workers, read total_tiles}:
        var per = ceildiv(total_tiles, num_workers)
        var start = worker_id * per
        var end = min(start + per, total_tiles)
        if start >= total_tiles:
            return
        var bp_panel = bp_buf + worker_id * k * NR
        var cur_col = -1
        # Column-major flat index so the worker's contiguous slice walks whole
        # columns: it packs a column once on entry and reuses it down the rows.
        for u in range(start, end):
            var col = u // num_i_panels
            var ip = u % num_i_panels
            var jr = col * NR
            var cols = min(NR, n - jr)
            if col != cur_col:
                _pack_b_slab[dtype, NR, NR_VECS, NELTS](
                    b_view, bp_panel, 0, jr, k, cols
                )
                cur_col = col
            var i0 = ip * MR
            var rows = min(MR, m - i0)
            if rows == MR and cols == NR:
                full_microkernel[dtype, MR, NR_VECS, NELTS, NR, KU](
                    c_view.sub(i0, jr), a_view.addr(i0, 0), 1, k,
                    bp_panel, k, True,
                )
            elif rows == MR:
                # Partial trailing column with all MR rows live: the full
                # KU-unrolled sweep, masking only the C store.
                partial_n_microkernel[dtype, MR, NR_VECS, NELTS, NR, KU](
                    c_view.sub(i0, jr), a_view.addr(i0, 0), 1, k,
                    bp_panel, k, cols, True,
                )
            else:
                # M-remainder rows (and possibly a partial column too): the
                # masked kernel reads A unpacked with a per-row guard and
                # stores only the live rows/columns.
                masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                    c_view.sub(i0, jr), a_view.sub(i0, 0), bp_panel, k,
                    rows, cols, True,
                )

    parallelize(worker, num_workers, num_workers)
    bp_buf.free()
