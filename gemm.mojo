from cpu_cache import l2_cache_size, compute_core_count
from matrix import Matrix
from sme_kernel import sme_gemm_ptr, sme_gemm_small_ptr
from tile import Tile
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.math import ceildiv, fma
from std.memory import memset_zero
from std.memory.unsafe_pointer import alloc
from std.sys import (
    CompilationTarget,
    num_physical_cores,
    simd_width_of,
    size_of,
)
from std.sys.intrinsics import prefetch, PrefetchOptions


# ===========================================================================
# The register tile
#
# Every kernel in this file is the same idea: hold an MR x (NR_VECS*NELTS)
# block of C entirely in SIMD registers, sweep it over K with rank-1 updates,
# then write it back once. `RegisterTile` is that block. Its accumulator is an
# `InlineArray` the compiler flattens into registers, so each method is a
# zero-cost abstraction. After `@always_inline` the tile emits exactly the
# FMA/load/store nest you would otherwise hand-write and hand-number. Naming it
# once lets each kernel below express only its own packing and loop scaffolding.
# ===========================================================================


struct RegisterTile[dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int](
    Copyable & Movable
):
    """An MR x (NR_VECS*NELTS) block of C, resident in SIMD registers."""

    var acc: InlineArray[SIMD[Self.dtype, Self.NELTS], Self.MR * Self.NR_VECS]

    @always_inline
    def __init__(out self):
        """Start at zero: a fresh C block, or the first K-panel of a sweep."""
        self.acc = InlineArray[
            SIMD[Self.dtype, Self.NELTS], Self.MR * Self.NR_VECS
        ](fill=SIMD[Self.dtype, Self.NELTS](0))

    @always_inline
    def rank1_update(
        mut self,
        a_col: InlineArray[Scalar[Self.dtype], Self.MR],
        b_row: InlineArray[SIMD[Self.dtype, Self.NELTS], Self.NR_VECS],
    ):
        """C_tile += a_col (x) b_row, one K-step. Broadcast each of the MR A
        scalars across the NR_VECS B vectors and FMA into the tile. This is the
        single inner step shared by every kernel here. It takes SIMD values
        only (no pointers), so it can never perturb the noalias B-load hoisting
        the hot loops depend on."""
        comptime for mr in range(Self.MR):
            var a_bc = SIMD[Self.dtype, Self.NELTS](a_col[mr])
            comptime for nr in range(Self.NR_VECS):
                self.acc[mr * Self.NR_VECS + nr] = a_bc.fma(
                    b_row[nr], self.acc[mr * Self.NR_VECS + nr]
                )

    @always_inline
    def load[org: MutOrigin](mut self, c: Tile[Self.dtype, org]):
        """Seed the tile from a C block, to accumulate onto a prior K-panel's
        partial. `c` is the block's view: c.row(0) is its top-left, rows
        c.stride apart."""
        comptime for mr in range(Self.MR):
            var row = c.row(mr)
            comptime for nr in range(Self.NR_VECS):
                self.acc[mr * Self.NR_VECS + nr] = row.load[width = Self.NELTS](
                    offset=nr * Self.NELTS
                )

    @always_inline
    def store[org: MutOrigin](self, c: Tile[Self.dtype, org]):
        """Write the finished tile back to a C block (see `load` for the view)."""
        comptime for mr in range(Self.MR):
            var row = c.row(mr)
            comptime for nr in range(Self.NR_VECS):
                row.store(
                    offset=nr * Self.NELTS, val=self.acc[mr * Self.NR_VECS + nr]
                )


# --- The two operands of one K-step ----------------------------------------
#
# Every K-step in every kernel feeds `rank1_update` the same pair: NR_VECS
# contiguous B vectors and MR A scalars. These two `@always_inline` loaders name
# that gather once. Like the tile itself they are zero-cost: the comptime loop
# over a register-flattened `InlineArray` lowers to the exact vmovupd nest you'd
# hand-write, so every kernel's inner loop collapses to a single readable line:
#
#     tile.rank1_update(load_a_col[MR](a, stride), load_b_row[NR_VECS, NELTS](b))


@always_inline
def load_b_row[
    dtype: DType, org: ImmutOrigin, //, NR_VECS: Int, NELTS: Int
](bp_k: UnsafePointer[Scalar[dtype], org]) -> InlineArray[
    SIMD[dtype, NELTS], NR_VECS
]:
    """The B operand of one K-step: NR_VECS contiguous NELTS-wide vectors."""
    var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](uninitialized=True)
    comptime for nr in range(NR_VECS):
        bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS)
    return bv^


@always_inline
def load_a_col[
    dtype: DType, org: ImmutOrigin, //, MR: Int
](a_base: UnsafePointer[Scalar[dtype], org], stride: Int) -> InlineArray[
    Scalar[dtype], MR
]:
    """The A operand of one K-step: MR scalars at `stride` apart. `stride == 1`
    reads a packed-A column; `stride == k` gathers a column straight from a
    row-major A (the no-pack kernels)."""
    var a_col = InlineArray[Scalar[dtype], MR](uninitialized=True)
    comptime for mr in range(MR):
        a_col[mr] = a_base[mr * stride]
    return a_col^


@always_inline
def _masked_microkernel[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int, NR: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: MutOrigin,
](
    c_block: Tile[dtype, c_org],
    a_block: Tile[dtype, a_org],
    bp_panel: UnsafePointer[Scalar[dtype], b_org],
    kc: Int,
    r: Int,
    jj_limit: Int,
    is_first_k: Bool,
):
    """Cold-path register tile: the leftover blocks the hot loop can't take.

    Handles an M-remainder (only `r` of MR rows active) and/or a partial
    NR-panel (only `jj_limit` of NR columns valid) by masking the C load and
    store; reads A unpacked (so it covers the un-packed remainder rows too) and
    B from the zero-padded packed panel. With r == MR and jj_limit == NR it is
    the unmasked full kernel, so this one function also serves the full-panel
    M-remainder.

    `c_block`/`a_block` are `sub`-views onto the block's corner, (i, j0+jr) in C
    and (i, pc) in A, so the kernel reads C via `c_block.row(mr)` and A via
    `a_block.row(mr)[pk]` instead of `c_ptr + i*n + j0 + jr` pointer math.
    """
    var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
    if not is_first_k:
        comptime for mr in range(MR):
            if mr < r:
                var cr = c_block.row(mr)
                comptime for nr in range(NR_VECS):
                    var col0 = nr * NELTS
                    if col0 + NELTS <= jj_limit:
                        tile.acc[mr * NR_VECS + nr] = cr.load[width=NELTS](offset=col0)
                    elif col0 < jj_limit:
                        var tmp = SIMD[dtype, NELTS](0)
                        for e in range(jj_limit - col0):
                            tmp[e] = cr[col0 + e]
                        tile.acc[mr * NR_VECS + nr] = tmp
    for pk in range(kc):
        var b_row = load_b_row[NR_VECS, NELTS](bp_panel + pk * NR)
        # A is gathered here (not via load_a_col) because the gather is guarded
        # by mr < r: rows past the M-remainder are out of bounds, so leave them
        # zero (their acc lanes are never stored).
        var a_col = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))
        comptime for mr in range(MR):
            if mr < r:
                a_col[mr] = a_block.row(mr)[pk]
        tile.rank1_update(a_col, b_row)
    comptime for mr in range(MR):
        if mr < r:
            var cr = c_block.row(mr)
            comptime for nr in range(NR_VECS):
                var col0 = nr * NELTS
                if col0 + NELTS <= jj_limit:
                    cr.store(offset=col0, val=tile.acc[mr * NR_VECS + nr])
                elif col0 < jj_limit:
                    var v = tile.acc[mr * NR_VECS + nr]
                    for e in range(jj_limit - col0):
                        cr[col0 + e] = v[e]


# ===========================================================================
# The packed prefill GEMM: the workhorse
# ===========================================================================


@always_inline
def _pack_b_slab[
    dtype: DType, NR: Int, NR_VECS: Int, NELTS: Int, PREFETCH_B_DIST: Int,
    b_org: ImmutOrigin, bp_org: MutOrigin,
](
    b: Tile[dtype, b_org],
    bp_worker: UnsafePointer[Scalar[dtype], bp_org],
    pc: Int,
    j0: Int,
    kc: Int,
    last_full_panel: Int,
    has_remainder: Bool,
    nr_actual: Int,
):
    """Pack one j-tile's slab of B into the worker buffer as [panel][k][NR],
    software-prefetching the next k-row. A partial trailing panel is padded out
    to full NR with zeros, so the micro-kernel can run it as a full panel (the
    zero columns contribute nothing and the masked store keeps only the valid
    ones). Pulled out of the worker so the K-panel loop reads as pack-then-compute."""
    for pk in range(kc):
        var row_base = b.addr(pc + pk, j0)
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b.addr(pc + pk + PREFETCH_B_DIST, j0)
        )
        for jp in range(last_full_panel):
            var src = row_base + jp * NR
            var dst = bp_worker + jp * kc * NR + pk * NR
            comptime for nv in range(NR_VECS):
                dst.store[width=NELTS](
                    offset=nv * NELTS,
                    val=src.load[width=NELTS](offset=nv * NELTS),
                )
        if has_remainder:
            var src = row_base + last_full_panel * NR
            var dst = bp_worker + last_full_panel * kc * NR + pk * NR
            for nr in range(nr_actual):
                dst[nr] = src[nr]
            for nr in range(nr_actual, NR):
                dst[nr] = Scalar[dtype](0)


def _packed_gemm[
    dtype: DType, MR: Int, NR: Int, KC: Int, KU: Int, TILE_N: Int,
    NC_TILES: Int, SHARED_A: Bool = False, PACK_A: Bool = True,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Packed C = A * B: pack A and B into cache-friendly panels, then run the
    `RegisterTile` micro-kernel over them, parallelized across N (j-tiles).

    Pointers are noalias to widen LLVM's hoisting; the NR_VECS B-loads per
    K-step are kept inline (not hidden behind a helper) so the compiler cannot
    re-issue them per accumulator row.

    SHARED_A packs the full A once up front instead of having every worker
    re-pack it per K-panel. Off the headline wide/tall shapes A is tiny next to
    the N-sweep, so the redundant per-worker pack is cheap and SHARED_A is left
    off; on a big square A is as large as B/C and packing it once is a real win
    (see DESIGN.md)."""
    comptime NELTS = simd_width_of[dtype]()
    comptime NR_VECS = NR // NELTS
    comptime PREFETCH_B_DIST = 8

    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols

    var num_j_tiles = ceildiv(n, TILE_N)
    var num_i_panels = ceildiv(m, MR)
    var num_workers = compute_core_count()

    var num_nr_panels = ceildiv(TILE_N, NR)
    var bp_per_worker = num_nr_panels * KC * NR + KU * NR
    var bp_total = num_workers * bp_per_worker
    var bp_buf = alloc[Scalar[dtype]](bp_total)

    var num_full_panels = m // MR
    var ap_per_worker: Int = 0
    var ap_total: Int = 1  # min size-1 alloc when PACK_A is off (A read unpacked)
    comptime if PACK_A:
        if SHARED_A:
            # One shared copy of packed A, laid out [i-panel][k][MR].
            # ap_per_worker = 0 so every worker indexes the same buffer.
            ap_per_worker = 0
            ap_total = num_i_panels * MR * k
        else:
            ap_per_worker = num_i_panels * MR * KC
            ap_total = num_workers * ap_per_worker
    var ap_buf = alloc[Scalar[dtype]](ap_total)

    comptime if PACK_A:
        if SHARED_A:
            # Pre-pack the full A once, in parallel over full MR-row i-panels. Each
            # panel block is MR*k contiguous, organized [k][MR] so the micro-kernel's
            # ap_panel + pk*MR + mr indexing matches the per-worker layout (just with
            # a full-k stride + pc offset instead of a per-panel kc stride).
            def pack_a_panel(ip: Int) {mut ap_buf, read a_view, read k}:
                var i0 = ip * MR
                var ap_panel = ap_buf + ip * MR * k
                for pk in range(k):
                    var dst = ap_panel + pk * MR
                    comptime for mr in range(MR):
                        dst[mr] = a_view.row(i0 + mr)[pk]
            parallelize(pack_a_panel, num_full_panels, num_workers)

    def process_worker(worker_id: Int) {read c_view, read a_view, read b_view, mut bp_buf, mut ap_buf, read m, read n, read k, read num_j_tiles, read num_workers, read bp_per_worker, read ap_per_worker}:
        var tiles_per_worker = ceildiv(num_j_tiles, num_workers)
        var j_tile_start = worker_id * tiles_per_worker
        var j_tile_end = min(j_tile_start + tiles_per_worker, num_j_tiles)
        if j_tile_start >= num_j_tiles:
            return

        var bp_worker = bp_buf + worker_id * bp_per_worker
        var ap_worker = ap_buf + worker_id * ap_per_worker

        # Hoisted captures for inner closures (declare type-only, see AGENTS.md).
        var bp_panel: type_of(bp_worker)
        var is_first_k: Bool

        var jt = j_tile_start
        while jt < j_tile_end:
            var jt_batch_end = min(jt + NC_TILES, j_tile_end)

            for pc in range(0, k, KC):
                var kc = min(KC, k - pc)
                is_first_k = (pc == 0)

                # Pack A: KC outer, MR inner so each pk gives MR contiguous
                # doubles. Skipped under SHARED_A (packed once up front) and under
                # not-PACK_A (the micro-kernel reads A unpacked, linalg-style).
                var i = 0
                var ip = 0
                comptime if PACK_A:
                    if not SHARED_A:
                        while i + MR <= m:
                            var ap_panel = ap_worker + ip * MR * kc
                            for pk in range(kc):
                                var dst = ap_panel + pk * MR
                                comptime for mr in range(MR):
                                    dst[mr] = a_view.row(i + mr)[pc + pk]
                            i += MR
                            ip += 1

                for j_tile_idx in range(jt, jt_batch_end):
                    var j0 = j_tile_idx * TILE_N
                    var tile_n = min(TILE_N, n - j0)
                    var num_panels = ceildiv(tile_n, NR)

                    var last_full_panel = num_panels
                    var has_remainder = False
                    var nr_actual = 0
                    if num_panels > 0:
                        var last_jr = (num_panels - 1) * NR
                        if last_jr + NR > tile_n:
                            last_full_panel = num_panels - 1
                            has_remainder = True
                            nr_actual = tile_n - last_jr

                    # Pack this j-tile's slab of B into [panel][k][NR] (a partial
                    # trailing panel is zero-padded to full NR, see _pack_b_slab).
                    _pack_b_slab[dtype, NR, NR_VECS, NELTS, PREFETCH_B_DIST](
                        b_view, bp_worker, pc, j0, kc,
                        last_full_panel, has_remainder, nr_actual,
                    )

                    for jp in range(num_panels):
                        var jr = jp * NR
                        bp_panel = bp_worker + jp * kc * NR

                        if jr + NR > tile_n:
                            # Partial NR-panel (tile_n not a multiple of NR, the
                            # last j-tile of an N-not-a-multiple-of-NR shape). The
                            # packed panel is zero-padded to full NR, so run the
                            # SAME register-tiled micro-kernel (the zero columns
                            # contribute nothing) and store back only the jj_limit
                            # valid columns. `_masked_microkernel` does exactly
                            # that: full MR-row i-panels first (r = MR), then the
                            # m % MR remainder rows (r = m - i).
                            var jj_limit = tile_n - jr
                            i = 0
                            while i + MR <= m:
                                _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                                    c_view.sub(i, j0 + jr), a_view.sub(i, pc),
                                    bp_panel, kc, MR, jj_limit, is_first_k,
                                )
                                i += MR
                            if i < m:
                                _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                                    c_view.sub(i, j0 + jr), a_view.sub(i, pc),
                                    bp_panel, kc, m - i, jj_limit, is_first_k,
                                )
                            continue

                        # ---- Full NR-panel: the register-tile micro-kernel ----
                        i = 0
                        ip = 0
                        while i + MR <= m:
                            # A operand source per K-step: PACK_A reads the packed
                            # panel at unit stride (SHARED_A: full-k stride per
                            # i-panel + pc offset; else per-worker per-kc layout);
                            # not-PACK_A reads A straight from source at stride k
                            # (a column gather), the linalg-style no-A-pack path.
                            var ap_panel = (
                                ap_worker + ip * MR * k + pc * MR
                            ) if SHARED_A else (ap_worker + ip * MR * kc)
                            var c_block = c_view.sub(i, j0 + jr)

                            comptime for mr in range(MR):
                                prefetch[PrefetchOptions().for_write().high_locality().to_data_cache()](
                                    c_block.row(mr)
                                )

                            var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
                            if not is_first_k:
                                tile.load(c_block)

                            # K-sweep, unrolled by KU so KU*NR_VECS B-vectors stay
                            # live per step. KU=2 keeps the 6x32 tile's 24 + 8 = 32
                            # accumulator+B vectors inside the AVX-512 register file
                            # (KU=4 needs 40 and spills). The bv loads stay inline
                            # so the noalias B-load hoist holds.
                            var pk = 0
                            var pk_end = kc - (kc % KU)
                            while pk < pk_end:
                                comptime for ku in range(KU):
                                    var step = pk + ku
                                    comptime if PACK_A:
                                        tile.rank1_update(
                                            load_a_col[MR](ap_panel + step * MR, 1),
                                            load_b_row[NR_VECS, NELTS](bp_panel + step * NR),
                                        )
                                    else:
                                        tile.rank1_update(
                                            load_a_col[MR](a_view.addr(i, pc + step), k),
                                            load_b_row[NR_VECS, NELTS](bp_panel + step * NR),
                                        )
                                pk += KU

                            while pk < kc:
                                comptime if PACK_A:
                                    tile.rank1_update(
                                        load_a_col[MR](ap_panel + pk * MR, 1),
                                        load_b_row[NR_VECS, NELTS](bp_panel + pk * NR),
                                    )
                                else:
                                    tile.rank1_update(
                                        load_a_col[MR](a_view.addr(i, pc + pk), k),
                                        load_b_row[NR_VECS, NELTS](bp_panel + pk * NR),
                                    )
                                pk += 1

                            tile.store(c_block)

                            i += MR
                            ip += 1

                        # M-remainder (m % MR rows): one register-tiled block, a
                        # single K-sweep with r x NR_VECS accumulators reusing the
                        # packed B panel at full NR width (jj_limit = NR, since the
                        # partial-tile case already `continue`d above).
                        if i < m:
                            _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                                c_view.sub(i, j0 + jr), a_view.sub(i, pc),
                                bp_panel, kc, m - i, NR, is_first_k,
                            )

            jt += NC_TILES

    parallelize(process_worker, num_workers, num_workers)
    bp_buf.free()
    ap_buf.free()


@always_inline
def _prefill[
    dtype: DType, KC: Int, TILE_N: Int, SHARED_A: Bool = False,
    PACK_A: Bool = True,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """The SOTA packed GEMM at its standard 6 x (4*NELTS) register tile (KU=2,
    NC_TILES=64). KC, TILE_N, SHARED_A and PACK_A are the only levers that vary
    across shapes, so naming the rest here keeps each dispatch branch a one-liner."""
    comptime NELTS = simd_width_of[dtype]()
    _packed_gemm[dtype, 6, 4 * NELTS, KC, 2, TILE_N, 64, SHARED_A, PACK_A](
        c, a, b
    )


@always_inline
def _prefill_kc[
    dtype: DType, TILE_N: Int, SHARED_A: Bool = False, PACK_A: Bool = True
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype], kc: Int):
    """Run `_prefill` at the comptime KC rung nearest the runtime cache-aware `kc`.
    KC must be a compile-time constant for the micro-kernel, so the measured rungs
    {512, 1024, 2048} are spelled out here and each large-M branch just passes its
    computed kc, snapping down (kc >= rung)."""
    if kc >= 2048:
        _prefill[dtype, 2048, TILE_N, SHARED_A, PACK_A](c, a, b)
    elif kc >= 1024:
        _prefill[dtype, 1024, TILE_N, SHARED_A, PACK_A](c, a, b)
    else:
        _prefill[dtype, 512, TILE_N, SHARED_A, PACK_A](c, a, b)


# ===========================================================================
# Decode GEMV (M = 1)
# ===========================================================================


@always_inline
def _decode_fma_chunk_unrolled[
    dtype: DType, KU: Int, NELTS: Int,
    c_origin: MutOrigin, a_origin: ImmutOrigin, b_origin: ImmutOrigin,
](
    ci: UnsafePointer[Scalar[dtype], c_origin],
    ai: UnsafePointer[Scalar[dtype], a_origin],
    b_col: UnsafePointer[Scalar[dtype], b_origin],
    p: Int,
    n: Int,
    chunk: Int,
):
    """KU-unrolled FMA chunk for the GEMV main loop.

    A top-level def so the inner closure can capture `p` and the pointers as
    function-scope bindings (a closure nested inside another closure cannot
    capture for/while-loop-body variables in current Mojo)."""
    def do_fma[width: Int](j: Int) {read ci, read ai, read b_col, read p, read n}:
        var acc = ci.load[width=width](offset=j)
        comptime for ku in range(KU):
            # Prefetch the same columns of the next KU-block of B rows: the KU
            # streams are n*8 bytes apart, too far for the HW prefetcher. May
            # reach past the end of B on the last block. Prefetch is
            # architecturally non-faulting, so that is safe.
            prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                b_col + (p + ku + KU) * n + j
            )
            var a_broadcast = SIMD[dtype, width](ai[p + ku])
            var b_vec = (b_col + (p + ku) * n).load[width=width, invariant=True](offset=j)
            acc = a_broadcast.fma(b_vec, acc)
        ci.store(offset=j, val=acc)
    vectorize[NELTS, unroll_factor=4](chunk, do_fma)


@always_inline
def _decode_fma_chunk_tail[
    dtype: DType, NELTS: Int,
    c_origin: MutOrigin, a_origin: ImmutOrigin, b_origin: ImmutOrigin,
](
    ci: UnsafePointer[Scalar[dtype], c_origin],
    ai: UnsafePointer[Scalar[dtype], a_origin],
    b_col: UnsafePointer[Scalar[dtype], b_origin],
    p: Int,
    n: Int,
    chunk: Int,
):
    """Single-step FMA chunk for the GEMV tail loop."""
    def do_fma_tail[width: Int](j: Int) {read ci, read ai, read b_col, read p, read n}:
        var a_broadcast = SIMD[dtype, width](ai[p])
        var b_vec = (b_col + p * n).load[width=width, invariant=True](offset=j)
        ci.store(offset=j, val=a_broadcast.fma(b_vec, ci.load[width=width](offset=j)))
    vectorize[NELTS, unroll_factor=4](chunk, do_fma_tail)


def _decode_gemv[
    dtype: DType, //, KU: Int = 8,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """J-parallel GEMV optimized for decode (small M, large K x N).

    Each worker owns a disjoint column chunk of C and sweeps all K rows. The
    per-k working set ~ (N/nw)*8 bytes of B + same for C fits L1 (e.g. 2752*8 =
    21 KB for N=11008, nw=4), so B streams past exactly once and no reduction
    is needed."""
    comptime assert KU > 0, "KU must be positive"
    comptime assert dtype.is_floating_point(), "GEMV requires floating-point dtype"
    comptime NELTS = simd_width_of[dtype]()

    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols
    var nw = compute_core_count()

    comptime if CompilationTarget.is_apple_silicon():
        # Apple Silicon: split by K-ROWS, not by N-columns. The column-split below
        # has each worker read a strided column slice of row-major B (stride N
        # between consecutive K rows), which only sustains ~132 GB/s of the ~242
        # GB/s the P-cores can read sequentially. Splitting by K instead lets each
        # worker stream a CONTIGUOUS block of B's rows (fully sequential, full
        # bandwidth) into a private full-width partial-C, then a parallel reduce
        # sums the partials. Decode 1x11008x2048: 33 -> ~58 GFLOPS (matches
        # Accelerate). The partials sum in a different order than the column-split,
        # so this is not bit-identical to the x86 path (~1e-12 f64 reorder, well
        # inside the 1e-7 dispatch tolerance), which is why it is Apple-gated.
        comptime W = 4 * NELTS
        var a_ptr = a_view.ptr
        var b_ptr = b_view.ptr
        var c_ptr = c_view.ptr
        # Over-decompose K into more chunks than workers (~4x) so parallelize
        # work-steals a balanced share regardless of how K divides across the
        # P-cores. A fixed nw-way split left ~1/3 of bandwidth on the table when K
        # did not divide evenly (1x4096x4096 ran 40 at nw=10 vs 61 at 40 chunks);
        # over-decomposition takes both decode shapes to ~102-103% of Accelerate.
        var nchunks = 4 * nw
        if nchunks > k: nchunks = k
        var partials = alloc[Scalar[dtype]](nchunks * m * n)
        def row_worker(ch: Int) {read partials, read a_ptr, read b_ptr, read m, read n, read k, read nchunks}:
            var per = ceildiv(k, nchunks)
            var k0 = ch * per
            var k1 = min(k0 + per, k)
            var pbase = partials + ch * m * n
            for x in range(m * n):
                pbase[x] = 0
            for kk in range(k0, k1):
                var brow = b_ptr + kk * n
                for i in range(m):
                    var aik = a_ptr[i * k + kk]
                    var p = pbase + i * n
                    var j = 0
                    while j + W <= n:
                        p.store(j, p.load[width=W](j) + aik * brow.load[width=W](j))
                        j += W
                    while j < n:
                        p.store(j, p.load(j) + aik * brow.load(j))
                        j += 1
        parallelize(row_worker, nchunks, nw)
        # Reduce the nchunks partials into C, parallelized over output elements.
        def reduce(x: Int) {read partials, read c_ptr, read m, read n, read nchunks}:
            var s = Scalar[dtype](0)
            for w in range(nchunks):
                s += partials[w * m * n + x]
            c_ptr[x] = s
        parallelize(reduce, m * n, nw)
        partials.free()
        return

    memset_zero(c_view.ptr, m * n)

    def worker(wid: Int) {read c_view, read a_view, read b_view, read m, read n, read k, read nw}:
        var cols_per = ceildiv(n, nw)
        var j0 = wid * cols_per
        var j1 = min(j0 + cols_per, n)
        var chunk = j1 - j0
        if chunk <= 0:
            return

        var b_col = b_view.addr(0, j0)  # base pointer into worker's column chunk
        var k_main = (k // KU) * KU

        for i in range(m):
            var ci = c_view.addr(i, j0)
            var ai = a_view.row(i)
            var p = 0

            while p < k_main:
                _decode_fma_chunk_unrolled[dtype=dtype, KU=KU, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                p += KU

            while p < k:
                _decode_fma_chunk_tail[dtype=dtype, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                p += 1

    parallelize(worker, nw, nw)


# ===========================================================================
# No-pack kernels: serial (tiny) and M-parallel thin-N / small box
#
# Both read A and B straight from source. For a tiny or cache-resident shape
# the data already fits L1/L2, so explicit packing buys nothing and the prefill
# kernel's packing + thread-launch overhead would dominate. Both reuse
# `RegisterTile` for the full MR-row panels and a 1-row tile for the M-tail.
# ===========================================================================


def _serial_gemm[
    dtype: DType, MR: Int, NR_VECS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Serial register-tiled GEMM for tiny shapes (no threads, no packing).

    Below the dispatch's tiny cutoff the parallel kernels' fixed cost (thread
    launch + per-worker buffers + packing) dwarfs the compute, running
    10-100x slower than this plain serial loop (an 8x8x8 measured 0.16 GFLOPS
    through the parallel path vs 14 GFLOPS here). Computes C = A * B; bit-
    identical to the parallel kernels. MR=6, NR_VECS=2 measured best."""
    comptime NELTS = simd_width_of[dtype]()
    comptime NR = NR_VECS * NELTS
    var c_view = c.view()
    var a_view = a.view()
    var b_view = b.view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols

    var j = 0
    while j + NR <= n:
        # Full MR-row register-tiled panels.
        var i = 0
        while i + MR <= m:
            var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
            for p in range(k):
                tile.rank1_update(
                    load_a_col[MR](a_view.addr(i, p), k),
                    load_b_row[NR_VECS, NELTS](b_view.addr(p, j)),
                )
            tile.store(c_view.sub(i, j))
            i += MR
        # M-remainder rows (m % MR): one row at a time, same NR-wide SIMD.
        while i < m:
            var tile = RegisterTile[dtype, 1, NR_VECS, NELTS]()
            for p in range(k):
                tile.rank1_update(
                    load_a_col[1](a_view.addr(i, p), k),
                    load_b_row[NR_VECS, NELTS](b_view.addr(p, j)),
                )
            tile.store(c_view.sub(i, j))
            i += 1
        j += NR
    # N-remainder columns (n % NR): vectorized tail over the full M extent,
    # with NELTS-wide chunks + an automatic scalar tail.
    if j < n:
        var jr = j
        for i in range(m):
            var c_row = c_view.row(i)
            var a_row = a_view.row(i)

            def tail[width: Int](jj: Int) {mut c_row, read a_row, read b_view, read k, read jr}:
                var acc = SIMD[dtype, width](0)
                for p in range(k):
                    acc = fma(
                        SIMD[dtype, width](a_row[p]),
                        b_view.addr(p, jr).load[width=width](offset=jj),
                        acc,
                    )
                c_row.store(offset=jr + jj, val=acc)

            vectorize[NELTS](n - jr, tail)


def _nopack_gemm[
    dtype: DType, MR: Int, NR_VECS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """M-parallel register-tiled GEMM with NO packing, for two regimes the
    N-parallel prefill kernel handles badly:

      * thin-N (small N, large M*K): the prefill kernel parallelizes only over
        N, so a thin N starves the cores (N=16 -> 1 j-tile -> 1 of 4 cores).
        Here the work is along M, with NR_VECS 1-2 (NR=8/16).
      * small M-dominant box whose B fits L2 (the sq96..256 gap): the prefill
        kernel's packing + per-worker buffers + launch overhead dwarfs the
        compute when the whole problem is cache-resident. Here NR=32 (NR_VECS=4).

    Either way every core owns a band of C's rows and sweeps the full N, reading
    A/B straight from source. Computes C = A * B; bit-identical to the other
    paths."""
    comptime NELTS = simd_width_of[dtype]()
    comptime NR = NR_VECS * NELTS
    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols
    # All physical cores (incl. Apple E-cores): the boxes that reach this kernel
    # are small and cache-resident (the Apple box-budget cut routes the heavier
    # ones to the packed P-core path), so the compute per block is tiny, the
    # E-cores never straggle, and capping to P-cores only idles 4 cores
    # (P-core no-pack measured worse: sq256 0.82 vs 1.11, sq320 0.91 vs 1.04).
    var nw = num_physical_cores()
    var num_blocks = ceildiv(m, MR)

    def worker(blk: Int) {read c_view, read a_view, read b_view, read m, read n, read k}:
        var i = blk * MR
        var r = min(MR, m - i)
        # A full MR-row block uses the comptime-unrolled MR tile (each B-load
        # reused across MR rows); a tail block (m % MR) falls back to 1 row.
        if r == MR:
            var j = 0
            while j + NR <= n:
                var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
                for p in range(k):
                    tile.rank1_update(
                        load_a_col[MR](a_view.addr(i, p), k),
                        load_b_row[NR_VECS, NELTS](b_view.addr(p, j)),
                    )
                tile.store(c_view.sub(i, j))
                j += NR
        else:
            for ii in range(i, i + r):
                var j2 = 0
                while j2 + NR <= n:
                    var tile = RegisterTile[dtype, 1, NR_VECS, NELTS]()
                    for p in range(k):
                        tile.rank1_update(
                            load_a_col[1](a_view.addr(ii, p), k),
                            load_b_row[NR_VECS, NELTS](b_view.addr(p, j2)),
                        )
                    tile.store(c_view.sub(ii, j2))
                    j2 += NR

        # N-remainder columns (n % NR < NR): scalar dot per (row, col). A tiny
        # strip (NR <= 16 cols), so a scalar tail costs nothing.
        var jr = (n // NR) * NR
        for ii in range(i, i + r):
            var a_row = a_view.row(ii)
            for jj in range(jr, n):
                var acc = Scalar[dtype](0)
                for p in range(k):
                    acc = fma(a_row[p], b_view.addr(p, jj)[0], acc)
                c_view.addr(ii, jj)[0] = acc

    parallelize(worker, num_blocks, nw)


# ===========================================================================
# Cache-aware tile heuristics
#
# Each picks a KC (K-panel depth) or a routing budget from the detected L2 so
# the working set stays cache-resident across machines. The measurement tables
# behind these numbers live in DESIGN.md.
# ===========================================================================


def _l2_resident_kc[dtype: DType](tile_n: Int, k: Int) -> Int:
    """Cache-aware KC for the prefill GEMM's large-M band: size the L2-resident
    packed-B tile (TILE_N x KC) at ~half the per-core L2 (the BLIS rule), leaving
    the rest for the streaming packed-A panel and the C accumulators. A 1 MB/core
    L2 yields KC=1024, a 2 MB/core L2 yields KC=2048; the caller snaps to its
    KC ladder."""
    comptime elem = size_of[Scalar[dtype]]()
    var l2 = l2_cache_size()
    if l2 == 0:
        # L2 undetectable (e.g. non-x86 without sysctl): pick a KC for a 1 MB L2.
        return min(512, k)
    var budget = (l2 // 2) // (tile_n * elem)
    return min(budget, k)


def _square_ish_kc(m: Int, n: Int, k: Int) -> Int:
    """Per-L2 KC for the square-ish branch: HALF the wide/tall branches' KC,
    because here the M*KC packed-A competes with the packed-B tile for L2. Yields
    KC=1024 from a 1 MB/core L2 up, KC=512 below, each the measured best.

    KC only bites when k > 512 (a single panel covers shorter K either way), and
    such a square-ish op (with M*N*K >= 2^28) is multi-ms, so the l2_cache_size()
    probe is < 2.5%. Smaller shapes skip the probe and take KC=512, keeping cpuid
    off the hot path for the few-us shapes. The 1 MB cut (was 1.5 MB, picking
    KC=512 on the 1 MB Skylake) followed a current-nightly interleaved A/B: KC1024
    beats KC512 on that part, sq1024 +3.6% / sq2048 +2.9% (2-sigma WIN), reusing
    each L2-resident C micro-tile across twice the K before its load/store. The
    512 KB packed-B tile (wide TILE_N x 1024) needs the full 1 MB L2 to coexist
    with packed A and C, so sub-1 MB parts stay on KC=512. DESIGN.md."""
    if k <= 512 or m * n * k < (1 << 28):
        return 512
    return 1024 if l2_cache_size() >= (1 << 20) else 512


def _box_l2_budget() -> Int:
    """Upper bound (bytes of B = k*n) for routing an M-dominant box to the no-pack
    `_nopack_gemm`. That kernel re-reads all of B once per MR-row block, so B must
    stay L2-resident *alongside* the packed-A panel, C, and prefetch headroom
    across the whole M-sweep, which holds only while B is ~1/3 of L2. Past that
    B spills mid-sweep and the packed prefill path wins, so cut at L2/3. Falls
    back to a compile-time 512 KB tier when L2 is undetectable."""
    var l2 = l2_cache_size()
    if l2 == 0:
        return (1 << 19)
    comptime if CompilationTarget.is_apple_silicon():
        # On Apple Silicon l2_cache_size() reports the cluster-shared L2 (16 MB
        # on M4 Max), so the Intel per-core l2/3 rule (~5.6 MB here) wildly
        # over-admits boxes to the no-pack route. That route skips packing but
        # re-reads all of B once per MR-row block and runs on all cores (incl.
        # the slow E-cores), so measured on M4 Max it only beats the packed
        # P-core path for genuinely small boxes: it wins to ~sq320 (B 800 KB,
        # ratio 1.04) and loses above (sq512 B 2 MB 0.76, box768 B 1 MB 0.73,
        # box640 0.83), where the packed path is both algorithmically better
        # (B packed once, reused) and P-core-only (no straggler). Cut at the
        # measured ~900 KB crossover.
        return (7 << 17)  # 896 KB
    return l2 // 3


# ===========================================================================
# Dispatch predicates
#
# One named gate per regime, tested top to bottom in matmul_dispatch below. Each
# returns only its own condition; the regime it selects is that condition with
# every gate above it having already failed (so _is_square_ish is just `n <= m`,
# because the narrow-N gate above it has ruled out n <= 192). The thresholds that
# differ across CPUs sit in _tiny_cutoff here and _box_l2_budget (cache heuristics
# above), and the SME gates compile away off Apple/f64. The measured rationale is
# in README.md / DESIGN.md.
# ===========================================================================


def _tiny_cutoff(m: Int) -> Int:
    """MAC count below which the serial loop beats the parallel kernels' launch +
    packing overhead. 2^19 on x86. On Apple the cheap parallel launch drops it to
    2^18 once there are enough M-row blocks to fill the cores (m >= 64); a small-M
    shape keeps 2^19 (too few row blocks to parallelize). DESIGN.md adaptation 3."""
    comptime if CompilationTarget.is_apple_silicon():
        if m >= 64:
            return 1 << 18
    return 1 << 19


def _is_tiny(m: Int, n: Int, k: Int) -> Bool:
    """Tiny shape: total work below the serial/parallel crossover, where the
    parallel kernels' launch + packing overhead dwarfs the compute. The crossover
    is hardware specific (_tiny_cutoff)."""
    return m * n * k < _tiny_cutoff(m)


def _is_decode(m: Int) -> Bool:
    """Decode GEMV: a single output row (M == 1)."""
    return m == 1


def _is_small_batch(m: Int) -> Bool:
    """Small-batch decode: 2..5 rows (M == 1 is already the GEMV), few enough to
    reuse one packed B panel across all rows with MR = M."""
    return m <= 5


def _sme_small_eligible[dtype: DType](m: Int, n: Int, k: Int) -> Bool:
    """True for the small-M SME batch (6 <= M < 16) on the 8x32 tile, which wastes
    far less FMOPA than the 16-row tile below 16 rows. n >= 256 keeps thin-N small-M
    on the NEON no-pack path. Comptime-false off Apple/f64."""
    comptime APPLE_F64 = CompilationTarget.is_apple_silicon() and (
        dtype == DType.float64
    )
    var ok = False
    comptime if APPLE_F64:
        ok = m >= 6 and m < 16 and n >= 256 and (m * n * k >= (1 << 21))
    return ok


def _sme_eligible[dtype: DType](m: Int, n: Int, k: Int) -> Bool:
    """True when the f64 SME coprocessor path (the 16x32 FMOPA micro-kernel) should
    run: Apple Silicon + f64, at least one 16-row tile and one 32-wide column tile,
    and enough work (>= 2^21 MACs) to amortize the streaming-mode entry. The kernel's
    overlap tiles cover non-divisible M/N. Comptime-false off Apple/f64, so the whole
    SME branch compiles away. DESIGN.md "SME"."""
    comptime APPLE_F64 = CompilationTarget.is_apple_silicon() and (
        dtype == DType.float64
    )
    var ok = False
    comptime if APPLE_F64:
        ok = m >= 16 and n >= 32 and (m * n * k >= (1 << 21))
    return ok


def _is_thin_n[dtype: DType](m: Int, n: Int) -> Bool:
    """Thin-N tall-M: N at most 8*NELTS with at least 64 rows. The work is along M,
    so it wants the no-pack kernel's M-parallelism instead of the prefill kernel's
    N-parallelism (which a thin N starves)."""
    comptime NELTS = simd_width_of[dtype]()
    return n <= NELTS * 8 and m >= 64


def _box_fits_l2[dtype: DType](m: Int, n: Int, k: Int) -> Bool:
    """True for a small M-dominant box (m >= 64, m >= n) whose B = k*n stays
    L2-resident, where the no-pack M-parallel kernel beats the packed path (whose
    packing + launch overhead dwarfs the compute on a cache-resident box). Two-tier:
    a compile-time 512 KB cut (no cpuid on the few-us shapes) then the L2-adaptive
    B <= L2/3 cut (_box_l2_budget). m >= n keeps it off every wide headline shape.
    DESIGN.md "_box_l2_budget"."""
    var b_bytes = k * n * size_of[Scalar[dtype]]()
    return (
        m >= 64
        and m >= n
        and (b_bytes <= (1 << 19) or b_bytes <= _box_l2_budget())
    )


def _is_narrow_n(n: Int) -> Bool:
    """Narrow N: N at most 192, too few TILE_N=64 j-tiles to fill the cores without
    dropping to the narrow NR=16 tile."""
    return n <= 3 * 64


def _is_square_ish(m: Int, n: Int) -> Bool:
    """Square-ish: N at most M (with N > 192 already, from the narrow-N gate above)."""
    return n <= m


def _is_wide_n(n: Int, k: Int) -> Bool:
    """Wide-N (up-proj-like): N at least K (with N > M already, from the square-ish
    gate above). The remaining shapes (N < K) are tall-K, the cascade's else."""
    return n >= k


# ===========================================================================
# Per-regime kernel selection
#
# One helper per dispatch regime: each picks the kernel + tile (KC, TILE_N, MR,
# SHARED_A) that measured fastest for its shape band and forwards to the kernel.
# Lifting the comptime tile picks out here keeps `matmul_dispatch` a flat regime
# cascade. The full per-branch measurements live in README.md / DESIGN.md.
# ===========================================================================


@always_inline
def _small_batch[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Small-batch decode (2 <= M <= 5): the packed micro-kernel with MR = M, so the
    packed B panel is streamed once and reused across all M rows (a per-row GEMV
    would re-stream B, ~2x slower at M=4). KU=2, KC=256, TILE_N=8*NELTS."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    if m == 2:
        _packed_gemm[dtype, 2, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 3:
        _packed_gemm[dtype, 3, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 4:
        _packed_gemm[dtype, 4, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    else:
        _packed_gemm[dtype, 5, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)


@always_inline
def _sme_small[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Small-M SME batch (6 <= M < 16) on the 8x32 FMOPA tile. f64 + Apple Silicon
    only; the body compiles away elsewhere (guarded callers never reach it)."""
    var m = a.rows
    var n = c.cols
    var k = a.cols
    comptime APPLE_F64 = CompilationTarget.is_apple_silicon() and (
        dtype == DType.float64
    )
    comptime if APPLE_F64:
        sme_gemm_small_ptr(
            c.data.unsafe_ptr().bitcast[Float64](),
            a.data.unsafe_ptr().bitcast[Float64](),
            b.data.unsafe_ptr().bitcast[Float64](),
            m, n, k, 2,
        )


@always_inline
def _sme[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Apple SME (matrix coprocessor) f64 path: the 16x32 FMOPA micro-kernel that
    breaks the ~515 GFLOPS NEON ceiling (two SME units on M4 Max). f64 + Apple
    Silicon only; compiles away elsewhere. Picks (KC, MC): one full sweep (no
    blocking) when A fits L2 with wide N and small K, or when N is narrow; else
    block, so the MC-tall A block stays L2-resident across the worker's j-tiles and
    KC keeps the B k-panel L1-resident. DESIGN.md "SME"."""
    var m = a.rows
    var n = c.cols
    var k = a.cols
    comptime APPLE_F64 = CompilationTarget.is_apple_silicon() and (
        dtype == DType.float64
    )
    comptime if APPLE_F64:
        var BIG = 1 << 30
        var kc = BIG
        var mc = BIG
        var a_fits_wide = n >= 4096 and m * k * 8 <= (12 << 20) and k <= 2048
        if not (a_fits_wide or n < 192):
            if k >= 4096:
                kc = 384
                mc = 256
            else:
                kc = 512
                mc = 128
        sme_gemm_ptr(
            c.data.unsafe_ptr().bitcast[Float64](),
            a.data.unsafe_ptr().bitcast[Float64](),
            b.data.unsafe_ptr().bitcast[Float64](),
            m, n, k, 2, kc, mc,
        )


@always_inline
def _thin_n[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Thin-N tall-M (N <= 8*NELTS, M >= 64): the work is along M, so parallelize
    over M-row blocks (the prefill kernel parallelizes only over N and starves the
    cores). Reads A/B from source; a thin N stays cache-resident, so packing buys
    nothing. NR_VECS=1 for sub-2*NELTS N, else NR_VECS=2."""
    comptime NELTS = simd_width_of[dtype]()
    var n = c.cols
    if n < 2 * NELTS:
        _nopack_gemm[dtype, 6, 1](c, a, b)
    else:
        _nopack_gemm[dtype, 6, 2](c, a, b)


@always_inline
def _small_box[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Small M-dominant box whose B is L2-resident (see _box_fits_l2).

    The bigger boxes (B ~ 512 KB, the top of the admission window) take the same
    pack-B-only / TileK=K path as the squares: packing B once and reading A
    unpacked beats re-reading all of B per MR-row block once B is large, as long
    as N splits into >= num_workers NR-tiles to fill the cores. Measured
    bonly/no-pack: sq256 1.14, 512x128x512 1.10, 256x128x512 1.11 (2-sigma WIN),
    while the genuinely small boxes lose it (sq192 0.76, sq128 0.95, sq96 0.82) and
    keep the no-pack M-parallel kernel. The cut is B > 384 KB. DESIGN.md.

    No-pack uses NR=32, MR=6 (24 accumulators, deepest ILP) when it divides M with
    no tail, else MR=4 (divides every multiple-of-4 box). DESIGN.md "_nopack_gemm MR"."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var b_bytes = k * n * size_of[Scalar[dtype]]()
    if b_bytes > (3 << 17) and ceildiv(n, 4 * NELTS) >= compute_core_count():
        # k <= 512 here (B <= 512 KB with N >= 4*NELTS), so KC=512 is a single
        # whole-K panel: C is swept over all of K and stored once.
        _prefill[dtype, 512, 4 * NELTS, False, False](c, a, b)
    elif m % 6 == 0:
        _nopack_gemm[dtype, 6, 4](c, a, b)
    else:
        _nopack_gemm[dtype, 4, 4](c, a, b)


@always_inline
def _narrow_n[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Narrow N (N <= 192): a narrow NR=16 / TILE_N=16 tile so a small N still splits
    into >= num_workers j-tiles instead of idling most cores. KU=4."""
    comptime NELTS = simd_width_of[dtype]()
    _packed_gemm[dtype, 6, 2 * NELTS, 256, 4, 2 * NELTS, 64](c, a, b)


@always_inline
def _square_ish[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Square-ish (192 < N <= M): the pack-B-only GEMM that matches stdlib linalg's
    edge here (found by reading linalg/matmul/cpu + utils and emit-asm). On a square
    A is as large as B, so packing A is pure overhead and the packed-A buffer
    competes with B/C in cache; linalg packs ONLY B and reads A unpacked (strided
    column broadcasts), and sets TileK = min(K, 2048) so each C micro-tile is swept
    over the whole K and STORED ONCE. We do the same: PACK_A=False, KC at the rung
    >= min(K, 2048), and TileN shrunk so the packed-B tile (TileN x KC) stays within
    a ~512 KB pack budget (TileN = 128 / 64 / 32 as K = 512 / 1024 / 2048). Measured
    interleaved A/B vs the old pack-both square path: sq512 +19%, sq768 +14%,
    sq1024 +12%, sq1536 +12%, sq2048 +6.5%.

    The pack-B-only path parallelizes over N (j-tiles), so it needs >= num_workers
    j-tiles to fill the cores. A small-N square-ish shape (or sq384, whose budget
    TileN=128 leaves only 3 tiles) keeps the old pack-both path. DESIGN.md
    "Square-ish: pack-B-only / TileK=K"."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var num_workers = compute_core_count()

    # linalg's budget rule: TileK = min(K, 2048) (C stored once), TileN shrunk so
    # the packed-B tile stays ~512 KB. Snap K to the comptime KC rung that still
    # covers it in a single panel.
    comptime TN_K512 = 16 * NELTS   # 128: packed-B 128 x 512 x 8 = 512 KB
    comptime TN_K1024 = 8 * NELTS   # 64:  64 x 1024 x 8 = 512 KB
    comptime TN_K2048 = 4 * NELTS   # 32:  32 x 2048 x 8 = 512 KB

    if k <= 512 and ceildiv(n, TN_K512) >= num_workers:
        _prefill[dtype, 512, TN_K512, False, False](c, a, b)
    elif k <= 1024 and ceildiv(n, TN_K1024) >= num_workers:
        _prefill[dtype, 1024, TN_K1024, False, False](c, a, b)
    elif k > 1024 and ceildiv(n, TN_K2048) >= num_workers:
        _prefill[dtype, 2048, TN_K2048, False, False](c, a, b)
    else:
        # Too few j-tiles for the pack-B-only N-parallel path to fill the cores
        # (small-N square-ish, sq384): keep the old pack-both path, L2-adaptive KC
        # + load-balanced TILE_N (wide 8*NELTS at >= 4 wide j-tiles/worker, else fine).
        comptime TN_WIDE = 8 * NELTS
        comptime TN_FINE = 4 * NELTS
        var njt_wide = ceildiv(n, TN_WIDE)
        var use_wide = njt_wide % num_workers == 0 and njt_wide // num_workers >= 4
        var kc = _square_ish_kc(m, n, k)
        if use_wide:
            _prefill_kc[dtype, TN_WIDE, True](c, a, b, kc)
        else:
            _prefill_kc[dtype, TN_FINE, True](c, a, b, kc)


@always_inline
def _wide_n[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Wide-N (N >= K, up-proj-like): the N-balanced 6x32 tile (TILE_N=8*NELTS). The
    older 8x24 tile only for very wide N with tiny M (the Qwen up-proj small batch).
    Small M packs A per worker; past the M~192/288 crossover SHARED_A; large M takes
    a cache-aware KC (half-L2-resident packed-B tile). DESIGN.md."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    if n >= 9 * 1024 and m <= 32:
        _packed_gemm[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS, 64](c, a, b)
    elif m <= 192:
        comptime if CompilationTarget.is_apple_silicon():
            _prefill[dtype, 256, 8 * NELTS, True](c, a, b)  # EXPERIMENT
        else:
            _prefill[dtype, 256, 8 * NELTS](c, a, b)
    elif m <= 288:
        _prefill[dtype, 512, 8 * NELTS, True](c, a, b)
    else:
        var kc = _l2_resident_kc[dtype](64, k)
        _prefill_kc[dtype, 8 * NELTS, True](c, a, b, kc)


@always_inline
def _tall_k[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Tall-K (N < K, down-proj-like): the uniform 6x32 tile (TILE_N=8*NELTS), whose
    masked M-remainder tail lets it beat 8x24 at every M. Small M packs A per worker;
    past the M~192 crossover SHARED_A; large M takes a cache-aware KC. DESIGN.md."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var k = a.cols
    if m <= 64:
        _prefill[dtype, 256, 8 * NELTS](c, a, b)
    elif m <= 256:
        if m >= 192:
            _prefill[dtype, 512, 8 * NELTS, True](c, a, b)
        else:
            comptime if CompilationTarget.is_apple_silicon():
                _prefill[dtype, 512, 8 * NELTS, True](c, a, b)  # EXPERIMENT
            else:
                _prefill[dtype, 512, 8 * NELTS](c, a, b)
    else:
        var kc = _l2_resident_kc[dtype](64, k)
        _prefill_kc[dtype, 8 * NELTS, True](c, a, b, kc)


# ===========================================================================
# Dispatch
# ===========================================================================


def matmul_dispatch[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Compute C = A * B, routed to the kernel + tile that measured fastest for the
    shape on a 4-core AVX-512 Xeon (f64). This table is the authoritative dispatch
    map; each row's helper holds the tile picks and the per-branch rationale:

        tiny  M*N*K < _tiny_cutoff      _serial_gemm  serial, no threads/packing
        M == 1                          _decode_gemv  j-parallel GEMV, streams B once
        M in 2..5                       _small_batch  packed MR=M, reuse B across rows
        SME small  6<=M<16 (Apple f64)  _sme_small    8x32 FMOPA coprocessor tile
        SME  M>=16 (Apple f64)          _sme          16x32 FMOPA coprocessor tile
        thin-N  N<=8*NELTS, M>=64       _thin_n       M-parallel, no packing
        small box  M>=N, B fits L2      _small_box    no-pack (pack-B-only at B>384KB)
        narrow-N  N<=192                _narrow_n     packed NR=16 -> >= 4 j-tiles
        square-ish  N<=M                _square_ish   pack-B-only 6x32, TileK=K
        wide-N  N>=K                    _wide_n       packed 6x32 (8x24 for N>=9k, M<=32)
        tall-K  N<K                     _tall_k       packed 6x32, cache-aware KC

    The N<=M and M>=N gates keep square-ish and the no-pack routes off every wide
    headline shape (the Qwen up/down projections are N >> M), where those kernels
    are catastrophic. Every tile pick is hardware-specific; the thresholds that
    differ across CPUs are isolated in the "Per-hardware dispatch knobs" helpers,
    and the full rationale + measurements are in README.md and DESIGN.md.

    On Apple Silicon the SME paths, P-core-only parallelism (compute_core_count),
    the box-budget cap, the 2^18 tiny cutoff (m >= 64), and SHARED_A in the small-M
    bands take over, each behind `comptime is_apple_silicon()` so x86 is
    byte-for-byte unchanged. See DESIGN.md "Apple Silicon" / "SME"."""
    var m = a.rows
    var n = c.cols
    var k = a.cols
    if _is_tiny(m, n, k):
        _serial_gemm[dtype, 6, 2](c, a, b)
    elif _is_decode(m):
        _decode_gemv(c, a, b)
    elif _is_small_batch(m):
        _small_batch(c, a, b)
    elif _sme_small_eligible[dtype](m, n, k):
        _sme_small(c, a, b)
    elif _sme_eligible[dtype](m, n, k):
        _sme(c, a, b)
    elif _is_thin_n[dtype](m, n):
        _thin_n(c, a, b)
    elif _box_fits_l2[dtype](m, n, k):
        _small_box(c, a, b)
    elif _is_narrow_n(n):
        _narrow_n(c, a, b)
    elif _is_square_ish(m, n):
        _square_ish(c, a, b)
    elif _is_wide_n(n, k):
        _wide_n(c, a, b)
    else:
        _tall_k(c, a, b)


# ===========================================================================
# Design notes
#
# The "why" behind every tuning constant above (SHARED_A, KU=2, the masked
# partial-N panel, the L2/3 no-pack cut, and the per-machine KC picks) lives in
# DESIGN.md, kept out of the source so the kernels read as code. README.md has
# the per-branch dispatch table and the full benchmark results.
# ===========================================================================
