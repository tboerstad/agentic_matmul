from cpu_cache import l2_cache_size
from matrix import Matrix
from tile import Tile
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.math import ceildiv, fma
from std.memory import memset_zero
from std.memory.unsafe_pointer import alloc
from std.sys import num_physical_cores, simd_width_of, size_of
from std.sys.intrinsics import prefetch, PrefetchOptions


# ===========================================================================
# The register tile
#
# Every kernel in this file is the same idea: hold an MR x (NR_VECS*NELTS)
# block of C entirely in SIMD registers, sweep it over K with rank-1 updates,
# then write it back once. `RegisterTile` is that block. Its accumulator is an
# `InlineArray` the compiler flattens into registers, so each method is a
# zero-cost abstraction that emits exactly the FMA/load/store nest you would
# otherwise hand-write. Each kernel below then expresses only its own packing
# and loop scaffolding.
# ===========================================================================


struct RegisterTile[dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int](
    Copyable
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
        """C_tile += a_col (x) b_row, one K-step: broadcast each of the MR A
        scalars across the NR_VECS B vectors and FMA into the tile. Takes SIMD
        values only (no pointers), so it can never perturb the noalias B-load
        hoisting the hot loops depend on."""
        comptime for mr in range(Self.MR):
            var a_bc = SIMD[Self.dtype, Self.NELTS](a_col[mr])
            comptime for nr in range(Self.NR_VECS):
                self.acc[mr * Self.NR_VECS + nr] = a_bc.fma(
                    b_row[nr], self.acc[mr * Self.NR_VECS + nr]
                )

    @always_inline
    def load[org: MutOrigin](mut self, c: Tile[Self.dtype, org]):
        """Seed the tile from a C block, to accumulate onto a prior K-panel's
        partial. `c.row(0)` is the block's top-left, rows c.stride apart."""
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
# contiguous B vectors and MR A scalars. These two loaders name that gather
# once, so every kernel's inner loop collapses to a single readable line:
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
    the unmasked full kernel.

    `c_block`/`a_block` are `sub`-views onto the block's corner, (i, j0+jr) in C
    and (i, pc) in A."""
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


@always_inline
def _full_microkernel[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int, NR: Int, KU: Int,
    PACK_A: Bool,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: MutOrigin, ap_org: MutOrigin,
](
    c_block: Tile[dtype, c_org],
    ap_panel: UnsafePointer[Scalar[dtype], ap_org],
    a_col_base: UnsafePointer[Scalar[dtype], a_org],
    a_stride: Int,
    bp_panel: UnsafePointer[Scalar[dtype], b_org],
    kc: Int,
    is_first_k: Bool,
):
    """Hot-path register tile for one full MR x NR block of C.

    The full-NR-panel counterpart of `_masked_microkernel`: no masking, every
    row and column live. Reads the packed B panel and, per K-step, the MR A
    scalars either from the packed-A panel at unit stride (PACK_A) or straight
    from a row-major A at column stride `a_stride` (the no-A-pack gather). The
    unused A source is ignored by the comptime branch.

    The K-sweep is unrolled by KU so KU*NR_VECS B-vectors stay live per step
    (KU=2 keeps the 6x32 tile's 24 + 8 = 32 accumulator+B vectors inside the
    AVX-512 register file; KU=4 needs 40 and spills). The B loads stay inline
    so the noalias B-load hoist holds."""
    comptime for mr in range(MR):
        prefetch[PrefetchOptions().for_write().high_locality().to_data_cache()](
            c_block.row(mr)
        )

    var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
    if not is_first_k:
        tile.load(c_block)

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
                    load_a_col[MR](a_col_base + step, a_stride),
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
                load_a_col[MR](a_col_base + pk, a_stride),
                load_b_row[NR_VECS, NELTS](bp_panel + pk * NR),
            )
        pk += 1

    tile.store(c_block)


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
    ones)."""
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
    re-pack it per K-panel. Worth it when A is large next to the N-sweep (big
    squares, large M); off the wide/tall shapes A is tiny and the redundant
    per-worker pack is cheap (see DESIGN.md)."""
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
    var num_workers = num_physical_cores()

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
            # Pre-pack the full A once, in parallel over full MR-row i-panels.
            # Each panel block is MR*k contiguous, organized [k][MR] so the
            # micro-kernel's ap_panel + pk*MR + mr indexing matches the
            # per-worker layout (full-k stride + pc offset instead of a
            # per-panel kc stride).
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
                # elements. Skipped under SHARED_A (packed once up front) and
                # under not-PACK_A (the micro-kernel reads A unpacked).
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

                    # Pack this j-tile's slab of B into [panel][k][NR] (a
                    # partial trailing panel is zero-padded to full NR).
                    _pack_b_slab[dtype, NR, NR_VECS, NELTS, PREFETCH_B_DIST](
                        b_view, bp_worker, pc, j0, kc,
                        last_full_panel, has_remainder, nr_actual,
                    )

                    for jp in range(num_panels):
                        var jr = jp * NR
                        bp_panel = bp_worker + jp * kc * NR

                        if jr + NR > tile_n:
                            # Partial NR-panel (tile_n not a multiple of NR).
                            # The packed panel is zero-padded to full NR, so run
                            # the SAME register-tiled micro-kernel (the zero
                            # columns contribute nothing) and store back only
                            # the jj_limit valid columns: full MR-row i-panels
                            # first (r = MR), then the m % MR remainder rows.
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
                        # PACK_A reads the packed panel at unit stride (SHARED_A:
                        # full-k stride per i-panel + pc offset; else per-worker
                        # per-kc layout); not-PACK_A reads A straight from source
                        # at stride k (a column gather).
                        i = 0
                        ip = 0
                        while i + MR <= m:
                            var ap_panel = (
                                ap_worker + ip * MR * k + pc * MR
                            ) if SHARED_A else (ap_worker + ip * MR * kc)
                            _full_microkernel[
                                dtype, MR, NR_VECS, NELTS, NR, KU, PACK_A
                            ](
                                c_view.sub(i, j0 + jr), ap_panel,
                                a_view.addr(i, pc), k, bp_panel, kc, is_first_k,
                            )
                            i += MR
                            ip += 1

                        # M-remainder (m % MR rows): one register-tiled block
                        # reusing the packed B panel at full NR width (jj_limit
                        # = NR, since the partial-tile case `continue`d above).
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
    """The packed GEMM at its standard 6 x (4*NELTS) register tile (KU=2,
    NC_TILES=64). KC, TILE_N, SHARED_A and PACK_A are the only levers that vary
    across shapes, so naming the rest here keeps each dispatch branch a
    one-liner."""
    comptime NELTS = simd_width_of[dtype]()
    _packed_gemm[dtype, 6, 4 * NELTS, KC, 2, TILE_N, 64, SHARED_A, PACK_A](
        c, a, b
    )


@always_inline
def _prefill_kc[
    dtype: DType, TILE_N: Int, SHARED_A: Bool = False, PACK_A: Bool = True
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype], kc: Int):
    """Run `_prefill` at the comptime KC rung nearest the runtime cache-aware
    `kc`. KC must be a compile-time constant for the micro-kernel, so the
    measured rungs {512, 1024, 2048} are spelled out here, snapping down."""
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
            # streams are n elements apart, too far for the HW prefetcher. May
            # reach past the end of B on the last block; prefetch is
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
    per-k working set of B plus C columns fits L1, so B streams past exactly
    once and no reduction is needed."""
    comptime assert KU > 0, "KU must be positive"
    comptime assert dtype.is_floating_point(), "GEMV requires floating-point dtype"
    comptime NELTS = simd_width_of[dtype]()

    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols
    var nw = num_physical_cores()

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
    10-100x slower than this plain serial loop. Bit-identical to the parallel
    kernels. MR=6, NR_VECS=2 measured best."""
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
        N, so a thin N starves the cores. Here the work is along M.
      * small M-dominant box whose B fits L2: the prefill kernel's packing +
        per-worker buffers + launch overhead dwarfs the compute when the whole
        problem is cache-resident.

    Every core owns a band of C's rows and sweeps the full N, reading A/B
    straight from source. Bit-identical to the other paths."""
    comptime NELTS = simd_width_of[dtype]()
    comptime NR = NR_VECS * NELTS
    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols
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
        # strip, so a scalar tail costs nothing.
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


def _l2_resident_kc[dtype: DType](tile_n: Int, k: Int, cap: Int = 2048) -> Int:
    """Cache-aware KC for the prefill GEMM's large-M band: size the packed-B
    tile (TILE_N x KC) to the per-core L2, then cap at a single `cap`-deep
    k-panel. The default cap=2048 means K <= 2048 sweeps in one panel, so each
    L2-resident C micro-tile is stored exactly once instead of once per
    k-panel. The tall-K band passes cap=1024: with K far above 2048 the
    stored-once benefit is unreachable, while a 2048-deep packed-A panel
    overflows L2. See DESIGN.md "_l2_resident_kc"."""
    comptime elem = size_of[Scalar[dtype]]()
    var l2 = l2_cache_size()
    if l2 == 0:
        return min(cap, k)
    var budget = l2 // (tile_n * elem)
    return min(min(budget, k), cap)


def _box_l2_budget() -> Int:
    """Upper bound (bytes of B = k*n) for routing an M-dominant box to the
    no-pack `_nopack_gemm`. That kernel re-reads all of B once per MR-row
    block, so B must stay L2-resident alongside the A panel, C, and prefetch
    headroom across the whole M-sweep, which holds only while B is ~1/3 of L2.
    Falls back to 512 KB when L2 is undetectable."""
    var l2 = l2_cache_size()
    if l2 == 0:
        return (1 << 19)
    return l2 // 3


# ===========================================================================
# Dispatch predicates
#
# One named gate per regime, tested top to bottom in matmul_dispatch below.
# Each returns only its own condition; the regime it selects is that condition
# with every gate above it having already failed (so _is_square_ish is just
# `n <= m`, because the narrow-N gate above it has ruled out n <= 192). The
# measured rationale is in README.md / DESIGN.md.
# ===========================================================================


def _is_tiny(m: Int, n: Int, k: Int) -> Bool:
    """Tiny shape: total work below the serial/parallel crossover, where the
    parallel kernels' launch + packing overhead dwarfs the compute."""
    return m * n * k < (1 << 19)


def _is_decode(m: Int) -> Bool:
    """Decode GEMV: a single output row (M == 1)."""
    return m == 1


def _is_small_batch(m: Int) -> Bool:
    """Small-batch decode: 2..5 rows (M == 1 is already the GEMV), few enough
    to reuse one packed B panel across all rows with MR = M."""
    return m <= 5


def _is_thin_n[dtype: DType](m: Int, n: Int) -> Bool:
    """Thin-N tall-M: N at most 8*NELTS with at least 64 rows. The work is
    along M, so it wants the no-pack kernel's M-parallelism instead of the
    prefill kernel's N-parallelism (which a thin N starves)."""
    comptime NELTS = simd_width_of[dtype]()
    return n <= NELTS * 8 and m >= 64


def _box_fits_l2[dtype: DType](m: Int, n: Int, k: Int) -> Bool:
    """True for a small M-dominant box (m >= 64, m >= n) whose B = k*n stays
    L2-resident, where the no-pack M-parallel kernel beats the packed path.
    Two-tier: a compile-time 512 KB cut (no cpuid on the few-us shapes) then
    the L2-adaptive B <= L2/3 cut. m >= n keeps it off every wide headline
    shape. DESIGN.md "_box_l2_budget"."""
    var b_bytes = k * n * size_of[Scalar[dtype]]()
    return (
        m >= 64
        and m >= n
        and (b_bytes <= (1 << 19) or b_bytes <= _box_l2_budget())
    )


def _is_narrow_n(n: Int) -> Bool:
    """Narrow N: N at most 192, too few TILE_N=64 j-tiles to fill the cores
    without dropping to the narrow NR=16 tile."""
    return n <= 3 * 64


def _is_square_ish(m: Int, n: Int) -> Bool:
    """Square-ish: N at most M (with N > 192 already, from the narrow-N gate
    above)."""
    return n <= m


def _is_wide_n(n: Int, k: Int) -> Bool:
    """Wide-N (up-proj-like): N at least K (with N > M already, from the
    square-ish gate above). The remaining shapes (N < K) are tall-K, the
    cascade's else."""
    return n >= k


# ===========================================================================
# Per-regime kernel selection
#
# One helper per dispatch regime: each picks the kernel + tile (KC, TILE_N, MR,
# SHARED_A) that measured fastest for its shape band and forwards to the
# kernel. The full per-branch measurements live in README.md / DESIGN.md.
# ===========================================================================


@always_inline
def _small_batch[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Small-batch decode (2 <= M <= 5): the packed micro-kernel with MR = M,
    so the packed B panel is streamed once and reused across all M rows (a
    per-row GEMV would re-stream B, ~2x slower at M=4)."""
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
def _thin_n[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Thin-N tall-M (N <= 8*NELTS, M >= 64): parallelize over M-row blocks
    (the prefill kernel parallelizes only over N and starves the cores). Reads
    A/B from source; a thin N stays cache-resident, so packing buys nothing."""
    comptime NELTS = simd_width_of[dtype]()
    var n = c.cols
    if n < 2 * NELTS:
        _nopack_gemm[dtype, 6, 1](c, a, b)
    elif n % (4 * NELTS) == 0:
        _nopack_gemm[dtype, 6, 4](c, a, b)
    else:
        _nopack_gemm[dtype, 6, 2](c, a, b)


@always_inline
def _small_box[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Small M-dominant box whose B is L2-resident (see _box_fits_l2).

    The bigger boxes (B above 384 KB, the top of the admission window) take
    the same pack-B-only / TileK=K path as the squares: packing B once and
    reading A unpacked beats re-reading all of B per MR-row block, as long as
    N splits into enough NR-tiles to fill the cores. The genuinely small boxes
    keep the no-pack M-parallel kernel: MR=6 (24 accumulators, deepest ILP)
    when it divides M with no tail, else MR=4. DESIGN.md "_nopack_gemm MR"."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var b_bytes = k * n * size_of[Scalar[dtype]]()
    if b_bytes > (3 << 17) and ceildiv(n, 4 * NELTS) >= num_physical_cores():
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
    """Narrow N (N <= 192): a narrow NR=16 / TILE_N=16 tile so a small N still
    splits into enough j-tiles instead of idling most cores. KU=4."""
    comptime NELTS = simd_width_of[dtype]()
    _packed_gemm[dtype, 6, 2 * NELTS, 256, 4, 2 * NELTS, 64](c, a, b)


@always_inline
def _pack_b_only_2d[
    dtype: DType, MR: Int, NR: Int, KU: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Pack-B-only GEMM parallelized over a 2D (column, MR-row-block) grid.

    The N-parallel pack-B-only path (`_square_ish`) hands each worker whole
    NR-wide columns; when the column count is not a multiple of the worker
    count the last round leaves cores idle, and no TILE_N choice removes that
    (the unit of work is a whole column). This path makes the unit of work one
    MR x NR C tile instead and distributes the columns x row-blocks grid
    evenly across workers in column-major order.

    Each worker packs into a private [k][NR] buffer the columns its contiguous
    tile range touches, reusing it across that column's row blocks, so B is
    packed once per worker-column with at most one shared boundary column
    repacked per worker pair. Single K-panel only: the caller gates this to
    k <= 2048 so each tile is one sweep over the whole K and C is stored once.
    A reads from source unpacked. DESIGN.md "Small-N square"."""
    comptime NELTS = simd_width_of[dtype]()
    comptime NR_VECS = NR // NELTS
    comptime PREFETCH_B_DIST = 8

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
    # worker crosses into each column it owns). An unused packed-A stand-in for
    # the PACK_A=False micro-kernel (the pointer is never read).
    var bp_buf = alloc[Scalar[dtype]](num_workers * k * NR)
    var ap_dummy = alloc[Scalar[dtype]](1)

    def compute_worker(worker_id: Int) {read c_view, read a_view, read b_view, mut bp_buf, read ap_dummy, read m, read n, read k, read num_cols, read num_i_panels, read num_workers, read total_tiles}:
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
                var full = cols == NR
                _pack_b_slab[dtype, NR, NR_VECS, NELTS, PREFETCH_B_DIST](
                    b_view, bp_panel, 0, jr, k,
                    1 if full else 0, not full, 0 if full else cols,
                )
                cur_col = col
            var i0 = ip * MR
            var rows = min(MR, m - i0)
            if rows == MR and cols == NR:
                _full_microkernel[dtype, MR, NR_VECS, NELTS, NR, KU, False](
                    c_view.sub(i0, jr), ap_dummy, a_view.addr(i0, 0), k,
                    bp_panel, k, True,
                )
            else:
                # M-remainder rows and/or a partial trailing column: the masked
                # kernel reads A unpacked and stores only the live rows/columns.
                _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                    c_view.sub(i0, jr), a_view.sub(i0, 0), bp_panel, k,
                    rows, cols, True,
                )

    parallelize(compute_worker, num_workers, num_workers)
    bp_buf.free()
    ap_dummy.free()


@always_inline
def _square_ish[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Square-ish (192 < N <= M): pack-B-only, TileK = min(K, 2048).

    On a square A is as large as B, so packing A is pure overhead and the
    packed-A buffer competes with B/C in cache: pack ONLY B, read A unpacked
    (strided column broadcasts), and sweep each C micro-tile over the whole K
    so it is stored once. TILE_N is the narrow 4*NELTS (one NR-panel per
    j-tile), the finest granularity the N-parallel kernel offers, which
    load-balances the band.

    When ceildiv(N, NR) splits unevenly across the workers the last round
    still idles cores; those shapes route to `_pack_b_only_2d`, which splits
    each column into MR-row blocks so the 2D tile grid balances. The 2D path
    carries a real cost (boundary columns repacked, more masked tiles), so it
    is gated to a single C-stored-once K-panel (k <= 2048) and a makespan it
    cuts by at least 1/8. DESIGN.md "Square-ish: pack-B-only / TileK=K"."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var num_workers = num_physical_cores()
    var num_cols = ceildiv(n, 4 * NELTS)
    var num_i_panels = ceildiv(m, 6)
    var col_makespan = ceildiv(num_cols, num_workers) * num_i_panels
    var grid_makespan = ceildiv(num_cols * num_i_panels, num_workers)
    if k <= 2048 and grid_makespan * 8 <= col_makespan * 7:
        _pack_b_only_2d[dtype, 6, 4 * NELTS, 2](c, a, b)
    else:
        _prefill_kc[dtype, 4 * NELTS, False, False](c, a, b, min(k, 2048))


@always_inline
def _wide_n[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Wide-N (N >= K, up-proj-like): the N-balanced 6x32 tile (TILE_N=8*NELTS).
    The older 8x24 tile only for very wide N with tiny M. Small M packs A per
    worker; past the M~192 crossover SHARED_A with a cache-aware KC
    (L2-resident packed-B tile). DESIGN.md."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    if n >= 9 * 1024 and m <= 32:
        _packed_gemm[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS, 64](c, a, b)
    elif k <= 2048 and k * n <= (1 << 20):
        # Small cache-resident wide box (B = k*n stays in L3, at any M): the
        # prefill path packs B per j-tile per worker, and that packing + launch
        # overhead dwarfs the compute when the whole problem is cache-hot. The
        # pack-B-only 2D grid packs B once per worker-column, reads A unpacked,
        # stores C once (k <= 2048 is one K-panel), and balances the column x
        # row-block tile grid across cores. The k*n cut keeps every wide
        # headline shape on the packed prefill path.
        _pack_b_only_2d[dtype, 6, 4 * NELTS, 2](c, a, b)
    elif m <= 192:
        _prefill[dtype, 256, 8 * NELTS](c, a, b)
    else:
        # m > 192: SHARED_A + a single C-stored-once k-panel (KC = min(K, 2048)
        # at the detected L2, see _l2_resident_kc).
        var kc = _l2_resident_kc[dtype](64, k)
        _prefill_kc[dtype, 8 * NELTS, True](c, a, b, kc)


@always_inline
def _tall_k[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Tall-K (N < K, down-proj-like): the 6x(4*NELTS) register tile whose
    masked M-remainder tail beats 8x24 at every M. Small M (< 192) packs A per
    worker on the wider TILE_N=8*NELTS; past the M~192 SHARED_A crossover the
    heavy band switches to the finer TILE_N=4*NELTS (2x more j-tiles, better
    load balance, smaller packed-B panels) and a KC capped at 1024 (with K far
    above 2048 the C-stored-once benefit of a deeper panel is unreachable,
    while the M*KC packed-A panel would overflow L2). DESIGN.md "Tall-K
    large-M"."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var k = a.cols
    if m <= 64:
        _prefill[dtype, 256, 8 * NELTS](c, a, b)
    elif m < 192:
        _prefill[dtype, 512, 8 * NELTS](c, a, b)
    elif m <= 256:
        # KC=512 single L1-resident k-panel, finer TILE_N.
        _prefill[dtype, 512, 4 * NELTS, True](c, a, b)
    else:
        var kc = _l2_resident_kc[dtype](32, k, 1024)
        _prefill_kc[dtype, 4 * NELTS, True](c, a, b, kc)


# ===========================================================================
# Dispatch
# ===========================================================================


def matmul_dispatch[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Compute C = A * B, routed to the kernel + tile that measured fastest
    for the shape on a 4-core AVX-512 Xeon (f64). This table is the
    authoritative dispatch map; each row's helper holds the tile picks and the
    per-branch rationale.

        tiny  M*N*K < 2^19       _serial_gemm  serial, no threads/packing
        M == 1                   _decode_gemv  j-parallel GEMV, streams B once
        M in 2..5                _small_batch  packed MR=M, reuse B across rows
        thin-N  N<=8*NELTS, M>=64  _thin_n     M-parallel no-pack
        small box  M>=N, B fits L2 _small_box  no-pack (pack-B-only at B>384KB)
        narrow-N  N<=192         _narrow_n     packed NR=16
        square-ish  N<=M         _square_ish   pack-B-only 6x32, TileK=K
        wide-N  N>=K             _wide_n       packed 6x32 (pack-B-only 2D for
                                               small L3-resident box)
        tall-K  N<K              _tall_k       packed 6x32, cache-aware KC

    The N<=M and M>=N gates keep square-ish and the no-pack routes off every
    wide headline shape (the LLM up/down projections are N >> M), where those
    kernels are catastrophic. The full rationale + measurements are in
    README.md and DESIGN.md."""
    var m = a.rows
    var n = c.cols
    var k = a.cols
    if _is_tiny(m, n, k):
        _serial_gemm[dtype, 6, 2](c, a, b)
    elif _is_decode(m):
        _decode_gemv(c, a, b)
    elif _is_small_batch(m):
        _small_batch(c, a, b)
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
