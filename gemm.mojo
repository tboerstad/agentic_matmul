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
# zero-cost abstraction — after `@always_inline` the tile emits exactly the
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
        """C_tile += a_col (x) b_row — one K-step. Broadcast each of the MR A
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
# that gather once. Like the tile itself they are zero-cost — the comptime loop
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
    c_ptr: UnsafePointer[Scalar[dtype], c_org],
    a_ptr: UnsafePointer[Scalar[dtype], a_org],
    bp_panel: UnsafePointer[Scalar[dtype], b_org],
    c_base: Int,
    a_base: Int,
    n: Int,
    k: Int,
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

    c_base = i*n + j0 + jr  (top-left of the block in C);
    a_base = i*k + pc       (top-left of the block's rows in A).
    """
    var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
    if not is_first_k:
        comptime for mr in range(MR):
            if mr < r:
                comptime for nr in range(NR_VECS):
                    var col0 = nr * NELTS
                    var cr = c_ptr + c_base + mr * n
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
                a_col[mr] = a_ptr[a_base + mr * k + pk]
        tile.rank1_update(a_col, b_row)
    comptime for mr in range(MR):
        if mr < r:
            comptime for nr in range(NR_VECS):
                var col0 = nr * NELTS
                var cr = c_ptr + c_base + mr * n
                if col0 + NELTS <= jj_limit:
                    cr.store(offset=col0, val=tile.acc[mr * NR_VECS + nr])
                elif col0 < jj_limit:
                    var v = tile.acc[mr * NR_VECS + nr]
                    for e in range(jj_limit - col0):
                        cr[col0 + e] = v[e]


# ===========================================================================
# The packed prefill GEMM — the workhorse
# ===========================================================================


@always_inline
def _pack_b_slab[
    dtype: DType, NR: Int, NR_VECS: Int, NELTS: Int, PREFETCH_B_DIST: Int,
    b_org: ImmutOrigin, bp_org: MutOrigin,
](
    b_ptr: UnsafePointer[Scalar[dtype], b_org],
    bp_worker: UnsafePointer[Scalar[dtype], bp_org],
    pc: Int,
    j0: Int,
    n: Int,
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
        var row_base = b_ptr + (pc + pk) * n + j0
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b_ptr + (pc + pk + PREFETCH_B_DIST) * n + j0
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
    NC_TILES: Int, SHARED_A: Bool = False,
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

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr().as_noalias_ptr()
    var a_ptr = a.data.unsafe_ptr().as_noalias_ptr()
    var b_ptr = b.data.unsafe_ptr().as_noalias_ptr()

    var num_j_tiles = ceildiv(n, TILE_N)
    var num_i_panels = ceildiv(m, MR)
    var num_workers = num_physical_cores()

    var num_nr_panels = ceildiv(TILE_N, NR)
    var bp_per_worker = num_nr_panels * KC * NR + KU * NR
    var bp_total = num_workers * bp_per_worker
    var bp_buf = alloc[Scalar[dtype]](bp_total)

    var num_full_panels = m // MR
    var ap_per_worker: Int
    var ap_total: Int
    if SHARED_A:
        # One shared copy of packed A, laid out [i-panel][k][MR]. ap_per_worker
        # = 0 so every worker indexes the same buffer (ap_worker == ap_buf).
        ap_per_worker = 0
        ap_total = num_i_panels * MR * k
    else:
        ap_per_worker = num_i_panels * MR * KC
        ap_total = num_workers * ap_per_worker
    var ap_buf = alloc[Scalar[dtype]](ap_total)

    if SHARED_A:
        # Pre-pack the full A once, in parallel over full MR-row i-panels. Each
        # panel block is MR*k contiguous, organized [k][MR] so the micro-kernel's
        # ap_panel + pk*MR + mr indexing matches the per-worker layout (just with
        # a full-k stride + pc offset instead of a per-panel kc stride).
        def pack_a_panel(ip: Int) {mut ap_buf, mut a_ptr, read k}:
            var i0 = ip * MR
            var ap_panel = ap_buf + ip * MR * k
            for pk in range(k):
                var dst = ap_panel + pk * MR
                comptime for mr in range(MR):
                    dst[mr] = a_ptr[(i0 + mr) * k + pk]
        parallelize(pack_a_panel, num_full_panels, num_workers)

    def process_worker(worker_id: Int) {mut c_ptr, mut a_ptr, mut b_ptr, mut bp_buf, mut ap_buf, read m, read n, read k, read num_j_tiles, read num_workers, read bp_per_worker, read ap_per_worker}:
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
                # doubles. Skipped under SHARED_A — A was packed once up front.
                var i = 0
                var ip = 0
                if not SHARED_A:
                    while i + MR <= m:
                        var ap_panel = ap_worker + ip * MR * kc
                        for pk in range(kc):
                            var dst = ap_panel + pk * MR
                            comptime for mr in range(MR):
                                dst[mr] = a_ptr[(i + mr) * k + pc + pk]
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
                    # trailing panel is zero-padded to full NR — see _pack_b_slab).
                    _pack_b_slab[dtype, NR, NR_VECS, NELTS, PREFETCH_B_DIST](
                        b_ptr, bp_worker, pc, j0, n, kc,
                        last_full_panel, has_remainder, nr_actual,
                    )

                    for jp in range(num_panels):
                        var jr = jp * NR
                        bp_panel = bp_worker + jp * kc * NR

                        if jr + NR > tile_n:
                            # Partial NR-panel (tile_n not a multiple of NR — the
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
                                    c_ptr, a_ptr, bp_panel,
                                    i * n + j0 + jr, i * k + pc,
                                    n, k, kc, MR, jj_limit, is_first_k,
                                )
                                i += MR
                            if i < m:
                                _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                                    c_ptr, a_ptr, bp_panel,
                                    i * n + j0 + jr, i * k + pc,
                                    n, k, kc, m - i, jj_limit, is_first_k,
                                )
                            continue

                        # ---- Full NR-panel: the register-tile micro-kernel ----
                        i = 0
                        ip = 0
                        while i + MR <= m:
                            # SHARED_A: full-k stride per i-panel + pc offset into
                            # the one shared pack; else per-worker per-kc layout.
                            var ap_panel = (
                                ap_worker + ip * MR * k + pc * MR
                            ) if SHARED_A else (ap_worker + ip * MR * kc)
                            var c_block = Tile(c_ptr, m, n, n).sub(i, j0 + jr)

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
                                    tile.rank1_update(
                                        load_a_col[MR](ap_panel + step * MR, 1),
                                        load_b_row[NR_VECS, NELTS](bp_panel + step * NR),
                                    )
                                pk += KU

                            while pk < kc:
                                tile.rank1_update(
                                    load_a_col[MR](ap_panel + pk * MR, 1),
                                    load_b_row[NR_VECS, NELTS](bp_panel + pk * NR),
                                )
                                pk += 1

                            tile.store(c_block)

                            i += MR
                            ip += 1

                        # M-remainder (m % MR rows): one register-tiled block, a
                        # single K-sweep with r x NR_VECS accumulators reusing the
                        # packed B panel at full NR width (jj_limit = NR — the
                        # partial-tile case already `continue`d above).
                        if i < m:
                            _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                                c_ptr, a_ptr, bp_panel,
                                i * n + j0 + jr, i * k + pc,
                                n, k, kc, m - i, NR, is_first_k,
                            )

            jt += NC_TILES

    parallelize(process_worker, num_workers, num_workers)
    bp_buf.free()
    ap_buf.free()


@always_inline
def _prefill[
    dtype: DType, KC: Int, TILE_N: Int, SHARED_A: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """The SOTA packed GEMM at its standard 6 x (4*NELTS) register tile (KU=2,
    NC_TILES=64). KC, TILE_N and SHARED_A are the only levers that vary across
    shapes, so naming the rest here keeps each dispatch branch a one-liner."""
    comptime NELTS = simd_width_of[dtype]()
    _packed_gemm[dtype, 6, 4 * NELTS, KC, 2, TILE_N, 64, SHARED_A](c, a, b)


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
            # reach past the end of B on the last block — prefetch is
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

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr().as_noalias_ptr()
    var a_ptr = a.data.unsafe_ptr().as_noalias_ptr()
    var b_ptr = b.data.unsafe_ptr().as_noalias_ptr()
    var nw = num_physical_cores()

    memset_zero(c_ptr, m * n)

    def worker(wid: Int) {mut c_ptr, mut a_ptr, mut b_ptr, read m, read n, read k, read nw}:
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
                _decode_fma_chunk_unrolled[dtype=dtype, KU=KU, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                p += KU

            while p < k:
                _decode_fma_chunk_tail[dtype=dtype, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                p += 1

    parallelize(worker, nw, nw)


# ===========================================================================
# No-pack kernels: serial (tiny) and M-parallel thin-N / small box
#
# Both read A and B straight from source — for a tiny or cache-resident shape
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
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_view = Tile(c.data.unsafe_ptr(), m, n, n)
    var a_view = Tile(a.data.unsafe_ptr(), m, k, k)
    var b_view = Tile(b.data.unsafe_ptr(), k, n, n)
    var b_ptr = b_view.ptr

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

            def tail[width: Int](jj: Int) {mut c_row, read a_row, read b_ptr, read k, read n, read jr}:
                var acc = SIMD[dtype, width](0)
                for p in range(k):
                    acc = fma(
                        SIMD[dtype, width](a_row[p]),
                        (b_ptr + p * n + jr).load[width=width](offset=jj),
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
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr().as_noalias_ptr()
    var a_ptr = a.data.unsafe_ptr().as_noalias_ptr()
    var b_ptr = b.data.unsafe_ptr().as_noalias_ptr()
    var nw = num_physical_cores()
    var num_blocks = ceildiv(m, MR)

    def worker(blk: Int) {mut c_ptr, mut a_ptr, mut b_ptr, read m, read n, read k}:
        var c_view = Tile(c_ptr, m, n, n)
        var a_view = Tile(a_ptr, m, k, k)
        var b_view = Tile(b_ptr, k, n, n)
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
    KC=512 on a 1 MB/core L2 and KC=1024 on a 2 MB/core L2 — each the measured
    best on its machine.

    KC only matters when k > 512, and such a square-ish op (with M*N*K >= 2^28)
    is multi-ms, so the l2_cache_size() probe is < 2.5%. Smaller shapes skip the
    probe and take KC=512, keeping cpuid off the hot path for the few-us shapes."""
    if k <= 512 or m * n * k < (1 << 28):
        return 512
    return 1024 if l2_cache_size() >= (3 << 19) else 512


def _box_l2_budget() -> Int:
    """Upper bound (bytes of B = k*n) for routing an M-dominant box to the no-pack
    `_nopack_gemm`. That kernel re-reads all of B once per MR-row block, so B must
    stay L2-resident *alongside* the packed-A panel, C, and prefetch headroom
    across the whole M-sweep — which holds only while B is ~1/3 of L2. Past that
    B spills mid-sweep and the packed prefill path wins, so cut at L2/3. Falls
    back to a compile-time 512 KB tier when L2 is undetectable."""
    var l2 = l2_cache_size()
    if l2 == 0:
        return (1 << 19)
    return l2 // 3


# ===========================================================================
# Dispatch
# ===========================================================================


def matmul_dispatch[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Compute C = A * B, routed to the kernel + tile that measured fastest for
    the shape on a 4-core AVX-512 Xeon (f64).

        tiny  M*N*K < 2^19          _serial_gemm   serial, no threads/packing
        M == 1                      _decode_gemv   j-parallel GEMV, streams B once
        M in 2..5                   packed MR=M    pack B once, reuse across rows
        thin-N  N<=64, M>=64        _nopack_gemm   M-parallel, no packing
        small box  M>=N, B fits L2  _nopack_gemm   M-parallel, no packing
        N <= 192                    packed NR=16   narrow tile -> >= 4 j-tiles
        square-ish  N <= M          packed 6x32    L2-adaptive KC, SHARED_A
        wide-N  N >= K              packed 6x32    (8x24 only for N>=9k & M<=32)
        tall-K  N < K               packed 6x32    cache-aware KC by M band

    Every KC/TILE_N/tile pick is hardware-specific; the full per-branch rationale
    and measurements live in README.md and DESIGN.md. The N<=M and M>=N gates keep
    the square-ish and no-pack routes off every
    wide headline shape (the Qwen up/down projections are N >> M), where those
    kernels are catastrophic."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols

    if m * n * k < (1 << 19):
        # Tiny: a plain serial register-tiled loop. Below ~2^19 MACs the parallel
        # kernels' fixed cost dwarfs the compute (sq8..32 ran 0.03-0.17x linalg
        # parallel vs 1.2-2.6x serial). Can never fire for a headline shape.
        _serial_gemm[dtype, 6, 2](c, a, b)
    elif m == 1:
        # Pure decode GEMV: each worker owns an L1-resident column chunk of C and
        # streams B exactly once.
        _decode_gemv(c, a, b)
    elif m <= 5:
        # Small-batch decode: the packed micro-kernel with MR = M, so the packed
        # B panel is streamed once and reused across all M rows (the GEMV would
        # re-stream B per row, ~2x slower at M=4). KU=2, TILE_N=8*NELTS.
        if m == 2:
            _packed_gemm[dtype, 2, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
        elif m == 3:
            _packed_gemm[dtype, 3, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
        elif m == 4:
            _packed_gemm[dtype, 4, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
        else:
            _packed_gemm[dtype, 5, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif n <= NELTS * 8 and m >= 64:
        # Thin-N, tall-M (N <= 64): the work is along M, but the prefill kernel
        # parallelizes only over N, starving the cores. _nopack_gemm parallelizes
        # over M-row blocks, reading A/B from source (a thin N stays cache-
        # resident, so packing buys nothing). Lifts the band from 0.10-0.58 to
        # 0.80-1.27 vs linalg. NR_VECS=1 for sub-16-wide N; NR_VECS=2 otherwise.
        if n < 2 * NELTS:
            _nopack_gemm[dtype, 6, 1](c, a, b)
        else:
            _nopack_gemm[dtype, 6, 2](c, a, b)
    elif (
        m >= 64
        and m >= n
        and (
            k * n * size_of[Scalar[dtype]]() <= (1 << 19)
            or k * n * size_of[Scalar[dtype]]() <= _box_l2_budget()
        )
    ):
        # Small M-dominant box whose B stays L2-resident: same no-pack M-parallel
        # kernel (NR=32). The packed kernel's packing + launch overhead dwarfs the
        # compute on a cache-resident box (sq96..256 ran 0.65-0.79 packed; no-pack
        # flips them to 1.0-1.18). The B-fits-L2 test is two-tiered: a compile-time
        # 512 KB cut (no cpuid on the few-us shapes) plus an L2-adaptive B <= L2/3
        # tier (_box_l2_budget). m >= n keeps it off every wide headline shape.
        _nopack_gemm[dtype, 6, 4](c, a, b)
    elif n <= 3 * 64:
        # Narrow N (<= 192): at TILE_N=64 such an N is < 4 j-tiles, idling most of
        # a 4-core box. A narrow NR=16 / TILE_N=16 tile splits N into >= num_workers
        # j-tiles (sq64 0.44->0.83). KU=4 here.
        _packed_gemm[dtype, 6, 2 * NELTS, 256, 4, 2 * NELTS, 64](c, a, b)
    elif n <= m:
        # Square-ish (192 < N <= M). The wide/tall branches' TILE_N=64 + big KC
        # fit a box-shaped C badly (few coarse j-tiles; a fat M*KC packed-A with no
        # C-traffic saving to amortize it). A finer tile + smaller KC keep both
        # packs L2-resident and multiply the j-tile count (sq512 0.74->0.88, sq2048
        # 0.70->0.85). Two levers at their measured best:
        #   * KC by detected L2 (_square_ish_kc): 512 on 1 MB/core, 1024 on 2 MB.
        #   * TILE_N by load balance: the fatter 8*NELTS only when N tiles evenly
        #     across workers (>= 2 each), else the finer 4*NELTS.
        # SHARED_A: here A is as large as B/C, so packing it once is a real win.
        comptime TN_WIDE = 8 * NELTS
        comptime TN_FINE = 4 * NELTS
        var njt_wide = ceildiv(n, TN_WIDE)
        var num_workers = num_physical_cores()
        var use_wide = njt_wide % num_workers == 0 and njt_wide // num_workers >= 2
        var kc = 1024 if _square_ish_kc(m, n, k) >= 1024 else 512
        if use_wide:
            if kc >= 1024:
                _prefill[dtype, 1024, TN_WIDE, True](c, a, b)
            else:
                _prefill[dtype, 512, TN_WIDE, True](c, a, b)
        else:
            if kc >= 1024:
                _prefill[dtype, 1024, TN_FINE, True](c, a, b)
            else:
                _prefill[dtype, 512, TN_FINE, True](c, a, b)
    elif n >= k:
        # Wide-N (up-proj-like). The N-balanced 6x32 tile (TILE_N=64, 32 even
        # j-tiles on N=2048) wins almost everywhere now that the masked
        # M-remainder tail removed the MR-divides-M tax.
        if n >= 9 * 1024 and m <= 32:
            # The one corner where the older 8x24 tile still edges 6x32 by ~2-4%:
            # very wide N and tiny M (the Qwen up-proj small batch). KU=2.
            _packed_gemm[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS, 64](c, a, b)
        elif m <= 192:
            # Small-M band: KC=256, per-worker A pack (below the SHARED_A crossover
            # at M~192, so the M=96 headline keeps the byte-for-byte path).
            _prefill[dtype, 256, 8 * NELTS](c, a, b)
        elif m <= 288:
            # Past the SHARED_A crossover: pack A once (+3%, up-proj M=256).
            _prefill[dtype, 512, 8 * NELTS, True](c, a, b)
        else:
            # Large M: cache-aware KC (half-L2 resident packed-B tile), capped at
            # 1024 when N <= M (a single huge k-panel thrashes the M*KC packed-A
            # of a square-ish shape). M > 288 is always past the SHARED_A crossover.
            var kc = _l2_resident_kc[dtype](64, k)
            if n <= m:
                kc = min(kc, 1024)
            if kc >= 2048:
                _prefill[dtype, 2048, 8 * NELTS, True](c, a, b)
            elif kc >= 1024:
                _prefill[dtype, 1024, 8 * NELTS, True](c, a, b)
            else:
                _prefill[dtype, 512, 8 * NELTS, True](c, a, b)
    else:
        # Tall-K (down-proj-like): the uniform 6x32 tile. TILE_N=64 splits N=2048
        # into 32 even j-tiles (perfect 4c balance), and the masked M-remainder
        # tail lets the balanced tile beat 8x24 at every M.
        if m <= 64:
            # Small M: KC=256, per-worker A pack.
            _prefill[dtype, 256, 8 * NELTS](c, a, b)
        elif m <= 256:
            # KC=512; pack A once past the M~192 crossover (down-proj headline
            # M=96 stays on the per-worker path).
            if m >= 192:
                _prefill[dtype, 512, 8 * NELTS, True](c, a, b)
            else:
                _prefill[dtype, 512, 8 * NELTS](c, a, b)
        else:
            # Large M: cache-aware KC (1024 on 1 MB/core L2, 2048 on 2 MB/core).
            var kc = _l2_resident_kc[dtype](64, k)
            if kc >= 2048:
                _prefill[dtype, 2048, 8 * NELTS, True](c, a, b)
            elif kc >= 1024:
                _prefill[dtype, 1024, 8 * NELTS, True](c, a, b)
            else:
                _prefill[dtype, 512, 8 * NELTS, True](c, a, b)


# ===========================================================================
# Design notes
#
# The "why" behind every tuning constant above — SHARED_A, KU=2, the masked
# partial-N panel, the L2/3 no-pack cut, and the per-machine KC picks — lives in
# DESIGN.md, kept out of the source so the kernels read as code. README.md has
# the per-branch dispatch table and the full benchmark results.
# ===========================================================================
