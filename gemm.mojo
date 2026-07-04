from cpu_cache import l2_cache_size
from matrix import Matrix
from tile import Tile
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.math import ceildiv
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

    @always_inline
    def load_masked[
        org: MutOrigin
    ](mut self, c: Tile[Self.dtype, org], rows: Int, cols: Int):
        """`load`, restricted to the first `rows` x `cols` of the block; the
        masked-out lanes keep their zero from `__init__`."""
        comptime for mr in range(Self.MR):
            if mr < rows:
                var row = c.row(mr)
                comptime for nr in range(Self.NR_VECS):
                    var col0 = nr * Self.NELTS
                    if col0 + Self.NELTS <= cols:
                        self.acc[mr * Self.NR_VECS + nr] = row.load[
                            width = Self.NELTS
                        ](offset=col0)
                    elif col0 < cols:
                        var v = SIMD[Self.dtype, Self.NELTS](0)
                        for e in range(cols - col0):
                            v[e] = row[col0 + e]
                        self.acc[mr * Self.NR_VECS + nr] = v

    @always_inline
    def store_masked[
        org: MutOrigin
    ](self, c: Tile[Self.dtype, org], rows: Int, cols: Int):
        """`store`, restricted to the first `rows` x `cols` of the block."""
        comptime for mr in range(Self.MR):
            if mr < rows:
                var row = c.row(mr)
                comptime for nr in range(Self.NR_VECS):
                    var col0 = nr * Self.NELTS
                    if col0 + Self.NELTS <= cols:
                        row.store(
                            offset=col0, val=self.acc[mr * Self.NR_VECS + nr]
                        )
                    elif col0 < cols:
                        var v = self.acc[mr * Self.NR_VECS + nr]
                        for e in range(cols - col0):
                            row[col0 + e] = v[e]


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
    rows: Int,
    cols: Int,
    is_first_k: Bool,
):
    """Cold-path register tile: the leftover blocks the hot loop can't take.

    Handles an M-remainder (only `rows` of MR rows active) and/or a partial
    NR-panel (only `cols` of NR columns valid) by masking the C load and store;
    reads A unpacked (so it covers the un-packed remainder rows too) and B from
    the zero-padded packed panel. With rows == MR and cols == NR it is the
    unmasked full kernel.

    `c_block`/`a_block` are `sub`-views onto the block's corner, (i, j0+jr) in C
    and (i, pc) in A."""
    var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
    if not is_first_k:
        tile.load_masked(c_block, rows, cols)
    for pk in range(kc):
        # A is gathered here (not via load_a_col) because the gather is guarded
        # by mr < rows: rows past the M-remainder are out of bounds, so leave
        # them zero (their acc lanes are never stored).
        var a_col = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))
        comptime for mr in range(MR):
            if mr < rows:
                a_col[mr] = a_block.row(mr)[pk]
        tile.rank1_update(a_col, load_b_row[NR_VECS, NELTS](bp_panel + pk * NR))
    tile.store_masked(c_block, rows, cols)


@always_inline
def _full_microkernel[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int, NR: Int, KU: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: MutOrigin,
](
    c_block: Tile[dtype, c_org],
    a_base: UnsafePointer[Scalar[dtype], a_org],
    a_k_step: Int,
    a_stride: Int,
    bp_panel: UnsafePointer[Scalar[dtype], b_org],
    kc: Int,
    is_first_k: Bool,
):
    """Hot-path register tile for one full MR x NR block of C.

    The full-NR-panel counterpart of `_masked_microkernel`: no masking, every
    row and column live. Reads the packed B panel and, per K-step, the MR A
    scalars from `a_base + pk * a_k_step` at element stride `a_stride`: a
    packed-A panel is (panel, MR, 1), a row-major A is (addr(i, pc), 1, k).
    Both are compile-time-known at every call site, so they fold away.

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
    while pk + KU <= kc:
        comptime for ku in range(KU):
            tile.rank1_update(
                load_a_col[MR](a_base + (pk + ku) * a_k_step, a_stride),
                load_b_row[NR_VECS, NELTS](bp_panel + (pk + ku) * NR),
            )
        pk += KU
    while pk < kc:
        tile.rank1_update(
            load_a_col[MR](a_base + pk * a_k_step, a_stride),
            load_b_row[NR_VECS, NELTS](bp_panel + pk * NR),
        )
        pk += 1

    tile.store(c_block)


# ===========================================================================
# The packed prefill GEMM: the workhorse
# ===========================================================================


@always_inline
def _pack_a_panel[
    dtype: DType, MR: Int, a_org: ImmutOrigin, ap_org: MutOrigin
](
    a: Tile[dtype, a_org],
    dst: UnsafePointer[Scalar[dtype], ap_org],
    i0: Int,
    pc: Int,
    kc: Int,
):
    """Pack MR rows of A (top-left (i0, pc), kc deep) as [k][MR], so each
    K-step's MR A scalars are contiguous."""
    for pk in range(kc):
        comptime for mr in range(MR):
            dst[pk * MR + mr] = a.row(i0 + mr)[pc + pk]


@always_inline
def _pack_b_slab[
    dtype: DType, NR: Int, NR_VECS: Int, NELTS: Int,
    b_org: ImmutOrigin, bp_org: MutOrigin,
](
    b: Tile[dtype, b_org],
    bp: UnsafePointer[Scalar[dtype], bp_org],
    pc: Int,
    j0: Int,
    kc: Int,
    tile_n: Int,
):
    """Pack a kc x tile_n slab of B (top-left (pc, j0)) into [panel][k][NR],
    software-prefetching the next k-row. A partial trailing panel is padded out
    to full NR with zeros, so the micro-kernel can run it as a full panel (the
    zero columns contribute nothing and the masked store keeps only the valid
    ones)."""
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
                    val=src.load[width=NELTS](offset=nv * NELTS),
                )
        if rem > 0:
            var src = row + full_panels * NR
            var dst = bp + full_panels * kc * NR + pk * NR
            for j in range(rem):
                dst[j] = src[j]
            for j in range(rem, NR):
                dst[j] = Scalar[dtype](0)


def _packed_gemm[
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
    DESIGN.md)."""
    comptime NELTS = simd_width_of[dtype]()
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
    var bp_buf = alloc[Scalar[dtype]](num_workers * bp_per_worker)

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
    var ap_buf = alloc[Scalar[dtype]](ap_total)

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
                                _full_microkernel[
                                    dtype, MR, NR_VECS, NELTS, NR, KU
                                ](
                                    c_view.sub(i, j0 + jr), ap, MR, 1,
                                    bp_panel, kc, is_first_k,
                                )
                            else:
                                _full_microkernel[
                                    dtype, MR, NR_VECS, NELTS, NR, KU
                                ](
                                    c_view.sub(i, j0 + jr),
                                    a_view.addr(i, pc), 1, k,
                                    bp_panel, kc, is_first_k,
                                )
                        else:
                            _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                                c_view.sub(i, j0 + jr), a_view.sub(i, pc),
                                bp_panel, kc, MR, cols, is_first_k,
                            )

                    # M-remainder (m % MR rows): one masked block reusing the
                    # same packed B panel.
                    var i = num_full_panels * MR
                    if i < m:
                        _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                            c_view.sub(i, j0 + jr), a_view.sub(i, pc),
                            bp_panel, kc, m - i, cols, is_first_k,
                        )

    parallelize(worker, num_workers, num_workers)
    bp_buf.free()
    ap_buf.free()


@always_inline
def _prefill[
    dtype: DType, KC: Int, TILE_N: Int, SHARED_A: Bool = False,
    PACK_A: Bool = True,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """The packed GEMM at its standard 6 x (4*NELTS) register tile (KU=2).
    KC, TILE_N, SHARED_A and PACK_A are the only levers that vary across
    shapes, so naming the rest here keeps each dispatch branch a one-liner."""
    comptime NELTS = simd_width_of[dtype]()
    _packed_gemm[dtype, 6, 4 * NELTS, KC, 2, TILE_N, SHARED_A, PACK_A](c, a, b)


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
def _decode_fma_chunk[
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
    """KU consecutive K-steps of the GEMV, vectorized across the worker's
    column chunk; the tail loop reuses it at KU=1.

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
                _decode_fma_chunk[dtype=dtype, KU=KU, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                p += KU
            while p < k:
                _decode_fma_chunk[dtype=dtype, KU=1, NELTS=NELTS](ci, ai, b_col, p, n, chunk)
                p += 1

    parallelize(worker, nw, nw)


# ===========================================================================
# No-pack kernels: serial (tiny) and M-parallel thin-N / small box
#
# Both read A and B straight from source. For a tiny or cache-resident shape
# the data already fits L1/L2, so explicit packing buys nothing and the prefill
# kernel's packing + thread-launch overhead would dominate. Both are the same
# per-block routine, `_nopack_rows`; only the driver differs (a plain loop vs
# `parallelize`), so they are bit-identical to each other and to the packed
# paths.
# ===========================================================================


@always_inline
def _nopack_tail_row[
    dtype: DType, NELTS: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: ImmutOrigin,
](
    c_row: UnsafePointer[Scalar[dtype], c_org],
    a_row: UnsafePointer[Scalar[dtype], a_org],
    b: Tile[dtype, b_org],
    jr: Int,
):
    """N-remainder columns (n % NR) of one C row: NELTS-wide chunks with an
    automatic scalar tail. A top-level def for the same closure-capture reason
    as `_decode_fma_chunk`."""
    def tail[width: Int](jj: Int) {read c_row, read a_row, read b, read jr}:
        var acc = SIMD[dtype, width](0)
        for p in range(b.rows):
            acc = SIMD[dtype, width](a_row[p]).fma(
                b.addr(p, jr).load[width=width](offset=jj), acc
            )
        c_row.store(offset=jr + jj, val=acc)
    vectorize[NELTS](b.cols - jr, tail)


@always_inline
def _nopack_rows[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: ImmutOrigin,
](
    c: Tile[dtype, c_org],
    a: Tile[dtype, a_org],
    b: Tile[dtype, b_org],
    i: Int,
    r: Int,
):
    """Rows [i, i+r) of C across the full N, reading A and B straight from
    source. A full MR-row block (r == MR) uses the comptime-unrolled MR tile
    (each B-load reused across MR rows); a tail block (m % MR) computes one row
    at a time with the same NR-wide SIMD."""
    comptime NR = NR_VECS * NELTS
    var n = c.cols
    var k = a.cols

    if r == MR:
        var j = 0
        while j + NR <= n:
            var tile = RegisterTile[dtype, MR, NR_VECS, NELTS]()
            for p in range(k):
                tile.rank1_update(
                    load_a_col[MR](a.addr(i, p), a.stride),
                    load_b_row[NR_VECS, NELTS](b.addr(p, j)),
                )
            tile.store(c.sub(i, j))
            j += NR
    else:
        for ii in range(i, i + r):
            var j = 0
            while j + NR <= n:
                var tile = RegisterTile[dtype, 1, NR_VECS, NELTS]()
                for p in range(k):
                    tile.rank1_update(
                        load_a_col[1](a.addr(ii, p), a.stride),
                        load_b_row[NR_VECS, NELTS](b.addr(p, j)),
                    )
                tile.store(c.sub(ii, j))
                j += NR

    var jr = (n // NR) * NR
    if jr < n:
        for ii in range(i, i + r):
            _nopack_tail_row[dtype, NELTS](c.row(ii), a.row(ii), b, jr)


def _serial_gemm[
    dtype: DType, MR: Int, NR_VECS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Serial register-tiled GEMM for tiny shapes (no threads, no packing).

    Below the dispatch's tiny cutoff the parallel kernels' fixed cost (thread
    launch + per-worker buffers + packing) dwarfs the compute, running
    10-100x slower than this plain serial loop. MR=6, NR_VECS=2 measured
    best."""
    comptime NELTS = simd_width_of[dtype]()
    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    for i in range(0, m, MR):
        _nopack_rows[dtype, MR, NR_VECS, NELTS](
            c_view, a_view, b_view, i, min(MR, m - i)
        )


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
    straight from source."""
    comptime NELTS = simd_width_of[dtype]()
    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows

    def worker(blk: Int) {read c_view, read a_view, read b_view, read m}:
        var i = blk * MR
        _nopack_rows[dtype, MR, NR_VECS, NELTS](
            c_view, a_view, b_view, i, min(MR, m - i)
        )

    parallelize(worker, ceildiv(m, MR), num_physical_cores())


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
    return min(min(l2 // (tile_n * elem), k), cap)


def _box_l2_budget() -> Int:
    """Upper bound (bytes of B = k*n) for routing an M-dominant box to the
    no-pack `_nopack_gemm`. That kernel re-reads all of B once per MR-row
    block, so B must stay L2-resident alongside the A panel, C, and prefetch
    headroom across the whole M-sweep, which holds only while B is ~1/3 of L2.
    Falls back to 512 KB when L2 is undetectable."""
    var l2 = l2_cache_size()
    if l2 == 0:
        return 1 << 19
    return l2 // 3


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
    comptime for MR in range(2, 6):
        if a.rows == MR:
            _packed_gemm[dtype, MR, 4 * NELTS, 256, 2, 8 * NELTS](c, a, b)
            return


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
    var bp_buf = alloc[Scalar[dtype]](num_workers * k * NR)

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
                _full_microkernel[dtype, MR, NR_VECS, NELTS, NR, KU](
                    c_view.sub(i0, jr), a_view.addr(i0, 0), 1, k,
                    bp_panel, k, True,
                )
            else:
                # M-remainder rows and/or a partial trailing column: the masked
                # kernel reads A unpacked and stores only the live rows/columns.
                _masked_microkernel[dtype, MR, NR_VECS, NELTS, NR](
                    c_view.sub(i0, jr), a_view.sub(i0, 0), bp_panel, k,
                    rows, cols, True,
                )

    parallelize(worker, num_workers, num_workers)
    bp_buf.free()


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
        _packed_gemm[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS](c, a, b)
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
        _prefill_kc[dtype, 8 * NELTS, True](c, a, b, _l2_resident_kc[dtype](64, k))


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
    for the shape on a 4-core AVX-512 Xeon (f64). This cascade is the
    authoritative dispatch map; each regime's helper holds the tile picks and
    the per-branch rationale.

        tiny  M*N*K < 2^19       _serial_gemm  serial, no threads/packing
        M == 1                   _decode_gemv  j-parallel GEMV, streams B once
        M in 2..5                _small_batch  packed MR=M, reuse B across rows
        thin-N  N<=8*NELTS, M>=64  _thin_n     M-parallel no-pack
        small box  M>=N, B fits L2 _small_box  no-pack (pack-B-only at B>384KB)
        narrow-N  N<=192         packed NR=16  too few TILE_N=64 j-tiles
        square-ish  N<=M         _square_ish   pack-B-only 6x32, TileK=K
        wide-N  N>=K             _wide_n       packed 6x32 (pack-B-only 2D for
                                               small L3-resident box)
        tall-K  N<K              _tall_k       packed 6x32, cache-aware KC

    The N<=M and M>=N gates keep square-ish and the no-pack routes off every
    wide headline shape (the LLM up/down projections are N >> M), where those
    kernels are catastrophic. The full rationale + measurements are in
    README.md and DESIGN.md."""
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    if m * n * k < (1 << 19):
        _serial_gemm[dtype, 6, 2](c, a, b)
    elif m == 1:
        _decode_gemv(c, a, b)
    elif m <= 5:
        _small_batch(c, a, b)
    elif n <= 8 * NELTS and m >= 64:
        _thin_n(c, a, b)
    elif _box_fits_l2[dtype](m, n, k):
        _small_box(c, a, b)
    elif n <= 192:
        # Narrow N: the NR=16 / TILE_N=16 tile (KU=4) so a small N still splits
        # into enough j-tiles to fill the cores.
        _packed_gemm[dtype, 6, 2 * NELTS, 256, 4, 2 * NELTS](c, a, b)
    elif n <= m:
        _square_ish(c, a, b)
    elif n >= k:
        _wide_n(c, a, b)
    else:
        _tall_k(c, a, b)
