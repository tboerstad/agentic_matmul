"""Shape-based kernel dispatch: route each GEMM to its fastest kernel.

`matmul_dispatch` is the package's entry point. It routes a C = A * B to one
of the kernels in `packed.mojo`, `gemv.mojo`, `nopack.mojo`, or `amx.mojo`
based on the shape (and dtype), using the cache-aware heuristics defined
here. Each regime helper below picks the kernel + tile (KC, TILE_N, MR,
SHARED_A) that measured fastest for its shape band; the measurement tables
behind every pick live in README.md and docs/DESIGN.md.
"""

from matmul.amx import amx_bf16_gemm, amx_bf16_usable, amx_shape_ok
from matmul.cpu_cache import l2_cache_size
from matmul.gemv import decode_gemv
from matmul.matrix import Matrix
from matmul.microkernel import compute_dtype
from matmul.nopack import nopack_gemm, serial_gemm
from matmul.packed import pack_b_only_2d, packed_gemm
from std.math import ceildiv
from std.sys import num_physical_cores, simd_width_of, size_of


# ===========================================================================
# The standard packed tile
# ===========================================================================


@always_inline
def _prefill[
    dtype: DType, KC: Int, TILE_N: Int, SHARED_A: Bool = False,
    PACK_A: Bool = True,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """The packed GEMM at its standard 6 x (4*NELTS) register tile (KU=2).
    KC, TILE_N, SHARED_A and PACK_A are the only levers that vary across
    shapes, so naming the rest here keeps each dispatch branch a one-liner."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
    packed_gemm[dtype, 6, 4 * NELTS, KC, 2, TILE_N, SHARED_A, PACK_A](c, a, b)


@always_inline
def _prefill_kc[
    dtype: DType, TILE_N: Int, SHARED_A: Bool = False, PACK_A: Bool = True
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype], kc: Int):
    """Run `_prefill` at the comptime KC rung nearest the runtime cache-aware
    `kc`. KC must be a compile-time constant for the micro-kernel, so the
    measured rungs {512, 1024, 2048, 4096} are spelled out here, snapping
    down."""
    if kc >= 4096:
        _prefill[dtype, 4096, TILE_N, SHARED_A, PACK_A](c, a, b)
    elif kc >= 2048:
        _prefill[dtype, 2048, TILE_N, SHARED_A, PACK_A](c, a, b)
    elif kc >= 1024:
        _prefill[dtype, 1024, TILE_N, SHARED_A, PACK_A](c, a, b)
    else:
        _prefill[dtype, 512, TILE_N, SHARED_A, PACK_A](c, a, b)


# ===========================================================================
# Cache-aware tile heuristics
#
# Each picks a KC (K-panel depth) or a routing budget from the detected L2 so
# the working set stays cache-resident across machines. The measurement tables
# behind these numbers live in docs/DESIGN.md.
# ===========================================================================


def _l2_resident_kc[dtype: DType](tile_n: Int, k: Int) -> Int:
    """Cache-aware KC for the packed GEMM's large-M band: size the packed-B
    tile (TILE_N x KC) to HALF the per-core L2, then cap the panel depth at
    16 KB per packed column (KC=2048 in f64, the depth the band was tuned
    at). The B tile shares L2 with the A panel sweeping through it and the
    C tiles, so a tile sized to the whole L2 evicts its own operands: on the
    1 MB/core machine an f32 K=4096 shape ran 0.84 vs linalg with the
    whole-L2 KC=4096 tile and 0.99 with the half-L2 KC=2048 (the 2 MB/core
    machine's picks all land on the cap and are unchanged). See docs/DESIGN.md
    "_l2_resident_kc".

    The packed tile is sized in COMPUTE-dtype elements: bf16 packs its B
    panels as f32, so its resident tile has f32 bytes even though the source
    matrix is bf16."""
    comptime elem = size_of[Scalar[compute_dtype[dtype]()]]()
    var cap = 16384 // elem
    var l2 = l2_cache_size()
    if l2 == 0:
        return min(cap, k)
    return min(min(l2 // (2 * tile_n * elem), k), cap)


def _box_l2_budget() -> Int:
    """Upper bound (bytes of B = k*n) for routing an M-dominant box to the
    no-pack `nopack_gemm`. That kernel re-reads all of B once per MR-row
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
    shape. docs/DESIGN.md "_box_l2_budget".

    Routing bytes use the COMPUTE dtype so bf16 takes the same route as f32
    on every shape: the byte windows were measured where storage and compute
    match, and the no-pack kernel this gate admits pays a per-load widening
    tax in bf16 that those windows never priced in (bf16 sq320 ran 0.54 vs
    linalg through the storage-byte gate, which kept it off the 2D grid that
    fixed the same shape in f32; routed as f32 it is a 1.18 WIN over 10
    epochs)."""
    var b_bytes = k * n * size_of[Scalar[compute_dtype[dtype]()]]()
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
# kernel. The full per-branch measurements live in README.md / docs/DESIGN.md.
# ===========================================================================


@always_inline
def _small_batch[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Small-batch decode (2 <= M <= 5): the packed micro-kernel with MR = M,
    so the packed B panel is streamed once and reused across all M rows (a
    per-row GEMV would re-stream B, ~2x slower at M=4)."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
    comptime for MR in range(2, 6):
        if a.rows == MR:
            packed_gemm[dtype, MR, 4 * NELTS, 256, 2, 8 * NELTS](c, a, b)
            return


@always_inline
def _thin_n[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Thin-N tall-M (N <= 8*NELTS, M >= 64): parallelize over M-row blocks
    (the packed kernel parallelizes only over N and starves the cores). Reads
    A/B from source; a thin N stays cache-resident, so packing buys nothing."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
    var n = c.cols
    if n < 2 * NELTS:
        nopack_gemm[dtype, 6, 1](c, a, b)
    elif n % (4 * NELTS) == 0:
        nopack_gemm[dtype, 6, 4](c, a, b)
    else:
        nopack_gemm[dtype, 6, 2](c, a, b)


@always_inline
def _small_box[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Small M-dominant box whose B is L2-resident (see _box_fits_l2).

    The bigger boxes (B above 384 KB, the top of the admission window) take
    the same pack-B-only path as the squares, via `_square_ish` so its
    makespan gate can also pick the balanced 2D grid when the column count
    splits the cores unevenly (f32 sq320: 5 columns on 4 cores ran 0.72 vs
    linalg on the 1D column path, 1.10 on the 2D grid). Packing B once and
    reading A unpacked beats re-reading all of B per MR-row block, as long as
    N splits into enough NR-tiles to fill the cores.

    The no-pack M-parallel kernel handles its N-remainder (n % NR) one row at
    a time with a single accumulator chain, so a wide remainder runs latency-
    bound at ~1/8 of the tile throughput and dominates the whole GEMM once it
    is a big enough slice of N (f32 sq300: a 44-wide remainder of N=300 sank
    the route to 0.25 vs linalg; the packed 2D grid runs it at 0.95). Shapes
    whose remainder is at least N/8 go to `_square_ish` too; its masked
    partial-panel microkernel keeps the remainder at full register-tile
    throughput.

    The genuinely small clean-N boxes keep the no-pack M-parallel kernel:
    MR=6 (24 accumulators, deepest ILP) when it divides M with no tail, else
    MR=4. docs/DESIGN.md "_nopack_gemm MR" and "Small-box f32 routing"."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    # Compute-dtype bytes, like _box_fits_l2, so bf16 routes as f32 here too.
    var b_bytes = k * n * size_of[Scalar[compute_dtype[dtype]()]]()
    var n_rem = n % (4 * NELTS)
    if b_bytes > (3 << 17) and ceildiv(n, 4 * NELTS) >= num_physical_cores():
        _square_ish(c, a, b)
    elif n_rem * 8 >= n:
        _square_ish(c, a, b)
    elif m % 6 == 0:
        nopack_gemm[dtype, 6, 4](c, a, b)
    else:
        nopack_gemm[dtype, 4, 4](c, a, b)


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
    still idles cores; those shapes route to `pack_b_only_2d`, which splits
    each column into MR-row blocks so the 2D tile grid balances. The 2D path
    carries a real cost (boundary columns repacked, more masked tiles), so it
    is gated to a single C-stored-once K-panel (k <= 2048) and a makespan it
    cuts by at least 1/8. docs/DESIGN.md "Square-ish: pack-B-only /
    TileK=K"."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var num_workers = num_physical_cores()
    var num_cols = ceildiv(n, 4 * NELTS)
    var num_i_panels = ceildiv(m, 6)
    var col_makespan = ceildiv(num_cols, num_workers) * num_i_panels
    var grid_makespan = ceildiv(num_cols * num_i_panels, num_workers)
    if k <= 2048 and grid_makespan * 8 <= col_makespan * 7:
        pack_b_only_2d[dtype, 6, 4 * NELTS, 2](c, a, b)
    else:
        _prefill_kc[dtype, 4 * NELTS, False, False](c, a, b, min(k, 2048))


@always_inline
def _wide_n[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Wide-N (N >= K, up-proj-like): the N-balanced 6x32 tile (TILE_N=8*NELTS).
    The older 8x24 tile only for very wide N with tiny M. Small M packs A per
    worker; past the M~192 crossover SHARED_A with a cache-aware KC
    (L2-resident packed-B tile) on the finer TILE_N=4*NELTS, the same
    heavy-band tile the tall-K branch uses: 2x more j-tiles for a tighter
    worker makespan (f32's doubled NR left N=11008 at 86 wide tiles, 21.5 per
    core, a ~2-3% straggler tax the f64 sweeps never saw) and a packed-B tile
    at half the L2 instead of all of it. docs/DESIGN.md "Wide-N heavy
    band"."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
    var m = a.rows
    var n = c.cols
    var k = a.cols
    if n >= 9 * 1024 and m <= 32:
        packed_gemm[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS](c, a, b)
    elif k <= 2048 and k * n <= (1 << 20):
        # Small cache-resident wide box (B = k*n stays in L3, at any M): the
        # packed path packs B per j-tile per worker, and that packing + launch
        # overhead dwarfs the compute when the whole problem is cache-hot. The
        # pack-B-only 2D grid packs B once per worker-column, reads A unpacked,
        # stores C once (k <= 2048 is one K-panel), and balances the column x
        # row-block tile grid across cores. The k*n cut keeps every wide
        # headline shape on the packed prefill path.
        pack_b_only_2d[dtype, 6, 4 * NELTS, 2](c, a, b)
    elif m <= 192:
        _prefill[dtype, 256, 8 * NELTS](c, a, b)
    else:
        # m > 192: SHARED_A + a single C-stored-once k-panel (KC = min(K, cap)
        # at the detected L2, see _l2_resident_kc) on the finer heavy-band tile.
        _prefill_kc[dtype, 4 * NELTS, True](
            c, a, b, _l2_resident_kc[dtype](4 * NELTS, k)
        )


@always_inline
def _tall_k[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """Tall-K (N < K, down-proj-like): the 6x(4*NELTS) register tile whose
    masked M-remainder tail beats 8x24 at every M. Small M (< 192) packs A per
    worker on the wider TILE_N=8*NELTS; past the M~192 crossover the heavy
    band switches to the finer TILE_N=4*NELTS (2x more j-tiles, better load
    balance, smaller packed-B panels). The m > 256 rung is pack-B-only: on a
    tall K the A matrix is the dominant operand (dn-m512's A is 45 MB, past
    L3), so packing it is a full extra DRAM sweep, and reading it unpacked
    (strided column broadcasts, the square-ish treatment) frees the KC cap to
    return to the deeper C-stored-once panel. docs/DESIGN.md "Tall-K
    large-M"."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
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
        _prefill_kc[dtype, 4 * NELTS, False, False](
            c, a, b, _l2_resident_kc[dtype](4 * NELTS, k)
        )


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

        bf16 + AMX, M%32=K%32=0  amx_bf16_gemm  tdpbf16ps tile kernel
        tiny  M*N*K < 2^19       serial_gemm   serial, no threads/packing
        M == 1                   decode_gemv   j-parallel GEMV, streams B once
        M in 2..5                _small_batch  packed MR=M, reuse B across rows
        thin-N  N<=8*NELTS, M>=64  _thin_n     M-parallel no-pack
        small box  M>=N, B fits L2 _small_box  no-pack (square-ish machinery
                                               at B>384KB or a >=N/8 remainder)
        narrow-N  N<=192         packed NR=16  too few TILE_N=64 j-tiles
        square-ish  N<=M         _square_ish   pack-B-only 6x32, TileK=K
        wide-N  N>=K             _wide_n       packed 6x32 (pack-B-only 2D for
                                               small L3-resident box)
        tall-K  N<K              _tall_k       packed 6x32, cache-aware KC

    The N<=M and M>=N gates keep square-ish and the no-pack routes off every
    wide headline shape (the LLM up/down projections are N >> M), where those
    kernels are catastrophic. The full rationale + measurements are in
    README.md and docs/DESIGN.md."""
    comptime NELTS = simd_width_of[compute_dtype[dtype]()]()
    var m = a.rows
    var n = c.cols
    var k = a.cols

    # bf16 on an AMX part: the tdpbf16ps tile kernel (docs/SOL.md idea 3),
    # gated to the tile geometry it handles (whole 32-row M blocks and 32-deep
    # K pair-steps; any N) and to above the tiny cutoff. Every other dtype,
    # machine, and shape falls through to the cascade unchanged.
    comptime if dtype == DType.bfloat16:
        if amx_shape_ok(m, n, k) and amx_bf16_usable():
            amx_bf16_gemm(c, a, b)
            return

    if m * n * k < (1 << 19):
        serial_gemm[dtype, 6, 2](c, a, b)
    elif m == 1:
        decode_gemv(c, a, b)
    elif m <= 5:
        _small_batch(c, a, b)
    elif n <= 8 * NELTS and m >= 64:
        _thin_n(c, a, b)
    elif _box_fits_l2[dtype](m, n, k):
        _small_box(c, a, b)
    elif n <= 192:
        # Narrow N: the NR=16 / TILE_N=16 tile (KU=4) so a small N still splits
        # into enough j-tiles to fill the cores.
        packed_gemm[dtype, 6, 2 * NELTS, 256, 4, 2 * NELTS](c, a, b)
    elif n <= m:
        _square_ish(c, a, b)
    elif n >= k:
        _wide_n(c, a, b)
    else:
        _tall_k(c, a, b)
