"""AMX bf16 GEMM: tdpbf16ps tile microkernel (docs/SOL.md idea 3).

Intel AMX does a 16x32 (bf16) by 32x16 (bf16, pair-interleaved) tile FMA into
a 16x16 f32 accumulator tile in ONE instruction (`tdpbf16ps`), ~1024
flops/cycle/core versus 64 for AVX-512 f32 FMA. The stdlib linalg bf16 path
does not use it, so on an AMX part this kernel raises the bf16 ceiling far
past anything the AVX-512 kernels can reach.

Shape of the kernel:

  * N-parallel like the packed kernels: each worker owns a contiguous range
    of 32-column j-tiles and walks it in GROUPS of j-tiles sized to fit the
    packed B group in about 3/8 of the per-core L2 (`_amx_group_tiles`).
  * Per group the worker pair-interleaves B into the VNNI layout tdpbf16ps
    needs (packed row k2 holds the (b[2k2][j], b[2k2+1][j]) pairs, 64 bytes
    per row per 16-column panel). The pack is row-outer over the whole
    group: each visit to a B row pair reads the group's contiguous column
    span (group x 64 bytes) in one go, so the TLB miss and the L3 fetch that
    every row of a wide B costs (rows are N x 2 bytes apart, more than a page
    for the wide-N shapes) are paid once per group instead of once per
    j-tile.
  * Then the worker sweeps M in 32-row blocks, and for each block runs every
    j-tile in the group: the 32-row A block is pulled from L3/L2 once per
    group and hits L2 for the rest of the group, instead of being re-streamed
    per j-tile.
  * Each 32x32 C block is a 2x2 grid of accumulator tiles (tmm0-3) that stays
    in tile registers across the WHOLE K sweep: C is written exactly once, so
    there is no per-K-panel C traffic at all. A is read straight from the
    row-major source by `tileloadd`'s strided row gather (no A pack), B rows
    stream from the L2-resident packed group.
  * The f32 accumulator tiles are stored to a small scratch and narrowed to
    the bf16 C once per block.

The grouping is the whole difference between the first version (one j-tile
packed and swept at a time) and this one, and it is worth 1.4-1.7x on every
wide-N or large-M shape on the 2.10 GHz Granite Rapids box (docs/DESIGN.md
"AMX bf16", third pass). Software prefetch of the next K-step's A/B lines
into L1 was also tried and lost 5-25% everywhere; the hardware prefetcher
already covers the L2-resident streams.

The tile instructions are LLVM's immediate-tile-register AMX intrinsics
(`llvm.x86.tileloadd64`, `llvm.x86.tdpbf16ps`, ..., the same family clang's
`_tile_loadd`-style intrinsics lower to) through `llvm_intrinsic`, with the
tile numbers as compile-time parameters so they land as the immediates the
intrinsics require. LLVM's other AMX family (`*.internal`, whose values the
register allocator assigns) needs the `x86_amx` IR type, which Mojo cannot
express, so the fixed-register form is the right fit; nothing else in a
worker touches tile state, so fixed tmm numbers are safe. Everything is
comptime-gated on `CompilationTarget.has_intel_amx()`: on a target without
the amx-tile feature none of the intrinsics are instantiated (they would be
rejected at instruction selection), and the dispatch gate folds to False.

Using AMX at all needs a per-process opt-in from the kernel:
`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` (one
`external_call["syscall"]`), memoized together with the cpuid feature check
in `amx_bf16_usable()`. Each worker invocation runs `ldtilecfg` (all 8 tiles
16 rows x 64 bytes) and `tilerelease` around its j-tile range.

The dispatch gate (`amx_shape_ok` + `amx_bf16_usable`, both checked in
dispatch.matmul_dispatch) keeps every other dtype and machine byte-identical:
this path runs only for bf16 with m % 32 == k % 32 == 0 (any n; a partial
trailing 16-column panel packs zero-padded and masks its C store) on a CPU
that reports AMX-TILE + AMX-BF16 and grants the tile-data xstate.

Numerics: tdpbf16ps truncates the f32 products of each bf16 pair and adds
them into the f32 accumulator per pair-step, which is not bit-identical to
the AVX-512 path's sequential f32 FMA over k. tests/test_dtypes.mojo gates the
result against a naive f64 reference at the same tolerance as the f32
compute path.
"""

from matmul.cpu_cache import cpuid, l2_cache_size
from matmul.matrix import Matrix
from matmul.tile import Tile
from std.algorithm.functional import parallelize
from std.ffi import _Global, external_call
from std.math import ceildiv
from std.memory import stack_allocation
from std.memory.unsafe_pointer import alloc
from std.sys import CompilationTarget, num_physical_cores
from std.sys.intrinsics import (
    llvm_intrinsic,
    prefetch,
    PrefetchOptions,
)


def _detect_amx_bf16() -> Int:
    """1 when the CPU reports AMX-TILE + AMX-BF16 (cpuid leaf 7 EDX bits 24
    and 22) AND the kernel grants the tile-data xstate
    (arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)); else 0. The cpuid
    re-check on top of the comptime target feature guards against running a
    binary on a different host than it was compiled on. The syscall is
    Linux-specific, which is fine: it only runs after both gates pass, and
    AMX parts are Linux servers."""
    comptime if not CompilationTarget.has_intel_amx():
        return 0
    var r = cpuid(7, 0)
    if (r.edx >> 24) & 1 == 0 or (r.edx >> 22) & 1 == 0:
        return 0
    # SYS_arch_prctl = 158, ARCH_REQ_XCOMP_PERM = 0x1023, XTILEDATA = 18.
    var rc = external_call["syscall", Int](Int(158), Int(0x1023), Int(18))
    return 1 if rc == 0 else 0


comptime _AMX_OK = _Global["agentic_matmul_amx_bf16", _detect_amx_bf16]


def amx_bf16_usable() -> Bool:
    """True when the AMX bf16 tile kernel can run on this machine, memoized
    (cpuid + one syscall on first call, a load afterwards). Comptime-False
    on targets without the amx-tile feature, so the dispatch branch folds
    away entirely there."""
    comptime if not CompilationTarget.has_intel_amx():
        return False
    try:
        return _AMX_OK.get_or_create_ptr()[] == 1
    except:
        return False


@always_inline
def amx_shape_ok(m: Int, n: Int, k: Int) -> Bool:
    """The shape half of the AMX dispatch gate: above the tiny cutoff, whole
    32-row M blocks, whole 32-deep K pair-steps. Any N works (a partial
    trailing 16-column panel packs zero-padded and masks its C store), but M
    and K remainders would read past the source A rows, so they stay on the
    AVX-512 routes. Shared by matmul_dispatch and the benchmark's roofline
    pick so the two can never disagree."""
    return (
        m * n * k >= (1 << 19)
        and m % 32 == 0
        and k % 32 == 0
    )


# --- The tile ops, one comptime-numbered intrinsic each ---------------------


@always_inline
def _tile_zero[t: Int]():
    llvm_intrinsic["llvm.x86.tilezero", NoneType](Int8(t))


@always_inline
def _tile_load[
    t: Int, dtype: DType, org: Origin
](p: UnsafePointer[Scalar[dtype], org], stride_bytes: Int):
    """tileloadd tmm{t} <- 16 rows of 64 bytes at `p`, rows `stride_bytes`
    apart."""
    llvm_intrinsic["llvm.x86.tileloadd64", NoneType](
        Int8(t), p, Int64(stride_bytes)
    )


@always_inline
def _tile_dpbf16ps[dst: Int, src_a: Int, src_b: Int]():
    """tmm{dst} (16x16 f32) += tmm{src_a} (16x32 bf16) * tmm{src_b} (32x16
    bf16, pair-interleaved)."""
    llvm_intrinsic["llvm.x86.tdpbf16ps", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b)
    )


@always_inline
def _tile_store[
    t: Int, org: MutOrigin
](p: UnsafePointer[Float32, org], stride_bytes: Int):
    llvm_intrinsic["llvm.x86.tilestored64", NoneType](
        Int8(t), p, Int64(stride_bytes)
    )


@always_inline
def _amx_configure():
    """Load the palette-1 tile config: all 8 tiles 16 rows x 64 bytes. Run
    once per worker invocation (tile config is per-thread state). The config
    block must be 64-byte aligned (ldtilecfg #GPs otherwise)."""
    var cfg = stack_allocation[64, UInt8, alignment=64]()
    for i in range(64):
        cfg[i] = 0
    cfg[0] = 1  # palette
    comptime for t in range(8):
        cfg[16 + 2 * t] = 64  # colsb, low byte
        cfg[48 + t] = 16  # rows
    llvm_intrinsic["llvm.x86.ldtilecfg", NoneType](cfg)


@always_inline
def _amx_release():
    """Return the tile file to the init state so later thread work carries no
    tile xstate."""
    llvm_intrinsic["llvm.x86.tilerelease", NoneType]()


def _amx_group_tiles(k: Int) -> Int:
    """How many 32-column j-tiles one worker packs and sweeps per group: as
    many as fit their VNNI-packed B (k x 64 bytes per j-tile) in 3/8 of the
    per-core L2, at least one. Swept on the Granite Rapids box (2 MB L2) at
    384 KB .. 1.5 MB: 768 KB is best or within noise on every shape (prefill
    1.56x, up-m512 1.70x, sq2048 1.56x, M512-g 1.49x vs the ungrouped
    kernel); 512 KB gives back 5-15% on the large-M band, 1 MB+ gives back
    5-20% on the small-M band where the whole A also wants to stay resident
    next to the group. Falls back to 768 KB when L2 is undetectable. The
    cpuid probe is memoized, so it costs one call per process."""
    var l2 = l2_cache_size()
    var budget = (3 * l2) // 8 if l2 > 0 else 768 * 1024
    return max(1, budget // (k * 64))


@always_inline
def _pack_vnni_group[
    dtype: DType, b_org: ImmutOrigin, bp_org: MutOrigin
](
    b: Tile[dtype, b_org],
    bp: UnsafePointer[Scalar[dtype], bp_org],
    j_base: Int,
    n: Int,
    k: Int,
    tiles: Int,
):
    """Pair-interleave `tiles` consecutive 32-column j-tiles of B (columns
    j_base .. j_base + 32*tiles, clipped to n) into the VNNI layout, row-outer:
    for each row pair (2k2, 2k2+1) walk the group's 16-column panels left to
    right. Packed panel p of the group lives at bp + (p//2)*k*32 +
    (p%2)*k*16, and its row k2 (64 bytes) holds the (b[2k2][j], b[2k2+1][j])
    pairs; `interleave` on two 16-lane rows emits exactly that permutation.
    A partial trailing panel (the last panel of a shape whose n is not a
    multiple of 16) packs element-wise and zero-fills the dead columns, so
    the tile FMA runs unmasked and the dead columns contribute nothing (the C
    store masks them off).

    Row-outer matters: each B row is n*2 bytes from the next, more than a
    4 KB page on the wide-N shapes, so a per-panel pack pays a TLB miss and
    an L3 line fetch per row per panel. Visiting each row pair once per group
    reads its contiguous tiles*64-byte span in one pass and amortizes both
    over the group; that alone was +48% on prefill (M=96), where the pack is
    a large share of the runtime."""
    var half = k * 16
    var tile_elems = k * 32
    var cols = min(32 * tiles, n - j_base)
    var full_panels = cols // 16
    var tail = cols - 16 * full_panels
    for k2 in range(k // 2):
        var r0 = b.addr(2 * k2, j_base)
        var r1 = b.addr(2 * k2 + 1, j_base)
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b.addr(2 * k2 + 8, j_base)
        )
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b.addr(2 * k2 + 9, j_base)
        )
        for p in range(full_panels):
            var v0 = (r0 + 16 * p).load[width=16]()
            var v1 = (r1 + 16 * p).load[width=16]()
            var dst = bp + (p // 2) * tile_elems + (p % 2) * half + k2 * 32
            dst.store(offset=0, val=v0.interleave(v1))
        if tail > 0:
            var p = full_panels
            var dst = bp + (p // 2) * tile_elems + (p % 2) * half + k2 * 32
            for j in range(tail):
                dst[2 * j] = r0[16 * p + j]
                dst[2 * j + 1] = r1[16 * p + j]
            for j in range(tail, 16):
                dst[2 * j] = Scalar[dtype](0)
                dst[2 * j + 1] = Scalar[dtype](0)


@always_inline
def _amx_store_c_tile[
    dtype: DType, c_org: MutOrigin, s_org: MutOrigin
](
    c: Tile[dtype, c_org],
    scratch: UnsafePointer[Float32, s_org],
    i0: Int,
    j0: Int,
    cols: Int,
):
    """Narrow one 16x16 f32 scratch tile into the bf16 C block at (i0, j0),
    keeping only the first `cols` columns (the zero-padded rest of a partial
    trailing panel): the single f32-to-bf16 rounding of the result."""
    if cols == 16:
        for r in range(16):
            var v = (scratch + r * 16).load[width=16]()
            c.addr(i0 + r, j0).store(offset=0, val=v.cast[dtype]())
    else:
        for r in range(16):
            var row = c.addr(i0 + r, j0)
            for j in range(cols):
                row[j] = scratch[r * 16 + j].cast[dtype]()


def amx_bf16_gemm[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """C = A * B on the AMX tile units. Caller guarantees (via the dispatch
    gate) bf16 element type, `amx_shape_ok(m, n, k)`, and
    `amx_bf16_usable()`."""
    comptime assert dtype == DType.bfloat16, "AMX kernel is bf16-only"
    # On a target without the amx-tile feature the intrinsics below cannot be
    # instruction-selected; `amx_bf16_usable()` is comptime-False there, so
    # this body is unreachable and can compile to nothing.
    comptime if not CompilationTarget.has_intel_amx():
        return

    var c_view = c.noalias_view()
    var a_view = a.noalias_view()
    var b_view = b.noalias_view()
    var m = a_view.rows
    var n = c_view.cols
    var k = a_view.cols

    var num_j_tiles = ceildiv(n, 32)
    var num_workers = num_physical_cores()
    var group = _amx_group_tiles(k)

    # Per worker: a group of VNNI-packed j-tiles (each two 16-column panels
    # of k/2 rows x 32 bf16) plus a 2x2-tile f32 scratch for the C narrowing.
    var tile_elems = k * 32
    var bp_per_worker = group * tile_elems
    var bp_buf = alloc[Scalar[dtype]](num_workers * bp_per_worker)
    var sc_per_worker = 32 * 32
    var sc_buf = alloc[Float32](num_workers * sc_per_worker)

    def worker(worker_id: Int) {read c_view, read a_view, read b_view, mut bp_buf, mut sc_buf, read m, read n, read k, read num_j_tiles, read num_workers, read group, read tile_elems, read bp_per_worker, read sc_per_worker}:
        var per = ceildiv(num_j_tiles, num_workers)
        var jt0 = worker_id * per
        var jt1 = min(jt0 + per, num_j_tiles)
        if jt0 >= num_j_tiles:
            return

        var bp = bp_buf + worker_id * bp_per_worker
        var sc = sc_buf + worker_id * sc_per_worker
        var a_stride = k * 2  # bytes per A row
        var half = k * 16  # bf16 elements per packed 16-column panel

        _amx_configure()

        var jg = jt0
        while jg < jt1:
            var tiles = min(group, jt1 - jg)
            _pack_vnni_group(b_view, bp, jg * 32, n, k, tiles)

            # M blocks outer, the group's j-tiles inner: the 32-row A block
            # is fetched once per group and reused from L2 for every j-tile.
            for i0 in range(0, m, 32):
                for g in range(tiles):
                    var j0 = (jg + g) * 32
                    var rem = min(32, n - j0)
                    var cols0 = min(16, rem)  # 1..16
                    var cols1 = rem - cols0  # 0..16; > 0 means a second panel
                    var two_wide = cols1 > 0
                    var bpg = bp + g * tile_elems

                    _tile_zero[0]()
                    _tile_zero[2]()
                    if two_wide:
                        _tile_zero[1]()
                        _tile_zero[3]()

                    # K sweep: 32 bf16 (16 pairs) per step; C tiles stay in
                    # registers for the whole sweep.
                    for p2 in range(0, k // 2, 16):
                        var a0 = a_view.addr(i0, 2 * p2)
                        var a1 = a_view.addr(i0 + 16, 2 * p2)
                        var b0 = bpg + p2 * 32
                        _tile_load[4](a0, a_stride)
                        _tile_load[5](a1, a_stride)
                        _tile_load[6](b0, 64)
                        _tile_dpbf16ps[0, 4, 6]()
                        _tile_dpbf16ps[2, 5, 6]()
                        if two_wide:
                            var b1 = bpg + half + p2 * 32
                            _tile_load[7](b1, 64)
                            _tile_dpbf16ps[1, 4, 7]()
                            _tile_dpbf16ps[3, 5, 7]()

                    # Store the f32 tiles once and narrow to bf16 C.
                    _tile_store[0](sc, 64)
                    _tile_store[2](sc + 512, 64)
                    _amx_store_c_tile(c_view, sc, i0, j0, cols0)
                    _amx_store_c_tile(c_view, sc + 512, i0 + 16, j0, cols0)
                    if two_wide:
                        _tile_store[1](sc + 256, 64)
                        _tile_store[3](sc + 768, 64)
                        _amx_store_c_tile(c_view, sc + 256, i0, j0 + 16, cols1)
                        _amx_store_c_tile(c_view, sc + 768, i0 + 16, j0 + 16, cols1)
            jg += tiles

        _amx_release()

    parallelize(worker, num_workers, num_workers)
    bp_buf.free()
    sc_buf.free()
