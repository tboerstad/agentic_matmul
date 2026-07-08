"""AMX bf16 GEMM: tdpbf16ps tile microkernel (SOL.md idea 3).

Intel AMX does a 16x32 (bf16) by 32x16 (bf16, pair-interleaved) tile FMA into
a 16x16 f32 accumulator tile in ONE instruction (`tdpbf16ps`), ~1024
flops/cycle/core versus 64 for AVX-512 f32 FMA. The stdlib linalg bf16 path
does not use it, so on an AMX part this kernel raises the bf16 ceiling far
past anything the AVX-512 kernels can reach.

Shape of the kernel:

  * N-parallel like the packed kernels: each worker owns a contiguous range
    of 32-column j-tiles.
  * Per j-tile the worker pair-interleaves B into the VNNI layout tdpbf16ps
    needs (packed row k2 holds the (b[2k2][j], b[2k2+1][j]) pairs, 64 bytes
    per row per 16-column panel) and then sweeps M in 32-row blocks.
  * Each 32x32 C block is a 2x2 grid of accumulator tiles (tmm0-3) that stays
    in tile registers across the WHOLE K sweep: C is written exactly once, so
    there is no per-K-panel C traffic at all. A is read straight from the
    row-major source by `tileloadd`'s strided row gather (no A pack), B rows
    stream from the L2-resident packed panel.
  * The f32 accumulator tiles are stored to a small scratch and narrowed to
    the bf16 C once per block.

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

The dispatch gate (`amx_bf16_usable` plus the shape divisibility checks in
gemm.matmul_dispatch) keeps every other dtype and machine byte-identical:
this path runs only for bf16 with m % 32 == n % 16 == k % 32 == 0 on a CPU
that reports AMX-TILE + AMX-BF16 and grants the tile-data xstate.

Numerics: tdpbf16ps truncates the f32 products of each bf16 pair and adds
them into the f32 accumulator per pair-step, which is not bit-identical to
the AVX-512 path's sequential f32 FMA over k. verify_f32_routes gates the
result against a naive f64 reference at the same tolerance as the f32
compute path.
"""

from cpu_cache import cpuid
from matrix import Matrix
from tile import Tile
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


@always_inline
def _pack_vnni_panel[
    dtype: DType, b_org: ImmutOrigin, bp_org: MutOrigin
](
    b: Tile[dtype, b_org],
    bp: UnsafePointer[Scalar[dtype], bp_org],
    j0: Int,
    k: Int,
):
    """Pair-interleave one 16-column panel of B into the VNNI layout: packed
    row k2 (64 bytes) holds (b[2k2][j0+j], b[2k2+1][j0+j]) pairs for j in
    0..16. `interleave` on two 16-lane rows emits exactly that permutation."""
    for k2 in range(k // 2):
        var r0 = b.addr(2 * k2, j0)
        var r1 = b.addr(2 * k2 + 1, j0)
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b.addr(2 * k2 + 16, j0)
        )
        var v0 = r0.load[width=16]()
        var v1 = r1.load[width=16]()
        (bp + k2 * 32).store(offset=0, val=v0.interleave(v1))


@always_inline
def _amx_store_c_tile[
    dtype: DType, c_org: MutOrigin, s_org: MutOrigin
](
    c: Tile[dtype, c_org],
    scratch: UnsafePointer[Float32, s_org],
    i0: Int,
    j0: Int,
):
    """Narrow one 16x16 f32 scratch tile into the bf16 C block at (i0, j0):
    the single f32-to-bf16 rounding of the result."""
    for r in range(16):
        var v = (scratch + r * 16).load[width=16]()
        c.addr(i0 + r, j0).store(offset=0, val=v.cast[dtype]())


def amx_bf16_gemm[
    dtype: DType
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """C = A * B on the AMX tile units. Caller guarantees (via the dispatch
    gate) bf16 element type, m % 32 == 0, n % 16 == 0, k % 32 == 0, and
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

    # Per worker: two VNNI-packed 16-column panels (k/2 rows x 32 bf16) plus
    # a 2x2-tile f32 scratch for the C narrowing.
    var bp_per_worker = k * 32
    var bp_buf = alloc[Scalar[dtype]](num_workers * bp_per_worker)
    var sc_per_worker = 32 * 32
    var sc_buf = alloc[Float32](num_workers * sc_per_worker)

    def worker(worker_id: Int) {read c_view, read a_view, read b_view, mut bp_buf, mut sc_buf, read m, read n, read k, read num_j_tiles, read num_workers, read bp_per_worker, read sc_per_worker}:
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

        for jt in range(jt0, jt1):
            var j0 = jt * 32
            var two_wide = j0 + 32 <= n  # else a single 16-column panel
            _pack_vnni_panel(b_view, bp, j0, k)
            if two_wide:
                _pack_vnni_panel(b_view, bp + half, j0 + 16, k)

            for i0 in range(0, m, 32):
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
                    var b0 = bp + p2 * 32
                    _tile_load[4](a0, a_stride)
                    _tile_load[5](a1, a_stride)
                    _tile_load[6](b0, 64)
                    _tile_dpbf16ps[0, 4, 6]()
                    _tile_dpbf16ps[2, 5, 6]()
                    if two_wide:
                        var b1 = bp + half + p2 * 32
                        _tile_load[7](b1, 64)
                        _tile_dpbf16ps[1, 4, 7]()
                        _tile_dpbf16ps[3, 5, 7]()

                # Store the f32 tiles once and narrow to bf16 C.
                _tile_store[0](sc, 64)
                _tile_store[2](sc + 512, 64)
                _amx_store_c_tile(c_view, sc, i0, j0)
                _amx_store_c_tile(c_view, sc + 512, i0 + 16, j0)
                if two_wide:
                    _tile_store[1](sc + 256, 64)
                    _tile_store[3](sc + 768, 64)
                    _amx_store_c_tile(c_view, sc + 256, i0, j0 + 16)
                    _amx_store_c_tile(c_view, sc + 768, i0 + 16, j0 + 16)

        _amx_release()

    parallelize(worker, num_workers, num_workers)
    bp_buf.free()
    sc_buf.free()
