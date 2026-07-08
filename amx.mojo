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
from std.sys.intrinsics import inlined_assembly, prefetch, PrefetchOptions


def _detect_amx_bf16() -> Int:
    """1 when the CPU reports AMX-TILE + AMX-BF16 (cpuid leaf 7 EDX bits 24
    and 22) AND the kernel grants the tile-data xstate
    (arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)); else 0. The
    syscall is Linux-specific, which is fine: the permission request only
    runs after the cpuid gate passes, and AMX parts are Linux servers."""
    comptime if not CompilationTarget.is_x86():
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
    (cpuid + one syscall on first call, a load afterwards)."""
    try:
        return _AMX_OK.get_or_create_ptr()[] == 1
    except:
        return False


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
    inlined_assembly[
        "ldtilecfg ($0)",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](cfg)


@always_inline
def _amx_release():
    """Return the tile file to the init state so later thread work carries no
    tile xstate."""
    inlined_assembly[
        "tilerelease", NoneType, constraints="", has_side_effect=True
    ]()


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
                if two_wide:
                    inlined_assembly[
                        "tilezero %tmm0\ntilezero %tmm1\ntilezero %tmm2\ntilezero %tmm3",
                        NoneType,
                        constraints="",
                        has_side_effect=True,
                    ]()
                else:
                    inlined_assembly[
                        "tilezero %tmm0\ntilezero %tmm2",
                        NoneType,
                        constraints="",
                        has_side_effect=True,
                    ]()

                # K sweep: 32 bf16 (16 pairs) per step; C tiles stay in
                # registers for the whole sweep.
                for p2 in range(0, k // 2, 16):
                    var a0 = a_view.addr(i0, 2 * p2)
                    var a1 = a_view.addr(i0 + 16, 2 * p2)
                    var b0 = bp + p2 * 32
                    if two_wide:
                        var b1 = bp + half + p2 * 32
                        inlined_assembly[
                            "tileloadd ($0,$1,1), %tmm4\ntileloadd ($2,$1,1), %tmm5\ntileloadd ($3,$4,1), %tmm6\ntileloadd ($5,$4,1), %tmm7\ntdpbf16ps %tmm6, %tmm4, %tmm0\ntdpbf16ps %tmm7, %tmm4, %tmm1\ntdpbf16ps %tmm6, %tmm5, %tmm2\ntdpbf16ps %tmm7, %tmm5, %tmm3",
                            NoneType,
                            constraints="r,r,r,r,r,r,~{memory}",
                            has_side_effect=True,
                        ](a0, a_stride, a1, b0, Int(64), b1)
                    else:
                        inlined_assembly[
                            "tileloadd ($0,$1,1), %tmm4\ntileloadd ($2,$1,1), %tmm5\ntileloadd ($3,$4,1), %tmm6\ntdpbf16ps %tmm6, %tmm4, %tmm0\ntdpbf16ps %tmm6, %tmm5, %tmm2",
                            NoneType,
                            constraints="r,r,r,r,r,~{memory}",
                            has_side_effect=True,
                        ](a0, a_stride, a1, b0, Int(64))

                # Store the f32 tiles once and narrow to bf16 C.
                if two_wide:
                    inlined_assembly[
                        "tilestored %tmm0, ($0,$4,1)\ntilestored %tmm1, ($1,$4,1)\ntilestored %tmm2, ($2,$4,1)\ntilestored %tmm3, ($3,$4,1)",
                        NoneType,
                        constraints="r,r,r,r,r,~{memory}",
                        has_side_effect=True,
                    ](sc, sc + 256, sc + 512, sc + 768, Int(64))
                    _amx_store_c_tile(c_view, sc, i0, j0)
                    _amx_store_c_tile(c_view, sc + 256, i0, j0 + 16)
                    _amx_store_c_tile(c_view, sc + 512, i0 + 16, j0)
                    _amx_store_c_tile(c_view, sc + 768, i0 + 16, j0 + 16)
                else:
                    inlined_assembly[
                        "tilestored %tmm0, ($0,$2,1)\ntilestored %tmm2, ($1,$2,1)",
                        NoneType,
                        constraints="r,r,r,~{memory}",
                        has_side_effect=True,
                    ](sc, sc + 256, Int(64))
                    _amx_store_c_tile(c_view, sc, i0, j0)
                    _amx_store_c_tile(c_view, sc + 256, i0 + 16, j0)

        _amx_release()

    parallelize(worker, num_workers, num_workers)
    bp_buf.free()
    sc_buf.free()
