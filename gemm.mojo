from cpu_cache import l2_cache_size
from matrix import Matrix
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.math import ceildiv, fma
from std.memory import memset_zero
from std.memory.unsafe_pointer import alloc
from std.sys import num_physical_cores, simd_width_of, size_of
from std.sys.intrinsics import prefetch, PrefetchOptions


# ---------------------------------------------------------------------------
# Register-tile micro-kernel building blocks
#
# Every register-blocked kernel in this file (the packed prefill kernel, the
# serial small-shape kernel, the no-pack thin-N kernel) computes an MR x NR
# block of C as a register-resident accumulator tile swept over K. These two
# helpers factor out that shared inner step so each kernel expresses only its
# own packing / loop scaffolding. Both are @always_inline, so after inlining
# they emit the same machine code as the hand-written nests they replace —
# `_fma_tile` takes SIMD values only (no pointers), so it cannot perturb the
# noalias B-load hoisting the hot loop depends on.
# ---------------------------------------------------------------------------


@always_inline
def _fma_tile[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int,
](
    mut acc: InlineArray[SIMD[dtype, NELTS], MR * NR_VECS],
    a_vals: InlineArray[Scalar[dtype], MR],
    bv: InlineArray[SIMD[dtype, NELTS], NR_VECS],
):
    """Broadcast each of the MR A scalars across the NR_VECS B vectors and FMA
    into the MR x NR_VECS register-tile accumulator. The one inner step shared
    by every register-blocked micro-kernel here."""
    comptime for mr in range(MR):
        var a_bc = SIMD[dtype, NELTS](a_vals[mr])
        comptime for nr in range(NR_VECS):
            acc[mr * NR_VECS + nr] = a_bc.fma(bv[nr], acc[mr * NR_VECS + nr])


@always_inline
def _micro_masked[
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
    """Cold-path MR x NR_VECS micro-kernel: the leftover blocks the hot loop
    can't take. Handles an M-remainder (only `r` of MR rows active) and/or a
    partial NR-panel (only `jj_limit` of NR columns valid) by masking the C
    load and store; reads A unpacked (so it works for the un-packed remainder
    rows too) and B from the zero-padded packed panel. With r == MR and
    jj_limit == NR it degenerates to the unmasked full kernel, so this one
    function also covers the full-panel M-remainder.

    c_base = i*n + j0 + jr  (top-left of the block in C);
    a_base = i*k + pc       (top-left of the block's rows in A)."""
    var acc = InlineArray[SIMD[dtype, NELTS], MR * NR_VECS](
        fill=SIMD[dtype, NELTS](0)
    )
    if not is_first_k:
        comptime for mr in range(MR):
            if mr < r:
                comptime for nr in range(NR_VECS):
                    var col0 = nr * NELTS
                    var cr = c_ptr + c_base + mr * n
                    if col0 + NELTS <= jj_limit:
                        acc[mr * NR_VECS + nr] = cr.load[width=NELTS](offset=col0)
                    elif col0 < jj_limit:
                        var tmp = SIMD[dtype, NELTS](0)
                        for e in range(jj_limit - col0):
                            tmp[e] = cr[col0 + e]
                        acc[mr * NR_VECS + nr] = tmp
    for pk in range(kc):
        var bp_k = bp_panel + pk * NR
        var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
            fill=SIMD[dtype, NELTS](0)
        )
        comptime for nr in range(NR_VECS):
            bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS)
        # Gather A guarded by mr < r: rows past the M-remainder are out of
        # bounds, so leave them zero (their acc lanes are never stored).
        var a_vals = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))
        comptime for mr in range(MR):
            if mr < r:
                a_vals[mr] = a_ptr[a_base + mr * k + pk]
        _fma_tile[dtype, MR, NR_VECS, NELTS](acc, a_vals, bv)
    comptime for mr in range(MR):
        if mr < r:
            comptime for nr in range(NR_VECS):
                var col0 = nr * NELTS
                var cr = c_ptr + c_base + mr * n
                if col0 + NELTS <= jj_limit:
                    cr.store(offset=col0, val=acc[mr * NR_VECS + nr])
                elif col0 < jj_limit:
                    var v = acc[mr * NR_VECS + nr]
                    for e in range(jj_limit - col0):
                        cr[col0 + e] = v[e]


def _prefill_gemm_v3[
    dtype: DType, MR: Int, NR: Int, KC: Int, KU: Int, TILE_N: Int,
    NC_TILES: Int, SHARED_A: Bool = False,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Prefill GEMM with explicit B-load hoisting in the microkernel.
    #
    # SHARED_A (square-ish only): by default every worker re-packs the FULL A
    # (all i-panels) for each k-panel into its own buffer — num_workers
    # redundant copies. On a wide/tall headline shape A is small relative to the
    # N-sweep, so that redundancy is cheap L3 traffic (measured a wash; the
    # default keeps it). On a big SQUARE, A is as large as B/C and the 4x
    # re-pack is a real cost (sq2048: ~134 MB of redundant pack traffic/call).
    # With SHARED_A the full A is packed ONCE up front (parallelized over
    # i-panels) into a single shared buffer keyed [i-panel][k][MR]; every worker
    # then reads that buffer. Halves the packed-A footprint and removes the
    # redundant packing. Enabled from the square-ish branch only, so every
    # headline shape keeps the byte-for-byte original path.
    # The NR_VECS SIMD loads of B per K-step are stored in an InlineArray
    # and reused across the MR broadcast-FMA inner loop, so the compiler
    # cannot accidentally re-issue them per `mr` iteration. Pointers are
    # marked noalias to widen LLVM's CSE/hoist opportunities.
    comptime NELTS = simd_width_of[dtype]()
    comptime NR_VECS = NR // NELTS
    comptime PREFETCH_B_DIST = 8
    comptime PREFETCH_DIST = 4

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
        # panel block is MR*k contiguous, organized [k][MR] so the microkernel's
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

        # Hoisted captures for inner closures (rule #5)
        var bp_panel: type_of(bp_worker)
        var is_first_k: Bool

        var jt = j_tile_start
        while jt < j_tile_end:
            var jt_batch_end = min(jt + NC_TILES, j_tile_end)

            for pc in range(0, k, KC):
                var kc = min(KC, k - pc)
                is_first_k = (pc == 0)

                # Pack A: KC outer, MR inner so each pk gives MR contiguous
                # doubles. Skipped under SHARED_A — the full A was packed once up
                # front into the shared buffer before the parallel region.
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

                    for pk in range(kc):
                        var row_base = b_ptr + (pc + pk) * n + j0
                        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                            b_ptr + (pc + pk + PREFETCH_B_DIST) * n + j0
                        )
                        for jp in range(last_full_panel):
                            var jr = jp * NR
                            var src = row_base + jr
                            var dst = bp_worker + jp * kc * NR + pk * NR
                            comptime for nv in range(NR_VECS):
                                dst.store[width=NELTS](
                                    offset=nv * NELTS,
                                    val=src.load[width=NELTS](offset=nv * NELTS),
                                )
                        if has_remainder:
                            var jr = last_full_panel * NR
                            var src = row_base + jr
                            var dst = bp_worker + last_full_panel * kc * NR + pk * NR
                            for nr in range(nr_actual):
                                dst[nr] = src[nr]
                            for nr in range(nr_actual, NR):
                                dst[nr] = Scalar[dtype](0)

                    for jp in range(num_panels):
                        var jr = jp * NR
                        bp_panel = bp_worker + jp * kc * NR

                        if jr + NR > tile_n:
                            # Partial NR-panel (tile_n not a multiple of NR, i.e.
                            # the last j-tile of an N-not-a-multiple-of-NR shape).
                            # The packed bp_panel is zero-padded to full NR (see the
                            # pack loop above), so run the SAME register-blocked
                            # MR×NR_VECS microkernel as a full panel — the zero
                            # columns contribute nothing — and store back ONLY the
                            # jj_limit valid columns (full-NELTS SIMD for each
                            # complete lane, scalar tail for the straddling lane).
                            #
                            # Replaces the old row-by-row `vectorize` tail, which
                            # swept K once per row with only NR_VECS-deep ILP — far
                            # too shallow to hide FMA latency, so the partial tile
                            # ran at a fraction of the microkernel's throughput. The
                            # cost scaled with the remainder width and dominated the
                            # straggler worker: N=11007 (rem=31) ran 0.86 vs linalg
                            # while the multiple-of-NR N=11008 hit parity. With the
                            # full-ILP microkernel the partial tile keeps pace, so
                            # any N (odd / not a multiple of NR) holds ~parity.
                            # _micro_masked runs that full-width kernel and stores
                            # only the jj_limit valid columns: full MR-row i-panels
                            # first, then the m % MR remainder rows (r = MR vs
                            # r = m - i).
                            var jj_limit = tile_n - jr
                            i = 0
                            while i + MR <= m:
                                _micro_masked[dtype, MR, NR_VECS, NELTS, NR](
                                    c_ptr, a_ptr, bp_panel,
                                    i * n + j0 + jr, i * k + pc,
                                    n, k, kc, MR, jj_limit, is_first_k,
                                )
                                i += MR
                            if i < m:
                                _micro_masked[dtype, MR, NR_VECS, NELTS, NR](
                                    c_ptr, a_ptr, bp_panel,
                                    i * n + j0 + jr, i * k + pc,
                                    n, k, kc, m - i, jj_limit, is_first_k,
                                )
                            continue

                        # ---- Full NR-panel: hoisted-load microkernel ----
                        i = 0
                        ip = 0
                        while i + MR <= m:
                            # SHARED_A: full-k stride per i-panel + pc offset into
                            # the one shared pack; else per-worker per-kc layout.
                            var ap_panel = (
                                ap_worker + ip * MR * k + pc * MR
                            ) if SHARED_A else (ap_worker + ip * MR * kc)

                            comptime for mr in range(MR):
                                prefetch[PrefetchOptions().for_write().high_locality().to_data_cache()](
                                    c_ptr + (i + mr) * n + j0 + jr
                                )

                            var acc = InlineArray[SIMD[dtype, NELTS], MR * NR_VECS](
                                fill=SIMD[dtype, NELTS](0)
                            )
                            if not is_first_k:
                                comptime for mr in range(MR):
                                    comptime for nr in range(NR_VECS):
                                        acc[mr * NR_VECS + nr] = (
                                            c_ptr + (i + mr) * n + j0 + jr
                                        ).load[width=NELTS](offset=nr * NELTS)

                            var pk = 0
                            var pk_end = kc - (kc % KU)
                            while pk < pk_end:
                                comptime for ku in range(KU):
                                    var bp_k = bp_panel + (pk + ku) * NR
                                    var ap_k = ap_panel + (pk + ku) * MR

                                    # Load B once per ku-step (NR_VECS SIMD loads,
                                    # kept inline so the noalias hoist holds),
                                    # gather the MR packed-A scalars, then FMA the
                                    # tile via the shared inner step.
                                    var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                                        fill=SIMD[dtype, NELTS](0)
                                    )
                                    comptime for nr in range(NR_VECS):
                                        bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS)
                                    var a_vals = InlineArray[Scalar[dtype], MR](
                                        fill=Scalar[dtype](0)
                                    )
                                    comptime for mr in range(MR):
                                        a_vals[mr] = ap_k[mr]
                                    _fma_tile[dtype, MR, NR_VECS, NELTS](acc, a_vals, bv)
                                pk += KU

                            while pk < kc:
                                var bp_k = bp_panel + pk * NR
                                var ap_k = ap_panel + pk * MR

                                var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                                    fill=SIMD[dtype, NELTS](0)
                                )
                                comptime for nr in range(NR_VECS):
                                    bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS)
                                var a_vals = InlineArray[Scalar[dtype], MR](
                                    fill=Scalar[dtype](0)
                                )
                                comptime for mr in range(MR):
                                    a_vals[mr] = ap_k[mr]
                                _fma_tile[dtype, MR, NR_VECS, NELTS](acc, a_vals, bv)
                                pk += 1

                            comptime for mr in range(MR):
                                comptime for nr in range(NR_VECS):
                                    (c_ptr + (i + mr) * n + j0 + jr).store(
                                        offset=nr * NELTS, val=acc[mr * NR_VECS + nr],
                                    )

                            i += MR
                            ip += 1

                        # M-remainder (m % MR rows, 1..MR-1): handle all r
                        # leftover rows as ONE register-blocked block — a single
                        # K-sweep with r×NR_VECS accumulators, reusing the packed
                        # B panel, full NR-width SIMD. (jr+NR <= tile_n here: the
                        # partial-tile case already `continue`d above, so the full
                        # NR width is valid — jj_limit = NR, no column masking.)
                        # Replaces the old row-by-row `vectorize` tail, which swept
                        # K once per row with only NR_VECS-deep ILP — too shallow
                        # to hide FMA latency, the tax that made MR-not-dividing-M
                        # tiles (e.g. 6x32 at M=256) lose to remainder-free ones.
                        if i < m:
                            _micro_masked[dtype, MR, NR_VECS, NELTS, NR](
                                c_ptr, a_ptr, bp_panel,
                                i * n + j0 + jr, i * k + pc,
                                n, k, kc, m - i, NR, is_first_k,
                            )

            jt += NC_TILES

    parallelize(process_worker, num_workers, num_workers)
    bp_buf.free()
    ap_buf.free()


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

    Extracted to a top-level def so the inner closure can capture `p` and the
    pointers directly as function-scope bindings (a closure nested inside
    another closure cannot capture for/while-loop-body variables in current
    Mojo)."""
    def do_fma[width: Int](j: Int) {read ci, read ai, read b_col, read p, read n}:
        var acc = ci.load[width=width](offset=j)
        comptime for ku in range(KU):
            # Prefetch the same columns of the next KU-block of B rows: the
            # KU streams are n*8 bytes apart, too far for the hardware
            # prefetcher to follow. May reach past the end of B on the last
            # block — prefetch is architecturally non-faulting, so that's safe.
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
    # J-parallel GEMV optimized for decode (small M, large K×N).
    #
    # Each worker owns a disjoint column chunk of C and sweeps all K rows.
    # Per-k working set ≈ (N/nw)*8 bytes of B + same for C, which fits L1
    # (e.g. 2752×8 = 21 KB for N=11008, nw=4).  No reduction needed.
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


def _l2_resident_kc[dtype: DType](tile_n: Int, k: Int) -> Int:
    # Cache-aware KC for the prefill GEMM's large-M band.
    #
    # The L2-resident working set is the packed-B tile (TILE_N x KC elements),
    # reused across every i-panel of a j-tile. Sizing that block at ~half of
    # the per-core L2 (the BLIS rule of thumb) leaves the other half for the
    # streaming packed-A micro-panel and the C accumulators.
    #
    # Driving KC from the detected L2 keeps the pick correct across machines:
    # a 1 MB/core L2 yields KC=1024, a 2 MB/core L2 yields KC=2048. The caller
    # snaps this budget to the kernel's KC ladder.
    comptime elem = size_of[Scalar[dtype]]()
    var l2 = l2_cache_size()
    if l2 == 0:
        # L2 undetectable (e.g. non-x86 without sysctl): pick a KC that fits a
        # small (1 MB) L2.
        return min(512, k)
    var budget = (l2 // 2) // (tile_n * elem)
    return min(budget, k)


def _square_ish_kc(m: Int, n: Int, k: Int) -> Int:
    # Per-L2-adaptive KC for the square-ish branch (n <= m). Square-ish wants
    # HALF the BLIS half-L2 KC the wide/tall branches use, because here the M*KC
    # packed-A competes with the packed-B tile for L2 (a box-shaped C is small,
    # so a bigger KC buys little C-traffic saving while inflating packed-A). That
    # works out to KC=512 on a 1 MB/core L2 and KC=1024 on a 2 MB/core L2 — and
    # each is the measured best ON its machine, the opposite conclusions a single
    # hardcoded KC can't satisfy: interleaved A/B (peak/25) gives Skylake (1 MB)
    # KC512 > KC1024 (sq2048 0.84 vs 0.72, sq1024 0.84 vs 0.80), while the Xeon
    # (2 MB) measured KC1024 > KC512 (sq768..2048 +3-6%).
    #
    # KC only matters when k > 512 (a smaller k is one <=512-wide panel either
    # way), and a square-ish shape with k > 512 and M*N*K >= 2^28 is a multi-ms
    # op where the l2_cache_size() probe (~6 cpuid -> ~61 us VM-exit on a KVM
    # guest) is < 2.5%. Smaller shapes skip the probe entirely and take KC=512
    # (the single-panel / small-L2-optimal pick) — keeping cpuid off the hot path
    # for the few-us shapes, the lesson from the small-box gate.
    if k <= 512 or m * n * k < (1 << 28):
        return 512
    return 1024 if l2_cache_size() >= (3 << 19) else 512


def _box_l2_budget() -> Int:
    # Upper bound (bytes of B = k*n) for routing an M-dominant box to the
    # no-pack M-parallel _thin_n_gemm. That kernel re-reads the whole of B once
    # per MR-row block, so B must stay L2-resident across the M-sweep; once B no
    # longer fits, the packed prefill path wins. The small-box gate's first tier
    # uses a COMPILE-TIME 512 KB cut (a quarter of this 2 MB-L2 part, half the
    # 1 MB Skylake) so it never probes on the few-us cache-resident boxes. This
    # second tier extends the route up the B range on a larger L2, where the
    # newly-eligible boxes (B > 512 KB, so m*n*k in the tens of millions) are big
    # enough that the one-time memoized cpuid is amortized to noise.
    #
    # Cut at L2/3, the re-measured crossover (interleaved A/B vs linalg on this
    # 2.10 GHz Xeon, 2 MB/core L2, peak over 50 runs x3). The no-pack route
    # re-reads ALL of B once per MR-row block, so B must stay L2-resident
    # *alongside* the packed-A micro-panel, the C output, and the HW prefetcher's
    # headroom across the whole M-sweep — that holds only while B occupies about
    # a third of L2. Past that, B spills mid-sweep and the route collapses:
    #   sq288 (B=648KB) nopack ~0.92-1.00 vs packed ~0.92  -> nopack (admit)
    #   sq320 (B=800KB) nopack ~0.64-0.71 vs packed ~0.82-0.86 -> PACKED
    #   sq352 (B=968KB) nopack ~0.55-0.58 vs packed ~0.77-0.95 -> PACKED
    #   sq384 (B=1.15M) nopack ~0.42      vs packed ~0.89-0.92 -> PACKED
    #   640x256x512 (B=1MB)   nopack 0.47 vs packed 0.86-0.91  -> PACKED
    #   512x320x512 (B=1.28M) nopack 0.46-0.50 vs packed 0.83-0.87 -> PACKED
    # The previous (2*L2)/3 = 1.33MB cut admitted sq320..sq416 and those tall
    # boxes to no-pack, where they were the worst losses in the whole sweep
    # (sq384 0.42); their earlier no-pack "wins" were measured on an older Mojo
    # nightly whose linalg.matmul was slower — the current stdlib kernel retakes
    # them, so the cut must tighten to L2/3 (= 682 KB here; admits sq288 at
    # 648 KB, excludes sq320 at 800 KB). On the 1 MB Skylake L2/3 = 341 KB sits
    # below the compile-time 512 KB tier-1 cut, so that part simply keeps no-pack
    # for B <= 512 KB (sq288/sq320 there already preferred packed). Falls back to
    # the compile-time 512 KB tier (no extension) when L2 is undetectable.
    var l2 = l2_cache_size()
    if l2 == 0:
        return (1 << 19)
    return l2 // 3


def _matmul_small[
    dtype: DType, MR: Int, NR_VECS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Serial register-blocked GEMM for tiny shapes. No thread spawn, no buffer
    # allocation, no A/B packing — for a handful-of-FLOPs matmul the prefill
    # kernel's fixed overhead (parallelize launch + per-worker bp/ap allocs +
    # packing) dwarfs the compute, so it runs 10-100x SLOWER than a plain
    # serial loop here (e.g. an 8x8x8 ran 0.16 GFLOPS through the parallel path
    # vs 14 GFLOPS serial). This kernel holds an MR x (NR_VECS*NELTS) register
    # tile of C across the K loop, reading A and B straight from their source
    # buffers (a tiny N/K stays L1-resident, so explicit packing buys nothing).
    # Computes C = A * B (overwrites C). MR=6, NR_VECS=2 measured best on the
    # 2.10 GHz Xeon across the captured band; correctness is bit-identical to
    # the parallel kernels (verify_dispatch / max_err 0.0).
    comptime NELTS = simd_width_of[dtype]()
    comptime NR = NR_VECS * NELTS
    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()

    var j = 0
    while j + NR <= n:
        # Full MR-row register-blocked panels.
        var i = 0
        while i + MR <= m:
            var acc = InlineArray[SIMD[dtype, NELTS], MR * NR_VECS](
                fill=SIMD[dtype, NELTS](0)
            )
            for p in range(k):
                var b_row = b_ptr + p * n + j
                var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                    fill=SIMD[dtype, NELTS](0)
                )
                comptime for nv in range(NR_VECS):
                    bv[nv] = b_row.load[width=NELTS](offset=nv * NELTS)
                var a_vals = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))
                comptime for mr in range(MR):
                    a_vals[mr] = a_ptr[(i + mr) * k + p]
                _fma_tile[dtype, MR, NR_VECS, NELTS](acc, a_vals, bv)
            comptime for mr in range(MR):
                comptime for nv in range(NR_VECS):
                    (c_ptr + (i + mr) * n + j).store(
                        offset=nv * NELTS, val=acc[mr * NR_VECS + nv]
                    )
            i += MR
        # M-remainder rows (m % MR): one row at a time, same NR-wide SIMD.
        while i < m:
            var acc = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                fill=SIMD[dtype, NELTS](0)
            )
            for p in range(k):
                var b_row = b_ptr + p * n + j
                var av = SIMD[dtype, NELTS](a_ptr[i * k + p])
                comptime for nv in range(NR_VECS):
                    acc[nv] = av.fma(
                        b_row.load[width=NELTS](offset=nv * NELTS), acc[nv]
                    )
            comptime for nv in range(NR_VECS):
                (c_ptr + i * n + j).store(offset=nv * NELTS, val=acc[nv])
            i += 1
        j += NR
    # N-remainder columns (n % NR): vectorized tail over the full M extent,
    # with NELTS-wide chunks + an automatic scalar tail.
    if j < n:
        var jr = j
        for i in range(m):
            var c_row = c_ptr + i * n
            var a_row = a_ptr + i * k

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


def _thin_n_gemm[
    dtype: DType, MR: Int, NR_VECS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # M-parallel register-blocked GEMM with NO packing. Used for two regimes
    # the N-parallel prefill kernel handles badly (see matmul_dispatch):
    #   * thin-N (small N, large M*K): the prefill kernel parallelizes only over
    #     N (j-tiles), so a thin N starves the cores — N=16 -> 1 j-tile -> 1 of 4
    #     cores busy. Here the work is along M, so NR_VECS is 1-2 (NR=8/16).
    #   * small box, M-dominant, B fits L2 (the sq96..sq256 gap): the prefill
    #     kernel's A/B packing + per-worker buffers + parallelize-launch overhead
    #     dwarfs the compute when the whole problem is cache-resident. Here NR=32
    #     (NR_VECS=4) covers the wider-but-still-small N.
    # Either way every core owns a disjoint band of C's rows and sweeps the full
    # N, reading A and B straight from source — B stays cache-resident across the
    # M-sweep so explicit packing buys nothing. Computes C = A * B (overwrites C);
    # bit-identical to the serial/parallel paths.
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
        var i = blk * MR
        var r = min(MR, m - i)
        # NR-wide register-blocked panels over the N tiles. A full MR-row block
        # uses the comptime-unrolled MR path (each B-load reused across MR rows);
        # a tail M-block (m % MR) falls back to one row at a time.
        if r == MR:
            var j = 0
            while j + NR <= n:
                var acc = InlineArray[SIMD[dtype, NELTS], MR * NR_VECS](
                    fill=SIMD[dtype, NELTS](0)
                )
                for p in range(k):
                    var b_row = b_ptr + p * n + j
                    var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                        fill=SIMD[dtype, NELTS](0)
                    )
                    comptime for nv in range(NR_VECS):
                        bv[nv] = b_row.load[width=NELTS](offset=nv * NELTS)
                    var a_vals = InlineArray[Scalar[dtype], MR](
                        fill=Scalar[dtype](0)
                    )
                    comptime for mr in range(MR):
                        a_vals[mr] = a_ptr[(i + mr) * k + p]
                    _fma_tile[dtype, MR, NR_VECS, NELTS](acc, a_vals, bv)
                comptime for mr in range(MR):
                    comptime for nv in range(NR_VECS):
                        (c_ptr + (i + mr) * n + j).store(
                            offset=nv * NELTS, val=acc[mr * NR_VECS + nv]
                        )
                j += NR
        else:
            for ii in range(i, i + r):
                var j2 = 0
                while j2 + NR <= n:
                    var acc = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                        fill=SIMD[dtype, NELTS](0)
                    )
                    for p in range(k):
                        var b_row = b_ptr + p * n + j2
                        var av = SIMD[dtype, NELTS](a_ptr[ii * k + p])
                        comptime for nv in range(NR_VECS):
                            acc[nv] = av.fma(
                                b_row.load[width=NELTS](offset=nv * NELTS), acc[nv]
                            )
                    comptime for nv in range(NR_VECS):
                        (c_ptr + ii * n + j2).store(offset=nv * NELTS, val=acc[nv])
                    j2 += NR

        # N-remainder columns (n % NR < NR): scalar dot per (row, col). This is a
        # tiny strip (NR is at most 16 cols) so a scalar tail costs nothing.
        var jr = (n // NR) * NR
        for ii in range(i, i + r):
            var a_row = a_ptr + ii * k
            for jj in range(jr, n):
                var acc = Scalar[dtype](0)
                for p in range(k):
                    acc = fma(a_row[p], b_ptr[p * n + jj], acc)
                c_ptr[ii * n + jj] = acc

    parallelize(worker, num_blocks, nw)


@always_inline
def _prefill[
    dtype: DType, KC: Int, TILE_N: Int, SHARED_A: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    """The SOTA packed prefill GEMM at its standard 6x(4*NELTS) register tile
    (KU=2, NC_TILES=64). KC, TILE_N and SHARED_A are the only levers that vary
    across shapes, so naming the rest here keeps each dispatch branch readable.

    KU=2 (not 4): the 6x32 tile holds MR*NR_VECS = 24 SIMD accumulators, and the
    comptime k-unroll keeps KU*NR_VECS B-vectors live per step. KU=2 needs
    24 + 8 = 32 zmm registers — exactly the AVX-512 file — while KU=4 needs
    24 + 16 = 40 and spills. Restoring KU=2 (it had drifted to 4 here) is
    bit-identical (codegen-only; verify_dispatch max_err 0.0) and measured a
    uniform +2-6% across the heavy-GEMM band, flipping the Qwen up-proj M>=256,
    down-proj M=256 and h4k/ffn-up8k M=512 from LOSE to WIN — see README."""
    comptime NELTS = simd_width_of[dtype]()
    _prefill_gemm_v3[dtype, 6, 4 * NELTS, KC, 2, TILE_N, 64, SHARED_A](c, a, b)


def matmul_dispatch[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # C = A * B, routed to the kernel + tile that measured fastest for the shape
    # on this class of CPU (4c AVX-512 Xeon, f64). The full per-branch rationale
    # and measurements live in README.md ("Dispatch logic"); the summary:
    #
    #   tiny  M*N*K < 2^19          _matmul_small  serial, no threads/packing
    #   M == 1                      _decode_gemv   j-parallel GEMV, streams B once
    #   M in 2..5                   prefill MR=M   pack B once, reuse across rows
    #   thin-N  N<=64, M>=64        _thin_n_gemm   M-parallel, no packing
    #   small box  M>=N, B fits L2  _thin_n_gemm   M-parallel, no packing
    #   N <= 192                    prefill NR=16  narrow tile -> >= 4 j-tiles
    #   square-ish  N <= M          prefill 6x32   L2-adaptive KC, SHARED_A
    #   wide-N  N >= K              prefill 6x32   (8x24 only for N>=9k & M<=32)
    #   tall-K  N < K               prefill 6x32   cache-aware KC by M band
    #
    # Every KC/TILE_N/tile pick is hardware-specific. The N<=M and m>=n gates keep
    # the square-ish and no-pack routes off every wide headline shape (Qwen
    # up/down proj are N >> M), where those kernels are catastrophic.
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols

    if m * n * k < (1 << 19):
        # Tiny total work: a plain serial register-blocked loop. Below ~2^19 MACs
        # the parallel kernels' fixed cost (thread launch + per-worker packing
        # buffers) dwarfs the compute (sq8..32 ran 0.03-0.17x linalg parallel vs
        # 1.2-2.6x serial). Can never fire for a headline shape.
        _matmul_small[dtype, 6, 2](c, a, b)
    elif m == 1:
        # Pure decode GEMV: each worker owns an L1-resident column chunk of C and
        # streams B exactly once.
        _decode_gemv(c, a, b)
    elif m <= 5:
        # Small-batch decode: the prefill micro-kernel with MR = M, so the packed
        # B panel is streamed once and reused across all M rows (the GEMV would
        # re-stream B per row, ~2x slower at M=4). KU=2, TILE_N=8*NELTS.
        if m == 2:
            _prefill_gemm_v3[dtype, 2, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
        elif m == 3:
            _prefill_gemm_v3[dtype, 3, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
        elif m == 4:
            _prefill_gemm_v3[dtype, 4, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
        else:
            _prefill_gemm_v3[dtype, 5, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif n <= NELTS * 8 and m >= 64:
        # Thin-N, tall-M (N <= 64): the work is along M, but the prefill kernel
        # parallelizes only over N, so a thin N starves the cores. _thin_n_gemm
        # parallelizes over M-row blocks instead, reading A/B straight from source
        # (a thin N stays cache-resident, so packing buys nothing). Lifts the band
        # from 0.10-0.58 to 0.80-1.27 vs linalg. NR_VECS=1 for sub-16-wide N so it
        # still fills a SIMD panel; NR_VECS=2 otherwise.
        if n < 2 * NELTS:
            _thin_n_gemm[dtype, 6, 1](c, a, b)
        else:
            _thin_n_gemm[dtype, 6, 2](c, a, b)
    elif (
        m >= 64
        and m >= n
        and (
            k * n * size_of[Scalar[dtype]]() <= (1 << 19)
            or k * n * size_of[Scalar[dtype]]() <= _box_l2_budget()
        )
    ):
        # Small M-dominant box whose B stays L2-resident: same no-pack M-parallel
        # kernel (NR=32). The packed prefill kernel's packing + thread-launch
        # overhead dwarfs the compute on a cache-resident box (sq96..256 ran
        # 0.65-0.79 packed; no-pack flips them to 1.0-1.18). The B-fits-L2 test is
        # two tiered: a compile-time 512 KB cut (no cpuid on the few-us shapes)
        # plus an L2-adaptive B <= L2/3 tier (_box_l2_budget). m >= n keeps it off
        # every wide headline shape; m >= 64 gives enough row-blocks to fill 4c.
        _thin_n_gemm[dtype, 6, 4](c, a, b)
    elif n <= 3 * 64:
        # Narrow N (<= 192): at the default TILE_N=64 such an N is < 4 j-tiles, so
        # the N-only parallelism idles most of a 4-core box. A narrow NR=16 /
        # TILE_N=16 tile splits N into >= num_workers j-tiles (sq64 0.44->0.83).
        _prefill_gemm_v3[dtype, 6, 2 * NELTS, 256, 4, 2 * NELTS, 64](c, a, b)
    elif n <= m:
        # Square-ish (192 < N <= M). The wide/tall branches' TILE_N=64 + big
        # cache-aware KC fit a box-shaped C badly (few coarse j-tiles; a fat M*KC
        # packed-A with no C-traffic saving to amortize it). A finer tile + smaller
        # KC keep packed-A and packed-B L2-resident and multiply the j-tile count
        # (sq512 0.74->0.88, sq2048 0.70->0.85). Two levers, each at its measured
        # best per machine:
        #   * KC by detected L2 (_square_ish_kc): 512 on a 1 MB/core L2, 1024 on
        #     2 MB/core (the two machines reach opposite picks a constant can't).
        #   * TILE_N by load balance (the #1 win lever): the fatter 8*NELTS only
        #     when N tiles evenly across workers (>= 2 each), else the finer
        #     4*NELTS, which collapses less on awkward N.
        # SHARED_A: here A is as large as B/C, so packing it once (not per-worker)
        # is a real win, unlike the wide/tall headline shapes where A is tiny.
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
        # Wide-N (up-proj-like). The N-balanced 6x32 tile (TILE_N=8*NELTS=64, 32
        # even j-tiles on N=2048) wins almost everywhere now that the masked
        # M-remainder tail removed the MR-divides-M tax.
        if n >= 9 * 1024 and m <= 32:
            # The one corner where the older 8x24 tile still edges 6x32 by ~2-4%:
            # very wide N and tiny M (the Qwen up-proj small batch). KU=2.
            _prefill_gemm_v3[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS, 64](c, a, b)
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
