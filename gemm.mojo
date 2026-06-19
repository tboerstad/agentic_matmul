from cpu_cache import l2_cache_size
from matrix import Matrix
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.math import ceildiv, fma
from std.memory import memset_zero
from std.memory.unsafe_pointer import alloc
from std.sys import num_physical_cores, simd_width_of, size_of
from std.sys.intrinsics import prefetch, PrefetchOptions


def matmul_naive[dtype: DType = DType.float64](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * B  —  simple triple-nested loop (ijk order).
    var m = a.rows
    var n = c.cols
    var k = a.cols
    for i in range(m):
        for j in range(n):
            var dot = Scalar[dtype](0)
            for p in range(k):
                var a_val = a[i, p]
                dot += a_val * b[p, j]

            c[i, j] = dot


def matmul_tiled[dtype: DType = DType.float64](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * B  —  tiled / cache-blocked version.
    #
    # Key optimizations over naive:
    #   1. Loop tiling: process TILE x TILE sub-blocks so data fits in L1/L2 cache
    #   2. Accumulate into a local register variable before writing back to C
    #   3. Loop order i→p→j inside tiles keeps A[i,p] reads sequential and
    #      reuses each loaded A element across the full j-tile
    comptime TILE = 32

    var m = a.rows
    var n = c.cols
    var k = a.cols

    _zero_fill[dtype](c)

    # Tile over all three dimensions
    for i0 in range(0, m, TILE):
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m
        for p0 in range(0, k, TILE):
            var p_end = p0 + TILE
            if p_end > k:
                p_end = k
            for j0 in range(0, n, TILE):
                var j_end = j0 + TILE
                if j_end > n:
                    j_end = n

                # Micro-kernel: multiply the (i0:i_end, p0:p_end) block of A
                # with the (p0:p_end, j0:j_end) block of B, accumulating into C
                for i in range(i0, i_end):
                    for p in range(p0, p_end):
                        var a_val = a[i, p]
                        for j in range(j0, j_end):
                            c[i, j] = c[i, j] + a_val * b[p, j]


def matmul_simd[dtype: DType = DType.float64](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * B  —  tiled + SIMD vectorized version.
    #
    # Uses Mojo's `vectorize` to auto-vectorize the innermost j-loop,
    # handling SIMD-width chunks and scalar remainders automatically.
    comptime TILE = 32
    comptime NELTS = simd_width_of[dtype]()

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()

    _zero_fill[dtype](c)

    # Tile over all three dimensions
    for i0 in range(0, m, TILE):
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m
        for p0 in range(0, k, TILE):
            var p_end = p0 + TILE
            if p_end > k:
                p_end = k
            for j0 in range(0, n, TILE):
                var j_end = j0 + TILE
                if j_end > n:
                    j_end = n
                var tile_n = j_end - j0

                # Micro-kernel with SIMD vectorization on j dimension
                for i in range(i0, i_end):
                    for p in range(p0, p_end):
                        var a_val = a[i, p]
                        var c_row = c_ptr + i * n + j0
                        var b_row = b_ptr + p * n + j0

                        def fma[width: Int](j: Int) {mut c_row, read b_row, read a_val}:
                            var c_vec = c_row.load[width=width](offset=j)
                            var b_vec = b_row.load[width=width](offset=j)
                            c_row.store(offset=j, val=c_vec + a_val * b_vec)

                        vectorize[NELTS](tile_n, fma)


def matmul_parallel[dtype: DType = DType.float64](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * B  —  tiled + SIMD + multi-threaded version.
    #
    # Parallelizes the outer i-tile loop across CPU cores while keeping
    # the SIMD-vectorized micro-kernel from matmul_simd. Each thread
    # owns a disjoint set of row tiles so no synchronization is needed.
    comptime TILE = 32
    comptime NELTS = simd_width_of[dtype]()

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    _zero_fill[dtype](c)

    # Number of row tiles
    var num_i_tiles = ceildiv(m, TILE)

    def process_i_tile(tile_idx: Int) {mut c_ptr, mut b_ptr, mut a_ptr, read m, read n, read k}:
        var i0 = tile_idx * TILE
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m

        # Hoisted captures for inner closure (rule #5)
        var a_val: Scalar[dtype]
        var c_row: type_of(c_ptr)
        var b_row: type_of(b_ptr)

        for p0 in range(0, k, TILE):
            var p_end = p0 + TILE
            if p_end > k:
                p_end = k
            for j0 in range(0, n, TILE):
                var j_end = j0 + TILE
                if j_end > n:
                    j_end = n
                var tile_n = j_end - j0

                # Micro-kernel with SIMD vectorization on j dimension
                for i in range(i0, i_end):
                    for p in range(p0, p_end):
                        a_val = a_ptr[i * k + p]
                        c_row = c_ptr + i * n + j0
                        b_row = b_ptr + p * n + j0

                        def fma[width: Int](j: Int) {mut c_row, read b_row, read a_val}:
                            var c_vec = c_row.load[width=width](offset=j)
                            var b_vec = b_row.load[width=width](offset=j)
                            c_row.store(offset=j, val=c_vec + a_val * b_vec)

                        vectorize[NELTS](tile_n, fma)

    parallelize(process_i_tile, num_i_tiles, num_physical_cores())


def matmul_register_blocked[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  tiled + SIMD + parallel + register-blocked.
    #
    # Key optimization over matmul_parallel:
    #   Register blocking (micro-kernel): processes MR=4 rows of C per inner
    #   loop iteration.  Each B vector loaded from cache is reused across all
    #   MR rows, cutting B-side memory traffic by 4x and improving the
    #   compute-to-load ratio of the inner loop.
    comptime TILE = 32
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 4  # rows of C per micro-kernel invocation

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    _zero_fill[dtype](c)

    var num_i_tiles = ceildiv(m, TILE)

    def process_i_tile(tile_idx: Int) {mut c_ptr, mut b_ptr, mut a_ptr, read m, read n, read k}:
        var i0 = tile_idx * TILE
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m

        # Hoisted captures for inner closures (rule #5)
        var a_vals = InlineArray[Scalar[dtype], MR](fill=Scalar[dtype](0))
        var b_row: type_of(b_ptr)
        var c_row: type_of(c_ptr)
        var a_val: Scalar[dtype]
        var i: Int
        var j0_h: Int

        for p0 in range(0, k, TILE):
            var p_end = p0 + TILE
            if p_end > k:
                p_end = k
            for j0 in range(0, n, TILE):
                var j_end = j0 + TILE
                if j_end > n:
                    j_end = n
                var tile_n = j_end - j0
                j0_h = j0

                # Register-blocked: process MR rows at a time
                i = i0
                while i + MR <= i_end:
                    for p in range(p0, p_end):
                        comptime for mr in range(MR):
                            a_vals[mr] = a_ptr[(i + mr) * k + p]
                        b_row = b_ptr + p * n + j0

                        def fma_mr[width: Int](j: Int) {mut c_ptr, read b_row, read a_vals, read i, read n, read j0_h}:
                            var b_vec = b_row.load[width=width](offset=j)
                            comptime for mr in range(MR):
                                var c_row_inner = c_ptr + (i + mr) * n + j0_h
                                c_row_inner.store(
                                    offset=j,
                                    val=c_row_inner.load[width=width](offset=j)
                                    + a_vals[mr] * b_vec,
                                )

                        vectorize[NELTS](tile_n, fma_mr)
                    i += MR

                # Handle remaining rows (< MR) with single-row SIMD
                while i < i_end:
                    for p in range(p0, p_end):
                        a_val = a_ptr[i * k + p]
                        c_row = c_ptr + i * n + j0
                        b_row = b_ptr + p * n + j0

                        def fma[width: Int](j: Int) {mut c_row, read b_row, read a_val}:
                            var c_vec = c_row.load[width=width](offset=j)
                            var b_vec = b_row.load[width=width](offset=j)
                            c_row.store(
                                offset=j, val=c_vec + a_val * b_vec
                            )

                        vectorize[NELTS](tile_n, fma)
                    i += 1

    parallelize(process_i_tile, num_i_tiles, num_physical_cores())


def matmul_packed[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  tiled + SIMD + parallel + register-blocked
    #                        + C-accumulation in registers.
    #
    # Key optimization over matmul_register_blocked:
    #   Register accumulation: the previous version loads and stores C vectors
    #   on every k-iteration (TILE times per tile).  This version restructures
    #   the micro-kernel loop order to j→p: for each j-block of NELTS elements,
    #   load MR C accumulators once, iterate over all k-values accumulating in
    #   registers, then store once.  This cuts C-side memory traffic by ~TILE×
    #   (32× for TILE=32), keeping the hottest data in CPU registers.
    comptime TILE = 32
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 4  # rows of C per micro-kernel invocation

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    _zero_fill[dtype](c)

    var num_i_tiles = ceildiv(m, TILE)

    def process_i_tile(tile_idx: Int) {mut c_ptr, mut b_ptr, mut a_ptr, read m, read n, read k}:
        var i0 = tile_idx * TILE
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m

        # Hoisted captures for inner closures (rule #5)
        var i: Int
        var j0_h: Int
        var p0_h: Int
        var tile_k_h: Int
        var c_row: type_of(c_ptr)

        for p0 in range(0, k, TILE):
            var p_end = p0 + TILE
            if p_end > k:
                p_end = k
            var tile_k = p_end - p0
            p0_h = p0
            tile_k_h = tile_k
            for j0 in range(0, n, TILE):
                var j_end = j0 + TILE
                if j_end > n:
                    j_end = n
                var tile_n = j_end - j0
                j0_h = j0

                # Register-accumulation micro-kernel: j→p loop order
                # For each j-block, load C into registers, accumulate
                # across all k-values, then store back once.
                i = i0
                while i + MR <= i_end:
                    def process_cols[width: Int](j: Int) {mut c_ptr, mut b_ptr, mut a_ptr, read i, read j0_h, read p0_h, read tile_k_h, read n, read k}:
                        var acc = InlineArray[SIMD[dtype, width], MR](
                            fill=SIMD[dtype, width](0)
                        )
                        comptime for mr in range(MR):
                            acc[mr] = (c_ptr + (i + mr) * n + j0_h).load[
                                width=width
                            ](offset=j)
                        for pk in range(tile_k_h):
                            var p = p0_h + pk
                            var b_vec = (b_ptr + p * n + j0_h).load[width=width](offset=j)
                            comptime for mr in range(MR):
                                acc[mr] += a_ptr[(i + mr) * k + p] * b_vec
                        comptime for mr in range(MR):
                            (c_ptr + (i + mr) * n + j0_h).store(
                                offset=j, val=acc[mr]
                            )

                    vectorize[NELTS](tile_n, process_cols)
                    i += MR

                # Handle remaining rows (< MR) with single-row accumulation
                while i < i_end:
                    c_row = c_ptr + i * n + j0

                    def process_tail_col[width: Int](j: Int) {mut c_row, mut b_ptr, mut a_ptr, read i, read j0_h, read p0_h, read tile_k_h, read n, read k}:
                        var acc = c_row.load[width=width](offset=j)
                        for pk in range(tile_k_h):
                            var p = p0_h + pk
                            acc += a_ptr[i * k + p] * (b_ptr + p * n + j0_h).load[width=width](offset=j)
                        c_row.store(offset=j, val=acc)

                    vectorize[NELTS](tile_n, process_tail_col)
                    i += 1

    parallelize(process_i_tile, num_i_tiles, num_physical_cores())


@always_inline
def _zero_fill[dtype: DType](mut c: Matrix[dtype]):
    """Vectorized zero-fill using SIMD stores."""
    comptime NELTS = simd_width_of[dtype]()
    var ptr = c.data.unsafe_ptr()
    var count = c.rows * c.cols
    def _zero[width: Int](idx: Int) {mut ptr}:
        ptr.store[width=width](offset=idx, val=SIMD[dtype, width](0))
    vectorize[NELTS](count, _zero)


def matmul_comptime[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  compile-time optimized GOTO-style GEMM.
    #
    # Key optimizations over matmul_packed:
    #   1. Parallelize over j-tiles (N-dimension) instead of i-tiles (M-dimension)
    #      for better load balance (172 j-tiles vs 3 i-tiles across 4 cores)
    #   2. j→k→i loop order: C panel stays in L1 across all k-tiles
    #   3. MR×NR register blocking with comptime-unrolled k-loop (KU=8)
    #   4. Software prefetching to hide B-load latency
    #
    # Uses Mojo's compile-time metaprogramming for the micro-kernel:
    #   - comptime for: generates MR×NR×KU unrolled FMA code at compile time
    #   - InlineArray[SIMD]: accumulator array that LLVM register-promotes
    #   - LLVM prefetch intrinsic: software prefetching for B data
    #   - vectorize: SIMD zero-fill with automatic remainder handling
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 6          # rows of C per micro-kernel
    comptime NR = 2          # SIMD vectors of C cols per micro-kernel
    comptime MICRO_N = NR * NELTS  # columns per micro-kernel pass
    comptime TILE_K = 256    # k-tile: B chunk = 256*64*8 = 128KB fits L2
    comptime TILE_N = 64     # j-tile: C panel = M*64*8 fits L1
    comptime KU = 8          # k-unroll factor (8 > 4: less loop overhead)
    comptime PREFETCH_DIST = 4  # prefetch distance in k-steps

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    _zero_fill[dtype](c)

    var num_j_tiles = ceildiv(n, TILE_N)

    def process_j_tile(j_tile_idx: Int) {mut c_ptr, mut b_ptr, mut a_ptr, read m, read n, read k}:
        var j0 = j_tile_idx * TILE_N
        var j_end = j0 + TILE_N
        if j_end > n:
            j_end = n
        var tile_n = j_end - j0

        # Hoisted captures for inner closures (rule #5)
        var i: Int
        var rem_base: Int
        var p0_h: Int
        var tile_k_h: Int
        var j0_h: Int
        var c_row: type_of(c_ptr)

        # j→k→i order: C panel stays in L1 across all k-tiles
        for p0 in range(0, k, TILE_K):
            var p_end = p0 + TILE_K
            if p_end > k:
                p_end = k
            var tile_k = p_end - p0
            p0_h = p0
            tile_k_h = tile_k

            # ---- MR-blocked rows with comptime micro-kernel ----
            i = 0
            while i + MR <= m:
                # Process MICRO_N columns at a time
                var j = 0
                while j + MICRO_N <= tile_n:
                    var jj = j0 + j  # absolute column index

                    # Load MR×NR accumulators from C into registers
                    var acc = InlineArray[SIMD[dtype, NELTS], MR * NR](
                        fill=SIMD[dtype, NELTS](0)
                    )
                    comptime for mr in range(MR):
                        comptime for nr in range(NR):
                            acc[mr * NR + nr] = (
                                c_ptr + (i + mr) * n + jj
                            ).load[width=NELTS](offset=nr * NELTS)

                    # K-loop with comptime KU unrolling
                    var pk = 0
                    var pk_end = tile_k - (tile_k % KU)
                    while pk < pk_end:
                        comptime for ku in range(KU):
                            var p = p0 + pk + ku
                            # Prefetch B data PREFETCH_DIST steps ahead
                            if pk + ku + PREFETCH_DIST < tile_k:
                                prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                                    b_ptr + (p + PREFETCH_DIST) * n + jj,
                                )
                            # Load MR A values into register-promoted array
                            var a_vals = InlineArray[Scalar[dtype], MR](
                                fill=Scalar[dtype](0)
                            )
                            comptime for mr in range(MR):
                                a_vals[mr] = a_ptr[(i + mr) * k + p]
                            # Load NR B vectors, FMA against all MR rows
                            comptime for nr in range(NR):
                                var bv = (b_ptr + p * n + jj).load[
                                    width=NELTS
                                ](offset=nr * NELTS)
                                comptime for mr in range(MR):
                                    acc[mr * NR + nr] += a_vals[mr] * bv
                        pk += KU

                    # K remainder (no unrolling)
                    while pk < tile_k:
                        var p = p0 + pk
                        var a_vals = InlineArray[Scalar[dtype], MR](
                            fill=Scalar[dtype](0)
                        )
                        comptime for mr in range(MR):
                            a_vals[mr] = a_ptr[(i + mr) * k + p]
                        comptime for nr in range(NR):
                            var bv = (b_ptr + p * n + jj).load[
                                width=NELTS
                            ](offset=nr * NELTS)
                            comptime for mr in range(MR):
                                acc[mr * NR + nr] += a_vals[mr] * bv
                        pk += 1

                    # Store accumulators back to C
                    comptime for mr in range(MR):
                        comptime for nr in range(NR):
                            (c_ptr + (i + mr) * n + jj).store(
                                offset=nr * NELTS, val=acc[mr * NR + nr]
                            )

                    j += MICRO_N

                # Remainder columns: vectorize handles SIMD + scalar
                var rem_n = tile_n - j
                rem_base = j0 + j

                def process_rem[width: Int](j_off: Int) {mut c_ptr, mut b_ptr, mut a_ptr, read i, read rem_base, read p0_h, read tile_k_h, read n, read k}:
                    var jj = rem_base + j_off
                    var acc_r = InlineArray[SIMD[dtype, width], MR](
                        fill=SIMD[dtype, width](0)
                    )
                    comptime for mr in range(MR):
                        acc_r[mr] = (c_ptr + (i + mr) * n + jj).load[
                            width=width
                        ]()
                    for pk in range(tile_k_h):
                        var p = p0_h + pk
                        var bv = (b_ptr + p * n + jj).load[width=width]()
                        comptime for mr in range(MR):
                            acc_r[mr] += a_ptr[(i + mr) * k + p] * bv
                    comptime for mr in range(MR):
                        (c_ptr + (i + mr) * n + jj).store(
                            val=acc_r[mr]
                        )

                vectorize[NELTS](rem_n, process_rem)
                i += MR

            # Handle remaining rows (< MR) with single-row SIMD
            j0_h = j0
            while i < m:
                c_row = c_ptr + i * n + j0

                def process_tail[width: Int](j: Int) {mut c_row, mut b_ptr, mut a_ptr, read i, read j0_h, read p0_h, read tile_k_h, read n, read k}:
                    var acc = c_row.load[width=width](offset=j)
                    for pk in range(tile_k_h):
                        var p = p0_h + pk
                        acc += a_ptr[i * k + p] * (b_ptr + p * n + j0_h).load[width=width](offset=j)
                    c_row.store(offset=j, val=acc)

                vectorize[NELTS](tile_n, process_tail)
                i += 1

    parallelize(process_j_tile, num_j_tiles, num_physical_cores())


def _goto_gemv[
    dtype: DType,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # GEMV path for small M (esp. M=1 decode).
    # Streams through B sequentially row-by-row with j-parallelism,
    # enabling hardware prefetching and maximizing DRAM bandwidth.
    comptime NELTS = simd_width_of[dtype]()
    comptime TILE_J = 1024  # C chunk = 1024*8 = 8KB, fits L1

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()

    var num_j_tiles = ceildiv(n, TILE_J)

    def process_gemv_tile(tile_idx: Int) {mut c_ptr, mut a_ptr, mut b_ptr, read m, read n, read k}:
        var j0 = tile_idx * TILE_J
        var tile_n = min(TILE_J, n - j0)

        # Hoisted captures for inner closure (rule #5)
        var c_row: type_of(c_ptr)
        var b_row: type_of(b_ptr)
        var a_val: Scalar[dtype]

        for i in range(m):
            c_row = c_ptr + i * n + j0
            for p in range(k):
                a_val = a_ptr[i * k + p]
                b_row = b_ptr + p * n + j0

                def fma_gemv[width: Int](j: Int) {mut c_row, read b_row, read a_val}:
                    c_row.store(
                        offset=j,
                        val=c_row.load[width=width](offset=j)
                        + a_val * b_row.load[width=width](offset=j),
                    )

                vectorize[NELTS](tile_n, fma_gemv)

    parallelize(process_gemv_tile, num_j_tiles, num_physical_cores())


def _goto_gemm[
    dtype: DType, MR: Int, NR: Int, KC: Int, KU: Int, TILE_N: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # j-parallel GOTO GEMM with per-tile B-panel packing.
    # j→k→i loop order keeps C panel in L2 across all k-tiles.
    # B is packed into NR-wide contiguous column panels for
    # sequential micro-kernel access.
    comptime NELTS = simd_width_of[dtype]()
    comptime NR_VECS = NR // NELTS  # number of SIMD vectors per NR panel
    comptime NUM_LOCAL_PANELS = TILE_N // NR

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()

    var num_j_tiles = ceildiv(n, TILE_N)

    # Opt 1: Zero-cost allocation — raw uninitialized memory instead of
    # List + resize (which zeroes the buffer needlessly).
    var bp_per_tile = NUM_LOCAL_PANELS * KC * NR + KU * NR
    var bp_total = num_j_tiles * bp_per_tile
    var bp_buf = alloc[Scalar[dtype]](bp_total)

    def process_j_tile(j_tile_idx: Int) {mut c_ptr, mut a_ptr, mut b_ptr, mut bp_buf, read m, read n, read k, read bp_per_tile}:
        var j0 = j_tile_idx * TILE_N
        var tile_n = min(TILE_N, n - j0)
        var num_panels = ceildiv(tile_n, NR)

        var bp_tile = bp_buf + j_tile_idx * bp_per_tile

        # Hoisted captures for inner closures (rule #5)
        var c_row: type_of(c_ptr)
        var a_row: type_of(a_ptr)
        var bp_panel: type_of(bp_tile)
        var first_k: Bool
        var kc_h: Int
        var pc_h: Int
        var i_h: Int

        for pc in range(0, k, KC):
            var kc = min(KC, k - pc)
            first_k = (pc == 0)
            kc_h = kc
            pc_h = pc

            # Pack B[pc:pc+kc, j0:j0+tile_n] into NR-panels
            for jp in range(num_panels):
                var jr = jp * NR
                var panel_base = bp_tile + jp * kc * NR
                if jr + NR <= tile_n:
                    for pk in range(kc):
                        var src = b_ptr + (pc + pk) * n + j0 + jr
                        var dst = panel_base + pk * NR
                        comptime for nv in range(NR_VECS):
                            dst.store[width=NELTS](
                                offset=nv * NELTS,
                                val=src.load[width=NELTS](offset=nv * NELTS),
                            )
                else:
                    var nr_actual = tile_n - jr
                    for pk in range(kc):
                        var src = b_ptr + (pc + pk) * n + j0 + jr
                        var dst = panel_base + pk * NR
                        for nr in range(nr_actual):
                            dst[nr] = src[nr]
                        for nr in range(nr_actual, NR):
                            dst[nr] = Scalar[dtype](0)

            # Micro-kernel: process MR rows at a time
            var i = 0
            while i + MR <= m:
                for jp in range(num_panels):
                    var jr = jp * NR
                    bp_panel = bp_tile + jp * kc * NR
                    if jr + NR > tile_n:
                        # Opt 2: vectorize handles SIMD + scalar remainder
                        var jj_limit = tile_n - jr
                        for ii in range(i, i + MR):
                            c_row = c_ptr + ii * n + j0 + jr
                            a_row = a_ptr + ii * k + pc

                            def fma_remainder[width: Int](jj: Int) {mut c_row, read a_row, read bp_panel, read first_k, read kc_h}:
                                var acc = c_row.load[width=width](offset=jj)
                                if first_k:
                                    acc = SIMD[dtype, width](0)
                                for ppk in range(kc_h):
                                    acc = fma(
                                        SIMD[dtype, width](a_row[ppk]),
                                        (bp_panel + ppk * NR).load[width=width](offset=jj),
                                        acc,
                                    )
                                c_row.store(offset=jj, val=acc)

                            vectorize[NELTS](jj_limit, fma_remainder)
                        continue

                    # ---- Full MR×NR micro-kernel ----
                    # Opt 5: Skip zero-fill + first-tile C load — on the first
                    # k-tile (pc==0), initialize accumulators to zero instead
                    # of loading from C. Saves writing 8.4MB of zeros
                    # (_zero_fill) and reading them back (first C load).
                    var acc = InlineArray[SIMD[dtype, NELTS], MR * NR_VECS](
                        fill=SIMD[dtype, NELTS](0)
                    )
                    if not first_k:
                        comptime for mr in range(MR):
                            comptime for nr in range(NR_VECS):
                                acc[mr * NR_VECS + nr] = (
                                    c_ptr + (i + mr) * n + j0 + jr
                                ).load[width=NELTS](offset=nr * NELTS)

                    # K-loop with KU unrolling + prefetching
                    var pk = 0
                    var pk_end = kc - (kc % KU)
                    while pk < pk_end:
                        comptime for ku in range(KU):
                            var bp_k = bp_panel + (pk + ku) * NR
                            # Opt 3: Built-in hardware prefetch targeting L1
                            prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                                bp_panel + (pk + ku + 4) * NR
                            )
                            var a_vals = InlineArray[Scalar[dtype], MR](
                                fill=Scalar[dtype](0)
                            )
                            comptime for mr in range(MR):
                                a_vals[mr] = a_ptr[
                                    (i + mr) * k + pc + pk + ku
                                ]
                            comptime for nr in range(NR_VECS):
                                var bv = bp_k.load[width=NELTS](
                                    offset=nr * NELTS
                                )
                                # Opt 4: Explicit FMA instruction
                                comptime for mr in range(MR):
                                    acc[mr * NR_VECS + nr] = fma(
                                        SIMD[dtype, NELTS](a_vals[mr]), bv, acc[mr * NR_VECS + nr]
                                    )
                        pk += KU

                    # K remainder
                    while pk < kc:
                        var bp_k = bp_panel + pk * NR
                        var a_vals = InlineArray[Scalar[dtype], MR](
                            fill=Scalar[dtype](0)
                        )
                        comptime for mr in range(MR):
                            a_vals[mr] = a_ptr[(i + mr) * k + pc + pk]
                        comptime for nr in range(NR_VECS):
                            var bv = bp_k.load[width=NELTS](
                                offset=nr * NELTS
                            )
                            comptime for mr in range(MR):
                                acc[mr * NR_VECS + nr] = fma(
                                    SIMD[dtype, NELTS](a_vals[mr]), bv, acc[mr * NR_VECS + nr]
                                )
                        pk += 1

                    # Store accumulators back to C
                    comptime for mr in range(MR):
                        comptime for nr in range(NR_VECS):
                            (c_ptr + (i + mr) * n + j0 + jr).store(
                                offset=nr * NELTS, val=acc[mr * NR_VECS + nr],
                            )

                i += MR

            # Handle remaining rows (< MR) — vectorize replaces manual loops
            while i < m:
                i_h = i
                for jp in range(num_panels):
                    var jr = jp * NR
                    bp_panel = bp_tile + jp * kc * NR
                    var jj_limit = min(NR, tile_n - jr)
                    c_row = c_ptr + i * n + j0 + jr

                    def fma_tail[width: Int](jj: Int) {mut c_row, mut a_ptr, read bp_panel, read first_k, read kc_h, read pc_h, read i_h, read k}:
                        var acc = c_row.load[width=width](offset=jj)
                        if first_k:
                            acc = SIMD[dtype, width](0)
                        for ppk in range(kc_h):
                            acc = fma(
                                SIMD[dtype, width](a_ptr[i_h * k + pc_h + ppk]),
                                (bp_panel + ppk * NR).load[width=width](offset=jj),
                                acc,
                            )
                        c_row.store(offset=jj, val=acc)

                    vectorize[NELTS](jj_limit, fma_tail)
                i += 1

    parallelize(process_j_tile, num_j_tiles, num_physical_cores())
    bp_buf.free()


def matmul_goto[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  GOTO-style GEMM with B-panel packing.
    # Dispatches to _goto_gemv (M < MR) or _goto_gemm (M >= MR).
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 8          # rows of C per micro-kernel (8 > 6: more B-reuse per A load)
    comptime NR = 2 * NELTS  # columns per micro-kernel (16 for float64 AVX-512)
    comptime KC = 512        # k-tile: fewer k-tiles halves B-packing + C load/store overhead
    comptime KU = 8          # k-unroll factor
    comptime TILE_N = 64     # j-tile: C panel = M*64*8 fits L2

    if a.rows < MR:
        _zero_fill[dtype](c)
        _goto_gemv[dtype](c, a, b)
    else:
        _goto_gemm[dtype, MR, NR, KC, KU, TILE_N](c, a, b)


def _prefill_gemm[
    dtype: DType, MR: Int, NR: Int, KC: Int, KU: Int, TILE_N: Int,
    NC_TILES: Int,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Optimized prefill GEMM with per-worker A-panel packing + B-panel packing.
    #
    # Key improvements over _goto_gemm:
    #   1. Worker-based parallelism: each thread gets a contiguous chunk of
    #      j-tiles instead of individual tiles. Reduces parallelize overhead.
    #   2. A-panel packing: pack A into MR-contiguous layout, shared across
    #      NC_TILES j-tiles per batch. Amortizes packing cost.
    #   3. Batched j-tile processing: process NC_TILES j-tiles per k-tile
    #      iteration to keep C panel in L2 while reusing packed A.
    #   4. Eliminates separate zero-fill pass: first k-tile uses zero accumulators.
    #   5. B-packing prefetch hides strided memory access latency.
    #   6. Packed-A prefetch in micro-kernel ensures data is in L1 before use.
    #   7. Row-major B packing: packs all NR-panels per row in inner loop,
    #      crossing the stride-N gap once per row instead of once per panel.
    #   8. C-panel prefetch with write intent before loading accumulators.
    comptime NELTS = simd_width_of[dtype]()
    comptime NR_VECS = NR // NELTS
    comptime PREFETCH_B_DIST = 8  # rows ahead to prefetch during B packing
    comptime PREFETCH_DIST = 4    # k-steps ahead to prefetch packed A/B

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()

    var num_j_tiles = ceildiv(n, TILE_N)
    var num_i_panels = ceildiv(m, MR)
    var num_workers = num_physical_cores()

    # Per-worker buffers
    # B pack: one TILE_N worth of NR-panels
    var num_nr_panels = ceildiv(TILE_N, NR)
    var bp_per_worker = num_nr_panels * KC * NR + KU * NR
    var bp_total = num_workers * bp_per_worker
    var bp_buf = alloc[Scalar[dtype]](bp_total)

    # A pack: all i-panels × MR × KC (shared across j-tiles within worker)
    var ap_per_worker = num_i_panels * MR * KC
    var ap_total = num_workers * ap_per_worker
    var ap_buf = alloc[Scalar[dtype]](ap_total)

    def process_worker(worker_id: Int) {mut c_ptr, mut a_ptr, mut b_ptr, mut bp_buf, mut ap_buf, read m, read n, read k, read num_j_tiles, read num_workers, read bp_per_worker, read ap_per_worker}:
        var tiles_per_worker = ceildiv(num_j_tiles, num_workers)
        var j_tile_start = worker_id * tiles_per_worker
        var j_tile_end = min(j_tile_start + tiles_per_worker, num_j_tiles)
        if j_tile_start >= num_j_tiles:
            return

        var bp_worker = bp_buf + worker_id * bp_per_worker
        var ap_worker = ap_buf + worker_id * ap_per_worker

        # Hoisted captures for inner closures (rule #5)
        var c_row: type_of(c_ptr)
        var a_row: type_of(a_ptr)
        var bp_panel: type_of(bp_worker)
        var is_first_k: Bool
        var kc_h: Int
        var pc_h: Int
        var i_h: Int

        # Process j-tiles in batches of NC_TILES to keep C in L2
        var jt = j_tile_start
        while jt < j_tile_end:
            var jt_batch_end = min(jt + NC_TILES, j_tile_end)

            for pc in range(0, k, KC):
                var kc = min(KC, k - pc)
                is_first_k = (pc == 0)
                kc_h = kc
                pc_h = pc

                # ---- Pack A once for this k-tile (shared across all j-tiles in batch) ----
                var i = 0
                var ip = 0
                while i + MR <= m:
                    var ap_panel = ap_worker + ip * MR * kc
                    for pk in range(kc):
                        var dst = ap_panel + pk * MR
                        comptime for mr in range(MR):
                            dst[mr] = a_ptr[(i + mr) * k + pc + pk]
                    i += MR
                    ip += 1

                # ---- Process each j-tile in this batch ----
                for j_tile_idx in range(jt, jt_batch_end):
                    var j0 = j_tile_idx * TILE_N
                    var tile_n = min(TILE_N, n - j0)
                    var num_panels = ceildiv(tile_n, NR)

                    # Pack B row-major: iterate rows (pk) in outer loop,
                    # panels (jp) in inner loop. Each B row's TILE_N columns
                    # are contiguous in memory so all panels read from the
                    # same cache lines, crossing the stride-N gap only once
                    # per row instead of once per panel (3× fewer strides).
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
                        # Prefetch entire tile width of next rows
                        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                            b_ptr + (pc + pk + PREFETCH_B_DIST) * n + j0
                        )
                        # Pack full NR-panels from this row
                        for jp in range(last_full_panel):
                            var jr = jp * NR
                            var src = row_base + jr
                            var dst = bp_worker + jp * kc * NR + pk * NR
                            comptime for nv in range(NR_VECS):
                                dst.store[width=NELTS](
                                    offset=nv * NELTS,
                                    val=src.load[width=NELTS](offset=nv * NELTS),
                                )
                        # Pack remainder panel (if any)
                        if has_remainder:
                            var jr = last_full_panel * NR
                            var src = row_base + jr
                            var dst = bp_worker + last_full_panel * kc * NR + pk * NR
                            for nr in range(nr_actual):
                                dst[nr] = src[nr]
                            for nr in range(nr_actual, NR):
                                dst[nr] = Scalar[dtype](0)

                    # Micro-kernel with packed A + packed B
                    # jp-outer, i-inner: B panel stays in L2 across all i-panels,
                    # reducing L2→L1 traffic by ~2.7× vs the i-outer order.
                    for jp in range(num_panels):
                        var jr = jp * NR
                        bp_panel = bp_worker + jp * kc * NR

                        if jr + NR > tile_n:
                            # Remainder columns: process all i-panels
                            var jj_limit = tile_n - jr
                            i = 0
                            while i + MR <= m:
                                for ii in range(i, i + MR):
                                    c_row = c_ptr + ii * n + j0 + jr
                                    a_row = a_ptr + ii * k + pc

                                    def pf_fma_remainder[width: Int](jj: Int) {mut c_row, read a_row, read bp_panel, read is_first_k, read kc_h}:
                                        var acc = c_row.load[width=width](offset=jj)
                                        if is_first_k:
                                            acc = SIMD[dtype, width](0)
                                        for ppk in range(kc_h):
                                            acc = fma(
                                                SIMD[dtype, width](a_row[ppk]),
                                                (bp_panel + ppk * NR).load[width=width](offset=jj),
                                                acc,
                                            )
                                        c_row.store(offset=jj, val=acc)

                                    vectorize[NELTS](jj_limit, pf_fma_remainder)
                                i += MR
                            # Remaining rows for remainder columns
                            while i < m:
                                i_h = i
                                c_row = c_ptr + i * n + j0 + jr

                                def pf_fma_tail_rem[width: Int](jj: Int) {mut c_row, mut a_ptr, read bp_panel, read is_first_k, read kc_h, read pc_h, read i_h, read k}:
                                    var acc = c_row.load[width=width](offset=jj)
                                    if is_first_k:
                                        acc = SIMD[dtype, width](0)
                                    for ppk in range(kc_h):
                                        acc = fma(
                                            SIMD[dtype, width](a_ptr[i_h * k + pc_h + ppk]),
                                            (bp_panel + ppk * NR).load[width=width](offset=jj),
                                            acc,
                                        )
                                    c_row.store(offset=jj, val=acc)

                                vectorize[NELTS](jj_limit, pf_fma_tail_rem)
                                i += 1
                            continue

                        # ---- Full NR-panel: process all i-panels ----
                        i = 0
                        ip = 0
                        while i + MR <= m:
                            var ap_panel = ap_worker + ip * MR * kc

                            # Prefetch C rows with write intent into L1
                            comptime for mr in range(MR):
                                prefetch[PrefetchOptions().for_write().high_locality().to_data_cache()](
                                    c_ptr + (i + mr) * n + j0 + jr
                                )

                            var acc = InlineArray[SIMD[dtype, NELTS], MR * NR_VECS](
                                fill=SIMD[dtype, NELTS](0)
                            )
                            # Load C accumulators (skip for first k-tile — already zero)
                            if not is_first_k:
                                comptime for mr in range(MR):
                                    comptime for nr in range(NR_VECS):
                                        acc[mr * NR_VECS + nr] = (
                                            c_ptr + (i + mr) * n + j0 + jr
                                        ).load[width=NELTS](offset=nr * NELTS)

                            # K-loop with KU unrolling, reading from packed A
                            var pk = 0
                            var pk_end = kc - (kc % KU)
                            while pk < pk_end:
                                comptime for ku in range(KU):
                                    var bp_k = bp_panel + (pk + ku) * NR
                                    prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                                        bp_panel + (pk + ku + PREFETCH_DIST) * NR
                                    )
                                    var ap_k = ap_panel + (pk + ku) * MR
                                    # Prefetch packed A ahead
                                    prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                                        ap_panel + (pk + ku + PREFETCH_DIST) * MR
                                    )
                                    var a_vals = InlineArray[Scalar[dtype], MR](
                                        fill=Scalar[dtype](0)
                                    )
                                    comptime for mr in range(MR):
                                        a_vals[mr] = ap_k[mr]
                                    comptime for nr in range(NR_VECS):
                                        var bv = bp_k.load[width=NELTS](
                                            offset=nr * NELTS
                                        )
                                        comptime for mr in range(MR):
                                            acc[mr * NR_VECS + nr] = fma(
                                                SIMD[dtype, NELTS](a_vals[mr]), bv, acc[mr * NR_VECS + nr]
                                            )
                                pk += KU

                            # K remainder
                            while pk < kc:
                                var bp_k = bp_panel + pk * NR
                                var a_vals = InlineArray[Scalar[dtype], MR](
                                    fill=Scalar[dtype](0)
                                )
                                var ap_k = ap_panel + pk * MR
                                comptime for mr in range(MR):
                                    a_vals[mr] = ap_k[mr]
                                comptime for nr in range(NR_VECS):
                                    var bv = bp_k.load[width=NELTS](
                                        offset=nr * NELTS
                                    )
                                    comptime for mr in range(MR):
                                        acc[mr * NR_VECS + nr] = fma(
                                            SIMD[dtype, NELTS](a_vals[mr]), bv, acc[mr * NR_VECS + nr]
                                        )
                                pk += 1

                            # Store accumulators back to C
                            comptime for mr in range(MR):
                                comptime for nr in range(NR_VECS):
                                    (c_ptr + (i + mr) * n + j0 + jr).store(
                                        offset=nr * NELTS, val=acc[mr * NR_VECS + nr],
                                    )

                            i += MR
                            ip += 1

                        # Handle remaining rows (< MR)
                        while i < m:
                            i_h = i
                            var jj_limit = min(NR, tile_n - jr)
                            c_row = c_ptr + i * n + j0 + jr

                            def pf_fma_tail[width: Int](jj: Int) {mut c_row, mut a_ptr, read bp_panel, read is_first_k, read kc_h, read pc_h, read i_h, read k}:
                                var acc = c_row.load[width=width](offset=jj)
                                if is_first_k:
                                    acc = SIMD[dtype, width](0)
                                for ppk in range(kc_h):
                                    acc = fma(
                                        SIMD[dtype, width](a_ptr[i_h * k + pc_h + ppk]),
                                        (bp_panel + ppk * NR).load[width=width](offset=jj),
                                        acc,
                                    )
                                c_row.store(offset=jj, val=acc)

                            vectorize[NELTS](jj_limit, pf_fma_tail)
                            i += 1

            jt += NC_TILES

    parallelize(process_worker, num_workers, num_workers)
    bp_buf.free()
    ap_buf.free()


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


def matmul_prefill[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  optimized for prefill shapes (M >= MR).
    # Worker-based parallelism with A-panel packing amortized across j-tile batches.
    # NR=3*NELTS improves compute intensity (6.0 vs 5.33 FLOP/byte), reducing memory
    # pressure. KC=512 halves k-tile count for less B packing + C traffic.
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 8
    comptime NR = 3 * NELTS   # 24 for float64: higher compute intensity (8×24)
    comptime KC = 512
    comptime KU = 8
    comptime TILE_N = 72      # 3 × NR = 3 full NR-panels per tile
    comptime NC_TILES = 256

    if a.rows < MR:
        _zero_fill[dtype](c)
        _goto_gemv[dtype](c, a, b)
    else:
        _prefill_gemm[dtype, MR, NR, KC, KU, TILE_N, NC_TILES](c, a, b)


def matmul_prefill_opt[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Optimized prefill kernel using v3 microkernel.
    # Tuned by empirical scan on Xeon Skylake AVX-512 (4 cores) for the
    # 96×11008×2048 prefill shape:
    #   MR=6, NR=4*NELTS=32 → 24 accumulators, 96/6=16 i-panels.
    #   KU=2 keeps the unrolled FMA body small (fits L1i) while still letting
    #     LLVM schedule cross-iteration. KU=4 and KU=8 are 5-8% slower.
    #   KC=256 picks the standard BLIS L2-resident B panel.
    #   TILE_N=64 ⇒ 172 j-tiles = exactly 43 per worker on 4 cores. The old
    #     TILE_N=128 gave 86 tiles split 22/22/22/20, idling one core ~9%
    #     of the time; the even split is worth ~2% peak GFLOPS.
    # Wins ~6% peak GFLOPS vs the old MR=8 NR=24 config on Skylake AVX-512.
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 6
    comptime NR = 4 * NELTS
    comptime KC = 256
    comptime KU = 2
    comptime TILE_N = 64
    comptime NC_TILES = 64

    if a.rows < MR:
        _zero_fill[dtype](c)
        _goto_gemv[dtype](c, a, b)
    else:
        _prefill_gemm_v3[dtype, MR, NR, KC, KU, TILE_N, NC_TILES](c, a, b)


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


def matmul_decode[
    dtype: DType, //,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  optimized for decode shapes (small M).
    # Uses j-parallel GEMV: each worker owns a column chunk that fits L1.
    _decode_gemv(c, a, b)


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


def matmul_dispatch[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  dispatches to the fastest kernel for the shape.
    #
    # Tuned by empirical scan across both the Qwen 2.5 VL 3B MLP projection
    # sweep (M = 1..512, both orientations) and a general (M,N,K) sweep
    # (square / wide-N / tall-K), most recently on the 2.10 GHz Xeon:
    #
    #   - M*N*K < 2^19 (tiny total work): a serial register-blocked kernel
    #     (_matmul_small). The parallel kernels' fixed overhead (thread launch +
    #     per-worker packing buffers) swamps a handful-of-FLOPs matmul, so they
    #     run 10-100x slower than a plain serial loop here. Flips sq1..sq64 from
    #     a 0.02-0.81 LOSE to a 1.1-3.4 WIN; cannot fire for any headline shape.
    #   - M == 1 (pure decode GEMV): j-parallel GEMV with L1-resident column
    #     chunks + software prefetch. Streams B exactly once.
    #   - 2 <= M <= 5 (small-batch decode): the v3 GEMM micro-kernel with
    #     MR = M, so the packed B panel is streamed once and reused across all
    #     M rows. The old path sent these to the GEMV, which re-streamed all of
    #     B once *per row* — ~2x slower at M=4. MR=M removes the remainder loop.
    #   - Small box, M-dominant, B fits L2 (M>=64, M>=N, B=K*N*8 L2-resident):
    #     the M-parallel no-pack _thin_n_gemm. The packed prefill kernel's
    #     packing + thread-launch overhead dwarfs the compute on a cache-resident
    #     box, so the worst general shapes lived here (sq96/sq128 0.65-0.71).
    #     Reading A/B unpacked flips them to 1.0-1.16. The L2-fit test is two
    #     tiered (compile-time 512 KB + L2-adaptive B<=L2/3); the route admits
    #     only B that fits ~1/3 of L2 (up to sq288 on the 2 MB Xeon), past which
    #     B spills mid-M-sweep and the packed path wins. See the branch comment.
    #   - M >= 6, N <= 192 (narrow N): the kernel parallelizes only over N
    #     (j-tiles), so a narrow N starves the cores (N=64 -> 1 j-tile at the
    #     default TILE_N=64 -> 1 of 4 cores busy). A narrow NR=16/TILE_N=16 tile
    #     splits N into >= num_workers j-tiles, recovering the small-square shapes
    #     (sq64 0.44->0.83). See the branch comment below.
    #   - M >= 6, 192 < N <= M (square-ish): a 6x32 tile with TILE_N=32 (4*NELTS)
    #     and a small fixed KC=512. The wide-N/tall-K branches below carry TILE_N=64
    #     + cache-aware big-KC picks that fit a box-shaped C badly; the narrower
    #     tile + small KC keep the M*KC packed-A L2-resident and double the j-tile
    #     count. Flips the square-ish band (sq512 0.74->0.88, 512x256x512 0.51->
    #     0.73) on the Skylake 2.80 GHz VM. N <= M never holds for a headline shape.
    #   - M >= 6, N > 192 (prefill GEMM): a 6x32 register tile (TILE_N=8*NELTS=64,
    #     32 even j-tiles on N=2048), KC stepping 256 (small M) -> 512 -> a
    #     cache-aware large KC. The register-blocked masked M-remainder tail (see
    #     _prefill_gemm_v3) removes the scalar tax that used to force MR-divides-M
    #     tiles, so the N-balanced 6x32 wins at almost every M. Two exceptions,
    #     each validated by interleaved A/B vs linalg (peak/20+):
    #       * wide-N (N >= K) small-M corner: very wide N (>= 9*1024) AND M <= 32
    #         keeps the older 8x24 tile (KU=2, TILE_N=9*NELTS), which still edges
    #         6x32 by ~2-4% there (the Qwen up-proj small batch).
    #       * square-ish large M (N <= M): cap the cache-aware KC at 1024 — a
    #         single huge k-panel thrashes the M*KC packed-A when N is too narrow
    #         to amortize it (sq2048 KC2048 0.87 -> KC1024 0.93).
    #     NB: memory-layout levers (2-D MC*KC blocking, 2-D (M,N) worker split,
    #     shared single-pack-of-A) were all measured wash-to-loss on this 33 MB-
    #     L3 part: A fits in L3, so "redundant" per-worker A re-packs are cheap
    #     L3 traffic and these shapes are compute-bound. The only residual losses
    #     (up-proj M>=256, down-proj M=256) are micro-kernel maturity vs linalg's
    #     hand-tuned AVX-512 kernel. (Memory levers may help where A exceeds L3.)
    comptime NELTS = simd_width_of[dtype]()
    var m = a.rows
    var n = c.cols
    var k = a.cols

    # Tiny total work: a serial register-blocked kernel. Below ~2^19 MAC ops
    # the parallel prefill kernel's fixed overhead (parallelize launch +
    # per-worker bp/ap buffer allocs + A/B packing) dominates the compute and
    # it runs an order of magnitude slower than a plain serial loop — e.g. on
    # the 2.10 GHz Xeon (4c) sq8/16/32 ran 0.03-0.17x linalg through the
    # parallel path vs 1.2-2.6x serial, and the gain stays positive out to
    # ~sq80 (wMNK=512000: serial 1.14x the parallel path) before the 4 cores
    # win again at sq88 (wMNK=681472). The 2^19 cut sits in that gap. This can
    # never fire for a headline/large shape (decode 1x11008x2048 and every
    # prefill batch have wMNK in the tens of millions). See _matmul_small.
    if m * n * k < (1 << 19):
        _matmul_small[dtype, 6, 2](c, a, b)
    elif m == 1:
        _decode_gemv(c, a, b)
    elif m == 2:
        _prefill_gemm_v3[dtype, 2, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 3:
        _prefill_gemm_v3[dtype, 3, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 4:
        _prefill_gemm_v3[dtype, 4, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 5:
        _prefill_gemm_v3[dtype, 5, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif n <= NELTS * 8 and m >= 64:
        # Thin-N, tall-M (N <= 64 on f64, the work is along M not N). The prefill
        # kernel parallelizes only over N (j-tiles), so a thin N starves the box
        # even with the narrow NR=16 tile below: at N=16 -> 1 j-tile, and the
        # remaining cores idle while one streams + packs all of B. These shapes
        # ran the worst ratios in the whole sweep — 8192x16x512 0.10, 2048x16x2048
        # 0.14, 4096x32x1024 0.18, 128x16x512 0.28 vs linalg — the deferred
        # "M-parallel thin-N kernel" follow-up. _thin_n_gemm parallelizes over
        # M-row blocks instead: every core owns a band of C's rows and sweeps the
        # full (tiny) N, reading A/B straight from source with no packing (a thin
        # N stays cache-resident, so packing buys nothing). Measured on the
        # 2.10 GHz Xeon (4c) it lifts the entire band from a 0.10-0.58 LOSE to
        # 0.80-1.27 — e.g. 8192x16x512 0.10->0.98, 2048x16x2048 0.14->0.89,
        # 128x16x512 0.28->1.27, 512x32x512 0.41->1.05 WIN, 64x32x2048 0.50->1.27
        # WIN. Above N=64 a large-K B no longer fits cache without packing and the
        # narrow packed path wins (2048x128x2048 dropped to 0.38 unpacked), so the
        # route caps at N=64. NR_VECS=1 (NR=NELTS) for N < 2*NELTS so a sub-16-wide
        # N still fills a full SIMD panel instead of falling to the scalar tail
        # (512x8x512 0.05->0.96); NR_VECS=2 (NR=16) otherwise. The N <= 64 cap and
        # the m >= 64 floor (mid-M is where it wins biggest) keep this clear of the
        # Qwen MLP shapes (N=2048/11008) entirely.
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
        # Small box, M-dominant, B fits L2 (the small-square gap). These are the
        # WORST shapes in the general sweep before this branch: square GEMMs in
        # the sq96..sq256 band ran 0.65-0.79 vs linalg, and tall small boxes
        # (e.g. 512x128x512) bottomed at 0.56 — all routed to the packed prefill
        # kernel (small-N or square-ish), whose A/B packing + per-worker buffer
        # alloc + parallelize-launch overhead dwarfs the compute when the whole
        # problem is cache-resident. The M-parallel _thin_n_gemm reads A/B
        # straight from source with NO packing (same idea that fixed the thin-N
        # band), so it pays none of that fixed cost; every core owns a band of
        # C's rows and sweeps the full N. Interleaved A/B vs linalg (peak/40) on
        # the Skylake 2.80 GHz VM (4c, AVX-512, 1 MB/core L2): sq96 0.70->1.18,
        # sq128 0.74->1.00, sq192 0.80->0.99, sq256 0.75->0.83, 512x128x512
        # 0.56->0.84, 256x128x512 0.63->0.90, 512x256x256 0.80->0.84.
        #
        # The three gates are exactly the win boundary, measured:
        #   * B (k*n elements) fits L2: _thin_n re-reads all of B per MR-row
        #     block, so B must STAY L2-resident across the M-sweep (with room for
        #     the A micro-panel and C). The fit is checked in two tiers:
        #       - A COMPILE-TIME 512 KB first tier (short-circuits before any
        #         probe) catches the genuinely tiny cache-resident boxes without
        #         touching l2_cache_size(): that probe runs ~6 cpuid instructions,
        #         which on a virtualized host (KVM) trap to the hypervisor at
        #         ~61 us/call — ruinous on a few-us op (it sank sq96 to 0.19 when
        #         the gate queried it live). 512 KB = a quarter of this 2 MB-L2
        #         part / half the 1 MB Skylake.
        #       - An L2-ADAPTIVE second tier, B <= L2/3 (_box_l2_budget),
        #         extends the route a little past 512 KB on a larger L2. It is only
        #         reached when the 512 KB tier already failed (B > 512 KB), so the
        #         newly-eligible boxes are tens-of-millions-of-MAC ops where the
        #         one-time memoized cpuid is amortized to noise. L2/3 is the
        #         re-measured crossover: B must stay L2-resident *alongside* the
        #         packed-A micro-panel + C + prefetch headroom across the whole
        #         M-sweep, which holds only to ~1/3 L2. On this 2 MB Xeon it admits
        #         sq288 (B=648KB, no-pack ~0.92-1.00 > packed) and EXCLUDES sq320
        #         (B=800KB, no-pack ~0.64-0.71 < packed ~0.82-0.86), sq352 (0.56 <
        #         0.95), sq384 (0.42 < 0.90), 640x256x512 (0.47 < 0.88) and the
        #         rest of the mid-square/tall-box band — all now packed. (The prior
        #         (2*L2)/3 = 1.35MB cut sent those to no-pack, where they were the
        #         worst losses in the sweep; their old no-pack "wins" were measured
        #         on an older nightly whose linalg was slower.) On the 1 MB Skylake
        #         L2/3 = 341 KB < the 512 KB tier-1, so that part keeps no-pack only
        #         for B <= 512 KB. Hardware-specific like every tile/KC pick here.
        #   * m >= n: _thin_n parallelizes over M, so it needs enough M-rows
        #     relative to the N each worker sweeps. At m < n it ties or loses
        #     (128x256x256 0.76 vs the 0.79 packed path) — and, crucially, this
        #     gate makes the branch UNREACHABLE for every wide/headline shape
        #     (Qwen up/down proj, all wide-N grid shapes are n >> m), where
        #     _thin_n is catastrophic (96x11008x2048 0.15, K128 0.37).
        #   * m >= 64: enough MR=6 row-blocks (>= ~10) to fill the 4 workers.
        _thin_n_gemm[dtype, 6, 4](c, a, b)
    elif n <= 3 * 64:
        # Small-N (N <= 192). The default TILE_N=64 splits such an N into fewer
        # than 4 j-tiles (N=64 -> 1, N=128 -> 2), and since the kernel only
        # parallelizes over j-tiles that idles most of a 4-core box — the single
        # biggest general-shape loss (square N=K<=128 ran 0.4-0.5x vs linalg).
        # A narrow NR=16 / TILE_N=16 tile splits N into >= num_workers j-tiles
        # (N=64 -> 4, N=96 -> 6, N=128 -> 8), filling the cores; measured on the
        # 2.10 GHz Xeon (4c) it recovers sq64 0.64->0.82, sq96 0.54->0.74,
        # sq128 0.52->0.66. N >= 256 already yields >= 4 tiles at TILE_N=64 and
        # keeps the wider tile below. (Cannot affect the Qwen MLP shapes, whose
        # N is 2048 or 11008 — this branch only fires for narrow N.)
        _prefill_gemm_v3[dtype, 6, 2 * NELTS, 256, 4, 2 * NELTS, 64](c, a, b)
    elif n <= m:
        # Square-ish (N <= M, N > 192 so the small-N branch above didn't fire).
        # NB: the small-box branch above now intercepts the cache-resident corner
        # (M >= N and B = K*N*8 fitting L2 — a compile-time 512 KB tier plus an
        # L2-adaptive B<=L2/3 tier, ~682 KB on this 2 MB-L2 part, i.e. up to sq288)
        # into the no-pack M-parallel kernel, so this branch handles only the
        # LARGER square-ish shapes (B above that cut: sq512/1024/2048 and the
        # awkward-N boxes) whose B can't stay L2-resident unpacked. The sq256
        # figures below predate that branch — they record how this packed branch
        # itself was tuned, not the current sq256 route.
        # Both the wide-N (N>=K) and tall-K (N<K) large-M branches below were
        # tuned on the two Qwen aspect ratios (N=2048/11008, always N >> M), and
        # carry their TILE_N=64 / cache-aware-KC picks into these square-ish
        # shapes where they fit badly: the box-shaped C is small, so the big KC
        # the cache-aware rule grows (KC=1024/2048 on this L2) inflates the M*KC
        # packed-A panel without buying any C-traffic saving, and TILE_N=64
        # leaves only a few coarse j-tiles to parallelize over (N=512 -> 8). On
        # the Skylake 2.80 GHz VM (4c, AVX-512, 1 MB/core L2) these were the
        # worst losses in the general sweep — square M=N=K ran 0.58-0.79x linalg
        # and the narrow-N-tall-M corner (e.g. 512x256x512) bottomed at 0.51x.
        #
        # A narrower TILE_N=32 (4*NELTS) + KC fixes both: it doubles the j-tile
        # count (N=512 -> 16 even tiles, 4/4/4/4) and keeps the packed-B tile
        # (32xKC) and packed-A panel (M*KC) L2-resident. Interleaved A/B vs linalg
        # (peak/30) on the small-L2 Skylake 2.80 GHz VM (1 MB/core) flipped the
        # whole band up with KC=512:
        #   sq256 0.72->0.73, sq512 0.73->0.88, sq1024 0.70->0.82, sq2048 0.70->0.85,
        #   512x256x512 0.51->0.73, 512x384x2048 0.57->0.78, 300x256x256 0.58->0.88.
        # The N<=M gate keeps this clear of every headline/wide shape: the Qwen
        # up/down proj and all wide-N grid shapes have N > M (N=2048..11008,
        # M<=512), so they never reach here and keep their tuned branches below.
        #
        # KC + TILE_N retune. KC is cache-size-dependent (see the file-level
        # note) and, for square-ish, the two target machines reach OPPOSITE picks
        # a single constant can't satisfy — so KC is now chosen by detected L2
        # (see _square_ish_kc): KC=512 on the 1 MB/core Skylake, KC=1024 on the
        # 2 MB/core Xeon. Measured per machine (interleaved A/B):
        #   * Skylake 1 MB (peak/25): KC512 beats KC1024 — sq1024 0.84 vs 0.80,
        #     sq2048 0.84 vs 0.72 (a single big k-panel over-grows the M*KC
        #     packed-A on the small L2). A hardcoded KC=1024 sank sq2048 to ~0.72.
        #   * Xeon 2 MB (peak/80, our-kernel A/B): KC1024 beats KC512 — sq512
        #     +0.8%, sq768 +6.5%, sq1024 +4.4%, sq1536 +3.1%, sq2048 +4.6% (the
        #     bigger L2 fits the 64xKC tile + M*KC packed-A, fewer k-panels mean
        #     C is written fewer times). _square_ish_kc probes L2 only for the
        #     large k>512 shapes where the choice bites and the cpuid is < 2.5%.
        # TILE_N then splits on load balance — the repo's #1 win lever, and HW-
        # agnostic. The fatter TILE_N=64 (fewer, fatter j-tiles -> less packing/
        # loop overhead) is best ONLY when N tiles evenly across the workers; when
        # ceildiv(N,64) is not a multiple of num_workers it idles a core and
        # collapses (512x384x2048 TN64 0.63, vs TN32 0.73). So use TN64 when N
        # gives an even, >=2-per-worker tiling (the M=N power-of-2 squares
        # 512..2048 all hit it) and the finer TN32 otherwise (the awkward-N
        # square-ish shapes), each at its measured best. sq256 no longer reaches
        # here (small-box branch above); were it to, k<=512 keeps KC=512 / TN32.
        comptime TN_WIDE = 8 * NELTS  # 64 on f64: fatter j-tile, even split only
        comptime TN_FINE = 4 * NELTS  # 32 on f64: finer j-tile, robust balance
        var njt_wide = ceildiv(n, TN_WIDE)
        var num_workers = num_physical_cores()
        var use_wide = njt_wide % num_workers == 0 and njt_wide // num_workers >= 2
        # SHARED_A=True: pack the full A once instead of per-worker. On these
        # square-ish shapes A is as large as B/C, so the default per-worker
        # 4x A re-pack is a real cost (unlike the wide/tall headline shapes,
        # where A is small and SHARED_A measured a wash).
        if _square_ish_kc(m, n, k) >= 1024:
            if use_wide:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 1024, 4, TN_WIDE, 64, True](c, a, b)
            else:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 1024, 4, TN_FINE, 64, True](c, a, b)
        else:
            if use_wide:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, TN_WIDE, 64, True](c, a, b)
            else:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, TN_FINE, 64, True](c, a, b)
    elif n >= k:
        # wide-N (up-proj-like). The small-M band uses the N-balanced 6x32 tile
        # (TILE_N = 2*NR = 8*NELTS = 64), same as the large-M band below.
        #
        # Large-M retune Jun 13 2026 on the 2.10 GHz Xeon (4c, AVX-512, L2 2MB/
        # core, L3 260MB) — a DIFFERENT machine from the Skylake 2.80 GHz VM the
        # old 8x24/KC512 picks were tuned on, with 2x the L2. Interleaved A/B vs
        # linalg, peak over 20 runs:
        #   * 6x32 beats 8x24 by a wide margin for every M>192 here (e.g. M=256
        #     8x24-KC512 0.93 -> 6x32-KC512 1.02; the higher MR of 8x24 no longer
        #     pays once N=11008's per-row SIMD width matters more than i-panel
        #     count). So the whole large-M band is now 6x32, flipping M=256 from
        #     LOSE to WIN.
        #   * KC: with the bigger L2 the C-traffic win from fewer k-panels now
        #     dominates. KC=512 is best up to M~288 (M=256 1.02, M=288 1.045);
        #     above that a single k-panel (KC=2048=K, no K split -> C written
        #     once) wins (M=384 0.999->1.046, M=512 0.953->1.038), flipping both
        #     of the old large-M losses to WINs. (On the small-L2 Skylake VM the
        #     opposite held — KC=512 beat KC=1024 there; this pick is HW-specific.)
        #
        # Small-M band: 6x32 instead of the old uniform 8x24 (KU=2, TILE_N=72).
        # The 8x24 pick was tuned only on the Qwen up-proj (N=11008); on
        # moderate-width wide-N it loses badly. Direct interleaved A/B 8x24 vs
        # 6x32 (peak/25) across N widths at K=2048 found a clean crossover:
        #   * N <= 4096: 6x32 wins at every M=8..128 (up to +15%, e.g. N=2048
        #     M=128 6/8=1.15; N=4096 M=32 1.05).
        #   * N = 8192: 6x32 wins M>=16, a tie at M=8.
        #   * N = 11008 (the Qwen up-proj): 8x24 still edges 6x32 by ~2-4% at
        #     M<=32 (M=8 1.34 vs 1.27, M=16 1.19 vs 1.14, M=32 1.14 vs 1.12),
        #     but 6x32 retakes it from M>=64 (and the headline prefill M=96
        #     already preferred 6x32).
        # So 8x24 only genuinely wins in one corner — very wide N AND M<=32 —
        # which is exactly the Qwen up-proj small-batch shape. Keep 8x24 there
        # (no headline regression) and use the N-balanced 6x32 everywhere else,
        # which flips the whole moderate-/square-N small-M band from LOSE to WIN.
        if n >= 9 * 1024 and m <= 32:
            _prefill_gemm_v3[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS, 64](c, a, b)
        elif m <= 192:
            # SHARED_A large-M gate. By default every worker re-packs the full A
            # (M*KC per k-panel), so on a wide-N shape A is re-packed num_workers
            # times. The README long held that to be "a wash on wide/tall
            # headline shapes (N >> M)" — but that was reasoned at the headline
            # M=96, where A is tiny next to the N-sweep. Once M grows, the 4x
            # A re-pack is a real cost. Interleaved A/B (peak/30) on the 2.10 GHz
            # Xeon found a clean crossover at M~192: M<=128 a wash (M=96 +0.6%,
            # so NO headline regression), M>=192 a clean +3% (e.g. up-proj M=256
            # 0.976->1.006, M=512 0.962->0.993; flips LOSE->WIN). Gate at m>=192
            # so the headline prefill (M=96) and the small-M band keep the exact
            # per-worker path. SHARED_A is bit-identical (same FMAs, A packed
            # once instead of num_workers times); verify_dispatch confirms it.
            if m >= 192:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 256, 4, 8 * NELTS, 64, True](c, a, b)
            else:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 256, 4, 8 * NELTS, 64](c, a, b)
        elif m <= 288:
            # M in 193..288: past the SHARED_A crossover (see above), so pack A
            # once. +3% (up-proj M=256 0.976->1.006).
            _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, 8 * NELTS, 64, True](c, a, b)
        else:
            # Cache-aware KC: size the resident packed-B tile (TILE_N x KC) to
            # half of the detected per-core L2 so the packed-A panel and C keep
            # the other half. This is KC=1024 on a 1 MB/core L2 and KC=2048 on a
            # 2 MB/core L2; a KC large enough to fill the whole L2 thrashes A/C.
            var kc = _l2_resident_kc[dtype](64, k)
            # Square-ish large-M guard: the big-KC pick assumes a wide N whose
            # large C amortizes the resulting big packed-A panel (M*KC). When N
            # is not wider than M (square-ish), C is small, so a single huge
            # k-panel buys almost no C-traffic saving while its M*KC packed-A
            # thrashes cache. Measured on the 2.10 GHz Xeon (interleaved A/B vs
            # linalg, peak/12): sq2048 (M=N=K=2048) KC=2048 0.86 -> KC=1024 0.93,
            # whereas a genuinely wide M=2048 N=8192 shape still prefers KC=2048.
            # Gating on n <= m caps only square-ish M>288 shapes and can never
            # touch a Qwen/headline shape (all M<=512, all N wider than M).
            if n <= m:
                kc = min(kc, 1024)
            # M>288: always past the SHARED_A crossover (see m<=192 branch).
            # up-proj M=512 0.962->0.993, h4k-m512 0.989->1.040 (peak/30).
            if kc >= 2048:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 2048, 4, 8 * NELTS, 64, True](c, a, b)
            elif kc >= 1024:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 1024, 4, 8 * NELTS, 64, True](c, a, b)
            else:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, 8 * NELTS, 64, True](c, a, b)
    else:
        # tall-K (down-proj-like): uniform 6x32 tile, TILE_N = 2*NR = 8*NELTS =
        # 64, KC=256 (M<=64) / 512 (M>64). TILE_N=64 splits N=2048 into 32 even
        # j-tiles (8/8/8/8 across 4 workers) — perfect load balance, zero
        # N-remainder. The M-remainder (MR=6 rarely divides M) is absorbed by
        # the register-blocked masked tail in _prefill_gemm_v3 (one K-sweep,
        # r×NR_VECS accumulators reusing the packed B), so the balanced tiling
        # no longer pays a scalar-remainder tax. Re-measured Jun 13 2026
        # (interleaved A/B vs linalg, peak/20): with that fix the balanced 6x32
        # beats 8x24 at every M — mid-M (M=8..64) flips from the noise floor to
        # clear WINS (1.01-1.28x), and large M improves to 0.92-0.97 (M=256
        # 0.88 -> 0.94). This replaces the old 8x24/6x32 three-band split.
        # Large-M KC bump (Jun 13 2026, 2.10 GHz Xeon, L2 2MB/core): for M>256
        # a larger KC=2048 (vs 512 -> ceil(11008/2048)=6 k-panels instead of 22)
        # cuts the per-panel B re-pack / re-read overhead and wins on the heavy
        # GEMMs — interleaved A/B vs linalg, peak/20: M=384 0.957->0.975, M=512
        # 0.936->0.988. Pushing KC further over-grows the packed panels and
        # collapses (KC=5504 0.75, full-K 0.45), and M<=256 keeps KC512 (M=128
        # needs it: KC512 1.10 vs KC1024 1.06; M=256 is a wash). KC256 stays for
        # the M<=64 band where the smaller packed panels help.
        if m <= 64:
            _prefill_gemm_v3[dtype, 6, 4 * NELTS, 256, 4, 8 * NELTS, 64](c, a, b)
        elif m <= 256:
            # SHARED_A large-M gate (see wide-N branch for the rationale). The
            # tall-K crossover is the same M~192: M<=128 a wash-to-slight-loss
            # INCLUDING the down-proj headline M=96 (-1%), M>=192 a +1-4% win
            # (down-proj M=256 0.956->0.966, M=512 0.931->0.965; peak/30). Gate
            # at m>=192 so the down-proj headline keeps the per-worker path.
            if m >= 192:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, 8 * NELTS, 64, True](c, a, b)
            else:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, 8 * NELTS, 64](c, a, b)
        else:
            # Cache-aware KC (see wide-N branch): half-L2 resident packed-B
            # tile, i.e. KC=1024 on a 1 MB/core L2, KC=2048 on a 2 MB/core L2.
            # M>256: always past the SHARED_A crossover, so pack A once.
            var kc = _l2_resident_kc[dtype](64, k)
            if kc >= 2048:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 2048, 4, 8 * NELTS, 64, True](c, a, b)
            elif kc >= 1024:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 1024, 4, 8 * NELTS, 64, True](c, a, b)
            else:
                _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, 8 * NELTS, 64, True](c, a, b)


# Default matmul points to the tiled version
def matmul[dtype: DType = DType.float64](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype](c, a, b)
