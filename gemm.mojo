from matrix import Matrix
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.math import ceildiv, fma
from std.memory import memset_zero
from std.memory.unsafe_pointer import alloc
from std.sys import num_physical_cores, simd_width_of
from std.sys.info import CompilationTarget
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


def _prefill_gemm_v3[
    dtype: DType, MR: Int, NR: Int, KC: Int, KU: Int, TILE_N: Int,
    NC_TILES: Int,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Prefill GEMM with explicit B-load hoisting in the microkernel.
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

        var jt = j_tile_start
        while jt < j_tile_end:
            var jt_batch_end = min(jt + NC_TILES, j_tile_end)

            for pc in range(0, k, KC):
                var kc = min(KC, k - pc)
                is_first_k = (pc == 0)
                kc_h = kc
                pc_h = pc

                # Pack A: KC outer, MR inner so each pk gives MR contiguous doubles
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
                            var jj_limit = tile_n - jr
                            i = 0
                            while i + MR <= m:
                                for ii in range(i, i + MR):
                                    c_row = c_ptr + ii * n + j0 + jr
                                    a_row = a_ptr + ii * k + pc

                                    def v3_fma_remainder[width: Int](jj: Int) {mut c_row, read a_row, read bp_panel, read is_first_k, read kc_h}:
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

                                    vectorize[NELTS](jj_limit, v3_fma_remainder)
                                i += MR
                            while i < m:
                                i_h = i
                                c_row = c_ptr + i * n + j0 + jr

                                def v3_fma_tail_rem[width: Int](jj: Int) {mut c_row, mut a_ptr, read bp_panel, read is_first_k, read kc_h, read pc_h, read i_h, read k}:
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

                                vectorize[NELTS](jj_limit, v3_fma_tail_rem)
                                i += 1
                            continue

                        # ---- Full NR-panel: hoisted-load microkernel ----
                        i = 0
                        ip = 0
                        while i + MR <= m:
                            var ap_panel = ap_worker + ip * MR * kc

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

                                    # Load B once per ku-step (NR_VECS SIMD loads)
                                    var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                                        fill=SIMD[dtype, NELTS](0)
                                    )
                                    comptime for nr in range(NR_VECS):
                                        bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS)

                                    # Broadcast each A scalar and FMA into NR_VECS accumulators
                                    comptime for mr in range(MR):
                                        var a_bc = SIMD[dtype, NELTS](ap_k[mr])
                                        comptime for nr in range(NR_VECS):
                                            acc[mr * NR_VECS + nr] = a_bc.fma(
                                                bv[nr], acc[mr * NR_VECS + nr]
                                            )
                                pk += KU

                            while pk < kc:
                                var bp_k = bp_panel + pk * NR
                                var ap_k = ap_panel + pk * MR

                                var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                                    fill=SIMD[dtype, NELTS](0)
                                )
                                comptime for nr in range(NR_VECS):
                                    bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS)

                                comptime for mr in range(MR):
                                    var a_bc = SIMD[dtype, NELTS](ap_k[mr])
                                    comptime for nr in range(NR_VECS):
                                        acc[mr * NR_VECS + nr] = a_bc.fma(
                                            bv[nr], acc[mr * NR_VECS + nr]
                                        )
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
                        # partial-tile case already `continue`d above.) Replaces
                        # the old row-by-row `vectorize` tail, which swept K once
                        # per row with only NR_VECS-deep ILP — too shallow to hide
                        # FMA latency, the tax that made MR-not-dividing-M tiles
                        # (e.g. 6x32 at M=256) lose to remainder-free ones.
                        if i < m:
                            var r = m - i
                            var racc = InlineArray[SIMD[dtype, NELTS], MR * NR_VECS](
                                fill=SIMD[dtype, NELTS](0)
                            )
                            if not is_first_k:
                                comptime for mr in range(MR):
                                    if mr < r:
                                        comptime for nr in range(NR_VECS):
                                            racc[mr * NR_VECS + nr] = (
                                                c_ptr + (i + mr) * n + j0 + jr
                                            ).load[width=NELTS](offset=nr * NELTS)
                            for pk in range(kc):
                                var bp_k = bp_panel + pk * NR
                                var bv = InlineArray[SIMD[dtype, NELTS], NR_VECS](
                                    fill=SIMD[dtype, NELTS](0)
                                )
                                comptime for nr in range(NR_VECS):
                                    bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS)
                                comptime for mr in range(MR):
                                    if mr < r:
                                        var a_bc = SIMD[dtype, NELTS](
                                            a_ptr[(i + mr) * k + pc + pk]
                                        )
                                        comptime for nr in range(NR_VECS):
                                            racc[mr * NR_VECS + nr] = a_bc.fma(
                                                bv[nr], racc[mr * NR_VECS + nr]
                                            )
                            comptime for mr in range(MR):
                                if mr < r:
                                    comptime for nr in range(NR_VECS):
                                        (c_ptr + (i + mr) * n + j0 + jr).store(
                                            offset=nr * NELTS,
                                            val=racc[mr * NR_VECS + nr],
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


def _host_is_small_l2() -> Bool:
    # True for host microarchitectures with a small (~1 MB) per-core L2, where
    # the heavy-GEMM (large-M) bands want smaller packed panels.
    #
    # Resolved at COMPILE TIME: `CompilationTarget._arch()` is the target uarch
    # the compiler is generating code for — the same mechanism that makes
    # `simd_width_of` return 8 on AVX-512 — so this folds to a constant with
    # zero runtime cost (no cache probe, no runtime branch). There is no
    # comptime cache-SIZE API, only the uarch name, so we enumerate the known
    # small-L2 part(s) and default everything else to the big-L2 profile (most
    # modern server parts have >=2 MB L2/core).
    #
    # This is correct precisely because this repo compiles on the machine it
    # runs on (`mojo bench_*.mojo`). An AOT binary built on one box and shipped
    # to a different one would bake in the BUILD host's uarch and would instead
    # need a runtime L2 probe (getconf LEVEL2_CACHE_SIZE / sysfs).
    comptime arch = CompilationTarget._arch()
    comptime if arch == "skylake-avx512":
        return True
    else:
        return False


def _large_m_kc() -> Int:
    # KC for the heavy-GEMM (large-M) bands, uarch-keyed at comptime via the L2
    # size. The optimum is L2-dependent and flips between machines:
    #   * Skylake-AVX512 (1 MB L2/core): KC=512. Bigger packed panels spill the
    #     1 MB L2, so the per-panel C-traffic a larger KC saves is outweighed
    #     (measured: KC=512 beat KC=1024 even for the 6x32 tile there).
    #   * Emerald/Granite/Sapphire Rapids, Zen, etc. (>=2 MB L2/core): KC=2048.
    #     The bigger L2 holds the packed B panel, so fewer k-panels (less B
    #     re-pack/re-read + less C re-traffic) wins — e.g. on the 2.10 GHz
    #     Emerald Rapids VM this flipped up/down-proj M=256/512 from LOSE to
    #     WIN/parity (see README "Large-M retune on the 2.10 GHz Xeon").
    comptime if _host_is_small_l2():
        return 512
    else:
        return 2048


def matmul_dispatch[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  dispatches to the fastest kernel for the shape.
    #
    # Tuned by empirical scan on Xeon Skylake AVX-512 (4 cores) across the
    # Qwen 2.5 VL 3B MLP projection sweep (M = 1..512, both orientations):
    #
    #   - M == 1 (pure decode GEMV): j-parallel GEMV with L1-resident column
    #     chunks + software prefetch. Streams B exactly once.
    #   - 2 <= M <= 5 (small-batch decode): the v3 GEMM micro-kernel with
    #     MR = M, so the packed B panel is streamed once and reused across all
    #     M rows. The old path sent these to the GEMV, which re-streamed all of
    #     B once *per row* — ~2x slower at M=4. MR=M removes the remainder loop.
    #   - M >= 6 (prefill GEMM): one 8x24 register tile per aspect ratio, with
    #     KC stepping 256 (small M) -> 512 (large M). Re-tuned Jun 2026 by
    #     interleaved A/B vs linalg (peak over 20 runs) on this Skylake AVX-512
    #     VM, which collapsed the old per-M tile zoo (4x48 / 6x32 / KC=1024)
    #     into a single tile shape per orientation:
    #       * wide-N (N >= K, e.g. gate/up proj): 8x24, KU=2, TILE_N=9*NELTS.
    #           - M <= 192: KC=256.   - M >  192: KC=512.
    #         8x24 KC=512 beat the old 4x48 (M=256, ~+4%) and 6x32-KC1024
    #         (M=512, ~+3%); MR=8 leaves no scalar M-remainder at M=256/512.
    #       * tall-K (N < K, e.g. down proj): 8x24 KU=4 TILE_N=6*NELTS=48 at
    #         the ends, 6x32 KC=512 in the middle band:
    #           - uniform 6x32, TILE_N=8*NELTS=64 (32 even j-tiles on N=2048),
    #             KC=256 (M<=64) / 512 (M>64). The register-blocked masked
    #             M-remainder tail (see _prefill_gemm_v3) removes the scalar tax
    #             that used to make MR=6 lose, so the perfectly N-balanced tile
    #             wins at every M (mid-M 1.01-1.28x, large M 0.92-0.97).
    #     NB: memory-layout levers (2-D MC*KC blocking, 2-D (M,N) worker split,
    #     shared single-pack-of-A) were all measured wash-to-loss on this 33 MB-
    #     L3 part: A fits in L3, so "redundant" per-worker A re-packs are cheap
    #     L3 traffic and these shapes are compute-bound. The only residual losses
    #     (up-proj M>=256, down-proj M=256) are micro-kernel maturity vs linalg's
    #     hand-tuned AVX-512 kernel. (Memory levers may help where A exceeds L3.)
    comptime NELTS = simd_width_of[dtype]()
    # Heavy-GEMM (large-M) tuning, chosen at COMPILE TIME from the host uarch's
    # L2 size (see _host_is_small_l2 / _large_m_kc). Both fold to constants —
    # no runtime branch, no cache probe. On the small-L2 Skylake VM these
    # restore that machine's documented tuning (8x24 wide-N tile, KC=512); on
    # >=2 MB-L2 parts they select the 6x32 / KC=2048 large-M retune.
    comptime SMALL_L2 = _host_is_small_l2()
    comptime KC_BIG = _large_m_kc()
    var m = a.rows
    var n = c.cols
    var k = a.cols

    if m == 1:
        _decode_gemv(c, a, b)
    elif m == 2:
        _prefill_gemm_v3[dtype, 2, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 3:
        _prefill_gemm_v3[dtype, 3, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 4:
        _prefill_gemm_v3[dtype, 4, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif m == 5:
        _prefill_gemm_v3[dtype, 5, 4 * NELTS, 256, 2, 8 * NELTS, 64](c, a, b)
    elif n >= k:
        # wide-N (up-proj-like). The small-M band (M<=192) keeps the 8x24 tile
        # (KU=2, TILE_N = 3*NR = 9*NELTS) that wins decisively everywhere. The
        # large-M band is uarch-keyed at comptime (SMALL_L2):
        #
        #   * big-L2 parts (>=2 MB/core, e.g. the 2.10 GHz Emerald Rapids VM):
        #     6x32 (TILE_N = 2*NR = 8*NELTS = 64) throughout M>192. Re-measured
        #     Jun 13 2026, interleaved A/B vs linalg, peak/20: 6x32 beats 8x24 by
        #     a wide margin at every M>192 here (M=256: 8x24-KC512 0.93 ->
        #     6x32-KC512 1.02 — N=11008's wide per-row SIMD outweighs 8x24's
        #     higher i-panel count), flipping M=256 from LOSE to WIN. With the
        #     bigger L2 the C-traffic win from fewer k-panels dominates: KC=512
        #     is best to M~288 (M=256 1.02, M=288 1.045), then KC_BIG=2048 (one
        #     k-panel over K=2048, C written once) wins (M=384 0.999->1.046,
        #     M=512 0.953->1.038).
        #   * small-L2 Skylake-AVX512 (1 MB/core): its documented tuning — 8x24
        #     KC=512 for 192<M<=256 (8x24 won there on that box), 6x32 for
        #     M>256 with KC_BIG=512 (the bigger panels spill its 1 MB L2, so
        #     KC=512 beat KC=1024 even for 6x32). Crossover stays at 256.
        if m <= 192:
            _prefill_gemm_v3[dtype, 8, 3 * NELTS, 256, 2, 9 * NELTS, 64](c, a, b)
        else:
            comptime if SMALL_L2:
                if m <= 256:
                    _prefill_gemm_v3[dtype, 8, 3 * NELTS, 512, 2, 9 * NELTS, 64](c, a, b)
                else:
                    _prefill_gemm_v3[dtype, 6, 4 * NELTS, KC_BIG, 4, 8 * NELTS, 64](c, a, b)
            else:
                if m <= 288:
                    _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, 8 * NELTS, 64](c, a, b)
                else:
                    _prefill_gemm_v3[dtype, 6, 4 * NELTS, KC_BIG, 4, 8 * NELTS, 64](c, a, b)
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
        # the M<=64 band where the smaller packed panels help. The M>256 KC is
        # KC_BIG (comptime, uarch-keyed): 2048 on this 2 MB-L2 part, 512 on the
        # 1 MB-L2 Skylake VM where the bigger panels spill.
        if m <= 64:
            _prefill_gemm_v3[dtype, 6, 4 * NELTS, 256, 4, 8 * NELTS, 64](c, a, b)
        elif m <= 256:
            _prefill_gemm_v3[dtype, 6, 4 * NELTS, 512, 4, 8 * NELTS, 64](c, a, b)
        else:
            _prefill_gemm_v3[dtype, 6, 4 * NELTS, KC_BIG, 4, 8 * NELTS, 64](c, a, b)


# Default matmul points to the tiled version
def matmul[dtype: DType = DType.float64](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype](c, a, b)
