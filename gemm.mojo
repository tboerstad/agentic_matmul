from matrix import Matrix
from std.algorithm.functional import parallelize, vectorize
from std.sys import num_physical_cores, simd_width_of


fn matmul_naive[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * op(B)  —  simple triple-nested loop (ijk order).
    var m = a.rows
    var n = c.cols
    var k = a.cols
    for i in range(m):
        for j in range(n):
            var dot = Scalar[dtype](0)
            for p in range(k):
                var a_val = a[i, p]

                comptime if transpose_b:
                    dot += a_val * b[j, p]
                else:
                    dot += a_val * b[p, j]

            c[i, j] = dot


fn matmul_tiled[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * op(B)  —  tiled / cache-blocked version.
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

    # Zero out C (tiles accumulate with +=)
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

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
                            comptime if transpose_b:
                                c[i, j] = c[i, j] + a_val * b[j, p]
                            else:
                                c[i, j] = c[i, j] + a_val * b[p, j]


fn matmul_simd[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * op(B)  —  tiled + SIMD vectorized version.
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

    # Zero out C (tiles accumulate with +=)
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

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
                comptime if transpose_b:
                    for i in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a[i, p]
                            for j in range(j0, j_end):
                                c[i, j] = c[i, j] + a_val * b[j, p]
                else:
                    for i in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a[i, p]
                            var c_row = c_ptr + i * n + j0
                            var b_row = b_ptr + p * n + j0

                            fn fma[width: Int](j: Int) unified {mut}:
                                var c_vec = c_row.load[width=width](offset=j)
                                var b_vec = b_row.load[width=width](offset=j)
                                c_row.store(offset=j, val=c_vec + a_val * b_vec)

                            vectorize[NELTS](tile_n, fma)


fn matmul_parallel[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * op(B)  —  tiled + SIMD + multi-threaded version.
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

    # Zero out C (tiles accumulate with +=)
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

    # Number of row tiles
    var num_i_tiles = (m + TILE - 1) // TILE

    fn process_i_tile(tile_idx: Int) capturing:
        var i0 = tile_idx * TILE
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
                comptime if transpose_b:
                    for i in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a_ptr[i * k + p]
                            for j in range(j0, j_end):
                                var idx = i * n + j
                                c_ptr[idx] = c_ptr[idx] + a_val * b_ptr[j * k + p]
                else:
                    for i in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a_ptr[i * k + p]
                            var c_row = c_ptr + i * n + j0
                            var b_row = b_ptr + p * n + j0

                            fn fma[width: Int](j: Int) unified {mut}:
                                var c_vec = c_row.load[width=width](offset=j)
                                var b_vec = b_row.load[width=width](offset=j)
                                c_row.store(offset=j, val=c_vec + a_val * b_vec)

                            vectorize[NELTS](tile_n, fma)

    parallelize[process_i_tile](num_i_tiles, num_physical_cores())


fn matmul_register_blocked[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  tiled + SIMD + parallel + register-blocked.
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

    # Zero out C (tiles accumulate with +=)
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

    var num_i_tiles = (m + TILE - 1) // TILE

    fn process_i_tile(tile_idx: Int) capturing:
        var i0 = tile_idx * TILE
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

                comptime if transpose_b:
                    # transpose_b path: no SIMD (non-contiguous B access)
                    for i in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a_ptr[i * k + p]
                            for j in range(j0, j_end):
                                var idx = i * n + j
                                c_ptr[idx] = c_ptr[idx] + a_val * b_ptr[j * k + p]
                else:
                    # Register-blocked: process MR rows at a time
                    var i = i0
                    while i + MR <= i_end:
                        for p in range(p0, p_end):
                            var a0 = a_ptr[i * k + p]
                            var a1 = a_ptr[(i + 1) * k + p]
                            var a2 = a_ptr[(i + 2) * k + p]
                            var a3 = a_ptr[(i + 3) * k + p]
                            var b_row = b_ptr + p * n + j0
                            var c_row0 = c_ptr + i * n + j0
                            var c_row1 = c_ptr + (i + 1) * n + j0
                            var c_row2 = c_ptr + (i + 2) * n + j0
                            var c_row3 = c_ptr + (i + 3) * n + j0

                            fn fma_mr[width: Int](j: Int) unified {mut}:
                                var b_vec = b_row.load[width=width](offset=j)
                                c_row0.store(
                                    offset=j,
                                    val=c_row0.load[width=width](offset=j)
                                    + a0 * b_vec,
                                )
                                c_row1.store(
                                    offset=j,
                                    val=c_row1.load[width=width](offset=j)
                                    + a1 * b_vec,
                                )
                                c_row2.store(
                                    offset=j,
                                    val=c_row2.load[width=width](offset=j)
                                    + a2 * b_vec,
                                )
                                c_row3.store(
                                    offset=j,
                                    val=c_row3.load[width=width](offset=j)
                                    + a3 * b_vec,
                                )

                            vectorize[NELTS](tile_n, fma_mr)
                        i += MR

                    # Handle remaining rows (< MR) with single-row SIMD
                    while i < i_end:
                        for p in range(p0, p_end):
                            var a_val = a_ptr[i * k + p]
                            var c_row = c_ptr + i * n + j0
                            var b_row = b_ptr + p * n + j0

                            fn fma[width: Int](j: Int) unified {mut}:
                                var c_vec = c_row.load[width=width](offset=j)
                                var b_vec = b_row.load[width=width](offset=j)
                                c_row.store(
                                    offset=j, val=c_vec + a_val * b_vec
                                )

                            vectorize[NELTS](tile_n, fma)
                        i += 1

    parallelize[process_i_tile](num_i_tiles, num_physical_cores())


fn matmul_packed[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  tiled + SIMD + parallel + register-blocked
    #                             + C-accumulation in registers.
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

    # Zero out C (tiles accumulate with +=)
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

    var num_i_tiles = (m + TILE - 1) // TILE

    fn process_i_tile(tile_idx: Int) capturing:
        var i0 = tile_idx * TILE
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m

        for p0 in range(0, k, TILE):
            var p_end = p0 + TILE
            if p_end > k:
                p_end = k
            var tile_k = p_end - p0
            for j0 in range(0, n, TILE):
                var j_end = j0 + TILE
                if j_end > n:
                    j_end = n
                var tile_n = j_end - j0

                comptime if transpose_b:
                    # transpose_b path: no SIMD (non-contiguous B access)
                    for i in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a_ptr[i * k + p]
                            for j in range(j0, j_end):
                                var idx = i * n + j
                                c_ptr[idx] = c_ptr[idx] + a_val * b_ptr[j * k + p]
                else:
                    # Register-accumulation micro-kernel: j→p loop order
                    # For each j-block, load C into registers, accumulate
                    # across all k-values, then store back once.
                    var i = i0
                    while i + MR <= i_end:
                        # MR-row micro-kernel with register accumulators
                        var c_row0 = c_ptr + i * n + j0
                        var c_row1 = c_ptr + (i + 1) * n + j0
                        var c_row2 = c_ptr + (i + 2) * n + j0
                        var c_row3 = c_ptr + (i + 3) * n + j0

                        # Process NELTS columns at a time
                        var j = 0
                        while j + NELTS <= tile_n:
                            # Load C accumulators from memory (once per tile)
                            var acc0 = c_row0.load[width=NELTS](offset=j)
                            var acc1 = c_row1.load[width=NELTS](offset=j)
                            var acc2 = c_row2.load[width=NELTS](offset=j)
                            var acc3 = c_row3.load[width=NELTS](offset=j)

                            # Accumulate across entire k-tile in registers
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_vec = (b_ptr + p * n + j0).load[width=NELTS](offset=j)
                                acc0 += a_ptr[i * k + p] * b_vec
                                acc1 += a_ptr[(i + 1) * k + p] * b_vec
                                acc2 += a_ptr[(i + 2) * k + p] * b_vec
                                acc3 += a_ptr[(i + 3) * k + p] * b_vec

                            # Store accumulators back (once per tile)
                            c_row0.store(offset=j, val=acc0)
                            c_row1.store(offset=j, val=acc1)
                            c_row2.store(offset=j, val=acc2)
                            c_row3.store(offset=j, val=acc3)
                            j += NELTS

                        # Scalar remainder for j
                        while j < tile_n:
                            var acc0 = c_row0[j]
                            var acc1 = c_row1[j]
                            var acc2 = c_row2[j]
                            var acc3 = c_row3[j]
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_val = b_ptr[p * n + j0 + j]
                                acc0 += a_ptr[i * k + p] * b_val
                                acc1 += a_ptr[(i + 1) * k + p] * b_val
                                acc2 += a_ptr[(i + 2) * k + p] * b_val
                                acc3 += a_ptr[(i + 3) * k + p] * b_val
                            c_row0[j] = acc0
                            c_row1[j] = acc1
                            c_row2[j] = acc2
                            c_row3[j] = acc3
                            j += 1

                        i += MR

                    # Handle remaining rows (< MR) with single-row accumulation
                    while i < i_end:
                        var c_row = c_ptr + i * n + j0
                        var j = 0
                        while j + NELTS <= tile_n:
                            var acc = c_row.load[width=NELTS](offset=j)
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_vec = (b_ptr + p * n + j0).load[width=NELTS](offset=j)
                                acc += a_ptr[i * k + p] * b_vec
                            c_row.store(offset=j, val=acc)
                            j += NELTS
                        while j < tile_n:
                            var acc = c_row[j]
                            for pk in range(tile_k):
                                var p = p0 + pk
                                acc += a_ptr[i * k + p] * b_ptr[p * n + j0 + j]
                            c_row[j] = acc
                            j += 1
                        i += 1

    parallelize[process_i_tile](num_i_tiles, num_physical_cores())


fn matmul_unrolled[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  cache-aware GOTO-style GEMM.
    #
    # Complete redesign vs previous kernels.  Two fundamental changes:
    #
    #   1. Parallelize over j-tiles (N-dimension) instead of i-tiles (M-dimension).
    #      With M=96/TILE=32 we only got 3 i-tiles across 4 cores (one idle).
    #      With N=11008/TILE=64 we get 172 j-tiles — perfect load balance.
    #      For decode (M=1), this turns 1-thread into 4-thread.
    #
    #   2. j→k→i loop order (vs old k→j→i).  For each j-tile:
    #      - C panel = M×TILE_N×8 = 48KB — fits L1, loaded/stored ONCE total
    #      - B chunk per k-tile = TILE_K×TILE_N×8 = 128KB — fits L2
    #      - B chunk reused across all M/MR = 24 i-blocks
    #      Old order swept all 11008 B-columns per k-tile (22MB — blew caches)
    #      and C (8.5MB) couldn't stay cached between k-tiles.
    comptime TILE_N = 64    # j-tile: C panel = M*64*8 fits L1
    comptime TILE_K = 256   # k-tile: B chunk = 256*64*8 = 128KB fits L2
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 4         # rows of C per micro-kernel
    comptime NR = 2         # SIMD vectors of C columns per micro-kernel
    comptime KU = 4         # k-loop unroll factor

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    # Zero out C
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

    var num_j_tiles = (n + TILE_N - 1) // TILE_N

    fn process_j_tile(j_tile_idx: Int) capturing:
        var j0 = j_tile_idx * TILE_N
        var j_end = j0 + TILE_N
        if j_end > n:
            j_end = n
        var tile_n = j_end - j0

        # j→k→i order: C panel stays in L1 across all k-tiles
        for p0 in range(0, k, TILE_K):
            var p_end = p0 + TILE_K
            if p_end > k:
                p_end = k
            var tile_k = p_end - p0

            comptime if transpose_b:
                for i in range(m):
                    for p in range(p0, p_end):
                        var a_val = a_ptr[i * k + p]
                        for j in range(j0, j_end):
                            var idx = i * n + j
                            c_ptr[idx] = c_ptr[idx] + a_val * b_ptr[j * k + p]
            else:
                # Process all rows in MR-sized blocks
                var i = 0
                while i + MR <= m:
                    var c_row0 = c_ptr + i * n + j0
                    var c_row1 = c_ptr + (i + 1) * n + j0
                    var c_row2 = c_ptr + (i + 2) * n + j0
                    var c_row3 = c_ptr + (i + 3) * n + j0

                    # Process NR SIMD vectors at a time
                    var j = 0
                    while j + NR * NELTS <= tile_n:
                        # Load C accumulators (from L1 — C panel is warm)
                        var acc00 = c_row0.load[width=NELTS](offset=j)
                        var acc01 = c_row0.load[width=NELTS](offset=j + NELTS)
                        var acc10 = c_row1.load[width=NELTS](offset=j)
                        var acc11 = c_row1.load[width=NELTS](offset=j + NELTS)
                        var acc20 = c_row2.load[width=NELTS](offset=j)
                        var acc21 = c_row2.load[width=NELTS](offset=j + NELTS)
                        var acc30 = c_row3.load[width=NELTS](offset=j)
                        var acc31 = c_row3.load[width=NELTS](offset=j + NELTS)

                        # K-unrolled accumulation
                        var pk = 0
                        var pk_end = tile_k - (tile_k % KU)
                        while pk < pk_end:
                            var p = p0 + pk

                            # Unroll 0
                            var b_row_0 = b_ptr + p * n + j0
                            var bv00 = b_row_0.load[width=NELTS](offset=j)
                            var bv01 = b_row_0.load[width=NELTS](offset=j + NELTS)
                            var a0_0 = a_ptr[i * k + p]
                            var a1_0 = a_ptr[(i + 1) * k + p]
                            var a2_0 = a_ptr[(i + 2) * k + p]
                            var a3_0 = a_ptr[(i + 3) * k + p]
                            acc00 += a0_0 * bv00
                            acc01 += a0_0 * bv01
                            acc10 += a1_0 * bv00
                            acc11 += a1_0 * bv01
                            acc20 += a2_0 * bv00
                            acc21 += a2_0 * bv01
                            acc30 += a3_0 * bv00
                            acc31 += a3_0 * bv01

                            # Unroll 1
                            var b_row_1 = b_ptr + (p + 1) * n + j0
                            var bv10 = b_row_1.load[width=NELTS](offset=j)
                            var bv11 = b_row_1.load[width=NELTS](offset=j + NELTS)
                            var a0_1 = a_ptr[i * k + p + 1]
                            var a1_1 = a_ptr[(i + 1) * k + p + 1]
                            var a2_1 = a_ptr[(i + 2) * k + p + 1]
                            var a3_1 = a_ptr[(i + 3) * k + p + 1]
                            acc00 += a0_1 * bv10
                            acc01 += a0_1 * bv11
                            acc10 += a1_1 * bv10
                            acc11 += a1_1 * bv11
                            acc20 += a2_1 * bv10
                            acc21 += a2_1 * bv11
                            acc30 += a3_1 * bv10
                            acc31 += a3_1 * bv11

                            # Unroll 2
                            var b_row_2 = b_ptr + (p + 2) * n + j0
                            var bv20 = b_row_2.load[width=NELTS](offset=j)
                            var bv21 = b_row_2.load[width=NELTS](offset=j + NELTS)
                            var a0_2 = a_ptr[i * k + p + 2]
                            var a1_2 = a_ptr[(i + 1) * k + p + 2]
                            var a2_2 = a_ptr[(i + 2) * k + p + 2]
                            var a3_2 = a_ptr[(i + 3) * k + p + 2]
                            acc00 += a0_2 * bv20
                            acc01 += a0_2 * bv21
                            acc10 += a1_2 * bv20
                            acc11 += a1_2 * bv21
                            acc20 += a2_2 * bv20
                            acc21 += a2_2 * bv21
                            acc30 += a3_2 * bv20
                            acc31 += a3_2 * bv21

                            # Unroll 3
                            var b_row_3 = b_ptr + (p + 3) * n + j0
                            var bv30 = b_row_3.load[width=NELTS](offset=j)
                            var bv31 = b_row_3.load[width=NELTS](offset=j + NELTS)
                            var a0_3 = a_ptr[i * k + p + 3]
                            var a1_3 = a_ptr[(i + 1) * k + p + 3]
                            var a2_3 = a_ptr[(i + 2) * k + p + 3]
                            var a3_3 = a_ptr[(i + 3) * k + p + 3]
                            acc00 += a0_3 * bv30
                            acc01 += a0_3 * bv31
                            acc10 += a1_3 * bv30
                            acc11 += a1_3 * bv31
                            acc20 += a2_3 * bv30
                            acc21 += a2_3 * bv31
                            acc30 += a3_3 * bv30
                            acc31 += a3_3 * bv31

                            pk += KU

                        # Handle remaining k-values
                        while pk < tile_k:
                            var p = p0 + pk
                            var b_row = b_ptr + p * n + j0
                            var bv0 = b_row.load[width=NELTS](offset=j)
                            var bv1 = b_row.load[width=NELTS](offset=j + NELTS)
                            var a0 = a_ptr[i * k + p]
                            var a1 = a_ptr[(i + 1) * k + p]
                            var a2 = a_ptr[(i + 2) * k + p]
                            var a3 = a_ptr[(i + 3) * k + p]
                            acc00 += a0 * bv0
                            acc01 += a0 * bv1
                            acc10 += a1 * bv0
                            acc11 += a1 * bv1
                            acc20 += a2 * bv0
                            acc21 += a2 * bv1
                            acc30 += a3 * bv0
                            acc31 += a3 * bv1
                            pk += 1

                        # Store accumulators back
                        c_row0.store(offset=j, val=acc00)
                        c_row0.store(offset=j + NELTS, val=acc01)
                        c_row1.store(offset=j, val=acc10)
                        c_row1.store(offset=j + NELTS, val=acc11)
                        c_row2.store(offset=j, val=acc20)
                        c_row2.store(offset=j + NELTS, val=acc21)
                        c_row3.store(offset=j, val=acc30)
                        c_row3.store(offset=j + NELTS, val=acc31)
                        j += NR * NELTS

                    # Handle remaining columns with single-vector path
                    while j + NELTS <= tile_n:
                        var acc0 = c_row0.load[width=NELTS](offset=j)
                        var acc1 = c_row1.load[width=NELTS](offset=j)
                        var acc2 = c_row2.load[width=NELTS](offset=j)
                        var acc3 = c_row3.load[width=NELTS](offset=j)
                        for pk in range(tile_k):
                            var p = p0 + pk
                            var bv = (b_ptr + p * n + j0).load[width=NELTS](offset=j)
                            acc0 += a_ptr[i * k + p] * bv
                            acc1 += a_ptr[(i + 1) * k + p] * bv
                            acc2 += a_ptr[(i + 2) * k + p] * bv
                            acc3 += a_ptr[(i + 3) * k + p] * bv
                        c_row0.store(offset=j, val=acc0)
                        c_row1.store(offset=j, val=acc1)
                        c_row2.store(offset=j, val=acc2)
                        c_row3.store(offset=j, val=acc3)
                        j += NELTS

                    # Scalar remainder for j
                    while j < tile_n:
                        var acc0 = c_row0[j]
                        var acc1 = c_row1[j]
                        var acc2 = c_row2[j]
                        var acc3 = c_row3[j]
                        for pk in range(tile_k):
                            var p = p0 + pk
                            var b_val = b_ptr[p * n + j0 + j]
                            acc0 += a_ptr[i * k + p] * b_val
                            acc1 += a_ptr[(i + 1) * k + p] * b_val
                            acc2 += a_ptr[(i + 2) * k + p] * b_val
                            acc3 += a_ptr[(i + 3) * k + p] * b_val
                        c_row0[j] = acc0
                        c_row1[j] = acc1
                        c_row2[j] = acc2
                        c_row3[j] = acc3
                        j += 1

                    i += MR

                # Handle remaining rows (< MR)
                while i < m:
                    var c_row = c_ptr + i * n + j0
                    var j = 0
                    while j + NELTS <= tile_n:
                        var acc = c_row.load[width=NELTS](offset=j)
                        for pk in range(tile_k):
                            var p = p0 + pk
                            var bv = (b_ptr + p * n + j0).load[width=NELTS](offset=j)
                            acc += a_ptr[i * k + p] * bv
                        c_row.store(offset=j, val=acc)
                        j += NELTS
                    while j < tile_n:
                        var acc = c_row[j]
                        for pk in range(tile_k):
                            var p = p0 + pk
                            acc += a_ptr[i * k + p] * b_ptr[p * n + j0 + j]
                        c_row[j] = acc
                        j += 1
                    i += 1

    parallelize[process_j_tile](num_j_tiles, num_physical_cores())


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
