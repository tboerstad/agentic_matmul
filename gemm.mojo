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


fn matmul_nr_blocked[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  tiled + SIMD + parallel + MR×NR register tile.
    #
    # Key optimization over matmul_packed:
    #   NR blocking: the packed kernel uses MR=4 rows × 1 SIMD-width column
    #   (4 zmm accumulators, 5 total).  This version tiles NR=4 SIMD-width
    #   columns as well, giving an MR=4 × NR=4 register tile:
    #     - 16 zmm accumulators (C tile)
    #     - 4 zmm for B vectors (reused across MR rows)
    #     - 20 zmm total (fits comfortably in 32 AVX-512 registers)
    #   Each A[i,p] scalar is now reused across NR=4 B vectors instead of 1,
    #   cutting A-side memory traffic by 4×.  Each B vector is still reused
    #   across MR=4 rows.  The result is 16 FMAs per k-iteration with only
    #   4 A loads + 4 B loads = excellent compute-to-load ratio.
    comptime TILE = 32
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 4   # rows of C per micro-kernel
    comptime NR = 4   # SIMD-width columns of C per micro-kernel

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
                    for i in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a_ptr[i * k + p]
                            for j in range(j0, j_end):
                                var idx = i * n + j
                                c_ptr[idx] = c_ptr[idx] + a_val * b_ptr[j * k + p]
                else:
                    # MR×NR micro-kernel with register accumulation
                    var i = i0
                    while i + MR <= i_end:
                        var c_row0 = c_ptr + i * n + j0
                        var c_row1 = c_ptr + (i + 1) * n + j0
                        var c_row2 = c_ptr + (i + 2) * n + j0
                        var c_row3 = c_ptr + (i + 3) * n + j0

                        # Process NR SIMD-width columns at a time
                        var j = 0
                        while j + NR * NELTS <= tile_n:
                            # Load MR×NR C accumulators (16 zmm registers)
                            var acc00 = c_row0.load[width=NELTS](offset=j)
                            var acc01 = c_row0.load[width=NELTS](offset=j + NELTS)
                            var acc02 = c_row0.load[width=NELTS](offset=j + 2 * NELTS)
                            var acc03 = c_row0.load[width=NELTS](offset=j + 3 * NELTS)
                            var acc10 = c_row1.load[width=NELTS](offset=j)
                            var acc11 = c_row1.load[width=NELTS](offset=j + NELTS)
                            var acc12 = c_row1.load[width=NELTS](offset=j + 2 * NELTS)
                            var acc13 = c_row1.load[width=NELTS](offset=j + 3 * NELTS)
                            var acc20 = c_row2.load[width=NELTS](offset=j)
                            var acc21 = c_row2.load[width=NELTS](offset=j + NELTS)
                            var acc22 = c_row2.load[width=NELTS](offset=j + 2 * NELTS)
                            var acc23 = c_row2.load[width=NELTS](offset=j + 3 * NELTS)
                            var acc30 = c_row3.load[width=NELTS](offset=j)
                            var acc31 = c_row3.load[width=NELTS](offset=j + NELTS)
                            var acc32 = c_row3.load[width=NELTS](offset=j + 2 * NELTS)
                            var acc33 = c_row3.load[width=NELTS](offset=j + 3 * NELTS)

                            # Accumulate across entire k-tile in registers
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_base = b_ptr + p * n + j0
                                # Load NR B vectors (4 zmm regs, reused across MR rows)
                                var b0 = b_base.load[width=NELTS](offset=j)
                                var b1 = b_base.load[width=NELTS](offset=j + NELTS)
                                var b2 = b_base.load[width=NELTS](offset=j + 2 * NELTS)
                                var b3 = b_base.load[width=NELTS](offset=j + 3 * NELTS)
                                # Load MR A scalars (reused across NR columns)
                                var a0 = a_ptr[i * k + p]
                                var a1 = a_ptr[(i + 1) * k + p]
                                var a2 = a_ptr[(i + 2) * k + p]
                                var a3 = a_ptr[(i + 3) * k + p]
                                # 16 FMAs: MR rows × NR columns
                                acc00 += a0 * b0
                                acc01 += a0 * b1
                                acc02 += a0 * b2
                                acc03 += a0 * b3
                                acc10 += a1 * b0
                                acc11 += a1 * b1
                                acc12 += a1 * b2
                                acc13 += a1 * b3
                                acc20 += a2 * b0
                                acc21 += a2 * b1
                                acc22 += a2 * b2
                                acc23 += a2 * b3
                                acc30 += a3 * b0
                                acc31 += a3 * b1
                                acc32 += a3 * b2
                                acc33 += a3 * b3

                            # Store MR×NR accumulators back (once per tile)
                            c_row0.store(offset=j, val=acc00)
                            c_row0.store(offset=j + NELTS, val=acc01)
                            c_row0.store(offset=j + 2 * NELTS, val=acc02)
                            c_row0.store(offset=j + 3 * NELTS, val=acc03)
                            c_row1.store(offset=j, val=acc10)
                            c_row1.store(offset=j + NELTS, val=acc11)
                            c_row1.store(offset=j + 2 * NELTS, val=acc12)
                            c_row1.store(offset=j + 3 * NELTS, val=acc13)
                            c_row2.store(offset=j, val=acc20)
                            c_row2.store(offset=j + NELTS, val=acc21)
                            c_row2.store(offset=j + 2 * NELTS, val=acc22)
                            c_row2.store(offset=j + 3 * NELTS, val=acc23)
                            c_row3.store(offset=j, val=acc30)
                            c_row3.store(offset=j + NELTS, val=acc31)
                            c_row3.store(offset=j + 2 * NELTS, val=acc32)
                            c_row3.store(offset=j + 3 * NELTS, val=acc33)
                            j += NR * NELTS

                        # Remainder: single SIMD-width columns (same as packed)
                        while j + NELTS <= tile_n:
                            var acc0 = c_row0.load[width=NELTS](offset=j)
                            var acc1 = c_row1.load[width=NELTS](offset=j)
                            var acc2 = c_row2.load[width=NELTS](offset=j)
                            var acc3 = c_row3.load[width=NELTS](offset=j)
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_vec = (b_ptr + p * n + j0).load[width=NELTS](offset=j)
                                acc0 += a_ptr[i * k + p] * b_vec
                                acc1 += a_ptr[(i + 1) * k + p] * b_vec
                                acc2 += a_ptr[(i + 2) * k + p] * b_vec
                                acc3 += a_ptr[(i + 3) * k + p] * b_vec
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


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
