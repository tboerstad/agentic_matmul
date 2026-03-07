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


fn matmul_blis[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Wide micro-kernel GEMM:
    #   Key improvement over matmul_packed: processes 2 SIMD vectors of C
    #   simultaneously with MR=6 rows, yielding 12 register accumulators.
    #   Each B vector pair is reused across 6 rows (vs 4 in packed), and
    #   each A scalar drives 2 FMAs (vs 1). This nearly doubles the
    #   compute-to-load ratio: 12 FMAs per 8 loads vs 4 FMAs per 5 loads.
    #   TILE_I=24 gives perfect 4-core utilization (96/24=4 tiles).
    #   24/6=4 MR blocks per tile with zero remainder rows.
    comptime TILE_I = 24
    comptime TILE = 32
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 6          # rows of C per micro-kernel invocation
    comptime NR_VECS = 2     # SIMD vectors wide
    comptime NR = NR_VECS * NELTS

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    # Zero out C
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

    var num_i_tiles = (m + TILE_I - 1) // TILE_I

    fn process_i_tile(tile_idx: Int) capturing:
        var i0 = tile_idx * TILE_I
        var i_end = i0 + TILE_I
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
                    for ii in range(i0, i_end):
                        for p in range(p0, p_end):
                            var a_val = a_ptr[ii * k + p]
                            for j in range(j0, j_end):
                                var idx = ii * n + j
                                c_ptr[idx] = c_ptr[idx] + a_val * b_ptr[j * k + p]
                else:
                    # MR=6 × NR=2*NELTS wide micro-kernel with register accumulators
                    var i = i0
                    while i + MR <= i_end:
                        var j = 0
                        while j + NR <= tile_n:
                            var c0 = c_ptr + i * n + j0 + j
                            var c1 = c_ptr + (i + 1) * n + j0 + j
                            var c2 = c_ptr + (i + 2) * n + j0 + j
                            var c3 = c_ptr + (i + 3) * n + j0 + j
                            var c4 = c_ptr + (i + 4) * n + j0 + j
                            var c5 = c_ptr + (i + 5) * n + j0 + j

                            # Load 12 SIMD accumulators (6 rows × 2 vectors)
                            var acc00 = c0.load[width=NELTS](offset=0)
                            var acc01 = c0.load[width=NELTS](offset=NELTS)
                            var acc10 = c1.load[width=NELTS](offset=0)
                            var acc11 = c1.load[width=NELTS](offset=NELTS)
                            var acc20 = c2.load[width=NELTS](offset=0)
                            var acc21 = c2.load[width=NELTS](offset=NELTS)
                            var acc30 = c3.load[width=NELTS](offset=0)
                            var acc31 = c3.load[width=NELTS](offset=NELTS)
                            var acc40 = c4.load[width=NELTS](offset=0)
                            var acc41 = c4.load[width=NELTS](offset=NELTS)
                            var acc50 = c5.load[width=NELTS](offset=0)
                            var acc51 = c5.load[width=NELTS](offset=NELTS)

                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_row = b_ptr + p * n + j0 + j
                                var b0 = b_row.load[width=NELTS](offset=0)
                                var b1 = b_row.load[width=NELTS](offset=NELTS)

                                var a0 = a_ptr[i * k + p]
                                var a1 = a_ptr[(i + 1) * k + p]
                                var a2 = a_ptr[(i + 2) * k + p]
                                var a3 = a_ptr[(i + 3) * k + p]
                                var a4 = a_ptr[(i + 4) * k + p]
                                var a5 = a_ptr[(i + 5) * k + p]

                                acc00 += a0 * b0
                                acc01 += a0 * b1
                                acc10 += a1 * b0
                                acc11 += a1 * b1
                                acc20 += a2 * b0
                                acc21 += a2 * b1
                                acc30 += a3 * b0
                                acc31 += a3 * b1
                                acc40 += a4 * b0
                                acc41 += a4 * b1
                                acc50 += a5 * b0
                                acc51 += a5 * b1

                            c0.store(offset=0, val=acc00)
                            c0.store(offset=NELTS, val=acc01)
                            c1.store(offset=0, val=acc10)
                            c1.store(offset=NELTS, val=acc11)
                            c2.store(offset=0, val=acc20)
                            c2.store(offset=NELTS, val=acc21)
                            c3.store(offset=0, val=acc30)
                            c3.store(offset=NELTS, val=acc31)
                            c4.store(offset=0, val=acc40)
                            c4.store(offset=NELTS, val=acc41)
                            c5.store(offset=0, val=acc50)
                            c5.store(offset=NELTS, val=acc51)
                            j += NR

                        # Remainder j with single SIMD vector, still MR rows
                        while j + NELTS <= tile_n:
                            var c0 = c_ptr + i * n + j0 + j
                            var c1 = c_ptr + (i + 1) * n + j0 + j
                            var c2 = c_ptr + (i + 2) * n + j0 + j
                            var c3 = c_ptr + (i + 3) * n + j0 + j
                            var c4 = c_ptr + (i + 4) * n + j0 + j
                            var c5 = c_ptr + (i + 5) * n + j0 + j
                            var acc0 = c0.load[width=NELTS]()
                            var acc1 = c1.load[width=NELTS]()
                            var acc2 = c2.load[width=NELTS]()
                            var acc3 = c3.load[width=NELTS]()
                            var acc4 = c4.load[width=NELTS]()
                            var acc5 = c5.load[width=NELTS]()
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_vec = (b_ptr + p * n + j0 + j).load[width=NELTS]()
                                acc0 += a_ptr[i * k + p] * b_vec
                                acc1 += a_ptr[(i + 1) * k + p] * b_vec
                                acc2 += a_ptr[(i + 2) * k + p] * b_vec
                                acc3 += a_ptr[(i + 3) * k + p] * b_vec
                                acc4 += a_ptr[(i + 4) * k + p] * b_vec
                                acc5 += a_ptr[(i + 5) * k + p] * b_vec
                            c0.store(val=acc0)
                            c1.store(val=acc1)
                            c2.store(val=acc2)
                            c3.store(val=acc3)
                            c4.store(val=acc4)
                            c5.store(val=acc5)
                            j += NELTS

                        while j < tile_n:
                            for ri in range(MR):
                                var acc = c_ptr[(i + ri) * n + j0 + j]
                                for pk in range(tile_k):
                                    var p = p0 + pk
                                    acc += a_ptr[(i + ri) * k + p] * b_ptr[p * n + j0 + j]
                                c_ptr[(i + ri) * n + j0 + j] = acc
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
