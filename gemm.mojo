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


fn matmul_v7[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  2D register-blocked micro-kernel with k-loop
    #                             unrolling, large K-tiles, SIMD, parallelism,
    #                             and register accumulation.
    #
    # Key optimizations over matmul_packed:
    #   1. 2D register blocking (MR=4 rows × NR=2 SIMD vectors = 4×16 doubles):
    #      Each B vector is reused across 4 rows, each A scalar across 2 column
    #      vectors.
    #   2. K-loop unrolling by 4: pre-loads 4 B vector pairs per unrolled
    #      iteration, giving the CPU's out-of-order engine more independent
    #      loads to overlap with FMA computation. Hides memory latency for
    #      B's stride-N access pattern.
    #   3. Larger K-tile (KC=256 vs 32): 8× more FMAs per C accumulator
    #      load/store, reducing C-side memory traffic by 8×.
    #   4. MC=24 i-tile: gives exactly 4 tiles for M=96 (perfect load balance
    #      on 4 cores) vs MC=32 which gives 3 tiles (one core idle).
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 4        # rows per micro-kernel
    comptime NR = 2        # SIMD vectors per micro-kernel column
    comptime NR_COLS = NR * NELTS  # actual columns per micro-kernel (16)
    comptime KC = 256      # k-tile
    comptime MC = 24       # i-tile (96/24 = 4 tiles for 4 cores)
    comptime KU = 4        # k-loop unroll factor

    var m = a.rows
    var n = c.cols
    var k = a.cols

    comptime if transpose_b:
        matmul_packed[dtype, transpose_b=True](c, a, b)
        return

    # For very small M (decode), fall back to packed kernel
    if m < MR:
        matmul_packed[dtype, transpose_b=False](c, a, b)
        return

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    # Zero out C
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

    var num_i_tiles = (m + MC - 1) // MC

    fn process_i_tile(tile_idx: Int) capturing:
        var i0 = tile_idx * MC
        var i_end = i0 + MC
        if i_end > m:
            i_end = m

        for p0 in range(0, k, KC):
            var p_end = p0 + KC
            if p_end > k:
                p_end = k
            var tile_k = p_end - p0

            # --- 2D register-blocked path (NR_COLS columns at a time) ---
            var j0 = 0
            while j0 + NR_COLS <= n:
                var i = i0
                while i + MR <= i_end:
                    var c_r0 = c_ptr + i * n + j0
                    var c_r1 = c_ptr + (i + 1) * n + j0
                    var c_r2 = c_ptr + (i + 2) * n + j0
                    var c_r3 = c_ptr + (i + 3) * n + j0

                    # Load MR × NR = 8 accumulator registers from C
                    var acc00 = c_r0.load[width=NELTS](offset=0)
                    var acc01 = c_r0.load[width=NELTS](offset=NELTS)
                    var acc10 = c_r1.load[width=NELTS](offset=0)
                    var acc11 = c_r1.load[width=NELTS](offset=NELTS)
                    var acc20 = c_r2.load[width=NELTS](offset=0)
                    var acc21 = c_r2.load[width=NELTS](offset=NELTS)
                    var acc30 = c_r3.load[width=NELTS](offset=0)
                    var acc31 = c_r3.load[width=NELTS](offset=NELTS)

                    # K-unrolled inner loop: process KU=4 k-values per iteration
                    var pk = 0
                    var tile_k_aligned = tile_k - (tile_k % KU)
                    while pk < tile_k_aligned:
                        var p = p0 + pk

                        # Unrolled k-step 0
                        var bb0 = b_ptr + p * n + j0
                        var b0_0 = bb0.load[width=NELTS](offset=0)
                        var b0_1 = bb0.load[width=NELTS](offset=NELTS)

                        var a00 = a_ptr[i * k + p]
                        var a01 = a_ptr[(i + 1) * k + p]
                        var a02 = a_ptr[(i + 2) * k + p]
                        var a03 = a_ptr[(i + 3) * k + p]

                        acc00 += a00 * b0_0
                        acc01 += a00 * b0_1
                        acc10 += a01 * b0_0
                        acc11 += a01 * b0_1
                        acc20 += a02 * b0_0
                        acc21 += a02 * b0_1
                        acc30 += a03 * b0_0
                        acc31 += a03 * b0_1

                        # Unrolled k-step 1
                        var bb1 = b_ptr + (p + 1) * n + j0
                        var b1_0 = bb1.load[width=NELTS](offset=0)
                        var b1_1 = bb1.load[width=NELTS](offset=NELTS)

                        var a10 = a_ptr[i * k + p + 1]
                        var a11 = a_ptr[(i + 1) * k + p + 1]
                        var a12 = a_ptr[(i + 2) * k + p + 1]
                        var a13 = a_ptr[(i + 3) * k + p + 1]

                        acc00 += a10 * b1_0
                        acc01 += a10 * b1_1
                        acc10 += a11 * b1_0
                        acc11 += a11 * b1_1
                        acc20 += a12 * b1_0
                        acc21 += a12 * b1_1
                        acc30 += a13 * b1_0
                        acc31 += a13 * b1_1

                        # Unrolled k-step 2
                        var bb2 = b_ptr + (p + 2) * n + j0
                        var b2_0 = bb2.load[width=NELTS](offset=0)
                        var b2_1 = bb2.load[width=NELTS](offset=NELTS)

                        var a20 = a_ptr[i * k + p + 2]
                        var a21 = a_ptr[(i + 1) * k + p + 2]
                        var a22 = a_ptr[(i + 2) * k + p + 2]
                        var a23 = a_ptr[(i + 3) * k + p + 2]

                        acc00 += a20 * b2_0
                        acc01 += a20 * b2_1
                        acc10 += a21 * b2_0
                        acc11 += a21 * b2_1
                        acc20 += a22 * b2_0
                        acc21 += a22 * b2_1
                        acc30 += a23 * b2_0
                        acc31 += a23 * b2_1

                        # Unrolled k-step 3
                        var bb3 = b_ptr + (p + 3) * n + j0
                        var b3_0 = bb3.load[width=NELTS](offset=0)
                        var b3_1 = bb3.load[width=NELTS](offset=NELTS)

                        var a30 = a_ptr[i * k + p + 3]
                        var a31 = a_ptr[(i + 1) * k + p + 3]
                        var a32 = a_ptr[(i + 2) * k + p + 3]
                        var a33 = a_ptr[(i + 3) * k + p + 3]

                        acc00 += a30 * b3_0
                        acc01 += a30 * b3_1
                        acc10 += a31 * b3_0
                        acc11 += a31 * b3_1
                        acc20 += a32 * b3_0
                        acc21 += a32 * b3_1
                        acc30 += a33 * b3_0
                        acc31 += a33 * b3_1

                        pk += KU

                    # Handle remaining k-values (< KU)
                    while pk < tile_k:
                        var p = p0 + pk
                        var bb = b_ptr + p * n + j0
                        var b0 = bb.load[width=NELTS](offset=0)
                        var b1 = bb.load[width=NELTS](offset=NELTS)

                        var a0 = a_ptr[i * k + p]
                        acc00 += a0 * b0
                        acc01 += a0 * b1
                        var a1 = a_ptr[(i + 1) * k + p]
                        acc10 += a1 * b0
                        acc11 += a1 * b1
                        var a2 = a_ptr[(i + 2) * k + p]
                        acc20 += a2 * b0
                        acc21 += a2 * b1
                        var a3 = a_ptr[(i + 3) * k + p]
                        acc30 += a3 * b0
                        acc31 += a3 * b1
                        pk += 1

                    # Store accumulators back to C (once per k-tile)
                    c_r0.store(offset=0, val=acc00)
                    c_r0.store(offset=NELTS, val=acc01)
                    c_r1.store(offset=0, val=acc10)
                    c_r1.store(offset=NELTS, val=acc11)
                    c_r2.store(offset=0, val=acc20)
                    c_r2.store(offset=NELTS, val=acc21)
                    c_r3.store(offset=0, val=acc30)
                    c_r3.store(offset=NELTS, val=acc31)
                    i += MR

                # Remaining rows (< MR): single-row accumulation
                while i < i_end:
                    var c_row = c_ptr + i * n + j0
                    var acc0 = c_row.load[width=NELTS](offset=0)
                    var acc1 = c_row.load[width=NELTS](offset=NELTS)
                    for pk in range(tile_k):
                        var p = p0 + pk
                        var bb = b_ptr + p * n + j0
                        var b0 = bb.load[width=NELTS](offset=0)
                        var b1 = bb.load[width=NELTS](offset=NELTS)
                        var a_val = a_ptr[i * k + p]
                        acc0 += a_val * b0
                        acc1 += a_val * b1
                    c_row.store(offset=0, val=acc0)
                    c_row.store(offset=NELTS, val=acc1)
                    i += 1
                j0 += NR_COLS

            # --- Remaining columns (< NR_COLS): single-vector fallback ---
            while j0 + NELTS <= n:
                var i = i0
                while i < i_end:
                    var c_row = c_ptr + i * n + j0
                    var acc = c_row.load[width=NELTS](offset=0)
                    for pk in range(tile_k):
                        var p = p0 + pk
                        var b_vec = (b_ptr + p * n + j0).load[width=NELTS](
                            offset=0
                        )
                        acc += a_ptr[i * k + p] * b_vec
                    c_row.store(offset=0, val=acc)
                    i += 1
                j0 += NELTS

            # --- Scalar remainder ---
            while j0 < n:
                var i = i0
                while i < i_end:
                    var acc = c_ptr[i * n + j0]
                    for pk in range(tile_k):
                        var p = p0 + pk
                        acc += a_ptr[i * k + p] * b_ptr[p * n + j0]
                    c_ptr[i * n + j0] = acc
                    i += 1
                j0 += 1

    parallelize[process_i_tile](num_i_tiles, num_physical_cores())


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
