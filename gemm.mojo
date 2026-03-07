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


fn matmul_b_packed[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  tiled + SIMD + parallel + register-blocked
    #                             + C-accumulation + B-panel packing.
    #
    # Key optimization over matmul_packed:
    #   B-panel packing: the previous version accesses B with stride N (11008
    #   elements = 88 KB per row for float64).  This causes TLB misses, cache
    #   conflict misses, and defeats the hardware prefetcher.  This version
    #   copies each B panel [p0:p_end, j0:j_end] into a contiguous packed
    #   buffer before the micro-kernel, so inner-loop B access is purely
    #   sequential.  The packing cost is amortized across all MR-row blocks.
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

    # Packed B buffer: TILE * TILE elements per thread, laid out as
    # num_j_blocks * tile_k * NELTS so that for a given j-block the k
    # values are contiguous (perfect for the j→p micro-kernel order).
    var num_threads = num_physical_cores()
    var pack_buf = List[Scalar[dtype]](capacity=num_threads * TILE * TILE)
    for _ in range(num_threads * TILE * TILE):
        pack_buf.append(Scalar[dtype](0))
    var pack_ptr = pack_buf.unsafe_ptr()

    fn process_i_tile(tile_idx: Int) capturing:
        var i0 = tile_idx * TILE
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m

        # Each thread gets its own slice of pack_buf
        var thread_id = tile_idx % num_threads
        var my_pack = pack_ptr + thread_id * TILE * TILE

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
                    # Pack B panel [p0:p_end, j0:j_end] into contiguous buffer.
                    # Layout: for j-block jb, packed[jb * tile_k * NELTS + pk * NELTS + lane]
                    # = B[p0+pk, j0 + jb*NELTS + lane]
                    # This puts k as the outer dimension within each j-block so the
                    # micro-kernel reads B sequentially.
                    var num_j_blocks = (tile_n + NELTS - 1) // NELTS
                    for jb in range(num_j_blocks):
                        var j_base = jb * NELTS
                        var width = NELTS
                        if j_base + width > tile_n:
                            width = tile_n - j_base
                        for pk in range(tile_k):
                            var src = b_ptr + (p0 + pk) * n + j0 + j_base
                            var dst = my_pack + jb * tile_k * NELTS + pk * NELTS
                            # Copy NELTS elements (or fewer for the last block)
                            if width == NELTS:
                                dst.store(src.load[width=NELTS]())
                            else:
                                for lane in range(width):
                                    dst[lane] = src[lane]

                    # Register-accumulation micro-kernel with packed B
                    var i = i0
                    while i + MR <= i_end:
                        var c_row0 = c_ptr + i * n + j0
                        var c_row1 = c_ptr + (i + 1) * n + j0
                        var c_row2 = c_ptr + (i + 2) * n + j0
                        var c_row3 = c_ptr + (i + 3) * n + j0

                        # Process full NELTS j-blocks
                        for jb in range(num_j_blocks):
                            var j_base = jb * NELTS
                            var bp = my_pack + jb * tile_k * NELTS

                            if j_base + NELTS <= tile_n:
                                # Load C accumulators
                                var acc0 = c_row0.load[width=NELTS](offset=j_base)
                                var acc1 = c_row1.load[width=NELTS](offset=j_base)
                                var acc2 = c_row2.load[width=NELTS](offset=j_base)
                                var acc3 = c_row3.load[width=NELTS](offset=j_base)

                                # Accumulate across k-tile — B access is now sequential
                                for pk in range(tile_k):
                                    var p = p0 + pk
                                    var b_vec = (bp + pk * NELTS).load[width=NELTS]()
                                    acc0 += a_ptr[i * k + p] * b_vec
                                    acc1 += a_ptr[(i + 1) * k + p] * b_vec
                                    acc2 += a_ptr[(i + 2) * k + p] * b_vec
                                    acc3 += a_ptr[(i + 3) * k + p] * b_vec

                                # Store back
                                c_row0.store(offset=j_base, val=acc0)
                                c_row1.store(offset=j_base, val=acc1)
                                c_row2.store(offset=j_base, val=acc2)
                                c_row3.store(offset=j_base, val=acc3)
                            else:
                                # Scalar remainder for last j-block
                                for lane in range(tile_n - j_base):
                                    var acc0s = c_row0[j_base + lane]
                                    var acc1s = c_row1[j_base + lane]
                                    var acc2s = c_row2[j_base + lane]
                                    var acc3s = c_row3[j_base + lane]
                                    for pk in range(tile_k):
                                        var p = p0 + pk
                                        var b_val = (bp + pk * NELTS)[lane]
                                        acc0s += a_ptr[i * k + p] * b_val
                                        acc1s += a_ptr[(i + 1) * k + p] * b_val
                                        acc2s += a_ptr[(i + 2) * k + p] * b_val
                                        acc3s += a_ptr[(i + 3) * k + p] * b_val
                                    c_row0[j_base + lane] = acc0s
                                    c_row1[j_base + lane] = acc1s
                                    c_row2[j_base + lane] = acc2s
                                    c_row3[j_base + lane] = acc3s

                        i += MR

                    # Handle remaining rows (< MR) with single-row accumulation
                    while i < i_end:
                        var c_row = c_ptr + i * n + j0
                        for jb in range(num_j_blocks):
                            var j_base = jb * NELTS
                            var bp = my_pack + jb * tile_k * NELTS

                            if j_base + NELTS <= tile_n:
                                var acc = c_row.load[width=NELTS](offset=j_base)
                                for pk in range(tile_k):
                                    var p = p0 + pk
                                    var b_vec = (bp + pk * NELTS).load[width=NELTS]()
                                    acc += a_ptr[i * k + p] * b_vec
                                c_row.store(offset=j_base, val=acc)
                            else:
                                for lane in range(tile_n - j_base):
                                    var acc = c_row[j_base + lane]
                                    for pk in range(tile_k):
                                        var p = p0 + pk
                                        acc += a_ptr[i * k + p] * (bp + pk * NELTS)[lane]
                                    c_row[j_base + lane] = acc
                        i += 1

    parallelize[process_i_tile](num_i_tiles, num_threads)


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
