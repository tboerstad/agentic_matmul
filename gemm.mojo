from matrix import Matrix
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.sys import num_physical_cores, simd_width_of
from std.sys.intrinsics import llvm_intrinsic


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
                            var a_vals = InlineArray[Scalar[dtype], MR](
                                fill=Scalar[dtype](0)
                            )
                            comptime for mr in range(MR):
                                a_vals[mr] = a_ptr[(i + mr) * k + p]
                            var b_row = b_ptr + p * n + j0

                            fn fma_mr[width: Int](j: Int) unified {mut}:
                                var b_vec = b_row.load[width=width](offset=j)
                                comptime for mr in range(MR):
                                    var c_row = c_ptr + (i + mr) * n + j0
                                    c_row.store(
                                        offset=j,
                                        val=c_row.load[width=width](offset=j)
                                        + a_vals[mr] * b_vec,
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
                        # Process NELTS columns at a time
                        var j = 0
                        while j + NELTS <= tile_n:
                            # Load MR C accumulators from memory (once per tile)
                            var acc = InlineArray[SIMD[dtype, NELTS], MR](
                                fill=SIMD[dtype, NELTS](0)
                            )
                            comptime for mr in range(MR):
                                acc[mr] = (c_ptr + (i + mr) * n + j0).load[
                                    width=NELTS
                                ](offset=j)

                            # Accumulate across entire k-tile in registers
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_vec = (b_ptr + p * n + j0).load[width=NELTS](offset=j)
                                comptime for mr in range(MR):
                                    acc[mr] += a_ptr[(i + mr) * k + p] * b_vec

                            # Store accumulators back (once per tile)
                            comptime for mr in range(MR):
                                (c_ptr + (i + mr) * n + j0).store(
                                    offset=j, val=acc[mr]
                                )
                            j += NELTS

                        # Scalar remainder for j
                        while j < tile_n:
                            var acc = InlineArray[Scalar[dtype], MR](
                                fill=Scalar[dtype](0)
                            )
                            comptime for mr in range(MR):
                                acc[mr] = (c_ptr + (i + mr) * n + j0)[j]
                            for pk in range(tile_k):
                                var p = p0 + pk
                                var b_val = b_ptr[p * n + j0 + j]
                                comptime for mr in range(MR):
                                    acc[mr] += a_ptr[(i + mr) * k + p] * b_val
                            comptime for mr in range(MR):
                                (c_ptr + (i + mr) * n + j0)[j] = acc[mr]
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


@always_inline
fn _prefetch_r[dtype: DType](ptr: UnsafePointer[Scalar[dtype], ...], offset: Int):
    """Prefetch for read, low temporal locality (L2), data cache."""
    llvm_intrinsic["llvm.prefetch", NoneType](
        (ptr + offset).bitcast[NoneType](), Int32(0), Int32(1), Int32(1)
    )


fn matmul_comptime[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  compile-time optimized GOTO-style GEMM.
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
    comptime MR = 6          # rows of C per micro-kernel (6 > 4: more B-reuse)
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

    # Vectorized zero-fill of C
    var total = m * n
    fn zero[width: Int](idx: Int) unified {mut}:
        c_ptr.store[width=width](offset=idx, val=SIMD[dtype, width](0))
    vectorize[NELTS](total, zero)

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
                # ---- MR-blocked rows with comptime micro-kernel ----
                var i = 0
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
                                    _prefetch_r[dtype](
                                        b_ptr,
                                        (p + PREFETCH_DIST) * n + jj,
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

                    # Remainder columns: single SIMD vector at a time
                    while j + NELTS <= tile_n:
                        var jj = j0 + j
                        var acc_r = InlineArray[SIMD[dtype, NELTS], MR](
                            fill=SIMD[dtype, NELTS](0)
                        )
                        comptime for mr in range(MR):
                            acc_r[mr] = (c_ptr + (i + mr) * n + jj).load[
                                width=NELTS
                            ]()
                        for pk in range(tile_k):
                            var p = p0 + pk
                            var bv = (b_ptr + p * n + jj).load[width=NELTS]()
                            comptime for mr in range(MR):
                                acc_r[mr] += a_ptr[(i + mr) * k + p] * bv
                        comptime for mr in range(MR):
                            (c_ptr + (i + mr) * n + jj).store(
                                val=acc_r[mr]
                            )
                        j += NELTS

                    # Scalar remainder for j
                    while j < tile_n:
                        var jj = j0 + j
                        comptime for mr in range(MR):
                            var acc_s = c_ptr[(i + mr) * n + jj]
                            for pk in range(tile_k):
                                acc_s += (
                                    a_ptr[(i + mr) * k + p0 + pk]
                                    * b_ptr[(p0 + pk) * n + jj]
                                )
                            c_ptr[(i + mr) * n + jj] = acc_s
                        j += 1

                    i += MR

                # Handle remaining rows (< MR) with single-row SIMD
                while i < m:
                    var c_row = c_ptr + i * n + j0
                    var j = 0
                    while j + NELTS <= tile_n:
                        var acc = c_row.load[width=NELTS](offset=j)
                        for pk in range(tile_k):
                            var p = p0 + pk
                            var bv = (b_ptr + p * n + j0).load[width=NELTS](
                                offset=j
                            )
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


fn _goto_gemv[
    dtype: DType, NELTS: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # GEMV path for small M (esp. M=1 decode).
    # Streams through B sequentially row-by-row with j-parallelism,
    # enabling hardware prefetching and maximizing DRAM bandwidth.
    comptime TILE_J = 1024  # C chunk = 1024*8 = 8KB, fits L1

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()

    var num_j_tiles = (n + TILE_J - 1) // TILE_J

    fn process_gemv_tile(tile_idx: Int) capturing:
        var j0 = tile_idx * TILE_J
        var j_end = j0 + TILE_J
        if j_end > n:
            j_end = n
        var tile_n = j_end - j0

        for i in range(m):
            var c_row = c_ptr + i * n + j0
            for p in range(k):
                var a_val = a_ptr[i * k + p]
                var b_row = b_ptr + p * n + j0

                fn fma_gemv[width: Int](j: Int) unified {mut}:
                    c_row.store(
                        offset=j,
                        val=c_row.load[width=width](offset=j)
                        + a_val * b_row.load[width=width](offset=j),
                    )

                vectorize[NELTS](tile_n, fma_gemv)

    parallelize[process_gemv_tile](num_j_tiles, num_physical_cores())


fn _goto_gemm[
    dtype: DType, NELTS: Int, MR: Int, NR: Int, KC: Int, KU: Int, TILE_N: Int
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # j-parallel GOTO GEMM with per-tile B-panel packing.
    # j→k→i loop order keeps C panel in L2 across all k-tiles.
    # B is packed into NR-wide contiguous column panels for
    # sequential micro-kernel access.
    comptime NUM_LOCAL_PANELS = TILE_N // NR

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()

    var num_j_tiles = (n + TILE_N - 1) // TILE_N

    var bp_per_tile = NUM_LOCAL_PANELS * KC * NR + KU * NR
    var bp_total = num_j_tiles * bp_per_tile
    var bp_list = List[Scalar[dtype]](capacity=bp_total)
    for _ in range(bp_total):
        bp_list.append(Scalar[dtype](0))
    var bp_buf = bp_list.unsafe_ptr()

    fn process_j_tile(j_tile_idx: Int) capturing:
        var j0 = j_tile_idx * TILE_N
        var j_end = j0 + TILE_N
        if j_end > n:
            j_end = n
        var tile_n = j_end - j0
        var num_panels = (tile_n + NR - 1) // NR

        var bp_tile = bp_buf + j_tile_idx * bp_per_tile

        for pc in range(0, k, KC):
            var kc = KC
            if pc + kc > k:
                kc = k - pc

            # Pack B[pc:pc+kc, j0:j0+tile_n] into NR-panels
            for jp in range(num_panels):
                var jr = jp * NR
                var panel_base = bp_tile + jp * kc * NR
                if jr + NR <= tile_n:
                    for pk in range(kc):
                        var src = b_ptr + (pc + pk) * n + j0 + jr
                        var dst = panel_base + pk * NR
                        dst.store[width=NELTS](val=src.load[width=NELTS]())
                        dst.store[width=NELTS](
                            offset=NELTS,
                            val=src.load[width=NELTS](offset=NELTS),
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
                    var bp_panel = bp_tile + jp * kc * NR
                    if jr + NR > tile_n:
                        # Remainder columns: scalar path
                        var jj_limit = tile_n - jr
                        for ii in range(i, i + MR):
                            var c_row = c_ptr + ii * n + j0 + jr
                            var jj = 0
                            while jj + NELTS <= jj_limit:
                                var acc = c_row.load[width=NELTS](offset=jj)
                                for ppk in range(kc):
                                    acc += a_ptr[
                                        ii * k + pc + ppk
                                    ] * (
                                        bp_panel + ppk * NR
                                    ).load[width=NELTS](offset=jj)
                                c_row.store(offset=jj, val=acc)
                                jj += NELTS
                            while jj < jj_limit:
                                var acc = c_row[jj]
                                for ppk in range(kc):
                                    acc += a_ptr[
                                        ii * k + pc + ppk
                                    ] * (bp_panel + ppk * NR)[jj]
                                c_row[jj] = acc
                                jj += 1
                        continue

                    # ---- Full MR×NR micro-kernel ----
                    var acc = InlineArray[SIMD[dtype, NELTS], MR * 2](
                        fill=SIMD[dtype, NELTS](0)
                    )
                    comptime for mr in range(MR):
                        comptime for nr in range(2):
                            acc[mr * 2 + nr] = (
                                c_ptr + (i + mr) * n + j0 + jr
                            ).load[width=NELTS](offset=nr * NELTS)

                    # K-loop with KU unrolling + prefetching
                    var pk = 0
                    var pk_end = kc - (kc % KU)
                    while pk < pk_end:
                        comptime for ku in range(KU):
                            var bp_k = bp_panel + (pk + ku) * NR
                            _prefetch_r[dtype](
                                bp_panel, (pk + ku + 4) * NR
                            )
                            var a_vals = InlineArray[Scalar[dtype], MR](
                                fill=Scalar[dtype](0)
                            )
                            comptime for mr in range(MR):
                                a_vals[mr] = a_ptr[
                                    (i + mr) * k + pc + pk + ku
                                ]
                            comptime for nr in range(2):
                                var bv = bp_k.load[width=NELTS](
                                    offset=nr * NELTS
                                )
                                comptime for mr in range(MR):
                                    acc[mr * 2 + nr] += a_vals[mr] * bv
                        pk += KU

                    # K remainder
                    while pk < kc:
                        var bp_k = bp_panel + pk * NR
                        var a_vals = InlineArray[Scalar[dtype], MR](
                            fill=Scalar[dtype](0)
                        )
                        comptime for mr in range(MR):
                            a_vals[mr] = a_ptr[(i + mr) * k + pc + pk]
                        comptime for nr in range(2):
                            var bv = bp_k.load[width=NELTS](
                                offset=nr * NELTS
                            )
                            comptime for mr in range(MR):
                                acc[mr * 2 + nr] += a_vals[mr] * bv
                        pk += 1

                    # Store accumulators back to C
                    comptime for mr in range(MR):
                        comptime for nr in range(2):
                            (c_ptr + (i + mr) * n + j0 + jr).store(
                                offset=nr * NELTS, val=acc[mr * 2 + nr],
                            )

                i += MR

            # Handle remaining rows (< MR)
            while i < m:
                for jp in range(num_panels):
                    var jr = jp * NR
                    var bp_panel = bp_tile + jp * kc * NR
                    var jj_limit = NR
                    if jr + NR > tile_n:
                        jj_limit = tile_n - jr
                    var c_row = c_ptr + i * n + j0 + jr
                    var jj = 0
                    while jj + NELTS <= jj_limit:
                        var acc = c_row.load[width=NELTS](offset=jj)
                        for ppk in range(kc):
                            var a_val = a_ptr[i * k + pc + ppk]
                            acc += a_val * (
                                bp_panel + ppk * NR
                            ).load[width=NELTS](offset=jj)
                        c_row.store(offset=jj, val=acc)
                        jj += NELTS
                    while jj < jj_limit:
                        var acc = c_row[jj]
                        for ppk in range(kc):
                            acc += a_ptr[
                                i * k + pc + ppk
                            ] * (bp_panel + ppk * NR)[jj]
                        c_row[jj] = acc
                        jj += 1
                i += 1

    parallelize[process_j_tile](num_j_tiles, num_physical_cores())
    _ = bp_list  # prevent early free


fn matmul_goto[
    dtype: DType = DType.float64, *, transpose_b: Bool = False
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * op(B)  —  GOTO-style GEMM with B-panel packing.
    # Dispatches to _goto_gemv (M < MR) or _goto_gemm (M >= MR).
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 6          # rows of C per micro-kernel
    comptime NR = 2 * NELTS  # columns per micro-kernel (16 for float64 AVX-512)
    comptime KC = 256        # k-tile: Bp NR-panel = 256*16*8 = 32KB fits L1
    comptime KU = 8          # k-unroll factor
    comptime TILE_N = 64     # j-tile: C panel = M*64*8 fits L2

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var c_ptr = c.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()

    # Vectorized zero-fill of C
    var total = m * n
    fn zero_fill[width: Int](idx: Int) unified {mut}:
        c_ptr.store[width=width](offset=idx, val=SIMD[dtype, width](0))
    vectorize[NELTS](total, zero_fill)

    comptime if transpose_b:
        for i in range(m):
            for p in range(k):
                var a_val = a_ptr[i * k + p]
                for j in range(n):
                    var idx = i * n + j
                    c_ptr[idx] = c_ptr[idx] + a_val * b_ptr[j * k + p]
    else:
        if m < MR:
            _goto_gemv[dtype, NELTS](c, a, b)
        else:
            _goto_gemm[dtype, NELTS, MR, NR, KC, KU, TILE_N](c, a, b)


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
