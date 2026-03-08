from matrix import Matrix
from std.algorithm.functional import parallelize, vectorize
from std.collections import InlineArray
from std.math import ceildiv, fma
from std.memory.unsafe_pointer import alloc
from std.sys import num_physical_cores, simd_width_of
from std.sys.intrinsics import prefetch, PrefetchOptions


fn matmul_naive[dtype: DType = DType.float64](
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


fn matmul_tiled[dtype: DType = DType.float64](
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


fn matmul_simd[dtype: DType = DType.float64](
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

                        fn fma[width: Int](j: Int) unified {mut}:
                            var c_vec = c_row.load[width=width](offset=j)
                            var b_vec = b_row.load[width=width](offset=j)
                            c_row.store(offset=j, val=c_vec + a_val * b_vec)

                        vectorize[NELTS](tile_n, fma)


fn matmul_parallel[dtype: DType = DType.float64](
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
fn _zero_fill[dtype: DType](mut c: Matrix[dtype]):
    """Vectorized zero-fill using SIMD stores."""
    comptime NELTS = simd_width_of[dtype]()
    var ptr = c.data.unsafe_ptr()
    var count = c.rows * c.cols
    fn _zero[width: Int](idx: Int) unified {mut}:
        ptr.store[width=width](offset=idx, val=SIMD[dtype, width](0))
    vectorize[NELTS](count, _zero)


fn matmul_comptime[
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

    _zero_fill[dtype](c)

    var num_j_tiles = ceildiv(n, TILE_N)

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

    fn process_gemv_tile(tile_idx: Int) capturing:
        var j0 = tile_idx * TILE_J
        var tile_n = min(TILE_J, n - j0)

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

    fn process_j_tile(j_tile_idx: Int) capturing:
        var j0 = j_tile_idx * TILE_N
        var tile_n = min(TILE_N, n - j0)
        var num_panels = ceildiv(tile_n, NR)

        var bp_tile = bp_buf + j_tile_idx * bp_per_tile

        for pc in range(0, k, KC):
            var kc = min(KC, k - pc)
            var first_k = (pc == 0)

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
                    var bp_panel = bp_tile + jp * kc * NR
                    if jr + NR > tile_n:
                        # Opt 2: vectorize handles SIMD + scalar remainder
                        var jj_limit = tile_n - jr
                        for ii in range(i, i + MR):
                            var c_row = c_ptr + ii * n + j0 + jr
                            var a_row = a_ptr + ii * k + pc

                            if first_k:
                                fn fma_rem_first[width: Int](jj: Int) unified {mut}:
                                    var acc = SIMD[dtype, width](0)
                                    for ppk in range(kc):
                                        acc = fma(
                                            SIMD[dtype, width](a_row[ppk]),
                                            (bp_panel + ppk * NR).load[width=width](offset=jj),
                                            acc,
                                        )
                                    c_row.store(offset=jj, val=acc)

                                vectorize[NELTS](jj_limit, fma_rem_first)
                            else:
                                fn fma_remainder[width: Int](jj: Int) unified {mut}:
                                    var acc = c_row.load[width=width](offset=jj)
                                    for ppk in range(kc):
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
                for jp in range(num_panels):
                    var jr = jp * NR
                    var bp_panel = bp_tile + jp * kc * NR
                    var jj_limit = min(NR, tile_n - jr)
                    var c_row = c_ptr + i * n + j0 + jr

                    if first_k:
                        fn fma_tail_first[width: Int](jj: Int) unified {mut}:
                            var acc = SIMD[dtype, width](0)
                            for ppk in range(kc):
                                acc = fma(
                                    SIMD[dtype, width](a_ptr[i * k + pc + ppk]),
                                    (bp_panel + ppk * NR).load[width=width](offset=jj),
                                    acc,
                                )
                            c_row.store(offset=jj, val=acc)

                        vectorize[NELTS](jj_limit, fma_tail_first)
                    else:
                        fn fma_tail[width: Int](jj: Int) unified {mut}:
                            var acc = c_row.load[width=width](offset=jj)
                            for ppk in range(kc):
                                acc = fma(
                                    SIMD[dtype, width](a_ptr[i * k + pc + ppk]),
                                    (bp_panel + ppk * NR).load[width=width](offset=jj),
                                    acc,
                                )
                            c_row.store(offset=jj, val=acc)

                        vectorize[NELTS](jj_limit, fma_tail)
                i += 1

    parallelize[process_j_tile](num_j_tiles, num_physical_cores())
    bp_buf.free()


fn matmul_goto[
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


fn _prefill_gemm[
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

    fn process_worker(worker_id: Int) capturing:
        var tiles_per_worker = ceildiv(num_j_tiles, num_workers)
        var j_tile_start = worker_id * tiles_per_worker
        var j_tile_end = min(j_tile_start + tiles_per_worker, num_j_tiles)
        if j_tile_start >= num_j_tiles:
            return

        var bp_worker = bp_buf + worker_id * bp_per_worker
        var ap_worker = ap_buf + worker_id * ap_per_worker

        # Process j-tiles in batches of NC_TILES to keep C in L2
        var jt = j_tile_start
        while jt < j_tile_end:
            var jt_batch_end = min(jt + NC_TILES, j_tile_end)

            for pc in range(0, k, KC):
                var kc = min(KC, k - pc)
                var is_first_k = (pc == 0)

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

                    # Pack B into NR-wide panels with prefetching
                    for jp in range(num_panels):
                        var jr = jp * NR
                        var panel_base = bp_worker + jp * kc * NR
                        if jr + NR <= tile_n:
                            for pk in range(kc):
                                var src = b_ptr + (pc + pk) * n + j0 + jr
                                var dst = panel_base + pk * NR
                                # Prefetch B rows ahead to hide stride-N latency
                                prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                                    b_ptr + (pc + pk + PREFETCH_B_DIST) * n + j0 + jr
                                )
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

                    # Micro-kernel with packed A + packed B
                    # jp-outer, i-inner: B panel stays in L2 across all i-panels,
                    # reducing L2→L1 traffic by ~2.7× vs the i-outer order.
                    for jp in range(num_panels):
                        var jr = jp * NR
                        var bp_panel = bp_worker + jp * kc * NR

                        if jr + NR > tile_n:
                            # Remainder columns: process all i-panels
                            var jj_limit = tile_n - jr
                            i = 0
                            while i + MR <= m:
                                for ii in range(i, i + MR):
                                    var c_row = c_ptr + ii * n + j0 + jr
                                    var a_row = a_ptr + ii * k + pc

                                    fn fma_remainder[width: Int](jj: Int) unified {mut}:
                                        var acc = c_row.load[width=width](offset=jj)
                                        if is_first_k:
                                            acc = SIMD[dtype, width](0)
                                        for ppk in range(kc):
                                            acc = fma(
                                                SIMD[dtype, width](a_row[ppk]),
                                                (bp_panel + ppk * NR).load[width=width](offset=jj),
                                                acc,
                                            )
                                        c_row.store(offset=jj, val=acc)

                                    vectorize[NELTS](jj_limit, fma_remainder)
                                i += MR
                            # Remaining rows for remainder columns
                            while i < m:
                                var c_row = c_ptr + i * n + j0 + jr

                                fn fma_tail_rem[width: Int](jj: Int) unified {mut}:
                                    var acc = c_row.load[width=width](offset=jj)
                                    if is_first_k:
                                        acc = SIMD[dtype, width](0)
                                    for ppk in range(kc):
                                        acc = fma(
                                            SIMD[dtype, width](a_ptr[i * k + pc + ppk]),
                                            (bp_panel + ppk * NR).load[width=width](offset=jj),
                                            acc,
                                        )
                                    c_row.store(offset=jj, val=acc)

                                vectorize[NELTS](jj_limit, fma_tail_rem)
                                i += 1
                            continue

                        # ---- Full NR-panel: process all i-panels ----
                        i = 0
                        ip = 0
                        while i + MR <= m:
                            var ap_panel = ap_worker + ip * MR * kc

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
                            var jj_limit = min(NR, tile_n - jr)
                            var c_row = c_ptr + i * n + j0 + jr

                            fn fma_tail[width: Int](jj: Int) unified {mut}:
                                var acc = c_row.load[width=width](offset=jj)
                                if is_first_k:
                                    acc = SIMD[dtype, width](0)
                                for ppk in range(kc):
                                    acc = fma(
                                        SIMD[dtype, width](a_ptr[i * k + pc + ppk]),
                                        (bp_panel + ppk * NR).load[width=width](offset=jj),
                                        acc,
                                    )
                                c_row.store(offset=jj, val=acc)

                            vectorize[NELTS](jj_limit, fma_tail)
                            i += 1

            jt += NC_TILES

    parallelize[process_worker](num_workers, num_workers)
    bp_buf.free()
    ap_buf.free()


fn matmul_prefill[
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


fn _decode_gemv[
    dtype: DType,
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # K-parallel GEMV optimized for decode (small M, large K×N).
    #
    # Flips goto_gemv's j-parallel (88KB B stride) to k-parallel
    # (sequential B scan). Each worker streams a contiguous slice of
    # B rows with p-outer/i-inner order so B is loaded once and reused
    # across all M output rows. Final lightweight reduction sums partials.
    comptime NELTS = simd_width_of[dtype]()
    comptime KU = 8

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var c_ptr = c.data.unsafe_ptr()
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var nw = num_physical_cores()

    var part_size = nw * m * n
    var part = alloc[Scalar[dtype]](part_size)

    fn _zero[width: Int](idx: Int) unified {mut}:
        part.store[width=width](offset=idx, val=SIMD[dtype, width](0))

    vectorize[NELTS, unroll_factor=4](part_size, _zero)

    fn worker(wid: Int) capturing:
        var rows_per = ceildiv(k, nw)
        var k0 = wid * rows_per
        var k1 = min(k0 + rows_per, k)
        var my_c = part + wid * m * n
        var p = k0

        while p + KU <= k1:
            var b_base = b_ptr + p * n

            prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
                b_ptr + (p + KU) * n
            )

            for i in range(m):
                var ci = my_c + i * n
                var ai = a_ptr + i * k
                var a_vals = InlineArray[Scalar[dtype], KU](fill=Scalar[dtype](0))
                comptime for ku in range(KU):
                    a_vals[ku] = ai[p + ku]

                fn do_fma[width: Int](j: Int) unified {mut}:
                    var acc = ci.load[width=width](offset=j)
                    comptime for ku in range(KU):
                        acc = fma(
                            SIMD[dtype, width](a_vals[ku]),
                            (b_base + ku * n).load[width=width](offset=j),
                            acc,
                        )
                    ci.store(offset=j, val=acc)

                vectorize[NELTS, unroll_factor=4](n, do_fma)
            p += KU

        # Remainder
        while p < k1:
            var bp = b_ptr + p * n
            for i in range(m):
                var a_val = a_ptr[i * k + p]
                var ci = my_c + i * n

                fn do_fma_tail[width: Int](j: Int) unified {mut}:
                    ci.store(
                        offset=j,
                        val=fma(
                            SIMD[dtype, width](a_val),
                            bp.load[width=width](offset=j),
                            ci.load[width=width](offset=j),
                        ),
                    )

                vectorize[NELTS](n, do_fma_tail)
            p += 1

    parallelize[worker](nw, nw)

    # Reduce partials into C
    for i in range(m):
        var ci_off = i * n

        fn reduce[width: Int](j: Int) unified {mut}:
            var s = SIMD[dtype, width](0)
            for w in range(nw):
                s += (part + w * m * n + ci_off).load[width=width](offset=j)
            (c_ptr + ci_off).store(offset=j, val=s)

        vectorize[NELTS](n, reduce)

    part.free()


fn matmul_decode[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  optimized for decode shapes (small M).
    # Uses k-parallel GEMV with sequential B streaming for maximum
    # memory bandwidth utilization.
    _decode_gemv[dtype](c, a, b)


fn matmul_dispatch[
    dtype: DType = DType.float64
](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    # Computes C = A * B  —  dispatches to the fastest kernel.
    #   - M < 8 (decode): k-parallel GEMV with sequential B streaming
    #   - M >= 8 (prefill): worker-based GEMM with A+B panel packing
    comptime NELTS = simd_width_of[dtype]()
    comptime MR = 8

    if a.rows < MR:
        _decode_gemv[dtype](c, a, b)
    else:
        comptime NR = 3 * NELTS
        comptime KC = 512
        comptime KU = 8
        comptime TILE_N = 72
        comptime NC_TILES = 256
        _prefill_gemm[dtype, MR, NR, KC, KU, TILE_N, NC_TILES](c, a, b)


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype](c, a, b)
