from std.sys import simd_width_of


# ===========================================================================
# Tile: a window into a row-major buffer
#
# Without it the kernels in gemm.mojo would index B and C by raw offset, like
# `c_ptr + i * n + j0 + jr` or `bp_worker + jp * kc * NR + pk * NR`. Each offset
# is correct, and each hides its meaning behind arithmetic. A `Tile` names that
# arithmetic once: a rows x cols rectangle whose rows sit `stride` elements apart
# (the parent row width). A sub-block of a bigger matrix is then a Tile with a
# smaller extent over the same buffer, no copy involved.
#
# Every method is `@always_inline` and does only the add and multiply the kernel
# would write by hand, so a Tile is a zero-cost rename (the same bet RegisterTile
# makes for the accumulator). Each kernel takes its tiles from
# `Matrix.noalias_view()` once at the top and works from those, with no raw
# pointers and no re-wrapping in loops.
# ===========================================================================


@fieldwise_init
struct Tile[dtype: DType, origin: Origin](Copyable & Movable):
    """A rows x cols window into a row-major buffer; `stride` is the parent row
    width. Generic over `origin` so the same type names a read-only operand (A, B)
    and a writable target (C); a store through an immutable-origin Tile is a type
    error, exactly as it should be."""

    var ptr: UnsafePointer[Scalar[Self.dtype], Self.origin]
    var rows: Int
    var cols: Int
    var stride: Int

    @always_inline
    def sub(self, r0: Int, c0: Int) -> Self:
        """The view whose top-left is element (r0, c0) of this one, same stride.
        Replaces `base + r0 * stride + c0` at the call sites that step by element
        offset (the packed kernel's j0 + jr is not a clean block index)."""
        return Self(
            self.ptr + r0 * self.stride + c0,
            self.rows - r0,
            self.cols - c0,
            self.stride,
        )

    @always_inline
    def tile[R: Int, C: Int](self, bi: Int, bj: Int) -> Self:
        """The (bi, bj)-th R x C block. Sugar for `sub(bi*R, bj*C)` that also
        carries the R x C extent. For loops that step in whole tiles (no-pack,
        serial), `c.tile[MR, NR](i // MR, j // NR)` reads as the block it is."""
        return Self(
            self.ptr + bi * R * self.stride + bj * C, R, C, self.stride
        )

    @always_inline
    def row(self, r: Int) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        """Pointer to the start of row r."""
        return self.ptr + r * self.stride

    @always_inline
    def addr(self, r: Int, c: Int) -> UnsafePointer[
        Scalar[Self.dtype], Self.origin
    ]:
        """Pointer to element (r, c), the base a strided operand loader walks."""
        return self.ptr + r * self.stride + c
