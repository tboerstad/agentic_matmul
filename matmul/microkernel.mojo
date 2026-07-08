"""The register-tile micro-kernel every GEMM in this package is built on.

Every kernel in this package is the same idea: hold an MR x (NR_VECS*NELTS)
block of C entirely in SIMD registers, sweep it over K with rank-1 updates,
then write it back once. `RegisterTile` is that block. Its accumulator is an
`InlineArray` the compiler flattens into registers, so each method is a
zero-cost abstraction that emits exactly the FMA/load/store nest you would
otherwise hand-write. Each kernel module then expresses only its own packing
and loop scaffolding.

This module holds the tile itself, the two operand loaders every K-step
feeds it, and the three micro-kernels (full, partial-N, masked) the drivers
in `packed.mojo` and `nopack.mojo` run over their blocks. It also defines
`compute_dtype`, the storage-to-compute dtype policy the whole package
follows.
"""

from matmul.tile import Tile
from std.collections import InlineArray
from std.sys.intrinsics import prefetch, PrefetchOptions


@always_inline
def compute_dtype[dtype: DType]() -> DType:
    """The dtype the kernels accumulate and FMA in for a given storage dtype.

    Identity for every dtype except bf16. AVX-512 has no bf16 SIMD FMA (and
    LLVM has nothing to lower one to), so a bf16 accumulator emulates
    element-wise at 4-5 GFLOPS, a 20-60x loss vs linalg. bf16 therefore keeps
    bf16 storage for A, B and C and computes in f32: the pack stages widen A
    and B panels to f32 as they copy (packing already touches every element),
    the no-pack and GEMV paths widen on load (a bf16 load plus a 16-bit shift
    makes an f32), and C is narrowed back to bf16 once per register-tile
    store. The ceiling becomes the f32 FMA peak. docs/SOL.md idea 2."""
    return DType.float32 if dtype == DType.bfloat16 else dtype


struct RegisterTile[dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int](
    Copyable
):
    """An MR x (NR_VECS*NELTS) block of C, resident in SIMD registers.

    `dtype` is the COMPUTE dtype (see `compute_dtype`). The C block in memory
    may be stored narrower (bf16 C under an f32 tile); `load`/`store` and
    their masked variants are generic over that storage dtype and cast at the
    boundary, which folds to nothing when the two dtypes match."""

    var acc: InlineArray[SIMD[Self.dtype, Self.NELTS], Self.MR * Self.NR_VECS]

    @always_inline
    def __init__(out self):
        """Start at zero: a fresh C block, or the first K-panel of a sweep."""
        self.acc = InlineArray[
            SIMD[Self.dtype, Self.NELTS], Self.MR * Self.NR_VECS
        ](fill=SIMD[Self.dtype, Self.NELTS](0))

    @always_inline
    def rank1_update(
        mut self,
        a_col: InlineArray[Scalar[Self.dtype], Self.MR],
        b_row: InlineArray[SIMD[Self.dtype, Self.NELTS], Self.NR_VECS],
    ):
        """C_tile += a_col (x) b_row, one K-step: broadcast each of the MR A
        scalars across the NR_VECS B vectors and FMA into the tile. Takes SIMD
        values only (no pointers), so it can never perturb the noalias B-load
        hoisting the hot loops depend on."""
        comptime for mr in range(Self.MR):
            var a_bc = SIMD[Self.dtype, Self.NELTS](a_col[mr])
            comptime for nr in range(Self.NR_VECS):
                self.acc[mr * Self.NR_VECS + nr] = a_bc.fma(
                    b_row[nr], self.acc[mr * Self.NR_VECS + nr]
                )

    @always_inline
    def load[sdtype: DType, org: MutOrigin](mut self, c: Tile[sdtype, org]):
        """Seed the tile from a C block, to accumulate onto a prior K-panel's
        partial. `c.row(0)` is the block's top-left, rows c.stride apart."""
        comptime for mr in range(Self.MR):
            var row = c.row(mr)
            comptime for nr in range(Self.NR_VECS):
                self.acc[mr * Self.NR_VECS + nr] = row.load[width = Self.NELTS](
                    offset=nr * Self.NELTS
                ).cast[Self.dtype]()

    @always_inline
    def store[sdtype: DType, org: MutOrigin](self, c: Tile[sdtype, org]):
        """Write the finished tile back to a C block (see `load` for the view)."""
        comptime for mr in range(Self.MR):
            var row = c.row(mr)
            comptime for nr in range(Self.NR_VECS):
                row.store(
                    offset=nr * Self.NELTS,
                    val=self.acc[mr * Self.NR_VECS + nr].cast[sdtype](),
                )

    @always_inline
    def load_masked[
        sdtype: DType, org: MutOrigin
    ](mut self, c: Tile[sdtype, org], rows: Int, cols: Int):
        """`load`, restricted to the first `rows` x `cols` of the block; the
        masked-out lanes keep their zero from `__init__`."""
        comptime for mr in range(Self.MR):
            if mr < rows:
                var row = c.row(mr)
                comptime for nr in range(Self.NR_VECS):
                    var col0 = nr * Self.NELTS
                    if col0 + Self.NELTS <= cols:
                        self.acc[mr * Self.NR_VECS + nr] = row.load[
                            width = Self.NELTS
                        ](offset=col0).cast[Self.dtype]()
                    elif col0 < cols:
                        var v = SIMD[Self.dtype, Self.NELTS](0)
                        for e in range(cols - col0):
                            v[e] = row[col0 + e].cast[Self.dtype]()
                        self.acc[mr * Self.NR_VECS + nr] = v

    @always_inline
    def store_masked[
        sdtype: DType, org: MutOrigin
    ](self, c: Tile[sdtype, org], rows: Int, cols: Int):
        """`store`, restricted to the first `rows` x `cols` of the block."""
        comptime for mr in range(Self.MR):
            if mr < rows:
                var row = c.row(mr)
                comptime for nr in range(Self.NR_VECS):
                    var col0 = nr * Self.NELTS
                    if col0 + Self.NELTS <= cols:
                        row.store(
                            offset=col0,
                            val=self.acc[mr * Self.NR_VECS + nr].cast[sdtype](),
                        )
                    elif col0 < cols:
                        var v = self.acc[mr * Self.NR_VECS + nr]
                        for e in range(cols - col0):
                            row[col0 + e] = v[e].cast[sdtype]()


# --- The two operands of one K-step ----------------------------------------
#
# Every K-step in every kernel feeds `rank1_update` the same pair: NR_VECS
# contiguous B vectors and MR A scalars. These two loaders name that gather
# once, so every kernel's inner loop collapses to a single readable line:
#
#     tile.rank1_update(load_a_col[MR](a, stride), load_b_row[NR_VECS, NELTS](b))


@always_inline
def load_b_row[
    dtype: DType, org: ImmutOrigin, //, NR_VECS: Int, NELTS: Int,
    cdtype: DType = dtype,
](bp_k: UnsafePointer[Scalar[dtype], org]) -> InlineArray[
    SIMD[cdtype, NELTS], NR_VECS
]:
    """The B operand of one K-step: NR_VECS contiguous NELTS-wide vectors,
    widened to the compute dtype when the source is stored narrower (the
    no-pack kernels reading bf16 B; the cast folds away when equal)."""
    var bv = InlineArray[SIMD[cdtype, NELTS], NR_VECS](uninitialized=True)
    comptime for nr in range(NR_VECS):
        bv[nr] = bp_k.load[width=NELTS](offset=nr * NELTS).cast[cdtype]()
    return bv^


@always_inline
def load_a_col[
    dtype: DType, org: ImmutOrigin, //, MR: Int, cdtype: DType = dtype
](a_base: UnsafePointer[Scalar[dtype], org], stride: Int) -> InlineArray[
    Scalar[cdtype], MR
]:
    """The A operand of one K-step: MR scalars at `stride` apart, widened to
    the compute dtype like `load_b_row`. `stride == 1` reads a packed-A
    column; `stride == k` gathers a column straight from a row-major A (the
    no-pack kernels)."""
    var a_col = InlineArray[Scalar[cdtype], MR](uninitialized=True)
    comptime for mr in range(MR):
        a_col[mr] = a_base[mr * stride].cast[cdtype]()
    return a_col^


@always_inline
def masked_microkernel[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int, NR: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: MutOrigin, cdtype: DType,
](
    c_block: Tile[dtype, c_org],
    a_block: Tile[dtype, a_org],
    bp_panel: UnsafePointer[Scalar[cdtype], b_org],
    kc: Int,
    rows: Int,
    cols: Int,
    is_first_k: Bool,
):
    """Cold-path register tile: the leftover blocks the hot loop can't take.

    Handles an M-remainder (only `rows` of MR rows active) and/or a partial
    NR-panel (only `cols` of NR columns valid) by masking the C load and store;
    reads A unpacked (so it covers the un-packed remainder rows too) and B from
    the zero-padded packed panel. With rows == MR and cols == NR it is the
    unmasked full kernel.

    `c_block`/`a_block` are `sub`-views onto the block's corner, (i, j0+jr) in C
    and (i, pc) in A."""
    var tile = RegisterTile[cdtype, MR, NR_VECS, NELTS]()
    if not is_first_k:
        tile.load_masked(c_block, rows, cols)
    for pk in range(kc):
        # A is gathered here (not via load_a_col) because the gather is guarded
        # by mr < rows: rows past the M-remainder are out of bounds, so leave
        # them zero (their acc lanes are never stored).
        var a_col = InlineArray[Scalar[cdtype], MR](fill=Scalar[cdtype](0))
        comptime for mr in range(MR):
            if mr < rows:
                a_col[mr] = a_block.row(mr)[pk].cast[cdtype]()
        tile.rank1_update(a_col, load_b_row[NR_VECS, NELTS](bp_panel + pk * NR))
    tile.store_masked(c_block, rows, cols)


@always_inline
def partial_n_microkernel[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int, NR: Int, KU: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: MutOrigin,
    adtype: DType, cdtype: DType,
](
    c_block: Tile[dtype, c_org],
    a_base: UnsafePointer[Scalar[adtype], a_org],
    a_k_step: Int,
    a_stride: Int,
    bp_panel: UnsafePointer[Scalar[cdtype], b_org],
    kc: Int,
    cols: Int,
    is_first_k: Bool,
):
    """Hot-path register tile for a partial NR-panel with all MR rows live.

    The packed B panel is zero-padded to full NR, so the K-sweep itself needs
    no masking at all: this is the same KU-unrolled loop as
    `full_microkernel` (see there for the A addressing), masking only the C
    load and store to the `cols` valid columns. That masking runs once per
    tile instead of once per K-step, so a partial column sustains full
    register-tile throughput. `masked_microkernel` used to take these tiles
    with its unguarded-per-K-step gather and no unroll; on a shape whose
    partial column is a big slice of N that ran at a fraction of the full
    tile's rate (f32 sq300: the 44-wide column is 1 of 5, and the whole GEMM
    sat at 0.83 vs linalg). Bit-identical: same FMA order per element, and
    the zero columns contribute nothing."""
    var tile = RegisterTile[cdtype, MR, NR_VECS, NELTS]()
    if not is_first_k:
        tile.load_masked(c_block, MR, cols)

    var pk = 0
    while pk + KU <= kc:
        comptime for ku in range(KU):
            tile.rank1_update(
                load_a_col[MR, cdtype](a_base + (pk + ku) * a_k_step, a_stride),
                load_b_row[NR_VECS, NELTS](bp_panel + (pk + ku) * NR),
            )
        pk += KU
    while pk < kc:
        tile.rank1_update(
            load_a_col[MR, cdtype](a_base + pk * a_k_step, a_stride),
            load_b_row[NR_VECS, NELTS](bp_panel + pk * NR),
        )
        pk += 1

    tile.store_masked(c_block, MR, cols)


@always_inline
def full_microkernel[
    dtype: DType, MR: Int, NR_VECS: Int, NELTS: Int, NR: Int, KU: Int,
    c_org: MutOrigin, a_org: ImmutOrigin, b_org: MutOrigin,
    adtype: DType, cdtype: DType,
](
    c_block: Tile[dtype, c_org],
    a_base: UnsafePointer[Scalar[adtype], a_org],
    a_k_step: Int,
    a_stride: Int,
    bp_panel: UnsafePointer[Scalar[cdtype], b_org],
    kc: Int,
    is_first_k: Bool,
):
    """Hot-path register tile for one full MR x NR block of C.

    The full-NR-panel counterpart of `masked_microkernel`: no masking, every
    row and column live. Reads the packed B panel and, per K-step, the MR A
    scalars from `a_base + pk * a_k_step` at element stride `a_stride`: a
    packed-A panel is (panel, MR, 1), a row-major A is (addr(i, pc), 1, k).
    Both are compile-time-known at every call site, so they fold away.

    The K-sweep is unrolled by KU so KU*NR_VECS B-vectors stay live per step
    (KU=2 keeps the 6x32 tile's 24 + 8 = 32 accumulator+B vectors inside the
    AVX-512 register file; KU=4 needs 40 and spills). The B loads stay inline
    so the noalias B-load hoist holds."""
    comptime for mr in range(MR):
        prefetch[PrefetchOptions().for_write().high_locality().to_data_cache()](
            c_block.row(mr)
        )

    var tile = RegisterTile[cdtype, MR, NR_VECS, NELTS]()
    if not is_first_k:
        tile.load(c_block)

    var pk = 0
    while pk + KU <= kc:
        comptime for ku in range(KU):
            tile.rank1_update(
                load_a_col[MR, cdtype](a_base + (pk + ku) * a_k_step, a_stride),
                load_b_row[NR_VECS, NELTS](bp_panel + (pk + ku) * NR),
            )
        pk += KU
    while pk < kc:
        tile.rank1_update(
            load_a_col[MR, cdtype](a_base + pk * a_k_step, a_stride),
            load_b_row[NR_VECS, NELTS](bp_panel + pk * NR),
        )
        pk += 1

    tile.store(c_block)
