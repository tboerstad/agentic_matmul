# Making the kernels beautiful — directions to choose from

The code is already *correct, fast, and superbly documented*. What it is **not**
is *readable at the level of intent*. Three things fight the reader:

1. **Raw pointer arithmetic is the lingua franca.** Every kernel drops to
   `c_ptr + i * n + j0 + jr`, `a_ptr[(i + mr) * k + pc + pk]`,
   `bp_worker + jp * kc * NR + pk * NR` within two lines. The *layout* of every
   buffer lives only in the reader's head and in comments.
2. **`Matrix` is anemic.** It can index a single element and hand out a flat
   pointer — nothing in between. So a "tile of C" or "the packed B panel" has no
   type; it's a pointer plus a mental note.
3. **Two megastructures.** `_packed_gemm` is ~230 lines fusing six concerns
   (buffer sizing, A-prepack, the worker split, B-packing, the full microkernel,
   the masked tails). `matmul_dispatch` is a ~150-line `if/elif` tower that
   interleaves *shape classification* with *config selection*.

`RegisterTile` already proves the thesis: **a named, `@always_inline`,
`InlineArray`-backed abstraction lowers to the exact code you'd hand-write.**
Every option below extends that proof from the *accumulator* to the *operands and
the loop nest*. None should move a single GFLOP — `verify_dispatch` (max_err 0.0)
is the gate for all of them.

---

## Option A — `LayoutTensor` (the idiomatic-Mojo north star)

Delete `Matrix` and the raw pointers; express every buffer as
`LayoutTensor[dtype, Layout]`. The packing layout stops being a comment and
becomes the type:

```mojo
# The packed-B panel is [panel][k][NR]. Today that's a comment above
#   dst = bp_worker + jp * kc * NR + pk * NR
# With a Layout it's the declaration, and the index math is generated:
alias BPanel = Layout.row_major(num_panels, kc, NR)
var bp = LayoutTensor[dtype, BPanel](bp_worker)
...
bp[jp, pk, nv] = ...                 # was: dst.store[width=NELTS](offset=nv*NELTS, ...)
```

The microkernel reads tiles instead of computing offsets:

```mojo
# was: var c_block = c_ptr + i * n + j0 + jr ; tile.load(c_block, n)
var c_tile = c.tile[MR, NR](i_panel, j_panel)     # a view, no copy
tile.load(c_tile)
...
tile.store(c_tile)
```

and packing collapses to a layout-to-layout copy (`dst.copy_from(a.tile[MR, KC](ip, pc))`),
with `.vectorize[1, NELTS]()` describing the SIMD shape declaratively.

- **Why it's the right north star:** layout algebra *is* the packing spec; the
  same abstraction is what a future GPU path would use; it's where Mojo itself is
  heading. Index bugs become impossible to write.
- **The risk that must be retired first:** the numbers in this repo come from
  exact register pressure (`KU=2` → 32 zmm, see DESIGN.md) and noalias B-load
  hoisting. We must prove `LayoutTensor` on **CPU** lowers the 6×32 hot loop to
  the *identical* `vfmadd231pd` nest before trusting it on the headline path.
  Gate: an interleaved A/B (peak GFLOPS) of one branch ported to LayoutTensor vs
  the pointer version, plus `verify_dispatch` max_err 0.0.
- **Cost:** largest diff; depends on the `layout` package (not vendored in this
  checkout). **Highest payoff, highest risk.**

## Option B — a tiny home-grown `Tile` view (metal-close, zero codegen risk)

The 80% of Option A's readability with 0% of its codegen risk: one small value
type you own completely, every method `@always_inline`, so the emitted loads are
provably the ones written today.

```mojo
struct Tile[dtype: DType, origin: Origin]:
    """A rows x cols view into a row-major buffer; `stride` is the parent width."""
    var ptr: UnsafePointer[Scalar[dtype], origin]
    var rows: Int
    var cols: Int
    var stride: Int

    @always_inline
    def tile[R: Int, C: Int](self, bi: Int, bj: Int) -> Self:
        return Self(self.ptr + bi * R * self.stride + bj * C, R, C, self.stride)

    @always_inline
    def at(self, r: Int, c: Int) -> ref [origin] Scalar[dtype]:
        return self.ptr[r * self.stride + c]

    @always_inline
    def copy_from(self, src: Tile[dtype, _]): ...   # the pack step, named once
```

The microkernel and packing then read as intent, while every offset is the same
one the compiler emits today:

```mojo
# before
var dst = bp_worker + jp * kc * NR + pk * NR
comptime for nv in range(NR_VECS):
    dst.store[width=NELTS](offset=nv*NELTS, val=src.load[width=NELTS](offset=nv*NELTS))

# after
b_panel.tile[KC, NR](jp, 0).row(pk).copy_from(b_src.row(pk))
```

`RegisterTile.load/store` take a `Tile` instead of `(ptr, stride)`, so the
`c_block + mr * n` walk lives inside one well-tested place.

- **Pros:** kills the index gymnastics with *zero* risk to the tuned numbers (you
  control every load); adoptable **one kernel at a time**; no toolchain
  dependency; trivially `verify_dispatch`-able after each file.
- **Cons:** you maintain a sliver of what `LayoutTensor` already gives; no GPU /
  composition story. Think of it as the **bridge** to Option A.

## Option C — separate the *schedule* from the *kernel* (CuTile-flavored)

The deepest un-beauty isn't the pointers — it's that *policy* (which tile, which
KC, pack-A-or-not) is tangled into *mechanism* (the loop nest) and smeared across
a 150-line dispatch tower. Pull policy into data and drive mechanism generically:

```mojo
@fieldwise_init
struct GemmConfig:                 # the "what", as a value not control flow
    var kernel: KernelKind         # Packed | Gemv | NoPack | Serial
    var mr: Int; var nr: Int
    var kc: Int; var ku: Int; var tile_n: Int
    var shared_a: Bool

def classify(m: Int, n: Int, k: Int) -> GemmConfig:
    """The whole dispatch tower, as one table-shaped function returning a value.
    Testable in isolation: assert classify(1, 11008, 2048).kernel == Gemv."""
    ...

def matmul_dispatch(mut c, a, b):
    run(classify(c.rows /*...*/), c, a, b)     # the entire body
```

`run` holds the *one* packed loop nest; the micro-kernel and the pack step are
the only varying pieces (already isolated as `RegisterTile` + a `pack` fn). This
is the CuTile spirit: **describe the tiling, let one driver move the data.**

- **Pros:** turns control flow into a readable table; a new shape is a new row,
  not a new branch; `classify` becomes unit-testable; pairs naturally with A or B
  for the buffers.
- **Cons:** most architectural; GEMV / no-pack / packed are genuinely different
  loop nests, so this is honestly "*config + 3 templates*," not "one kernel" —
  over-unifying them would be worse than the tower. Comptime params (MR/NR/KC)
  must stay comptime, so `GemmConfig` splits into a comptime shape + a runtime
  part. Apply judiciously.

## Option D — quick wins (orthogonal, zero-risk, land today)

Independent of the big choice, three changes are pure upside:

- **Extract `classify` from the tower even without the rest of C** — just so the
  routing reads top-to-bottom as a table of shape → kernel.
- **Split `_packed_gemm`** into named `_pack_a_shared`, `_pack_b_tile`,
  `_micro_panel`, `_micro_tail` helpers. Same code, six concerns become six
  names. Biggest readability win per line changed.
- **`Matrix` ergonomics:** `__init__(rows, cols, fill)`, a `from_list`
  constructor, and a `tile()` view (the seed of Option B) — the benches/tests
  build matrices by hand today.

---

## Recommendation

1. **Now:** Option D (D is free) + **Option B**. B gives the headline beauty win
   — the kernels stop speaking pointer — with a guarantee of byte-identical
   codegen, adoptable file by file behind `verify_dispatch`.
2. **Next:** layer **Option C**'s `classify` on top — the dispatch tower is the
   single ugliest reader-facing thing and C fixes it without touching a hot loop.
3. **North star:** **Option A**, once a one-branch spike proves `LayoutTensor`
   CPU codegen is zero-cost on the 6×32 / `KU=2` path. B is deliberately shaped
   so its call sites (`.tile`, `.copy_from`, `.at`) map 1:1 onto `LayoutTensor`,
   making the eventual swap mechanical.

Every step is gated by `verify_dispatch` (max_err 0.0) and, for anything on a hot
loop, an interleaved A/B peak-GFLOPS check per the methodology note in README.md.

---

## Status: D + B landed

The chosen direction (D + B) is implemented:

- **`tile.mojo`** — the `Tile` view (Option B), `origin`-generic, every accessor
  `@always_inline`.
- **`Matrix`** — keyword-only `fill=` constructor, `from_rows` builder, `view()`.
- **`RegisterTile.load/store`** take a `Tile` instead of `(ptr, stride)`; the
  serial, no-pack, and packed full-panel call sites address through
  `sub`/`addr`/`row` instead of raw offsets.
- **`_pack_b_slab`** — the packed worker's B-slab packing extracted to a named
  helper (Option D's de-monolith).

Each landed behind `verify_dispatch` (max_err 0.0) with `--iterate` sweep ratios
unchanged within VM noise. **Next** per the recommendation: Option C's `classify`
for the dispatch tower, then an Option A `LayoutTensor` spike — the `Tile` call
sites (`sub`/`tile`/`row`/`addr`) are shaped to map onto `LayoutTensor` so that
swap stays mechanical.
