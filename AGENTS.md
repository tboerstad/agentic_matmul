# Agent Instructions

## Verify hardware FIRST

Before running any benchmarks or interpreting any results, identify the hardware you are running on.
All performance numbers in this repo are **hardware-specific** — they will vary significantly across
different CPUs, core counts, cache sizes, SIMD widths, and virtualization environments.

Run this immediately:
```bash
lscpu | grep -E "Model name|CPU\(s\)|MHz|cache|Flags"
python -c "import numpy; numpy.show_config()"
```

Check for:
- **CPU model and clock speed** (e.g. Intel Xeon @ 2.80 GHz)
- **Core count** (physical cores available for parallelism)
- **SIMD support** (AVX-512, AVX2, SSE4 — determines FLOPS/cycle)
- **Virtualization** (KVM/container overhead reduces effective throughput)
- **BLAS backend** (OpenBLAS version, MKL, etc.)

The theoretical peak GFLOPS formula is:
```
peak = clock_GHz × doubles_per_SIMD × 2(FMA) × FMA_units × cores
```
For example: 2.8 GHz × 8 (AVX-512) × 2 × 2 (dual FMA) × 4 cores = 358.4 GFLOPS.

That formula uses the base clock and can undershoot badly: under full AVX-512 FMA
load these VMs turbo above base (Machine B's paper peak is 268.8 GFLOPS f64, the
measured peak is ~310). Measure the real ceiling in the same boot with
`bash sol/run.sh` (FMA peak + DRAM/L3/L2 bandwidth) and use it as the
denominator for any efficiency claim. See SOL.md for the full speed-of-light
analysis, per-shape rooflines, and the current % of SOL standings.

**All benchmark numbers in README.md and SOL.md were measured on specific machines
(Machine A: Intel Xeon @ 2.80 GHz Skylake; Machine B: Intel Xeon @ 2.10 GHz
Granite Rapids; both 4 cores, AVX-512, KVM). Your results WILL differ.**

## First-time setup after cloning

1. Run `bash setup.sh` to install dependencies (uv, latest Mojo nightly — MAX 26.5 → Mojo 1.0.0b3)
2. Activate the venv: `source .venv/bin/activate`
3. Verify the setup works: `mojo main.mojo`

If an existing `.venv` has a stale nightly, the build can fail with errors like
`'def' functions will soon stop implying 'raises'` (an intermediate nightly
promoted that deprecation to an error). Upgrade in place:
`uv pip install --upgrade modular --index https://whl.modular.com/nightly/simple/ --prerelease allow`.
The code targets Mojo 1.0.0b3+.

## Mojo 101

- Import from `std`: `from std.collections import List`
- Use `SIMD[DType.float64, N]` for fixed-size arrays
- Declare functions with `def`, variables with `var`. There is no `fn` keyword — every function, including `@always_inline` helpers and nested closures, is a `def`
- Return owned `List` values with `^` (e.g. `return result^`); without it the compiler rejects the implicit copy
- Use `mut` (not `inout`) for mutable function parameters: `def foo(mut x: List[Float64])`
- Use `comptime if` (not `@parameter if`) for compile-time branching on parameter values
- Inside a struct, reference its parameters with `Self.` prefix: `List[Scalar[Self.dtype]]`, not `List[Scalar[dtype]]`
- Use `vectorize[simd_width](size, closure)` from `std.algorithm.functional` to auto-vectorize loops with automatic remainder handling — no manual SIMD + scalar tail loop needed
- Closures declare an explicit capture list in braces after the parameters — one `mut <name>` or `read <name>` entry per enclosing variable they touch: `def name[width: Int](i: Int) {mut acc, read a_val, read n}:`. The per-variable `mut`/`read` annotations make each capture's mutability explicit. The same form is used for both `vectorize` and `parallelize` closures
- Use `parallelize[func](num_work_items, num_workers)` from `std.algorithm.functional` to distribute work across threads — the worker closure uses the same brace capture list: `def worker(i: Int) {mut c_ptr, read m, read n, read k}:`
- Use `InlineArray[T, N]` (from `std.collections`) + `comptime for` to replace hand-numbered variables (e.g. `acc0`–`acc3`). The compiler flattens comptime-indexed `InlineArray` elements into registers, producing identical machine code to manual variables — but the code scales when you change tile sizes like MR/NR/KU
- Declare hoisted closure-capture bindings type-only (`var c_row: type_of(c_ptr)`, `var kc_h: Int`) rather than with a dummy initializer (`var c_row = c_ptr`). The dummy value is always overwritten before the inner closure reads it, and Mojo 1.0.0b1+ flags it as a dead store (`assignment ... was never used`). `type_of(x)` recovers an `UnsafePointer`'s full type including its `origin` parameter

## Development

- Mojo source files use the `.mojo` extension
- The Mojo compiler and runtime are installed in `.venv/` via `uv`
- Always activate the venv before running `mojo` commands

## Judging a perf change (read before you claim a win or a loss)

A single dispatch/linalg ratio is noise. On shared VMs it swings ±5–10% with the
turbo/thermal state a process launches into, and we have repeatedly reported those
swings as real kernel wins or losses (e.g. up-m512 at 0.79 in one launch, 1.04 in
the next, byte-identical code). Do not trust one ratio, and do not compare
absolute GFLOPS across separate process launches.

To decide whether a shape genuinely won or lost, run `mojo bench_focus.mojo`
(no flag). It runs the shape set for 10 independent epochs, each a peak over 12
interleaved A/B reps, and prints the ratio's mean ± stdev with a 2σ verdict:
WIN (mean − 2σ > 1.0), LOSE (mean + 2σ < 1.0), or tie (band straddles 1.0). Only
call a shape a win or loss when its verdict says so; a `tie` is within noise.
`--quick` gives the old single-epoch ratios for a fast edit-loop check, but it is
NOT a basis for judging a change.

**Judge by % of SOL, not the linalg ratio.** `bench_focus` now measures this
machine's own ceilings in the same process (all-core FMA peak, L3/DRAM read
bandwidth, L3 size, via `sol.mojo`, a Mojo port of `sol/`) and prints each shape
as a **%SOL** column: the dispatch GFLOPS as a fraction of that shape's roofline
(min of the FMA peak and bandwidth × arithmetic intensity), with a `compute`/`bw`
bound tag. The linalg ratio keeps pointing at the wrong work, because its
denominator drifts with the nightly and the host: a shape at 0.99 vs linalg can
be at 92% of SOL (at the wall, nothing to win) while a shape that WINS vs linalg
sits at 63% of SOL (the real gap). Use %SOL to pick what to work on and to state
efficiency, and the ratio + 2σ verdict only to gate a regression. A bandwidth-
bound shape reading over 100% of SOL means the harness is holding its operand
cache-hot across reps versus the cold-DRAM roofline (see SOL.md idea 5). The SOL
numbers are re-measured every run, so they transfer across machines and
nightlies; the tables in SOL.md are from specific boots and will not match yours.

## Style preferences

- When providing links to the user, always use raw plain text. Do not wrap them in markdown bold (`**`) or other formatting
- Write code comments and docstrings in plain human language. Avoid contrastive negations (the "not X, but Y" or "X, not Y" shape). Say the thing you mean directly. Avoid em-dashes; use a period, comma, or parentheses instead

## Creating a Pull Request

Use the GitHub REST API with `$GH_TOKEN` (available in the environment) to create PRs directly:

```bash
curl -s -X POST "https://api.github.com/repos/tboerstad/agentic_matmul/pulls" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GH_TOKEN" \
  -d '{
  "title": "PR title here",
  "head": "claude/your-branch-name",
  "base": "main",
  "body": "## Summary\n- Description of changes\n\n## Test plan\n- [ ] Testing steps"
}'
```

The response JSON includes `html_url` — provide that link to the user.

For a quick comparison link (no PR creation), use:

```
https://github.com/tboerstad/agentic_matmul/compare/main...claude/your-branch-name
```
