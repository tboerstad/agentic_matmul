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

**All benchmark numbers in SOTA_RESULTS.md were measured on specific machines — see the Hardware
Configurations table there for details (Machine A: Intel Xeon @ 2.80 GHz Skylake; Machine B:
Intel Xeon @ 2.10 GHz Granite Rapids; both 4 cores, AVX-512, KVM). Your results WILL differ.**

## First-time setup after cloning

1. Run `bash setup.sh` to install dependencies (uv, latest Mojo nightly — MAX 26.5 → Mojo 1.0.0b3)
2. Activate the venv: `source .venv/bin/activate`
3. Verify the setup works: `mojo main.mojo`

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
