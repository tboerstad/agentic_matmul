# Agent Instructions

## First-time setup after cloning

1. Run `bash setup.sh` to install dependencies (uv, Mojo nightly)
2. Activate the venv: `source .venv/bin/activate`
3. Verify the setup works: `mojo main.mojo`

## Mojo 101

- Import from `std`: `from std.collections import List`
- Use `SIMD[DType.float64, N]` for fixed-size arrays
- Declare functions with `fn`, variables with `var`
- Return owned `List` values with `^` (e.g. `return result^`); without it the compiler rejects the implicit copy
- Use `mut` (not `inout`) for mutable function parameters: `fn foo(mut x: List[Float64])`
- Use `comptime if` (not `@parameter if`) for compile-time branching on parameter values
- Inside a struct, reference its parameters with `Self.` prefix: `List[Scalar[Self.dtype]]`, not `List[Scalar[dtype]]`
- Use `vectorize[simd_width](size, closure)` from `std.algorithm.functional` to auto-vectorize loops with automatic remainder handling — no manual SIMD + scalar tail loop needed
- Closures passed to `vectorize` use the `unified {mut}` syntax: `fn name[width: Int](i: Int) unified {mut}:` — `unified` means it works in both parametric and runtime contexts, `{mut}` allows capturing and mutating enclosing variables
- Use `parallelize[func](num_work_items, num_workers)` from `std.algorithm.functional` to distribute work across threads — the closure must use `capturing` (not `unified {mut}`): `fn worker(i: Int) capturing:`
- Use `InlineArray[T, N]` (from `std.collections`) + `comptime for` to replace hand-numbered variables (e.g. `acc0`–`acc3`). The compiler flattens comptime-indexed `InlineArray` elements into registers, producing identical machine code to manual variables — but the code scales when you change tile sizes like MR/NR/KU

## Development

- Mojo source files use the `.mojo` extension
- The Mojo compiler and runtime are installed in `.venv/` via `uv`
- Always activate the venv before running `mojo` commands

## Style preferences

- When providing links to the user, always use raw plain text — never wrap in markdown bold (`**`) or other formatting

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
