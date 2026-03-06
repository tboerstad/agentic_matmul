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

## Development

- Mojo source files use the `.mojo` extension
- The Mojo compiler and runtime are installed in `.venv/` via `uv`
- Always activate the venv before running `mojo` commands

## Style preferences

- When providing links to the user, always use raw plain text — never wrap in markdown bold (`**`) or other formatting

## Creating a Pull Request

To create a PR link without needing `gh` CLI, use the GitHub compare URL format:

```
https://github.com/[owner]/[repo]/compare/[base-branch]...[feature-branch]
```

Example:
```
https://github.com/tboerstad/agentic_matmul/compare/main...claude/update-a-matrix-constants-b1TaV
```

This link allows you to review all changes between branches and create a PR directly from GitHub.
