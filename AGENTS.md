# Agent Instructions

## First-time setup after cloning

1. Run `bash setup.sh` to install dependencies (uv, Mojo nightly)
2. Activate the venv: `source .venv/bin/activate`
3. Verify the setup works: `mojo main.mojo`

## Mojo 101

- Import from `std`: `from std.collections import List`
- Use `SIMD[DType.float64, N]` for fixed-size arrays
- Declare functions with `fn`, variables with `var`

## Development

- Mojo source files use the `.mojo` extension
- The Mojo compiler and runtime are installed in `.venv/` via `uv`
- Always activate the venv before running `mojo` commands

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
