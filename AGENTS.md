# Agent Instructions

## First-time setup after cloning

1. Run `bash setup.sh` to install dependencies (uv, Mojo nightly)
2. Activate the venv: `source .venv/bin/activate`
3. Verify the setup works: `mojo main.mojo`

## Mojo 101

`StaticTuple` and `InlineArray` are not available in nightly builds — use `SIMD[DType.float64, N]` as a fixed-size array of floats, which supports indexing and arithmetic out of the box. Standard library imports like `from collections import List` still work but emit a deprecation warning; prefix with `std.` instead (e.g. `from std.collections import List`). Mojo functions are declared with `fn` and variables with `var`, and the type system is strict — mismatched or unavailable types produce compile errors, not runtime ones.

## Development

- Mojo source files use the `.mojo` extension
- The Mojo compiler and runtime are installed in `.venv/` via `uv`
- Always activate the venv before running `mojo` commands
