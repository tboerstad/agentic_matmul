#!/bin/bash
set -euo pipefail

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Initialize project if needed
if [ ! -f pyproject.toml ]; then
    uv init
fi

# Create venv and install the Mojo nightly the code targets (MAX 26.5 →
# Mojo 1.0.0b3). Pinned: the 26.6 nightlies (Mojo 1.1.0rc0) removed the
# `read` capture keyword, `ImmutOrigin`, and moved `parallelize`/`alloc`,
# so they no longer compile this repo. Bump the pin together with a port.
uv venv
source .venv/bin/activate
uv pip install "modular==26.5.0.dev2026070806" --index https://whl.modular.com/nightly/simple/ --prerelease allow

# Install the Python benchmark deps (numpy/scipy/mkl) so `python bench/sota.py`
# works out of the box, including the Intel MKL dgemm comparison.
uv pip install numpy scipy mkl

echo ""
echo "Setup complete! To get started:"
echo "  source .venv/bin/activate"
echo "  mojo -I . examples/demo.mojo"
echo "  python bench/sota.py   # NumPy / SciPy / MKL benchmarks"
