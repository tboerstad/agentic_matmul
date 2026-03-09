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

# Create venv and install mojo nightly
uv venv
source .venv/bin/activate
uv pip install modular --index https://whl.modular.com/nightly/simple/ --prerelease allow

echo ""
echo "Setup complete! To get started:"
echo "  source .venv/bin/activate"
echo "  mojo main.mojo"
