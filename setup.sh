#!/bin/bash
set -euo pipefail

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install gh CLI from precompiled binary
if ! command -v gh &> /dev/null; then
    echo "Installing gh CLI..."
    GH_VERSION="2.87.3"
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        GH_ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ]; then
        GH_ARCH="arm64"
    else
        GH_ARCH=$ARCH
    fi
    GH_URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${GH_ARCH}.tar.gz"
    GH_TEMP=$(mktemp -d)
    curl -fsSL "$GH_URL" -o "$GH_TEMP/gh.tar.gz"
    tar -xzf "$GH_TEMP/gh.tar.gz" -C "$GH_TEMP"
    sudo mv "$GH_TEMP/gh_${GH_VERSION}_linux_${GH_ARCH}/bin/gh" /usr/local/bin/gh
    sudo chmod +x /usr/local/bin/gh
    rm -rf "$GH_TEMP"
    echo "gh CLI installed successfully"
fi

# Initialize project if needed
if [ ! -f pyproject.toml ]; then
    uv init
fi

# Create venv and install mojo nightly
uv venv
source .venv/bin/activate
uv pip install mojo --index https://whl.modular.com/nightly/simple/ --prerelease allow

echo ""
echo "Setup complete! To get started:"
echo "  source .venv/bin/activate"
echo "  mojo main.mojo"
