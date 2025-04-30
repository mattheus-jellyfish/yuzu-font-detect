#!/bin/bash

# Install uv if not already installed
echo "Checking for uv installation..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell configuration to make uv available
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
    elif [ -f ~/.zshrc ]; then
        source ~/.zshrc
    fi
fi

# Create a virtual environment
echo "Creating virtual environment..."
uv venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
uv pip install -e .

# If on L4 GPU, install specific CUDA versions from wheels
if [ "$1" == "--l4gpu" ]; then
    echo "Installing L4 GPU specific packages..."
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi

echo "Virtual environment setup complete!"
echo "You can activate the environment with:"
echo "    source .venv/bin/activate"
echo ""
echo "If you're using this on an L4 GPU, run with:"
echo "    ./setup_venv.sh --l4gpu" 