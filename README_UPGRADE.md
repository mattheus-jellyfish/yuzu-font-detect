# YuzuMarker.FontDetection - Dependency Management

This document provides instructions for managing dependencies in the YuzuMarker.FontDetection project.

## Using uv sync

The `uv` tool is used for fast, reliable Python package management. Here are common commands for keeping your environment in sync:

### Basic Sync

To sync your environment with all dependencies defined in pyproject.toml:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Sync all dependencies
uv sync
```

### L4 GPU-specific Sync

For synchronizing with specific CUDA packages for L4 GPU:

```bash
# Create a requirements file for L4 GPU
echo "torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121" > requirements-l4gpu.txt
echo "torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121" >> requirements-l4gpu.txt

# Sync with L4 GPU requirements
uv sync -r requirements-l4gpu.txt
```

### Additional Commands

```bash
# Sync only from pyproject.toml
uv pip sync

# Sync from multiple requirements files
uv sync -r requirements.txt -r requirements-dev.txt

# Upgrade all packages to their latest versions within constraints
uv sync --upgrade

# Install development dependencies
uv sync -r requirements-dev.txt
```

## Setting Up a New Environment

To set up a new environment from scratch:

```bash
# Create and set up environment with standard dependencies
./setup_venv.sh

# For L4 GPU with CUDA 12.1 support
./setup_venv.sh --l4gpu
```

# Running model training
```bash
python train.py -m resnet18 -p -b 128 -i -a v3 -n resnet18_$(date +"%y%m%d_%H%M")_135_fonts_128_batch -f
```
