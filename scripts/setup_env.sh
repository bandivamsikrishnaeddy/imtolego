#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# Bootstrap: Create conda environment and validate imports
# --------------------------------------------------------------------------- #
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="imtolego"

cd "$PROJECT_ROOT"

echo "========================================"
echo "  ImToLego Environment Setup"
echo "========================================"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install conda/miniconda first."
    exit 1
fi

echo "Using conda: $(conda --version)"

# If env already exists, offer to remove it
if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
    echo "Environment '${ENV_NAME}' already exists."
    read -r -p "Remove and recreate? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Exiting. Activate existing env with: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "========================================"
echo "  Environment '${ENV_NAME}' created."
echo "  Activate with: conda activate ${ENV_NAME}"
echo "========================================"

echo ""
echo "Validating imports..."
conda run -n "$ENV_NAME" python -c "
import torch, torchvision, transformers, einops, omegaconf, trimesh, rembg
print('Core ML imports: OK')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import fastapi, uvicorn, structlog
print('Web API imports: OK')
print()
print('All checks passed. Ready to start!')
"

echo ""
echo "To start the API server:"
echo "  conda activate ${ENV_NAME}"
echo "  ./scripts/run.sh"
