#!/bin/bash
# Run BEM-UNet Synapse web demo
# Uses the mamba23 conda environment (has torch 2.5.1 + CUDA)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null

conda activate mamba23

# Install Flask if missing
python -c "import flask" 2>/dev/null || pip install flask --quiet

# Run from project root so imports resolve correctly
cd "$SCRIPT_DIR/.."
echo "========================================"
echo "  BEM-UNet Synapse Demo"
echo "  http://localhost:5000"
echo "========================================"
python demo/app.py
