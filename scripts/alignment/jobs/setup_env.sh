#!/bin/bash

# ========================================
# Environment Setup for Training
# Sets up Python environment and dependencies
# ========================================

echo "=========================================="
echo "Setting up environment for training"
echo "=========================================="

# Option 1: Using conda (recommended)
if command -v conda &> /dev/null; then
    echo "Found conda, creating environment..."

    # Create conda environment
    conda create -n neuro_gene python=3.10 -y

    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate neuro_gene

    # Install PyTorch (adjust CUDA version as needed)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

    # Install other dependencies
    pip install -r scripts/alignment/requirements.txt

    echo ""
    echo "Conda environment 'neuro_gene' created!"
    echo "Activate with: conda activate neuro_gene"

# Option 2: Using virtualenv
elif command -v python3 &> /dev/null; then
    echo "Using virtualenv..."

    # Create virtual environment
    python3 -m venv venv_neuro_gene

    # Activate environment
    source venv_neuro_gene/bin/activate

    # Install dependencies
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r scripts/alignment/requirements.txt

    echo ""
    echo "Virtual environment created!"
    echo "Activate with: source venv_neuro_gene/bin/activate"

else
    echo "Error: Neither conda nor python3 found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Environment setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment"
echo "2. Test import: python -c 'from scripts.alignment import MultiGeneAlignmentModel'"
echo "3. Submit a job: sbatch scripts/alignment/jobs/train_sequence.sh"
echo ""
