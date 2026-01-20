#!/bin/bash

# ========================================
# Quick Test Script (No SLURM)
# Tests the training script locally with small settings
# Usage: bash scripts/alignment/jobs/test_quick.sh [single|sequence]
# ========================================

MODE=${1:-"sequence"}

echo "=========================================="
echo "Quick Test Mode: $MODE"
echo "=========================================="

# Set paths
SCRIPT_DIR="/home/user/Text2CT"
BRAIN_LATENT_DIR="/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent"
GENE_EMBEDDING_DIR="/scratch/connectome/mieuxmin/Brain_Gene_FM"
OUTPUT_DIR="./outputs/test_${MODE}"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Change to script directory
cd $SCRIPT_DIR

echo ""
echo "Testing with minimal settings..."
echo "This will train for only 2 epochs"
echo ""

if [ "$MODE" = "single" ]; then
    # Single gene test
    python scripts/alignment/train_multi_gene_alignment.py \
        --mode single \
        --single_gene_name APOE \
        --brain_latent_dir $BRAIN_LATENT_DIR \
        --gene_embedding_dir $GENE_EMBEDDING_DIR \
        --output_dir $OUTPUT_DIR \
        --batch_size 16 \
        --num_epochs 2 \
        --learning_rate 1e-4 \
        --projection_dim 256 \
        --val_split 0.2 \
        --num_workers 2 \
        --save_every 1

elif [ "$MODE" = "sequence" ]; then
    # Sequence test
    python scripts/alignment/train_multi_gene_alignment.py \
        --mode sequence \
        --brain_latent_dir $BRAIN_LATENT_DIR \
        --gene_embedding_dir $GENE_EMBEDDING_DIR \
        --output_dir $OUTPUT_DIR \
        --batch_size 8 \
        --num_epochs 2 \
        --learning_rate 1e-4 \
        --projection_dim 256 \
        --transformer_hidden_dim 256 \
        --transformer_num_layers 2 \
        --transformer_num_heads 4 \
        --transformer_pooling mean \
        --val_split 0.2 \
        --num_workers 2 \
        --save_every 1

else
    echo "Error: Invalid mode. Use 'single' or 'sequence'"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test completed!"
echo "If this works, you can submit the full job with sbatch"
echo "=========================================="
