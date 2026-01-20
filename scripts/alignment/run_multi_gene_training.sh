#!/bin/bash

# Training script for multi-gene neuroimaging alignment

# Default settings
GENE_EMBEDDING_DIR="/scratch/connectome/mieuxmin/Brain_Gene_FM"
BRAIN_LATENT_DIR="/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent"
OUTPUT_DIR="./outputs/multi_gene_alignment"
MODE="sequence"
SINGLE_GENE_NAME=""

BATCH_SIZE=64
NUM_EPOCHS=100
LEARNING_RATE=1e-4
PROJECTION_DIM=512

# Transformer settings (for sequence mode)
TRANSFORMER_HIDDEN_DIM=512
TRANSFORMER_NUM_LAYERS=4
TRANSFORMER_NUM_HEADS=8
TRANSFORMER_POOLING="mean"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --single_gene_name)
            SINGLE_GENE_NAME="$2"
            shift 2
            ;;
        --gene_embedding_dir)
            GENE_EMBEDDING_DIR="$2"
            shift 2
            ;;
        --brain_latent_dir)
            BRAIN_LATENT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --transformer_hidden_dim)
            TRANSFORMER_HIDDEN_DIM="$2"
            shift 2
            ;;
        --transformer_num_layers)
            TRANSFORMER_NUM_LAYERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validation
if [ "$MODE" = "single" ] && [ -z "$SINGLE_GENE_NAME" ]; then
    echo "Error: --single_gene_name required for single mode"
    echo "Usage: $0 --mode single --single_gene_name APOE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Multi-Gene Neuroimaging Alignment"
echo "=========================================="
echo "Mode: $MODE"
if [ "$MODE" = "single" ]; then
    echo "Gene: $SINGLE_GENE_NAME"
fi
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "=========================================="

# Build command
CMD="python scripts/alignment/train_multi_gene_alignment.py \
    --mode $MODE \
    --gene_embedding_dir $GENE_EMBEDDING_DIR \
    --brain_latent_dir $BRAIN_LATENT_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --projection_dim $PROJECTION_DIM \
    --val_split 0.1 \
    --num_workers 4 \
    --save_every 10"

# Add single gene name if provided
if [ -n "$SINGLE_GENE_NAME" ]; then
    CMD="$CMD --single_gene_name $SINGLE_GENE_NAME"
fi

# Add transformer settings for sequence mode
if [ "$MODE" = "sequence" ]; then
    CMD="$CMD \
        --transformer_hidden_dim $TRANSFORMER_HIDDEN_DIM \
        --transformer_num_layers $TRANSFORMER_NUM_LAYERS \
        --transformer_num_heads $TRANSFORMER_NUM_HEADS \
        --transformer_pooling $TRANSFORMER_POOLING"
fi

# Run training
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
