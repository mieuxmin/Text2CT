#!/bin/bash

# Example training script for neuroimaging-gene alignment

# Default settings
GENE_EMBEDDING_PATH="/path/to/gene_embeddings.csv"
OUTPUT_DIR="./outputs/neuro_gene_alignment"
BATCH_SIZE=128
NUM_EPOCHS=100
LEARNING_RATE=1e-4
PROJECTION_DIM=512
HIDDEN_DIM=1024

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gene_embedding_path)
            GENE_EMBEDDING_PATH="$2"
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
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --projection_dim)
            PROJECTION_DIM="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if gene embedding path is provided
if [ "$GENE_EMBEDDING_PATH" = "/path/to/gene_embeddings.csv" ]; then
    echo "Error: Please provide --gene_embedding_path"
    echo "Usage: $0 --gene_embedding_path /path/to/gene_embeddings.csv [OPTIONS]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python scripts/alignment/train_neuro_gene_alignment.py \
    --gene_embedding_path "$GENE_EMBEDDING_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --projection_dim "$PROJECTION_DIM" \
    --hidden_dim "$HIDDEN_DIM" \
    --val_split 0.1 \
    --num_workers 4 \
    --save_every 10

echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
