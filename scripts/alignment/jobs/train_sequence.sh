#!/bin/bash
#SBATCH --job-name=neuro_gene_sequence
#SBATCH --output=logs/sequence_%j.out
#SBATCH --error=logs/sequence_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ========================================
# Multi-Gene Sequence Alignment Training
# Processes all 111 genes as a sequence using Transformer
# Usage: sbatch scripts/alignment/jobs/train_sequence.sh
# ========================================

echo "=========================================="
echo "Multi-Gene Sequence Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Set paths
SCRIPT_DIR="/home/user/Text2CT"
BRAIN_LATENT_DIR="/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent"
GENE_EMBEDDING_DIR="/scratch/connectome/mieuxmin/Brain_Gene_FM"
OUTPUT_DIR="./outputs/multi_gene_sequence"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Change to script directory
cd $SCRIPT_DIR

# Activate conda environment (if needed)
# source ~/.bashrc
# conda activate your_env

# Print GPU info
nvidia-smi

echo ""
echo "Starting multi-gene sequence training"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training with transformer
python scripts/alignment/train_multi_gene_alignment.py \
    --mode sequence \
    --brain_latent_dir $BRAIN_LATENT_DIR \
    --gene_embedding_dir $GENE_EMBEDDING_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --projection_dim 512 \
    --transformer_hidden_dim 512 \
    --transformer_num_layers 4 \
    --transformer_num_heads 8 \
    --transformer_pooling mean \
    --val_split 0.1 \
    --num_workers 8 \
    --save_every 10 \
    --seed 42

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
