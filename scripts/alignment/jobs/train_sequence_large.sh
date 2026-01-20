#!/bin/bash
#SBATCH --job-name=neuro_gene_seq_large
#SBATCH --output=logs/sequence_large_%j.out
#SBATCH --error=logs/sequence_large_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# ========================================
# Large Multi-Gene Sequence Training
# Uses larger transformer (6 layers, 12 heads, 768 hidden dim)
# Requires more GPU memory
# Usage: sbatch scripts/alignment/jobs/train_sequence_large.sh
# ========================================

echo "=========================================="
echo "Large Multi-Gene Sequence Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "=========================================="

# Set paths
SCRIPT_DIR="/home/user/Text2CT"
BRAIN_LATENT_DIR="/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent"
GENE_EMBEDDING_DIR="/scratch/connectome/mieuxmin/Brain_Gene_FM"
OUTPUT_DIR="./outputs/multi_gene_sequence_large"

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
echo "Starting large transformer training"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training with larger transformer
python scripts/alignment/train_multi_gene_alignment.py \
    --mode sequence \
    --brain_latent_dir $BRAIN_LATENT_DIR \
    --gene_embedding_dir $GENE_EMBEDDING_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --num_epochs 150 \
    --learning_rate 5e-5 \
    --projection_dim 768 \
    --transformer_hidden_dim 768 \
    --transformer_num_layers 6 \
    --transformer_num_heads 12 \
    --transformer_pooling cls \
    --neuro_hidden_dim 1024 \
    --val_split 0.1 \
    --num_workers 16 \
    --save_every 10 \
    --seed 42

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
