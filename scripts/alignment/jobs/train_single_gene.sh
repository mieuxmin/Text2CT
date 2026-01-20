#!/bin/bash
#SBATCH --job-name=neuro_gene_single
#SBATCH --output=logs/single_gene_%j.out
#SBATCH --error=logs/single_gene_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ========================================
# Single Gene Alignment Training
# Usage: sbatch scripts/alignment/jobs/train_single_gene.sh GENE_NAME
# Example: sbatch scripts/alignment/jobs/train_single_gene.sh APOE
# ========================================

# Get gene name from command line argument
GENE_NAME=${1:-"APOE"}

echo "=========================================="
echo "Single Gene Training: $GENE_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Set paths
SCRIPT_DIR="/home/user/Text2CT"
BRAIN_LATENT_DIR="/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent"
GENE_EMBEDDING_DIR="/scratch/connectome/mieuxmin/Brain_Gene_FM"
OUTPUT_DIR="./outputs/single_gene/${GENE_NAME}"

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
echo "Starting training for gene: $GENE_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python scripts/alignment/train_multi_gene_alignment.py \
    --mode single \
    --single_gene_name $GENE_NAME \
    --brain_latent_dir $BRAIN_LATENT_DIR \
    --gene_embedding_dir $GENE_EMBEDDING_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --projection_dim 512 \
    --val_split 0.1 \
    --num_workers 8 \
    --save_every 10 \
    --seed 42

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
