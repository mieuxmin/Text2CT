#!/bin/bash

# ========================================
# Train all genes individually (single mode)
# Submits multiple jobs, one per gene
# Usage: bash scripts/alignment/jobs/train_all_genes.sh
# ========================================

SCRIPT_DIR="/home/user/Text2CT"
GENE_EMBEDDING_DIR="/scratch/connectome/mieuxmin/Brain_Gene_FM"

cd $SCRIPT_DIR

echo "=========================================="
echo "Submitting jobs for all genes"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Get list of all genes from the directory
GENE_FILES=$(ls ${GENE_EMBEDDING_DIR}/*_brain_gene_embeddingUKB.csv)

JOB_COUNT=0

for GENE_FILE in $GENE_FILES; do
    # Extract gene name from filename
    BASENAME=$(basename $GENE_FILE)
    GENE_NAME=${BASENAME/_brain_gene_embeddingUKB.csv/}

    echo "Submitting job for gene: $GENE_NAME"

    # Submit job
    sbatch scripts/alignment/jobs/train_single_gene.sh $GENE_NAME

    JOB_COUNT=$((JOB_COUNT + 1))

    # Optional: Add delay to avoid overwhelming scheduler
    sleep 1
done

echo ""
echo "=========================================="
echo "Submitted $JOB_COUNT jobs"
echo "Check status with: squeue -u $USER"
echo "=========================================="
