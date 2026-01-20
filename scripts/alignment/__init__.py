"""
Neuroimaging-Gene Embedding Alignment Module

This module provides tools for aligning neuroimaging and gene embeddings
using contrastive learning (CLIP-style).
"""

from .neuro_gene_dataset import NeuroGeneDataset
from .neuro_gene_model import NeuroGeneAlignmentModel

__all__ = [
    'NeuroGeneDataset',
    'NeuroGeneAlignmentModel',
]
