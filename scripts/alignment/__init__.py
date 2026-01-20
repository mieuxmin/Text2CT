"""
Neuroimaging-Gene Embedding Alignment Module

This module provides tools for aligning neuroimaging and gene embeddings
using contrastive learning (CLIP-style).

Features:
- Single gene alignment: Simple projection-based alignment
- Multi-gene alignment: Transformer-based sequence encoding
"""

from .neuro_gene_dataset import NeuroGeneDataset
from .neuro_gene_model import NeuroGeneAlignmentModel
from .multi_gene_dataset import MultiGeneNeuroDataset
from .multi_gene_model import MultiGeneAlignmentModel, GeneSequenceEncoder

__all__ = [
    'NeuroGeneDataset',
    'NeuroGeneAlignmentModel',
    'MultiGeneNeuroDataset',
    'MultiGeneAlignmentModel',
    'GeneSequenceEncoder',
]
