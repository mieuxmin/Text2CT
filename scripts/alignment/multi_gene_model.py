"""
Models for aligning neuroimaging with single or multiple gene embeddings.
Supports both simple projection and transformer-based sequence encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class GeneSequenceEncoder(nn.Module):
    """
    Transformer-based encoder for gene sequences.
    Processes (num_genes, gene_dim) sequences.
    """

    def __init__(
        self,
        gene_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling: str = 'mean',  # 'mean', 'max', or 'cls'
    ):
        super().__init__()

        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Input projection
        self.input_projection = nn.Linear(gene_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # CLS token (if using cls pooling)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"Initialized GeneSequenceEncoder:")
        logger.info(f"  Gene dim: {gene_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Heads: {num_heads}")
        logger.info(f"  Pooling: {pooling}")

    def forward(self, gene_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_sequence: (batch_size, num_genes, gene_dim)

        Returns:
            (batch_size, hidden_dim)
        """
        batch_size, num_genes, gene_dim = gene_sequence.shape

        # Project to hidden dimension
        x = self.input_projection(gene_sequence)  # (B, N, H)

        # Add CLS token if needed
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, H)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, H)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, N, H) or (B, N+1, H)

        # Pooling
        if self.pooling == 'cls':
            # Use CLS token
            output = x[:, 0, :]  # (B, H)
        elif self.pooling == 'mean':
            # Mean pooling
            if self.pooling == 'cls':
                output = x[:, 1:, :].mean(dim=1)  # Exclude CLS
            else:
                output = x.mean(dim=1)  # (B, H)
        elif self.pooling == 'max':
            # Max pooling
            output = x.max(dim=1)[0]  # (B, H)
        else:
            raise ValueError(f"Invalid pooling: {self.pooling}")

        # Layer norm
        output = self.layer_norm(output)

        return output


class MultiGeneAlignmentModel(nn.Module):
    """
    Model for aligning neuroimaging with gene sequences.

    Supports two modes:
    1. Single gene: Simple projection layers
    2. Sequence: Transformer encoder + projection

    Args:
        neuro_input_dim: Neuroimaging dimension (default: 12150)
        gene_input_dim: Gene dimension (default: 256)
        num_genes: Number of genes in sequence (default: 1 for single mode)
        projection_dim: Shared embedding space dimension (default: 512)
        use_transformer: Use transformer for gene encoding (default: True if num_genes > 1)
        transformer_hidden_dim: Transformer hidden dimension (default: 512)
        transformer_num_layers: Number of transformer layers (default: 4)
        transformer_num_heads: Number of attention heads (default: 8)
        transformer_pooling: Pooling method ('mean', 'max', 'cls')
        neuro_hidden_dim: Hidden dimension for neuro projection (default: None)
        dropout: Dropout rate (default: 0.1)
        logit_scale_init: Initial temperature (default: 2.6592)
    """

    def __init__(
        self,
        neuro_input_dim: int = 12150,
        gene_input_dim: int = 256,
        num_genes: int = 1,
        projection_dim: int = 512,
        use_transformer: Optional[bool] = None,
        transformer_hidden_dim: int = 512,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 8,
        transformer_pooling: str = 'mean',
        neuro_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        logit_scale_init: float = 2.6592,
    ):
        super().__init__()

        self.neuro_input_dim = neuro_input_dim
        self.gene_input_dim = gene_input_dim
        self.num_genes = num_genes
        self.projection_dim = projection_dim

        # Decide whether to use transformer
        if use_transformer is None:
            use_transformer = num_genes > 1

        self.use_transformer = use_transformer

        # Neuroimaging projection
        if neuro_hidden_dim is not None:
            self.neuro_projection = nn.Sequential(
                nn.Linear(neuro_input_dim, neuro_hidden_dim),
                nn.LayerNorm(neuro_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(neuro_hidden_dim, projection_dim),
            )
        else:
            self.neuro_projection = nn.Linear(neuro_input_dim, projection_dim, bias=False)

        # Gene encoding
        if use_transformer:
            # Transformer encoder + projection
            self.gene_encoder = GeneSequenceEncoder(
                gene_dim=gene_input_dim,
                hidden_dim=transformer_hidden_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                dropout=dropout,
                pooling=transformer_pooling,
            )

            self.gene_projection = nn.Linear(
                transformer_hidden_dim, projection_dim, bias=False
            )

        else:
            # Simple projection for single gene
            self.gene_encoder = None
            self.gene_projection = nn.Linear(gene_input_dim, projection_dim, bias=False)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        # Initialize weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized MultiGeneAlignmentModel:")
        logger.info(f"  Neuro input dim: {neuro_input_dim}")
        logger.info(f"  Gene input dim: {gene_input_dim}")
        logger.info(f"  Num genes: {num_genes}")
        logger.info(f"  Projection dim: {projection_dim}")
        logger.info(f"  Use transformer: {use_transformer}")
        logger.info(f"  Total parameters: {total_params:,}")

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode_neuro(
        self, neuro_embeddings: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode neuroimaging embeddings.

        Args:
            neuro_embeddings: (batch_size, neuro_input_dim)
            normalize: Whether to L2-normalize

        Returns:
            (batch_size, projection_dim)
        """
        projected = self.neuro_projection(neuro_embeddings)

        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected

    def encode_gene(
        self,
        gene_input: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode gene embeddings.

        Args:
            gene_input:
                - If transformer: (batch_size, num_genes, gene_dim)
                - If no transformer: (batch_size, gene_dim)
            normalize: Whether to L2-normalize

        Returns:
            (batch_size, projection_dim)
        """
        if self.use_transformer:
            # gene_input: (B, num_genes, gene_dim)
            encoded = self.gene_encoder(gene_input)  # (B, hidden_dim)
            projected = self.gene_projection(encoded)  # (B, projection_dim)
        else:
            # gene_input: (B, gene_dim)
            projected = self.gene_projection(gene_input)  # (B, projection_dim)

        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected

    def forward(
        self,
        neuro_embeddings: torch.Tensor,
        gene_input: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive loss.

        Args:
            neuro_embeddings: (batch_size, neuro_input_dim)
            gene_input:
                - If transformer: (batch_size, num_genes, gene_dim)
                - If no transformer: (batch_size, gene_dim)
            return_loss: Whether to compute loss

        Returns:
            Dictionary with features, logits, and loss
        """
        # Encode and normalize
        neuro_features = self.encode_neuro(neuro_embeddings, normalize=True)
        gene_features = self.encode_gene(gene_input, normalize=True)

        # Compute similarity
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_neuro = logit_scale * neuro_features @ gene_features.t()
        logits_per_gene = logits_per_neuro.t()

        outputs = {
            'neuro_features': neuro_features,
            'gene_features': gene_features,
            'logits_per_neuro': logits_per_neuro,
            'logits_per_gene': logits_per_gene,
            'logit_scale': logit_scale,
        }

        if return_loss:
            loss = self.contrastive_loss(logits_per_neuro, logits_per_gene)
            outputs['loss'] = loss

        return outputs

    @staticmethod
    def contrastive_loss(
        logits_per_neuro: torch.Tensor,
        logits_per_gene: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric contrastive loss."""
        batch_size = logits_per_neuro.shape[0]
        labels = torch.arange(batch_size, device=logits_per_neuro.device)

        loss_neuro = F.cross_entropy(logits_per_neuro, labels)
        loss_gene = F.cross_entropy(logits_per_gene, labels)

        return (loss_neuro + loss_gene) / 2.0

    def get_similarity_matrix(
        self,
        neuro_embeddings: torch.Tensor,
        gene_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity matrix."""
        with torch.no_grad():
            neuro_features = self.encode_neuro(neuro_embeddings, normalize=True)
            gene_features = self.encode_gene(gene_input, normalize=True)

            logit_scale = self.logit_scale.exp().clamp(max=100)
            similarity = logit_scale * neuro_features @ gene_features.t()

        return similarity
