"""
Model for aligning neuroimaging and gene embeddings using contrastive learning.
Uses small projection layers to map each modality to a shared embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NeuroGeneAlignmentModel(nn.Module):
    """
    Dual-encoder model for aligning neuroimaging and gene embeddings.

    Uses simple projection layers to map each modality to a shared embedding space,
    then applies contrastive learning (CLIP-style) for alignment.

    Args:
        neuro_input_dim: Dimension of flattened neuroimaging latent (default: 12150 = 3*15*18*15)
        gene_input_dim: Dimension of gene embeddings (default: 256)
        projection_dim: Dimension of shared embedding space (default: 512)
        hidden_dim: Optional hidden layer dimension for projections (None = direct projection)
        dropout: Dropout rate for projection layers (default: 0.1)
        logit_scale_init: Initial value for learnable temperature (default: 2.6592 = ln(1/0.07))
    """

    def __init__(
        self,
        neuro_input_dim: int = 12150,
        gene_input_dim: int = 256,
        projection_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        logit_scale_init: float = 2.6592,  # ln(1/0.07) as in CLIP
    ):
        super().__init__()

        self.neuro_input_dim = neuro_input_dim
        self.gene_input_dim = gene_input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim

        # Neuroimaging projection network
        if hidden_dim is not None:
            # Two-layer MLP with hidden layer
            self.neuro_projection = nn.Sequential(
                nn.Linear(neuro_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, projection_dim),
            )
        else:
            # Direct linear projection
            self.neuro_projection = nn.Linear(neuro_input_dim, projection_dim, bias=False)

        # Gene projection network
        if hidden_dim is not None:
            # Two-layer MLP with hidden layer
            self.gene_projection = nn.Sequential(
                nn.Linear(gene_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, projection_dim),
            )
        else:
            # Direct linear projection
            self.gene_projection = nn.Linear(gene_input_dim, projection_dim, bias=False)

        # Learnable temperature parameter (as in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        # Initialize weights
        self._init_weights()

        logger.info(f"Initialized NeuroGeneAlignmentModel:")
        logger.info(f"  Neuro input dim: {neuro_input_dim}")
        logger.info(f"  Gene input dim: {gene_input_dim}")
        logger.info(f"  Projection dim: {projection_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode_neuro(self, neuro_embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode neuroimaging embeddings to shared space.

        Args:
            neuro_embeddings: Tensor of shape (batch_size, neuro_input_dim)
            normalize: Whether to L2-normalize the output (default: True)

        Returns:
            Tensor of shape (batch_size, projection_dim)
        """
        projected = self.neuro_projection(neuro_embeddings)

        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected

    def encode_gene(self, gene_embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode gene embeddings to shared space.

        Args:
            gene_embeddings: Tensor of shape (batch_size, gene_input_dim)
            normalize: Whether to L2-normalize the output (default: True)

        Returns:
            Tensor of shape (batch_size, projection_dim)
        """
        projected = self.gene_projection(gene_embeddings)

        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected

    def forward(
        self,
        neuro_embeddings: torch.Tensor,
        gene_embeddings: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive loss computation.

        Args:
            neuro_embeddings: Tensor of shape (batch_size, neuro_input_dim)
            gene_embeddings: Tensor of shape (batch_size, gene_input_dim)
            return_loss: Whether to compute and return contrastive loss

        Returns:
            Dictionary containing:
                - neuro_features: Normalized neuro projections (batch_size, projection_dim)
                - gene_features: Normalized gene projections (batch_size, projection_dim)
                - logits_per_neuro: Similarity matrix (batch_size, batch_size)
                - logits_per_gene: Similarity matrix (batch_size, batch_size)
                - loss: Contrastive loss (if return_loss=True)
                - logit_scale: Current temperature value
        """
        # Encode and normalize
        neuro_features = self.encode_neuro(neuro_embeddings, normalize=True)
        gene_features = self.encode_gene(gene_embeddings, normalize=True)

        # Compute similarity with learnable temperature
        # Clamp logit_scale to prevent instability (as in CLIP)
        logit_scale = self.logit_scale.exp().clamp(max=100)

        # Compute cosine similarity scaled by temperature
        # Shape: (batch_size, batch_size)
        logits_per_neuro = logit_scale * neuro_features @ gene_features.t()
        logits_per_gene = logits_per_neuro.t()

        outputs = {
            'neuro_features': neuro_features,
            'gene_features': gene_features,
            'logits_per_neuro': logits_per_neuro,
            'logits_per_gene': logits_per_gene,
            'logit_scale': logit_scale,
        }

        # Compute contrastive loss
        if return_loss:
            loss = self.contrastive_loss(logits_per_neuro, logits_per_gene)
            outputs['loss'] = loss

        return outputs

    @staticmethod
    def contrastive_loss(
        logits_per_neuro: torch.Tensor,
        logits_per_gene: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symmetric contrastive loss (CLIP-style).

        Args:
            logits_per_neuro: Similarity matrix (batch_size, batch_size)
            logits_per_gene: Similarity matrix (batch_size, batch_size)

        Returns:
            Scalar loss value
        """
        batch_size = logits_per_neuro.shape[0]
        labels = torch.arange(batch_size, device=logits_per_neuro.device)

        # Symmetric cross-entropy loss
        loss_neuro = F.cross_entropy(logits_per_neuro, labels)
        loss_gene = F.cross_entropy(logits_per_gene, labels)

        loss = (loss_neuro + loss_gene) / 2.0

        return loss

    def get_similarity_matrix(
        self,
        neuro_embeddings: torch.Tensor,
        gene_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between neuroimaging and gene embeddings.

        Args:
            neuro_embeddings: Tensor of shape (N, neuro_input_dim)
            gene_embeddings: Tensor of shape (M, gene_input_dim)

        Returns:
            Similarity matrix of shape (N, M)
        """
        with torch.no_grad():
            neuro_features = self.encode_neuro(neuro_embeddings, normalize=True)
            gene_features = self.encode_gene(gene_embeddings, normalize=True)

            logit_scale = self.logit_scale.exp().clamp(max=100)
            similarity = logit_scale * neuro_features @ gene_features.t()

        return similarity

    def save_embeddings(
        self,
        dataloader,
        device: str = 'cuda',
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Extract and save all embeddings from a dataloader.

        Args:
            dataloader: DataLoader with neuro and gene embeddings
            device: Device to run inference on

        Returns:
            (neuro_embeddings, gene_embeddings, iids)
        """
        self.eval()
        all_neuro_features = []
        all_gene_features = []
        all_iids = []

        with torch.no_grad():
            for batch in dataloader:
                neuro_emb = batch['neuro_embedding'].to(device)
                gene_emb = batch['gene_embedding'].to(device)
                iids = batch['iid']

                neuro_feat = self.encode_neuro(neuro_emb, normalize=True)
                gene_feat = self.encode_gene(gene_emb, normalize=True)

                all_neuro_features.append(neuro_feat.cpu())
                all_gene_features.append(gene_feat.cpu())
                all_iids.extend(iids)

        all_neuro_features = torch.cat(all_neuro_features, dim=0)
        all_gene_features = torch.cat(all_gene_features, dim=0)

        return all_neuro_features, all_gene_features, all_iids
