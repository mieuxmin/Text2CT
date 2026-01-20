"""
Dataset class for neuroimaging-gene embedding alignment.
Loads paired neuroimaging latents and gene embeddings by IID.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NeuroGeneDataset(Dataset):
    """
    Dataset for loading paired neuroimaging latents and gene embeddings.

    Args:
        brain_latent_dir: Directory containing {IID}_latent.npz files
        gene_embedding_path: Path to CSV/parquet file with IID and gene embeddings
        iid_column: Name of the IID column in gene embedding file (default: 'IID')
        gene_embedding_dim: Expected dimension of gene embeddings (default: 256)
        transform: Optional transform to apply to neuroimaging latents
    """

    def __init__(
        self,
        brain_latent_dir: str,
        gene_embedding_path: str,
        iid_column: str = 'IID',
        gene_embedding_dim: int = 256,
        transform=None,
    ):
        super().__init__()

        self.brain_latent_dir = brain_latent_dir
        self.gene_embedding_path = gene_embedding_path
        self.iid_column = iid_column
        self.gene_embedding_dim = gene_embedding_dim
        self.transform = transform

        # Load gene embeddings
        logger.info(f"Loading gene embeddings from {gene_embedding_path}")
        if gene_embedding_path.endswith('.parquet'):
            self.gene_df = pd.read_parquet(gene_embedding_path)
        elif gene_embedding_path.endswith('.csv'):
            self.gene_df = pd.read_csv(gene_embedding_path)
        else:
            raise ValueError(f"Unsupported file format: {gene_embedding_path}")

        # Verify IID column exists
        if iid_column not in self.gene_df.columns:
            raise ValueError(f"Column '{iid_column}' not found in gene embedding file")

        # Get embedding columns (all columns except IID)
        self.embedding_columns = [col for col in self.gene_df.columns if col != iid_column]

        if len(self.embedding_columns) != gene_embedding_dim:
            logger.warning(
                f"Expected {gene_embedding_dim} embedding dimensions, "
                f"but found {len(self.embedding_columns)} columns"
            )

        # Find paired samples (IIDs that have both brain latent and gene embedding)
        available_brain_iids = set()
        for fname in os.listdir(brain_latent_dir):
            if fname.endswith('_latent.npz'):
                iid = fname.replace('_latent.npz', '')
                available_brain_iids.add(iid)

        available_gene_iids = set(self.gene_df[iid_column].astype(str).values)

        # Find intersection
        self.paired_iids = sorted(list(available_brain_iids & available_gene_iids))

        logger.info(f"Found {len(available_brain_iids)} brain latents")
        logger.info(f"Found {len(available_gene_iids)} gene embeddings")
        logger.info(f"Found {len(self.paired_iids)} paired samples")

        if len(self.paired_iids) == 0:
            raise ValueError("No paired samples found! Check IID matching.")

        # Create gene embedding lookup dict for faster access
        self.gene_df[iid_column] = self.gene_df[iid_column].astype(str)
        self.gene_df = self.gene_df.set_index(iid_column)

    def __len__(self) -> int:
        return len(self.paired_iids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - 'neuro_embedding': flattened neuroimaging latent, shape (12150,)
                - 'gene_embedding': gene embedding, shape (gene_embedding_dim,)
                - 'iid': subject ID
        """
        iid = self.paired_iids[idx]

        # Load neuroimaging latent
        brain_latent_path = os.path.join(self.brain_latent_dir, f"{iid}_latent.npz")

        try:
            with np.load(brain_latent_path) as data:
                # Assuming the latent is stored with key 'latent' or 'arr_0'
                if 'latent' in data:
                    brain_latent = data['latent']
                elif 'arr_0' in data:
                    brain_latent = data['arr_0']
                else:
                    # Use the first array in the npz file
                    brain_latent = data[list(data.keys())[0]]

            # Expected shape: (3, 15, 18, 15)
            # Flatten to (12150,)
            brain_latent_flat = brain_latent.flatten()

            # Apply transform if provided
            if self.transform is not None:
                brain_latent_flat = self.transform(brain_latent_flat)

            # Convert to tensor
            neuro_embedding = torch.from_numpy(brain_latent_flat).float()

        except Exception as e:
            logger.error(f"Error loading brain latent for IID {iid}: {e}")
            raise

        # Load gene embedding
        try:
            gene_row = self.gene_df.loc[iid, self.embedding_columns]
            gene_embedding = torch.from_numpy(gene_row.values.astype(np.float32))
        except Exception as e:
            logger.error(f"Error loading gene embedding for IID {iid}: {e}")
            raise

        return {
            'neuro_embedding': neuro_embedding,
            'gene_embedding': gene_embedding,
            'iid': iid
        }

    def get_embedding_dims(self) -> Tuple[int, int]:
        """
        Returns:
            (neuro_dim, gene_dim) - dimensions of neuroimaging and gene embeddings
        """
        # Neuroimaging: 3 * 15 * 18 * 15 = 12150
        neuro_dim = 3 * 15 * 18 * 15
        gene_dim = len(self.embedding_columns)
        return neuro_dim, gene_dim
