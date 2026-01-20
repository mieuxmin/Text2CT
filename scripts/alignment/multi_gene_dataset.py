"""
Dataset class for multi-gene neuroimaging alignment.
Supports both single-gene and multi-gene sequence approaches.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class MultiGeneNeuroDataset(Dataset):
    """
    Dataset for loading paired neuroimaging latents and multiple gene embeddings.

    Supports two modes:
    1. Single gene mode: Returns one gene at a time
    2. Sequence mode: Returns all genes as a sequence (111 x 256)

    Args:
        brain_latent_dir: Directory containing {IID}_latent.npz files
        gene_embedding_dir: Directory containing {gene_name}_brain_gene_embeddingUKB.csv files
        iid_column: Name of the IID column (default: 'IID')
        gene_embedding_dim: Dimension per gene (default: 256)
        mode: 'single' or 'sequence' (default: 'sequence')
        single_gene_name: Gene name for single mode (required if mode='single')
    """

    def __init__(
        self,
        brain_latent_dir: str,
        gene_embedding_dir: str,
        iid_column: str = 'IID',
        gene_embedding_dim: int = 256,
        mode: str = 'sequence',
        single_gene_name: Optional[str] = None,
        transform=None,
    ):
        super().__init__()

        self.brain_latent_dir = brain_latent_dir
        self.gene_embedding_dir = gene_embedding_dir
        self.iid_column = iid_column
        self.gene_embedding_dim = gene_embedding_dim
        self.mode = mode
        self.single_gene_name = single_gene_name
        self.transform = transform

        if mode == 'single' and single_gene_name is None:
            raise ValueError("single_gene_name must be provided when mode='single'")

        # Find all gene embedding files
        gene_files = glob.glob(
            os.path.join(gene_embedding_dir, '*_brain_gene_embeddingUKB.csv')
        )

        if len(gene_files) == 0:
            raise ValueError(f"No gene embedding files found in {gene_embedding_dir}")

        logger.info(f"Found {len(gene_files)} gene embedding files")

        # Parse gene names
        self.gene_names = []
        self.gene_files = {}

        for fpath in gene_files:
            fname = os.path.basename(fpath)
            # Extract gene name: {gene_name}_brain_gene_embeddingUKB.csv
            gene_name = fname.replace('_brain_gene_embeddingUKB.csv', '')
            self.gene_names.append(gene_name)
            self.gene_files[gene_name] = fpath

        self.gene_names = sorted(self.gene_names)
        logger.info(f"Gene names: {', '.join(self.gene_names[:5])}... (showing first 5)")

        # Load gene embeddings based on mode
        if mode == 'single':
            if single_gene_name not in self.gene_files:
                raise ValueError(
                    f"Gene '{single_gene_name}' not found. "
                    f"Available genes: {', '.join(self.gene_names)}"
                )

            logger.info(f"Loading single gene: {single_gene_name}")
            self.gene_df = self._load_gene_file(self.gene_files[single_gene_name])

        elif mode == 'sequence':
            logger.info("Loading all genes for sequence mode...")
            self.gene_dfs = {}

            for gene_name in self.gene_names:
                df = self._load_gene_file(self.gene_files[gene_name])
                self.gene_dfs[gene_name] = df

            logger.info(f"Loaded {len(self.gene_dfs)} gene embeddings")

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'single' or 'sequence'")

        # Find paired samples
        available_brain_iids = set()
        for fname in os.listdir(brain_latent_dir):
            if fname.endswith('_latent.npz'):
                iid = fname.replace('_latent.npz', '')
                available_brain_iids.add(iid)

        # Get IIDs that have all gene embeddings
        if mode == 'single':
            # IID is now the index, not a column
            available_gene_iids = set(self.gene_df.index.astype(str).values)
        else:
            # Intersection of all genes
            available_gene_iids = None
            for gene_name, df in self.gene_dfs.items():
                # IID is now the index, not a column
                gene_iids = set(df.index.astype(str).values)
                if available_gene_iids is None:
                    available_gene_iids = gene_iids
                else:
                    available_gene_iids &= gene_iids

        # Find intersection with brain latents
        self.paired_iids = sorted(list(available_brain_iids & available_gene_iids))

        logger.info(f"Found {len(available_brain_iids)} brain latents")
        logger.info(f"Found {len(available_gene_iids)} gene embeddings (with all genes)")
        logger.info(f"Found {len(self.paired_iids)} paired samples")

        if len(self.paired_iids) == 0:
            raise ValueError("No paired samples found! Check IID matching.")

    def _load_gene_file(self, fpath: str) -> pd.DataFrame:
        """Load and process a single gene embedding file."""
        df = pd.read_csv(fpath)

        if self.iid_column not in df.columns:
            raise ValueError(f"Column '{self.iid_column}' not found in {fpath}")

        # Set IID as index for faster lookup
        df[self.iid_column] = df[self.iid_column].astype(str)
        df = df.set_index(self.iid_column)

        # Get embedding columns
        embedding_cols = [col for col in df.columns if col != self.iid_column]

        if len(embedding_cols) != self.gene_embedding_dim:
            logger.warning(
                f"Expected {self.gene_embedding_dim} dims, "
                f"but found {len(embedding_cols)} in {os.path.basename(fpath)}"
            )

        return df

    def __len__(self) -> int:
        return len(self.paired_iids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            For single mode:
                - 'neuro_embedding': (12150,)
                - 'gene_embedding': (256,)
                - 'iid': str
                - 'gene_name': str

            For sequence mode:
                - 'neuro_embedding': (12150,)
                - 'gene_sequence': (num_genes, 256) - e.g., (111, 256)
                - 'iid': str
                - 'gene_names': List[str]
        """
        iid = self.paired_iids[idx]

        # Load neuroimaging latent
        brain_latent_path = os.path.join(self.brain_latent_dir, f"{iid}_latent.npz")

        try:
            with np.load(brain_latent_path) as data:
                if 'latent' in data:
                    brain_latent = data['latent']
                elif 'arr_0' in data:
                    brain_latent = data['arr_0']
                else:
                    brain_latent = data[list(data.keys())[0]]

            brain_latent_flat = brain_latent.flatten()

            if self.transform is not None:
                brain_latent_flat = self.transform(brain_latent_flat)

            neuro_embedding = torch.from_numpy(brain_latent_flat).float()

        except Exception as e:
            logger.error(f"Error loading brain latent for IID {iid}: {e}")
            raise

        # Load gene embedding(s)
        if self.mode == 'single':
            try:
                gene_row = self.gene_df.loc[iid]
                gene_embedding = torch.from_numpy(
                    gene_row.values.astype(np.float32)
                )
            except Exception as e:
                logger.error(f"Error loading gene embedding for IID {iid}: {e}")
                raise

            return {
                'neuro_embedding': neuro_embedding,
                'gene_embedding': gene_embedding,
                'iid': iid,
                'gene_name': self.single_gene_name,
            }

        else:  # sequence mode
            try:
                gene_embeddings = []

                for gene_name in self.gene_names:
                    gene_row = self.gene_dfs[gene_name].loc[iid]
                    gene_emb = gene_row.values.astype(np.float32)
                    gene_embeddings.append(gene_emb)

                # Stack to (num_genes, embedding_dim)
                gene_sequence = np.stack(gene_embeddings, axis=0)
                gene_sequence = torch.from_numpy(gene_sequence).float()

            except Exception as e:
                logger.error(f"Error loading gene sequence for IID {iid}: {e}")
                raise

            return {
                'neuro_embedding': neuro_embedding,
                'gene_sequence': gene_sequence,
                'iid': iid,
                'gene_names': self.gene_names,
            }

    def get_embedding_dims(self) -> Tuple[int, int]:
        """
        Returns:
            (neuro_dim, gene_dim)
            - For single mode: (12150, 256)
            - For sequence mode: (12150, 111*256=28416) or (12150, (111, 256))
        """
        neuro_dim = 3 * 15 * 18 * 15  # 12150

        if self.mode == 'single':
            gene_dim = self.gene_embedding_dim
        else:
            gene_dim = len(self.gene_names) * self.gene_embedding_dim

        return neuro_dim, gene_dim

    def get_num_genes(self) -> int:
        """Get number of genes."""
        if self.mode == 'single':
            return 1
        else:
            return len(self.gene_names)

    @classmethod
    def get_all_gene_names(cls, gene_embedding_dir: str) -> List[str]:
        """Get all available gene names from directory."""
        gene_files = glob.glob(
            os.path.join(gene_embedding_dir, '*_brain_gene_embeddingUKB.csv')
        )

        gene_names = []
        for fpath in gene_files:
            fname = os.path.basename(fpath)
            gene_name = fname.replace('_brain_gene_embeddingUKB.csv', '')
            gene_names.append(gene_name)

        return sorted(gene_names)
