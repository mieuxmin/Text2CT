"""
Script to extract aligned embeddings from a trained model.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repository root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.alignment.neuro_gene_dataset import NeuroGeneDataset
from scripts.alignment.neuro_gene_model import NeuroGeneAlignmentModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main extraction function."""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = NeuroGeneDataset(
        brain_latent_dir=args.brain_latent_dir,
        gene_embedding_path=args.gene_embedding_path,
        iid_column=args.iid_column,
        gene_embedding_dim=args.gene_embedding_dim,
    )

    logger.info(f"Total samples: {len(dataset)}")

    # Get embedding dimensions
    neuro_dim, gene_dim = dataset.get_embedding_dims()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model
    logger.info("Loading model...")
    model = NeuroGeneAlignmentModel(
        neuro_input_dim=neuro_dim,
        gene_input_dim=gene_dim,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Val accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")

    # Extract embeddings
    logger.info("Extracting embeddings...")
    neuro_features, gene_features, iids = model.save_embeddings(dataloader, device)

    logger.info(f"Extracted {len(iids)} embeddings")
    logger.info(f"Neuro features shape: {neuro_features.shape}")
    logger.info(f"Gene features shape: {gene_features.shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as PyTorch tensors
    if args.save_pt:
        pt_path = os.path.join(args.output_dir, 'aligned_embeddings.pt')
        torch.save({
            'neuro_features': neuro_features,
            'gene_features': gene_features,
            'iids': iids,
            'checkpoint_path': args.checkpoint_path,
            'projection_dim': args.projection_dim,
        }, pt_path)
        logger.info(f"Saved PyTorch tensors to {pt_path}")

    # Save as CSV
    if args.save_csv:
        logger.info("Converting to CSV...")

        # Create dataframe
        df = pd.DataFrame({'IID': iids})

        # Add neuro features
        for i in range(neuro_features.shape[1]):
            df[f'neuro_{i}'] = neuro_features[:, i].numpy()

        # Add gene features
        for i in range(gene_features.shape[1]):
            df[f'gene_{i}'] = gene_features[:, i].numpy()

        csv_path = os.path.join(args.output_dir, 'aligned_embeddings.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")

    # Save as Parquet (more efficient)
    if args.save_parquet:
        logger.info("Converting to Parquet...")

        # Create dataframe
        df = pd.DataFrame({'IID': iids})

        # Add neuro features
        for i in range(neuro_features.shape[1]):
            df[f'neuro_{i}'] = neuro_features[:, i].numpy()

        # Add gene features
        for i in range(gene_features.shape[1]):
            df[f'gene_{i}'] = gene_features[:, i].numpy()

        parquet_path = os.path.join(args.output_dir, 'aligned_embeddings.parquet')
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet to {parquet_path}")

    # Compute and save similarity matrix if requested
    if args.save_similarity:
        logger.info("Computing similarity matrix...")
        similarity = neuro_features @ gene_features.t()

        sim_path = os.path.join(args.output_dir, 'similarity_matrix.pt')
        torch.save({
            'similarity': similarity,
            'iids': iids,
        }, sim_path)
        logger.info(f"Saved similarity matrix to {sim_path}")

        # Also save top-k most similar pairs
        logger.info(f"Finding top-{args.top_k} most similar pairs...")

        top_similarities = []
        for i in range(len(iids)):
            topk_values, topk_indices = similarity[i].topk(args.top_k)

            for rank, (idx, score) in enumerate(zip(topk_indices, topk_values)):
                top_similarities.append({
                    'neuro_iid': iids[i],
                    'gene_iid': iids[idx.item()],
                    'rank': rank + 1,
                    'similarity': score.item(),
                })

        top_sim_df = pd.DataFrame(top_similarities)
        top_sim_path = os.path.join(args.output_dir, f'top{args.top_k}_similarities.csv')
        top_sim_df.to_csv(top_sim_path, index=False)
        logger.info(f"Saved top-{args.top_k} similarities to {top_sim_path}")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract aligned embeddings from trained model")

    # Data paths
    parser.add_argument(
        '--brain_latent_dir',
        type=str,
        default='/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent',
        help='Directory containing {IID}_latent.npz files'
    )
    parser.add_argument(
        '--gene_embedding_path',
        type=str,
        required=True,
        help='Path to gene embedding CSV/parquet file'
    )
    parser.add_argument(
        '--iid_column',
        type=str,
        default='IID',
        help='Name of IID column in gene embedding file'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/aligned_embeddings',
        help='Output directory for embeddings'
    )

    # Model parameters (must match training)
    parser.add_argument(
        '--gene_embedding_dim',
        type=int,
        default=256,
        help='Dimension of gene embeddings'
    )
    parser.add_argument(
        '--projection_dim',
        type=int,
        default=512,
        help='Dimension of shared embedding space'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=None,
        help='Hidden layer dimension for projection (None = direct projection)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (not used in inference, but needed for model init)'
    )

    # Extraction parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for extraction'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )

    # Output format options
    parser.add_argument(
        '--save_pt',
        action='store_true',
        default=True,
        help='Save as PyTorch tensors (.pt)'
    )
    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='Save as CSV file'
    )
    parser.add_argument(
        '--save_parquet',
        action='store_true',
        help='Save as Parquet file'
    )
    parser.add_argument(
        '--save_similarity',
        action='store_true',
        help='Compute and save similarity matrix'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top similar pairs to save (if --save_similarity)'
    )

    args = parser.parse_args()

    main(args)
