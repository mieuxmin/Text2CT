"""
Training script for neuroimaging-gene embedding alignment.
Uses contrastive learning to align two modalities in a shared embedding space.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

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


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    epoch: int,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """
    Train for one epoch.

    Args:
        model: NeuroGeneAlignmentModel
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        gradient_accumulation_steps: Number of steps to accumulate gradients

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    total_neuro_loss = 0.0
    total_gene_loss = 0.0
    num_batches = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        neuro_emb = batch['neuro_embedding'].to(device)
        gene_emb = batch['gene_embedding'].to(device)

        # Forward pass
        outputs = model(neuro_emb, gene_emb, return_loss=True)

        loss = outputs['loss']
        logits_per_neuro = outputs['logits_per_neuro']
        logits_per_gene = outputs['logits_per_gene']

        # Compute individual losses for logging
        batch_size = logits_per_neuro.shape[0]
        labels = torch.arange(batch_size, device=device)
        neuro_loss = F.cross_entropy(logits_per_neuro, labels)
        gene_loss = F.cross_entropy(logits_per_gene, labels)

        # Scale loss by accumulation steps
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        total_loss += loss.item() * gradient_accumulation_steps
        total_neuro_loss += neuro_loss.item()
        total_gene_loss += gene_loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            'scale': f"{outputs['logit_scale'].item():.2f}"
        })

    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_neuro_loss = total_neuro_loss / num_batches
    avg_gene_loss = total_gene_loss / num_batches

    return {
        'loss': avg_loss,
        'neuro_loss': avg_neuro_loss,
        'gene_loss': avg_gene_loss,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """
    Evaluate model on validation set.

    Args:
        model: NeuroGeneAlignmentModel
        dataloader: Validation dataloader
        device: Device to evaluate on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_neuro_loss = 0.0
    total_gene_loss = 0.0
    num_batches = 0

    # Track retrieval accuracy
    correct_neuro_to_gene = 0
    correct_gene_to_neuro = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Validation"):
        neuro_emb = batch['neuro_embedding'].to(device)
        gene_emb = batch['gene_embedding'].to(device)

        # Forward pass
        outputs = model(neuro_emb, gene_emb, return_loss=True)

        loss = outputs['loss']
        logits_per_neuro = outputs['logits_per_neuro']
        logits_per_gene = outputs['logits_per_gene']

        # Compute individual losses
        batch_size = logits_per_neuro.shape[0]
        labels = torch.arange(batch_size, device=device)
        neuro_loss = F.cross_entropy(logits_per_neuro, labels)
        gene_loss = F.cross_entropy(logits_per_gene, labels)

        # Accumulate losses
        total_loss += loss.item()
        total_neuro_loss += neuro_loss.item()
        total_gene_loss += gene_loss.item()
        num_batches += 1

        # Compute retrieval accuracy (top-1)
        neuro_to_gene_pred = logits_per_neuro.argmax(dim=1)
        gene_to_neuro_pred = logits_per_gene.argmax(dim=1)

        correct_neuro_to_gene += (neuro_to_gene_pred == labels).sum().item()
        correct_gene_to_neuro += (gene_to_neuro_pred == labels).sum().item()
        total_samples += batch_size

    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_neuro_loss = total_neuro_loss / num_batches
    avg_gene_loss = total_gene_loss / num_batches

    neuro_to_gene_acc = correct_neuro_to_gene / total_samples
    gene_to_neuro_acc = correct_gene_to_neuro / total_samples

    return {
        'loss': avg_loss,
        'neuro_loss': avg_neuro_loss,
        'gene_loss': avg_gene_loss,
        'neuro_to_gene_acc': neuro_to_gene_acc,
        'gene_to_neuro_acc': gene_to_neuro_acc,
    }


def main(args):
    """Main training function."""

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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
    logger.info(f"Neuroimaging embedding dim: {neuro_dim}")
    logger.info(f"Gene embedding dim: {gene_dim}")

    # Split dataset
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model
    logger.info("Initializing model...")
    model = NeuroGeneAlignmentModel(
        neuro_input_dim=neuro_dim,
        gene_input_dim=gene_dim,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        logit_scale_init=args.logit_scale_init,
    )
    model = model.to(device)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # Setup scheduler
    total_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    logger.info(f"Total training steps: {total_steps}")

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Neuro Loss: {train_metrics['neuro_loss']:.4f}")
        logger.info(f"Train Gene Loss: {train_metrics['gene_loss']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Neuro->Gene Acc: {val_metrics['neuro_to_gene_acc']:.4f}")
        logger.info(f"Val Gene->Neuro Acc: {val_metrics['gene_to_neuro_acc']:.4f}")

        # Save best model
        avg_val_acc = (val_metrics['neuro_to_gene_acc'] + val_metrics['gene_to_neuro_acc']) / 2

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_val_loss = val_metrics['loss']

            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': avg_val_acc,
                'args': vars(args),
            }, checkpoint_path)

            logger.info(f"Saved best model to {checkpoint_path}")

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': avg_val_acc,
                'args': vars(args),
            }, checkpoint_path)

    logger.info(f"\n{'='*50}")
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neuroimaging-gene alignment model")

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
        '--output_dir',
        type=str,
        default='./outputs/neuro_gene_alignment',
        help='Output directory for checkpoints'
    )

    # Model parameters
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
        help='Dropout rate'
    )
    parser.add_argument(
        '--logit_scale_init',
        type=float,
        default=2.6592,  # ln(1/0.07)
        help='Initial value for learnable temperature'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )

    args = parser.parse_args()

    main(args)
