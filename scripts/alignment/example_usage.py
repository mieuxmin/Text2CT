"""
Example usage of neuroimaging-gene alignment model.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from neuro_gene_dataset import NeuroGeneDataset
from neuro_gene_model import NeuroGeneAlignmentModel


def example_training():
    """Example: Basic training setup"""

    # 1. Create dataset
    dataset = NeuroGeneDataset(
        brain_latent_dir='/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent',
        gene_embedding_path='/path/to/gene_embeddings.csv',
        iid_column='IID',
        gene_embedding_dim=256,
    )

    print(f"Dataset size: {len(dataset)}")

    # Get embedding dimensions
    neuro_dim, gene_dim = dataset.get_embedding_dims()
    print(f"Neuro dim: {neuro_dim}, Gene dim: {gene_dim}")

    # 2. Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )

    # 3. Initialize model
    model = NeuroGeneAlignmentModel(
        neuro_input_dim=neuro_dim,
        gene_input_dim=gene_dim,
        projection_dim=512,
        hidden_dim=1024,  # Use 2-layer MLP
        dropout=0.1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 4. Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for batch in dataloader:
            neuro_emb = batch['neuro_embedding'].to(device)
            gene_emb = batch['gene_embedding'].to(device)

            # Forward pass
            outputs = model(neuro_emb, gene_emb, return_loss=True)
            loss = outputs['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def example_inference():
    """Example: Load trained model and perform inference"""

    # 1. Load model
    model = NeuroGeneAlignmentModel(
        neuro_input_dim=12150,
        gene_input_dim=256,
        projection_dim=512,
    )

    checkpoint = torch.load('outputs/neuro_gene_alignment/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 2. Load some data
    neuro_embeddings = torch.randn(10, 12150).to(device)  # 10 samples
    gene_embeddings = torch.randn(10, 256).to(device)

    # 3. Encode to shared space
    with torch.no_grad():
        neuro_features = model.encode_neuro(neuro_embeddings)
        gene_features = model.encode_gene(gene_embeddings)

        print(f"Neuro features shape: {neuro_features.shape}")  # (10, 512)
        print(f"Gene features shape: {gene_features.shape}")    # (10, 512)

        # 4. Compute similarity
        similarity = model.get_similarity_matrix(neuro_embeddings, gene_embeddings)
        print(f"Similarity matrix shape: {similarity.shape}")  # (10, 10)

        # Find most similar pairs
        for i in range(len(similarity)):
            most_similar_idx = similarity[i].argmax().item()
            similarity_score = similarity[i, most_similar_idx].item()
            print(f"Neuro {i} <-> Gene {most_similar_idx}: {similarity_score:.4f}")


def example_save_embeddings():
    """Example: Extract and save all embeddings"""

    # Load dataset
    dataset = NeuroGeneDataset(
        brain_latent_dir='/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent',
        gene_embedding_path='/path/to/gene_embeddings.csv',
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load model
    model = NeuroGeneAlignmentModel(
        neuro_input_dim=12150,
        gene_input_dim=256,
        projection_dim=512,
    )

    checkpoint = torch.load('outputs/neuro_gene_alignment/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Extract all embeddings
    neuro_features, gene_features, iids = model.save_embeddings(
        dataloader,
        device=device,
    )

    print(f"Extracted {len(iids)} embeddings")
    print(f"Neuro features shape: {neuro_features.shape}")
    print(f"Gene features shape: {gene_features.shape}")

    # Save to file
    output = {
        'neuro_features': neuro_features,
        'gene_features': gene_features,
        'iids': iids,
    }
    torch.save(output, 'aligned_embeddings.pt')
    print("Saved to aligned_embeddings.pt")

    # Also save as CSV for easy analysis
    df = pd.DataFrame({
        'IID': iids,
    })

    # Add neuro features
    for i in range(neuro_features.shape[1]):
        df[f'neuro_{i}'] = neuro_features[:, i].numpy()

    # Add gene features
    for i in range(gene_features.shape[1]):
        df[f'gene_{i}'] = gene_features[:, i].numpy()

    df.to_csv('aligned_embeddings.csv', index=False)
    print("Saved to aligned_embeddings.csv")


def example_cross_modal_retrieval():
    """Example: Cross-modal retrieval (neuro <-> gene)"""

    # Load model
    model = NeuroGeneAlignmentModel(
        neuro_input_dim=12150,
        gene_input_dim=256,
        projection_dim=512,
    )

    checkpoint = torch.load('outputs/neuro_gene_alignment/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load dataset
    dataset = NeuroGeneDataset(
        brain_latent_dir='/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent',
        gene_embedding_path='/path/to/gene_embeddings.csv',
    )

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    # Extract all embeddings
    neuro_features, gene_features, iids = model.save_embeddings(
        dataloader,
        device=device,
    )

    # Example: Given a neuroimaging sample, find top-k most similar gene profiles
    query_idx = 0
    query_neuro = neuro_features[query_idx:query_idx+1].to(device)

    # Compute similarity with all gene embeddings
    all_gene_features = gene_features.to(device)
    similarities = (query_neuro @ all_gene_features.t()).squeeze()

    # Get top-5
    top5_values, top5_indices = similarities.topk(5)

    print(f"\nTop-5 genes for neuroimaging sample {iids[query_idx]}:")
    for rank, (idx, score) in enumerate(zip(top5_indices, top5_values)):
        print(f"  {rank+1}. IID={iids[idx]}, similarity={score.item():.4f}")

    # Example: Given a gene profile, find top-k most similar neuroimaging samples
    query_idx = 0
    query_gene = gene_features[query_idx:query_idx+1].to(device)

    all_neuro_features = neuro_features.to(device)
    similarities = (query_gene @ all_neuro_features.t()).squeeze()

    top5_values, top5_indices = similarities.topk(5)

    print(f"\nTop-5 neuroimaging for gene profile {iids[query_idx]}:")
    for rank, (idx, score) in enumerate(zip(top5_indices, top5_values)):
        print(f"  {rank+1}. IID={iids[idx]}, similarity={score.item():.4f}")


if __name__ == "__main__":
    print("=== Neuroimaging-Gene Alignment Examples ===\n")

    print("1. Training example")
    print("-" * 50)
    # example_training()  # Uncomment to run

    print("\n2. Inference example")
    print("-" * 50)
    # example_inference()  # Uncomment to run

    print("\n3. Save embeddings example")
    print("-" * 50)
    # example_save_embeddings()  # Uncomment to run

    print("\n4. Cross-modal retrieval example")
    print("-" * 50)
    # example_cross_modal_retrieval()  # Uncomment to run
