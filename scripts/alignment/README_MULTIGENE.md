# Multi-Gene Neuroimaging Alignment

111개의 gene embedding을 neuroimaging latent와 align하는 두 가지 방법을 제공합니다.

## 🎯 두 가지 접근 방법

### 1. **Single Gene Mode** - 개별 Gene Alignment
각 gene을 개별적으로 neuroimaging과 align합니다.

```bash
python scripts/alignment/train_multi_gene_alignment.py \
    --mode single \
    --single_gene_name APOE \
    --output_dir ./outputs/single_gene/APOE
```

**장점**:
- 각 gene의 독립적인 영향 파악
- 간단한 모델 구조
- 빠른 학습

**사용 사례**:
- 특정 gene의 neuroimaging 연관성 분석
- Gene별 독립적인 예측 모델

### 2. **Sequence Mode** - Multi-Gene Transformer Alignment
111개의 gene을 sequence로 처리하여 transformer encoder로 학습합니다.

```bash
python scripts/alignment/train_multi_gene_alignment.py \
    --mode sequence \
    --output_dir ./outputs/multi_gene_sequence
```

**장점**:
- Gene 간 상호작용 학습
- 전체적인 유전자 프로파일 활용
- CLIP의 text encoder와 유사한 구조

**사용 사례**:
- 복합적인 유전자-뇌 관계 모델링
- 다중 gene을 활용한 neuroimaging 예측

## 📁 데이터 구조

```
/scratch/connectome/mieuxmin/Brain_Gene_FM/
├── APOE_brain_gene_embeddingUKB.csv      # IID + 256 embedding dims
├── BDNF_brain_gene_embeddingUKB.csv
├── ...
└── {gene_name}_brain_gene_embeddingUKB.csv  # 총 111개

/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent/
├── 1234567_latent.npz   # shape: (3, 15, 18, 15)
├── 1234568_latent.npz
└── ...
```

## 🏗️ 모델 구조

### Single Gene Mode

```
Neuro (12150) ──→ [Linear] ──→ L2-Normalize ──┐
                                                │
                                                ├──→ Contrastive Loss
                                                │
Gene (256)    ──→ [Linear] ──→ L2-Normalize ──┘
```

### Sequence Mode (Transformer)

```
Neuro (12150) ──→ [Linear] ──→ L2-Normalize ──────┐
                                                    │
                                                    ├──→ Contrastive Loss
                                                    │
Gene Sequence                                       │
(111, 256)                                          │
    │                                               │
    ├──→ [Input Projection] ──→ (111, 512)         │
    │                                               │
    ├──→ [Positional Encoding]                     │
    │                                               │
    ├──→ [Transformer Encoder]                     │
    │    - 4 layers                                 │
    │    - 8 attention heads                        │
    │    - GELU activation                          │
    │                                               │
    ├──→ [Pooling: mean/max/cls] ──→ (512)         │
    │                                               │
    └──→ [Projection] ──→ L2-Normalize ────────────┘
```

## 🚀 사용 방법

### 옵션 1: Single Gene Training (개별 gene)

```bash
# APOE gene만 학습
python scripts/alignment/train_multi_gene_alignment.py \
    --mode single \
    --single_gene_name APOE \
    --batch_size 128 \
    --num_epochs 50 \
    --projection_dim 512 \
    --output_dir ./outputs/single_gene/APOE

# 모든 gene에 대해 반복 실행
for gene in APOE BDNF COMT ...; do
    python scripts/alignment/train_multi_gene_alignment.py \
        --mode single \
        --single_gene_name $gene \
        --output_dir ./outputs/single_gene/$gene
done
```

### 옵션 2: Multi-Gene Sequence Training

```bash
# 기본 설정 (mean pooling)
python scripts/alignment/train_multi_gene_alignment.py \
    --mode sequence \
    --batch_size 64 \
    --num_epochs 100 \
    --projection_dim 512 \
    --transformer_hidden_dim 512 \
    --transformer_num_layers 4 \
    --transformer_num_heads 8 \
    --transformer_pooling mean \
    --output_dir ./outputs/multi_gene_sequence

# CLS token pooling (BERT-style)
python scripts/alignment/train_multi_gene_alignment.py \
    --mode sequence \
    --transformer_pooling cls \
    --output_dir ./outputs/multi_gene_sequence_cls

# 더 큰 transformer
python scripts/alignment/train_multi_gene_alignment.py \
    --mode sequence \
    --transformer_hidden_dim 768 \
    --transformer_num_layers 6 \
    --transformer_num_heads 12 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --output_dir ./outputs/multi_gene_large
```

## 📊 하이퍼파라미터 가이드

### Single Gene Mode

| Parameter | Recommended | Description |
|-----------|------------|-------------|
| `--batch_size` | 128 | 더 클수록 좋음 |
| `--projection_dim` | 512 | 공유 공간 차원 |
| `--learning_rate` | 1e-4 | 학습률 |
| `--num_epochs` | 50-100 | 빠르게 수렴 |

### Sequence Mode

| Parameter | Recommended | Description |
|-----------|------------|-------------|
| `--batch_size` | 32-64 | Transformer는 메모리 많이 사용 |
| `--projection_dim` | 512 | 공유 공간 차원 |
| `--transformer_hidden_dim` | 512-768 | Transformer 내부 차원 |
| `--transformer_num_layers` | 4-6 | Layer 개수 |
| `--transformer_num_heads` | 8-12 | Attention head 개수 |
| `--transformer_pooling` | mean/cls | Pooling 방법 |
| `--learning_rate` | 1e-4 | 학습률 |
| `--num_epochs` | 100-200 | Transformer는 더 오래 학습 |

## 🔬 Transformer Pooling 방법

### 1. Mean Pooling (기본값)
- 모든 gene embedding의 평균
- 가장 안정적
- 모든 gene을 동등하게 고려

```python
output = transformer_output.mean(dim=1)  # (B, hidden_dim)
```

### 2. Max Pooling
- 각 차원의 최대값
- 중요한 feature 강조

```python
output = transformer_output.max(dim=1)[0]  # (B, hidden_dim)
```

### 3. CLS Token (BERT-style)
- 학습 가능한 special token 추가
- Sequence 전체 정보를 압축

```python
cls_token = learnable_parameter  # (1, 1, hidden_dim)
x = concat([cls_token, gene_sequence], dim=1)
output = transformer(x)[:, 0, :]  # Use CLS token
```

## 📈 성능 비교

실험 설정에 따른 예상 성능:

| Mode | Model Size | Training Time | Memory | Accuracy* |
|------|-----------|---------------|--------|-----------|
| Single | ~6M params | 10 min/epoch | 2GB | 0.65-0.75 |
| Sequence (small) | ~15M params | 30 min/epoch | 8GB | 0.70-0.80 |
| Sequence (large) | ~30M params | 60 min/epoch | 16GB | 0.75-0.85 |

*Accuracy: Cross-modal retrieval top-1 accuracy

## 💡 사용 예제

### 학습 후 Inference

```python
import torch
from scripts.alignment.multi_gene_dataset import MultiGeneNeuroDataset
from scripts.alignment.multi_gene_model import MultiGeneAlignmentModel

# Load model
model = MultiGeneAlignmentModel(
    neuro_input_dim=12150,
    gene_input_dim=256,
    num_genes=111,  # sequence mode
    projection_dim=512,
    use_transformer=True,
    transformer_hidden_dim=512,
    transformer_num_layers=4,
    transformer_num_heads=8,
)

checkpoint = torch.load('outputs/multi_gene_sequence/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
dataset = MultiGeneNeuroDataset(
    brain_latent_dir='/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent',
    gene_embedding_dir='/scratch/connectome/mieuxmin/Brain_Gene_FM',
    mode='sequence',
)

# Get a sample
sample = dataset[0]
neuro_emb = sample['neuro_embedding'].unsqueeze(0)  # (1, 12150)
gene_seq = sample['gene_sequence'].unsqueeze(0)     # (1, 111, 256)

# Encode
with torch.no_grad():
    neuro_feat = model.encode_neuro(neuro_emb)      # (1, 512)
    gene_feat = model.encode_gene(gene_seq)         # (1, 512)

    # Compute similarity
    similarity = (neuro_feat * gene_feat).sum()
    print(f"Similarity: {similarity.item():.4f}")
```

### Gene 간 Attention 시각화

```python
# Extract attention weights from transformer
# (sequence mode에서만 가능)

import matplotlib.pyplot as plt
import seaborn as sns

# Forward pass with attention weights
model.gene_encoder.transformer.layers[0].self_attn.register_forward_hook(
    lambda module, input, output: attention_weights.append(output[1])
)

attention_weights = []
_ = model(neuro_emb, gene_seq)

# Visualize attention (111 x 111)
attn = attention_weights[0][0].mean(0).cpu().numpy()  # Average over heads

plt.figure(figsize=(12, 10))
sns.heatmap(attn, xticklabels=dataset.gene_names, yticklabels=dataset.gene_names)
plt.title('Gene-Gene Attention Weights')
plt.tight_layout()
plt.savefig('gene_attention.png')
```

## 🛠️ 문제 해결

### 1. Gene 파일을 찾을 수 없음

```
ValueError: No gene embedding files found
```

**해결책**:
- `--gene_embedding_dir` 경로 확인
- `*_brain_gene_embeddingUKB.csv` 파일 형식 확인

### 2. Memory 부족 (Sequence Mode)

```
CUDA out of memory
```

**해결책**:
```bash
# 배치 크기 줄이기
--batch_size 16

# Gradient accumulation 사용
--gradient_accumulation_steps 4

# Transformer 크기 줄이기
--transformer_hidden_dim 256
--transformer_num_layers 2
```

### 3. 특정 Gene이 없음 (Single Mode)

```
ValueError: Gene 'XXX' not found
```

**해결책**:
```python
# 사용 가능한 gene 목록 확인
from scripts.alignment.multi_gene_dataset import MultiGeneNeuroDataset

genes = MultiGeneNeuroDataset.get_all_gene_names(
    '/scratch/connectome/mieuxmin/Brain_Gene_FM'
)
print(f"Available genes: {genes}")
```

## 📚 참고 자료

- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Transformer**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **BERT**: [Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

## 🎓 논문 작성 시 인용

```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={International Conference on Machine Learning},
  year={2021}
}
```
