# Neuroimaging-Gene Embedding Alignment

이 모듈은 neuroimaging latent와 gene embedding을 contrastive learning으로 align하는 코드입니다.

## 구조

```
scripts/alignment/
├── __init__.py                      # 모듈 초기화
├── neuro_gene_dataset.py            # 데이터셋 클래스
├── neuro_gene_model.py              # Alignment 모델
├── train_neuro_gene_alignment.py   # 학습 스크립트
└── README.md                        # 문서
```

## 데이터 준비

### 1. Neuroimaging Latent

- **위치**: `/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent`
- **형식**: `{IID}_latent.npz`
- **Shape**: (3, 15, 18, 15)
- **처리**: flatten하여 (12150,) 벡터로 사용

### 2. Gene Embedding

- **형식**: CSV 또는 Parquet 파일
- **구조**:
  - IID 열: Subject ID
  - 256개의 embedding 열
- **예시**:
  ```
  IID,emb_0,emb_1,...,emb_255
  1234567,0.123,0.456,...,0.789
  1234568,0.234,0.567,...,0.890
  ```

## 사용 방법

### 기본 학습 실행

```bash
python scripts/alignment/train_neuro_gene_alignment.py \
    --gene_embedding_path /path/to/gene_embeddings.csv \
    --output_dir ./outputs/neuro_gene_alignment \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

### 주요 하이퍼파라미터

#### 데이터 관련
- `--brain_latent_dir`: Neuroimaging latent 디렉토리 (기본값: `/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent`)
- `--gene_embedding_path`: Gene embedding 파일 경로 (필수)
- `--iid_column`: IID 열 이름 (기본값: `IID`)
- `--gene_embedding_dim`: Gene embedding 차원 (기본값: 256)

#### 모델 구조
- `--projection_dim`: 공유 embedding 공간 차원 (기본값: 512)
- `--hidden_dim`: Projection layer의 hidden 차원 (기본값: None - direct projection)
- `--dropout`: Dropout rate (기본값: 0.1)

#### 학습 설정
- `--batch_size`: 배치 크기 (기본값: 128)
- `--num_epochs`: 학습 epoch 수 (기본값: 100)
- `--learning_rate`: 학습률 (기본값: 1e-4)
- `--weight_decay`: Weight decay (기본값: 0.01)
- `--val_split`: Validation 분할 비율 (기본값: 0.1)

### 고급 설정 예제

```bash
# Hidden layer를 사용하는 2-layer projection
python scripts/alignment/train_neuro_gene_alignment.py \
    --gene_embedding_path /path/to/gene_embeddings.csv \
    --projection_dim 512 \
    --hidden_dim 1024 \
    --dropout 0.2 \
    --batch_size 64 \
    --num_epochs 200 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2
```

## 모델 구조

### NeuroGeneAlignmentModel

```
Neuroimaging (12150,) ──┐
                        │
        ┌───────────────▼───────────────┐
        │  Neuro Projection Layer       │
        │  (12150 → [hidden] → 512)     │
        └───────────────┬───────────────┘
                        │
                        ▼
                L2 Normalize
                        │
                        ├─────────┐
                        │         │
Gene (256,) ────┐       │         │
                │       │         │
    ┌───────────▼───────▼─────────▼───┐
    │  Gene Projection Layer          │
    │  (256 → [hidden] → 512)         │
    └───────────┬─────────────────────┘
                │
                ▼
          L2 Normalize
                │
                └─────────┐
                          │
                    ┌─────▼──────┐
                    │  Similarity │
                    │   (scaled)  │
                    └─────┬───────┘
                          │
                    ┌─────▼──────┐
                    │ Contrastive│
                    │    Loss    │
                    └────────────┘
```

### Contrastive Loss

CLIP-style symmetric contrastive loss:

```python
# Similarity matrix with learnable temperature
logit_scale = exp(learnable_param).clamp(max=100)
similarity = logit_scale * neuro_features @ gene_features.T

# Symmetric cross-entropy
loss_neuro = CrossEntropy(similarity, labels)
loss_gene = CrossEntropy(similarity.T, labels)
loss = (loss_neuro + loss_gene) / 2
```

## 출력

학습 중에는 다음과 같은 출력을 볼 수 있습니다:

```
Epoch 10/100
100%|████████████| 100/100 [00:05<00:00, 18.23it/s, loss=0.4523, lr=9.5e-05, scale=14.23]
Train Loss: 0.4523
Train Neuro Loss: 0.4512
Train Gene Loss: 0.4534
Validation: 100%|████████████| 11/11 [00:00<00:00, 45.67it/s]
Val Loss: 0.3821
Val Neuro->Gene Acc: 0.7234
Val Gene->Neuro Acc: 0.7156
Saved best model to ./outputs/neuro_gene_alignment/best_model.pt
```

## 체크포인트 사용

### 학습된 모델 로드

```python
import torch
from scripts.alignment import NeuroGeneAlignmentModel

# 모델 초기화
model = NeuroGeneAlignmentModel(
    neuro_input_dim=12150,
    gene_input_dim=256,
    projection_dim=512,
)

# 체크포인트 로드
checkpoint = torch.load('outputs/neuro_gene_alignment/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 추론
with torch.no_grad():
    neuro_features = model.encode_neuro(neuro_embeddings)
    gene_features = model.encode_gene(gene_embeddings)

    # Similarity 계산
    similarity = model.get_similarity_matrix(neuro_embeddings, gene_embeddings)
```

### Embedding 추출

```python
from torch.utils.data import DataLoader
from scripts.alignment import NeuroGeneDataset

# 데이터셋 로드
dataset = NeuroGeneDataset(
    brain_latent_dir='/storage/bigdata/UKB_LDM/autoencoder_output/run_77481/brain_latent',
    gene_embedding_path='/path/to/gene_embeddings.csv',
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# 모든 embedding 추출
neuro_features, gene_features, iids = model.save_embeddings(
    dataloader,
    device='cuda'
)

# 저장
torch.save({
    'neuro_features': neuro_features,
    'gene_features': gene_features,
    'iids': iids,
}, 'aligned_embeddings.pt')
```

## 성능 지표

학습 중 추적되는 지표:

1. **Loss**: Contrastive loss (낮을수록 좋음)
2. **Neuro Loss**: Neuroimaging → Gene 방향 loss
3. **Gene Loss**: Gene → Neuroimaging 방향 loss
4. **Neuro→Gene Accuracy**: Neuroimaging에서 올바른 gene을 찾는 top-1 정확도
5. **Gene→Neuro Accuracy**: Gene에서 올바른 neuroimaging을 찾는 top-1 정확도

## 구현 세부사항

### Projection Layer 옵션

1. **Direct Projection** (`hidden_dim=None`):
   ```python
   neuro_projection = Linear(12150 → 512)
   gene_projection = Linear(256 → 512)
   ```

2. **Two-layer MLP** (`hidden_dim=1024`):
   ```python
   neuro_projection = Sequential(
       Linear(12150 → 1024),
       LayerNorm(1024),
       GELU(),
       Dropout(0.1),
       Linear(1024 → 512),
   )
   ```

### Temperature Scaling

- Learnable temperature parameter (as in CLIP)
- 초기값: `exp(2.6592) ≈ 14.3` (1/0.07)
- 최대값으로 clamp: 100
- 학습 중 자동으로 최적화됨

## 문제 해결

### IID 매칭 오류

```
ValueError: No paired samples found! Check IID matching.
```

**해결책**:
- Neuroimaging latent 파일명이 `{IID}_latent.npz` 형식인지 확인
- Gene embedding 파일의 IID 열이 올바른지 확인
- IID 형식이 일치하는지 확인 (문자열 vs 숫자)

### Memory 부족

```
CUDA out of memory
```

**해결책**:
- `--batch_size` 줄이기
- `--gradient_accumulation_steps` 늘리기
- `--projection_dim` 줄이기

## 참고자료

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Contrastive Learning Tutorial](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
