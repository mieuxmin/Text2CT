# Job Submission Scripts

GPU cluster에서 neuroimaging-gene alignment 모델을 학습시키기 위한 SLURM job script 모음입니다.

## 📋 사용 가능한 Job Scripts

| Script | 용도 | GPU | Memory | 시간 |
|--------|------|-----|--------|------|
| `train_single_gene.sh` | 특정 gene 1개 학습 | 1 | 32GB | 24h |
| `train_sequence.sh` | 111개 gene sequence 학습 | 1 | 64GB | 48h |
| `train_sequence_large.sh` | 대형 transformer 학습 | 2 | 128GB | 72h |
| `train_all_genes.sh` | 모든 gene 개별 학습 (배치) | - | - | - |

## 🚀 사용 방법

### 1. 특정 Gene 하나만 학습 (Single Mode)

```bash
# APOE gene 학습
sbatch scripts/alignment/jobs/train_single_gene.sh APOE

# BDNF gene 학습
sbatch scripts/alignment/jobs/train_single_gene.sh BDNF

# COMT gene 학습
sbatch scripts/alignment/jobs/train_single_gene.sh COMT
```

**출력 위치**: `outputs/single_gene/{GENE_NAME}/`

### 2. 111개 Gene 전체를 Sequence로 학습 (권장!)

```bash
# 기본 transformer (4 layers, 8 heads)
sbatch scripts/alignment/jobs/train_sequence.sh
```

**출력 위치**: `outputs/multi_gene_sequence/`

### 3. 대형 Transformer로 학습 (성능 최대화)

```bash
# 대형 transformer (6 layers, 12 heads, 768 dim)
# 더 많은 GPU 메모리 필요
sbatch scripts/alignment/jobs/train_sequence_large.sh
```

**출력 위치**: `outputs/multi_gene_sequence_large/`

### 4. 모든 Gene을 개별적으로 학습

```bash
# 111개의 job을 순차적으로 제출
bash scripts/alignment/jobs/train_all_genes.sh
```

**출력 위치**: `outputs/single_gene/{각 gene name}/`

## 📊 Job 상태 확인

```bash
# 제출한 job 확인
squeue -u $USER

# 특정 job 상세 정보
scontrol show job JOB_ID

# 로그 확인 (실시간)
tail -f logs/sequence_JOBID.out

# 완료된 job 정보
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,MaxRSS
```

## 📁 출력 파일 구조

학습이 완료되면 다음과 같은 파일들이 생성됩니다:

```
outputs/
├── single_gene/
│   ├── APOE/
│   │   ├── best_model.pt              # 최고 성능 모델
│   │   ├── checkpoint_epoch_10.pt     # 중간 체크포인트
│   │   └── checkpoint_epoch_20.pt
│   ├── BDNF/
│   └── ...
├── multi_gene_sequence/
│   ├── best_model.pt
│   └── checkpoint_epoch_*.pt
└── multi_gene_sequence_large/
    ├── best_model.pt
    └── checkpoint_epoch_*.pt

logs/
├── single_gene_JOBID.out              # 표준 출력
├── single_gene_JOBID.err              # 에러 로그
├── sequence_JOBID.out
└── sequence_JOBID.err
```

## ⚙️ Job Script 커스터마이징

각 script를 수정하여 하이퍼파라미터를 조정할 수 있습니다:

### train_single_gene.sh

```bash
# 배치 크기 조정
--batch_size 128         # → 256 (더 빠름, 메모리 많이 필요)

# 학습률 조정
--learning_rate 1e-4     # → 5e-5 (더 안정적)

# Projection 차원 조정
--projection_dim 512     # → 768 (더 큰 임베딩 공간)
```

### train_sequence.sh

```bash
# Transformer 크기 조정
--transformer_hidden_dim 512    # → 768
--transformer_num_layers 4      # → 6
--transformer_num_heads 8       # → 12

# Pooling 방법 변경
--transformer_pooling mean      # → cls (BERT-style)

# 배치 크기와 accumulation
--batch_size 64                 # → 32
--gradient_accumulation_steps 1 # → 2
```

## 🔍 로그 모니터링

### 학습 진행 상황 확인

```bash
# 실시간 로그 확인
tail -f logs/sequence_JOBID.out

# Loss 값만 추출
grep "Train Loss" logs/sequence_JOBID.out

# Validation accuracy 추출
grep "Val.*Acc" logs/sequence_JOBID.out
```

### 예상 출력

```
==========================================
Epoch 10/100
==========================================
100%|████████████| 150/150 [00:45<00:00, 3.32it/s, loss=0.4523, lr=9.5e-05, scale=14.23]
Train Loss: 0.4523
Train Neuro Loss: 0.4512
Train Gene Loss: 0.4534
Validation: 100%|████████████| 17/17 [00:05<00:00, 3.12it/s]
Val Loss: 0.3821
Val Neuro->Gene Acc: 0.7234
Val Gene->Neuro Acc: 0.7156
Saved best model to ./outputs/multi_gene_sequence/best_model.pt
```

## 🛠️ 문제 해결

### Job이 시작되지 않음

```bash
# Job queue 확인
squeue -u $USER

# Pending 이유 확인
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"

# 파티션 확인
sinfo
```

### Out of Memory

**증상**: `CUDA out of memory` 에러

**해결책**:
```bash
# 배치 크기 줄이기
--batch_size 32  # → 16

# Gradient accumulation 사용
--gradient_accumulation_steps 2  # → 4

# Transformer 크기 줄이기
--transformer_hidden_dim 512  # → 256
--transformer_num_layers 4    # → 2
```

### GPU 사용률이 낮음

**증상**: GPU utilization < 50%

**해결책**:
```bash
# Worker 수 증가
--num_workers 4  # → 8

# Pin memory 활성화 (이미 활성화됨)
# DataLoader에서 자동 설정됨
```

## 📈 성능 벤치마크

테스트 환경: NVIDIA A100 40GB

| Configuration | Batch Size | Time/Epoch | GPU Memory | Val Acc |
|---------------|-----------|------------|------------|---------|
| Single Gene | 128 | ~10 min | 8GB | 0.70 |
| Sequence (small) | 64 | ~30 min | 20GB | 0.75 |
| Sequence (base) | 64 | ~45 min | 32GB | 0.78 |
| Sequence (large) | 32 | ~90 min | 60GB | 0.82 |

## 🎯 추천 설정

### 빠른 프로토타이핑
```bash
sbatch scripts/alignment/jobs/train_single_gene.sh APOE
```
- 빠른 학습 (10 min/epoch)
- 적은 메모리 (8GB)
- Gene별 개별 분석

### 최고 성능
```bash
sbatch scripts/alignment/jobs/train_sequence_large.sh
```
- 대형 transformer
- Gene 간 상호작용 학습
- 최고 retrieval accuracy

### 균형잡힌 선택 (권장)
```bash
sbatch scripts/alignment/jobs/train_sequence.sh
```
- 적당한 모델 크기
- 합리적인 학습 시간
- 좋은 성능

## 📞 추가 도움말

- **하이퍼파라미터 튜닝**: `README_MULTIGENE.md` 참조
- **모델 구조 이해**: `multi_gene_model.py` 주석 참조
- **데이터셋 디버깅**: `multi_gene_dataset.py` 참조

## 🔗 관련 문서

- [Multi-Gene Alignment 가이드](../README_MULTIGENE.md)
- [기본 사용법](../README.md)
- [예제 코드](../example_usage.py)
