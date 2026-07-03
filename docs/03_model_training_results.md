# 03. Model, Training, and Results

## 1. 문서 목적
이 문서는 `CNN_Transformer` 모델이 **왜 이런 구조를 택했는지**, 각 블록이 **무슨 역할을 하는지**, 그리고 실제 학습/평가가 어떤 방식으로 이루어지는지를 설명합니다.

---

## 2. 왜 CNN + Transformer 구조를 사용하는가

<p align="center"><img src="figures/infographics/model_rationale_detailed.png" width="100%"></p>

### 2-1. 문제 구조
이 프로젝트의 입력은 이미지가 아니라, **HRV 4개 특징이 시간축으로 12 프레임 쌓인 2차원 시퀀스**입니다.

- 입력 형태: `(N, 1, 12, 4)`
- 의미: `배치 × 채널 × 시간 × 특징`

즉, 단일 벡터 분류가 아니라 **작은 시계열-격자 입력**을 다루는 문제입니다.

### 2-2. 왜 CNN을 먼저 쓰는가
CNN은 인접한 시간프레임과 특징축 사이의 **국소적(local) 패턴**을 추출하는 데 유리합니다.

이 프로젝트에서는 다음 구조가 그것을 담당합니다.
- `Conv2d(1→32, 3×3)`
- `BatchNorm2d(32)`
- `GELU`
- `Conv2d(32→64, 3×3)`
- `BatchNorm2d(64)`
- `GELU`
- `Dropout2d(0.30)`

즉, CNN은 “MeanRR와 SDNN이 인접 시간에서 같이 움직이는 패턴”, “짧은 구간의 진동 양상” 같은 **로컬 구조**를 feature map으로 바꿉니다.

### 2-3. 왜 Transformer를 뒤에 붙이는가
CNN만 쓰면 국소 패턴은 잘 잡지만, **시퀀스 전체 구간 사이의 장거리 관계**를 명시적으로 비교하는 데는 한계가 있습니다.  
Transformer는 self-attention으로 **멀리 떨어진 패치 간 관계**를 모델링할 수 있습니다.

즉,
- CNN: **local pattern extraction**
- Transformer: **global sequence relation modeling**

이라는 역할 분담입니다.

---

## 3. 전체 모델 구조

<p align="center"><img src="figures/architecture/cnn_transformer_model_architecture.png" width="85%"></p>
<p align="center"><img src="figures/infographics/model_inference_pipeline.png" width="100%"></p>

모델은 크게 다음 4단계로 구성됩니다.

1. **CNN feature extractor**
2. **Patch embedding**
3. **Transformer encoder**
4. **Classifier head**

---

## 4. 모델 세부 구성

### 4-1. 입력
- shape: `(N, 1, 12, 4)`
- 의미: 배치 × 채널 × 시간 × 특징

여기서 `12`는 시간 단계 수이고, `4`는 `MeanRR, SDNN, RMSSD, LF/HF`를 의미합니다.

### 4-2. CNN feature extractor
#### 개념
CNN은 입력 HRV 격자에서 **국소 패턴**을 추출합니다.

#### 코드 구성
- `Conv2d(1→32, 3×3, padding=1)`
- `BatchNorm2d(32)`
- `GELU`
- `Conv2d(32→64, 3×3, padding=1)`
- `BatchNorm2d(64)`
- `GELU`
- `Dropout2d(0.30)`

#### 출력 shape
`(N, 64, 12, 4)`

즉, 입력의 작은 시계열-특징 격자가 64채널 feature map으로 확장됩니다.

### 4-3. Patch embedding
#### 개념
Transformer는 보통 token sequence를 입력으로 받습니다.  
따라서 CNN feature map을 **시간축 패치(token)** 로 잘라서 embedding해야 합니다.

#### 코드 파라미터
- `patch_len = 2`
- `stride = 1`

#### 패치 수 계산
`num_patches = (12 - 2) / 1 + 1 = 11`

#### 패치 차원
CNN 출력 한 패치는 다음 차원을 가집니다.

`patch_dim = 2 × 4 × 64 = 512`

#### 임베딩
`Linear(512 → 128)`

즉, 각 패치는 512차원에서 128차원 token으로 압축됩니다.

### 4-4. Transformer encoder
#### 개념
Transformer는 token들 사이의 관계를 self-attention으로 계산합니다.

#### 기본 self-attention 식
`Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) V`

이 프로젝트에서는 이 메커니즘을 여러 head에서 병렬로 수행합니다.

#### 코드 파라미터
- `d_model = 128`
- `n_heads = 8`
- `num_layers = 3`
- `dim_feedforward = 512`
- `activation = GELU`
- `dropout = 0.30`

#### 추가 요소
- learnable `[CLS] token`
- positional embedding

`[CLS]` 토큰은 전체 시퀀스를 대표하는 요약 벡터 역할을 합니다.

### 4-5. 출력층
최종적으로 `[CLS]` 토큰을 선택해 다음 연산을 수행합니다.

`LayerNorm(128) → Linear(128→1) → Sigmoid`

확률 표현은 다음과 같습니다.

`p_hat = sigmoid(W · h_CLS + b)`

즉, 최종 출력은 **High 클래스일 확률**입니다.

---

## 5. 이 모델을 구조도로 다시 해석하면

### 5-1. 파이프라인 관점
1. HRV 4개 특징이 시간축으로 쌓인 입력을 받음
2. CNN이 local pattern을 추출함
3. 패치 분할로 토큰 시퀀스로 바꿈
4. Transformer가 장거리 관계를 학습함
5. `[CLS]` 토큰으로 전체 시퀀스를 요약하고 분류함

### 5-2. 왜 이 구조가 이 프로젝트에 맞는가
- 입력 길이가 길지 않음 (`12 step`)
- 특징 차원이 작음 (`4 feature`)
- 단순 RNN보다 병렬 처리에 유리함
- CNN으로 노이즈성 local variation을 완충하고, Transformer로 전역 구조를 포착 가능

즉, 이 모델은 “생체신호용 대형 foundation model”이 아니라, **작은 HRV 시퀀스에 맞춘 경량 하이브리드 분류기**입니다.

---

## 6. 학습 설정

| 항목 | 값 |
|---|---:|
| Epochs | 400 |
| Minimum epochs | 200 |
| LR | 3e-4 |
| Batch size | 64 |
| Warmup epochs | 10 |
| Early stopping patience | 40 |
| Optimizer | AdamW |
| Dropout | 0.30 |

### 6-1. 왜 이런 설정이 중요한가
- `MIN_EPOCHS = 200`: 너무 이르게 멈추지 않도록 함
- `Warmup = 10`: 초기 학습 안정화
- `AdamW`: weight decay와 함께 안정적 최적화
- `Dropout = 0.30`: 과적합 완화

---

## 7. 평가 방식

### 7-1. 5-Fold Stratified Cross Validation
데이터를 5개 fold로 나누고, 각 fold에서 High / Low 비율이 유지되도록 평가합니다.

### 7-2. 8:2 Holdout Test
추가적으로 학습/검증 분리 외에 holdout set 기반 성능도 계산합니다.

### 7-3. 출력 파일
각 fold 및 holdout 결과는 다음을 포함합니다.
- confusion matrix
- ROC / PR / calibration curve
- loss curve
- Youden J statistic bar
- `predictions.csv`

즉, 단순 accuracy 하나만 저장하는 구조가 아니라, **분류 성능의 다양한 측면을 검토할 수 있게 설계된 평가 아카이브**입니다.

---

## 8. 결과 해석

### 8-1. 논문에 기재된 요약 성능
- 평균 AUC: `0.85`
- F1-score: `0.82`

이 값은 논문 본문 서술의 최종 요약값입니다.

### 8-2. 저장소 결과 파일 기준 5-fold 평균
- 평균 AUC: **0.999**
- 평균 Accuracy: **0.981**
- 평균 F1-score: **0.978**

이 값은 현재 저장소에 포함된 결과 이미지와 `summary_metrics.csv`를 기준으로 정리한 값입니다.

### 8-3. 대표 시각화
| Fold 5 confusion | Fold 5 loss | Fold 5 ROC / PR / calibration |
|---|---|---|
| <img src="figures/results/fold_5/confusion_high_low.jpg" width="100%"> | <img src="figures/results/fold_5/loss.jpg" width="100%"> | <img src="figures/results/fold_5/roc_pr_cal.jpg" width="100%"> |

| Holdout confusion | Holdout loss | Holdout ROC / PR / calibration |
|---|---|---|
| <img src="figures/results/holdout_test/confusion_high_low.jpg" width="100%"> | <img src="figures/results/holdout_test/loss.jpg" width="100%"> | <img src="figures/results/holdout_test/roc_pr_cal.jpg" width="100%"> |

### 8-4. 5-Fold 요약 표
| Fold | AUC | ACC | F1 | N_val | High | Low | Best threshold | Youden J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 140 | 60 | 80 | 0.979 | 1.000 |
| 2 | 0.999 | 0.943 | 0.929 | 140 | 60 | 80 | 0.364 | 0.971 |
| 3 | 0.995 | 0.979 | 0.976 | 140 | 60 | 80 | 0.963 | 0.971 |
| 4 | 1.000 | 0.993 | 0.992 | 140 | 60 | 80 | 0.977 | 1.000 |
| 5 | 1.000 | 0.993 | 0.992 | 140 | 60 | 80 | 0.473 | 1.000 |

---

## 9. 이 문서의 핵심 해석
이 저장소의 모델은 단순 CNN도 아니고, 단순 Transformer도 아닙니다.  
핵심은 다음과 같습니다.

> **CNN이 짧은 구간의 HRV 패턴을 먼저 구조화하고, Transformer가 그 패턴들을 시퀀스 전체 수준에서 다시 연결해 해석한다.**

즉, 모델 설계 의도는 “복잡해 보여서 붙인 구조”가 아니라, **작은 HRV 시퀀스에서 local + global 정보를 함께 보려는 구조적 선택**이라고 해석하는 것이 맞습니다.
