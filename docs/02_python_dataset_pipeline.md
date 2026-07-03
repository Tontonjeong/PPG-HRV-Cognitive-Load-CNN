# 02. Python Dataset Pipeline

## 1. 문서 목적
이 문서는 `src/model/model.py`의 실제 로직을 기준으로, **STM32에서 넘어온 PPG/IBI 데이터가 어떻게 HRV 시퀀스로 바뀌고, 왜 그 특징을 사용하며, 어떤 방식으로 스케일링과 클래스 균형화가 수행되는지**를 정리합니다.  
핵심은 단순 코드 나열이 아니라, **왜 이 과정을 쓰는지 → 어떤 수식이 쓰이는지 → 코드가 실제로 어떻게 구현하는지**를 연결해서 설명하는 것입니다.

---

## 2. 왜 HRV를 사용하는가

<p align="center"><img src="figures/infographics/hrv_why_and_features.png" width="100%"></p>

### 2-1. 개념 설명
이 프로젝트의 최종 목적은 **N-back 인지 과제 수행 능력**을 분류하는 것입니다.  
그러나 원시 PPG 파형은 센서 부착 상태, 광학 세기, 접촉 압력, 아날로그 회로 이득, 움직임 잡음에 영향을 많이 받습니다. 반면 **HRV(Heart Rate Variability)** 는 연속 심박 간격의 변동성을 요약한 특징 집합이기 때문에, 더 생리학적이고 비교 가능한 표현이 됩니다.

### 2-2. 왜 HRV가 인지 부하와 연결되는가
인지 과제 난이도가 올라가면 자율신경계 조절 양상이 바뀌고, 그 결과 **박동 간 간격(RR / IBI)의 평균과 변동성 패턴**이 달라질 수 있습니다.  
즉, 이 저장소는 다음 전제를 사용합니다.

1. **PPG 피크 간 시간차 = IBI**
2. **IBI 연속열 = 심박 변동성 정보의 원천**
3. **HRV 특징 = 인지 부하 변화에 반응하는 요약 표현**

### 2-3. 이 프로젝트에서 HRV를 쓰는 실질적 이유
- **생리학적 의미**: 단순 진폭이 아니라 심박 간 간격 변동을 봅니다.
- **강건성**: 원시 PPG 진폭보다 센서 부착 편차에 덜 민감합니다.
- **모델 입력 구조화**: 불규칙한 생체신호를 `(12, 4)`의 고정 길이 시퀀스로 바꿀 수 있습니다.
- **저장소 코드와 직접 연결**: `model.py`는 최종적으로 HRV 4개 특징만 사용합니다.

---

## 3. 이 코드가 사용하는 HRV 특징 4개

<p align="center"><img src="figures/infographics/hrv_feature_formula_guide.png" width="100%"></p>

이 프로젝트는 총 4개의 HRV 특징만 사용합니다.

1. **MeanRR**
2. **SDNN**
3. **RMSSD**
4. **LF/HF**

각 특징은 서로 다른 의미를 갖기 때문에, 단순히 “특징 4개”라고 보면 안 됩니다.  
각 특징은 **평균 수준 / 전체 변동성 / 단기 변동성 / 주파수영역 분포**를 각각 대표합니다.

### 3-1. MeanRR
#### 개념
한 윈도우 안에서의 **평균 박동 간 간격**입니다.

#### 공식
`MeanRR = (1/N) · Σ RR_i`

#### 의미
- MeanRR가 커지면 평균 심박수는 낮아지는 경향이 있습니다.
- 이 값은 “변동성”보다 **평균 심박 간격 수준**을 보여줍니다.

### 3-2. SDNN
#### 개념
전체 RR 간격의 **표준편차**입니다.

#### 공식
`SDNN = sqrt( (1/(N-1)) · Σ (RR_i - MeanRR)^2 )`

#### 의미
- RR 간격이 넓게 퍼져 있으면 SDNN이 커집니다.
- 전체적인 HRV의 spread를 나타냅니다.

### 3-3. RMSSD
#### 개념
연속 RR 간격 차이의 제곱평균제곱근입니다.

#### 공식
`RMSSD = sqrt( (1/(N-1)) · Σ (RR_(i+1) - RR_i)^2 )`

#### 의미
- 이웃한 박동 사이 변화가 빠를수록 값이 커집니다.
- **단기 변동성**을 강하게 반영합니다.

### 3-4. LF/HF
#### 개념
주파수영역에서 저주파(LF)와 고주파(HF) 파워의 비율입니다.

#### 공식
`LF/HF = integral(P(f), 0.04..0.15) / integral(P(f), 0.15..0.40)`

#### 코드상 구현 방식
- `scipy.signal.lombscargle()` 사용
- 분석 주파수축: `0.01–0.5 Hz`
- LF 대역: `0.04–0.15 Hz`
- HF 대역: `0.15–0.40 Hz`
- 적분: Simpson 적분 사용

#### 의미
- 단순 파형이 아니라 **주파수 분포 비율**입니다.
- RR 간격이 불규칙 간격 샘플이므로 일반 FFT 대신 **Lomb–Scargle**를 쓰는 점이 중요합니다.

---

## 4. 데이터 로딩과 라벨 생성

### 4-1. 데이터 로딩
`model.py`는 다음 두 입력을 핵심으로 사용합니다.

- `data/PPG_*.txt`: 각 실험 파일의 `ppg`, `ibi` 열
- `nback_results.csv`: `filename`, `accuracy`

파일명은 `_norm_name`, `_subject_key` 로직을 통해 정규화되고, subject 단위로 매칭됩니다.

### 4-2. 라벨 생성
정답률은 먼저 `0~1` 범위로 정규화됩니다.

- `accuracy > 1.0` 이면 퍼센트로 간주하고 `100`으로 나눔
- 기본 기준: `acc >= 0.5` → High(1), 그 외 Low(0)
- 클래스가 한쪽으로 치우치면 median / quantile fallback 사용

즉, 이 프로젝트의 label은 생리신호로부터 직접 나온 것이 아니라, **N-back 과제 성능(accuracy)을 기준으로 생성된 supervision signal**입니다.

---

## 5. 슬라이딩 윈도우 기반 HRV 시퀀스 생성

<p align="center"><img src="figures/infographics/dataset_scaling_pipeline.png" width="100%"></p>

### 5-1. 왜 슬라이딩 윈도우를 쓰는가
IBI는 이벤트 기반 데이터이므로 샘플 간격이 일정하지 않습니다.  
따라서 한 번에 전체 trial을 하나의 숫자로 요약하면 시간 변화가 사라집니다. 이 문제를 해결하기 위해 **고정 시간 윈도우에서 HRV를 반복 계산**하고, 이를 연속 시퀀스로 쌓습니다.

### 5-2. 코드 파라미터
| 변수 | 값 |
|---|---:|
| `WIN_SEC` | 60 |
| `STEP_SEC` | 4 |
| `TIME_STEPS` | 12 |
| `MIN_WIN_SAMPLES` | 10 |

### 5-3. 생성 절차
1. IBI 누적합으로 시간축 `t`를 만듭니다.
2. `start`부터 `start + 60초` 구간의 IBI를 모읍니다.
3. 해당 구간에서 HRV 4개 특징을 계산합니다.
4. 이를 4초 간격으로 반복합니다.
5. 연속된 12개 HRV 프레임이 모이면 하나의 시퀀스로 사용합니다.

### 5-4. 최종 형태
- 한 프레임: `4개 HRV 특징`
- 한 시퀀스: `12개 프레임`
- 최종 샘플 형태: `(12, 4)`
- `HRVDataset` 내부 텐서 형태: `(1, 12, 4)`

즉, 모델 입력은 **시간축 12 × 특징축 4**의 2차원 시퀀스입니다.

---

## 6. StandardScaler를 왜 사용하는가

### 6-1. 개념
HRV 특징은 서로 단위와 값 범위가 다릅니다.

- MeanRR: 수백~천 단위 ms
- SDNN / RMSSD: 수 ms ~ 수백 ms
- LF/HF: 대체로 작은 ratio

이 상태 그대로 학습하면 특정 특징 스케일이 과도하게 우세해질 수 있습니다. 그래서 특징별 평균과 분산을 정규화합니다.

### 6-2. 공식
`z = (x - μ) / σ`

### 6-3. 코드 구현 포인트
코드는 `X.reshape(-1, 4)`로 모든 time frame을 feature 축 기준으로 펼친 뒤 `fit/transform` 하고, 이후 원래 시퀀스 shape로 되돌립니다.

### 6-4. 왜 필요한가
- 각 특징의 수치 범위를 맞춤
- 학습 안정성 향상
- optimizer가 공정하게 feature를 보도록 도움

---

## 7. KDE 기반 synthetic balancing

<p align="center"><img src="figures/infographics/kde_theory_usage.png" width="100%"></p>
<p align="center"><img src="figures/infographics/kde_balancing_pipeline.png" width="100%"></p>

### 7-1. 왜 필요한가
원본 클래스 분포가 불균형하면, 모델은 다수 클래스 쪽으로 편향될 수 있습니다.  
이 프로젝트는 단순 복제(oversampling) 대신 **KDE(Kernel Density Estimation) 기반 샘플링**을 사용해 HRV feature space 안에서 새로운 시퀀스를 생성합니다.

### 7-2. KDE가 무엇인가
KDE는 관측 샘플들 위에 부드러운 커널을 얹어서 **확률밀도함수**를 추정하는 방법입니다.

#### 공식
`p_hat(x) = (1 / (n·h)) · Σ K( (x - x_i) / h )`

여기서,
- `x_i`: 관측된 샘플
- `K`: 커널 함수 (이 코드에서는 Gaussian kernel 기반)
- `h`: bandwidth

즉, 소수 클래스 샘플들을 그대로 복제하는 것이 아니라, 그 **주변 분포를 매끄럽게 근사**한 뒤 새로운 점을 샘플링합니다.

### 7-3. 코드 구현 절차
1. 특정 클래스 샘플만 골라 `(모든 시간프레임, 4)` feature pool을 만듭니다.
2. `gaussian_kde()`로 분포를 적합합니다.
3. 필요한 수만큼 시퀀스를 샘플링합니다.
4. 시간 방향 smoothing을 적용합니다.
5. 생리적 범위를 벗어나지 않도록 clipping합니다.

### 7-4. smoothing 공식
`synthetic[t] = 0.6 · x_t + 0.4 · synthetic[t-1]`

이 단계가 중요한 이유는, 독립 샘플링만 하면 시간축에서 값이 너무 튈 수 있기 때문입니다.

### 7-5. clipping 범위
- MeanRR: `300–1800`
- SDNN: `5–400`
- RMSSD: `5–400`
- LF/HF: `0–5`

### 7-6. 왜 이 방법을 쓰는가
- 단순 duplication보다 과적합 위험이 낮음
- feature space가 4차원 연속값이므로 KDE와 잘 맞음
- raw PPG가 아니라 **HRV 특징공간**에서 다룸으로써 구현 복잡도가 낮음

### 7-7. 실제 / 합성 분포 비교
| High | Low |
|---|---|
| <img src="figures/results/boxplot_real_vs_synth_High.jpg" width="100%"> | <img src="figures/results/boxplot_real_vs_synth_Low.jpg" width="100%"> |

---

## 8. 이 문서의 핵심 해석
Python 파이프라인은 단순한 “후처리 코드”가 아닙니다.  
오히려 이 저장소에서 Python 파트는 **STM32가 만든 IBI 스트림을, 생리학적으로 해석 가능한 HRV 시퀀스로 바꾸고, 이를 학습 가능한 형태로 재구성하는 중심부**입니다.

즉, 이 문서의 핵심은 다음 한 줄로 요약됩니다.

> **원시 PPG → IBI → HRV 특징 → 정규화 → 균형화 → 시퀀스 텐서화** 가 실제 모델 성능을 좌우하는 핵심 경로이다.
