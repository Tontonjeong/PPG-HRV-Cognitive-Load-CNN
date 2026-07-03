# CNN 기반 HRV 분석을 통한 인지 과제 수행 능력 예측 저장소

<p align="center">
  <img src="https://img.shields.io/badge/Project-PPG%20HRV%20Cognitive%20Load-0A66C2" alt="project badge">
  <img src="https://img.shields.io/badge/MCU-STM32F411RETx-03234B" alt="stm32 badge">
  <img src="https://img.shields.io/badge/Model-CNN%20%2B%20Transformer-8A2BE2" alt="model badge">
  <img src="https://img.shields.io/badge/Signal-PPG%20%2F%20HRV-2E8B57" alt="signal badge">
</p>

이 저장소는 **귓불 PPG(Photoplethysmography) 신호로부터 심박간 간격(IBI, RR)을 검출하고, HRV(Heart Rate Variability) 특징을 추출한 뒤, CNN + Transformer 기반 모델로 N-back 인지 과제 수행 능력을 분류하는 전체 연구 파이프라인**을 정리한 저장소입니다.  
단순히 코드만 보관하는 용도가 아니라, **하드웨어 구조 → STM32 펌웨어 → Python 전처리 → 데이터셋 구성 → 데이터 균형화 → 딥러닝 모델 → 결과 해석**까지를 한 번에 추적할 수 있도록 다시 구성했습니다.

> **중요**  
> - 이 README는 최대한 **논문, 업로드된 코드, 실험 사진, STM32 설정 스크린샷, 결과 이미지**를 서로 연결해서 설명하도록 재구성했습니다.  
> - 문서 내 설명은 업로드된 `model.py`, `stm코드.txt`, STM32CubeMX 스크린샷, 실험 사진, 논문 PDF를 기준으로 정리했습니다.  
> - 논문 본문에 기재된 요약 성능과, 코드 실행 결과 폴더(`paper_outputs_final`)에 저장된 성능은 서로 다른 층위의 정보이므로 **구분해서** 설명합니다.

---

## 1. 빠른 요약

- **센서/회로**: 귓불 PPG 센서 + 아날로그 증폭/필터 회로 + STM32F411RETx ADC
- **펌웨어**: 이동평균, CMSIS-DSP 기반 IIR 필터, 적응형 임계값, 1·2차 미분 기반 FSM 피크 검출, IBI 계산, UART 전송
- **Python 전처리**: N-back 결과 기반 라벨링, 60초 윈도우 / 4초 스텝 HRV 추출, StandardScaler 정규화
- **데이터 균형화**: `gaussian_kde` 기반 HRV 시퀀스 합성
- **모델**: CNN feature extractor + patch embedding + Transformer encoder + binary classifier
- **출력 목표**: N-back 수행 결과를 기준으로 **High / Low 그룹 분류**

---

## 2. 문서 구성 안내

README만으로도 전체를 이해할 수 있도록 길게 작성했으며, 세부 내용은 아래 문서로 연결됩니다.

- [`docs/index.md`](docs/index.md): 문서 허브
- [`docs/01_experiment_hardware_firmware.md`](docs/01_experiment_hardware_firmware.md): 실험 구성, 하드웨어, STM32 설정, 펌웨어 및 신호처리
- [`docs/02_python_dataset_pipeline.md`](docs/02_python_dataset_pipeline.md): Python 코드 분석, HRV 특징, 스케일링, 데이터셋, KDE 기반 증강
- [`docs/03_model_training_results.md`](docs/03_model_training_results.md): 모델 구조, 학습 루프, 평가 방식, 결과 해석
- [`docs/04_repository_map_reproducibility.md`](docs/04_repository_map_reproducibility.md): 저장소 구조, 파일 맵, 재현 절차

---

## 2-1. README 목차

- [1. 빠른 요약](#1-빠른-요약)
- [3. 전체 시스템 개요](#3-전체-시스템-개요)
- [4. 실험 환경과 실제 촬영 사진](#4-실험-환경과-실제-촬영-사진)
- [5. 하드웨어 구성](#5-하드웨어-구성)
- [6. STM32 설정과 펌웨어 처리 흐름](#6-stm32-설정과-펌웨어-처리-흐름)
- [7. Python 코드 분석: HRV 특징 추출과 데이터셋 생성](#7-python-코드-분석-hrv-특징-추출과-데이터셋-생성)
- [8. 데이터 균형화: KDE 기반 합성 HRV 시퀀스](#8-데이터-균형화-kde-기반-합성-hrv-시퀀스)
- [9. 모델 구조](#9-모델-구조)
- [10. 학습 방식과 평가 절차](#10-학습-방식과-평가-절차)

---

## 3. 전체 시스템 개요

<p align="center">
  <img src="docs/figures/infographics/end_to_end_overview.png" width="100%" alt="end to end overview">
</p>

<p align="center">
  <img src="docs/figures/architecture/overall_system_architecture.png" width="85%" alt="overall architecture">
</p>

위 두 그림은 이 연구를 한 장으로 요약한 것입니다.  
실제 파이프라인은 다음 순서를 따릅니다.

1. **귓불 PPG 센서**가 혈류 변화에 따른 광학 신호를 수집합니다.  
2. **PPG 아날로그 회로**가 미약한 신호를 증폭하고 잡음을 줄입니다.  
3. **STM32F411RETx**가 ADC로 신호를 샘플링하고, 실시간 피크 검출을 통해 IBI를 계산합니다.  
4. PC 측 Python 코드가 수집된 IBI를 기반으로 **MeanRR, SDNN, RMSSD, LF/HF**를 계산합니다.  
5. 60초 윈도우 기반 HRV 시퀀스를 만들고, StandardScaler와 KDE 기반 합성으로 학습용 데이터셋을 구성합니다.  
6. 최종적으로 **CNN + Transformer 모델**이 N-back 과제 수행 능력을 High / Low로 분류합니다.

---

## 4. 실험 환경과 실제 촬영 사진

### 4-1. 실험 장면

| 귓불 PPG 센서 부착 | STM32 + 브레드보드 + N-back 실험 장면 |
|---|---|
| <img src="docs/figures/experiment_photos/exp01_earlobe_ppg_sensor_closeup.jpg" width="100%"> | <img src="docs/figures/experiment_photos/exp02_full_setup_with_stm32_and_nback.jpg" width="100%"> |

| N-back 자극 화면 | 과제 수행 중 피험자 측면 | PPG 파형 모니터링 |
|---|---|---|
| <img src="docs/figures/experiment_photos/exp03_nback_screen_closeup.jpg" width="100%"> | <img src="docs/figures/experiment_photos/exp04_subject_side_nback_task.jpg" width="100%"> | <img src="docs/figures/experiment_photos/exp05_ppg_waveform_monitoring.jpg" width="100%"> |

이 사진들은 문서의 핵심 근거입니다. 즉, 이 저장소는 단순한 시뮬레이션 예제가 아니라, **실제 귓불 PPG 하드웨어와 STM32 수집 보드, 그리고 N-back 인지 과제 환경을 기반으로 수행한 프로젝트**라는 점을 보여줍니다.

### 4-2. N-back 결과 화면 예시

<p align="center">
  <img src="docs/figures/results/nback_block_result_example.png" width="70%" alt="nback result">
</p>

<p align="center">
  <img src="docs/figures/infographics/nback_labeling_explainer.png" width="100%" alt="nback labeling explainer">
</p>

실험 종료 후에는 위와 같은 결과 화면을 기반으로 **과제 정답률(accuracy)** 을 확보합니다.  
Python 코드에서는 이 정답률을 정규화하여 High / Low 라벨을 부여합니다.

---

## 5. 하드웨어 구성

### 5-1. 하드웨어 구조 요약 인포그래픽

<p align="center">
  <img src="docs/figures/infographics/hardware_overview.png" width="100%" alt="hardware overview">
</p>

### 5-2. PPG 아날로그 회로도

<p align="center">
  <img src="docs/figures/circuit/ppg_analog_frontend_circuit.png" width="90%" alt="ppg circuit">
</p>

<p align="center">
  <img src="docs/figures/infographics/ppg_circuit_signal_path_detailed.png" width="100%" alt="ppg circuit signal path detailed">
</p>

논문과 업로드된 그림 기준으로, 이 시스템의 하드웨어는 다음처럼 해석할 수 있습니다.

- **광원(LED)**: 귓불에 빛을 조사합니다.
- **포토다이오드**: 혈류 변화에 따라 반사/투과 광량의 변화를 전류 형태로 감지합니다.
- **아날로그 전처리 회로**: OPA2333xxD 연산증폭기와 RC 네트워크를 이용하여 매우 약한 PPG 신호를 증폭/정형합니다.
- **STM32 ADC 입력부**: 최종 아날로그 PPG 파형을 디지털 값으로 변환합니다.

아날로그-디지털 변환은 펌웨어에서 다음 식으로 전압으로 환산됩니다.

```math
V[n] = ADC[n] × 3.3 / 4095
```

여기서 4095는 12비트 ADC의 최대 카운트입니다.

---

## 6. STM32 설정과 펌웨어 처리 흐름

### 6-1. STM32CubeMX 설정 스크린샷

| Pinout | ADC1 | TIM1 |
|---|---|---|
| <img src="docs/figures/stm_screenshots/stm_pinout_overview.png" width="100%"> | <img src="docs/figures/stm_screenshots/stm_adc1_parameter_settings.png" width="100%"> | <img src="docs/figures/stm_screenshots/stm_tim1_settings.png" width="100%"> |

| USART2 | RCC | Clock |
|---|---|---|
| <img src="docs/figures/stm_screenshots/stm_usart2_settings.png" width="100%"> | <img src="docs/figures/stm_screenshots/stm_rcc_settings.png" width="100%"> | <img src="docs/figures/stm_screenshots/stm_clock_configuration.png" width="100%"> |

업로드된 `stm코드.txt`와 스크린샷을 종합하면 다음 설정이 핵심입니다.

- **MCU**: `STM32F411RETx`
- **ADC1**: 12-bit, Scan mode 사용, Regular conversion 2개, software trigger
- **TIM1**: prescaler = 999, period = 999
- **TIM2**: prescaler = 83, 1 MHz 기반의 microsecond timestamp 역할
- **USART2**: 115200 baud, 8N1
- **Clock**: HSE bypass + PLL 기반 100 MHz 시스템 클럭

### 6-2. 펌웨어 / 신호처리 인포그래픽

<p align="center">
  <img src="docs/figures/infographics/stm32_signal_pipeline.png" width="100%" alt="stm32 signal pipeline">
</p>

<p align="center">
  <img src="docs/figures/infographics/stm32_formula_guide.png" width="100%" alt="stm32 formula guide">
</p>

<p align="center">
  <img src="docs/figures/infographics/peak_fsm_detailed_explainer.png" width="100%" alt="peak fsm detailed explainer">
</p>

이 저장소의 STM32 펌웨어는 아래 순서대로 동작합니다.

#### (1) ADC 샘플링
`HAL_ADC_Start()` → `HAL_ADC_PollForConversion()` → `HAL_ADC_GetValue()` 순서로 샘플을 취득합니다.

#### (2) 5-포인트 이동평균
노이즈를 완화하기 위해 최근 5개 샘플의 평균을 구합니다.

```math
y[n] = (1/M) · Σ[k=0..M-1] x[n-k]`, where `M = 5`
```

#### (3) CMSIS-DSP 기반 IIR 필터
펌웨어는 `arm_biquad_cascade_df2T_f32()`를 사용하여 2-stage biquad 필터를 적용합니다.  
코드 주석상 구조는 **0.5 Hz 고역통과 + 8 Hz 저역통과** 체인을 의도하고 있으며, 결과적으로 PPG의 유효 맥파 성분만 남기도록 설계되어 있습니다.

#### (4) 적응형 임계값(Adaptive threshold)
필터 출력의 절댓값을 사용해 envelope `env`를 추적하고, 그 일정 비율을 threshold로 사용합니다.

```math
env[n] = a_up·|x[n]| + (1-a_up)·env[n-1]` when `|x[n]| > env[n-1]
```
```math
env[n] = a_dn·|x[n]| + (1-a_dn)·env[n-1]` when `|x[n]| <= env[n-1]
```
```math
`thr[n] = K · env[n]`, where `a_up = 0.40`, `a_dn = 0.02`, `K = 0.15`
```

#### (5) 미분 기반 FSM 피크 검출
코드는 단순 임계 비교가 아니라, **1차 미분 / 2차 미분 + 상태기계(FSM)** 를 사용합니다.

- 1차 미분:  ```math
  ( d_1[n] = x[n] - x[n-1] \)
```
- 2차 미분: ```math
d_2[n] = d_1[n] - d_1[n-1]
```

FSM 상태는 다음과 같습니다.

- `ST_IDLE`: 상승 시작점을 기다림
- `ST_RISING`: threshold를 넘고 기울기가 증가하는 구간
- `ST_SLOPEMAXED`: 기울기 최대점을 지난 뒤 실제 peak를 기다림
- `PEAK`: 유효 peak 확정 및 IBI 계산

#### (6) IBI 계산 및 유효성 검사
연속 두 peak의 시간차를 사용하여 IBI를 계산합니다.

`IBI_ms = (t_peak,i - t_peak,i-1) / 1000`

유효 범위는 코드상 다음과 같습니다.

- `IBI_MIN_MS = 250`
- `IBI_MAX_MS = 2000`
- `REFRACTORY_US = 300000`

즉, 너무 짧거나 너무 긴 interval은 버리고, 300 ms 이내의 중복 검출은 억제합니다.

#### (7) UART 출력
- peak가 검출된 샘플: `filtered_ppg,ibi_ms`
- 나머지 샘플: `filtered_ppg,0`

이 형식이 이후 Python 전처리 단계로 넘어갑니다.

---

## 7. Python 코드 분석: HRV 특징 추출과 데이터셋 생성

### 7-1. 왜 HRV를 쓰는가

<p align="center">
  <img src="docs/figures/infographics/hrv_why_and_features.png" width="100%" alt="why hrv">
</p>

이 프로젝트는 원시 PPG 진폭 자체를 모델에 넣지 않고, **PPG에서 peak를 찾은 뒤 IBI를 만들고, 그 IBI로부터 HRV 특징을 계산하는 경로**를 사용합니다. 그 이유는 다음과 같습니다.

- **생리학적 의미**: HRV는 beat-to-beat 간격 변동을 직접 반영합니다.
- **강건성**: 원시 PPG 진폭은 센서 부착과 접촉 상태에 민감하지만, HRV는 상대적으로 더 비교 가능한 feature space를 제공합니다.
- **구조화 가능성**: irregular IBI stream을 `(12, 4)` HRV 시퀀스로 바꾸면 학습 입력으로 쓰기 쉬워집니다.

즉, 이 저장소는 **원시 파형 분류**가 아니라, **생리학적으로 요약된 HRV 시퀀스 분류**를 수행하는 구조입니다.

### 7-2. 데이터셋 및 스케일링 인포그래픽

<p align="center">
  <img src="docs/figures/infographics/dataset_scaling_pipeline.png" width="100%" alt="dataset and scaling pipeline">
</p>

`model.py`는 원시 PPG/IBI 텍스트 파일과 `nback_results.csv`를 읽어 다음 과정을 수행합니다.

1. `PPG_*.txt`에서 `ppg`, `ibi` 열을 읽습니다.  
2. `nback_results.csv`의 `accuracy` 열을 기반으로 과제 수행 라벨을 생성합니다.  
3. IBI 누적시간 축에서 **60초 윈도우 / 4초 스텝**으로 HRV 특징을 계산합니다.  
4. 최소 `TIME_STEPS = 12`개의 HRV 프레임이 모이면 하나의 시퀀스로 사용합니다.  
5. 생성된 배열을 `StandardScaler`로 표준화하고, `HRVDataset`이 `(N, 1, 12, 4)` 텐서로 바꿉니다.

### 7-3. N-back 결과 기반 라벨링

정답률은 먼저 0~1 범위로 정규화됩니다.

- 값이 1보다 크면 퍼센트로 간주하여 100으로 나눕니다.
- 기본 라벨링 기준은 `acc >= 0.5` 입니다.
- 단일 클래스 쏠림이 발생하면 median / quantile fallback을 적용합니다.

기본 규칙은 다음과 같습니다.

`y = 1 if accuracy_norm >= 0.5 else 0`

### 7-4. HRV 특징의 개념, 공식, 의미

<p align="center">
  <img src="docs/figures/infographics/hrv_feature_formula_guide.png" width="100%" alt="hrv feature formula guide">
</p>

이 저장소가 사용하는 HRV 특징은 총 4개입니다.

1. **MeanRR**: 평균 RR 간격  
   `MeanRR = (1/N) · Σ RR_i`
2. **SDNN**: RR 간격 표준편차  
   `SDNN = sqrt( (1/(N-1)) · Σ (RR_i - MeanRR)^2 )`
3. **RMSSD**: 연속 RR 차이 제곱평균제곱근  
   `RMSSD = sqrt( (1/(N-1)) · Σ (RR_(i+1) - RR_i)^2 )`
4. **LF/HF**: 주파수영역 파워 비율  
   `LF/HF = integral(P(f), 0.04..0.15) / integral(P(f), 0.15..0.40)`

코드는 `scipy.signal.lombscargle`를 사용합니다.  
주파수 범위는 `0.01–0.5 Hz`, 적분 구간은 LF `0.04–0.15 Hz`, HF `0.15–0.40 Hz`입니다.

### 7-5. 데이터셋 구성 파라미터

| 항목 | 값 |
|---|---:|
| Window length | 60 s |
| Step size | 4 s |
| Time steps | 12 |
| Feature dimension | 4 |
| 최종 입력 shape | `(N, 1, 12, 4)` |

즉, 한 샘플은 **12개의 시간 프레임 × 4개의 HRV 특징**으로 구성된 2차원 시퀀스이며, CNN 입력을 위해 channel dimension이 하나 추가됩니다.

### 7-6. 왜 StandardScaler를 쓰는가

HRV 특징은 스케일이 서로 다르므로, 표준화 없이 바로 학습시키면 특정 특징이 수치적으로 우세해질 수 있습니다. 그래서 코드는 다음 공식을 적용합니다.

`z = (x - μ) / σ`

구현은 `X.reshape(-1, 4)` 형태로 feature axis 기준으로 fit/transform한 뒤, 다시 원래 shape로 복원하는 방식입니다.

## 8. 데이터 균형화: KDE 기반 합성 HRV 시퀀스

### 8-1. KDE 개념 / 공식 / 사용 이유

<p align="center">
  <img src="docs/figures/infographics/kde_theory_usage.png" width="100%" alt="kde theory">
</p>
<p align="center">
  <img src="docs/figures/infographics/kde_balancing_pipeline.png" width="100%" alt="kde balancing">
</p>

이 저장소에서 흔히 생각하는 이미지식 augmentation(shift, crop 등)은 사용되지 않습니다.  
대신 **클래스 불균형을 해결하기 위해 KDE 기반의 synthetic HRV sequence generation**을 사용합니다.

KDE는 다음과 같은 확률밀도 추정식으로 이해할 수 있습니다.

`p_hat(x) = (1 / (n·h)) · Σ K( (x - x_i) / h )`

여기서 `x_i`는 관측 샘플, `K`는 커널 함수, `h`는 bandwidth입니다. 즉, 소수 클래스 샘플을 그대로 복제하는 대신 **그 분포를 부드럽게 근사한 뒤 그 안에서 새로운 점을 샘플링**합니다.

이 방법을 쓰는 이유는 다음과 같습니다.

- 단순 복제보다 과적합 위험이 낮음
- HRV feature space가 4차원 연속값이므로 KDE와 잘 맞음
- raw PPG가 아니라 HRV 특징 공간에서 합성하므로 구현이 간결함

### 8-2. 코드 흐름

1. 특정 클래스의 시퀀스 `X[class]`를 모두 모아 `(전체 프레임 수, 4)` 형태로 펼칩니다.
2. `scipy.stats.gaussian_kde`로 클래스별 분포를 추정합니다.
3. 필요한 개수만큼 HRV feature vector를 샘플링합니다.
4. 시간 축 방향으로 smoothing을 적용합니다.
5. 생리적으로 불가능한 범위를 방지하기 위해 clipping을 적용합니다.

코드에 반영된 clipping 범위는 아래와 같습니다.

- MeanRR: 300 ~ 1800 ms
- SDNN: 5 ~ 400 ms
- RMSSD: 5 ~ 400 ms
- LF/HF: 0 ~ 5

시간 smoothing은 다음처럼 구현되어 있습니다.

`s_t = 0.6·x_t + 0.4·s_(t-1)`

### 8-3. 실제/합성 분포 비교 그림

| High group | Low group |
|---|---|
| <img src="docs/figures/results/boxplot_real_vs_synth_High.jpg" width="100%"> | <img src="docs/figures/results/boxplot_real_vs_synth_Low.jpg" width="100%"> |

이 그림들은 실제와 합성 데이터가 같은 feature space 상에서 어떤 관계를 갖는지 보여줍니다.

## 9. 모델 구조

### 9-1. 왜 CNN + Transformer인가

<p align="center">
  <img src="docs/figures/infographics/model_rationale_detailed.png" width="100%" alt="model rationale">
</p>

입력은 `(N, 1, 12, 4)` 형태의 작은 HRV 시퀀스입니다. 따라서 이 모델은 다음 역할 분담을 갖습니다.

- **CNN**: 인접 시간 프레임과 HRV 특징들 사이의 국소 패턴을 추출
- **Patch embedding**: CNN feature map을 Transformer가 읽을 수 있는 토큰 시퀀스로 변환
- **Transformer**: 멀리 떨어진 패치 간 관계를 self-attention으로 학습
- **Classifier**: `[CLS]` 토큰으로 전체 시퀀스를 요약하여 High / Low를 분류

### 9-2. 업로드된 구조도

<p align="center">
  <img src="docs/figures/architecture/cnn_transformer_model_architecture.png" width="85%" alt="cnn transformer architecture">
</p>

### 9-3. 코드 기반 모델 인포그래픽

<p align="center">
  <img src="docs/figures/infographics/model_inference_pipeline.png" width="100%" alt="model pipeline">
</p>

`CNN_Transformer` 클래스는 다음 구조를 가집니다.

#### (1) CNN feature extractor
- `Conv2d(1 → 32, 3×3, padding=1)`
- `BatchNorm2d(32)`
- `GELU`
- `Conv2d(32 → 64, 3×3, padding=1)`
- `BatchNorm2d(64)`
- `GELU`
- `Dropout2d(p=0.30)`

따라서 입력 `(N,1,12,4)`는 CNN 이후 `(N,64,12,4)` feature map이 됩니다.

#### (2) Patch embedding
시간 방향으로 `patch_len = 2`, `stride = 1`로 unfold합니다.  
그러면 patch 개수는

`num_patches = (12 - 2) / 1 + 1 = 11`

각 patch의 차원은

`2 × 4 × 64 = 512`

이며, 이후 `Linear(512 → 128)`로 임베딩됩니다.

#### (3) Transformer encoder
- `d_model = 128`
- `n_heads = 8`
- `num_layers = 3`
- `dim_feedforward = 512`
- activation = `gelu`
- dropout = `0.30`

또한 learnable `[CLS] token`과 positional embedding이 추가됩니다. Self-attention의 기본식은 다음과 같습니다.

`Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) V`

#### (4) Classifier head
마지막으로 `[CLS]` 토큰을 선택해

- `LayerNorm(128)`
- `Linear(128 → 1)`

을 통과시키고, sigmoid를 적용하여 High 그룹일 확률을 얻습니다.

`p_hat = sigmoid(W · h_CLS + b)`

## 10. 학습 방식과 평가 절차

`model.py`는 다음 평가 시나리오를 모두 포함합니다.

1. **5-Fold Stratified Cross Validation**
2. **8:2 Holdout Test**

학습 설정은 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| Epochs | 400 |
| Minimum epochs | 200 |
| Learning rate | 3e-4 |
| Batch size | 64 |
| Dropout | 0.30 |
| Early stopping patience | 40 |
| Warmup epochs | 10 |
| Optimizer | AdamW |

특히 early stopping은 `MIN_EPOCHS = 200` 이후에만 동작합니다. 즉, 너무 일찍 학습이 멈추지 않도록 설계되어 있습니다.

### 평가 지표
- ROC-AUC
- Accuracy
- F1-score
- Confusion matrix
- ROC / PR / Calibration curve
- Youden J statistic
- Loss curve

<p align="center">
  <img src="docs/figures/infographics/evaluation_metrics_explainer.png" width="100%" alt="evaluation metrics explainer">
</p>

---

## 11. 결과 해석

### 11-1. 논문 요약 성능
논문 본문 요약에는 **평균 AUC 0.85, F1-score 0.82**가 보고되어 있습니다.  
이 값은 논문의 최종 서술 결과입니다.

### 11-2. 코드 실행 결과 폴더 기준 성능
현재 저장소에 포함된 `paper_outputs_final` 결과 이미지와 `summary_metrics.csv` 기준으로 5-fold 평균은 다음과 같습니다.

- 평균 AUC: **0.999**
- 평균 Accuracy: **0.981**
- 평균 F1-score: **0.978**

즉, 논문 요약 성능과 코드 실행 결과 아카이브는 분리해서 보아야 하며, README에서는 둘을 혼동하지 않도록 명시합니다.

### 11-3. 대표 결과 이미지

| Fold 5 Confusion | Fold 5 Loss | Fold 5 ROC / PR / Calibration |
|---|---|---|
| <img src="docs/figures/results/fold_5/confusion_high_low.jpg" width="100%"> | <img src="docs/figures/results/fold_5/loss.jpg" width="100%"> | <img src="docs/figures/results/fold_5/roc_pr_cal.jpg" width="100%"> |

| Holdout Confusion | Holdout Loss | Holdout ROC / PR / Calibration |
|---|---|---|
| <img src="docs/figures/results/holdout_test/confusion_high_low.jpg" width="100%"> | <img src="docs/figures/results/holdout_test/loss.jpg" width="100%"> | <img src="docs/figures/results/holdout_test/roc_pr_cal.jpg" width="100%"> |

### 11-4. 5-Fold 요약 표

| Fold | AUC | Accuracy | F1 | Validation N | High N | Low N | Best threshold | Youden J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 140 | 60 | 80 | 0.979 | 1.000 |
| 2 | 0.999 | 0.943 | 0.929 | 140 | 60 | 80 | 0.364 | 0.971 |
| 3 | 0.995 | 0.979 | 0.976 | 140 | 60 | 80 | 0.963 | 0.971 |
| 4 | 1.000 | 0.993 | 0.992 | 140 | 60 | 80 | 0.977 | 1.000 |
| 5 | 1.000 | 0.993 | 0.992 | 140 | 60 | 80 | 0.473 | 1.000 |


---

## 12. 저장소 구조

```text
PPG-HRV-Cognitive-Load-CNN/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ sample/
├─ docs/
│  ├─ index.md
│  ├─ 01_experiment_hardware_firmware.md
│  ├─ 02_python_dataset_pipeline.md
│  ├─ 03_model_training_results.md
│  ├─ 04_repository_map_reproducibility.md
│  └─ figures/
│     ├─ architecture/
│     ├─ circuit/
│     ├─ experiment_photos/
│     ├─ infographics/
│     ├─ results/
│     └─ stm_screenshots/
├─ src/
│  ├─ model/
│  │  └─ model.py
│  └─ stm32/
│     └─ main.c
└─ references/
   └─ paper.pdf
```

---

## 13. 주요 파일 설명

- `src/model/model.py`  
  Python 기반 전체 학습/평가 코드입니다. HRV 특징 추출, KDE 합성, Dataset 정의, CNN+Transformer 모델, 5-fold/holdout 평가가 모두 들어 있습니다.

- `src/stm32/main.c`  
  STM32F411RETx 기반 PPG 수집 및 실시간 피크 검출 코드입니다. ADC, moving average, CMSIS-DSP IIR, adaptive threshold, derivative FSM, IBI 출력이 포함됩니다.

- `docs/figures/circuit/ppg_analog_frontend_circuit.png`  
  논문 및 프로젝트 설명에 사용되는 PPG 회로도입니다.

- `docs/figures/experiment_photos/*`  
  실제 N-back 실험, 센서 부착, STM32 보드 구성, 파형 모니터링 사진입니다.

- `docs/figures/results/*`  
  cross-validation과 holdout 평가 결과 이미지와 CSV입니다.

---

## 14. 실행 및 재현 절차

### 14-1. Python 환경

```bash
pip install -r requirements.txt
python src/model/model.py
```

### 14-2. STM32 측

- STM32CubeMX에서 제공된 설정을 기준으로 프로젝트를 생성합니다.
- `src/stm32/main.c`를 반영합니다.
- 보드에서 UART2를 통해 `filtered_ppg,ibi_ms` 스트림을 PC로 전송합니다.
- 저장된 텍스트 파일을 Python 코드가 읽어 학습용 데이터셋으로 사용합니다.

---

## 15. 참고 문서 바로가기

- [`docs/01_experiment_hardware_firmware.md`](docs/01_experiment_hardware_firmware.md)
- [`docs/02_python_dataset_pipeline.md`](docs/02_python_dataset_pipeline.md)
- [`docs/03_model_training_results.md`](docs/03_model_training_results.md)
- [`docs/04_repository_map_reproducibility.md`](docs/04_repository_map_reproducibility.md)
- [`references/paper.pdf`](references/paper.pdf)

---

## 16. 결론

이 저장소의 핵심은 **코드와 문서가 분리되지 않는 것**입니다.  
즉, 센서 사진과 회로도, STM32 설정, 펌웨어 로직, Python 전처리, HRV 공식, CNN/Transformer 구조, 평가 결과가 모두 서로 연결되어 있어야 이 프로젝트의 흐름을 제대로 이해할 수 있습니다.

이번 재구성본은 다음을 목표로 했습니다.

1. README 하나만 읽어도 프로젝트 전체 흐름을 이해할 수 있을 것  
2. 세부 문서가 중복되지 않고, 주제별로 통합되어 있을 것  
3. 인포그래픽이 실제 설명 내용과 맞고, 글씨가 충분히 커서 읽을 수 있을 것  
4. STM32 기반 프로젝트라는 정체성이 문서 전반에 일관되게 드러날 것  

