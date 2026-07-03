# 04. Repository Map and Reproducibility

## 1. 문서 목적
이 문서는 저장소 안의 파일들이 어떤 역할을 하는지 정리하고, 다른 사람이 최소한의 절차로 프로젝트 흐름을 따라갈 수 있도록 돕는 문서입니다.

---

## 2. 저장소 범위
이 저장소는 **STM32 기반 PPG-HRV 인지 과제 수행능력 분류 프로젝트**입니다.  
Raspberry Pi 포팅 문서는 범위에서 제거했습니다.

```text
Earlobe PPG sensor
→ PPG analog circuit
→ STM32F411RETx ADC + firmware
→ filtered PPG / IBI UART stream
→ Python HRV feature extraction
→ KDE-based synthetic balancing
→ CNN + Transformer classifier
→ N-back High / Low classification result
```

---

## 3. 폴더 구조

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

## 4. 핵심 코드 파일

| 파일 | 역할 |
|---|---|
| `src/stm32/main.c` | STM32F411RETx 기반 PPG ADC 수집, moving average, CMSIS-DSP IIR, adaptive threshold, derivative FSM peak detector, IBI/UART 출력 |
| `src/model/model.py` | HRV 특징 추출, N-back 라벨링, sliding window dataset, StandardScaler, KDE synthetic balancing, CNN+Transformer 학습/평가 |

---

## 5. 핵심 문서 파일

| 문서 | 설명 |
|---|---|
| `README.md` | 전체 프로젝트를 길게 설명하는 메인 문서 |
| `docs/01_experiment_hardware_firmware.md` | 실험, 하드웨어, STM32 설정, 펌웨어 알고리즘 상세 설명 |
| `docs/02_python_dataset_pipeline.md` | HRV 사용 이유, HRV 공식, 데이터셋 구성, 스케일링, KDE 설명 |
| `docs/03_model_training_results.md` | CNN+Transformer 설계 이유, 모델 구조, 학습/평가 결과 설명 |
| `docs/04_repository_map_reproducibility.md` | 저장소 구조, 재현 절차, 주의점 |

---

## 6. 핵심 그림 파일

| 파일 | 설명 |
|---|---|
| `docs/figures/circuit/ppg_analog_frontend_circuit.png` | 실제 PPG 아날로그 회로도 |
| `docs/figures/infographics/ppg_circuit_signal_path_detailed.png` | 회로 단계별 역할 설명 인포그래픽 |
| `docs/figures/infographics/stm32_formula_guide.png` | STM32 펌웨어 알고리즘 공식 요약 |
| `docs/figures/infographics/peak_fsm_detailed_explainer.png` | 미분 기반 FSM peak detector 상세 설명 |
| `docs/figures/infographics/hrv_why_and_features.png` | HRV 사용 이유와 PPG→HRV 흐름 |
| `docs/figures/infographics/hrv_feature_formula_guide.png` | HRV 4개 특징 공식과 해석 |
| `docs/figures/infographics/kde_theory_usage.png` | KDE 개념/공식/사용 이유 |
| `docs/figures/infographics/kde_balancing_pipeline.png` | KDE 기반 synthetic balancing 흐름 |
| `docs/figures/infographics/model_rationale_detailed.png` | CNN+Transformer 구조 선택 이유 |
| `docs/figures/infographics/model_inference_pipeline.png` | 모델 inference shape 흐름 |
| `docs/figures/infographics/evaluation_metrics_explainer.png` | 평가 지표 해석 |

---

## 7. 최소 재현 절차

### 7-1. STM32 측
1. STM32CubeMX 설정을 기반으로 STM32F411RETx 프로젝트를 생성합니다.
2. `src/stm32/main.c`의 로직을 반영합니다.
3. PPG 아날로그 회로 출력을 ADC1 입력에 연결합니다.
4. USART2를 통해 PC로 데이터를 전송합니다.
5. 출력 포맷은 `filtered_ppg,ibi_ms`입니다.

### 7-2. Python 측
```bash
pip install -r requirements.txt
python src/model/model.py
```

실행 전 `model.py` 내부 경로 설정을 실제 데이터 위치에 맞게 수정해야 합니다.

---

## 8. 결과 해석 시 주의점

1. 논문 본문에 적힌 성능과 `summary_metrics.csv`에 저장된 실행 결과는 같은 의미가 아닐 수 있습니다.
2. KDE synthetic balancing을 사용하므로, 실제 데이터와 합성 데이터의 구분이 중요합니다.
3. HRV feature는 raw PPG waveform이 아니라 IBI 기반 요약값입니다.
4. label은 PPG에서 만들어진 것이 아니라 N-back accuracy에서 만들어집니다.
5. STM32 펌웨어가 생성하는 IBI 품질이 Python HRV feature 품질에 직접 영향을 줍니다.

---

## 9. 재현 체크리스트

- [ ] PPG 회로 출력이 STM32 ADC 입력 범위 안에 들어오는지 확인
- [ ] UART가 `filtered_ppg,ibi_ms` 형태로 출력되는지 확인
- [ ] IBI가 250–2000 ms 범위 안에서 안정적으로 검출되는지 확인
- [ ] `nback_results.csv`에 filename과 accuracy가 정리되어 있는지 확인
- [ ] Python 코드의 data path가 현재 PC 경로와 맞는지 확인
- [ ] StandardScaler가 train/test split leakage 없이 적용되는지 확인할 것
- [ ] synthetic data와 real data를 구분해 결과를 해석할 것

