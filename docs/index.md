# Documentation Index

이 문서는 저장소 세부 문서의 허브입니다. README는 전체 개요이고, 아래 문서들은 주제별 상세 분석입니다.

## Core documents

| 문서 | 내용 |
|---|---|
| [01. Experiment, Hardware, and Firmware](01_experiment_hardware_firmware.md) | 실험 사진, PPG 회로, STM32CubeMX 설정, ADC/IIR/threshold/FSM/IBI/UART 처리 설명 |
| [02. Python Dataset Pipeline](02_python_dataset_pipeline.md) | 왜 HRV를 쓰는지, HRV 공식, 데이터 로딩, 라벨링, sliding window, StandardScaler, KDE synthetic balancing |
| [03. Model, Training, and Results](03_model_training_results.md) | 왜 CNN+Transformer인지, 모델 구조, patch embedding, self-attention, 학습 설정, 평가 결과 |
| [04. Repository Map and Reproducibility](04_repository_map_reproducibility.md) | 저장소 구조, 파일 맵, 재현 절차, 결과 해석 시 주의점 |

## Key infographics

| 그림 | 역할 |
|---|---|
| `figures/infographics/ppg_circuit_signal_path_detailed.png` | PPG 회로가 왜 필요한지와 각 아날로그 블록 역할 |
| `figures/infographics/stm32_formula_guide.png` | ADC 변환, moving average, IIR, threshold, derivative, IBI 공식 정리 |
| `figures/infographics/peak_fsm_detailed_explainer.png` | STM32 미분 기반 FSM peak detector 설명 |
| `figures/infographics/hrv_why_and_features.png` | 왜 HRV를 쓰는지와 PPG→IBI→HRV→model 입력 흐름 |
| `figures/infographics/hrv_feature_formula_guide.png` | MeanRR, SDNN, RMSSD, LF/HF 개념과 공식 |
| `figures/infographics/kde_theory_usage.png` | KDE 개념, 공식, 사용 이유 |
| `figures/infographics/model_rationale_detailed.png` | CNN+Transformer 구조를 쓰는 이유와 block별 역할 |
| `figures/infographics/evaluation_metrics_explainer.png` | confusion, ROC, PR, calibration, F1, Youden J 해석 |

## Scope note

이 저장소 문서는 **STM32 기반 프로젝트**로 정리되어 있습니다. Raspberry Pi 포팅 문서는 제거했으며, 남은 핵심 경로는 다음과 같습니다.

```text
PPG analog front-end → STM32 ADC / firmware → UART IBI stream → Python HRV features → CNN + Transformer model
```
