# C — 수율 및 모델 결과 해석

이 디렉터리의 CSV가 발표 수치의 단일 출처다. 노트북 출력이나 화면에 적은 예시
숫자를 발표에 인용하지 않는다. 모든 결과는 seed 20260810으로 재현된다.

## 핵심 결과

- 본선 모델: LightGBM + 단조 제약
- 홀드아웃 PR-AUC 0.9436, recall 0.9333, macro-F1 0.8251
- Brier score 0.1088, ECE 0.0267
- 홀드아웃 실제 수율 67.44%, 평균 예측 수율 67.16% (차이 -0.28%p)
- 동일한 max 3개·변수 범위 폭의 ±18% 제약에서 모델 예상 수율
  68.20% → 88.24%
- 모든 추천을 먼저 동결한 뒤 계산한 합성 사후 oracle 기대수율
  68.47% → 86.98% (+18.51%p, 95% bootstrap CI +15.71~+21.37%p)

## 파일 안내

| 파일 | 해석 |
|---|---|
| [model_comparison.csv](./model_comparison.csv) | 3개 알고리즘 계열, 4개 모델 설정의 홀드아웃·OOF 성능 비교 |
| [calibration_curve.csv](./calibration_curve.csv) | 예측확률 구간별 평균 예측 수율과 실제 관측 수율 |
| [yield_summary.csv](./yield_summary.csv) | 홀드아웃 전체의 실제 수율·평균 예측 수율·보정 차이 |
| [permutation_ablation.csv](./permutation_ablation.csv) | 피처 하나를 섞었을 때 PR-AUC 감소와 Brier 증가 |
| [feature_sweep_summary.csv](./feature_sweep_summary.csv) | 다른 조건을 고정하고 각 피처를 관측 q10~q90에서 바꾼 모델 민감도 |
| [yield_experiment_summary.csv](./yield_experiment_summary.csv) | 네 모델에 동일한 안전 제약을 적용한 예측·사후 oracle 수율 감사 |
| [yield_primary_presets.csv](./yield_primary_presets.csv) | 본선 모델의 발표용 고정 시나리오별 현재·추천 조건 |
| [yield_preset_model_comparison.csv](./yield_preset_model_comparison.csv) | 동일 프리셋에 대한 네 모델 결과 비교 |

## 결과 해석

### 1. 수율 예측

수율은 개별 다이의 pickup_success 확률 평균이다. 홀드아웃에서 실제 수율은
67.44%, 모델 평균 예측은 67.16%로 배치 수준 차이는 -0.28%p였다.

### 2. 주요 영향 변수

Permutation ablation의 PR-AUC 감소 기준 상위 변수는 다음 순서다.

1. uv_time: 0.0472
2. head_vacuum: 0.0264
3. pin_height: 0.0245
4. pin_speed: 0.0211
5. temperature: 0.0209

이 순위는 모델이 예측에 사용한 정보량이다. 실제 원인 또는 인과효과 순위가 아니다.

### 3. 변수값과 모델 예상 수율

관측 q10~q90 범위의 단변수 sweep에서 모델 예상 수율 변화폭이 컸던 예시는 다음과 같다.

- uv_time 4.0 → 5.9 s: 48.74% → 80.45%
- head_vacuum -62.7 → -67.5 kPa: 52.54% → 76.89%
- pin_speed 1.40 → 0.96 mm/s: 51.50% → 75.31%

이는 다른 피처를 각 행의 원래 값에 둔 모델 민감도 평균이며, 세 값을 동시에
적용했을 때의 보장 수율이나 현장 인과효과가 아니다.

### 4. 조건 추천 감사

모델은 최적화 후 예상 수율로 선택하지 않았다. 홀드아웃 PR-AUC를 우선하고 Brier를
동률 기준으로 사용해 LightGBM + 단조 제약을 먼저 고정했다. 그 뒤 모든 추천을
동결하고 숨은 합성 physics를 사후 감사에만 사용했다.

- 200행 평균 모델 예상: 68.20% → 88.24% (+20.04%p)
- 200행 평균 합성 사후 oracle: 68.47% → 86.98% (+18.51%p)
- 추천 후 oracle 확률이 90% 이상인 행: 83.5%
- 사후 oracle에서 개선 관측: 89.5%, 악화 관측: 0/200
- 정상 프리셋 oracle: 88.35% → 99.59%
- 얇은 다이 + 빠른 핀 프리셋 oracle: 18.52% → 90.66%
- 마모 + 약한 진공 프리셋 oracle: 16.80% → 74.49%

프리셋은 데모 사례이고 전체 평균이 아니다. 특히 마모 + 약한 진공 조건은 모델
예상 91.53%와 사후 oracle 74.49%가 크게 달라, 설비 상태가 나쁜 조건에서 모델
추천을 그대로 적용하면 안 된다는 한계를 보여준다.

## 재생성

저장소 루트에서 다음 순서로 실행한다.

1. make data
2. make train
3. python scripts/analyze_model.py
4. python scripts/run_yield_experiment.py

마지막 수율 실험은 200행 × 4모델 추천을 계산하므로 현재 환경에서 약 8분 걸린다.

## 반드시 함께 말할 한계

- 합성 데이터 결과이며 실제 기업·설비에서 검증된 수율이 아니다.
- SHAP, permutation ablation, feature sweep은 상관·모델 민감도 해석이지 인과 추정이 아니다.
- 사후 oracle은 모델 선택·학습·후보 생성·최적화에 사용하지 않았다.
- 악화 관측 0/200은 현장 위험이 0이라는 뜻이 아니다.
- 조건 추천은 현장 PoC 검증 대상이며 자동 적용값이 아니다.
