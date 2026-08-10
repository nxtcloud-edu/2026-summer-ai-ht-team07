"""4모델 비교 + 안전 제약 조건 추천 + 합성 oracle 사후 감사 CLI.

주의: ``physics``는 모델 학습이나 추천에 전달되지 않는다. 모든 모델과 추천 조건을
먼저 동결한 뒤, 이 스크립트에서만 관측 조건 기반 합성 oracle로 사후 채점한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yeda.data.physics import latent_score, sigmoid  # noqa: E402
from yeda.io_utils import load_config  # noqa: E402
from yeda.models.yield_experiment import DISCLAIMER, run_yield_experiment  # noqa: E402
from yeda.schema import FEATURE_NAMES  # noqa: E402


def synthetic_posthoc_oracle(frame: pd.DataFrame) -> np.ndarray:
    """동결된 관측 조건에 합성 생성식을 적용해 사후 채점한다."""
    config = load_config("data_gen")
    columns = {
        feature: frame[feature].to_numpy(dtype=float) for feature in FEATURE_NAMES
    }
    return sigmoid(latent_score(columns, config))


def parse_args() -> argparse.Namespace:
    """실험 표본 수만 CLI에서 안전하게 재정의한다."""
    parser = argparse.ArgumentParser(description="4모델 수율 최적화 사후 감사")
    parser.add_argument(
        "--cohort-size",
        type=int,
        default=None,
        help="holdout 표본 수 (기본: configs/yield_experiment.yaml)",
    )
    args = parser.parse_args()
    if args.cohort_size is not None and args.cohort_size <= 0:
        parser.error("--cohort-size는 양수여야 합니다.")
    return args


def main() -> int:
    args = parse_args()
    experiment_config = load_config("yield_experiment")
    if args.cohort_size is not None:
        experiment_config = dict(experiment_config)
        experiment_config["cohort_size"] = args.cohort_size

    started = perf_counter()
    result = run_yield_experiment(
        experiment_config=experiment_config,
        oracle=synthetic_posthoc_oracle,
    )
    elapsed = perf_counter() - started

    display = result.summary.copy()
    percent_columns = [
        "cohort_actual_yield",
        "baseline_predicted_yield",
        "optimized_predicted_yield",
        "baseline_oracle_yield",
        "optimized_oracle_yield",
        "optimized_oracle_90pct_row_rate",
        "true_improvement_rate",
        "true_worsening_rate",
    ]
    for column in percent_columns:
        display[f"{column}_pct"] = display[column] * 100.0

    print("=== 4모델 예측 품질 + 동일 제약 수율 최적화 사후 감사 ===")
    print(f"선정 모델: {result.recommended_model}")
    columns = [
        "model",
        "selected_model",
        "holdout_pr_auc",
        "holdout_brier",
        "baseline_oracle_yield_pct",
        "optimized_oracle_yield_pct",
        "oracle_improvement_pp",
        "optimized_oracle_90pct_row_rate_pct",
        "true_worsening_rate_pct",
    ]
    print(
        display.loc[:, columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\n=== 선정 모델의 발표용 고정 시나리오 ===")
    presets = result.primary_presets.copy()
    for column in (
        "baseline_predicted_yield",
        "optimized_predicted_yield",
        "baseline_oracle_yield",
        "optimized_oracle_yield",
    ):
        presets[f"{column}_pct"] = presets[column] * 100.0
    print(
        presets.loc[
            :,
            [
                "sample_id",
                "baseline_oracle_yield_pct",
                "optimized_oracle_yield_pct",
                "oracle_gain_pp",
                "changed_features",
            ],
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    print(f"\n주의: {DISCLAIMER}")
    print(f"실행시간: {elapsed:.1f}초")
    print("\n생성 파일:")
    for name, path in result.paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
