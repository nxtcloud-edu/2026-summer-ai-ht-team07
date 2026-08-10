"""본선 모델의 수율·순열 어블레이션·피처 스윕 분석 CLI.

먼저 make data 및 make train 을 실행한 뒤 사용한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yeda.models.analysis import CAUSALITY_WARNING, run_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YEDA 본선 모델의 수율 및 모델 민감도 분석"
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=None,
        help="피처별 순열 반복 수 (기본: configs/model.yaml)",
    )
    parser.add_argument(
        "--sweep-points",
        type=int,
        default=None,
        help="연속형 피처별 q10~q90 grid 수 (기본: configs/model.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="순열 재현 시드 (기본: configs/model.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = perf_counter()
    result = run_analysis(
        permutation_repeats=args.permutation_repeats,
        sweep_grid_points=args.sweep_points,
        seed=args.seed,
    )
    elapsed = perf_counter() - started

    yield_row = result.yield_summary.iloc[0]
    print("=== 홀드아웃 배치 수율 보정 요약 ===")
    print(f"실제 수율       : {yield_row['actual_yield_pct']:.2f}%")
    print(f"평균 예상 수율  : {yield_row['predicted_yield_pct']:.2f}%")
    print(f"Calibration gap: {yield_row['calibration_gap_pp']:+.2f}%p")

    print("\n=== 순열 어블레이션 상위 피처 ===")
    columns = ["feature", "pr_auc_drop_mean", "brier_increase_mean"]
    print(
        result.permutation_summary.loc[:, columns]
        .head(8)
        .to_string(index=False, float_format=lambda value: f"{value:.5f}")
    )

    print("\n=== 관측 범위 내 예측 수율 영향폭 상위 피처 ===")
    impact = result.feature_sweep_summary.head(8).copy()
    impact["min_yield_pct"] = impact["min_predicted_yield"] * 100.0
    impact["max_yield_pct"] = impact["max_predicted_yield"] * 100.0
    impact_columns = [
        "feature",
        "korean",
        "adjustable",
        "effect_span_pp",
        "min_predicted_yield_value",
        "min_yield_pct",
        "max_predicted_yield_value",
        "max_yield_pct",
    ]
    print(
        impact.loc[:, impact_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print(f"\n주의: {CAUSALITY_WARNING}")
    print(f"실행시간: {elapsed:.1f}초")
    print("\n생성 파일:")
    for name, path in result.paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
