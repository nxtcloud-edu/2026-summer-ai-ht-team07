"""모델 학습 CLI — `make train` 진입점.

사용::

    python scripts/train.py            # 4종 비교 + OOF 임계값 선택
    python scripts/train.py --fast     # 호환용 별칭 (동일한 검증 실행)

소유자: C(데이터·모델).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yeda.models.train import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="YEDA 모델 학습 및 4종 비교")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="이전 CLI 호환용 별칭 (OOF 임계값 선택은 항상 실행)",
    )
    args = parser.parse_args()

    if args.fast:
        print("주의: --fast는 호환용이며 임계값 누수 방지를 위한 OOF 검증은 생략하지 않습니다.")
    comparison = run()
    columns = [
        c
        for c in (
            "label",
            "threshold",
            "accuracy",
            "pr_auc",
            "recall",
            "f1_macro",
            "brier",
            "ece",
        )
        if c in comparison.columns
    ]
    print(comparison[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if comparison.attrs.get("monotone_verified"):
        count = comparison.attrs.get("monotone_feature_count", "?")
        print(f"\nLightGBM 단조 제약 설정 및 전체 {count}개 피처 grid 검증: PASS")
    print("\n비교표 저장: artifacts/metrics/model_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
