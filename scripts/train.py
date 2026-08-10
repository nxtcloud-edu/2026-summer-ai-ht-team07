"""모델 학습 CLI — `make train` 진입점.

사용::

    python scripts/train.py            # 4종 비교 + 교차검증
    python scripts/train.py --fast     # 교차검증 생략 (시간이 급할 때)

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
    parser.add_argument("--fast", action="store_true", help="교차검증 생략")
    args = parser.parse_args()

    comparison = run(with_cv=not args.fast)
    columns = [
        c for c in ("label", "accuracy", "pr_auc", "recall", "f1_macro", "brier", "ece")
        if c in comparison.columns
    ]
    print(comparison[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if comparison.attrs.get("monotone_verified"):
        print("\nLightGBM 단조 제약 설정 및 pin_speed 예측 방향 검증: PASS")
    print("\n비교표 저장: artifacts/metrics/model_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
