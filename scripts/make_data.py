"""데이터 생성 CLI — `make data` 진입점.

사용::

    python scripts/make_data.py            # configs/data_gen.yaml 사용
    python scripts/make_data.py --check    # 생성 없이 기존 데이터 검수만
    python scripts/make_data.py --seed 20260811 --output data/raw/yeda_synthetic_seed20260811.csv

소유자: C(데이터·모델).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yeda.data.generator import generate_and_save  # noqa: E402
from yeda.data.preprocess import load_raw  # noqa: E402
from yeda.io_utils import load_config  # noqa: E402
from yeda.schema import TARGET, validate_frame  # noqa: E402


def main() -> int:
    """생성 후 검수 결과를 출력한다.

    Returns:
        종료 코드. 검수 경고가 있으면 1 (CI에서 잡히게).
    """
    parser = argparse.ArgumentParser(description="YEDA 합성 데이터 생성")
    parser.add_argument("--check", action="store_true", help="생성하지 않고 기존 데이터만 검수")
    parser.add_argument("--seed", type=int, help="설정 파일의 생성 시드를 재정의")
    parser.add_argument("--n-samples", type=int, help="설정 파일의 생성 행 수를 재정의")
    parser.add_argument("--output", help="설정 파일의 CSV 출력 경로를 재정의")
    args = parser.parse_args()

    if args.check:
        df = load_raw()
        report = validate_frame(df)
        print(f"행 수: {len(df):,} / 성공률: {df[TARGET].mean():.3f}")
        for warning in report.warnings:
            print(f"  [경고] {warning}")
        return 0 if report.ok else 1

    config = load_config("data_gen")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_samples is not None:
        if args.n_samples < 1:
            parser.error("--n-samples는 1 이상이어야 합니다")
        config["n_samples"] = args.n_samples
    if args.output is not None:
        config["output_path"] = args.output

    report = generate_and_save(config)
    print(f"생성 완료: {report.n_samples:,}행 (seed={report.seed})")
    print(f"  성공률          : {report.success_rate:.3f}")
    print(f"  베이즈 정확도 상한 : {report.bayes_accuracy:.3f}   ← 학습 정확도가 이 값을 크게 넘으면 누수 의심")
    print(f"  확률 표준편차    : {report.prob_std:.3f}")
    print(f"  결측률          : {report.missing_rate:.3f}")

    for warning in report.warnings:
        print(f"  [경고] {warning}")
    return 1 if report.warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
