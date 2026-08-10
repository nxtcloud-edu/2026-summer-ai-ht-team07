"""학습 파이프라인: 4종 비교 → 본선 모델 저장.

``make train`` 의 진입점. 실행하면 다음이 만들어진다::

    artifacts/models/primary_model.joblib     서빙용 번들
    artifacts/metrics/model_comparison.csv    4종 비교표 (발표자료 직결)
    artifacts/metrics/model_metrics.json      전체 지표
    artifacts/metrics/calibration_curve.csv   신뢰도 곡선 데이터
    artifacts/metrics/monotone_verification.json  0이 아닌 모든 단조 제약 grid 검증
    artifacts/figures/calibration_curve.png   홀드아웃 신뢰도 곡선

소유자: C(데이터·모델).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..io_utils import load_config, resolve, save_json
from ..schema import FEATURE_NAMES, MONOTONE_CONSTRAINTS
from ..data.preprocess import Split, make_split
from .evaluate import (
    calibration_table,
    comparison_frame,
    compute_metrics,
    expected_calibration_error,
    save_calibration_plot,
    select_decision_threshold,
)
from .registry import (
    ModelBundle,
    build_model,
    save_bundle,
    verify_all_monotone_predictions,
)


def train_one(name: str, split: Split, config: dict[str, Any]) -> tuple[Any, np.ndarray]:
    """모델 하나를 학습하고 홀드아웃 확률을 반환한다.

    Args:
        name: 모델 키.
        split: 전처리된 학습/홀드아웃 세트.
        config: ``configs/model.yaml``.

    Returns:
        ``(학습된 모델, 홀드아웃 성공확률 배열)``.
    """
    model = build_model(name, config)
    model.fit(split.X_train, split.y_train)
    y_prob = model.predict_proba(split.X_test)[:, 1]
    return model, y_prob


def cross_val_probabilities(name: str, split: Split, config: dict[str, Any]) -> np.ndarray:
    """학습 세트에 대한 교차검증 예측 확률.

    홀드아웃은 한 번만 쓰는 최종 보고용이므로, 임계값 조정이나 신뢰도 점검처럼
    반복해서 봐야 하는 작업은 교차검증 예측으로 한다.

    Args:
        name: 모델 키.
        split: 전처리된 세트.
        config: 모델 설정.

    Returns:
        학습 세트 순서와 같은 길이의 확률 배열.
    """
    cv = StratifiedKFold(
        n_splits=int(config["cv_folds"]), shuffle=True, random_state=int(config["seed"])
    )
    model = build_model(name, config)
    probs = cross_val_predict(model, split.X_train, split.y_train, cv=cv, method="predict_proba")
    return probs[:, 1]


def run(config: dict[str, Any] | None = None, *, with_cv: bool = True) -> pd.DataFrame:
    """전체 학습·비교·저장 파이프라인.

    Args:
        config: 모델 설정. None 이면 ``configs/model.yaml``.
        with_cv: 이전 CLI 호환 인자. 임계값 누수를 막기 위해 False여도 OOF
            교차검증은 반드시 실행한다.

    Returns:
        모델 비교표 DataFrame.

    Raises:
        FileNotFoundError: 데이터가 없을 때. 먼저 ``make data``.
    """
    cfg = config if config is not None else load_config("model")
    if not with_cv:
        warnings.warn(
            "OOF 임계값 선택은 필수이므로 with_cv=False를 무시합니다.",
            UserWarning,
            stacklevel=2,
        )
    split = make_split(test_size=float(cfg["test_size"]), seed=int(cfg["seed"]))
    threshold_cfg = cfg["threshold_selection"]
    objective = str(threshold_cfg["objective"])
    min_recall = float(threshold_cfg["min_recall"])

    enabled = [k for k, v in cfg["models"].items() if v.get("enabled", True)]
    labels = {k: cfg["models"][k].get("label", k) for k in enabled}

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for name in enabled:
        # 임계값은 학습 세트 OOF 확률만 보고 먼저 확정한다. 아래 홀드아웃
        # 확률은 이 선택 과정에 절대 들어가지 않는다.
        oof_prob = cross_val_probabilities(name, split, cfg)
        selected = select_decision_threshold(
            split.y_train.to_numpy(),
            oof_prob,
            min_recall=min_recall,
            objective=objective,
        )

        model, y_prob = train_one(name, split, cfg)
        metrics = compute_metrics(split.y_test, y_prob, selected.threshold)
        oof_metrics = compute_metrics(split.y_train, oof_prob, selected.threshold)
        metrics.update(
            {
                "threshold": float(selected.threshold),
                "oof_pr_auc": oof_metrics["pr_auc"],
                "oof_recall": oof_metrics["recall"],
                "oof_f1_macro": oof_metrics["f1_macro"],
                "oof_brier": oof_metrics["brier"],
            }
        )

        table = calibration_table(split.y_test, y_prob, int(cfg["evaluation"]["calibration_bins"]))
        metrics["ece"] = expected_calibration_error(table)

        results[name] = metrics
        fitted[name] = model
        probabilities[name] = y_prob
        thresholds[name] = float(selected.threshold)

    primary = cfg["primary_model"]
    if primary not in fitted:
        raise ValueError(f"primary_model 이 활성 모델 목록에 없습니다: {primary}")

    cal = calibration_table(
        split.y_test, probabilities[primary], int(cfg["evaluation"]["calibration_bins"])
    )
    bundle = ModelBundle(
        model=fitted[primary],
        name=primary,
        feature_names=list(FEATURE_NAMES),
        imputer_values=split.imputer_values,
        threshold=thresholds[primary],
        metrics=results[primary],
    )

    verification = cfg.get("verification", {})
    monotone_names = [
        name for name in enabled if cfg["models"][name].get("apply_monotone", False)
    ]
    if not monotone_names:
        raise ValueError("활성 모델 중 단조 제약 검증 대상이 없습니다.")
    monotone_name = primary if primary in monotone_names else monotone_names[0]
    monotone_bundle = ModelBundle(
        model=fitted[monotone_name],
        name=monotone_name,
        feature_names=list(FEATURE_NAMES),
        imputer_values=split.imputer_values,
        threshold=thresholds[monotone_name],
        metrics=results[monotone_name],
    )
    monotone_report = verify_all_monotone_predictions(
        monotone_bundle,
        n_points=int(verification["monotone_grid_points"]),
        atol=float(verification["monotone_atol"]),
    )
    verification_scope = str(verification.get("monotone_features", "all_nonzero"))
    if verification_scope != "all_nonzero":
        raise ValueError("verification.monotone_features 는 all_nonzero 여야 합니다.")
    expected_count = sum(direction != 0 for direction in MONOTONE_CONSTRAINTS)
    monotone_report["expected_nonzero_features"] = int(expected_count)
    monotone_report["feature_count_matches"] = (
        monotone_report["constrained_feature_count"] == expected_count
    )
    monotone_report["passed"] = bool(
        monotone_report["passed"] and monotone_report["feature_count_matches"]
    )
    save_json(monotone_report, verification["output_json"])
    if not monotone_report["passed"]:
        failed = [
            item["feature"] for item in monotone_report["features"] if not item["passed"]
        ]
        raise AssertionError(
            "LightGBM 전체 단조 제약 검증 실패: "
            f"constraints_match={monotone_report['constraints_match_schema']}, "
            f"count={monotone_report['constrained_feature_count']}/{expected_count}, "
            f"failed={failed}"
        )

    comparison = comparison_frame(results, labels)
    comparison.attrs["monotone_verified"] = True
    comparison.attrs["monotone_feature_count"] = int(
        monotone_report["constrained_feature_count"]
    )
    comparison.to_csv(resolve(cfg["evaluation"]["comparison_csv"]), index=False, encoding="utf-8")
    save_json(results, cfg["evaluation"]["detail_json"])
    cal.to_csv(resolve(cfg["evaluation"]["calibration_csv"]), index=False, encoding="utf-8")
    save_calibration_plot(
        split.y_test.to_numpy(),
        probabilities[primary],
        resolve(cfg["evaluation"]["calibration_figure"]),
        n_bins=int(cfg["evaluation"]["calibration_bins"]),
        label=primary,
        dpi=int(cfg["evaluation"]["figure_dpi"]),
    )
    save_bundle(bundle, cfg)
    return comparison


def main() -> None:
    """CLI 진입점 — 비교표를 콘솔에 출력한다."""
    comparison = run()
    cols = [c for c in ("label", "accuracy", "pr_auc", "recall", "f1_macro", "brier", "ece") if c in comparison.columns]
    print(comparison[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
