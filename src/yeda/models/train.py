"""학습 파이프라인: 3종 비교 → 본선 모델 저장.

``make train`` 의 진입점. 실행하면 다음이 만들어진다::

    artifacts/models/primary_model.joblib     서빙용 번들
    artifacts/metrics/model_comparison.csv    3종 비교표 (발표자료 직결)
    artifacts/metrics/model_metrics.json      전체 지표
    artifacts/metrics/calibration_curve.csv   신뢰도 곡선 데이터

소유자: C(데이터·모델).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..io_utils import load_config, resolve, save_json
from ..schema import FEATURE_NAMES
from ..data.preprocess import Split, make_split
from .evaluate import (
    calibration_table,
    comparison_frame,
    compute_metrics,
    expected_calibration_error,
)
from .registry import ModelBundle, build_model, save_bundle


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
        with_cv: True 면 교차검증 PR-AUC 도 비교표에 추가한다.
            시간이 급하면 False 로 두면 된다(홀드아웃 지표만으로도 발표는 가능).

    Returns:
        모델 비교표 DataFrame.

    Raises:
        FileNotFoundError: 데이터가 없을 때. 먼저 ``make data``.
    """
    cfg = config if config is not None else load_config("model")
    split = make_split(test_size=float(cfg["test_size"]), seed=int(cfg["seed"]))
    threshold = float(cfg["decision_threshold"])

    enabled = [k for k, v in cfg["models"].items() if v.get("enabled", True)]
    labels = {k: cfg["models"][k].get("label", k) for k in enabled}

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}

    for name in enabled:
        model, y_prob = train_one(name, split, cfg)
        metrics = compute_metrics(split.y_test, y_prob, threshold)

        if with_cv:
            cv_prob = cross_val_probabilities(name, split, cfg)
            cv_metrics = compute_metrics(split.y_train, cv_prob, threshold)
            metrics["cv_pr_auc"] = cv_metrics["pr_auc"]
            metrics["cv_accuracy"] = cv_metrics["accuracy"]

        table = calibration_table(split.y_test, y_prob, int(cfg["evaluation"]["calibration_bins"]))
        metrics["ece"] = expected_calibration_error(table)

        results[name] = metrics
        fitted[name] = model
        probabilities[name] = y_prob

    comparison = comparison_frame(results, labels)
    comparison.to_csv(resolve(cfg["evaluation"]["comparison_csv"]), index=False, encoding="utf-8")
    save_json(results, cfg["evaluation"]["detail_json"])

    primary = cfg["primary_model"]
    cal = calibration_table(
        split.y_test, probabilities[primary], int(cfg["evaluation"]["calibration_bins"])
    )
    cal.to_csv(resolve(cfg["evaluation"]["calibration_csv"]), index=False, encoding="utf-8")

    bundle = ModelBundle(
        model=fitted[primary],
        name=primary,
        feature_names=list(FEATURE_NAMES),
        imputer_values=split.imputer_values,
        threshold=threshold,
        metrics=results[primary],
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
