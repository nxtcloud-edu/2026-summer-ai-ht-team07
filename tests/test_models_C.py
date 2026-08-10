"""C 소유 모델 번들·임계값·단조 제약의 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_recall_curve, recall_score

from yeda.models.evaluate import save_calibration_plot, select_decision_threshold
from yeda.models.registry import ModelBundle, verify_all_monotone_predictions
from yeda.schema import BOUNDS, FEATURE_NAMES, MONOTONE_CONSTRAINTS


class RecordingEstimator:
    """번들이 모델에 넘긴 최종 DataFrame을 기록하는 테스트 대역."""

    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.seen = X.copy()
        success = np.full(len(X), 0.75, dtype=float)
        return np.column_stack((1.0 - success, success))


class SchemaMonotoneEstimator:
    """스키마 방향대로만 움직이는 결정적 확률 모델 테스트 대역."""

    def get_params(self, deep: bool = True) -> dict[str, list[int]]:
        del deep
        return {"monotone_constraints": list(MONOTONE_CONSTRAINTS)}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(X), dtype=float)
        for feature, direction in zip(FEATURE_NAMES, MONOTONE_CONSTRAINTS):
            if direction == 0:
                continue
            low, high = BOUNDS[feature]
            scaled = (X[feature].to_numpy(dtype=float) - low) / (high - low)
            score += 0.2 * direction * scaled
        success = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack((1.0 - success, success))


def test_bundle_imputes_missing_and_nan_then_restores_feature_order() -> None:
    model = RecordingEstimator()
    bundle = ModelBundle(
        model=model,
        name="recording",
        feature_names=["second", "first"],
        imputer_values={"first": 11.0, "second": 22.0},
    )
    raw = pd.DataFrame({"first": [np.nan, 5.0], "ignored": [99.0, 98.0]})

    probability = bundle.predict_proba(raw)

    assert probability.tolist() == [0.75, 0.75]
    assert model.seen is not None
    assert model.seen.columns.tolist() == ["second", "first"]
    assert model.seen.to_numpy().tolist() == [[22.0, 11.0], [22.0, 5.0]]
    assert not model.seen.isna().any().any()


def test_threshold_selector_enforces_recall_and_breaks_ties_deterministically() -> None:
    # threshold 0.9와 0.7의 macro-F1이 같고, 0.7의 recall이 더 크다.
    target = np.array([1, 0, 1, 0])
    probability = np.array([0.9, 0.8, 0.7, 0.6])

    first = select_decision_threshold(target, probability, min_recall=0.5)
    second = select_decision_threshold(target, probability, min_recall=0.5)

    assert first == second
    assert first.threshold == pytest.approx(0.7)
    assert first.recall == pytest.approx(1.0)
    assert first.recall >= first.min_recall
    assert first.candidate_count == 4
    assert first.feasible_candidate_count == 4


def test_vectorized_threshold_selector_matches_brute_force() -> None:
    rng = np.random.default_rng(20260810)
    target = np.array([0, 1] * 40)
    probability = rng.uniform(0.01, 0.99, size=len(target))
    min_recall = 0.75
    _, _, thresholds = precision_recall_curve(target, probability)
    brute_candidates = []
    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        recall = recall_score(target, prediction, zero_division=0)
        if recall + 1e-12 < min_recall:
            continue
        brute_candidates.append(
            (
                f1_score(target, prediction, average="macro", zero_division=0),
                recall,
                float(threshold),
            )
        )
    expected = max(brute_candidates, key=lambda row: (row[0], row[1], -row[2]))

    selected = select_decision_threshold(target, probability, min_recall=min_recall)

    assert selected.threshold == pytest.approx(expected[2])
    assert selected.objective_value == pytest.approx(expected[0])
    assert selected.recall == pytest.approx(expected[1])


@pytest.mark.parametrize(
    ("target", "probability", "min_recall"),
    [
        (np.array([1, 1]), np.array([0.2, 0.8]), 0.5),
        (np.array([0, 1]), np.array([0.2, 1.2]), 0.5),
        (np.array([0, 1]), np.array([0.2, 0.8]), 1.1),
    ],
)
def test_threshold_selector_rejects_invalid_input(
    target: np.ndarray,
    probability: np.ndarray,
    min_recall: float,
) -> None:
    with pytest.raises(ValueError):
        select_decision_threshold(target, probability, min_recall=min_recall)


def test_all_nonzero_schema_constraints_are_grid_verified() -> None:
    imputer_values = {
        feature: float(sum(BOUNDS[feature]) / 2.0) for feature in FEATURE_NAMES
    }
    bundle = ModelBundle(
        model=SchemaMonotoneEstimator(),
        name="schema_monotone_test",
        feature_names=list(FEATURE_NAMES),
        imputer_values=imputer_values,
    )

    report = verify_all_monotone_predictions(bundle, n_points=17)
    expected = {
        feature
        for feature, direction in zip(FEATURE_NAMES, MONOTONE_CONSTRAINTS)
        if direction != 0
    }

    assert report["passed"] is True
    assert report["constrained_feature_count"] == len(expected) == 10
    assert {row["feature"] for row in report["features"]} == expected
    assert all(row["passed"] for row in report["features"])


def test_calibration_plot_is_written(tmp_path) -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.3, 0.7, 0.9])
    output = tmp_path / "calibration.png"

    returned = save_calibration_plot(target, probability, output, n_bins=4)

    assert returned == output
    assert output.is_file()
    assert output.stat().st_size > 0
