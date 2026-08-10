"""수율 최적화 실험의 선택·단위 계약 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yeda.models.yield_experiment import (
    SELECTION_RULE,
    _posthoc_oracle_audit,
    _select_model_without_optimized_yield,
)
from yeda.schema import FEATURE_NAMES, SPEC_BY_NAME


def test_selection_is_strict_pr_auc_then_brier() -> None:
    """자기 주장 수율이나 단조 tolerance가 PR-AUC 1위를 뒤집지 못한다."""
    metrics = pd.DataFrame(
        [
            {
                "model": "higher_pr_auc",
                "holdout_pr_auc": 0.950,
                "holdout_brier": 0.130,
                "optimized_predicted_yield": 0.70,
            },
            {
                "model": "inflated_yield",
                "holdout_pr_auc": 0.946,
                "holdout_brier": 0.100,
                "optimized_predicted_yield": 0.999,
            },
        ]
    )
    selected, reason = _select_model_without_optimized_yield(metrics)
    assert selected == "higher_pr_auc"
    assert "PR-AUC" in reason
    assert "optimized_predicted_yield_not_used" in SELECTION_RULE


def test_selection_uses_brier_only_as_tie_break() -> None:
    metrics = pd.DataFrame(
        [
            {"model": "worse_brier", "holdout_pr_auc": 0.95, "holdout_brier": 0.13},
            {"model": "better_brier", "holdout_pr_auc": 0.95, "holdout_brier": 0.11},
        ]
    )
    selected, _ = _select_model_without_optimized_yield(metrics)
    assert selected == "better_brier"


def _condition(uv_time: float) -> dict[str, float]:
    condition = {
        name: (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2.0
        for name in FEATURE_NAMES
    }
    condition["uv_time"] = uv_time
    return condition


def test_posthoc_audit_keeps_probability_and_percentage_point_units() -> None:
    """yield는 0~1, gain은 %p이고 개선·악화 방향이 정확해야 한다."""
    frozen = []
    for sample_id, start, end in (("up", 4.0, 6.0), ("down", 6.0, 4.0)):
        frozen.append(
            {
                "group": "test",
                "sample_id": sample_id,
                "model": "fake",
                "actual_label": 1,
                "baseline_predicted": start / 10.0,
                "optimized_predicted": end / 10.0,
                "predicted_gain_pp": (end - start) * 10.0,
                "n_changed": 1,
                "changed_features": "uv_time",
                "n_evaluations": 2,
                "current": _condition(start),
                "suggested": _condition(end),
            }
        )

    audited = _posthoc_oracle_audit(
        frozen,
        oracle=lambda frame: 0.35 + frame["uv_time"].to_numpy(dtype=float) / 10.0,
        target_yield=0.9,
    ).set_index("sample_id")

    assert audited.loc["up", "baseline_oracle_yield"] == pytest.approx(0.75)
    assert audited.loc["up", "optimized_oracle_yield"] == pytest.approx(0.95)
    assert audited.loc["up", "oracle_gain_pp"] == pytest.approx(20.0)
    assert bool(audited.loc["up", "true_improved"])
    assert not bool(audited.loc["up", "true_worsened"])
    assert bool(audited.loc["up", "optimized_oracle_90pct_hit"])

    assert audited.loc["down", "oracle_gain_pp"] == pytest.approx(-20.0)
    assert bool(audited.loc["down", "true_worsened"])
    assert np.isfinite(audited["optimized_oracle_yield"]).all()
