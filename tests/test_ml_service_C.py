"""C 소유 ML 서비스 접점 회귀 테스트.

실제 모델 파일 유무와 무관하게 mock 계약을 직접 검증한다. mock은 성능 수치를
주장하는 모델이 아니지만, 피처 방향·설명 합·최적화 안전 제약은 실서비스와 같아야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.services.ml_service import MLService
from yeda.schema import ADJUSTABLE, FEATURE_NAMES, FIXED, SPEC_BY_NAME


def _mock_service() -> MLService:
    service = MLService.__new__(MLService)
    service.is_mock = True
    service.model_name = None
    service.status = "test mock"
    service._bundle = None
    service._explainer = None
    service._background = None
    return service


def _midpoint_values() -> dict[str, float]:
    return {
        name: (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2.0
        for name in FEATURE_NAMES
    }


def _poor_values() -> dict[str, float]:
    values: dict[str, float] = {}
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        if spec.monotone > 0:
            values[name] = spec.low
        elif spec.monotone < 0:
            values[name] = spec.high
        elif name in {"pin_height", "temperature"}:
            values[name] = spec.low
        else:
            values[name] = (spec.low + spec.high) / 2.0
    return values


def test_mock_state_contract() -> None:
    service = _mock_service()
    assert service.is_mock is True
    assert service.is_loaded is False


def test_mock_vacuum_sign_matches_schema() -> None:
    service = _mock_service()
    weak = _midpoint_values()
    strong = dict(weak)
    weak["head_vacuum"] = SPEC_BY_NAME["head_vacuum"].high
    strong["head_vacuum"] = SPEC_BY_NAME["head_vacuum"].low

    assert service.predict_one(strong) > service.predict_one(weak)


def test_mock_explanation_reconstructs_prediction() -> None:
    service = _mock_service()
    values = _poor_values()
    result = service.explain(values)

    reconstructed = result["base_value"] + sum(
        row["shap_value_pp"] for row in result["shap_values"]
    ) / 100.0
    assert reconstructed == pytest.approx(service.predict_one(values), abs=1e-12)
    assert result == service.explain(values)


def test_mock_uses_real_optimizer_constraints() -> None:
    service = _mock_service()
    values = _poor_values()
    result = service.optimize(values)

    assert result["gain_pp"] >= -1e-10
    assert len(result["recommendations"]) <= 3
    changed = {row["feature"] for row in result["recommendations"]}
    assert changed <= set(ADJUSTABLE)
    assert not changed & set(FIXED)

    for row in result["recommendations"]:
        spec = SPEC_BY_NAME[row["feature"]]
        max_move = (spec.high - spec.low) * 0.18
        assert abs(row["delta"]) <= max_move + spec.resolution
        assert spec.low <= row["suggested_value"] <= spec.high
        quotient = row["suggested_value"] / spec.resolution
        assert quotient == pytest.approx(round(quotient), abs=1e-8)


def test_batch_yield_and_defect_are_complements() -> None:
    service = _mock_service()
    first = _midpoint_values()
    second = _poor_values()
    rows = pd.DataFrame([first, second], columns=FEATURE_NAMES)

    probabilities = service.predict_proba(rows)
    expected_yield = float(np.mean(probabilities) * 100.0)
    assert service.predict_yield(rows) == pytest.approx(expected_yield)
    assert service.predict_defect_rate(rows) == pytest.approx(100.0 - expected_yield)
