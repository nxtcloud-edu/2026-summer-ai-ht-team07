"""최적화 계약 검증.

학습된 모델 없이 **가짜 점수 함수**로 검사한다. 모델 학습을 기다리지 않으므로
D(설명·최적화)가 C(데이터·모델)와 독립적으로 개발할 수 있다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yeda.io_utils import load_config
from yeda.optimize.search import (
    local_search,
    random_search,
    recommend,
    tunable_features,
)
from yeda.schema import ADJUSTABLE, BOUNDS, FEATURE_NAMES, FIXED, SPEC_BY_NAME


def fake_score(X: pd.DataFrame) -> np.ndarray:
    """가짜 대리 모델 — uv_time 과 pin_pressure 가 높을수록 좋다고 본다.

    실제 물리와 무관하다. 최적화기가 **제약을 지키는지**만 검사하는 것이 목적이다.
    """
    uv = (X["uv_time"] - 3.0) / 4.0
    pressure = (X["pin_pressure"] - 20.0) / 20.0
    thickness = (X["die_thickness"] - 100.0) / 50.0
    return np.clip(0.2 + 0.35 * uv + 0.35 * pressure + 0.3 * thickness, 0.0, 1.0).to_numpy()


@pytest.fixture
def baseline() -> dict[str, float]:
    """모든 변수를 범위 중앙에 둔 기준 조건."""
    return {
        name: (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2 for name in FEATURE_NAMES
    }


def test_tunable_equals_schema_adjustable() -> None:
    """기본 탐색 대상은 스키마의 조정 가능 변수와 같다."""
    assert set(tunable_features()) == set(ADJUSTABLE)


def test_tunable_override_cannot_widen() -> None:
    """설정으로 고정 변수를 탐색 대상에 끼워 넣을 수 없다.

    구조적으로 막지 않으면 언젠가 누군가 YAML 한 줄로 뚫는다.
    """
    cfg = {**load_config("optimize"), "tunable_override": ["die_thickness"]}
    with pytest.raises(ValueError, match="조정 불가"):
        tunable_features(cfg)


def test_fixed_features_never_change(baseline: dict[str, float]) -> None:
    """**핵심 계약** — 최적화가 제품 사양·자재·설비 상태를 절대 바꾸지 않는다.

    ``die_thickness`` 를 올리라는 제안은 "제품을 바꾸라"는 말과 같다.
    ``fake_score`` 는 일부러 die_thickness 를 좋아하도록 만들었는데도 바뀌면 안 된다.
    """
    result = recommend(fake_score, baseline)
    for name in FIXED:
        assert result.suggested[name] == baseline[name], f"{name} 이 변경되었습니다"
    assert not set(result.recommendations["feature"]) & set(FIXED)


def test_suggestions_within_physical_bounds(baseline: dict[str, float]) -> None:
    """제안값이 물리적 허용 범위를 벗어나지 않는다."""
    result = recommend(fake_score, baseline)
    for name, value in result.suggested.items():
        low, high = BOUNDS[name]
        assert low - 1e-9 <= value <= high + 1e-9, f"{name}={value} 범위 이탈"


def test_suggestions_snapped_to_resolution(baseline: dict[str, float]) -> None:
    """제안값이 설비 설정 분해능에 맞춰진다.

    "핀 압력 37.8213N" 같은 제안은 실무자가 설정할 수 없다.
    """
    result = recommend(fake_score, baseline)
    for _, row in result.recommendations.iterrows():
        spec = SPEC_BY_NAME[row["feature"]]
        ratio = row["suggested_value"] / spec.resolution
        assert abs(ratio - round(ratio)) < 1e-6, f"{row['feature']} 이 분해능에 맞지 않습니다"


def test_max_changed_features_respected(baseline: dict[str, float]) -> None:
    """한 번에 제안하는 변수 개수가 상한을 넘지 않는다 (현장 수용성)."""
    cfg = load_config("optimize")
    result = recommend(fake_score, baseline, cfg)
    assert len(result.recommendations) <= int(cfg["constraints"]["max_changed_features"])


def test_gain_is_non_negative(baseline: dict[str, float]) -> None:
    """제안이 현재 조건보다 나쁘지 않다."""
    result = recommend(fake_score, baseline)
    assert result.optimized_prob >= result.baseline_prob - 1e-9
    assert result.gain_pp >= -1e-9


def test_min_gain_filter(baseline: dict[str, float]) -> None:
    """노이즈 수준의 개선은 제안 목록에서 걸러진다."""
    cfg = load_config("optimize")
    result = recommend(fake_score, baseline, cfg)
    minimum = float(cfg["constraints"]["min_gain_pp"])
    assert all(result.recommendations["expected_gain_pp"] >= minimum)


def test_recommend_requires_all_features() -> None:
    """피처가 빠지면 조용히 넘어가지 않고 명확히 실패한다."""
    with pytest.raises(KeyError):
        recommend(fake_score, {"uv_time": 5.0})


def test_local_search_is_deterministic(baseline: dict[str, float]) -> None:
    """좌표 하강이 결정론적이다 — 리허설과 본 시연 결과가 같아야 한다."""
    first, first_score, _ = local_search(fake_score, baseline)
    second, second_score, _ = local_search(fake_score, baseline)
    assert first == second
    assert first_score == pytest.approx(second_score)


def test_random_search_respects_fixed(baseline: dict[str, float]) -> None:
    """랜덤 서치도 고정 변수를 건드리지 않는다."""
    best, _, _ = random_search(fake_score, baseline, seed=1)
    for name in FIXED:
        assert best[name] == baseline[name]


def test_move_limit_respected(baseline: dict[str, float]) -> None:
    """현장 수용성 제약(한 번에 움직일 수 있는 폭)이 지켜진다."""
    cfg = load_config("optimize")
    max_move = float(cfg["constraints"]["max_relative_move"])
    best, _, _ = local_search(fake_score, baseline, cfg)
    for name in tunable_features(cfg):
        low, high = BOUNDS[name]
        moved = abs(best[name] - baseline[name])
        assert moved <= (high - low) * max_move + 1e-6, f"{name} 이동 폭 초과"
