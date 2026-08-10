"""데이터 생성기 검증.

기존 프로토타입의 결함(결정론적 AND 규칙, 99.86% 정확도)이 **재발하지 않는지**
자동으로 지킨다. 특히 ``test_label_is_not_deterministic`` 과
``test_no_single_rule_reproduces_target`` 은 그 결함을 정확히 겨냥한 테스트다.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from yeda.data.generator import _sample_conditions, generate
from yeda.data.physics import bayes_accuracy, latent_score, sigmoid
from yeda.io_utils import load_config
from yeda.schema import FEATURE_NAMES, MONOTONE_CONSTRAINTS, SPEC_BY_NAME, TARGET, validate_frame


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config("data_gen")


@pytest.fixture(scope="module")
def small_config(config: dict) -> dict:
    """테스트 속도를 위해 표본을 줄인 설정."""
    cfg = {**config, "n_samples": 2000}
    return cfg


@pytest.fixture(scope="module")
def dataset(small_config: dict):
    return generate(small_config)


def test_schema_contract(dataset) -> None:
    """생성 결과가 스키마 계약을 지킨다."""
    df, _ = dataset
    report = validate_frame(df)
    assert report.ok, report.errors


def test_reproducible(small_config: dict) -> None:
    """같은 시드는 같은 데이터를 만든다."""
    first, _ = generate(small_config)
    second, _ = generate(small_config)
    assert first.equals(second)


def test_configured_target_ranges(dataset, config: dict) -> None:
    """성공률과 베이즈 정확도가 YAML 에 선언된 목표 대역 안에 있다."""
    _, report = dataset
    for metric in ("success_rate", "bayes_accuracy"):
        low, high = config["targets"][metric]
        value = getattr(report, metric)
        assert low <= value <= high, f"{metric}={value:.3f}, target=[{low}, {high}]"


def test_recipe_sampling_is_truncated_and_deterministic(small_config: dict) -> None:
    """레시피 표본은 경계에 쌓이지 않는 진짜 절단정규분포이고 재현 가능하다.

    산포를 의도적으로 크게 잡아 ``normal + clip`` 구현이라면 양 끝점에 상당한
    질량이 생기게 한다. 거절 샘플링 구현에서는 연속분포 표본이 정확히 경계값과
    같아질 확률이 0 이므로 경계 pile-up 이 없어야 한다.
    """
    cfg = deepcopy(small_config)
    cfg["n_samples"] = 4096
    cfg["sampling"]["recipe_ratio"] = 1.0
    cfg["sampling"]["spread_ratio"] = 0.50

    first = _sample_conditions(cfg, np.random.default_rng(20260810))
    second = _sample_conditions(cfg, np.random.default_rng(20260810))

    for name in FEATURE_NAMES:
        np.testing.assert_array_equal(first[name], second[name])
        if name == "tape_type":
            continue
        spec = SPEC_BY_NAME[name]
        assert np.all(first[name] > spec.low), f"{name} 하한에 표본이 쌓였습니다"
        assert np.all(first[name] < spec.high), f"{name} 상한에 표본이 쌓였습니다"


def test_label_is_not_deterministic(small_config: dict) -> None:
    """**핵심 회귀 테스트** — 라벨이 확률적이어야 한다.

    베이즈 정확도가 0.95 를 넘으면 사실상 결정론적 규칙이 된 것이고,
    이는 기존 프로토타입(99.86%)과 똑같은 실패다.
    """
    _, report = generate(small_config)
    assert report.bayes_accuracy < 0.95, (
        f"베이즈 정확도 {report.bayes_accuracy:.3f} — 라벨이 결정론에 가깝습니다. "
        "latent.scale 을 낮추세요."
    )


def test_same_condition_yields_different_labels(small_config: dict) -> None:
    """동일한 공정 조건에서도 결과가 갈린다 (베르누이 샘플링 확인)."""
    rng = np.random.default_rng(0)
    fixed = {
        name: np.full(500, (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2)
        for name in FEATURE_NAMES
    }
    p = sigmoid(latent_score(fixed, small_config))
    labels = (rng.random(len(p)) < p).astype(int)
    assert 0 < labels.mean() < 1, "동일 조건에서 라벨이 한쪽으로만 나옵니다"


def test_no_single_rule_reproduces_target(dataset) -> None:
    """**핵심 회귀 테스트** — 단일 임계값 규칙으로 타겟이 재현되지 않는다.

    기존 데이터는 4개 부등식의 AND 로 5,000행 전부가 맞아떨어졌다.
    여기서는 어떤 변수의 어떤 단일 임계값으로도 정확도 90%를 넘지 못해야 한다.
    """
    df, _ = dataset
    y = df[TARGET].to_numpy()
    for name in FEATURE_NAMES:
        column = df[name].to_numpy(dtype=float)
        valid = ~np.isnan(column)
        values, labels = column[valid], y[valid]
        for threshold in np.percentile(values, [10, 25, 50, 75, 90]):
            for rule in ((values >= threshold), (values <= threshold)):
                accuracy = float((rule.astype(int) == labels).mean())
                assert accuracy < 0.90, (
                    f"{name} 단일 규칙(임계값 {threshold:.3g})이 정확도 {accuracy:.3f} 를 냅니다. "
                    "결정론적 규칙이 남아 있습니다."
                )


def test_all_features_contribute(small_config: dict) -> None:
    """13개 변수 전부가 잠재 점수에 기여한다 (순수 난수 변수가 없다)."""
    df, _ = generate(small_config)
    columns = {name: df[name].fillna(df[name].median()).to_numpy() for name in FEATURE_NAMES}
    _, parts = latent_score(columns, small_config, return_parts=True)

    for name in FEATURE_NAMES:
        assert name in parts, f"{name} 의 기여항이 없습니다"
        assert np.std(parts[name]) > 1e-6, f"{name} 의 기여가 상수입니다 (사실상 무기여)"


def test_interactions_present(small_config: dict) -> None:
    """현재 YAML 계약의 상호작용항 정확히 3개가 모두 실제로 작동한다."""
    df, _ = generate(small_config)
    columns = {name: df[name].fillna(df[name].median()).to_numpy() for name in FEATURE_NAMES}
    _, parts = latent_score(columns, small_config, return_parts=True)

    expected = set(small_config["interactions"])
    actual = set(parts) - set(FEATURE_NAMES)
    assert len(expected) == 3, f"현재 설정의 상호작용항은 정확히 3개여야 합니다: {sorted(expected)}"
    assert actual == expected, f"설정/실행 상호작용 불일치: expected={expected}, actual={actual}"
    for key in expected:
        assert np.std(parts[key]) > 1e-6, f"{key} 가 상수입니다"


def test_monotonicity_matches_schema(small_config: dict) -> None:
    """생성기의 기여 방향이 스키마의 단조 제약과 어긋나지 않는다.

    어긋나면 LightGBM 의 단조 제약이 데이터와 싸우게 되어 성능이 떨어진다.
    변수 하나만 움직여 잠재 점수 변화 방향을 직접 확인한다.
    """
    baseline = {
        name: np.full(64, (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2)
        for name in FEATURE_NAMES
    }

    for name, expected in zip(FEATURE_NAMES, MONOTONE_CONSTRAINTS):
        if expected == 0:
            continue
        spec = SPEC_BY_NAME[name]
        grid = np.linspace(spec.low, spec.high, 64)
        columns = {**baseline, name: grid}
        # tape_type 상호작용은 두 수준 모두에서 성립해야 한다.
        for tape in (0.0, 1.0):
            columns_variant = {**columns, "tape_type": np.full(64, tape)}
            if name == "tape_type":
                columns_variant = columns
            scores = latent_score(columns_variant, small_config)
            diffs = np.diff(scores)
            direction = np.sign(expected)
            assert np.all(diffs * direction >= -1e-9), (
                f"{name}: 단조 제약 {expected} 와 생성기 방향이 어긋납니다 (tape_type={tape})"
            )


def test_missing_values_injected(dataset, config: dict) -> None:
    """결측이 1~3% 수준으로 들어가 있고, 타겟에는 없다."""
    df, report = dataset
    assert 0.005 < report.missing_rate < 0.05, f"결측률 {report.missing_rate:.3f}"
    assert df[TARGET].isna().sum() == 0, "타겟에 결측이 있습니다"


def test_values_are_rounded(dataset) -> None:
    """관측값이 센서 분해능으로 반올림되어 있다 (소수점 9자리 잔여 없음)."""
    df, _ = dataset
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        values = df[name].dropna().to_numpy(dtype=float)
        remainder = np.abs(values / spec.resolution - np.round(values / spec.resolution))
        assert np.all(remainder < 1e-6), f"{name} 이 분해능 {spec.resolution} 로 반올림되지 않았습니다"


def test_values_within_bounds(dataset) -> None:
    """모든 관측값이 물리적 허용 범위 안에 있다."""
    df, _ = dataset
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        values = df[name].dropna()
        assert values.min() >= spec.low - 1e-9, f"{name} 하한 이탈"
        assert values.max() <= spec.high + 1e-9, f"{name} 상한 이탈"


def test_bayes_accuracy_helper() -> None:
    """베이즈 정확도 계산이 정의대로 동작한다."""
    assert bayes_accuracy(np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert bayes_accuracy(np.array([1.0, 0.0])) == pytest.approx(1.0)
