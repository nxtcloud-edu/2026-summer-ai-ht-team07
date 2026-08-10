"""스키마 · 전처리 · 텍스트 유틸 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yeda.data.preprocess import apply_imputer, fit_imputer, row_to_frame
from yeda.schema import (
    FEATURE_NAMES,
    RAW_COLUMNS,
    RISK_LEVELS,
    SPEC_BY_NAME,
    TARGET,
    validate_frame,
)
from yeda.text_utils import eul_reul, format_value, has_final_consonant


def _frame(n: int = 10) -> pd.DataFrame:
    """스키마를 만족하는 최소 DataFrame."""
    data = {
        name: np.full(n, (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2)
        for name in FEATURE_NAMES
    }
    data[TARGET] = np.tile([0, 1], n // 2)
    return pd.DataFrame(data)


def test_feature_names_unique_and_ordered() -> None:
    """피처명이 중복 없이 유일하며 RAW_COLUMNS 와 정합적이다."""
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES) == 13
    assert set(FEATURE_NAMES).issubset(set(RAW_COLUMNS))


def test_every_feature_has_rationale() -> None:
    """모든 변수에 물리적 근거가 적혀 있다.

    발표 Q&A에서 "이 변수는 왜 넣었나요"에 답하지 못하는 변수가 있으면 안 된다.
    """
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        assert len(spec.rationale) > 20, f"{name}: 근거 설명이 너무 짧습니다"
        assert spec.korean, f"{name}: 한글명이 없습니다"
        assert spec.low < spec.high, f"{name}: 범위가 잘못되었습니다"
        assert spec.resolution > 0, f"{name}: 분해능이 0 이하입니다"


def test_validate_detects_missing_column() -> None:
    """필수 피처가 빠지면 오류로 잡는다."""
    df = _frame().drop(columns=["uv_time"])
    report = validate_frame(df)
    assert not report.ok
    assert any("uv_time" in error for error in report.errors)


def test_validate_detects_bad_target() -> None:
    """타겟에 0/1 이외 값이 있으면 오류로 잡는다."""
    df = _frame()
    df.loc[0, TARGET] = 7
    assert not validate_frame(df).ok


def test_validate_out_of_range_is_warning_not_error() -> None:
    """범위 이탈은 경고일 뿐 오류가 아니다 (데모 중 예외로 죽지 않게)."""
    df = _frame()
    df.loc[0, "uv_time"] = 99.0
    report = validate_frame(df)
    assert report.ok
    assert any("uv_time" in warning for warning in report.warnings)


def test_validate_without_target() -> None:
    """추론 입력(라벨 없음)도 검증할 수 있다."""
    df = _frame().drop(columns=[TARGET])
    assert validate_frame(df, require_target=False).ok


def test_imputer_roundtrip() -> None:
    """결측이 학습 세트 통계로 채워지고 피처 순서가 정렬된다."""
    df = _frame(20)[list(FEATURE_NAMES)]
    df.loc[0, "humidity"] = np.nan
    values = fit_imputer(df)
    filled = apply_imputer(df, values)

    assert filled.isna().sum().sum() == 0
    assert list(filled.columns) == list(FEATURE_NAMES)


def test_row_to_frame_orders_columns() -> None:
    """UI dict 가 뒤죽박죽 순서로 와도 모델 입력 순서로 정렬된다.

    이 정렬이 없으면 조용히 틀린 예측이 나온다 — 가장 잡기 어려운 버그다.
    """
    df = _frame(10)[list(FEATURE_NAMES)]
    values = fit_imputer(df)
    shuffled = {name: values[name] for name in reversed(FEATURE_NAMES)}
    assert list(row_to_frame(shuffled, values).columns) == list(FEATURE_NAMES)


def test_row_to_frame_fills_absent_feature() -> None:
    """일부 피처가 빠져 있어도 대치값으로 채워 1행을 만든다."""
    df = _frame(10)[list(FEATURE_NAMES)]
    values = fit_imputer(df)
    frame = row_to_frame({"uv_time": 5.0}, values)
    assert len(frame) == 1
    assert frame.isna().sum().sum() == 0


def test_risk_levels_defined() -> None:
    assert RISK_LEVELS == ("critical", "warning", "normal")


@pytest.mark.parametrize(
    ("word", "expected"),
    [("핀 압력", True), ("UV 조사 시간", True), ("온도", False), ("29N", False)],
)
def test_has_final_consonant(word: str, expected: bool) -> None:
    assert has_final_consonant(word) is expected


def test_josa_selection() -> None:
    """조사가 받침에 맞게 선택된다 (발표 화면 완성도)."""
    assert eul_reul("핀 압력") == "핀 압력을"
    assert eul_reul("공정 온도") == "공정 온도를"


def test_format_value_omits_unitless() -> None:
    assert format_value(29.5, "N") == "29.5N"
    assert format_value(1.0, "-") == "1"
