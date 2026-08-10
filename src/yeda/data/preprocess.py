"""전처리: 로드 → 결측 처리 → 홀드아웃 분리.

학습 파이프라인과 추론 파이프라인이 **같은 전처리**를 쓰도록 여기 한 곳에 모은다.
UI에서 손으로 입력한 한 행도 반드시 이 모듈을 거쳐 모델로 들어간다.

소유자: C(데이터·모델).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..io_utils import load_config, resolve
from ..schema import CATEGORICAL, FEATURE_NAMES, TARGET, validate_frame


@dataclass
class Split:
    """홀드아웃 분리 결과.

    Attributes:
        X_train, y_train: 학습 세트.
        X_test, y_test: 홀드아웃(최종 보고용). **튜닝에 절대 쓰지 않는다.**
        imputer_values: 학습 세트에서 계산한 대치값. 추론 시 그대로 재사용한다.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    imputer_values: dict[str, float]


def load_raw(path: str | None = None) -> pd.DataFrame:
    """생성된 원시 CSV 를 읽고 스키마를 검증한다.

    Args:
        path: CSV 경로. None 이면 ``configs/data_gen.yaml`` 의 ``output_path``.

    Returns:
        원시 DataFrame.

    Raises:
        FileNotFoundError: 데이터가 아직 생성되지 않은 경우. ``make data`` 안내 포함.
        ValueError: 스키마 계약 위반(컬럼 누락 등).
    """
    target = resolve(path or load_config("data_gen")["output_path"])
    if not target.exists():
        raise FileNotFoundError(f"데이터 없음: {target}\n먼저 `make data` 를 실행하세요.")

    df = pd.read_csv(target)
    report = validate_frame(df)
    if not report.ok:
        raise ValueError("스키마 검증 실패:\n" + "\n".join(report.errors))
    return df


def fit_imputer(X: pd.DataFrame) -> dict[str, float]:
    """학습 세트 기준 결측 대치값을 계산한다.

    연속형은 중앙값(이상치에 강함), 범주형은 최빈값을 쓴다.
    **테스트 세트로 계산하면 누수**이므로 반드시 학습 세트만 넘길 것.

    Args:
        X: 학습 피처 DataFrame.

    Returns:
        ``{피처명: 대치값}``.
    """
    values: dict[str, float] = {}
    for name in FEATURE_NAMES:
        col = X[name]
        if name in CATEGORICAL:
            mode = col.mode(dropna=True)
            values[name] = float(mode.iloc[0]) if len(mode) else 0.0
        else:
            values[name] = float(col.median(skipna=True))
    return values


def apply_imputer(X: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    """저장된 대치값으로 결측을 채우고 피처 순서를 정렬한다.

    Args:
        X: 결측이 있을 수 있는 DataFrame. 피처 일부만 있어도 된다.
        values: ``fit_imputer`` 결과.

    Returns:
        ``FEATURE_NAMES`` 순서로 정렬되고 결측이 없는 새 DataFrame.

    Note:
        컬럼 순서 정렬이 핵심이다. LightGBM/sklearn 은 순서로 피처를 식별하므로
        UI에서 dict 로 만든 한 행이 다른 순서로 들어오면 조용히 틀린 예측이 나온다.
    """
    out = X.copy()
    for name in FEATURE_NAMES:
        if name not in out.columns:
            out[name] = values[name]
        else:
            out[name] = out[name].fillna(values[name])
    return out[list(FEATURE_NAMES)].astype(float)


def make_split(
    df: pd.DataFrame | None = None,
    *,
    test_size: float = 0.2,
    seed: int = 20260810,
) -> Split:
    """계층 추출로 홀드아웃을 분리하고 결측을 대치한다.

    Args:
        df: 원시 DataFrame. None 이면 ``load_raw()``.
        test_size: 홀드아웃 비율.
        seed: 재현성을 위한 시드. 팀 전체가 같은 값을 쓴다.

    Returns:
        Split 객체.
    """
    data = df if df is not None else load_raw()
    X = data[list(FEATURE_NAMES)]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    values = fit_imputer(X_train)
    return Split(
        X_train=apply_imputer(X_train, values),
        y_train=y_train.reset_index(drop=True),
        X_test=apply_imputer(X_test, values),
        y_test=y_test.reset_index(drop=True),
        imputer_values=values,
    )


def row_to_frame(values: dict[str, float], imputer_values: dict[str, float]) -> pd.DataFrame:
    """UI 입력 한 행(dict)을 모델 입력 DataFrame 으로 바꾼다.

    Args:
        values: ``{피처명: 값}``. 일부 피처가 빠져 있어도 대치값으로 채운다.
        imputer_values: 학습 시 저장된 대치값.

    Returns:
        1행짜리 DataFrame (피처 순서 정렬 완료).
    """
    row = {name: values.get(name, np.nan) for name in FEATURE_NAMES}
    return apply_imputer(pd.DataFrame([row]), imputer_values)
