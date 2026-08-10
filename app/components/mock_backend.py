"""Mock 백엔드 — 다른 모듈이 미완성이어도 UI 개발이 멈추지 않게 하는 stub.

이 파일의 존재 이유는 하나다: **E(UI·통합)가 킥오프 직후부터 화면을 만들 수 있어야 한다.**
C의 모델도, D의 SHAP/최적화도 아직 없는 상태에서 UI가 대기하면 24시간 중 절반을 날린다.

사용 규칙
    - 모든 함수는 실제 모듈과 **동일한 시그니처와 반환 계약**을 지킨다.
    - 실물이 준비되면 ``app/streamlit_app.py`` 의 백엔드 선택 로직이 자동으로 실물을 쓴다.
      UI 코드는 한 줄도 바꾸지 않는다.
    - Mock 이 켜져 있으면 화면 상단에 항상 배지를 띄운다.
      **모의 데이터를 실제 결과로 착각한 채 발표하는 것이 최악의 사고다.**

소유자: E(UI·통합).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yeda.schema import (
    ADJUSTABLE,
    EXPLANATION_COLUMNS,
    FEATURE_NAMES,
    ID_COL,
    RECOMMENDATION_COLUMNS,
    SPEC_BY_NAME,
    clip_to_bounds,
)

MOCK_BADGE: str = "⚠️ MOCK 모드 — 학습된 모델이 없어 모의 결과를 표시 중입니다 (`make train` 필요)"
"""화면 상단 배지 문구. 실제 결과와 혼동하지 않도록 항상 노출한다."""


def predict_proba(X: pd.DataFrame) -> np.ndarray:
    """모의 성공 확률.

    실제 모델과 비슷한 방향성을 갖도록 몇 개 변수의 선형 결합을 쓴다.
    **정확도가 목적이 아니라 UI 배선 검증이 목적**이므로 단순하게 둔다.

    Args:
        X: 피처 DataFrame.

    Returns:
        길이 ``len(X)`` 의 확률 배열.
    """
    frame = X[list(FEATURE_NAMES)].astype(float)
    z = np.zeros(len(frame))
    for name, weight in (("uv_time", 1.0), ("pin_pressure", 0.8), ("head_vacuum", -0.7), ("runtime_hours", -0.5)):
        spec = SPEC_BY_NAME[name]
        centered = (frame[name] - (spec.low + spec.high) / 2) / ((spec.high - spec.low) / 2)
        z += weight * centered.to_numpy()
    return 1.0 / (1.0 + np.exp(-(1.2 + 1.5 * z)))


def explain_frame(X: pd.DataFrame, ids: pd.Series | None = None, top_k: int | None = None) -> pd.DataFrame:
    """모의 SHAP 기여도 (``EXPLANATION_COLUMNS`` 계약 준수).

    Args:
        X: 설명할 행들.
        ids: 다이 식별자.
        top_k: 행당 상위 개수.

    Returns:
        실제 ``explain.explain_frame()`` 과 같은 형태의 DataFrame. 단위는 %p.
    """
    rng = np.random.default_rng(0)
    die_ids = list(ids) if ids is not None else [f"row_{i}" for i in range(len(X))]
    rows = []
    for i, die_id in enumerate(die_ids):
        values = rng.normal(0, 6, size=len(FEATURE_NAMES))
        order = np.argsort(-np.abs(values))
        if top_k is not None:
            order = order[:top_k]
        for j in order:
            rows.append(
                {
                    ID_COL: die_id,
                    "feature": FEATURE_NAMES[j],
                    "shap_value_pp": float(values[j]),
                    "feature_value": float(X.iloc[i][FEATURE_NAMES[j]]),
                    "direction": "기여" if values[j] >= 0 else "위험",
                }
            )
    return pd.DataFrame(rows, columns=list(EXPLANATION_COLUMNS))


def recommend(current: dict[str, float]) -> pd.DataFrame:
    """모의 개선 가이드 (``RECOMMENDATION_COLUMNS`` 계약 준수).

    조정 가능 변수 중 앞 3개를 범위 중앙 쪽으로 살짝 옮기라고 제안한다.
    **고정 변수는 절대 건드리지 않는다** — mock 조차 이 규칙을 지켜야 UI 검증이 의미 있다.

    Args:
        current: 현재 조건.

    Returns:
        추천 DataFrame.
    """
    rows = []
    for name in list(ADJUSTABLE)[:3]:
        spec = SPEC_BY_NAME[name]
        center = (spec.low + spec.high) / 2
        target = clip_to_bounds({name: current[name] + (center - current[name]) * 0.5})[name]
        if abs(target - current[name]) < spec.resolution:
            continue
        rows.append(
            {
                "feature": name,
                "current_value": float(current[name]),
                "suggested_value": float(target),
                "delta": float(target - current[name]),
                "unit": spec.unit,
                "expected_gain_pp": float(abs(target - current[name]) / (spec.high - spec.low) * 20),
            }
        )
    return pd.DataFrame(rows, columns=list(RECOMMENDATION_COLUMNS))


def sample_frame(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """모의 배치 데이터 — 데이터 생성기가 아직 없을 때 표/차트를 채우는 용도.

    Args:
        n: 행 수.
        seed: 시드.

    Returns:
        ``die_id`` + 13개 피처 DataFrame.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        if spec.kind == "categorical":
            data[name] = rng.integers(0, 2, size=n).astype(float)
        else:
            data[name] = np.round(rng.uniform(spec.low, spec.high, size=n), 2)
    frame = pd.DataFrame(data)
    frame.insert(0, ID_COL, [f"MOCK{i:04d}" for i in range(n)])
    return frame
