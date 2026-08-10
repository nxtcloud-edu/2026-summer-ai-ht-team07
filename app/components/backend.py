"""백엔드 어댑터 — 실물 모듈과 mock 사이를 자동 전환한다.

UI 코드는 이 파일의 ``Backend`` 만 호출한다. 학습된 모델이 있으면 실물을,
없으면 mock 을 쓰며 **UI 코드는 한 줄도 달라지지 않는다.**

이 구조가 있어야 E(UI·통합)가 C·D의 산출물을 기다리지 않고 킥오프부터 일할 수 있다.

소유자: E(UI·통합).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from yeda.io_utils import load_config
from yeda.schema import FEATURE_NAMES

from . import mock_backend


@dataclass
class Backend:
    """예측 · 설명 · 최적화를 한 곳에서 제공하는 파사드.

    Attributes:
        is_mock: mock 모드 여부. True 면 화면에 경고 배지를 띄워야 한다.
        bundle: 실제 ``ModelBundle`` (mock 이면 None).
        metrics: 홀드아웃 성능 지표 (UI 표시용).
        status: 사람이 읽는 상태 문자열.
    """

    is_mock: bool
    bundle: Any = None
    metrics: dict[str, float] | None = None
    status: str = ""
    _explainer: Any = None
    _background: pd.DataFrame | None = None

    # ---------------------------------------------------------------- 예측
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """픽업 성공 확률을 반환한다 (0~1)."""
        if self.is_mock:
            return mock_backend.predict_proba(X)
        return self.bundle.predict_proba(X)

    def predict_one(self, values: dict[str, float]) -> float:
        """조건 dict 하나에 대한 성공 확률."""
        frame = self.to_frame(values)
        return float(self.predict_proba(frame)[0])

    def to_frame(self, values: dict[str, float]) -> pd.DataFrame:
        """조건 dict → 모델 입력 DataFrame (피처 순서 정렬 포함).

        UI 슬라이더가 만든 dict 는 순서가 보장되지 않는다. 여기서 반드시 정렬한다.
        """
        if self.is_mock:
            return pd.DataFrame([{name: values[name] for name in FEATURE_NAMES}])
        from yeda.data.preprocess import row_to_frame

        return row_to_frame(values, self.bundle.imputer_values)

    # ---------------------------------------------------------------- 설명
    def explain(self, X: pd.DataFrame, ids: list[str] | None = None, top_k: int | None = None) -> pd.DataFrame:
        """SHAP 기여도를 %p 단위 long format 으로 반환한다."""
        if self.is_mock:
            return mock_backend.explain_frame(X, ids=pd.Series(ids) if ids else None, top_k=top_k)

        from yeda.explain.shap_explainer import explain_frame

        return explain_frame(self._ensure_explainer(), X, ids=pd.Series(ids) if ids else None, top_k=top_k)

    def base_value(self) -> float:
        """워터폴 시작점이 되는 기준 확률."""
        if self.is_mock:
            return 0.70
        return self._ensure_explainer().base_value

    def _ensure_explainer(self):
        """SHAP explainer 를 지연 생성한다 (앱 첫 로딩을 빠르게 유지)."""
        if self._explainer is None:
            from yeda.data.preprocess import make_split
            from yeda.explain.shap_explainer import build_explainer

            cfg = load_config("app")
            if self._background is None:
                self._background = make_split().X_train
            self._explainer = build_explainer(
                self.bundle, self._background, int(cfg["app"]["shap_background_size"])
            )
        return self._explainer

    # ------------------------------------------------------------- 최적화
    def recommend(self, current: dict[str, float]):
        """개선 가이드를 계산한다.

        Returns:
            실물이면 ``optimize.OptimizationResult``, mock 이면 같은 필드를 갖는 경량 객체.
        """
        if self.is_mock:
            table = mock_backend.recommend(current)
            baseline = self.predict_one(current)
            return _MockResult(
                baseline_prob=baseline,
                optimized_prob=min(1.0, baseline + table["expected_gain_pp"].sum() / 100),
                gain_pp=float(table["expected_gain_pp"].sum()),
                current=dict(current),
                suggested=dict(current),
                recommendations=table,
                n_evaluations=0,
            )
        from yeda.optimize.search import recommend

        return recommend(self.predict_proba, current)


@dataclass
class _MockResult:
    """``OptimizationResult`` 와 필드가 같은 mock 결과 객체."""

    baseline_prob: float
    optimized_prob: float
    gain_pp: float
    current: dict[str, float]
    suggested: dict[str, float]
    recommendations: pd.DataFrame
    n_evaluations: int


def get_backend() -> Backend:
    """모델이 있으면 실물, 없으면 mock 백엔드를 만든다.

    Returns:
        Backend.

    Note:
        예외를 던지지 않는다. 데모 중 파일 하나가 없다고 앱이 죽는 것이 최악이므로,
        어떤 실패도 mock 모드로 흡수하고 사유를 ``status`` 에 남긴다.
    """
    cfg = load_config("app")
    try:
        from yeda.models.registry import load_bundle

        bundle = load_bundle()
        return Backend(
            is_mock=False,
            bundle=bundle,
            metrics=bundle.metrics,
            status=f"학습된 모델 로드 완료 ({bundle.name})",
        )
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 mock 으로 폴백
        if not cfg["app"].get("allow_mock_mode", True):
            raise
        return Backend(is_mock=True, status=f"모델 로드 실패 → mock 모드: {type(exc).__name__}")
