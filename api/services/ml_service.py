"""ML 서비스 레이어 — FastAPI 용 모델 로드·예측·설명·최적화 파사드.

기존 ``app/components/backend.py`` 의 ``Backend`` 패턴을 그대로 따른다:
- 모델이 있으면 실물, 없으면 mock 모드로 폴백 (절대 크래시하지 않음)
- SHAP explainer 는 지연 초기화 (첫 explain 호출 시 생성)

소유자: D(API 라우터·설명·최적화) + C(ml_service 모델 로딩 부분).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# src/ 를 import path 에 추가
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from yeda.alerts.email import notify, risk_level  # noqa: E402
from yeda.io_utils import load_config  # noqa: E402
from yeda.optimize.search import format_recommendations, recommend  # noqa: E402
from yeda.schema import FEATURE_NAMES  # noqa: E402


class MLService:
    """예측·설명·최적화를 한 곳에서 제공하는 서비스 싱글톤.

    Attributes:
        is_mock: mock 모드 여부.
        model_name: 로드된 모델 이름.
        status: 사람이 읽는 상태 문자열.
    """

    def __init__(self) -> None:
        self.is_mock: bool = True
        self.model_name: str | None = None
        self.status: str = ""
        self._bundle: Any = None
        self._explainer: Any = None
        self._background: pd.DataFrame | None = None
        self._initialize()

    def _initialize(self) -> None:
        """모델 로드를 시도하고, 실패하면 mock 모드로 진입한다."""
        try:
            from yeda.models.registry import load_bundle

            bundle = self._bundle = load_bundle()
            self.is_mock = False
            self.model_name = bundle.name
            self.status = f"학습된 모델 로드 완료 ({bundle.name})"
        except Exception as exc:  # noqa: BLE001
            self.is_mock = True
            self.model_name = None
            self.status = f"모델 로드 실패 → mock 모드: {type(exc).__name__}"

    # ---------------------------------------------------------------- 예측

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """픽업 성공 확률 배열을 반환한다."""
        if self.is_mock:
            return self._mock_predict(X)
        return self._bundle.predict_proba(X)

    def predict_one(self, values: dict[str, float]) -> float:
        """조건 dict 하나에 대한 성공 확률."""
        frame = self._to_frame(values)
        return float(self.predict_proba(frame)[0])

    def get_risk_level(self, probability: float) -> str:
        """확률 → 위험 등급."""
        return risk_level(probability)

    # ---------------------------------------------------------------- 설명

    def explain(
        self, values: dict[str, float], die_id: str | None = None, top_k: int | None = None
    ) -> dict[str, Any]:
        """SHAP 기여도 분해 결과를 반환한다.

        Returns:
            {"base_value": float, "shap_values": list[dict], "disclaimer": str}
        """
        from yeda.explain.shap_explainer import DISCLAIMER

        frame = self._to_frame(values)
        ids = pd.Series([die_id or "input"])

        if self.is_mock:
            explanation = self._mock_explain(frame, ids, top_k)
            base_value = 0.70
        else:
            from yeda.explain.shap_explainer import explain_frame

            explainer_bundle = self._ensure_explainer()
            explanation = explain_frame(explainer_bundle, frame, ids=ids, top_k=top_k)
            base_value = explainer_bundle.base_value

        shap_list = []
        for _, row in explanation.iterrows():
            shap_list.append(
                {
                    "feature": row["feature"],
                    "shap_value_pp": float(row["shap_value_pp"]),
                    "feature_value": float(row["feature_value"]),
                    "direction": row["direction"],
                }
            )

        return {
            "base_value": base_value,
            "shap_values": shap_list,
            "disclaimer": DISCLAIMER,
        }

    def _ensure_explainer(self):
        """SHAP explainer 를 지연 생성한다."""
        if self._explainer is None:
            from yeda.data.preprocess import make_split
            from yeda.explain.shap_explainer import build_explainer

            cfg = load_config("app")
            if self._background is None:
                self._background = make_split().X_train
            self._explainer = build_explainer(
                self._bundle, self._background, int(cfg["app"]["shap_background_size"])
            )
        return self._explainer

    # ------------------------------------------------------------- 최적화

    def optimize(self, values: dict[str, float]) -> dict[str, Any]:
        """개선 가이드를 계산한다.

        Returns:
            OptimizationResult 를 dict 로 변환한 결과.
        """
        if self.is_mock:
            return self._mock_optimize(values)

        result = recommend(self.predict_proba, values)
        formatted = format_recommendations(result)

        recommendations = []
        for _, row in result.recommendations.iterrows():
            recommendations.append(
                {
                    "feature": row["feature"],
                    "current_value": float(row["current_value"]),
                    "suggested_value": float(row["suggested_value"]),
                    "delta": float(row["delta"]),
                    "unit": row["unit"],
                    "expected_gain_pp": float(row["expected_gain_pp"]),
                }
            )

        return {
            "baseline_prob": result.baseline_prob,
            "optimized_prob": result.optimized_prob,
            "gain_pp": result.gain_pp,
            "recommendations": recommendations,
            "formatted_text": formatted,
            "n_evaluations": result.n_evaluations,
        }

    # ------------------------------------------------------------- 알림

    def send_alert(
        self,
        die_id: str,
        probability: float,
        risk_features: list[dict] | None = None,
        recommendations: list[str] | None = None,
    ) -> dict[str, Any]:
        """알림을 트리거한다.

        Returns:
            AlertResponse 호환 dict.
        """
        cfg = load_config("app")
        result = notify(die_id, probability, risk_features, recommendations, cfg)

        if result is None:
            return {
                "subject": None,
                "body": None,
                "recipients": [],
                "sent": False,
                "dry_run": cfg["alerts"].get("dry_run", True),
                "error": "알림 조건에 해당하지 않습니다 (등급/쿨다운).",
            }

        return {
            "subject": result.subject,
            "body": result.body,
            "recipients": result.recipients,
            "sent": result.sent,
            "dry_run": cfg["alerts"].get("dry_run", True),
            "error": result.error,
        }

    # ------------------------------------------------------------- 프리셋

    def get_presets(self) -> list[dict[str, Any]]:
        """configs/app.yaml 의 데모 프리셋 목록을 반환한다."""
        cfg = load_config("app")
        return cfg.get("demo_presets", [])

    # ------------------------------------------------------------- 내부

    def _to_frame(self, values: dict[str, float]) -> pd.DataFrame:
        """조건 dict → 정렬된 DataFrame."""
        return pd.DataFrame([{name: values[name] for name in FEATURE_NAMES}])

    # ------------------------------------------------------------- Mock

    def _mock_predict(self, X: pd.DataFrame) -> np.ndarray:
        """mock 예측: 선형 조합 기반 시그모이드."""
        uv = (X["uv_time"] - 3.0) / 4.0
        pressure = (X["pin_pressure"] - 20.0) / 20.0
        vacuum = (X["head_vacuum"] + 70.0) / 10.0
        runtime = 1.0 - X["runtime_hours"] / 24.0
        logit = -1.0 + 2.0 * uv + 1.5 * pressure + 1.0 * vacuum + 0.5 * runtime
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob.to_numpy()

    def _mock_explain(
        self, X: pd.DataFrame, ids: pd.Series, top_k: int | None
    ) -> pd.DataFrame:
        """mock SHAP: 랜덤 기여도 생성."""
        from yeda.schema import EXPLANATION_COLUMNS, ID_COL

        rng = np.random.default_rng(42)
        rows = []
        for i in range(len(X)):
            values = rng.normal(0, 5, size=len(FEATURE_NAMES))
            order = np.argsort(-np.abs(values))
            if top_k:
                order = order[:top_k]
            for j in order:
                rows.append(
                    {
                        ID_COL: str(ids.iloc[i]),
                        "feature": FEATURE_NAMES[j],
                        "shap_value_pp": float(values[j]),
                        "feature_value": float(X.iloc[i][FEATURE_NAMES[j]]),
                        "direction": "기여" if values[j] >= 0 else "위험",
                    }
                )
        return pd.DataFrame(rows, columns=list(EXPLANATION_COLUMNS))

    def _mock_optimize(self, values: dict[str, float]) -> dict[str, Any]:
        """mock 최적화: 조정 가능 변수를 범위 중앙으로 이동."""
        from yeda.schema import ADJUSTABLE, SPEC_BY_NAME

        baseline_prob = self.predict_one(values)
        recommendations = []
        suggested = dict(values)

        for name in list(ADJUSTABLE)[:3]:
            spec = SPEC_BY_NAME[name]
            mid = (spec.low + spec.high) / 2
            if abs(values[name] - mid) < 1e-6:
                continue
            delta = mid - values[name]
            suggested[name] = mid
            recommendations.append(
                {
                    "feature": name,
                    "current_value": values[name],
                    "suggested_value": mid,
                    "delta": delta,
                    "unit": spec.unit,
                    "expected_gain_pp": 2.0,
                }
            )

        optimized_prob = self.predict_one(suggested)
        gain_pp = (optimized_prob - baseline_prob) * 100.0

        return {
            "baseline_prob": baseline_prob,
            "optimized_prob": optimized_prob,
            "gain_pp": gain_pp,
            "recommendations": recommendations,
            "formatted_text": [f"Mock: {len(recommendations)}개 변수 조정 제안"],
            "n_evaluations": 0,
        }


@lru_cache(maxsize=1)
def get_ml_service() -> MLService:
    """MLService 싱글톤을 반환한다 (FastAPI Depends 용)."""
    return get_ml_service._instance


# 모듈 로드 시 인스턴스 생성하지 않고, 앱 startup 에서 초기화
get_ml_service._instance = None  # type: ignore[attr-defined]


def initialize_service() -> MLService:
    """서비스를 초기화하고 캐시한다. FastAPI lifespan 에서 호출."""
    service = MLService()
    get_ml_service._instance = service  # type: ignore[attr-defined]
    get_ml_service.cache_clear()
    return service
