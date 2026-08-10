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
from yeda.schema import FEATURE_NAMES, SPEC_BY_NAME  # noqa: E402


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

    @property
    def is_loaded(self) -> bool:
        """실제 학습 번들이 정상 로드됐는지 반환한다.

        is_mock의 반대 의미를 명시적인 서비스 계약으로 제공한다. UI/API가
        내부 _bundle 구현을 직접 들여다보지 않게 하기 위한 상태값이다.
        """
        return self._bundle is not None and not self.is_mock

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

    def _prediction_frame(self, values: Any) -> pd.DataFrame:
        """단건 dict 또는 배치 DataFrame/list를 피처 순서에 맞춘다."""
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
        elif isinstance(values, dict):
            frame = pd.DataFrame([values])
        elif (
            isinstance(values, (list, tuple))
            and values
            and all(isinstance(row, dict) for row in values)
        ):
            frame = pd.DataFrame(values)
        else:
            raise TypeError("입력은 dict, dict 목록 또는 DataFrame이어야 합니다.")

        missing = [name for name in FEATURE_NAMES if name not in frame.columns]
        if missing:
            raise KeyError(f"피처 누락: {missing}")
        return frame.loc[:, FEATURE_NAMES].astype(float)

    def predict(self, values: Any) -> float | list[float]:
        """성공확률을 반환하며 배치는 JSON 직렬화 가능한 목록으로 반환한다."""
        probabilities = self.predict_proba(self._prediction_frame(values))
        if len(probabilities) == 1:
            return float(probabilities[0])
        return probabilities.astype(float).tolist()

    def predict_yield(self, values: Any) -> float:
        """배치 평균 픽업 성공확률을 예상 수율(%)로 반환한다."""
        probabilities = self.predict_proba(self._prediction_frame(values))
        return float(np.mean(probabilities) * 100.0)

    def predict_defect_rate(self, values: Any) -> float:
        """예상 불량률(%)을 ``100 - 예상 수율``로 반환한다."""
        return float(100.0 - self.predict_yield(values))

    def get_risk_level(self, probability: float) -> str:
        """확률 → 위험 등급."""
        return risk_level(probability)

    # ---------------------------------------------------------------- 설명

    def explain(
        self,
        values: dict[str, float],
        die_id: str | None = None,
        top_k: int | None = None,
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
            base_value = self._mock_base_probability()
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

    def get_shap_background(
        self,
        n_samples: int = 200,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """결측 처리된 SHAP 배경 표본을 결정론적으로 반환한다."""
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise ValueError("n_samples는 양의 정수여야 합니다.")
        if n_samples <= 0:
            raise ValueError("n_samples는 양의 정수여야 합니다.")

        if self._background is None:
            try:
                from yeda.data.preprocess import make_split

                self._background = make_split().X_train
            except Exception:  # 데이터 생성 전에도 서비스는 동작해야 한다.
                if self._bundle is not None:
                    values = dict(self._bundle.imputer_values)
                else:
                    from yeda.schema import SPEC_BY_NAME

                    values = {
                        name: (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2
                        for name in FEATURE_NAMES
                    }
                self._background = pd.DataFrame([values], columns=FEATURE_NAMES)

        background = self._background.loc[:, FEATURE_NAMES]
        if len(background) > n_samples:
            background = background.sample(
                n=n_samples,
                random_state=random_state,
            )
        return background.reset_index(drop=True)

    def _ensure_explainer(self):
        """SHAP explainer 를 지연 생성한다."""
        if self._explainer is None:
            from yeda.explain.shap_explainer import build_explainer

            cfg = load_config("app")
            size = int(cfg["app"]["shap_background_size"])
            background = self.get_shap_background(size)
            self._explainer = build_explainer(self._bundle, background, size)
        return self._explainer

    # ------------------------------------------------------------- 최적화

    def optimize(self, values: dict[str, float]) -> dict[str, Any]:
        """개선 가이드를 계산한다.

        Returns:
            OptimizationResult 를 dict 로 변환한 결과.
        """
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
        """스키마 방향만 따르는 결정론적 mock 예측.

        mock은 실제 모델의 대체 성능을 주장하지 않는다. 다만 모델 파일이 없는
        개발 환경에서도 진공 부호를 포함한 schema.MONOTONE 방향과 API 계약은
        어기지 않아야 한다. 생성기의 잠재 점수나 계수는 의도적으로 참조하지 않는다.
        """
        logit = np.full(len(X), np.log(0.70 / 0.30), dtype=float)

        for name in FEATURE_NAMES:
            spec = SPEC_BY_NAME[name]
            span = float(spec.high - spec.low)
            normalized = np.clip(
                (X[name].to_numpy(dtype=float) - float(spec.low)) / span,
                0.0,
                1.0,
            )
            if spec.monotone:
                # 값 기준 방향이다. 진공 변수의 -1은 더 음수일수록 확률이 높다.
                logit += 0.45 * float(spec.monotone) * (normalized - 0.5)
            elif name in {"pin_height", "temperature"}:
                # 단조 제약이 없는 두 연속 변수는 중앙 설정점에서 가장 안정적이라는
                # 중립적인 데모 곡선만 둔다. tape_type에는 임의 효과를 만들지 않는다.
                logit -= 0.30 * np.square(2.0 * normalized - 1.0)

        return 1.0 / (1.0 + np.exp(-logit))

    @staticmethod
    def _mock_reference_frame(n_rows: int = 1) -> pd.DataFrame:
        """mock 설명의 고정 기준점(각 스키마 범위 중앙)을 만든다."""
        midpoint = {
            name: (SPEC_BY_NAME[name].low + SPEC_BY_NAME[name].high) / 2.0
            for name in FEATURE_NAMES
        }
        return pd.DataFrame([midpoint] * n_rows, columns=FEATURE_NAMES)

    def _mock_base_probability(self) -> float:
        """mock 설명 기준점의 성공 확률."""
        return float(self._mock_predict(self._mock_reference_frame())[0])

    def _mock_explain(
        self, X: pd.DataFrame, ids: pd.Series, top_k: int | None
    ) -> pd.DataFrame:
        """mock 확률을 재현하는 결정론적 순차 기여도 분해.

        전체 피처를 반환할 때 base + sum(contribution) == prediction이
        부동소수 오차 안에서 성립한다. 이는 SHAP 자체가 아니라 모델 미탑재 시
        UI 계약을 검증하기 위한 확률공간 waterfall이다.
        """
        from yeda.schema import EXPLANATION_COLUMNS, ID_COL

        rows = []
        references = self._mock_reference_frame(len(X))
        for i in range(len(X)):
            running = references.iloc[i].to_dict()
            previous = float(self._mock_predict(pd.DataFrame([running]))[0])
            contributions: list[float] = []
            for name in FEATURE_NAMES:
                running[name] = float(X.iloc[i][name])
                current = float(self._mock_predict(pd.DataFrame([running]))[0])
                contributions.append((current - previous) * 100.0)
                previous = current

            values = np.asarray(contributions, dtype=float)
            order = np.argsort(-np.abs(values), kind="stable")
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


def _current_service() -> MLService:
    """Lifespan 전 직접 호출에도 동일한 DI singleton을 반환한다."""
    service = get_ml_service()
    return service if service is not None else initialize_service()


def predict(values: Any) -> float | list[float]:
    """모듈 수준 픽업 성공확률 API."""
    return _current_service().predict(values)


def predict_yield(values: Any) -> float:
    """모듈 수준 배치 예상 수율(%) API."""
    return _current_service().predict_yield(values)


def predict_defect_rate(values: Any) -> float:
    """모듈 수준 배치 예상 불량률(%) API."""
    return _current_service().predict_defect_rate(values)


def get_shap_background(
    n_samples: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """모듈 수준 SHAP 배경 데이터 API."""
    return _current_service().get_shap_background(n_samples, random_state)


__all__ = [
    "MLService",
    "get_ml_service",
    "initialize_service",
    "predict",
    "predict_yield",
    "predict_defect_rate",
    "get_shap_background",
]
