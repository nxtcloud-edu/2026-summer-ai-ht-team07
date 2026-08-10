"""API 응답 스키마.

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """헬스체크 응답."""

    status: str = "ok"
    model_loaded: bool
    is_mock: bool
    model_name: str | None = None


class PredictResponse(BaseModel):
    """수율 예측 응답."""

    die_id: str | None = None
    probability: float = Field(..., description="픽업 성공 확률 (0~1)")
    risk_level: str = Field(..., description="위험 등급: critical / warning / normal")


class ShapFeature(BaseModel):
    """SHAP 기여도 개별 피처."""

    feature: str
    shap_value_pp: float = Field(..., description="기여도 (%p 단위)")
    feature_value: float
    direction: str = Field(..., description="기여 / 위험")


class ExplainResponse(BaseModel):
    """SHAP 기여도 분해 응답."""

    die_id: str | None = None
    base_value: float = Field(..., description="기준 성공률 (0~1)")
    shap_values: list[ShapFeature]
    disclaimer: str


class RecommendationItem(BaseModel):
    """최적화 개별 제안 항목."""

    feature: str
    current_value: float
    suggested_value: float
    delta: float
    unit: str
    expected_gain_pp: float


class OptimizeResponse(BaseModel):
    """최적화 제안 응답."""

    baseline_prob: float
    optimized_prob: float
    gain_pp: float
    recommendations: list[RecommendationItem]
    formatted_text: list[str] = Field(default_factory=list, description="한국어 개선 가이드 문장")
    n_evaluations: int


class AlertResponse(BaseModel):
    """알림 트리거 응답."""

    subject: str | None = None
    body: str | None = None
    recipients: list[str] = Field(default_factory=list)
    sent: bool = False
    dry_run: bool = True
    error: str | None = None


class PresetResponse(BaseModel):
    """데모 프리셋."""

    name: str
    description: str
    values: dict[str, float]


class HistoryItem(BaseModel):
    """예측 이력 항목."""

    id: int
    created_at: datetime
    input_json: dict
    probability: float
    risk_level: str
