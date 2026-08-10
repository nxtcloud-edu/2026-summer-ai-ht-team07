"""API 요청 스키마 — schema.py 의 FEATURE_NAMES, BOUNDS 기반.

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """수율 예측 요청. 13개 공정 변수를 모두 포함해야 한다."""

    die_id: str | None = Field(default=None, description="다이 식별자 (선택)")
    uv_time: float = Field(..., ge=3.0, le=7.0, description="UV 조사 시간 (s)")
    uv_intensity: float = Field(..., ge=1000.0, le=1400.0, description="UV 광량 (mW/cm^2)")
    pin_speed: float = Field(..., ge=0.8, le=1.6, description="이젝터 핀 상승 속도 (mm/s)")
    pin_pressure: float = Field(..., ge=20.0, le=40.0, description="핀 압력 (N)")
    pin_height: float = Field(..., ge=0.3, le=0.7, description="핀 상승 높이 (mm)")
    head_vacuum: float = Field(..., ge=-70.0, le=-60.0, description="픽업 헤드 진공압 (kPa)")
    pin_vacuum: float = Field(..., ge=-55.0, le=-45.0, description="이젝터 진공압 (kPa)")
    temperature: float = Field(..., ge=20.0, le=30.0, description="공정 온도 (degC)")
    humidity: float = Field(..., ge=30.0, le=60.0, description="상대습도 (%RH)")
    vacuum_status: float = Field(..., ge=-100.0, le=-90.0, description="진공 시스템 상태 (kPa)")
    runtime_hours: float = Field(..., ge=0.0, le=24.0, description="설비 연속 가동 시간 (h)")
    tape_type: float = Field(..., ge=0.0, le=1.0, description="다이싱 테이프 종류 (0/1)")
    die_thickness: float = Field(..., ge=100.0, le=150.0, description="다이 두께 (um)")

    def to_feature_dict(self) -> dict[str, float]:
        """Pydantic 모델을 피처 dict 로 변환한다."""
        return {
            "uv_time": self.uv_time,
            "uv_intensity": self.uv_intensity,
            "pin_speed": self.pin_speed,
            "pin_pressure": self.pin_pressure,
            "pin_height": self.pin_height,
            "head_vacuum": self.head_vacuum,
            "pin_vacuum": self.pin_vacuum,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "vacuum_status": self.vacuum_status,
            "runtime_hours": self.runtime_hours,
            "tape_type": self.tape_type,
            "die_thickness": self.die_thickness,
        }


class ExplainRequest(PredictRequest):
    """SHAP 기여도 분해 요청. PredictRequest 와 동일한 피처 + top_k 옵션."""

    top_k: int | None = Field(default=None, ge=1, le=13, description="상위 몇 개 변수만 반환할지")


class OptimizeRequest(PredictRequest):
    """최적화 제안 요청. 현재 공정 조건 13개 피처를 포함한다."""

    pass


class AlertRequest(BaseModel):
    """알림 트리거 요청."""

    die_id: str = Field(..., description="다이 식별자")
    probability: float = Field(..., ge=0.0, le=1.0, description="예측 성공 확률")
    risk_features: list[dict] | None = Field(default=None, description="SHAP 상위 위험 변수")
    recommendations: list[str] | None = Field(default=None, description="개선 가이드 문장")
