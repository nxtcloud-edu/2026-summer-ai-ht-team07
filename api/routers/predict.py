"""예측 및 SHAP 설명 라우터.

엔드포인트:
    POST /api/predict — 조건 입력 → 픽업 성공 확률
    POST /api/explain — 조건 입력 → SHAP 기여도 분해

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.models.request import ExplainRequest, PredictRequest
from api.models.response import ExplainResponse, PredictResponse, ShapFeature
from api.routers.history import save_prediction
from api.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, service: MLService = Depends(get_ml_service)):
    """조건 입력 → 픽업 성공 확률 + 위험 등급."""
    try:
        values = request.to_feature_dict()
        probability = service.predict_one(values)
        level = service.get_risk_level(probability)
        # 이력 저장 (B의 MySQL 연동 전까지 인메모리)
        save_prediction(values, probability, level)
        return PredictResponse(
            die_id=request.die_id,
            probability=probability,
            risk_level=level,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"예측 실패: {exc}") from exc


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest, service: MLService = Depends(get_ml_service)):
    """조건 입력 → SHAP 기여도 분해."""
    try:
        values = request.to_feature_dict()
        result = service.explain(values, die_id=request.die_id, top_k=request.top_k)
        shap_features = [ShapFeature(**item) for item in result["shap_values"]]
        return ExplainResponse(
            die_id=request.die_id,
            base_value=result["base_value"],
            shap_values=shap_features,
            disclaimer=result["disclaimer"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"설명 실패: {exc}") from exc
