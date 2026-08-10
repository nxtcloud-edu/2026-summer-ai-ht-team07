"""알림 라우터.

엔드포인트:
    POST /api/alert — 알림 트리거 (dry-run 기본)

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.models.request import AlertRequest
from api.models.response import AlertResponse
from api.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/api", tags=["alert"])


@router.post("/alert", response_model=AlertResponse)
def alert(request: AlertRequest, service: MLService = Depends(get_ml_service)):
    """알림을 트리거한다. 기본은 dry-run (실제 발송 안 함)."""
    try:
        result = service.send_alert(
            die_id=request.die_id,
            probability=request.probability,
            risk_features=request.risk_features,
            recommendations=request.recommendations,
        )
        return AlertResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"알림 실패: {exc}") from exc
