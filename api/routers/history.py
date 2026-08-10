"""이력 및 프리셋 라우터.

엔드포인트:
    GET /api/presets — 데모 프리셋 목록
    GET /api/history — 예측 이력 조회 (현재는 인메모리 스텁, B가 MySQL 연동 예정)

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query

from api.models.response import HistoryItem, PresetResponse
from api.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/api", tags=["history"])

# 인메모리 예측 이력 스텁 (B가 MySQL 연동 시 교체 예정)
_prediction_history: list[dict[str, Any]] = []
_history_counter: int = 0


def save_prediction(input_json: dict, probability: float, risk_level: str) -> int:
    """예측 결과를 이력에 저장한다. B의 DB 연동 전까지 인메모리."""
    global _history_counter
    _history_counter += 1
    _prediction_history.append(
        {
            "id": _history_counter,
            "created_at": datetime.now(timezone.utc),
            "input_json": input_json,
            "probability": probability,
            "risk_level": risk_level,
        }
    )
    return _history_counter


@router.get("/presets", response_model=list[PresetResponse])
def get_presets(service: MLService = Depends(get_ml_service)):
    """데모 프리셋 목록을 반환한다."""
    presets = service.get_presets()
    return [
        PresetResponse(
            name=p["name"],
            description=p["description"],
            values=p["values"],
        )
        for p in presets
    ]


@router.get("/history", response_model=list[HistoryItem])
def get_history(limit: int = Query(default=20, ge=1, le=100)):
    """최근 예측 이력을 반환한다 (최신순)."""
    items = sorted(_prediction_history, key=lambda x: x["id"], reverse=True)[:limit]
    return [HistoryItem(**item) for item in items]
