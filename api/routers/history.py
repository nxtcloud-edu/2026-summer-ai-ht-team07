"""이력 및 프리셋 라우터.

엔드포인트:
    GET /api/presets — 데모 프리셋 목록
    GET /api/history — 예측 이력 조회 (MySQL 연동, DB 미접속 시 인메모리 폴백)

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query

from api.models.response import HistoryItem, PresetResponse
from api.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["history"])

# DB 사용 가능 여부 플래그 — import 실패 시 인메모리 폴백
_use_db = False
try:
    from api.db.queries import get_history as db_get_history
    from api.db.queries import save_alert as db_save_alert
    from api.db.queries import save_prediction as db_save_prediction

    _use_db = True
    logger.info("DB 연동 활성화: api.db.queries 로드 성공")
except Exception as exc:
    logger.warning(f"DB 연동 불가 → 인메모리 폴백: {exc}")

# 인메모리 폴백 (DB 미접속 환경용)
_prediction_history: list[dict[str, Any]] = []
_history_counter: int = 0


def save_prediction(
    input_json: dict,
    probability: float,
    risk_level: str,
    die_id: str | None = None,
    model_name: str | None = None,
) -> int:
    """예측 결과를 저장한다. DB 접속 가능 시 MySQL, 불가 시 인메모리."""
    if _use_db:
        try:
            return db_save_prediction(
                input_json=input_json,
                probability=probability,
                risk_level=risk_level,
                die_id=die_id,
                model_name=model_name,
            )
        except Exception as exc:
            logger.warning(f"DB save_prediction 실패 → 인메모리 폴백: {exc}")

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


def save_alert_record(
    prediction_id: int | None,
    subject: str,
    body: str,
    recipients: list[str],
    sent: bool,
    dry_run: bool,
    error: str | None = None,
) -> int | None:
    """알림 이력을 저장한다. DB 접속 불가 시 무시."""
    if _use_db:
        try:
            return db_save_alert(
                prediction_id=prediction_id,
                subject=subject,
                body=body,
                recipients=recipients,
                sent=sent,
                dry_run=dry_run,
                error=error,
            )
        except Exception as exc:
            logger.warning(f"DB save_alert 실패: {exc}")
    return None


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
    if _use_db:
        try:
            items = db_get_history(limit=limit)
            return [HistoryItem(**item) for item in items]
        except Exception as exc:
            logger.warning(f"DB get_history 실패 → 인메모리 폴백: {exc}")

    items = sorted(_prediction_history, key=lambda x: x["id"], reverse=True)[:limit]
    return [HistoryItem(**item) for item in items]
