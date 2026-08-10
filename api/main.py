"""YEDA FastAPI 앱 엔트리포인트.

실행::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

소유자: E(프론트엔드·통합). 라우터 등록은 D가 관리.
"""

from __future__ import annotations

import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# src/ 를 import path 에 추가
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.models.response import HealthResponse  # noqa: E402
from api.routers import alert, history, optimize, predict  # noqa: E402
from api.services.ml_service import get_ml_service, initialize_service  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 ML 서비스를 초기화한다."""
    initialize_service()
    print("=" * 60)
    print("  YEDA API Server")
    print("  Version: 1.0.0")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60)
    yield


app = FastAPI(
    title="YEDA API",
    description="반도체 후공정 다이 픽업 수율 예측 · 원인 분해 · 조건 최적화 API",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — 로컬 개발 및 S3 프론트엔드 허용
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 S3 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 전역 예외 핸들러 — 데모 중 스택 트레이스가 사용자에게 노출되지 않게 한다
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """예상하지 못한 예외를 JSON 에러로 변환한다."""
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "서버 내부 오류", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# D의 라우터 등록
# ---------------------------------------------------------------------------
app.include_router(predict.router)
app.include_router(optimize.router)
app.include_router(alert.router)
app.include_router(history.router)


# ---------------------------------------------------------------------------
# E 직접 구현 엔드포인트
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    """헬스체크 — 모델 로드 상태를 확인한다."""
    service = get_ml_service()
    return HealthResponse(
        status="ok",
        model_loaded=not service.is_mock,
        is_mock=service.is_mock,
        model_name=service.model_name,
    )


@app.get("/api/warning", tags=["system"])
def warning():
    """수율 저하 조기경보 상태.

    DB에 이력이 쌓이기 전까지는 NORMAL을 반환한다.
    """
    return {
        "status": "NORMAL",
        "recent_mean": None,
        "previous_mean": None,
        "drop": 0,
        "recent_count": 10,
        "message": "",
    }


@app.post("/api/predict/batch", tags=["predict"])
async def predict_batch(request: Request):
    """배치 예측 — CSV 업로드 다건 처리용.

    요청 형식: { "records": [ {13개 변수 or null}, ... ] }
    결측치(null)는 ML 서비스에서 imputation 후 예측한다.
    """
    body = await request.json()
    records: list[dict[str, Any]] = body.get("records", [])

    service = get_ml_service()
    results = []

    for row in records:
        try:
            probability = service.predict_one(row)
            level = service.get_risk_level(probability)
            results.append({
                "probability": probability,
                "risk_level": level,
                "lower_bound": max(0.0, probability - 0.03),
                "upper_bound": min(1.0, probability + 0.03),
                "model_name": service.model_name,
            })
        except Exception:
            # 개별 행 실패 시 null로 채움 (배치 전체를 실패시키지 않음)
            results.append({
                "probability": None,
                "risk_level": None,
                "lower_bound": None,
                "upper_bound": None,
                "model_name": None,
                "error": "예측 실패",
            })

    return {"results": results, "total": len(results)}
