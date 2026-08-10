"""YEDA FastAPI 앱 엔트리포인트.

실행::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

소유자: E(프론트엔드·통합). 라우터 등록은 D가 관리.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    yield


app = FastAPI(
    title="YEDA API",
    description="반도체 후공정 다이 픽업 수율 예측 · 원인 분해 · 조건 최적화 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — 로컬 개발 및 S3 프론트엔드 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 S3 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(predict.router)
app.include_router(optimize.router)
app.include_router(alert.router)
app.include_router(history.router)


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
