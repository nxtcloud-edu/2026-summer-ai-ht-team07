"""YEDA FastAPI 앱 엔트리포인트.

소유자: E (프론트엔드·통합·데모)

이 파일은 다음을 담당한다:
  - FastAPI 앱 인스턴스 생성
  - CORS 미들웨어 설정
  - 라우터 등록 (D가 완성하면 주석 해제)
  - 전역 예외 핸들러 (데모 중 500 노출 방지)
  - /api/health, /api/presets, /api/warning 엔드포인트 (E 직접 구현)

실행:
  make api
  또는
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# src/ 를 import path에 추가 (프로젝트 루트에서 실행할 때)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# 앱 생성
# ---------------------------------------------------------------------------

app = FastAPI(
    title="YEDA API",
    version="1.0.0",
    description="다이 픽업 수율 예측 · 원인 분해 · 조건 최적화 · 경보 API",
)

# ---------------------------------------------------------------------------
# CORS — 로컬 개발 + S3 배포 도메인 허용
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",    # VS Code Live Server
        "http://127.0.0.1:5500",
    ],
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
    # 개발 중에는 콘솔에 출력
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "서버 내부 오류",
            "detail": str(exc),
        },
    )


# ---------------------------------------------------------------------------
# 라우터 등록 — D가 라우터를 완성하면 순차적으로 주석 해제
# ---------------------------------------------------------------------------

# try:
#     from api.routers import predict
#     app.include_router(predict.router, prefix="/api", tags=["predict"])
# except ImportError:
#     pass

# try:
#     from api.routers import optimize
#     app.include_router(optimize.router, prefix="/api", tags=["optimize"])
# except ImportError:
#     pass

# try:
#     from api.routers import alert
#     app.include_router(alert.router, prefix="/api", tags=["alert"])
# except ImportError:
#     pass

# try:
#     from api.routers import history
#     app.include_router(history.router, prefix="/api", tags=["history"])
# except ImportError:
#     pass


# ---------------------------------------------------------------------------
# E 직접 구현 엔드포인트
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    """서버 상태 + 모델 로드 여부.

    프론트엔드가 주기적으로 폴링하여 서버 연결 상태를 표시한다.
    """
    model_loaded = False
    mock_mode = True

    # C의 ml_service가 준비되면 실제 상태 반환
    try:
        from yeda.io_utils import MODEL_DIR

        primary_model = MODEL_DIR / "primary_model.joblib"
        if primary_model.exists():
            model_loaded = True
            mock_mode = False
    except Exception:
        pass

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "mock_mode": mock_mode,
        "version": "1.0.0",
    }


@app.get("/api/presets")
async def presets():
    """데모 프리셋 목록 반환 (configs/app.yaml에서 로드)."""
    try:
        from yeda.io_utils import load_config

        cfg = load_config("app")
        return {"presets": cfg.get("demo_presets", [])}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "프리셋 로드 실패", "detail": str(exc)},
        )


@app.get("/api/warning")
async def warning():
    """수율 저하 조기경보 상태.

    DB에 이력이 쌓이기 전까지는 NORMAL을 반환한다.
    D가 실 구현을 완료하면 이 mock을 교체한다.
    """
    return {
        "status": "NORMAL",
        "recent_mean": None,
        "previous_mean": None,
        "drop": 0,
        "recent_count": 10,
        "message": "",
    }


# ---------------------------------------------------------------------------
# Mock 엔드포인트 — D의 라우터가 준비되기 전 프론트엔드 개발용
# 라우터가 등록되면 아래 mock들은 자동으로 덮어써진다.
# ---------------------------------------------------------------------------


@app.post("/api/predict")
async def mock_predict(request: Request):
    """Mock 예측 — 프론트엔드 개발용.

    실제로는 D의 predict 라우터가 이 경로를 덮어쓴다.
    """
    body = await request.json()

    # 간단한 mock: 변수 조합으로 임의의 확률 생성
    import hashlib
    import json

    seed = hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()
    # seed에서 0~1 사이 값 추출
    raw = int(seed[:8], 16) / 0xFFFFFFFF
    probability = 0.55 + raw * 0.40  # 55% ~ 95% 범위

    # risk level 판정
    if probability < 0.60:
        risk_level = "critical"
    elif probability < 0.80:
        risk_level = "warning"
    else:
        risk_level = "normal"

    # 예측 범위 (mock: ±3%p)
    lower = max(0.0, probability - 0.03)
    upper = min(1.0, probability + 0.03)

    return {
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "prediction_id": int(seed[:6], 16),
        "model_name": "mock",
    }


@app.post("/api/predict/batch")
async def mock_predict_batch(request: Request):
    """Mock 배치 예측 — CSV 업로드 배치 처리용.

    요청 형식: { "records": [ {13개 변수}, {13개 변수}, ... ] }
    """
    import hashlib
    import json

    body = await request.json()
    records = body.get("records", [])

    results = []
    for row in records:
        seed = hashlib.md5(json.dumps(row, sort_keys=True).encode()).hexdigest()
        raw = int(seed[:8], 16) / 0xFFFFFFFF
        probability = 0.55 + raw * 0.40

        if probability < 0.60:
            risk_level = "critical"
        elif probability < 0.80:
            risk_level = "warning"
        else:
            risk_level = "normal"

        lower = max(0.0, probability - 0.03)
        upper = min(1.0, probability + 0.03)

        results.append({
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "model_name": "mock",
        })

    return {"results": results, "total": len(results)}


@app.post("/api/explain")
async def mock_explain(request: Request):
    """Mock SHAP 설명 — 프론트엔드 개발용."""
    features_mock = [
        {"feature": "uv_time", "korean": "UV 조사 시간", "shap_value_pp": 8.2, "feature_value": 4.8, "direction": "positive"},
        {"feature": "pin_speed", "korean": "이젝터 핀 상승 속도", "shap_value_pp": -5.1, "feature_value": 1.4, "direction": "negative"},
        {"feature": "head_vacuum", "korean": "픽업 헤드 진공압", "shap_value_pp": 4.3, "feature_value": -65.5, "direction": "positive"},
        {"feature": "die_thickness", "korean": "다이 두께", "shap_value_pp": -3.7, "feature_value": 112.0, "direction": "negative"},
        {"feature": "pin_pressure", "korean": "핀 압력", "shap_value_pp": 2.9, "feature_value": 29.0, "direction": "positive"},
        {"feature": "humidity", "korean": "상대습도", "shap_value_pp": -2.1, "feature_value": 44.0, "direction": "negative"},
        {"feature": "temperature", "korean": "공정 온도", "shap_value_pp": 1.5, "feature_value": 24.0, "direction": "positive"},
        {"feature": "runtime_hours", "korean": "설비 연속 가동 시간", "shap_value_pp": -1.2, "feature_value": 9.0, "direction": "negative"},
        {"feature": "pin_vacuum", "korean": "이젝터 진공압", "shap_value_pp": 0.8, "feature_value": -50.0, "direction": "positive"},
        {"feature": "uv_intensity", "korean": "UV 광량", "shap_value_pp": 0.6, "feature_value": 1180, "direction": "positive"},
        {"feature": "pin_height", "korean": "핀 상승 높이", "shap_value_pp": -0.4, "feature_value": 0.5, "direction": "negative"},
        {"feature": "vacuum_status", "korean": "진공 시스템 상태", "shap_value_pp": 0.3, "feature_value": -95.5, "direction": "positive"},
        {"feature": "tape_type", "korean": "다이싱 테이프 종류", "shap_value_pp": 0.1, "feature_value": 1, "direction": "positive"},
    ]

    return {
        "base_value": 0.75,
        "contributions": features_mock,
    }


@app.post("/api/optimize")
async def mock_optimize(request: Request):
    """Mock 최적화 — 프론트엔드 개발용."""
    return {
        "current_probability": 0.723,
        "optimized_probability": 0.891,
        "gain_pp": 16.8,
        "recommendations": [
            {"feature": "uv_time", "korean": "UV 조사 시간", "current_value": 3.5, "suggested_value": 5.2, "unit": "s", "expected_gain_pp": 6.3},
            {"feature": "pin_speed", "korean": "이젝터 핀 상승 속도", "current_value": 1.40, "suggested_value": 1.05, "unit": "mm/s", "expected_gain_pp": 5.1},
            {"feature": "head_vacuum", "korean": "픽업 헤드 진공압", "current_value": -62.0, "suggested_value": -67.5, "unit": "kPa", "expected_gain_pp": 3.2},
            {"feature": "pin_pressure", "korean": "핀 압력", "current_value": 25.0, "suggested_value": 31.5, "unit": "N", "expected_gain_pp": 2.2},
        ],
        "limitations": [
            "runtime_hours = 20.0h → 콜릿 교체가 필요합니다.",
            "die_thickness = 112μm → 제품 사양이므로 조정할 수 없습니다.",
        ],
    }


@app.post("/api/alert")
async def mock_alert(request: Request):
    """Mock 알림 — dry-run 본문 반환."""
    body = await request.json()
    probability = body.get("probability", 0.5)
    risk_level = body.get("risk_level", "warning")

    return {
        "subject": f"[YEDA 경보] 예측 픽업 성공률 {probability * 100:.1f}%",
        "body": (
            f"예측 픽업 성공률이 {probability * 100:.1f}%로 임계값을 밑돌았습니다.\n\n"
            f"[주요 위험 변수 — SHAP 기여도]\n"
            f"  - 이젝터 핀 상승 속도: 현재 1.40mm/s → 성공률 5.1%p 감소 기여\n"
            f"  - 다이 두께: 현재 112μm → 성공률 3.7%p 감소 기여\n\n"
            f"[개선 가이드]\n"
            f"  - 핀 상승 속도를 1.05mm/s로 낮추세요 (+5.1%p 기대)\n"
            f"  - 픽업 헤드 진공압을 -67.5kPa로 강화하세요 (+3.2%p 기대)\n\n"
            f"{'─' * 50}\n"
            f"※ 본 알림은 dry-run 모드입니다. 실제 발송되지 않았습니다.\n"
            f"— YEDA 수율 모니터링"
        ),
        "recipients": ["(dry-run: 수신자 없음)"],
        "sent": False,
        "dry_run": True,
    }


@app.get("/api/history")
async def mock_history():
    """Mock 이력 — 프론트엔드 개발용 샘플 데이터."""
    import random
    from datetime import datetime, timedelta

    now = datetime.now()
    records = []
    for i in range(20):
        ts = now - timedelta(minutes=(20 - i) * 5)
        prob = 0.85 + random.uniform(-0.15, 0.10)
        prob = max(0.50, min(1.0, prob))

        if prob < 0.60:
            level = "critical"
        elif prob < 0.80:
            level = "warning"
        else:
            level = "normal"

        records.append({
            "timestamp": ts.isoformat(),
            "probability": round(prob, 4),
            "risk_level": level,
            "model_name": "mock",
            "actual_yield": None,
        })

    return {"records": records}


# ---------------------------------------------------------------------------
# 앱 시작 이벤트
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 환경 확인."""
    print("=" * 60)
    print("  YEDA API Server")
    print(f"  Version: 1.0.0")
    print(f"  Docs: http://localhost:8000/docs")
    print("=" * 60)
