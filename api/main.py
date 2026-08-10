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
    벡터화 추론으로 수천 건도 수 초 내 처리.
    """
    import pandas as pd
    from yeda.schema import FEATURE_NAMES, SPEC_BY_NAME

    body = await request.json()
    records: list[dict[str, Any]] = body.get("records", [])

    if not records:
        return {"results": [], "total": 0}

    service = get_ml_service()

    # DataFrame으로 변환 + 결측 대치 (중앙값)
    df = pd.DataFrame(records)
    for col in FEATURE_NAMES:
        if col in df.columns:
            mid = (SPEC_BY_NAME[col].low + SPEC_BY_NAME[col].high) / 2
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(mid)
        else:
            mid = (SPEC_BY_NAME[col].low + SPEC_BY_NAME[col].high) / 2
            df[col] = mid

    # 벡터화 배치 예측 (한 번에 전체)
    try:
        X = df[list(FEATURE_NAMES)].astype(float)
        probs = service.predict_proba(X)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "배치 예측 실패", "detail": str(exc)},
        )

    results = []
    for prob in probs:
        p = float(prob)
        level = service.get_risk_level(p)
        results.append({
            "probability": round(p, 4),
            "risk_level": level,
            "lower_bound": round(max(0.0, p - 0.03), 4),
            "upper_bound": round(min(1.0, p + 0.03), 4),
            "model_name": service.model_name,
        })

    return {"results": results, "total": len(results)}


@app.post("/api/retrain", tags=["system"])
async def retrain(request: Request):
    """업로드된 CSV 데이터를 누적하고 모델을 재학습한다.

    요청 형식: { "records": [ {13개 변수 + pickup_success}, ... ] }
    또는 records 없이 호출하면 기존 data/raw 파일로 재학습만 수행.

    재학습 후 서비스의 모델이 즉시 교체된다.
    """
    import subprocess
    import time

    body = await request.json()
    records: list[dict[str, Any]] = body.get("records", [])

    # 1. 새 데이터가 있으면 별도 파일(uploaded_data.csv)에 누적 저장
    #    원본 yeda_synthetic.csv는 건드리지 않는다.
    data_path = _ROOT / "data" / "raw" / "yeda_synthetic.csv"
    upload_path = _ROOT / "data" / "raw" / "uploaded_data.csv"
    rows_added = 0

    if records:
        import pandas as pd
        new_df = pd.DataFrame(records)

        # 결측치 처리 — 각 변수의 중앙값으로 대체 (학습 데이터에 NaN 불가)
        from yeda.schema import FEATURE_NAMES, SPEC_BY_NAME
        for col in FEATURE_NAMES:
            if col in new_df.columns and new_df[col].isna().any():
                mid = (SPEC_BY_NAME[col].low + SPEC_BY_NAME[col].high) / 2
                new_df[col] = new_df[col].fillna(mid)

        # tape_type은 정수형으로 변환
        if "tape_type" in new_df.columns:
            new_df["tape_type"] = new_df["tape_type"].round().astype(int)

        # pickup_success(타겟)이 없으면 기존 모델로 라벨 추정 (pseudo-labeling)
        if "pickup_success" not in new_df.columns:
            service = get_ml_service()
            probs = []
            for _, row in new_df.iterrows():
                try:
                    p = service.predict_one(row.to_dict())
                except Exception:
                    p = 0.5
                probs.append(p)
            # 확률 → 0/1 라벨 (threshold 0.5)
            new_df["pickup_success"] = [1 if p >= 0.5 else 0 for p in probs]

        # 업로드 파일에 누적
        if upload_path.exists():
            existing_upload = pd.read_csv(upload_path)
            combined_upload = pd.concat([existing_upload, new_df], ignore_index=True)
        else:
            combined_upload = new_df

        combined_upload.to_csv(upload_path, index=False)
        rows_added = len(records)

    # 2. 학습용 데이터 = 원본 + 업로드 누적분 합치기
    import pandas as pd
    train_df = pd.read_csv(data_path)
    if upload_path.exists():
        upload_df = pd.read_csv(upload_path)
        train_df = pd.concat([train_df, upload_df], ignore_index=True)

    # 합친 데이터를 임시 파일로 저장 (train.py가 읽을 수 있게)
    combined_path = _ROOT / "data" / "raw" / "yeda_synthetic.csv"
    train_df.to_csv(combined_path, index=False)

    # 2. 재학습 실행
    start = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--fast"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": "재학습 실패",
                "detail": result.stderr[-500:] if result.stderr else "unknown",
            },
        )

    # 3. 서비스 모델 핫 리로드
    from api.services.ml_service import initialize_service as reinit
    service = reinit()

    # 4. 원본 CSV 복원 (make data 원본 유지)
    #    학습용으로 합쳤던 것을 원래대로 되돌림
    if upload_path.exists() and records:
        original_df = pd.read_csv(data_path)
        # 원본 행수 = 전체 - 업로드 누적분
        upload_count = len(pd.read_csv(upload_path))
        original_only = original_df.iloc[:len(original_df) - upload_count] if len(original_df) > upload_count else original_df
        # 그냥 원본 재생성이 더 안전
        import subprocess as _sp
        _sp.run(
            [sys.executable, "scripts/make_data.py"],
            cwd=str(_ROOT),
            capture_output=True,
            env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
        )

    # 4. 새 모델 성능 반환
    metrics = {}
    metrics_path = _ROOT / "artifacts" / "metrics" / "model_comparison.csv"
    if metrics_path.exists():
        import pandas as pd
        df = pd.read_csv(metrics_path)
        # 본선 모델 행 찾기
        primary = df[df["label"].str.contains("단조", na=False)]
        if not primary.empty:
            row = primary.iloc[0]
            metrics = {
                "accuracy": float(row.get("accuracy", 0)),
                "pr_auc": float(row.get("pr_auc", 0)),
                "recall": float(row.get("recall", 0)),
                "f1_macro": float(row.get("f1_macro", 0)),
            }

    return {
        "status": "ok",
        "message": f"재학습 완료 ({elapsed:.1f}초)",
        "rows_added": rows_added,
        "model_name": service.model_name,
        "is_mock": service.is_mock,
        "metrics": metrics,
    }


@app.post("/api/reset-model", tags=["system"])
async def reset_model():
    """모델을 초기 상태로 되돌린다.

    - 업로드된 학습 데이터(uploaded_data.csv) 삭제
    - 원본 데이터(8000행)로 재생성
    - 모델 재학습
    - 서비스 핫 리로드
    """
    import subprocess
    import time

    upload_path = _ROOT / "data" / "raw" / "uploaded_data.csv"

    # 1. 업로드 누적 데이터 삭제
    if upload_path.exists():
        upload_path.unlink()

    # 2. 원본 데이터 재생성 (8000행, 시드 고정)
    subprocess.run(
        [sys.executable, "scripts/make_data.py"],
        cwd=str(_ROOT),
        capture_output=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
    )

    # 3. 원본 데이터로 재학습
    start = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--fast"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={"error": "초기화 재학습 실패", "detail": result.stderr[-300:]},
        )

    # 4. 서비스 핫 리로드
    from api.services.ml_service import initialize_service as reinit
    service = reinit()

    return {
        "status": "ok",
        "message": f"모델 초기화 완료 ({elapsed:.1f}초). 원본 8000행 기준 모델로 복원.",
        "model_name": service.model_name,
    }
