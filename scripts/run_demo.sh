#!/usr/bin/env bash
# YEDA 3-Tier 데모 실행 — `make demo` 가 호출한다.
#
# 데이터/모델이 없으면 자동으로 만든 뒤 백엔드+프론트엔드를 띄운다.
# 발표 직전 새 노트북에서도 이 한 줄이면 화면이 뜨는 것이 목표다.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=src

if [ ! -f data/raw/yeda_synthetic.csv ]; then
  echo "[demo] 데이터가 없어 생성합니다..."
  python scripts/make_data.py
fi

if [ ! -f artifacts/models/primary_model.joblib ]; then
  echo "[demo] 학습된 모델이 없어 학습합니다..."
  python scripts/train.py --fast
fi

echo "[demo] 백엔드 시작 (포트 8000)..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 2

echo "[demo] 프론트엔드 시작 (포트 3000)..."
python -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

echo "============================================"
echo "  YEDA 데모 실행 중"
echo "  프론트엔드: http://localhost:3000"
echo "  백엔드 API: http://localhost:8000/docs"
echo "  종료: Ctrl+C"
echo "============================================"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" EXIT INT TERM
wait
