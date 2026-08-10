#!/usr/bin/env bash
# YEDA 데모 실행 — `make demo` 가 호출한다.
#
# 데이터/모델이 없으면 자동으로 만든 뒤 앱을 띄운다.
# 발표 직전 새 노트북에서도 이 한 줄이면 화면이 뜨는 것이 목표다.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f data/raw/yeda_synthetic.csv ]; then
  echo "[demo] 데이터가 없어 생성합니다..."
  python scripts/make_data.py
fi

if [ ! -f artifacts/models/primary_model.joblib ]; then
  echo "[demo] 학습된 모델이 없어 학습합니다..."
  python scripts/train.py --fast
fi

echo "[demo] Streamlit 실행 — 브라우저가 열리지 않으면 http://localhost:8501 로 접속하세요."
exec streamlit run app/streamlit_app.py "$@"
