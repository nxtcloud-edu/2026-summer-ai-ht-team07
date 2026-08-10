# =============================================================================
# YEDA — 단축 명령
# =============================================================================
# 발표 직전 새 노트북에서도 `make setup && make demo` 두 줄이면 화면이 떠야 한다.
# 명령을 추가할 때는 반드시 README 의 실행 방법도 함께 고칠 것.
# =============================================================================

PYTHON ?= python
STREAMLIT ?= streamlit
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help setup data train demo test lint clean all secom secom-data check

help:  ## 사용 가능한 명령 보기
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## 의존성 설치
	$(PYTHON) -m pip install -r requirements.txt

data:  ## 합성 데이터 생성 (configs/data_gen.yaml)
	$(PYTHON) scripts/make_data.py

check:  ## 기존 데이터 검수만 수행 (생성하지 않음)
	$(PYTHON) scripts/make_data.py --check

train:  ## 모델 3종 학습·비교 후 본선 모델 저장
	$(PYTHON) scripts/train.py

train-fast:  ## 교차검증 생략 학습 (시간이 급할 때)
	$(PYTHON) scripts/train.py --fast

demo:  ## Streamlit 데모 실행 (없으면 데이터·모델 자동 생성)
	bash scripts/run_demo.sh

test:  ## pytest 실행
	$(PYTHON) -m pytest tests -q

all: data train test  ## 데이터 → 학습 → 테스트 전체 파이프라인

secom-data:  ## UCI SECOM 원본 다운로드 (우선순위 낮음)
	$(PYTHON) scripts/download_secom.py

secom:  ## SECOM 일반화 검증 실행 (우선순위 낮음)
	$(PYTHON) -c "import sys; sys.path.insert(0,'src'); from yeda.secom.pipeline import run; print(run())"

clean:  ## 생성물 삭제 (data/raw, artifacts). 설정과 코드는 건드리지 않는다.
	rm -f data/raw/*.csv data/processed/*.csv
	rm -f artifacts/models/*.joblib artifacts/models/*.json
	rm -f artifacts/metrics/*.csv artifacts/metrics/*.json
	rm -f artifacts/figures/*.png
	@echo "생성물을 삭제했습니다. 'make all' 로 재생성하세요."
