# ROLES.md — 3-Tier 아키텍처 역할 분담과 파일 소유권

> **이 문서의 목적은 24시간 동안 5명이 서로 안 부딪히게 하는 것이다.**
> 해커톤에서 진짜 리스크는 실력이 아니라 **git 충돌과 대기 시간**이다.
> 아래 소유권 표를 지키면 대부분의 충돌이 원천 차단된다.
>
> **아키텍처**: 3-Tier (S3 프론트엔드 / EC2:8000 백엔드 / MySQL)

## 팀원 배정

| 코드 | 전공 | 역할 | 한 줄 요약 |
|---|---|---|---|
| **A** | 전자공학과 | 공정물리 · 데이터 검수 | 물리 관계를 정의하고 생성기를 검수한다 |
| **B** | 전기정보공학과 | DB · 인프라 · Q&A 방어 | MySQL 스키마, 배포 설계, 기술 방어 논리를 만든다 |
| **C** | 인공지능·소프트웨어학과 | 데이터 · 모델 · ML 서비스 | 생성기 구현, 3종 모델 비교, API 모델 서비스 |
| **D** | 인공지능·소프트웨어학과 | API 라우터 · 설명 · 최적화 | FastAPI 라우터, SHAP 분해, 조건 탐색 |
| **E** | 인공지능·소프트웨어학과 | 프론트엔드 · 통합 · 데모 | 정적 프론트엔드, FastAPI 앱 초기화, 안정화 |

> **TODO(팀장): 킥오프 때 A~E 자리에 실제 이름을 채워 넣을 것.**

---

## 왜 이렇게 나눴는가

### 3-Tier 분업 원칙

기존 Streamlit 1-Tier와 달리 3개 계층이 존재한다.
E 한 명에게 3개 계층을 모두 맡기면 병목이 된다.
따라서 **계층별로 소유자를 분산**한다:

| 계층 | 주 담당 | 보조 |
|---|---|---|
| Tier 1 — Frontend (S3) | **E** | D (API 연동 인터페이스) |
| Tier 2 — Backend API (EC2:8000) | **D** (라우터) + **C** (ML 서비스) | E (main.py, CORS) |
| Tier 3 — Database (MySQL) | **B** | D (쿼리 호출) |

### 하드웨어 전공자(A·B)의 역할

- **A**: 프로젝트의 차별점인 "공정 물리를 모델 제약으로 주입했다"는 서사의 신뢰도를 만든다.
- **B**: 기존 "센서 아키텍처" 역할에 **MySQL 스키마 설계와 배포 인프라**를 추가한다.
  SQL DDL 작성과 AWS 배포 가이드는 전기정보공학 전공의 시스템 설계 역량과 일치한다.

### AI/SW 3명 분업

- **C(데이터·모델·ML서비스)**: ML 파이프라인 + `api/services/ml_service.py`. 모델을 학습하고, API에서 로드할 수 있게 서비스 레이어를 제공한다.
- **D(API 라우터·설명·최적화)**: 기존 SHAP/최적화 로직을 FastAPI 엔드포인트로 노출한다. C의 학습을 기다릴 필요 없이 `fake_score`로 개발 시작.
- **E(프론트엔드·통합)**: 정적 프론트엔드 + `api/main.py` 앱 초기화. mock 응답으로 킥오프 직후부터 화면을 만든다.

---

## 파일 소유권 (충돌 방지의 핵심)

**규칙: 자기 소유가 아닌 파일은 고치지 않는다.** 고쳐야 하면 소유자에게 말한다.

---

### A — 공정물리 · 데이터 검수

| 소유 파일 | 내용 |
|---|---|
| `configs/data_gen.yaml` | 변수별 기여 가중치, 상호작용 계수, 샘플링 설정 |
| `docs/PHYSICS_RATIONALE.md` | 변수 13개의 물리적 근거 (발표 Q&A 원본) |
| `docs/DATA_CARD.md` | 합성 데이터 명세 카드 |

**완료 정의 (DoD)**
- [ ] 13개 변수 각각에 대해 "방향(+/-)·근거 1문장·출처"가 `PHYSICS_RATIONALE.md`에 적혀 있다
- [x] `schema.py`의 `monotone` 값 8개가 물리적으로 타당함을 A가 확인하고 서명했다
- [ ] 상호작용 3개 각각에 대해 "왜 이 두 변수가 곱해지는가"를 한 문단으로 설명할 수 있다
- [x] 생성된 데이터의 변수별 기여 곡선을 보고 "물리적으로 말이 된다"고 승인했다
- [ ] "이 데이터는 당신들이 만든 규칙 아닙니까" 질문의 답변을 준비했다

**인터페이스 접점**
- → C: `configs/data_gen.yaml`을 넘긴다. C는 읽기만 한다.
- → B: 단조 제약 방향 공동 확정.
- ← C: 생성 결과 검수 리포트를 받아 판정한다.

**즉시 착수 태스크**: `configs/data_gen.yaml` 가중치 조정 + `PHYSICS_RATIONALE.md` 작성 시작

---

### B — DB · 인프라 · Q&A 방어

| 소유 파일 | 내용 |
|---|---|
| `scripts/init_db.sql` | MySQL 데이터베이스·테이블 생성 DDL |
| `api/db/connection.py` | MySQL 연결 풀 관리 |
| `api/db/queries.py` | 예측 저장, 알림 저장, 이력 조회 SQL |
| `scripts/deploy.sh` | S3 업로드 + EC2 배포 스크립트 |
| `docs/ARCHITECTURE.md` | 3-Tier 아키텍처 흐름도, AWS 배치안 |
| `docs/QA_DEFENSE.md` | 예상 질문 20개와 답변 |

**완료 정의 (DoD)**
- [x] `scripts/init_db.sql` 실행 시 `yeda` DB + `predictions`, `alerts` 테이블 생성
- [x] `api/db/connection.py`가 `.env`의 DB 접속 정보로 연결 풀을 생성한다
- [x] `api/db/queries.py`에 `save_prediction()`, `save_alert()`, `get_history()` 구현
- [x] `ARCHITECTURE.md`에 3-Tier 다이어그램, 각 계층 역할, 통신 방식이 적혀 있다
- [ ] `QA_DEFENSE.md`에 최소 20개 질문·답변 (5개는 "왜 3-Tier인가" 관련)
  - 현재 16개 중 Q5, Q9, Q17~Q20이 TODO 상태
- [x] `scripts/deploy.sh`에 S3 sync + EC2 서비스 재시작 명령이 있다
- [x] DB 비밀번호가 코드에 하드코딩되지 않고 `.env`에서만 읽힌다

**인터페이스 접점**
- → D: `api/db/queries.py`의 함수를 D가 라우터에서 호출한다.
- → E: 배포 스크립트를 E가 `Makefile`에 연결한다.
- → A: 단조 제약 방향 공동 확정.
- → 전원: 발표 30분 전 Q&A 리허설을 주도한다.

**즉시 착수 태스크**: `scripts/init_db.sql` 작성 + `api/db/connection.py` 구현 + `ARCHITECTURE.md` 3-Tier 버전 작성

---

### C — 데이터 · 모델 · ML 서비스

| 소유 파일 | 내용 |
|---|---|
| `src/yeda/data/generator.py` | 합성 데이터 생성기 |
| `src/yeda/data/physics.py` | 잠재 점수 함수 |
| `src/yeda/data/preprocess.py` | 결측 처리, 홀드아웃 분리 |
| `src/yeda/models/*.py` | 학습, 평가, 레지스트리 |
| `api/services/ml_service.py` | 모델 로드·예측·SHAP 배경 데이터 제공 |
| `configs/model.yaml` | 모델 하이퍼파라미터 |
| `scripts/make_data.py`, `scripts/train.py` | CLI |
| `tests/test_generator.py` | 생성기 회귀 테스트 |

**완료 정의 (DoD)**
- [x] `make data`가 경고 없이 통과 (성공률 65~80%, 베이즈 정확도 0.82~0.90)
  - ✅ 실측: 성공률 66.8%, 베이즈 정확도 0.872, 경고 0건
- [x] `make train`이 4종 비교표를 `artifacts/metrics/model_comparison.csv`로 출력
  - ✅ 4종: LogReg(0.809) / RF(0.832) / LightGBM(0.849) / LightGBM+단조(0.854)
- [x] 홀드아웃 정확도 80% 초중반 (95%+ 시 생성기 재점검)
  - ✅ 본선 모델 accuracy=0.854, PR-AUC=0.944, ECE=0.027
- [x] `api/services/ml_service.py`가 모델 로드 + `predict()` + `get_shap_background()` 제공
  - ✅ C가 `is_loaded` 프로퍼티, 모델 분석 모듈 추가 확장
- [x] 모델이 없으면 mock 모드 폴백 (기존 로직 유지)
- [x] `pytest tests/test_generator.py` 전부 통과
  - ✅ + `test_ml_service_C.py`, `test_models_C.py`, `test_yield_experiment.py` 추가

**인터페이스 접점**
- ← A: `configs/data_gen.yaml`을 받아 그대로 사용
- → D: `api/services/ml_service.py`의 함수를 D가 라우터에서 호출
- → E: 모델 상태 (`is_loaded`, `is_mock`) 제공

**즉시 착수 태스크**: `make data` → `make train` 실행 + `api/services/ml_service.py` 뼈대 작성

---

### D — API 라우터 · 설명 · 최적화

| 소유 파일 | 내용 |
|---|---|
| `api/routers/predict.py` | `/api/predict`, `/api/explain` 엔드포인트 |
| `api/routers/optimize.py` | `/api/optimize` 엔드포인트 |
| `api/routers/alert.py` | `/api/alert` 엔드포인트 |
| `api/routers/history.py` | `/api/history`, `/api/presets` 엔드포인트 |
| `api/models/request.py` | Pydantic 요청 스키마 |
| `api/models/response.py` | Pydantic 응답 스키마 |
| `src/yeda/explain/shap_explainer.py` | SHAP 기여도 분해 |
| `src/yeda/optimize/search.py` | 조건 탐색 및 가이드 생성 |
| `configs/optimize.yaml` | 탐색 제약 |
| `tests/test_optimize.py` | 최적화 계약 테스트 |
| `tests/test_api.py` | API 엔드포인트 테스트 |

**완료 정의 (DoD)**
- [x] `POST /api/predict` — 13개 피처 입력 → 확률 + risk_level 반환
- [x] `POST /api/explain` — SHAP 기여도 JSON 반환 (mock 모드 동작 확인, 실물은 모델 로드 후)
- [x] `POST /api/optimize` — 고정 변수를 건드리지 않는 최적화 제안 반환
- [x] `POST /api/alert` — dry-run 본문 반환
- [x] `GET /api/history` — 인메모리 이력 조회 (B의 MySQL 연동 대기)
- [x] `GET /api/presets` — `configs/app.yaml`의 프리셋 목록 반환
- [x] `GET /api/health` — 모델 로드 상태 확인
- [x] 모든 엔드포인트에 적절한 에러 처리 및 HTTP 상태 코드
- [x] `pytest tests/test_api.py` 통과 (23개 테스트)

**인터페이스 접점**
- ← C: `api/services/ml_service.py`의 함수를 호출
- ← B: `api/db/queries.py`의 함수를 호출
- → E: 라우터를 `api/main.py`에 등록 (E가 import)
- → E: API 응답 형식을 프론트엔드에서 사용

**즉시 착수 태스크**: `api/models/request.py` + `api/models/response.py` 정의 → `api/routers/predict.py` 뼈대 (fake_score로 개발)

---

### E — 프론트엔드 · 통합 · 데모

| 소유 파일 | 내용 |
|---|---|
| `frontend/index.html` | 메인 페이지 |
| `frontend/js/*.js` | API 호출, 탭 전환, 렌더링 |
| `frontend/css/*.css` | 스타일 |
| `api/main.py` | FastAPI 앱 엔트리포인트 (CORS, 라우터 등록) |
| `src/yeda/alerts/email.py` | 이메일 알림 |
| `src/yeda/schema.py` | **공용 계약** (변경 시 전원 공지) |
| `src/yeda/io_utils.py`, `src/yeda/text_utils.py` | 공용 유틸 |
| `configs/app.yaml` | UI · 알림 설정 |
| `.env.example` | 환경변수 템플릿 |
| `Makefile`, `scripts/run_demo.sh` | 실행 편의 |
| `requirements.txt` | 의존성 (변경 시 전원 공지) |

**완료 정의 (DoD)**
- [ ] `make api` → uvicorn 서버가 포트 8000에서 뜬다
- [ ] `make frontend` → 프론트엔드가 로컬에서 서빙된다
- [ ] 프론트엔드 4개 탭(예측/원인/가이드/알림)이 API를 호출하여 동작한다
- [ ] 백엔드가 죽어도 프론트엔드 페이지가 로드된다 (에러 메시지 표시)
- [ ] 데모 프리셋 버튼으로 원클릭 입력 가능
- [x] `api/main.py`에 CORS 설정으로 S3 도메인 허용
- [ ] `.env.example`에 DB 접속 정보(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME) 추가
- [x] `requirements.txt`에 `fastapi`, `uvicorn[standard]`, `httpx` 추가
- [ ] **마지막 3시간**: 기능 추가 중단, 리허설·안정화만

**인터페이스 접점**
- ← D: `api/routers/`를 `api/main.py`에서 import·등록
- ← B: 배포 스크립트를 Makefile에 연결
- → 전원: `schema.py`, `requirements.txt` 변경은 E가 머지하고 전원 공지

**즉시 착수 태스크**: `api/main.py` 초기화 (CORS + health 엔드포인트) + `frontend/` 디렉토리 구조 생성 + Makefile 업데이트

---

## 킥오프 즉시 병렬 착수 맵

```
시간 0h                                                              1h
 │                                                                    │
 A ──── data_gen.yaml 조정 + PHYSICS_RATIONALE.md 작성 ──────────────▶
 B ──── init_db.sql + api/db/ 구현 + ARCHITECTURE.md 3-Tier ─────────▶
 C ──── make data → make train + api/services/ml_service.py ─────────▶
 D ──── api/models/ 스키마 + api/routers/ 뼈대 (fake_score) ─────────▶
 E ──── api/main.py + frontend/ 구조 + Makefile 업데이트 ────────────▶
 │                                                                    │
 ╰─── 전원: schema.py 확정, .env.example 확인, 브랜치 생성 ───────────╯
```

**의존 관계 (화살표 = "이게 있어야 다음 단계 가능")**:
```
A → (data_gen.yaml) → C → (ml_service) → D → (라우터) → E → (통합)
B → (db/queries.py) → D → (history 라우터)
E → (main.py + CORS) → D → (라우터 등록)
```

**핵심**: A·B·C·D·E 모두 **첫 1시간부터 독립 작업 가능**하다.
의존성이 생기는 시점은 M2(4h) 이후 통합 단계부터다.

---

## 공용 파일 변경 규칙

| 파일 | 확정 시점 | 변경 절차 |
|---|---|---|
| `src/yeda/schema.py` | M0 (킥오프 +1h) | E에게 요청 → E가 수정·공지 → 전원 pull |
| `configs/data_gen.yaml` | M1 | A만 수정. 변경 시 C에게 알림 |
| `requirements.txt` | M0 | E만 수정. 버전 변경 시 전원 공지 |
| `api/models/request.py` | M0 | D가 확정. 변경 시 E에게 알림 (프론트엔드 영향) |
| `.env.example` | M0 | E가 관리. 새 변수 추가 시 전원 공지 |

---

## 브랜치 전략

| 브랜치 | 소유자 | 목적 |
|---|---|---|
| `feat/data-C` | C | 데이터 생성 + 모델 학습 |
| `feat/api-D` | D | FastAPI 라우터 + Pydantic 스키마 |
| `feat/frontend-E` | E | 프론트엔드 + main.py |
| `feat/db-B` | B | MySQL 스키마 + DB 연동 |
| `docs/physics-A` | A | 물리 근거 문서 |
| `docs/arch-B` | B | 아키텍처 + QA 문서 |

---

## M0에서 합의해야 하는 것 (30분 안에)

1. **API 엔드포인트 스펙** — D와 E가 같은 JSON 형식을 보고 일한다
2. **MySQL 테이블 스키마** — B가 제안, D가 확인 (쿼리 호출 주체)
3. **`.env` 변수 이름** — DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
4. **CORS 허용 도메인** — 로컬 개발 시 `http://localhost:*`, 배포 시 S3 URL
5. **프리셋 데이터 소스** — `configs/app.yaml`의 기존 프리셋을 API가 그대로 사용

---

## 리스크와 백업

| 리스크 | 징후 | 대응 | 담당 |
|---|---|---|---|
| MySQL 연결 실패 | API 시작 시 에러 | SQLite 폴백 모드 추가 | B → D |
| 생성기 정확도 95%+ | `make data` 경고 | `latent.scale` 하향 | A → C |
| API 응답 느림 (SHAP) | explain 5초+ | 배경 표본 200 → 50 축소 | D |
| S3 배포 권한 없음 | aws cli 에러 | 로컬 http.server로 데모 | E |
| 프론트-백 JSON 불일치 | 화면에 데이터 안 뜸 | request/response 스키마 재확인 | D + E |
| 팀원 1명 이탈 | — | 아래 백업 담당자가 DoD 축소판 수행 | 팀장 |

**백업 담당**: A↔B (도메인·인프라 교차), C↔D (ML·API 교차), E는 D가 백업

---

## 시간 부족 시 범위 축소 순서

앞에 있을수록 먼저 버린다:

1. ~~SECOM 일반화 검증~~
2. ~~이메일 실제 발송~~ (dry-run 본문 표시로 충분)
3. ~~Optuna 최적화~~ (좌표 하강으로 충분)
4. ~~예측 이력 UI~~ (MySQL 저장은 되되 프론트엔드 탭 생략)
5. ~~프론트엔드 디자인 고도화~~ (기본 기능만 동작)
6. ~~S3 실배포~~ (로컬 static 서빙으로 3-Tier 시연)

**절대 버리지 않는 것**:
- 3-Tier 분리 자체 (프론트/백엔드/DB 분리 동작)
- 합성 데이터 명시
- 단조 제약 서사
- `make api` + 프론트엔드로 예측이 뜨는 것

---

> **전원 공통 규칙**: 막히면 **30분 안에** 팀에 알린다.
> 혼자 붙들고 있는 시간이 24시간짜리 일정에서 가장 비싸다.
