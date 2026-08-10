# Yield X 해커톤 — 3-Tier 아키텍처 파이프라인 구축

> **목표**: 현재 Streamlit 1-Tier 스켈레톤을 **3-Tier 아키텍처**로 전환하여
> 프레젠테이션(S3) / 애플리케이션(EC2:8000) / 데이터(MySQL) 계층을 분리한다.
>
> **핵심 원칙**: M3(12h 지점)에서 파이프라인이 조악하게라도 end-to-end로 돌아야 한다.

---

## Requirements

### Requirement 1: 3-Tier 아키텍처 분리

**Description**: 기존 Streamlit 모놀리식 앱을 3개 계층으로 분리한다.

| 계층 | 역할 | 기술 스택 |
|---|---|---|
| Presentation (Tier 1) | 정적 프론트엔드 | S3 호스팅 (React 또는 HTML/JS) |
| Application (Tier 2) | 비즈니스 로직 API | EC2 :8000 (FastAPI) |
| Data (Tier 3) | 영속 데이터 저장 | MySQL (RDS 또는 EC2 내 설치) |

**Acceptance Criteria**:
- [ ] 프론트엔드는 S3에서 정적 파일로 서빙된다
- [ ] 백엔드 API 서버가 EC2의 포트 8000에서 동작한다
- [ ] 데이터는 MySQL에 영속 저장되어 서버 재시작 시에도 유지된다
- [ ] 백엔드 서버가 멈춰도 프론트엔드 화면은 뜬다 (S3 독립 호스팅)
- [ ] DB 비밀번호는 EC2 내부에서만 관리된다 (.env, 절대 프론트엔드에 노출 안 됨)

---

### Requirement 2: Backend API 서버 구축 (EC2 :8000)

**Description**: 기존 `app/components/backend.py`의 로직을 FastAPI REST API로 전환한다.

**Acceptance Criteria**:
- [ ] FastAPI 서버가 포트 8000에서 실행된다
- [ ] `POST /api/predict` — 조건 입력 → 픽업 성공 확률 반환
- [ ] `POST /api/explain` — SHAP 기여도 분해 결과 반환
- [ ] `POST /api/optimize` — 조정 가능 변수 최적화 제안 반환
- [ ] `POST /api/alert` — 알림 트리거 (dry-run 지원)
- [ ] `GET /api/health` — 헬스체크 엔드포인트
- [ ] `GET /api/presets` — 데모 프리셋 목록 반환
- [ ] `GET /api/history` — 예측 이력 조회 (MySQL에서)
- [ ] 에러 발생 시 적절한 HTTP 상태 코드와 메시지 반환
- [ ] CORS 설정으로 S3 프론트엔드에서의 요청 허용

---

### Requirement 3: MySQL 데이터 계층 구축

**Description**: 예측 요청/결과/이력을 MySQL에 저장하여 서버 재시작 시에도 데이터가 유지되도록 한다.

**Acceptance Criteria**:
- [ ] MySQL 데이터베이스 `yeda` 생성
- [ ] 예측 이력 테이블 (`predictions`) — 입력 조건, 예측 결과, 타임스탬프 저장
- [ ] 알림 이력 테이블 (`alerts`) — 발송/dry-run 이력 저장
- [ ] DB 접속 정보는 `.env`에서 읽으며 코드에 하드코딩하지 않는다
- [ ] 서버 재시작 후에도 과거 예측 이력이 조회된다
- [ ] DB 마이그레이션 스크립트 제공 (`scripts/init_db.sql` 또는 ORM migration)

---

### Requirement 4: 프론트엔드 (S3 정적 호스팅)

**Description**: 기존 Streamlit 4개 탭 기능을 정적 웹 프론트엔드로 전환하여 S3에서 호스팅한다.

**Acceptance Criteria**:
- [ ] 정적 HTML/JS/CSS (또는 React 빌드 산출물)로 S3 배포 가능
- [ ] 4개 기능 탭 구현: 수율 예측 / 원인 분해 / 개선 가이드 / 알림
- [ ] 백엔드 API(EC2:8000)를 호출하여 데이터를 표시
- [ ] 백엔드가 죽어도 프론트엔드 페이지 자체는 로드됨 (에러 메시지 표시)
- [ ] 데모 프리셋 버튼으로 고정 조건을 원클릭 입력 가능
- [ ] 반응형 레이아웃으로 발표 시 보기 좋은 UI

---

### Requirement 5: 환경 셋업 및 배포 자동화

**Description**: 3-Tier 환경을 로컬/AWS에서 쉽게 구성할 수 있도록 스크립트와 설정을 제공한다.

**Acceptance Criteria**:
- [ ] `.env.example`에 MySQL 접속 정보 항목 추가 (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
- [ ] `make api` — FastAPI 서버 로컬 실행 (uvicorn, port 8000)
- [ ] `make frontend` — 프론트엔드 로컬 개발 서버 또는 빌드
- [ ] `make init-db` — MySQL 스키마 초기화
- [ ] `scripts/deploy.sh` — S3 업로드 + EC2 배포 가이드 (또는 자동화)
- [ ] `requirements.txt`에 FastAPI, uvicorn, mysql-connector (또는 SQLAlchemy+pymysql) 추가
- [ ] 로컬에서 `make api` + 브라우저로 전체 플로우 테스트 가능

---

### Requirement 6: 데이터 파이프라인 유지 (기존 ML 파이프라인)

**Description**: 기존 `make data` → `make train` ML 파이프라인을 유지하고, 학습된 모델을 API 서버가 로드하도록 연결한다.

**Acceptance Criteria**:
- [ ] `make data` → 합성 데이터 생성 (기존과 동일)
- [ ] `make train` → 모델 학습 및 `artifacts/models/` 저장 (기존과 동일)
- [ ] API 서버 시작 시 `artifacts/models/primary_model.joblib` 자동 로드
- [ ] 모델이 없으면 mock 모드로 동작 (기존 로직 유지)
- [ ] `pytest tests/` 기존 테스트 통과 유지

---

### Requirement 7: 발표 및 데모 안정화

**Description**: 24시간 해커톤 발표에서 3-Tier 구조를 시연할 수 있도록 안정화한다.

**Acceptance Criteria**:
- [ ] 리허설 3회 중 최소 2회 사고 없이 완료
- [ ] `make demo`가 3-Tier 모드로 동작 (API 서버 + 프론트엔드)
- [ ] 백엔드 장애 시 프론트엔드에서 "서버 연결 실패" 안내 표시
- [ ] 데모 화면 녹화본 백업 준비
- [ ] Q&A 대비: "왜 3-Tier인가" 설명 가능 (데이터 영속성, 프론트 독립성, 보안)

---

## Design

### 3-Tier 아키텍처 다이어그램

```
┌─────────────────┐       HTTP        ┌─────────────────────┐       SQL       ┌──────────────┐
│   Tier 1        │  ───────────────▶  │      Tier 2         │  ────────────▶  │   Tier 3     │
│   Frontend      │  ◀───────────────  │      Backend        │  ◀────────────  │   Database   │
│                 │     JSON Response  │                     │     Query Result│              │
│   S3 Static     │                    │   EC2 :8000         │                 │   MySQL      │
│   HTML/JS/CSS   │                    │   FastAPI (Python)  │                 │              │
└─────────────────┘                    └─────────────────────┘                 └──────────────┘
                                              │
                                              ▼
                                    artifacts/models/
                                    (joblib 모델 파일)
```

### 1-Tier vs 3-Tier 비교

| 항목 | 1-Tier (기존 Streamlit) | 3-Tier (목표) |
|---|---|---|
| 화면이 있는 곳 | EC2 (Streamlit) | **S3** |
| 로직이 도는 곳 | 같은 EC2 | **EC2 :8000** |
| 데이터 위치 | 서버 메모리 | **MySQL** |
| 서버 재시작 | 데이터 사라짐 | **유지됨** |
| 서버가 멈추면 | 화면도 안 뜸 | **화면은 뜸** |
| DB 비밀번호 | 해당 없음 | **EC2 안에만** |

### 디렉토리 구조 (변경 후)

```
2026-summer-ai-ht-team07/
├── frontend/                    # Tier 1 — S3 배포용 정적 파일
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── assets/
├── api/                         # Tier 2 — FastAPI 백엔드
│   ├── main.py                  # FastAPI 앱 엔트리포인트
│   ├── routers/
│   │   ├── predict.py           # /api/predict, /api/explain
│   │   ├── optimize.py          # /api/optimize
│   │   ├── alert.py             # /api/alert
│   │   └── history.py           # /api/history
│   ├── models/                  # Pydantic 요청/응답 스키마
│   │   ├── request.py
│   │   └── response.py
│   ├── db/
│   │   ├── connection.py        # MySQL 연결 관리
│   │   └── queries.py           # SQL 쿼리
│   └── services/
│       └── ml_service.py        # 기존 backend.py 로직 래핑
├── scripts/
│   ├── init_db.sql              # MySQL 스키마 초기화
│   ├── deploy.sh                # 배포 스크립트
│   ├── make_data.py             # (기존 유지)
│   └── train.py                 # (기존 유지)
├── src/yeda/                    # (기존 ML 코드 유지)
├── app/                         # (기존 Streamlit — 레거시 참조용)
├── configs/
├── artifacts/
├── data/
├── tests/
│   ├── test_api.py              # API 엔드포인트 테스트
│   └── ...                      # (기존 테스트 유지)
├── .env.example
├── requirements.txt
└── Makefile
```

### API 엔드포인트 설계

| Method | Path | 설명 | 요청 | 응답 |
|---|---|---|---|---|
| GET | `/api/health` | 헬스체크 | - | `{"status": "ok", "model_loaded": true}` |
| GET | `/api/presets` | 데모 프리셋 | - | `[{name, description, values}]` |
| POST | `/api/predict` | 수율 예측 | `{feature_values}` | `{probability, risk_level}` |
| POST | `/api/explain` | SHAP 분해 | `{feature_values}` | `{shap_values, base_value}` |
| POST | `/api/optimize` | 최적화 제안 | `{feature_values}` | `{recommendations, expected_improvement}` |
| POST | `/api/alert` | 알림 트리거 | `{feature_values, probability}` | `{sent, message_body}` |
| GET | `/api/history` | 예측 이력 | `?limit=20` | `[{id, timestamp, input, result}]` |

### MySQL 스키마 설계

```sql
CREATE DATABASE IF NOT EXISTS yeda;

CREATE TABLE predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_json JSON NOT NULL,
    probability FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    shap_json JSON
);

CREATE TABLE alerts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_id BIGINT,
    trigger_level VARCHAR(20),
    dry_run BOOLEAN DEFAULT TRUE,
    recipients TEXT,
    message_body TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
```

### 보안 원칙

- DB 비밀번호는 `.env` → EC2 환경변수로만 주입
- 프론트엔드(S3)에는 API URL만 설정, 시크릿 절대 노출 안 됨
- CORS 화이트리스트로 허용 도메인 제한
- API 입력값 Pydantic 검증으로 인젝션 방지

---

## Tasks

### Task 1: FastAPI 백엔드 프로젝트 구조 생성
- **Requirement**: 2
- [ ] `api/` 디렉토리 및 하위 구조 생성 (`main.py`, `routers/`, `models/`, `db/`, `services/`)
- [ ] `api/main.py` — FastAPI 앱 초기화, CORS 설정, 라우터 등록
- [ ] `requirements.txt`에 `fastapi`, `uvicorn[standard]`, `pymysql`, `sqlalchemy` 추가

### Task 2: Pydantic 요청/응답 스키마 정의
- **Requirement**: 2
- [ ] `api/models/request.py` — `PredictRequest` (13개 피처), `OptimizeRequest` 등
- [ ] `api/models/response.py` — `PredictResponse`, `ExplainResponse`, `OptimizeResponse`, `HealthResponse`
- [ ] 기존 `src/yeda/schema.py`의 변수 정의와 일관성 유지

### Task 3: ML 서비스 레이어 구현
- **Requirement**: 2, 6
- [ ] `api/services/ml_service.py` — 기존 `app/components/backend.py` 로직을 서비스 클래스로 래핑
- [ ] 모델 로드 (`artifacts/models/primary_model.joblib`), mock 폴백
- [ ] SHAP explainer 초기화 및 캐싱
- [ ] 최적화 함수 연결 (`src/yeda/optimize/`)

### Task 4: API 라우터 구현
- **Requirement**: 2
- [ ] `api/routers/predict.py` — `/api/predict`, `/api/explain`
- [ ] `api/routers/optimize.py` — `/api/optimize`
- [ ] `api/routers/alert.py` — `/api/alert`
- [ ] `api/routers/history.py` — `/api/history`, `/api/presets`
- [ ] `/api/health` 헬스체크

### Task 5: MySQL 데이터 계층 구현
- **Requirement**: 3
- [ ] `scripts/init_db.sql` — 데이터베이스 및 테이블 생성 DDL
- [ ] `api/db/connection.py` — MySQL 연결 풀 관리 (SQLAlchemy 또는 pymysql)
- [ ] `api/db/queries.py` — 예측 저장, 알림 저장, 이력 조회 함수
- [ ] `.env.example`에 DB 접속 정보 항목 추가

### Task 6: 프론트엔드 구현
- **Requirement**: 4
- [ ] `frontend/index.html` — 메인 페이지, 4개 탭 네비게이션
- [ ] `frontend/js/api.js` — 백엔드 API 호출 유틸리티 (fetch wrapper)
- [ ] `frontend/js/app.js` — 탭 전환, 폼 제출, 결과 렌더링 로직
- [ ] `frontend/css/style.css` — 발표용 깔끔한 스타일
- [ ] 프리셋 버튼 구현 (API에서 프리셋 목록 가져와 원클릭 입력)
- [ ] 백엔드 장애 시 에러 안내 UI 구현

### Task 7: Makefile 및 환경 설정 업데이트
- **Requirement**: 5
- [ ] `Makefile`에 `api`, `frontend`, `init-db` 타겟 추가
- [ ] `make api` — `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- [ ] `make init-db` — MySQL 스키마 초기화 실행
- [ ] `make frontend` — 프론트엔드 로컬 서빙 (python -m http.server 등)
- [ ] `make demo` 업데이트 — 3-Tier 모드 실행

### Task 8: 배포 스크립트 및 가이드
- **Requirement**: 5
- [ ] `scripts/deploy.sh` — S3 업로드 (`aws s3 sync frontend/ s3://bucket-name`)
- [ ] EC2 배포 가이드 (systemd 서비스 또는 docker-compose)
- [ ] MySQL 초기 설정 가이드
- [ ] README에 3-Tier 아키텍처 배포 방법 추가

### Task 9: API 테스트 작성
- **Requirement**: 2, 6
- [ ] `tests/test_api.py` — FastAPI TestClient 기반 엔드포인트 테스트
- [ ] `/api/health` 정상 응답 확인
- [ ] `/api/predict` 정상 케이스 + 잘못된 입력 에러 확인
- [ ] `/api/history` DB 연동 확인 (테스트용 SQLite 또는 mock)
- [ ] 기존 `pytest tests/` 전부 통과 유지

### Task 10: End-to-End 통합 검증
- **Requirement**: 1, 7
- [ ] 로컬에서 MySQL + API 서버 + 프론트엔드 3개 모두 실행
- [ ] 프론트엔드에서 프리셋 선택 → 예측 → SHAP → 최적화 → 알림 플로우 동작 확인
- [ ] 예측 결과가 MySQL에 저장되고 이력 조회 가능 확인
- [ ] API 서버 종료 후 프론트엔드가 여전히 뜨는지 확인
- [ ] API 서버 재시작 후 MySQL 데이터 유지 확인

### Task 11: 발표 준비 및 리허설
- **Requirement**: 7
- [ ] 3-Tier 아키텍처 설명 슬라이드 준비
- [ ] "1-Tier vs 3-Tier" 비교표 슬라이드 (이미지 기반)
- [ ] 데모 시나리오 대본 — 3-Tier 장점 시연 (서버 끄기 → 화면 유지, DB 영속성)
- [ ] 리허설 3회 수행
- [ ] 데모 녹화 백업

---

## 마일스톤 타임라인 (3-Tier 기준)

| 마일스톤 | 경과 | 핵심 산출물 |
|---|---|---|
| **M0** 셋업 | 0~1h | 환경 셋업, 역할 확정 |
| **M1** 데이터 | 1~4h | `make data`, `make train` 완료 |
| **M2** API 서버 | 4~8h | FastAPI + MySQL 동작, `/api/predict` 응답 |
| **M3** E2E 통합 | 8~12h | **프론트엔드 → API → DB 전체 플로우 동작** |
| **M4** 품질 개선 | 12~18h | UI 개선, 에러 처리, 이력 기능 |
| **M5** 발표 자료 | 18~21h | 슬라이드 + 3-Tier 비교 설명 |
| **M6** 안정화 | 21~24h | 리허설 3회, 백업, 기능 추가 금지 |

---

## 범위 축소 전략 (시간 부족 시)

버리는 순서 (앞에 있을수록 먼저 버림):
1. ~~SECOM 일반화 검증~~
2. ~~이메일 실제 발송~~ (dry-run으로 충분)
3. ~~Optuna 최적화~~ (좌표 하강으로 대체)
4. ~~예측 이력 기능~~ (MySQL 연동은 있되 UI 생략 가능)
5. ~~프론트엔드 디자인 고도화~~ (기본 기능만 동작하면 됨)

**절대 버리지 않는 것**:
- 3-Tier 분리 자체 (S3 / EC2:8000 / MySQL)
- `make demo`로 전체 플로우가 뜨는 것
- 합성 데이터 명시
- 단조 제약 서사
