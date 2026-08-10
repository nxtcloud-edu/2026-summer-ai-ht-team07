# ARCHITECTURE.md — YIELD-X 시스템 아키텍처

> **소유자: B (전기정보공학과 — DB · 인프라 · Q&A 방어)**
>
> 이 문서는 YIELD-X 3-Tier 아키텍처의 설계, 현재 구현 상태, 그리고
> 논리적 계층과 물리적 배치의 차이를 명확히 기술한다.
>
> **원칙**: 완료된 것과 진행 중인 것을 구분해서 쓴다. 숨기면 Q&A에서 무너진다.

---

## 1. YIELD-X 3-Tier Architecture

```
┌─────────────────────┐        HTTP/JSON        ┌───────────────────────────────┐        SQL         ┌─────────────────┐
│  Presentation Tier  │  ─────────────────────▶  │       Application Tier        │  ───────────────▶  │    Data Tier    │
│                     │  ◀─────────────────────  │                               │  ◀───────────────  │                 │
│  S3 Static Website  │                          │  EC2                          │                    │  MariaDB        │
│  YIELD-X Frontend   │                          │  FastAPI :8000                │                    │  127.0.0.1:3306 │
│  (HTML/JS/CSS)      │                          │  - ML inference (predict)     │                    │  database: yeda │
│                     │                          │  - SHAP explain               │                    │  - predictions  │
│                     │                          │  - Optimization guide         │                    │  - alerts       │
│                     │                          │  - Alert API                  │                    │                 │
└─────────────────────┘                          └───────────────────────────────┘                    └─────────────────┘
```

---

## 2. 통신 방식

| 구간 | 프로토콜 | 설명 |
|---|---|---|
| Browser → S3 | HTTP/HTTPS | 정적 파일 호스팅 (HTML/JS/CSS) |
| Frontend → FastAPI | HTTP/JSON REST API | `/api/predict`, `/api/explain` 등 호출 |
| FastAPI → MariaDB | SQLAlchemy + PyMySQL / SQL | parameter-bound raw SQL 쿼리 |

**포트 정책**:

| 포트 | 용도 | 외부 접근 |
|---|---|---|
| **8000** | FastAPI API 서버 | ✅ 공개 (Security Group 허용) |
| **3306** | MariaDB | ❌ 비공개 (127.0.0.1 전용) |

---

## 3. 논리적 3-Tier vs 물리적 배치

### 논리적 분리

| 계층 | 역할 | 기술 |
|---|---|---|
| Presentation | UI 렌더링, 사용자 입력 | 정적 HTML/JS/CSS (S3) |
| Application | 비즈니스 로직, ML 추론 | FastAPI + uvicorn |
| Data | 영속 저장, 이력 관리 | MariaDB (InnoDB) |

### 물리적 배치 — 해커톤 MVP

```
┌──────────────────────────────────────────────────────────────────────┐
│  EC2 Instance (Amazon Linux 2023)                                    │
│                                                                      │
│  ┌──────────────────────────┐   ┌──────────────────────────────┐    │
│  │  Application Tier        │   │  Data Tier                    │    │
│  │                          │   │                               │    │
│  │  yieldx-api.service      │   │  mariadb.service              │    │
│  │  uvicorn → FastAPI       │──▶│  MariaDB 10.5                 │    │
│  │  0.0.0.0:8000            │   │  127.0.0.1:3306               │    │
│  │                          │   │  database: yeda               │    │
│  └──────────────────────────┘   └──────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  S3 (Presentation Tier) — 구현 예정                                   │
│  정적 프론트엔드 호스팅                                                  │
└──────────────────────────────────────────────────────────────────────┘
```

**MVP에서 Application과 Data를 동일 EC2에 배치한 이유**:

1. **비용 절감** — RDS 별도 인스턴스 비용 불필요
2. **운영 단순화** — 네트워크 지연 없이 localhost 통신
3. **시간 제약** — 24시간 해커톤에서 인프라 복잡도 최소화
4. **충분한 성능** — MVP 규모에서 단일 인스턴스로 충분

**Production 분리 가능성**:

Production 환경에서는 Data Tier를 Amazon RDS (또는 Aurora)로 분리할 수 있다.
현재 코드는 환경변수(`DB_HOST`, `DB_PORT`)만 변경하면 외부 DB를 가리키도록
설계되어 있어, 코드 수정 없이 분리가 가능하다.

---

## 4. systemd 배포 구조

```
.env (EC2 로컬, 권한 600, Git 미포함)
 │
 ▼
systemd EnvironmentFile
 │
 ▼
yieldx-api.service (enabled, active)
 │
 ▼
/home/ec2-user/.pyenv/versions/3.12.12/bin/uvicorn
 │
 ▼
FastAPI (api.main:app)
 │
 ├──▶ ML Service (예측/SHAP/최적화)
 └──▶ MariaDB (SQLAlchemy + PyMySQL → 127.0.0.1:3306)
```

### 서비스 설정

| 항목 | 값 |
|---|---|
| 서비스명 | `yieldx-api.service` |
| User / Group | `ec2-user` |
| WorkingDirectory | `/home/ec2-user/2026-summer-ai-ht-team07` |
| EnvironmentFile | `/home/ec2-user/2026-summer-ai-ht-team07/.env` |
| ExecStart | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| Restart | `on-failure` (RestartSec=5) |
| 부팅 자동 시작 | ✅ enabled (`multi-user.target`) |
| MariaDB 의존성 | `After=mariadb.service`, `Wants=mariadb.service` |

### 환경변수 구조 (.env)

```
DB_HOST=<값>
DB_PORT=<값>
DB_NAME=<값>
DB_USER=<값>
DB_PASSWORD=<값>
```

> `.env` 파일의 실제 값은 이 문서에 기재하지 않는다.
> EC2 로컬에서만 관리되며, Git에 포함되지 않는다.

---

## 5. 보안

| 항목 | 정책 |
|---|---|
| DB 비밀번호 | 코드/Git 하드코딩 금지. `.env`에서만 관리 |
| `.env` 파일 권한 | `600` (소유자만 읽기/쓰기) |
| `.env` Git 관리 | `.gitignore`에 등록. `.env.example`만 커밋 |
| AWS 접근 | IAM Role 기반 |
| MariaDB 접근 | 127.0.0.1 전용. 3306 포트 외부 비공개 |
| API 접근 | 8000 포트만 필요한 범위에서 공개 |
| CORS | `api/main.py`에서 설정 (현재 `allow_origins=["*"]`, 배포 시 S3 도메인으로 제한) |
| SQL Injection 방어 | SQLAlchemy `text()` + parameter binding 사용 |

---

## 6. API 엔드포인트

| Method | Path | 설명 | DB 연동 |
|---|---|---|---|
| GET | `/api/health` | 헬스체크 (모델 로드 상태) | — |
| GET | `/api/presets` | 데모 프리셋 목록 | — |
| POST | `/api/predict` | 13개 변수 → 확률 + 위험등급 | `save_prediction()` |
| POST | `/api/explain` | SHAP 기여도 분해 | — |
| POST | `/api/optimize` | 조정 가능 변수 최적화 제안 | — |
| POST | `/api/alert` | 알림 트리거 (dry-run 기본) | `save_alert()` |
| GET | `/api/history` | 예측 이력 조회 (최신 N건) | `get_history()` |

### DB 연동 흐름

```
POST /api/predict
 → api/routers/predict.py
   → api/routers/history.save_prediction()
     → api/db/queries.save_prediction()  [DB 성공 시]
     → 인메모리 폴백                      [DB 실패 시]

GET /api/history
 → api/routers/history.get_history()
   → api/db/queries.get_history()        [DB 성공 시]
   → 인메모리 폴백                        [DB 실패 시]

POST /api/alert
 → api/routers/alert.py
   → api/routers/history.save_alert_record()
     → api/db/queries.save_alert()       [DB 성공 시]
```

> D의 라우터는 B의 `api/db/queries.py`를 직접 import하며,
> DB 연결 실패 시 인메모리 폴백으로 graceful degradation한다.

---

## 7. 현재 구현 및 검증 상태

### ✅ 완료 — EC2에서 검증됨

| 항목 | 상태 | 검증 내용 |
|---|---|---|
| MariaDB 10.5 설치 및 실행 | 완료 | `systemctl status mariadb` → active (running) |
| `yeda` DB + `predictions`/`alerts` 테이블 | 완료 | `init_db.sql` 실행, FK 포함 정상 생성 |
| `api/db/connection.py` DB 연결 | 완료 | `check_db_connection()` → True |
| `api/db/queries.py` E2E 테스트 | 완료 | `save_prediction`, `get_history`, `save_alert` 모두 PASS |
| `.env` 환경변수 관리 | 완료 | 존재, 권한 600, gitignore 대상 |
| systemd `yieldx-api.service` | 완료 | enabled + active (running) |
| FastAPI 서버 기동 | 완료 | `0.0.0.0:8000`, `/api/health` 정상 응답 |
| EC2 재부팅 후 자동 시작 | 완료 | systemd enabled, mariadb.service 의존성 설정 |
| `/api/health` 응답 | 완료 | `{"status":"ok","model_loaded":false,"is_mock":true}` |
| `/docs` (Swagger UI) 접근 | 완료 | 외부에서 접근 가능 |
| D 라우터 → B DB 쿼리 통합 | 완료 | `predict.py`/`alert.py` → `history.py` → `db/queries.py` 호출 확인 |
| DB 3306 외부 비공개 | 완료 | Security Group에서 3306 미허용 |

### 🔄 통합 진행 중

| 항목 | 상태 | 비고 |
|---|---|---|
| 모델 artifact 배치 | 미완료 | `artifacts/models/`에 `.joblib` 파일 없음. 현재 mock 모드로 동작 |
| Frontend (S3 정적 호스팅) | 미구현 | `frontend/` 디렉토리 미존재. E 파트 작업 대기 |
| S3 배포 | 미완료 | `scripts/deploy.sh` 미존재. 프론트엔드 완성 후 진행 |
| Makefile `api` / `frontend` 타겟 | 미추가 | Makefile에 해당 타겟 없음 |
| `.env.example` DB 항목 추가 | 미완료 | 현재 SMTP 항목만 존재 |

### 📋 향후 확장

| 항목 | 설명 |
|---|---|
| Data Tier를 RDS로 분리 | 환경변수만 변경하면 코드 수정 없이 가능 |
| CORS 도메인 제한 | S3 배포 URL 확정 후 `allow_origins` 제한 |
| HTTPS/TLS | ALB 또는 Nginx reverse proxy 추가 |
| CI/CD | GitHub Actions → EC2 자동 배포 |
| 알림 채널 확장 | Slack, SMS (현재 이메일 dry-run만) |

---

## 8. ML 파이프라인과 Mock 폴백

```
configs/data_gen.yaml ──→ make data ──→ data/raw/yeda_synthetic.csv
                                              │
configs/model.yaml ─────→ make train ─→ artifacts/models/primary_model.joblib
                                              │
                                              ▼
                              api/services/ml_service.py (모델 자동 로드)
                                     │
          ┌────────────────┬─────────┼─────────────┐
          ▼                ▼         ▼             ▼
   shap_explainer.py   search.py   email.py    predict_proba
   (SHAP %p 분해)     (조건 탐색)  (알림)       (수율 예측)
```

**현재 상태**: 모델 파일(`primary_model.joblib`)이 EC2에 배치되지 않아
`ml_service.py`가 mock 모드로 동작 중. 이는 인프라 장애가 아니라
모델 학습 artifact 미배치에 따른 정상적인 폴백이다.

**mock 모드의 의미**:
- API 서버 자체는 정상 기동
- `/api/predict` 등은 mock 응답을 반환
- 모델 파일을 `artifacts/models/`에 배치하면 자동으로 실물 모드 전환
- 데모가 멈추는 일이 없도록 설계됨

---

## 9. 데이터베이스 스키마

```sql
-- database: yeda (MariaDB 10.5, InnoDB, utf8mb4)

predictions
├── id              BIGINT UNSIGNED AUTO_INCREMENT (PK)
├── created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
├── input_json      JSON (13개 feature 원본)
├── probability     DECIMAL(6,5)
├── risk_level      ENUM('critical','warning','normal')
├── die_id          VARCHAR(32) NULL
└── model_name      VARCHAR(100) NULL

alerts
├── id              BIGINT UNSIGNED AUTO_INCREMENT (PK)
├── created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
├── prediction_id   BIGINT UNSIGNED NULL → FK(predictions.id, ON DELETE SET NULL)
├── subject         VARCHAR(255)
├── body            TEXT
├── recipients      JSON
├── sent            BOOLEAN DEFAULT FALSE
├── dry_run         BOOLEAN DEFAULT TRUE
└── error           TEXT NULL
```

---

## 10. 1-Tier(Streamlit) vs 3-Tier 비교

| 항목 | 1-Tier (레거시 Streamlit) | 3-Tier (현재) |
|---|---|---|
| 화면이 있는 곳 | EC2 (Streamlit :8501) | **S3** (예정) |
| 로직이 도는 곳 | 같은 EC2 | **EC2 :8000 (FastAPI)** |
| 데이터 위치 | 서버 메모리 | **MariaDB (영속)** |
| 서버 재시작 시 데이터 | 사라짐 | **유지됨** |
| 서버가 멈추면 | 화면도 안 뜸 | **화면은 뜸** (S3 독립) |
| DB 비밀번호 노출 | 해당 없음 | **EC2 안에만** |

> Streamlit 데모(`make demo`)는 레거시 모드로 여전히 사용 가능하다.

---

## 11. 향후 산업현장 확장 — 설비 연동 설계

> 아래는 실제 산업현장 배포 시의 아키텍처 확장안이다.
> 현재 해커톤 MVP에는 포함되지 않으며, PoC 이후 단계에서 구현한다.

### 목표 아키텍처

```
┌────────────────┐
│  다이 본더 설비  │
│                │
│  PLC / 컨트롤러 │
└───────┬────────┘
        │  ① 데이터 취득
        │     (OPC-UA / MODBUS / 벤더 SDK — 확인 필요)
        ▼
┌────────────────┐      ┌─────────────────────┐
│  Edge Collector │─────→│  MES / 데이터 레이크  │
│  (게이트웨이)    │      └────────┬────────────┘
└────────────────┘               │  ② 배치/스트림 적재
                                 ▼
                    ┌──────────────────────────┐
                    │   AWS                     │
                    │   S3 (원시 로그)           │
                    │     ↓                    │
                    │   추론 서비스              │
                    │   (Lambda / ECS 확장 가능) │
                    │     ↓                    │
                    │   SES (알림)              │
                    └────────┬─────────────────┘
                             │  ③ 결과 제공
                             ▼
                    ┌──────────────────────────┐
                    │  실무자 대시보드            │
                    └──────────────────────────┘
```

### 변수별 취득 경로 (조사 필요)

| 변수 | 취득 방식(추정) | 취득 주기 | 확인 여부 |
|---|---|---|---|
| `uv_time` | 레시피 파라미터 | 레시피 변경 시 | ☐ |
| `uv_intensity` | UV 램프 컨트롤러 | 실시간 | ☐ |
| `pin_speed` | 이젝터 서보 설정값 | 레시피 변경 시 | ☐ |
| `pin_pressure` | 이젝터 압력 센서 | 실시간 | ☐ |
| `pin_height` | 이젝터 위치 설정값 | 레시피 변경 시 | ☐ |
| `head_vacuum` | 진공 압력 센서 | 실시간 | ☐ |
| `pin_vacuum` | 진공 압력 센서 | 실시간 | ☐ |
| `temperature` | 챔버 온도 센서 | 실시간 | ☐ |
| `humidity` | 클린룸 환경 센서 | 실시간 | ☐ |
| `vacuum_status` | 공용 진공 라인 센서 | 실시간 | ☐ |
| `runtime_hours` | 설비 가동 로그 | 누적 | ☐ |
| `tape_type` | 자재 마스터 / 로트 정보 | 로트 단위 | ☐ |
| `die_thickness` | 제품 사양 / 로트 정보 | 로트 단위 | ☐ |
| `pickup_success` | 설비 픽업 결과 카운터 | 다이 단위 | ☐ |

> **주의**: 변수마다 취득 주기가 다르다. 레시피 파라미터는 로트 단위 고정,
> 센서값은 실시간이다. 이를 **다이 단위로 조인**하는 것이 실제 배포의
> 첫 번째 기술 과제다.

### 현재와 목표 사이의 간극

| # | 간극 | 영향 | 해소 방법 |
|---|---|---|---|
| 1 | 실시간 수집 미구현 | 수동 입력만 가능 | Edge Collector 개발 (PoC 단계) |
| 2 | 설비 인터페이스 미확인 | 벤더별로 다를 수 있음 | 도입 기업 설비 사양 확인 필요 |
| 3 | 다이 단위 라벨 취득 방법 미확인 | 학습 데이터 구성 불확실 | 설비 카운터 로그 확인 필요 |
| 4 | 설비별 편차 미반영 | 단일 모델 가정 | 설비 ID를 변수로 추가 |
| 5 | 모델 재학습 주기 미정 | 드리프트 대응 없음 | 주기적 재학습 파이프라인 |

---

## 12. 알림 설계

**현재**: SMTP 기반 이메일, 기본 dry-run.

| 설계 요소 | 내용 |
|---|---|
| 쿨다운 | 같은 다이에 반복 발송 방지 (기본 300초) |
| 행동 가능한 본문 | 무엇이 문제 / 왜 / 무엇을 하면 되는지 3요소 |
| 한계 고지 | 메일 하단에 SHAP 한계·합성 데이터 사실 명시 |
| DB 이력 저장 | `alerts` 테이블에 발송/dry-run 이력 영속 저장 |

**향후**: Slack/SMS 채널 확장, 등급별 수신자 분리 (critical → 반장, warning → 로그만).

---

## 13. 검수 체크리스트 (B 담당)

- [x] 3-Tier 다이어그램, 각 계층 역할, 통신 방식이 이 문서에 적혀 있다
- [x] MariaDB 설치·실행·DB/테이블 생성 검증 완료
- [x] `api/db/connection.py` 연결 성공 검증 완료
- [x] `api/db/queries.py` 3개 함수 E2E 테스트 PASS
- [x] systemd 서비스 구성 및 자동 시작 검증 완료
- [x] 보안 정책 (비밀번호 관리, 포트 정책) 문서화
- [ ] 변수 취득 경로 표를 최소 절반 이상 채웠다 (현장 연동은 PoC 이후)
- [ ] `QA_DEFENSE.md`에 최소 20개 질문·답변 완성
