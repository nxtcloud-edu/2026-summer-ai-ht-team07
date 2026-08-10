# ARCHITECTURE.md — 3-Tier 시스템 아키텍처 및 설비 연동 설계

> **소유자: B(전기정보공학과)**
>
> 이 문서의 목적은 두 가지다:
> 1. 현재 구현된 **3-Tier 아키텍처**의 구조와 통신 방식을 명확히 한다.
> 2. "지금 데모"와 "실제 배포" 사이의 간극을 **먼저 밝혀** 로드맵으로 만든다.

---

## 1. 3-Tier 아키텍처 (현재 구현)

```
┌─────────────────────┐        HTTP/JSON         ┌─────────────────────────┐        SQL          ┌──────────────────┐
│     Tier 1          │  ─────────────────────▶  │       Tier 2            │  ────────────────▶  │    Tier 3        │
│     Frontend        │  ◀─────────────────────  │       Backend API       │  ◀────────────────  │    Database      │
│                     │                          │                         │                     │                  │
│  S3 Static Hosting  │                          │  EC2 :8000 (FastAPI)    │                     │  MySQL (MariaDB) │
│  HTML / JS / CSS    │                          │  uvicorn                │                     │  yeda DB         │
└─────────────────────┘                          └─────────────────────────┘                     └──────────────────┘
        │                                                   │                                            │
        │  정적 파일 서빙                                     │  비즈니스 로직                                │  영속 저장
        │  서버 독립적                                        │  ML 모델 로드                                │  서버 재시작 시 유지
        │                                                   │  SHAP / 최적화                              │  predictions / alerts
        └───────────────────────────────────────────────────┴────────────────────────────────────────────┘
```

### 1-Tier vs 3-Tier 비교

| 항목 | 1-Tier (기존 Streamlit) | 3-Tier (현재) |
|---|---|---|
| 화면이 있는 곳 | EC2 (Streamlit) | **S3** (정적 호스팅) |
| 로직이 도는 곳 | 같은 EC2 | **EC2 :8000** (FastAPI) |
| 데이터 위치 | 서버 메모리 | **MySQL** (MariaDB) |
| 서버 재시작 | 데이터 사라짐 | **유지됨** |
| 서버가 멈추면 | 화면도 안 뜸 | **화면은 뜸** |
| DB 비밀번호 | 해당 없음 | **EC2 안에만** (.env) |

### 왜 3-Tier인가

1. **프론트-백 분리**: 백엔드 장애 시에도 프론트엔드는 살아있어 "서버 점검 중" 안내 가능
2. **데이터 영속성**: 예측 이력/알림 이력이 서버 재시작에 무관하게 보존
3. **보안**: DB 접속 정보가 백엔드 인스턴스 내부에만 존재, 프론트엔드에 노출 안 됨
4. **확장성**: 프론트/백/DB를 독립적으로 스케일 가능

---

## 2. API 엔드포인트 (Tier 2 상세)

| Method | Path | 설명 | 담당 |
|---|---|---|---|
| GET | `/api/health` | 헬스체크 (모델 로드 상태) | D |
| GET | `/api/presets` | 데모 프리셋 목록 | D |
| POST | `/api/predict` | 13개 변수 → 성공 확률 + 위험 등급 | D |
| POST | `/api/explain` | SHAP 기여도 분해 (%p 단위) | D |
| POST | `/api/optimize` | 조정 가능 변수 최적화 제안 | D |
| POST | `/api/alert` | 알림 트리거 (dry-run 기본) | D |
| GET | `/api/history` | 예측 이력 조회 (MySQL) | D+B |

### 요청/응답 흐름

```
[Frontend]                      [FastAPI :8000]                    [MySQL]
    │                                │                                │
    │── POST /api/predict ──────────▶│                                │
    │                                │── predict_proba() ───────────▶ │ (모델 추론)
    │                                │                                │
    │                                │── save_prediction() ─────────▶ │ (DB INSERT)
    │                                │◀─ prediction_id ──────────────│
    │◀─ {probability, risk_level} ──│                                │
    │                                │                                │
    │── GET /api/history ───────────▶│                                │
    │                                │── get_history(limit) ────────▶ │ (DB SELECT)
    │                                │◀─ [rows] ────────────────────│
    │◀─ [{id, created_at, ...}] ───│                                │
```

---

## 3. 데이터베이스 스키마 (Tier 3 상세)

```sql
-- scripts/init_db.sql 기준

CREATE DATABASE IF NOT EXISTS yeda;

CREATE TABLE predictions (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_json      JSON NOT NULL,           -- 13개 feature dict
    probability     DECIMAL(6,5) NOT NULL,   -- [0, 1]
    risk_level      ENUM('critical','warning','normal') NOT NULL,
    die_id          VARCHAR(32) NULL,
    model_name      VARCHAR(100) NULL
);

CREATE TABLE alerts (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    prediction_id   BIGINT UNSIGNED NULL,    -- FK → predictions(id)
    subject         VARCHAR(255) NOT NULL,
    body            TEXT NOT NULL,
    recipients      JSON NOT NULL,
    sent            BOOLEAN DEFAULT FALSE,
    dry_run         BOOLEAN DEFAULT TRUE,
    error           TEXT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
);
```

### DB 접속 구성

- 접속 정보: `.env`의 `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- 드라이버: SQLAlchemy + PyMySQL (`mysql+pymysql://`)
- 연결 풀: `pool_pre_ping=True`, `pool_recycle=1800`
- 코드: `api/db/connection.py` (engine) + `api/db/queries.py` (쿼리 3개)

---

## 4. ML 파이프라인 (모델 계층)

```
configs/data_gen.yaml ──→ scripts/make_data.py ──→ data/raw/yeda_synthetic.csv
                                                         │
configs/model.yaml ─────→ scripts/train.py ──────→ artifacts/models/primary_model.joblib
                                                         │
                                                         ▼
                                           api/services/ml_service.py (자동 로드)
                                                  │
                   ┌──────────────┬────────────────┼──────────────────┐
                   ▼              ▼                ▼                  ▼
            predict_proba   shap_explainer    search.py          email.py
            (수율 예측)     (SHAP %p 분해)   (조건 탐색)         (알림)
```

모델이 없으면 mock 모드로 자동 폴백 — 데모가 멈추는 일 없음.

---

## 5. 배포 구성 (AWS)

```
┌─────────────────────────────────────────────────────────┐
│  AWS                                                     │
│                                                         │
│  ┌──────────┐     ┌───────────────────────────┐         │
│  │   S3     │     │   EC2 (Amazon Linux)      │         │
│  │          │     │                           │         │
│  │  index.  │     │  FastAPI :8000 (systemd)  │         │
│  │  html    │     │  .venv + requirements.txt │         │
│  │  js/css  │     │  .env (DB_USER/PASSWORD)  │         │
│  │          │     │                           │         │
│  └──────────┘     │  MariaDB :3306 (local)    │         │
│                   │  yeda DB                  │         │
│                   └───────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

- S3: 정적 파일 호스팅 (CloudFront 옵션)
- EC2: FastAPI + MariaDB 동일 인스턴스 (해커톤 규모)
- 프로덕션 확장 시: RDS 분리, ALB 추가, Auto Scaling

---

## 6. 변수별 취득 경로 (실제 배포 시)

| 변수 | 취득 방식(추정) | 취득 주기 |
|---|---|---|
| `uv_time` | 레시피 파라미터 (설비 컨트롤러) | 레시피 변경 시 |
| `uv_intensity` | UV 램프 컨트롤러 출력값 | 실시간 |
| `pin_speed` | 이젝터 서보 설정값 | 레시피 변경 시 |
| `pin_pressure` | 이젝터 압력 센서 (PLC 태그) | 실시간 |
| `pin_height` | 이젝터 위치 설정값 | 레시피 변경 시 |
| `head_vacuum` | 진공 압력 센서 (PLC 태그) | 실시간 |
| `pin_vacuum` | 진공 압력 센서 (PLC 태그) | 실시간 |
| `temperature` | 챔버 온도 센서 | 실시간 |
| `humidity` | 클린룸 환경 센서 | 실시간 |
| `vacuum_status` | 공용 진공 라인 센서 | 실시간 |
| `runtime_hours` | 설비 가동 로그 누적 | 누적 |
| `tape_type` | 자재 마스터 / MES 로트 정보 | 로트 단위 |
| `die_thickness` | 제품 사양 / MES 로트 정보 | 로트 단위 |
| `pickup_success` | 설비 픽업 결과 카운터 | 다이 단위 |

> 변수마다 취득 주기가 다르다. 레시피 파라미터는 로트 단위로 고정이고,
> 센서값은 실시간이다. 이를 **다이 단위로 조인**하는 것이 실제 배포의 첫 번째 기술 과제다.

---

## 7. 현재와 목표 사이의 간극

| # | 간극 | 영향 | 해소 방법 |
|---|---|---|---|
| 1 | 실시간 수집 미구현 | 수동 입력만 가능 | Edge Collector 개발 (PoC 단계) |
| 2 | 설비 인터페이스 미확인 | 벤더별 프로토콜 상이 | 도입 기업 설비 사양 확인 필요 |
| 3 | 다이 단위 라벨 취득 방법 미확인 | 학습 데이터 구성 불확실 | 설비 카운터 로그 확인 필요 |
| 4 | 설비별 편차 미반영 | 단일 모델 가정 | 설비 ID를 변수로 추가 |
| 5 | 모델 재학습 주기 미정 | 드리프트 대응 없음 | 주기적 재학습 파이프라인 |

---

## 8. 알림 설계

**현재**: SMTP 기반 이메일, 기본 dry-run. alerts 테이블에 이력 저장.

**설계상 고려한 것**:
- **쿨다운** (300초) — 같은 다이에 반복 발송 방지
- **행동 가능한 본문** — 무엇이 문제인지 / 왜 그런지 / 무엇을 하면 되는지
- **한계 고지** — SHAP 한계와 합성 데이터 사실을 메일 하단에 명시

**향후**: Slack / SMS 채널 확장, 등급별 수신자 분리.

---

## 9. 검수 체크리스트

- [x] 3-Tier 다이어그램이 문서에 있다
- [x] 각 Tier의 역할과 통신 방식이 명시되어 있다
- [x] DB 스키마(predictions/alerts)가 문서화되어 있다
- [x] API 엔드포인트 목록과 요청/응답 흐름이 있다
- [x] 변수 취득 경로 표가 있다
- [x] 현재와 목표의 간극을 발표에서 설명할 수 있다
- [ ] 알림 메일 본문을 실무자 관점에서 검수했다
