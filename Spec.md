# Spec.md — E역할: 프론트엔드 · 통합 · 데모

> **담당**: E (인공지능·소프트웨어학과)
> **한 줄 요약**: 정적 프론트엔드, FastAPI 앱 초기화, 안정화
> **브랜치**: `feat/frontend-E`

---

## 1. 역할 개요

E는 사용자가 직접 보는 **프론트엔드 UI**와, 백엔드의 **진입점(main.py)**을 소유한다.
D가 만든 라우터를 등록하고, B의 배포 스크립트를 Makefile에 연결하며,
공용 파일(`schema.py`, `requirements.txt`)의 변경을 머지·공지하는 통합 책임도 갖는다.

---

## 2. 소유 파일

| 파일 | 설명 |
|---|---|
| `frontend/index.html` | 메인 페이지 (정적 HTML) |
| `frontend/js/*.js` | API 호출, 탭 전환, 렌더링 로직 |
| `frontend/css/*.css` | 스타일시트 |
| `api/main.py` | FastAPI 앱 엔트리포인트 (CORS, 라우터 등록) |
| `src/yeda/alerts/email.py` | 이메일 알림 모듈 |
| `src/yeda/schema.py` | 공용 계약 (변경 시 전원 공지) |
| `src/yeda/io_utils.py` | 경로·설정·저장 유틸리티 |
| `src/yeda/text_utils.py` | 한국어 조사·포맷 유틸 |
| `configs/app.yaml` | UI·알림 설정 |
| `.env.example` | 환경변수 템플릿 |
| `Makefile` | 실행 단축 명령 |
| `scripts/run_demo.sh` | 데모 실행 스크립트 |
| `requirements.txt` | 의존성 목록 (변경 시 전원 공지) |

---

## 3. 완료 정의 (Definition of Done)

- [ ] `make api` → uvicorn 서버가 포트 8000에서 정상 기동
- [ ] `make frontend` → 프론트엔드가 로컬에서 서빙됨
- [ ] 프론트엔드 4개 탭(예측/원인/가이드/알림)이 API를 호출하여 동작함
- [ ] 백엔드가 죽어도 프론트엔드 페이지가 로드됨 (에러 메시지 표시)
- [ ] 데모 프리셋 버튼으로 원클릭 입력 가능
- [ ] `api/main.py`에 CORS 설정으로 S3 도메인 허용
- [ ] `.env.example`에 DB 접속 정보(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME) 추가
- [ ] `requirements.txt`에 `fastapi`, `uvicorn[standard]`, `pymysql`, `sqlalchemy` 추가
- [ ] **마지막 3시간**: 기능 추가 중단, 리허설·안정화만 수행

---

## 4. 구현 태스크

### Phase 0 — 즉시 착수 (킥오프 ~ 1h)

| # | 태스크 | 산출물 | 비고 |
|---|---|---|---|
| 0-1 | `api/main.py` 초기화 | FastAPI 인스턴스 + CORS 미들웨어 + `/api/health` 엔드포인트 | D가 라우터 등록할 기반 |
| 0-2 | `frontend/` 디렉토리 구조 생성 | `index.html`, `js/app.js`, `css/style.css` | 빈 탭 4개 구조 |
| 0-3 | Makefile 업데이트 | `make api`, `make frontend` 타겟 추가 | |
| 0-4 | `.env.example` 업데이트 | DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME 추가 | |
| 0-5 | `requirements.txt` 업데이트 | fastapi, uvicorn, pymysql, sqlalchemy 추가 | 전원 공지 |

### Phase 1 — 프론트엔드 기본 구현 (1h ~ 4h)

| # | 태스크 | 산출물 | 비고 |
|---|---|---|---|
| 1-1 | 탭 네비게이션 구현 | 예측/원인/가이드/알림 4개 탭 전환 | JS로 SPA 방식 |
| 1-2 | 예측 탭 UI | 13개 변수 입력 폼 + 프리셋 버튼 + 결과 표시 영역 | `configs/app.yaml`의 `demo_presets` 활용 |
| 1-3 | 원인 탭 UI | SHAP 기여도 바 차트 렌더링 | `/api/explain` 응답 시각화 |
| 1-4 | 가이드 탭 UI | 최적화 제안 테이블 | `/api/optimize` 응답 표시 |
| 1-5 | 알림 탭 UI | 알림 발송(dry-run) 결과 표시 | `/api/alert` 응답 표시 |
| 1-6 | API 호출 모듈 작성 | `frontend/js/api.js` — fetch 래퍼 + 에러 핸들링 | 백엔드 다운 시 에러 메시지 |

### Phase 2 — 백엔드 통합 (4h ~ 8h)

| # | 태스크 | 산출물 | 비고 |
|---|---|---|---|
| 2-1 | D의 라우터 등록 | `api/main.py`에 predict, explain, optimize, alert, history 라우터 import | D와 협의 |
| 2-2 | 정적 파일 서빙 설정 | FastAPI `StaticFiles` 마운트 또는 별도 서빙 | 로컬 개발용 |
| 2-3 | 에러 핸들링 미들웨어 | 전역 예외 → JSON 에러 응답 | 데모 중 500 노출 방지 |
| 2-4 | B의 배포 스크립트 연결 | Makefile에 `make deploy` 타겟 추가 | B와 협의 |

### Phase 3 — 안정화 · 데모 (8h ~ 마감)

| # | 태스크 | 산출물 | 비고 |
|---|---|---|---|
| 3-1 | 엔드투엔드 데모 시나리오 검증 | 3개 프리셋 전부 정상 동작 확인 | 정상/위험(얇은다이)/위험(마모) |
| 3-2 | UI 반응형 보정 | 발표 해상도(1920x1080)에서 레이아웃 확인 | |
| 3-3 | 에러 케이스 점검 | 백엔드 다운, 모델 미로드, 잘못된 입력값 처리 | graceful degradation |
| 3-4 | `scripts/run_demo.sh` 업데이트 | 3-Tier 버전에 맞게 실행 흐름 수정 | |
| 3-5 | 최종 리허설 | 발표 시나리오 1회 완주 | 마지막 3시간 이내 |

---

## 5. 인터페이스 접점

| 방향 | 상대 | 내용 |
|---|---|---|
| ← D | `api/routers/`를 `api/main.py`에서 import·등록 | D가 라우터 완성 후 E가 등록 |
| ← B | `scripts/deploy.sh`를 Makefile에 연결 | B가 스크립트 완성 후 E가 타겟 추가 |
| → 전원 | `schema.py`, `requirements.txt` 변경 시 머지·공지 | E가 게이트키퍼 |
| ← C | 모델 상태(`is_loaded`, `is_mock`) 참조 | health 엔드포인트에 반영 |
| → D | API 응답 형식을 프론트엔드에서 사용 | 형식 합의 필수 (M0) |

---

## 6. 기술 스택

| 영역 | 기술 | 비고 |
|---|---|---|
| 프론트엔드 | HTML5 + Vanilla JS + CSS3 | 정적 파일, S3 호스팅 가능 |
| 백엔드 프레임워크 | FastAPI | 비동기, 자동 OpenAPI 문서 |
| ASGI 서버 | uvicorn | `make api`로 기동 |
| CORS | `fastapi.middleware.cors` | localhost + S3 도메인 허용 |
| 차트 | Chart.js 또는 D3.js (CDN) | SHAP 바 차트 렌더링 |
| HTTP 클라이언트 | Fetch API | 프론트엔드 → 백엔드 |

---

## 7. API 엔드포인트 (E가 호출하는 것)

| 메서드 | 경로 | 설명 | 담당 |
|---|---|---|---|
| GET | `/api/health` | 서버 상태 + 모델 로드 여부 | **E** (직접 구현) |
| GET | `/api/presets` | 데모 프리셋 목록 | D |
| POST | `/api/predict` | 13개 피처 → 확률 + risk_level | D |
| POST | `/api/explain` | SHAP 기여도 분해 | D |
| POST | `/api/optimize` | 최적화 제안 | D |
| POST | `/api/alert` | 알림 발송 (dry-run) | D |
| GET | `/api/history` | 예측 이력 조회 | D |

---

## 8. 디렉토리 구조 (생성 목표)

```
frontend/
├── index.html          # 메인 SPA 페이지
├── css/
│   └── style.css       # 전체 스타일
└── js/
    ├── app.js          # 탭 전환, 초기화
    ├── api.js          # fetch 래퍼, 에러 핸들링
    ├── predict.js      # 예측 탭 로직
    ├── explain.js      # 원인 탭 로직 (차트)
    ├── optimize.js     # 가이드 탭 로직
    └── alert.js        # 알림 탭 로직

api/
├── main.py             # FastAPI 앱 초기화, CORS, 라우터 등록
├── routers/            # (D 소유)
│   ├── predict.py
│   ├── optimize.py
│   ├── alert.py
│   └── history.py
├── models/             # (D 소유)
│   ├── request.py
│   └── response.py
├── services/           # (C 소유)
│   └── ml_service.py
└── db/                 # (B 소유)
    ├── connection.py
    └── queries.py
```

---

## 9. `api/main.py` 설계

```python
"""FastAPI 앱 엔트리포인트. 소유자: E."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="YEDA API", version="1.0.0")

# CORS — 로컬 개발 + S3 배포 도메인
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],  # 배포 시 S3 URL 추가
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (D가 완성하면 주석 해제)
# from api.routers import predict, optimize, alert, history
# app.include_router(predict.router, prefix="/api")
# app.include_router(optimize.router, prefix="/api")
# app.include_router(alert.router, prefix="/api")
# app.include_router(history.router, prefix="/api")

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": False, "mock_mode": True}
```

---

## 10. Makefile 추가 타겟

```makefile
api:  ## FastAPI 백엔드 서버 실행 (포트 8000)
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:  ## 프론트엔드 정적 파일 서빙 (포트 3000)
	$(PYTHON) -m http.server 3000 --directory frontend

deploy:  ## S3 + EC2 배포 (B의 스크립트 호출)
	bash scripts/deploy.sh
```

---

## 11. 리스크 & 대응

| 리스크 | 징후 | 대응 |
|---|---|---|
| S3 배포 권한 없음 | aws cli 에러 | 로컬 `python -m http.server`로 3-Tier 시연 |
| 프론트-백 JSON 불일치 | 화면에 데이터 안 뜸 | D와 request/response 스키마 재확인 |
| 백엔드 기동 실패 | import 에러 | mock 모드로 폴백, 모든 라우터 옵셔널 import |
| CORS 에러 | 브라우저 콘솔 에러 | allow_origins 와일드카드 확인 |

---

## 12. 범위 축소 시 보존 우선순위

**절대 버리지 않는 것**:
1. 3-Tier 분리 동작 (프론트 ↔ API ↔ DB)
2. `make api` + 프론트엔드에서 예측이 뜨는 것
3. 프리셋 버튼 동작

**시간 부족 시 버리는 것** (우선순위 순):
1. 예측 이력 UI 탭
2. 프론트엔드 디자인 고도화
3. S3 실배포 (로컬 static 서빙으로 대체)
4. 이메일 실제 발송 (dry-run으로 충분)

---

## 13. 타임라인 (24시간 기준)

| 시간 | 마일스톤 | 핵심 산출물 |
|---|---|---|
| 0~1h | M0: 착수 | `api/main.py` + `frontend/` 구조 + Makefile |
| 1~4h | M1: 프론트 기본 | 4개 탭 UI 완성 (mock 데이터로 동작) |
| 4~8h | M2: 통합 시작 | D의 라우터 등록, 실제 API 연동 |
| 8~16h | M3: 기능 완성 | 모든 탭이 실 API와 동작 |
| 16~21h | M4: 안정화 | 에러 처리, 엣지 케이스 |
| 21~24h | M5: 리허설 | 데모 시나리오 검증, 최종 점검 |

---

## 14. 즉시 착수 체크리스트

킥오프 직후 아래를 순서대로 실행:

1. `feat/frontend-E` 브랜치 생성
2. `api/main.py` 작성 (CORS + health)
3. `frontend/` 디렉토리 생성 + `index.html` 뼈대
4. Makefile에 `api`, `frontend` 타겟 추가
5. `.env.example`에 DB 변수 추가
6. `requirements.txt`에 FastAPI 관련 패키지 추가 → **전원 공지**
7. `make api` 로 서버 기동 확인
8. 커밋 & 푸시

---

> **핵심 원칙**: mock 응답으로 킥오프 직후부터 화면을 만든다. 의존성이 준비되면 실 API로 전환.
