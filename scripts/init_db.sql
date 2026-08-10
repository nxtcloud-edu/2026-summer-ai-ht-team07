-- =============================================================================
-- YIELD-X : DB 초기화 스크립트 (MySQL / MariaDB)
--
-- 담당: B (DB / 인프라 / Q&A 방어)
-- 목적: predictions / alerts 테이블 생성.
--
-- 외부 서비스명은 YIELD-X 로 변경되었고 내부 Python package 명은 여전히
-- `yeda` (src/yeda) 이지만, 내부 DB 이름은 ROLES.md 와 spec.md 계약을 그대로
-- 따라 `yeda` 를 사용한다.
--
-- 참고(Single source of truth):
--   - DB 이름 `yeda`, 예측 이력/알림 이력 테이블 요건 : ROLES.md (B DoD)
--   - predictions/alerts 컬럼 설계 예시(input_json/probability/risk_level) :
--     .kiro/specs/hackathon-pipeline/spec.md :: "MySQL 스키마 설계"
--   - risk_level 허용값                       : src/yeda/schema.py :: RISK_LEVELS
--   - DB 함수 인터페이스(save_prediction/save_alert/get_history) :
--     ROLES.md (B DoD) — 13개 feature 는 API 요청/응답 계층(D 소유)에서
--     Pydantic 으로 검증되고, DB 계층은 원본 입력을 `input_json` 하나로 보관한다.
--
-- 안전 원칙:
--   - DROP TABLE / DROP DATABASE 등 기존 데이터를 파괴하는 명령을 포함하지 않는다.
--   - CREATE DATABASE / CREATE TABLE 모두 IF NOT EXISTS 를 사용해 반복 실행에 안전하다.
--   - InnoDB 엔진을 사용해 alerts -> predictions 외래키 참조를 보장한다.
-- =============================================================================

CREATE DATABASE IF NOT EXISTS yeda
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE yeda;

-- -----------------------------------------------------------------------------
-- predictions : 예측 요청 원본(input_json) + 결과(probability/risk_level) 이력
--
-- 13개 feature 는 더 이상 개별 컬럼으로 저장하지 않는다. API 요청/응답 스키마
-- (D 소유, api/models/request.py)가 이미 13개 feature 를 검증하므로, DB 계층은
-- 검증된 원본 입력 전체를 `input_json` 하나로 보관해 스키마 변경(feature 추가/삭제)에
-- 유연하게 대응한다. get_history() 조회는 created_at 정렬을 기준으로 한다.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    created_at      DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP
        COMMENT '예측 저장 시각. get_history(limit=20) 정렬/필터 기준',

    input_json      JSON             NOT NULL
        COMMENT '예측 요청 원본 입력(13개 feature 등). API 요청 스키마(D 소유)가 이미 검증한 값을 그대로 보관',

    probability     DECIMAL(6,5)     NOT NULL
        COMMENT '픽업 성공 확률 [0,1]. 기존 y_prob 계약을 계승(이름만 probability 로 통일)',

    risk_level      ENUM('critical', 'warning', 'normal') NOT NULL
        COMMENT '위험 등급. src/yeda/schema.py::RISK_LEVELS 3종 고정',

    die_id          VARCHAR(32)      NULL
        COMMENT '다이 식별자(선택). 생성기 포맷은 "D000123" 이나, 없을 수 있어 nullable',

    model_name      VARCHAR(100)     NULL
        COMMENT '예측에 사용된 모델 이름/버전 식별자(선택)',

    PRIMARY KEY (id),
    KEY idx_predictions_created_at (created_at)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='YIELD-X 예측 요청/결과 이력';

-- -----------------------------------------------------------------------------
-- alerts : predictions 에서 파생된 경보(알림) 이력
--
-- prediction_id 는 predictions.id 를 참조하는 nullable FK 이다. 예측과 무관하게
-- 생성되는 경보(예: 수동 테스트 알림)도 저장할 수 있도록 nullable 로 둔다.
-- 참조된 예측이 삭제되어도 경보 이력 자체는 남아야 하므로 ON DELETE SET NULL.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    created_at      DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP
        COMMENT '경보 생성 시각',

    prediction_id   BIGINT UNSIGNED  NULL
        COMMENT '경보를 유발한 예측 레코드(선택). predictions.id 참조(FK)',

    subject         VARCHAR(255)     NOT NULL
        COMMENT '경보 이메일 제목',

    body            TEXT             NOT NULL
        COMMENT '경보 이메일 본문',

    recipients      JSON             NOT NULL
        COMMENT '수신자 목록. 예: ["a@example.com", "b@example.com"]',

    sent            BOOLEAN          NOT NULL DEFAULT FALSE
        COMMENT '실제 발송 성공 여부',

    dry_run         BOOLEAN          NOT NULL DEFAULT TRUE
        COMMENT 'dry-run 여부. TRUE 면 실제 발송 없이 미리보기만 수행됨',

    error           TEXT             NULL
        COMMENT '발송 실패 시 오류 메시지(선택). 성공 시 NULL',

    PRIMARY KEY (id),
    KEY idx_alerts_created_at (created_at),
    KEY idx_alerts_prediction_id (prediction_id),

    CONSTRAINT fk_alerts_prediction_id
        FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='YIELD-X predictions 기반 경보 이력';
