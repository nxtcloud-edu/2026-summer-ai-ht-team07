"""YIELD-X API : DB 쿼리 함수 (MariaDB / MySQL).

담당: B (DB · 인프라 · Q&A 방어)

역할
    이 모듈의 책임은 "predictions / alerts 테이블에 대한 쿼리 3개"뿐이다.
    연결 풀 관리는 ``api/db/connection.py`` 의 ``engine`` 을 그대로 재사용하고,
    이 파일에서는 새로운 engine 이나 커넥션 설정을 만들지 않는다.
    ORM 모델 선언도 하지 않고, ``sqlalchemy.text`` 로 작성한 raw SQL +
    parameter binding 만 사용한다.

스키마 대응(``scripts/init_db.sql`` 기준, DB 이름 ``yeda``)
    predictions(id, created_at, input_json, probability, risk_level, die_id, model_name)
    alerts(id, created_at, prediction_id, subject, body, recipients, sent, dry_run, error)

.. note::
   내부 DB 이름은 ``yeda`` 이다(``ROLES.md`` / ``spec.md`` 계약).
   외부 서비스명 YIELD-X 와 내부 DB 이름 ``yeda`` 를 혼동하지 않는다.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from .connection import engine


def _dump_json(value: Any) -> str:
    """dict/list 를 MySQL/MariaDB JSON 컬럼에 안전하게 저장할 문자열로 직렬화한다.

    PyMySQL 은 Python dict/list 를 JSON 컬럼에 그대로 바인딩해주지 않으므로,
    여기서 명시적으로 ``json.dumps`` 를 호출해 문자열로 만들어 전달한다.
    MariaDB/MySQL 의 JSON 컬럼은 문자열 입력을 파싱해 저장한다.

    Args:
        value: 저장할 dict 또는 list. 이미 문자열이면 그대로 통과시킨다
            (호출자가 이미 직렬화해 넘긴 경우를 대비).

    Returns:
        JSON 문자열.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _load_json(value: Any) -> Any:
    """JSON 컬럼 조회 결과를 dict/list 로 복원한다.

    드라이버/버전에 따라 JSON 컬럼이 이미 dict/list 로 디코딩되어 오는 경우와
    문자열로 오는 경우가 둘 다 있을 수 있어, 두 상황을 모두 안전하게 처리한다.

    Args:
        value: DB 에서 읽은 JSON 컬럼 값 (str 또는 이미 dict/list).

    Returns:
        dict 또는 list. 파싱할 수 없는 문자열이면 원본 값을 그대로 반환한다.
    """
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def save_prediction(
    input_json: dict,
    probability: float,
    risk_level: str,
    die_id: str | None = None,
    model_name: str | None = None,
) -> int:
    """예측 요청/결과를 ``predictions`` 테이블에 한 행 저장한다.

    D 의 기존 호출부가 앞의 세 인자(``input_json``, ``probability``,
    ``risk_level``)만 넘기는 버전이므로, ``die_id`` / ``model_name`` 은
    생략 가능한 keyword 인자로 두어 하위 호환을 유지한다.

    Args:
        input_json: 예측 요청 원본 입력(13개 feature 등). dict 로 전달하면
            내부에서 JSON 문자열로 직렬화해 저장한다.
        probability: 픽업 성공 확률 [0, 1].
        risk_level: 위험 등급. ``"critical" | "warning" | "normal"`` 중 하나
            (``predictions.risk_level`` 컬럼이 ENUM 으로 강제하므로, 여기서는
            추가 검증 없이 그대로 전달한다).
        die_id: 다이 식별자(선택). 없으면 NULL 로 저장된다.
        model_name: 예측에 사용된 모델 이름/버전(선택). 없으면 NULL 로 저장된다.

    Returns:
        int: 새로 생성된 ``predictions.id``.
    """
    stmt = text(
        """
        INSERT INTO predictions (input_json, probability, risk_level, die_id, model_name)
        VALUES (:input_json, :probability, :risk_level, :die_id, :model_name)
        """
    )
    with engine.begin() as conn:
        result = conn.execute(
            stmt,
            {
                "input_json": _dump_json(input_json),
                "probability": probability,
                "risk_level": risk_level,
                "die_id": die_id,
                "model_name": model_name,
            },
        )
        return int(result.lastrowid)


def save_alert(
    prediction_id: int | None,
    subject: str,
    body: str,
    recipients: list[str],
    sent: bool,
    dry_run: bool,
    error: str | None = None,
) -> int:
    """경보(알림) 발송/미리보기 이력을 ``alerts`` 테이블에 한 행 저장한다.

    Args:
        prediction_id: 이 경보를 유발한 ``predictions.id`` (선택). 없으면 NULL.
        subject: 경보 이메일 제목.
        body: 경보 이메일 본문.
        recipients: 수신자 이메일 목록. 내부에서 JSON 문자열로 직렬화해 저장한다.
        sent: 실제 발송 성공 여부.
        dry_run: dry-run 여부(True 면 실제 발송 없이 미리보기만 수행됨).
        error: 발송 실패 시 오류 메시지(선택). 성공 시 None.

    Returns:
        int: 새로 생성된 ``alerts.id``.
    """
    stmt = text(
        """
        INSERT INTO alerts (prediction_id, subject, body, recipients, sent, dry_run, error)
        VALUES (:prediction_id, :subject, :body, :recipients, :sent, :dry_run, :error)
        """
    )
    with engine.begin() as conn:
        result = conn.execute(
            stmt,
            {
                "prediction_id": prediction_id,
                "subject": subject,
                "body": body,
                "recipients": _dump_json(recipients),
                "sent": sent,
                "dry_run": dry_run,
                "error": error,
            },
        )
        return int(result.lastrowid)


def get_history(limit: int = 20) -> list[dict]:
    """``predictions`` 테이블에서 최신 예측 이력을 조회한다.

    정렬은 ``created_at DESC`` 를 우선하고, 같은 시각에 여러 행이 저장된
    경우를 대비해 ``id DESC`` 를 보조 정렬 기준으로 사용한다.

    Args:
        limit: 반환할 최대 행 수 (기본 20). SQL 문자열에 직접 넣지 않고
            parameter binding 으로 전달한다.

    Returns:
        list[dict]: 각 dict 는 다음 키만 가진다 — ``id``, ``created_at``,
        ``input_json``, ``probability``, ``risk_level``.
        ``die_id`` / ``model_name`` 은 DB 에는 존재하지만 현재 history 응답
        계약에는 포함하지 않는다.
    """
    stmt = text(
        """
        SELECT id, created_at, input_json, probability, risk_level
        FROM predictions
        ORDER BY created_at DESC, id DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"limit": limit}).mappings().all()

    return [
        {
            "id": int(row["id"]),
            "created_at": row["created_at"],
            "input_json": _load_json(row["input_json"]),
            "probability": float(row["probability"]),
            "risk_level": row["risk_level"],
        }
        for row in rows
    ]
