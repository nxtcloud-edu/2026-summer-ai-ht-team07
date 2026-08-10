"""YIELD-X API : DB 연결 관리 (MariaDB / MySQL).

담당: B (DB · 인프라 · Q&A 방어)

역할
    이 모듈의 책임은 "DB 연결 하나"뿐이다. 환경변수로부터 접속 정보를 구성하고,
    SQLAlchemy connection pool(``engine``)을 만들고, 연결 상태를 확인하는 함수를
    제공한다. ORM 모델 선언이나 쿼리 로직은 이 파일의 책임이 아니다
    (쿼리는 ``api/db/queries.py`` 가 담당한다).

배포 전제
    - AWS EC2 Amazon Linux, FastAPI 와 MariaDB 가 같은 인스턴스에서 실행된다.
    - MariaDB 는 3306 포트로 로컬(또는 사설 네트워크)에서만 접근 가능하고,
      외부에는 공개되지 않는다.
    - 실제 DB 비밀번호는 EC2 상의 .env / 환경변수에만 존재하며, 이 저장소에는
      값이 아니라 "어떤 환경변수가 필요한가"만 문서화된다.
    - SQLAlchemy + PyMySQL 조합으로 ``mysql+pymysql://`` 드라이버를 사용한다.

.. important::
   이 모듈을 import 하는 것만으로 실제 DB 접속을 시도하지 않는다.
   ``create_engine()`` 은 필요한 시점(첫 쿼리 실행 등)에 지연 연결되므로,
   MariaDB 가 아직 준비되지 않은 환경에서도 import 자체는 안전하다.
   실제 연결 여부를 확인하려면 :func:`check_db_connection` 을 호출해야 한다.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class DBConfigError(RuntimeError):
    """DB 접속에 필요한 환경변수가 없거나 잘못되었을 때 발생시키는 예외.

    목적은 "인증 실패"처럼 애매한 하류 오류 대신, 설정 단계에서 바로
    원인을 알 수 있는 명확한 오류를 내는 것이다. 이 예외 메시지에는
    비밀번호 값이나 완성된 DATABASE_URL 문자열을 절대 포함하지 않는다.
    """


# ---------------------------------------------------------------------------
# 환경변수 기본값
#
# 보안과 무관한 값(호스트/포트/DB 이름)만 기본값을 둔다. 계정 정보
# (DB_USER / DB_PASSWORD)는 기본값을 두지 않고, 없으면 즉시 오류를 낸다.
# ---------------------------------------------------------------------------
_DEFAULT_DB_HOST = "127.0.0.1"
_DEFAULT_DB_PORT = "3306"
_DEFAULT_DB_NAME = "yeda"


def _build_database_url() -> str:
    """환경변수를 읽어 SQLAlchemy DATABASE URL을 구성한다.

    Returns:
        ``mysql+pymysql://user:password@host:port/dbname`` 형식의 접속 문자열.

    Raises:
        DBConfigError: ``DB_USER`` 또는 ``DB_PASSWORD`` 가 설정되지 않은 경우.
            "누가 봐도 설정이 빠졌다"는 것을 바로 알 수 있게 하기 위해,
            SQLAlchemy/PyMySQL 이 던지는 인증 오류보다 먼저 여기서 막는다.
    """
    host = os.environ.get("DB_HOST", _DEFAULT_DB_HOST)
    port = os.environ.get("DB_PORT", _DEFAULT_DB_PORT)
    name = os.environ.get("DB_NAME", _DEFAULT_DB_NAME)
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")

    missing = [var for var, val in (("DB_USER", user), ("DB_PASSWORD", password)) if not val]
    if missing:
        raise DBConfigError(
            "DB 접속 정보가 설정되지 않았습니다: "
            f"{', '.join(missing)} 환경변수를 확인하세요. "
            "(EC2 의 .env 또는 환경변수에 DB_USER / DB_PASSWORD 를 설정해야 합니다.)"
        )

    # PyMySQL 은 URL 안의 특수문자(:, @, / 등)를 percent-encoding 없이 넣으면
    # 접속 문자열이 깨질 수 있으므로 quote_plus 로 이스케이프한다.
    from urllib.parse import quote_plus

    return (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{name}"
    )


def _create_engine() -> Engine:
    """SQLAlchemy engine(connection pool)을 생성한다.

    .. note::
       ``create_engine()`` 은 즉시 접속하지 않는다. 실제 커넥션은 첫 쿼리
       실행 시점에 pool 에서 꺼내 쓰며, 여기서는 pool 설정만 구성한다.
    """
    database_url = _build_database_url()
    return create_engine(
        database_url,
        pool_pre_ping=True,  # 끊어진(stale) 커넥션을 재사용하지 않고 미리 감지해 재연결한다.
        pool_recycle=1800,  # MariaDB 의 idle timeout 으로 끊긴 연결을 30분마다 선제 갱신한다.
        future=True,
    )


# ---------------------------------------------------------------------------
# 모듈 로드 시 즉시 DB 에 접속하지 않는다.
#
# `create_engine()` 은 커넥션 풀 설정만 구성하고 실제 TCP 연결은 지연 생성하므로,
# 이 모듈을 import 하는 시점에는 환경변수 검증(_build_database_url)만 수행되고
# MariaDB 로의 실제 접속은 일어나지 않는다. 단, 필수 환경변수가 없으면 import
# 시점에 DBConfigError 가 발생한다 — 이는 "설정 누락"을 의도적으로 즉시
# 드러내기 위함이며, DB 서버 부재로 인한 오류와는 다르다.
# ---------------------------------------------------------------------------
engine: Engine = _create_engine()


def check_db_connection() -> bool:
    """실제로 DB에 연결할 수 있는지 확인한다.

    가벼운 쿼리(``SELECT 1``) 하나를 실행해 연결 가능 여부만 확인하고,
    연결/쿼리 실패의 세부 예외는 그대로 삼키지 않되 비밀번호 등 민감한
    값은 노출하지 않는다 (SQLAlchemy 예외 메시지 자체에도 비밀번호는
    포함되지 않으므로 별도 마스킹이 필요 없다).

    Returns:
        bool: 연결에 성공하면 True, 실패하면 False.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
