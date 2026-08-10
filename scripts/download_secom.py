"""UCI SECOM 다운로드 — `make secom-data` 진입점.

TODO(B): urllib.request 로 두 파일을 data/external/ 에 저장.
         해커톤장 네트워크 실패 시 명확한 안내를 출력하고 종료 코드 1 을 반환할 것.
         이 파트가 실패해도 메인 데모에는 아무 영향이 없어야 한다.

소유자: B(계측·IoT).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yeda.io_utils import EXTERNAL_DIR  # noqa: E402
from yeda.secom.pipeline import SECOM_DATA_URL, SECOM_LABEL_URL  # noqa: E402


def main() -> int:
    """SECOM 원본 2개 파일을 내려받는다."""
    print(f"TODO(B): {SECOM_DATA_URL} / {SECOM_LABEL_URL} → {EXTERNAL_DIR}")
    print("이 파트는 우선순위가 낮습니다. 메인 데모가 완성된 뒤에 착수하세요.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
