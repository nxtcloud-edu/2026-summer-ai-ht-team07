"""경로·설정·산출물 저장 유틸리티.

모든 모듈이 경로를 하드코딩하지 않고 여기를 거친다. Streamlit 은 실행 위치가
프로젝트 루트가 아닐 수 있어서, 리포지토리 루트를 파일 기준으로 역추적한다.

소유자: E(UI·통합).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
"""리포지토리 루트. ``src/yeda/io_utils.py`` 기준 두 단계 위."""

CONFIG_DIR: Path = ROOT / "configs"
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"
ARTIFACT_DIR: Path = ROOT / "artifacts"
MODEL_DIR: Path = ARTIFACT_DIR / "models"
METRIC_DIR: Path = ARTIFACT_DIR / "metrics"
FIGURE_DIR: Path = ARTIFACT_DIR / "figures"


def load_config(name: str) -> dict[str, Any]:
    """``configs/`` 아래 YAML 설정을 읽는다.

    Args:
        name: 파일명. 확장자는 생략 가능 (``"data_gen"`` → ``configs/data_gen.yaml``).

    Returns:
        파싱된 dict.

    Raises:
        FileNotFoundError: 설정 파일이 없을 때. 메시지에 실제 경로를 담는다.
    """
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"설정 파일 없음: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """상대 경로를 리포지토리 루트 기준 절대 경로로 바꾼다."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def ensure_dirs() -> None:
    """산출물 디렉토리를 미리 만든다. 이미 있으면 아무 일도 하지 않는다."""
    for d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODEL_DIR, METRIC_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> Path:
    """dict/list 를 UTF-8 JSON 으로 저장한다 (한글 그대로, 들여쓰기 2).

    Args:
        obj: 직렬화할 객체. numpy 스칼라는 미리 ``float()`` 로 변환할 것.
        path: 저장 경로 (상대 경로 허용).

    Returns:
        실제로 쓴 절대 경로.
    """
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)
    return target


def load_json(path: str | Path) -> Any:
    """UTF-8 JSON 을 읽는다."""
    with resolve(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
