"""모듈 간 계약 검사 — 이 테스트가 깨지면 프로젝트의 정당성이 깨진다.

특히 ``test_no_generator_leak`` 은 이 리포지토리에서 **가장 중요한 테스트**다.
데이터 생성 규칙이 최적화·설명 쪽으로 새면 "정답을 넣고 정답을 꺼냈다"가 되고,
그 지적에는 방어할 방법이 없다. 사람이 기억으로 지키는 대신 CI가 강제한다.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from yeda.schema import (
    ADJUSTABLE,
    BOUNDS,
    FEATURE_NAMES,
    FIXED,
    MONOTONE_CONSTRAINTS,
    SPEC_BY_NAME,
    clip_to_bounds,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "yeda"

FORBIDDEN_IMPORTERS = ["models", "explain", "optimize", "secom"]
"""생성기를 import 하면 안 되는 패키지들."""

GENERATOR_MODULES = {"physics", "generator"}
"""import 가 금지된 모듈명."""


def _imported_modules(path: Path) -> set[str]:
    """파일이 import 하는 모듈 경로를 모두 수집한다 (상대 import 포함)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("package", FORBIDDEN_IMPORTERS)
def test_no_generator_leak(package: str) -> None:
    """모델·설명·최적화·SECOM 이 데이터 생성 규칙을 import 하지 않는지 검사한다.

    생성기의 임계값을 최적화 가이드의 정답으로 재사용하는 구조를 구조적으로 차단한다.
    가이드는 오직 **학습된 모델의 predict_proba** 로부터만 유도되어야 한다.
    """
    for path in (SRC / package).rglob("*.py"):
        imports = _imported_modules(path)
        for imported in imports:
            tail = imported.split(".")[-1]
            if tail in GENERATOR_MODULES and "data" in imported:
                pytest.fail(
                    f"{path.relative_to(SRC.parent.parent)} 이 생성 규칙을 import 했습니다: {imported}\n"
                    "→ 생성기의 임계값을 정답으로 재사용하면 순환 논리가 됩니다. "
                    "가이드는 학습된 모델에서만 유도해야 합니다."
                )


def test_adjustable_and_fixed_partition() -> None:
    """조정 가능/불가능 변수가 전체를 빠짐없이, 겹침 없이 나눈다."""
    assert set(ADJUSTABLE) | set(FIXED) == set(FEATURE_NAMES)
    assert not set(ADJUSTABLE) & set(FIXED)


def test_product_spec_is_not_adjustable() -> None:
    """제품 사양·자재·설비 상태는 절대 조정 대상이 아니다.

    다이 두께를 바꾸라는 제안은 "제품을 바꾸라"는 말과 같다. 그 순간 실무자 신뢰를 잃는다.
    """
    for name in ("die_thickness", "tape_type", "runtime_hours", "vacuum_status"):
        assert not SPEC_BY_NAME[name].adjustable, f"{name} 이 조정 가능으로 잘못 설정됨"


def test_monotone_constraints_aligned() -> None:
    """단조 제약 리스트가 피처 순서와 길이·순서 모두 일치한다.

    이 정렬이 어긋나면 제약이 엉뚱한 변수에 걸리는데, 학습은 조용히 성공하므로
    끝까지 발견되지 않을 수 있다.
    """
    assert len(MONOTONE_CONSTRAINTS) == len(FEATURE_NAMES)
    for constraint, name in zip(MONOTONE_CONSTRAINTS, FEATURE_NAMES):
        assert constraint in (-1, 0, 1), f"{name}: 잘못된 제약값 {constraint}"


def test_vacuum_sign_convention() -> None:
    """음수 kPa 게이지압 변수의 제약 방향이 부호 규약과 맞는지 확인한다.

    "진공이 세다 = 값이 더 음수다" 이므로, 진공이 셀수록 성공률이 오르는 관계는
    부호 있는 값 기준으로 **단조 감소(-1)** 여야 한다. 이 부호를 놓치면 방향이 통째로 뒤집힌다.
    """
    for name in ("head_vacuum", "pin_vacuum", "vacuum_status"):
        spec = SPEC_BY_NAME[name]
        assert spec.high <= 0, f"{name} 은 음수 게이지압이어야 합니다"
        assert spec.monotone == -1, f"{name} 의 제약 방향이 부호 규약과 어긋납니다"


def test_clip_to_bounds_respects_range_and_resolution() -> None:
    """제안값이 허용 범위를 넘지 않고 설비 설정 분해능으로 스냅된다."""
    clipped = clip_to_bounds({"pin_pressure": 999.0, "uv_time": -5.0, "pin_speed": 1.23456})
    assert clipped["pin_pressure"] == BOUNDS["pin_pressure"][1]
    assert clipped["uv_time"] == BOUNDS["uv_time"][0]
    assert clipped["pin_speed"] == pytest.approx(1.23)


def test_clip_passes_unknown_keys_through() -> None:
    """스키마에 없는 키는 그대로 통과시킨다 (SECOM 등 외부 데이터 호환)."""
    assert clip_to_bounds({"unknown_sensor": 3.14})["unknown_sensor"] == 3.14
