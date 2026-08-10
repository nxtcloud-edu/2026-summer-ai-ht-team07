"""다이 픽업 공정의 물리 관계 → 잠재 점수(latent score) 변환.

이 모듈은 **합성 데이터 생성 전용**이다.
기여 함수의 형태와 계수는 공정 물리에서 유도되며, 근거는 ``docs/PHYSICS_RATIONALE.md``
에 변수 단위로 기록한다.

.. danger::
   **절대 금지 (프로젝트 1순위 규칙)**

   이 모듈은 ``yeda.models`` / ``yeda.explain`` / ``yeda.optimize`` 어디에서도
   import 되어서는 안 된다. 생성 규칙을 최적화 가이드의 정답으로 재사용하는 순간
   전체 결과가 순환 논리가 된다("정답을 넣고 정답을 꺼냈다").

   이 규칙은 ``tests/test_contracts.py::test_no_generator_leak`` 가 자동 검사한다.

소유자: A(공정물리) — 계수와 함수 형태의 결정권자. C(데이터·모델)는 구현만 돕는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from ..schema import SPEC_BY_NAME

Shape = Literal["linear", "sat", "peak"]
"""기여 함수 형태.

linear : 범위 내에서 효과가 포화되지 않는 변수 (예: 마모, 습도)
sat    : 효과가 포화되는 단조 변수 — tanh (예: UV 조사, 핀 압력)
peak   : 내부 최적점을 갖는 변수 — 이차 페널티 (예: 핀 높이, 온도)
"""


def normalize(x: np.ndarray, name: str) -> np.ndarray:
    """물리값을 허용 범위 기준 [-1, 1] 로 정규화한다.

    Args:
        x: 원시 물리값 배열.
        name: 피처명 (``schema.SPEC_BY_NAME`` 조회용).

    Returns:
        low → -1, high → +1 로 선형 사상된 배열. 범위를 벗어나면 그대로 넘어간다.
    """
    spec = SPEC_BY_NAME[name]
    center = (spec.high + spec.low) / 2.0
    half = (spec.high - spec.low) / 2.0
    return (np.asarray(x, dtype=float) - center) / half


@dataclass(frozen=True)
class Contribution:
    """단일 변수의 잠재 점수 기여 정의.

    Attributes:
        name: 피처명.
        shape: 기여 함수 형태.
        weight: **부호 있는** 가중치.
            양수 = 값이 커질수록 픽업 성공 확률 상승.
            ``shape="peak"`` 일 때는 최적점에서 벗어난 정도에 대한 **페널티 크기**(양수)로 쓴다.
        optimum: ``shape="peak"`` 전용. 물리 단위의 최적값.
        sharpness: ``shape="sat"`` 전용 tanh 기울기. 클수록 빨리 포화된다.
    """

    name: str
    shape: Shape
    weight: float
    optimum: float | None = None
    sharpness: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """물리값 배열 → 잠재 점수 기여분."""
        u = normalize(x, self.name)
        if self.shape == "linear":
            return self.weight * u
        if self.shape == "sat":
            return self.weight * np.tanh(self.sharpness * u)
        if self.shape == "peak":
            if self.optimum is None:
                raise ValueError(f"{self.name}: shape='peak' 인데 optimum 이 없음")
            u_opt = normalize(np.asarray([self.optimum]), self.name)[0]
            return -abs(self.weight) * (u - u_opt) ** 2
        raise ValueError(f"알 수 없는 shape: {self.shape}")


# --------------------------------------------------------------------------
# 상호작용항
# --------------------------------------------------------------------------
# 선형 모델이 잡을 수 없고 트리 앙상블이 필요한 구조를 의도적으로 만든다.
# 각 항은 (이름, 설명, 함수) 로 등록되며 함수는 {피처명: 배열} 를 받아 기여분을 반환한다.

InteractionFn = Callable[[dict[str, np.ndarray], float], np.ndarray]


def _int_speed_thickness(cols: dict[str, np.ndarray], c: float) -> np.ndarray:
    """얇은 다이 x 빠른 핀 = 크랙.

    ``u_speed * u_thickness`` 곱. 두꺼운 다이(+1)는 빠른 핀(+1)을 견디지만(+c),
    얇은 다이(-1)에 빠른 핀(+1)이 오면 굽힘 응력으로 크랙이 난다(-c).
    부호가 뒤집히는 진짜 교차 상호작용이라 선형항으로는 재현 불가능하다.
    """
    return c * normalize(cols["pin_speed"], "pin_speed") * normalize(
        cols["die_thickness"], "die_thickness"
    )


def _int_uv_tape(cols: dict[str, np.ndarray], c: float) -> np.ndarray:
    """UV 경화형 테이프(type=1)에서만 UV 조사 효과가 증폭된다.

    일반 테이프(type=0)는 UV에 거의 반응하지 않으므로 상호작용항이 0이 된다.
    → uv_time 의 단조 증가 제약을 깨지 않으면서(추가항이 tape=0 일 때 소거),
      tape 종류에 따라 최적 UV 조건이 달라지는 실제 현장 상황을 재현한다.
    """
    tape = np.asarray(cols["tape_type"], dtype=float)
    return c * normalize(cols["uv_time"], "uv_time") * tape


def _int_wear_vacuum(cols: dict[str, np.ndarray], c: float) -> np.ndarray:
    """마모된 콜릿 x 약한 헤드 진공 = 복합 악화.

    두 조건이 **동시에** 나쁠 때만 추가 손실이 발생하는 AND형 위험.
    각 인자를 [0,1] 로 사상해 곱하므로 한쪽이 정상이면 페널티가 사라진다.
    """
    wear = (normalize(cols["runtime_hours"], "runtime_hours") + 1.0) / 2.0
    weak_vac = (normalize(cols["head_vacuum"], "head_vacuum") + 1.0) / 2.0
    return -abs(c) * wear * weak_vac


INTERACTIONS: dict[str, InteractionFn] = {
    "pin_speed_x_die_thickness": _int_speed_thickness,
    "uv_time_x_tape_type": _int_uv_tape,
    "runtime_hours_x_head_vacuum": _int_wear_vacuum,
}
"""등록된 상호작용항. 키는 ``configs/data_gen.yaml`` 의 ``interactions`` 키와 일치해야 한다."""


# --------------------------------------------------------------------------
# 잠재 점수
# --------------------------------------------------------------------------

def build_contributions(config: dict) -> list[Contribution]:
    """YAML 설정에서 ``Contribution`` 목록을 만든다.

    Args:
        config: ``configs/data_gen.yaml`` 을 로드한 dict. ``contributions`` 키를 읽는다.

    Returns:
        설정에 정의된 순서대로의 Contribution 리스트.

    Raises:
        KeyError: 스키마에 없는 피처명이 설정에 있는 경우.
    """
    out: list[Contribution] = []
    for name, spec in config["contributions"].items():
        if name not in SPEC_BY_NAME:
            raise KeyError(f"스키마에 없는 피처가 data_gen.yaml 에 있음: {name}")
        out.append(
            Contribution(
                name=name,
                shape=spec["shape"],
                weight=float(spec["weight"]),
                optimum=spec.get("optimum"),
                sharpness=float(spec.get("sharpness", 1.0)),
            )
        )
    return out


def latent_score(
    cols: dict[str, np.ndarray],
    config: dict,
    *,
    return_parts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """공정 조건 → 잠재 점수(로그 오즈).

    ``p = sigmoid(latent)`` 이고 라벨은 ``y ~ Bernoulli(p)`` 로 뽑는다.
    즉 **같은 조건에서도 결과가 갈린다.** 기존 프로토타입의 결정론적 AND 규칙을
    폐기한 것이 이 함수의 존재 이유다.

    Args:
        cols: ``{피처명: 값 배열}``. 13개 피처가 모두 있어야 한다.
        config: ``configs/data_gen.yaml`` 로드 결과.
        return_parts: True 면 항별 기여분 dict 를 함께 반환한다(검수/문서화용).

    Returns:
        잠재 점수 배열. ``return_parts=True`` 면 ``(latent, parts)``.
    """
    parts: dict[str, np.ndarray] = {}
    n = len(next(iter(cols.values())))
    total = np.zeros(n, dtype=float)

    for contrib in build_contributions(config):
        value = contrib(cols[contrib.name])
        parts[contrib.name] = value
        total = total + value

    for key, coef in config.get("interactions", {}).items():
        if key not in INTERACTIONS:
            raise KeyError(f"등록되지 않은 상호작용항: {key}")
        value = INTERACTIONS[key](cols, float(coef))
        parts[key] = value
        total = total + value

    scale = float(config["latent"]["scale"])
    intercept = float(config["latent"]["intercept"])
    latent = intercept + scale * total

    if return_parts:
        return latent, parts
    return latent


def sigmoid(z: np.ndarray) -> np.ndarray:
    """수치적으로 안전한 로지스틱 함수."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def bayes_accuracy(p: np.ndarray) -> float:
    """이 생성기로 도달 가능한 **이론적 최대 정확도**.

    라벨이 확률적이므로 완벽한 모델(=진짜 p를 아는 모델)조차 틀린다.
    ``E[max(p, 1-p)]`` 가 그 상한이며, 학습 모델 정확도가 이 값을 크게 넘으면
    데이터 누수를 의심해야 한다.

    Args:
        p: 성공 확률 배열.

    Returns:
        베이즈 정확도 상한 (0~1).
    """
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.maximum(p, 1.0 - p)))
