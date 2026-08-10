"""Yield X 합성 데이터 생성기.

기존 프로토타입(``yeda_simulated_data.xlsx``)의 치명적 결함을 대체한다.

기존 데이터의 문제
    타겟이 다음 4개 부등식의 결정론적 AND 로 5,000행 **전부** 재현되었다::

        uv_time >= 4.2 AND pin_pressure >= 26 AND head_vacuum <= -64 AND pin_vacuum <= -47

    나머지 9개 변수는 타겟 상관계수 -0.03~0.01 의 순수 난수였고, 그 결과 RandomForest
    정확도 99.86% / depth 8 단일 트리 99.84% 가 나왔다. 모델이 배운 것은 공정 물리가
    아니라 부등식 4개였고, "주요 문제 변수"와 최적화 가이드는 사전에 하드코딩한 규칙을
    되읽는 순환 논리였다. pandas 세 줄로 심사위원이 검증 가능한 수준의 결함이다.

이 생성기의 대응
    1. 결정론적 AND 폐기 → 연속 잠재 점수 + 로지스틱 + 베르누이 샘플링(확률적 라벨).
    2. 13개 변수 전부가 크기가 다른 기여를 갖는다.
    3. 상호작용항 3개 — 선형 모델로는 못 잡고 트리 앙상블이 필요한 구조.
    4. 라벨 생성 후 관측값에 센서 노이즈 + 분해능 반올림.
    5. 1~3% 결측 삽입.
    6. 성공률 65~80%, 베이즈 정확도 82~90% 대역으로 자동 검증.
    7. 시드 고정 + 파라미터 YAML 외부화.

소유자: C(데이터·모델). 물리 계수(``configs/data_gen.yaml``)의 결정권은 A(공정물리).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from ..io_utils import load_config, resolve, save_json
from ..schema import (
    CATEGORICAL,
    FEATURE_NAMES,
    ID_COL,
    RAW_COLUMNS,
    RESOLUTION,
    SPEC_BY_NAME,
    TARGET,
)
from .physics import bayes_accuracy, latent_score, sigmoid


@dataclass
class GenerationReport:
    """생성 결과 요약. ``artifacts/metrics/data_generation.json`` 으로 저장된다.

    Attributes:
        n_samples: 생성 행 수.
        seed: 사용한 시드.
        success_rate: 타겟 평균(=픽업 성공률).
        bayes_accuracy: 도달 가능한 이론적 최대 정확도.
        mean_probability: 평균 성공 확률 p.
        prob_std: p 의 표준편차. 너무 작으면 변수가 라벨을 설명하지 못한다는 뜻.
        missing_rate: 전체 결측 비율.
        warnings: 목표 대역 이탈 등 검수 경고.
    """

    n_samples: int
    seed: int
    success_rate: float
    bayes_accuracy: float
    mean_probability: float
    prob_std: float
    missing_rate: float
    warnings: list[str]


def _sample_truncated_normal(
    rng: np.random.Generator,
    *,
    center: float,
    sigma: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """거절 샘플링으로 범위 안의 절단정규분포를 뽑는다.

    ``normal`` 표본을 사후 ``clip`` 하면 범위 밖 질량이 양 끝점에 쌓여 실제
    절단정규분포가 되지 않는다. 여기서는 범위 밖 표본을 버리고 필요한 개수만큼
    다시 뽑는다. 전달받은 ``Generator`` 만 사용하므로 시드 재현성도 유지된다.

    Raises:
        ValueError: 범위·표준편차·설정점 또는 표본 수가 유효하지 않은 경우.
    """
    if size < 0:
        raise ValueError(f"size 는 0 이상이어야 합니다: {size}")
    if not np.isfinite([center, sigma, low, high]).all():
        raise ValueError("절단정규분포 파라미터는 모두 유한해야 합니다")
    if low >= high:
        raise ValueError(f"절단 범위가 잘못되었습니다: [{low}, {high}]")
    if sigma <= 0:
        raise ValueError(f"sigma 는 양수여야 합니다: {sigma}")
    if not low <= center <= high:
        raise ValueError(f"설정점 {center} 이 허용 범위 [{low}, {high}] 밖입니다")

    samples = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        candidates = rng.normal(center, sigma, size=size - filled)
        accepted = candidates[(candidates >= low) & (candidates <= high)]
        take = min(len(accepted), size - filled)
        samples[filled : filled + take] = accepted[:take]
        filled += take
    return samples


def _sample_conditions(config: dict[str, Any], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """공정 조건의 '참값'을 샘플링한다 (센서 노이즈 이전).

    실제 양산 데이터는 균등분포가 아니라 레시피 설정점 주변에 몰린다. 동시에 조건
    탐색(DOE) 로트는 범위 전체를 훑는다. 두 성분을 섞어야
      (a) 분포가 현실적이고, (b) 모델이 범위 가장자리도 학습해 최적화 제안이 안전해진다.

    Args:
        config: ``configs/data_gen.yaml`` 로드 결과.
        rng: numpy 난수 생성기.

    Returns:
        ``{피처명: 참값 배열}``. 레시피 성분은 거절 샘플링한 절단정규분포이며,
        모든 값이 허용 범위 안에 있다.
    """
    n = int(config["n_samples"])
    smp = config["sampling"]
    recipe_mask = rng.random(n) < float(smp["recipe_ratio"])
    spread_ratio = float(smp["spread_ratio"])

    cols: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        if name in CATEGORICAL:
            cols[name] = (rng.random(n) < float(smp["tape_type_p1"])).astype(float)
            continue

        center = float(smp["setpoints"][name])
        sigma = (spec.high - spec.low) * spread_ratio
        values = np.empty(n, dtype=float)
        n_recipe = int(recipe_mask.sum())
        values[~recipe_mask] = rng.uniform(spec.low, spec.high, size=n - n_recipe)
        values[recipe_mask] = _sample_truncated_normal(
            rng,
            center=center,
            sigma=sigma,
            low=spec.low,
            high=spec.high,
            size=n_recipe,
        )
        cols[name] = values

    return cols


def _apply_sensor_noise(
    cols: dict[str, np.ndarray], config: dict[str, Any], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """관측값에 센서 노이즈를 입히고 분해능으로 반올림한다.

    라벨은 '참값'으로 이미 생성된 뒤이므로, 여기서 생기는 참값-관측값 괴리가
    곧 **환원 불가능한 오차**가 된다. 현실의 계측 한계를 그대로 반영한 것이다.

    Args:
        cols: 참값 dict.
        config: 생성 설정.
        rng: 난수 생성기.

    Returns:
        노이즈가 적용된 새 dict (원본은 보존).
    """
    noise_cfg = config["noise"]
    k = float(noise_cfg["sensor_sigma_in_resolution"])
    do_round = bool(noise_cfg.get("round_to_resolution", True))

    observed: dict[str, np.ndarray] = {}
    for name, values in cols.items():
        spec = SPEC_BY_NAME[name]
        if name in CATEGORICAL:
            observed[name] = values.copy()
            continue

        noisy = values + rng.normal(0.0, k * RESOLUTION[name], size=len(values))
        noisy = np.clip(noisy, spec.low, spec.high)
        if do_round:
            res = RESOLUTION[name]
            noisy = np.round(noisy / res) * res
            decimals = max(0, len(str(res).split(".")[1]) if "." in str(res) else 0)
            noisy = np.round(noisy, decimals)
        observed[name] = noisy
    return observed


def _inject_missing(df: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """피처에 랜덤 결측을 삽입한다 (타겟/식별자는 제외).

    전처리 로직이 반드시 필요한 상태를 만들기 위한 의도적 오염이다.
    MCAR(완전 무작위 결측)로 두어 24시간 안에 다룰 수 있는 난이도를 유지한다.

    Args:
        df: 원본 DataFrame (변경되지 않음).
        config: 생성 설정.
        rng: 난수 생성기.

    Returns:
        결측이 삽입된 복사본.
    """
    miss_cfg = config["missing"]
    default_rate = float(miss_cfg["default_rate"])
    per_feature: dict[str, float] = miss_cfg.get("per_feature") or {}

    out = df.copy()
    for name in FEATURE_NAMES:
        rate = float(per_feature.get(name, default_rate))
        if rate <= 0:
            continue
        mask = rng.random(len(out)) < rate
        out.loc[mask, name] = np.nan
    return out


def generate(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, GenerationReport]:
    """합성 데이터셋을 생성한다.

    파이프라인::

        참값 샘플링 → 잠재 점수 → p=sigmoid → y~Bernoulli(p)
                   → 센서 노이즈 + 반올림 → 결측 삽입 → 검수

    Args:
        config: 생성 설정. None 이면 ``configs/data_gen.yaml`` 을 읽는다.

    Returns:
        ``(df, report)``. ``df`` 는 ``schema.RAW_COLUMNS`` 순서를 따른다.

    Note:
        반환된 df 에는 잠재 점수나 참확률 p 가 **포함되지 않는다.** 학습·최적화 쪽으로
        정답이 새는 것을 막기 위함이다. 검수용 통계만 report 로 나간다.
    """
    cfg = config if config is not None else load_config("data_gen")
    rng = np.random.default_rng(int(cfg["seed"]))

    truth = _sample_conditions(cfg, rng)
    latent = latent_score(truth, cfg)
    p = sigmoid(latent)
    y = (rng.random(len(p)) < p).astype(int)

    observed = _apply_sensor_noise(truth, cfg, rng)

    df = pd.DataFrame({name: observed[name] for name in FEATURE_NAMES})
    df[TARGET] = y
    df.insert(0, ID_COL, [f"D{i:06d}" for i in range(len(df))])
    df = df[list(RAW_COLUMNS)]

    df = _inject_missing(df, cfg, rng)

    report = _build_report(df, p, cfg)
    return df, report


def _build_report(df: pd.DataFrame, p: np.ndarray, cfg: dict[str, Any]) -> GenerationReport:
    """생성 결과를 검수하고 목표 대역 이탈을 경고로 남긴다.

    성공률이나 베이즈 정확도가 목표를 벗어나면 ``latent.scale`` / ``latent.intercept``
    를 재보정해야 한다는 신호다. 특히 베이즈 정확도가 0.95 를 넘으면 라벨이 사실상
    결정론이 된 것이므로 **기존 프로토타입과 같은 실패**다.
    """
    success_rate = float(df[TARGET].mean())
    bayes = bayes_accuracy(p)
    missing_rate = float(df[list(FEATURE_NAMES)].isna().mean().mean())

    warnings: list[str] = []
    lo, hi = cfg["targets"]["success_rate"]
    if not lo <= success_rate <= hi:
        warnings.append(f"성공률 {success_rate:.3f} 가 목표 [{lo}, {hi}] 밖 → latent.intercept 재보정 필요")
    b_lo, b_hi = cfg["targets"]["bayes_accuracy"]
    if not b_lo <= bayes <= b_hi:
        warnings.append(f"베이즈 정확도 {bayes:.3f} 가 목표 [{b_lo}, {b_hi}] 밖 → latent.scale 재보정 필요")
    if bayes > 0.95:
        warnings.append("베이즈 정확도 0.95 초과: 라벨이 사실상 결정론적이다. 기존 프로토타입과 같은 실패.")

    return GenerationReport(
        n_samples=len(df),
        seed=int(cfg["seed"]),
        success_rate=success_rate,
        bayes_accuracy=bayes,
        mean_probability=float(np.mean(p)),
        prob_std=float(np.std(p)),
        missing_rate=missing_rate,
        warnings=warnings,
    )


def generate_and_save(config: dict[str, Any] | None = None) -> GenerationReport:
    """생성 후 CSV 와 검수 리포트를 디스크에 쓴다.

    ``make data`` 의 진입점이다.

    Args:
        config: 생성 설정. None 이면 ``configs/data_gen.yaml``.

    Returns:
        검수 리포트. ``warnings`` 가 비어 있어야 정상.
    """
    cfg = config if config is not None else load_config("data_gen")
    df, report = generate(cfg)

    out_path = resolve(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    save_json(asdict(report), "artifacts/metrics/data_generation.json")
    return report
