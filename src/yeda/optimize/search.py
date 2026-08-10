"""학습된 모델을 대리 모델(surrogate)로 둔 공정 조건 탐색.

출력 형태는 항상 하나다::

    "무엇을 얼마로 바꾸면 예측 성공률이 몇 %p 오른다"

.. danger::
   **이 모듈은 ``yeda.data.physics`` / ``yeda.data.generator`` 를 import 하지 않는다.**
   생성 규칙의 임계값을 정답으로 재사용하면 전체 결과가 순환 논리가 된다.
   가이드는 오직 학습된 모델의 ``predict_proba`` 만 질의해서 만든다.
   (``tests/test_contracts.py::test_no_generator_leak`` 가 자동 검사)

설계 원칙
    1. **조정 가능/불가능 분리** — ``die_thickness``(제품 사양), ``tape_type``(자재),
       ``runtime_hours``·``vacuum_status``(설비 상태)는 고정. 이걸 바꾸라고 제안하는
       순간 실무자에게 신뢰를 잃는다.
    2. **물리적 허용 범위 제약** — ``schema.BOUNDS`` 밖은 탐색하지 않는다.
    3. **현장 수용성** — 한 번에 바꿀 변수 개수와 이동 폭을 제한한다.
    4. **설정 가능한 값만 제안** — 센서 분해능으로 스냅한다(37.8213N 같은 제안 금지).

소유자: D(설명·최적화).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..io_utils import load_config
from ..text_utils import eul_reul, format_value
from ..schema import (
    ADJUSTABLE,
    BOUNDS,
    FEATURE_NAMES,
    RECOMMENDATION_COLUMNS,
    SPEC_BY_NAME,
    clip_to_bounds,
)

ScoreFn = Callable[[pd.DataFrame], np.ndarray]
"""조건 DataFrame → 성공 확률 배열. 보통 ``bundle.predict_proba``."""


@dataclass
class OptimizationResult:
    """탐색 결과.

    Attributes:
        baseline_prob: 현재 조건의 예측 성공 확률.
        optimized_prob: 제안 조건의 예측 성공 확률.
        gain_pp: 개선 폭 (%p).
        current: 현재 조건 dict.
        suggested: 제안 조건 dict (변경된 변수만이 아니라 전체).
        recommendations: ``RECOMMENDATION_COLUMNS`` 계약을 따르는 DataFrame.
        n_evaluations: 모델 질의 횟수 (성능 리포트용).
    """

    baseline_prob: float
    optimized_prob: float
    gain_pp: float
    current: dict[str, float]
    suggested: dict[str, float]
    recommendations: pd.DataFrame
    n_evaluations: int


def tunable_features(config: dict[str, Any] | None = None) -> list[str]:
    """탐색 대상 변수 목록.

    ``schema.ADJUSTABLE`` 이 상한이다. 설정으로 **더 좁힐 수만** 있고 넓힐 수는 없다.
    고정 변수를 실수로 탐색에 넣는 사고를 구조적으로 막는다.

    Args:
        config: ``configs/optimize.yaml``. None 이면 파일에서 읽는다.

    Returns:
        탐색 가능한 피처명 리스트.
    """
    cfg = config if config is not None else load_config("optimize")
    override = cfg.get("tunable_override")
    if not override:
        return list(ADJUSTABLE)
    invalid = [f for f in override if f not in ADJUSTABLE]
    if invalid:
        raise ValueError(f"조정 불가 변수를 탐색 대상으로 지정할 수 없습니다: {invalid}")
    return list(override)


def _candidate_bounds(name: str, current: float, max_relative_move: float) -> tuple[float, float]:
    """한 변수의 탐색 구간 = 물리 허용 범위 ∩ 현재값 주변 이동 제한.

    현장에서 "핀 압력을 20에서 40으로 두 배 올리세요"는 받아들여지지 않는다.
    허용 범위 폭의 일정 비율 안에서만 제안한다.
    """
    low, high = BOUNDS[name]
    reach = (high - low) * max_relative_move
    return max(low, current - reach), min(high, current + reach)


def _grid(name: str, low: float, high: float, n: int) -> list[float]:
    """분해능으로 스냅된 후보값 그리드 (중복 제거)."""
    raw = np.linspace(low, high, n)
    snapped = sorted({clip_to_bounds({name: float(v)})[name] for v in raw})
    return snapped


def _score_single(score_fn: ScoreFn, values: dict[str, float]) -> float:
    """조건 하나를 모델에 질의한다."""
    frame = pd.DataFrame([{k: values[k] for k in FEATURE_NAMES}])
    return float(score_fn(frame)[0])


def _score_batch(score_fn: ScoreFn, rows: Sequence[dict[str, float]]) -> np.ndarray:
    """조건 여러 개를 한 번에 질의한다 (좌표 하강의 속도 핵심)."""
    frame = pd.DataFrame([{k: r[k] for k in FEATURE_NAMES} for r in rows])
    return np.asarray(score_fn(frame), dtype=float)


def local_search(
    score_fn: ScoreFn,
    current: dict[str, float],
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], float, int]:
    """좌표 하강(coordinate ascent) 탐색.

    변수 하나씩 그리드를 훑어 가장 좋은 값으로 갱신하고, 이를 여러 라운드 반복한다.
    상호작용을 완전히 잡지는 못하지만 **결정론적이고 빠르다**. 데모 리허설에서
    같은 입력에 같은 답이 나오는 것이 랜덤 서치보다 훨씬 중요하다.

    Args:
        score_fn: 조건 → 확률 함수.
        current: 현재 조건 (13개 피처 전부 필요).
        config: 최적화 설정.

    Returns:
        ``(최적 조건, 최적 확률, 모델 질의 횟수)``.
    """
    cfg = config if config is not None else load_config("optimize")
    max_move = float(cfg["constraints"]["max_relative_move"])
    grid_points = int(cfg["search"]["local"]["grid_points"])
    rounds = int(cfg["search"]["local"]["rounds"])
    names = tunable_features(cfg)

    best = dict(current)
    best_score = _score_single(score_fn, best)
    evaluations = 1

    for _ in range(rounds):
        improved = False
        for name in names:
            low, high = _candidate_bounds(name, float(current[name]), max_move)
            candidates = _grid(name, low, high, grid_points)
            rows = [{**best, name: value} for value in candidates]
            scores = _score_batch(score_fn, rows)
            evaluations += len(rows)

            top = int(np.argmax(scores))
            if scores[top] > best_score + 1e-9:
                best = rows[top]
                best_score = float(scores[top])
                improved = True
        if not improved:
            break

    return best, best_score, evaluations


def random_search(
    score_fn: ScoreFn,
    current: dict[str, float],
    config: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[dict[str, float], float, int]:
    """랜덤 서치 — 상호작용이 강할 때 좌표 하강의 지역 최적을 보완한다.

    Args:
        score_fn: 조건 → 확률 함수.
        current: 현재 조건.
        config: 최적화 설정.
        seed: 재현성 시드.

    Returns:
        ``(최적 조건, 최적 확률, 질의 횟수)``.
    """
    cfg = config if config is not None else load_config("optimize")
    max_move = float(cfg["constraints"]["max_relative_move"])
    n_trials = int(cfg["search"]["random"]["n_trials"])
    names = tunable_features(cfg)
    rng = np.random.default_rng(seed)

    rows = [dict(current)]
    for _ in range(n_trials):
        trial = dict(current)
        for name in names:
            low, high = _candidate_bounds(name, float(current[name]), max_move)
            trial[name] = float(rng.uniform(low, high))
        trial.update(clip_to_bounds({n: trial[n] for n in names}))
        rows.append(trial)

    scores = _score_batch(score_fn, rows)
    top = int(np.argmax(scores))
    return rows[top], float(scores[top]), len(rows)


def optuna_search(
    score_fn: ScoreFn,
    current: dict[str, float],
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], float, int]:
    """Optuna TPE 탐색 (선택). optuna 미설치 시 랜덤 서치로 폴백한다.

    TODO(D): 시간이 남으면(M5 이후) 켠다. 데모 안정성이 우선이므로 기본 전략은 local.

    Args:
        score_fn: 조건 → 확률 함수.
        current: 현재 조건.
        config: 최적화 설정.

    Returns:
        ``(최적 조건, 최적 확률, 질의 횟수)``.
    """
    cfg = config if config is not None else load_config("optimize")
    try:
        import optuna
    except ImportError:
        return random_search(score_fn, current, cfg)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    max_move = float(cfg["constraints"]["max_relative_move"])
    names = tunable_features(cfg)
    counter = {"n": 0}

    def objective(trial: "optuna.Trial") -> float:
        candidate = dict(current)
        for name in names:
            low, high = _candidate_bounds(name, float(current[name]), max_move)
            candidate[name] = trial.suggest_float(name, low, high)
        candidate.update(clip_to_bounds({n: candidate[n] for n in names}))
        counter["n"] += 1
        return _score_single(score_fn, candidate)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(
        objective,
        n_trials=int(cfg["search"]["optuna"]["n_trials"]),
        timeout=cfg["search"]["optuna"].get("timeout_seconds"),
    )
    best = dict(current)
    best.update(clip_to_bounds(study.best_params))
    return best, float(study.best_value), counter["n"]


_STRATEGIES = {"local": local_search, "random": random_search, "optuna": optuna_search}


def _build_recommendations(
    score_fn: ScoreFn,
    current: dict[str, float],
    suggested: dict[str, float],
    baseline_prob: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float], float]:
    """제안을 기여 큰 순으로 정렬하고, 수용성 제약에 맞게 잘라낸다.

    변수를 하나씩 **누적 적용**하면서 실제 개선 폭을 다시 측정한다. 개별 변수의
    단독 효과를 그냥 더하면 상호작용 때문에 실제와 어긋나는데, 발표에서 "%p 오른다"고
    말한 숫자가 화면 값과 다르면 그 자리에서 신뢰를 잃는다.

    Returns:
        ``(추천표, 최종 채택 조건, 최종 확률)``.
    """
    max_changed = int(config["constraints"]["max_changed_features"])
    min_gain = float(config["constraints"]["min_gain_pp"])

    changed = [n for n in suggested if abs(suggested[n] - current[n]) > 1e-12]
    if not changed:
        return pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS)), dict(current), baseline_prob

    # 각 변수를 단독 적용했을 때의 이득으로 우선순위를 매긴다.
    solo_rows = [{**current, name: suggested[name]} for name in changed]
    solo_scores = _score_batch(score_fn, solo_rows)
    order = sorted(range(len(changed)), key=lambda i: -solo_scores[i])

    rows: list[dict[str, Any]] = []
    accepted = dict(current)
    running_prob = baseline_prob

    for idx in order[:max_changed]:
        name = changed[idx]
        trial = {**accepted, name: suggested[name]}
        prob = _score_single(score_fn, trial)
        gain_pp = (prob - running_prob) * 100.0
        if gain_pp < min_gain:
            continue

        spec = SPEC_BY_NAME[name]
        rows.append(
            {
                "feature": name,
                "current_value": float(current[name]),
                "suggested_value": float(suggested[name]),
                "delta": float(suggested[name] - current[name]),
                "unit": spec.unit,
                "expected_gain_pp": float(gain_pp),
            }
        )
        accepted = trial
        running_prob = prob

    return pd.DataFrame(rows, columns=list(RECOMMENDATION_COLUMNS)), accepted, running_prob


def recommend(
    score_fn: ScoreFn,
    current: dict[str, float],
    config: dict[str, Any] | None = None,
) -> OptimizationResult:
    """현재 조건에 대한 개선 가이드를 생성한다.

    이 함수가 최적화 파트의 **공개 계약**이다. UI는 이것만 호출한다.

    Args:
        score_fn: 조건 DataFrame → 성공 확률 배열. 보통 ``bundle.predict_proba``.
        current: 현재 공정 조건. 13개 피처가 모두 있어야 한다.
        config: ``configs/optimize.yaml``. None 이면 파일에서 읽는다.

    Returns:
        OptimizationResult. ``recommendations`` 가 비어 있으면 "현재 조건이 이미
        국소 최적이거나, 의미 있는(>= min_gain_pp) 개선을 찾지 못했다"는 뜻이다.

    Raises:
        KeyError: ``current`` 에 빠진 피처가 있는 경우.
    """
    cfg = config if config is not None else load_config("optimize")

    missing = [n for n in FEATURE_NAMES if n not in current]
    if missing:
        raise KeyError(f"현재 조건에 빠진 피처: {missing}")

    baseline_prob = _score_single(score_fn, current)
    strategy = _STRATEGIES.get(cfg.get("strategy", "local"), local_search)
    raw_best, _, evaluations = strategy(score_fn, current, cfg)

    recommendations, accepted, final_prob = _build_recommendations(
        score_fn, current, raw_best, baseline_prob, cfg
    )

    return OptimizationResult(
        baseline_prob=baseline_prob,
        optimized_prob=final_prob,
        gain_pp=(final_prob - baseline_prob) * 100.0,
        current=dict(current),
        suggested=accepted,
        recommendations=recommendations,
        n_evaluations=evaluations,
    )


def format_recommendations(result: OptimizationResult) -> list[str]:
    """추천표를 사람이 읽는 한 줄 문장들로 바꾼다.

    Args:
        result: ``recommend()`` 결과.

    Returns:
        ``"핀 압력을 29.0N → 33.5N 으로 올리면 예측 성공률이 4.2%p 오릅니다."`` 형태의 문자열 리스트.
    """
    lines = []
    for _, row in result.recommendations.iterrows():
        spec = SPEC_BY_NAME[row["feature"]]
        verb = "올리면" if row["delta"] > 0 else "내리면"
        current = format_value(row["current_value"], spec.unit)
        suggested = format_value(row["suggested_value"], spec.unit)
        lines.append(
            f"{eul_reul(spec.korean)} {current} → {suggested} 으로 {verb} "
            f"예측 성공률이 {row['expected_gain_pp']:.1f}%p 오릅니다."
        )
    return lines
