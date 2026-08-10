"""모델 생성 / 저장 / 로드.

학습(``train.py``)과 서빙(Streamlit)이 **같은 방식으로** 모델을 다루도록 여기 모은다.
저장 번들에는 모델뿐 아니라 대치값과 피처 순서가 함께 들어간다. 이것들이 빠지면
추론 시점에 조용히 틀린 예측이 나온다.

소유자: C(데이터·모델).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..io_utils import load_config, resolve, save_json
from ..schema import BOUNDS, FEATURE_NAMES, MONOTONE_CONSTRAINTS


@dataclass
class ModelBundle:
    """추론에 필요한 모든 것을 담은 번들.

    Attributes:
        model: 학습된 sklearn 호환 분류기 (``predict_proba`` 보유).
        name: 모델 키 (``configs/model.yaml`` 의 키).
        feature_names: 학습 시점의 피처 순서. 추론 입력 정렬에 사용.
        imputer_values: 학습 세트 기준 결측 대치값.
        threshold: 성공/실패 판정 임계값.
        metrics: 홀드아웃 성능 요약 (UI 표시용).
    """

    model: Any
    name: str
    feature_names: list[str]
    imputer_values: dict[str, float]
    threshold: float = 0.5
    metrics: dict[str, float] | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """픽업 성공 확률을 반환한다.

        Args:
            X: 피처 DataFrame. 순서가 달라도 여기서 정렬한다.

        Returns:
            길이 ``len(X)`` 의 성공 확률 배열 (0~1).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ModelBundle.predict_proba 입력은 pandas DataFrame 이어야 합니다.")

        prepared = X.copy()
        for feature in self.feature_names:
            if feature not in self.imputer_values:
                raise KeyError(f"번들 대치값에 피처가 없습니다: {feature}")
            fill_value = float(self.imputer_values[feature])
            if feature not in prepared.columns:
                prepared[feature] = fill_value
            else:
                prepared[feature] = prepared[feature].fillna(fill_value)

        prepared = prepared[self.feature_names].astype(float)
        return np.asarray(self.model.predict_proba(prepared)[:, 1], dtype=float)

    def is_tree(self) -> bool:
        """SHAP ``TreeExplainer`` 를 쓸 수 있는 트리 계열인지 여부."""
        return hasattr(self.model, "booster_") or hasattr(self.model, "estimators_")


def build_model(name: str, config: dict[str, Any] | None = None) -> Any:
    """설정 이름으로 학습 전 모델 객체를 만든다.

    Args:
        name: ``configs/model.yaml`` 의 ``models`` 키.
        config: 모델 설정. None 이면 ``configs/model.yaml``.

    Returns:
        학습 전 sklearn 호환 추정기.

    Raises:
        KeyError: 설정에 없는 모델 이름.

    Note:
        ``lightgbm_monotone`` 의 제약 벡터는 ``schema.MONOTONE_CONSTRAINTS`` 에서
        주입한다. YAML 에 숫자를 복사해두면 스키마와 어긋난 순간 방향이 뒤집힌다.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    cfg = config if config is not None else load_config("model")
    if name not in cfg["models"]:
        raise KeyError(f"configs/model.yaml 에 없는 모델: {name}")

    entry = cfg["models"][name]
    params = dict(entry.get("params") or {})
    seed = int(cfg["seed"])

    if name == "logreg":
        # 선형 모델은 스케일에 민감하다. 표준화 파이프라인을 붙여야 공정한 베이스라인이 된다.
        return make_pipeline(
            StandardScaler(), LogisticRegression(random_state=seed, **params)
        )
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name.startswith("lightgbm"):
        if entry.get("apply_monotone"):
            params["monotone_constraints"] = list(MONOTONE_CONSTRAINTS)
        return lgb.LGBMClassifier(random_state=seed, **params)

    raise KeyError(f"빌더가 정의되지 않은 모델: {name}")


def save_bundle(bundle: ModelBundle, config: dict[str, Any] | None = None) -> Path:
    """번들을 joblib 로 저장하고 메타데이터 JSON 을 함께 남긴다.

    Args:
        bundle: 저장할 번들.
        config: 모델 설정. None 이면 ``configs/model.yaml``.

    Returns:
        저장된 모델 파일 경로.
    """
    cfg = config if config is not None else load_config("model")
    path = resolve(cfg["output"]["model_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    save_json(
        {
            "name": bundle.name,
            "feature_names": bundle.feature_names,
            "threshold": bundle.threshold,
            "metrics": bundle.metrics or {},
            "imputer_values": bundle.imputer_values,
        },
        cfg["output"]["metadata_path"],
    )
    return path


def load_bundle(config: dict[str, Any] | None = None) -> ModelBundle:
    """저장된 번들을 읽는다.

    Args:
        config: 모델 설정. None 이면 ``configs/model.yaml``.

    Returns:
        ModelBundle.

    Raises:
        FileNotFoundError: 아직 학습하지 않은 경우. ``make train`` 안내 포함.
    """
    cfg = config if config is not None else load_config("model")
    path = resolve(cfg["output"]["model_path"])
    if not path.exists():
        raise FileNotFoundError(f"학습된 모델 없음: {path}\n먼저 `make train` 을 실행하세요.")
    return joblib.load(path)


def load_bundle_or_none(config: dict[str, Any] | None = None) -> ModelBundle | None:
    """모델이 있으면 로드하고, 없으면 None 을 반환한다.

    UI 가 모델 없이도 mock 모드로 뜨게 하기 위한 진입점이다.
    데모 중 파일 하나 없다고 앱이 죽는 것이 가장 나쁜 시나리오다.
    """
    try:
        return load_bundle(config)
    except (FileNotFoundError, Exception):  # noqa: B014 - joblib 버전 불일치까지 흡수
        return None


def default_feature_order() -> list[str]:
    """스키마 기준 기본 피처 순서."""
    return list(FEATURE_NAMES)


def assert_monotone_prediction(
    bundle: ModelBundle,
    *,
    feature: str = "pin_speed",
    n_points: int = 101,
    atol: float = 1e-10,
) -> bool:
    """단조 제약 설정과 실제 예측 방향을 함께 검사한다.

    다른 피처는 학습 세트 대치값에 고정하고 feature 하나만 물리 범위
    전체에서 변화시킨다. 설정 벡터가 스키마와 다르거나 예측 방향을 위반하면
    학습 파이프라인을 즉시 실패시킨다.
    """
    if feature not in FEATURE_NAMES:
        raise KeyError(f"스키마에 없는 피처: {feature}")
    if n_points < 2:
        raise ValueError("n_points 는 2 이상이어야 합니다.")
    if atol < 0:
        raise ValueError("atol 은 0 이상이어야 합니다.")

    feature_index = FEATURE_NAMES.index(feature)
    direction = MONOTONE_CONSTRAINTS[feature_index]
    if direction == 0:
        raise ValueError(f"단조 제약이 없는 피처는 검증할 수 없습니다: {feature}")

    actual = bundle.model.get_params().get("monotone_constraints")
    if actual is None or list(actual) != list(MONOTONE_CONSTRAINTS):
        raise AssertionError(
            "학습 모델의 monotone_constraints 가 schema 순서와 일치하지 않습니다."
        )

    low, high = BOUNDS[feature]
    probe = pd.DataFrame(
        [bundle.imputer_values.copy() for _ in range(n_points)],
        columns=bundle.feature_names,
    )
    probe[feature] = np.linspace(float(low), float(high), n_points)
    delta = np.diff(bundle.predict_proba(probe))
    violation = delta < -atol if direction > 0 else delta > atol
    if np.any(violation):
        first = int(np.flatnonzero(violation)[0])
        raise AssertionError(
            f"{feature} 단조 제약 위반: grid index {first} -> {first + 1}"
        )
    return True


def verify_all_monotone_predictions(
    bundle: ModelBundle,
    *,
    n_points: int = 101,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """0이 아닌 모든 스키마 단조 제약을 예측 grid에서 검증한다.

    각 피처만 물리 범위 전체에서 움직이고 나머지 피처는 학습 세트 대치값에
    고정한다. 반환값은 그대로 JSON으로 저장할 수 있으며, 호출자가 ``passed``를
    확인해 학습을 실패시킬 수 있다. 검증 결과를 먼저 저장하기 위해 이 함수
    자체는 위반 시 예외를 던지지 않는다.

    Args:
        bundle: 단조 제약 LightGBM 번들.
        n_points: 피처별 등간격 grid 점 수.
        atol: 부동소수 오차 허용치.

    Returns:
        모델 설정 일치 여부와 피처별 위반 수를 담은 JSON 직렬화 가능 dict.
    """
    if n_points < 2:
        raise ValueError("n_points 는 2 이상이어야 합니다.")
    if atol < 0:
        raise ValueError("atol 은 0 이상이어야 합니다.")

    constrained = [
        (feature, int(direction))
        for feature, direction in zip(FEATURE_NAMES, MONOTONE_CONSTRAINTS)
        if direction != 0
    ]
    actual = bundle.model.get_params().get("monotone_constraints")
    constraints_match = actual is not None and list(actual) == list(MONOTONE_CONSTRAINTS)

    feature_reports: list[dict[str, Any]] = []
    for feature, direction in constrained:
        low, high = BOUNDS[feature]
        probe = pd.DataFrame(
            [bundle.imputer_values.copy() for _ in range(n_points)],
            columns=bundle.feature_names,
        )
        probe[feature] = np.linspace(float(low), float(high), n_points)
        probabilities = bundle.predict_proba(probe)
        deltas = np.diff(probabilities)
        violations = deltas < -atol if direction > 0 else deltas > atol
        violation_indices = np.flatnonzero(violations).astype(int).tolist()
        feature_reports.append(
            {
                "feature": feature,
                "direction": direction,
                "bounds": [float(low), float(high)],
                "grid_points": int(n_points),
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
                "delta_min": float(deltas.min()),
                "delta_max": float(deltas.max()),
                "violation_count": len(violation_indices),
                "violation_indices": violation_indices,
                "passed": not violation_indices,
            }
        )

    passed = constraints_match and all(item["passed"] for item in feature_reports)
    return {
        "model": bundle.name,
        "passed": bool(passed),
        "constraints_match_schema": bool(constraints_match),
        "constrained_feature_count": len(constrained),
        "grid_points": int(n_points),
        "atol": float(atol),
        "features": feature_reports,
    }
