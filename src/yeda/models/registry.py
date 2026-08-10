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
from ..schema import FEATURE_NAMES, MONOTONE_CONSTRAINTS


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
        return self.model.predict_proba(X[self.feature_names])[:, 1]

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
