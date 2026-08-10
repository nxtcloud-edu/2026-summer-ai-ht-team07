"""SHAP 기반 개별 다이 실패 원인 분해.

.. important::
   **확률 공간에서 계산한다.** ``shap.TreeExplainer`` 의 기본 출력은 로그 오즈(log-odds)다.
   그대로 화면에 띄우면 "-0.83" 같은 값을 실무자에게 보여주게 되고, 발표에서
   단위를 잘못 말하게 된다. ``model_output="probability"`` 를 주면 기여도가
   **확률 %p 단위**로 나와 "이 변수가 성공률을 12%p 깎았다"고 말할 수 있다.

.. warning::
   **SHAP 은 상관 기반 설명이지 인과 추정이 아니다.**
   "이 변수를 바꾸면 이만큼 좋아진다"는 인과 주장이 아니라 "학습된 모델이 이 예측을
   이렇게 분해했다"는 진술이다. UI 문구와 README 에 이 한계를 명시하고,
   포지션은 항상 **"SHAP 으로 후보 변수를 좁히고, 실제 개선 폭은 PoC 실증으로 검증한다"** 를 유지한다.

소유자: D(설명·최적화).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..schema import EXPLANATION_COLUMNS, ID_COL, SPEC_BY_NAME

DISCLAIMER: str = (
    "SHAP 기여도는 학습된 모델이 이 예측을 어떻게 분해했는지를 보여주는 **상관 기반 설명**이며, "
    "인과 효과 추정이 아닙니다. 개선 후보 변수를 좁히는 용도로 사용하고, 실제 개선 폭은 "
    "현장 PoC로 검증해야 합니다."
)
"""UI와 발표자료에 그대로 노출할 한계 고지 문구. 문구를 바꾸려면 팀 합의 후 이 상수만 고친다."""


@dataclass
class ExplainerBundle:
    """SHAP explainer 와 기준값.

    Attributes:
        explainer: ``shap.TreeExplainer`` 인스턴스.
        base_value: 기준 확률(모든 기여가 0일 때의 예측). 워터폴 시작점.
        feature_names: 기여도 배열의 컬럼 순서.
    """

    explainer: Any
    base_value: float
    feature_names: list[str]


def build_explainer(bundle, background: pd.DataFrame, max_background: int = 200) -> ExplainerBundle:
    """확률 공간 TreeExplainer 를 만든다.

    Args:
        bundle: ``registry.ModelBundle``. 트리 계열이어야 한다.
        background: 배경 데이터셋 (보통 학습 세트 일부). ``model_output="probability"``
            는 배경 분포가 필요하며, 클수록 느려지므로 샘플링한다.
        max_background: 배경 표본 상한. 200행이면 데모 응답성이 충분하다.

    Returns:
        ExplainerBundle.

    Raises:
        TypeError: 트리 계열이 아닌 모델을 넘긴 경우 (LogisticRegression 등).
    """
    import shap

    if not bundle.is_tree():
        raise TypeError(
            f"TreeExplainer 는 트리 계열 전용입니다. 현재 모델: {bundle.name}. "
            "configs/model.yaml 의 primary_model 을 트리 계열로 두세요."
        )

    bg = background[bundle.feature_names]
    if len(bg) > max_background:
        bg = bg.sample(max_background, random_state=0)

    explainer = shap.TreeExplainer(
        bundle.model,
        data=bg,
        model_output="probability",
        feature_perturbation="interventional",
    )
    base = explainer.expected_value
    base_value = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)
    return ExplainerBundle(
        explainer=explainer, base_value=base_value, feature_names=list(bundle.feature_names)
    )


def shap_values(explainer_bundle: ExplainerBundle, X: pd.DataFrame) -> np.ndarray:
    """SHAP 기여도를 **확률 단위**로 계산한다.

    Args:
        explainer_bundle: ``build_explainer`` 결과.
        X: 설명할 행들.

    Returns:
        ``(n_rows, n_features)`` 배열. 단위는 확률(0~1 스케일).
        %p 로 보여줄 때는 100을 곱한다.

    Note:
        LightGBM 이진 분류에서 shap 라이브러리 버전에 따라 (n, f) 또는 (n, f, 2)
        형태가 나온다. 양성 클래스 축만 뽑아 형태를 통일한다.
    """
    values = explainer_bundle.explainer.shap_values(X[explainer_bundle.feature_names])
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[..., -1]
    elif isinstance(values, list):
        arr = np.asarray(values[-1])
    return arr


def explain_frame(
    explainer_bundle: ExplainerBundle,
    X: pd.DataFrame,
    ids: pd.Series | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """개별 다이 기여도를 long format 계약(``EXPLANATION_COLUMNS``)으로 반환한다.

    Args:
        explainer_bundle: ``build_explainer`` 결과.
        X: 설명할 행들.
        ids: 다이 식별자. None 이면 행 번호로 만든다.
        top_k: 행당 상위 몇 개 변수만 남길지 (절대값 기준). None 이면 전부.

    Returns:
        ``[die_id, feature, shap_value_pp, feature_value, direction]`` DataFrame.
        ``shap_value_pp`` 는 **%p 단위**(예: -12.4 = 성공률을 12.4%p 깎음).
        ``direction`` 은 ``"위험"``/``"기여"`` — UI 색상 분기에 사용.
    """
    arr = shap_values(explainer_bundle, X) * 100.0
    names = explainer_bundle.feature_names
    die_ids = list(ids) if ids is not None else [f"row_{i}" for i in range(len(X))]

    rows = []
    for i, die_id in enumerate(die_ids):
        contributions = arr[i]
        order = np.argsort(-np.abs(contributions))
        if top_k is not None:
            order = order[:top_k]
        for j in order:
            rows.append(
                {
                    ID_COL: die_id,
                    "feature": names[j],
                    "shap_value_pp": float(contributions[j]),
                    "feature_value": float(X.iloc[i][names[j]]),
                    "direction": "기여" if contributions[j] >= 0 else "위험",
                }
            )
    return pd.DataFrame(rows, columns=list(EXPLANATION_COLUMNS))


def waterfall_frame(explanation: pd.DataFrame, base_value: float, die_id: str) -> pd.DataFrame:
    """워터폴 차트용 누적 표를 만든다.

    Args:
        explanation: ``explain_frame`` 결과.
        base_value: 기준 확률 (0~1).
        die_id: 대상 다이.

    Returns:
        ``[label, shap_value_pp, cumulative_pp]`` DataFrame.
        첫 행은 기준선(``기준 성공률``), 마지막 행은 최종 예측이다.

    Note:
        Streamlit 기본 차트로도 그릴 수 있는 형태로 반환한다. shap 의 matplotlib
        플롯은 Streamlit 재실행 때 종종 깨져서, 데모 안정성을 위해 표를 직접 만든다.
    """
    subset = explanation[explanation[ID_COL] == die_id].copy()
    cumulative = base_value * 100.0
    rows = [{"label": "기준 성공률", "shap_value_pp": 0.0, "cumulative_pp": cumulative}]
    for _, row in subset.iterrows():
        cumulative += row["shap_value_pp"]
        spec = SPEC_BY_NAME.get(row["feature"])
        label = f"{spec.korean} ({row['feature_value']:g}{spec.unit})" if spec else row["feature"]
        rows.append(
            {
                "label": label,
                "shap_value_pp": float(row["shap_value_pp"]),
                "cumulative_pp": float(cumulative),
            }
        )
    rows.append({"label": "최종 예측", "shap_value_pp": 0.0, "cumulative_pp": cumulative})
    return pd.DataFrame(rows)


def top_risk_features(explanation: pd.DataFrame, die_id: str, k: int = 3) -> list[dict[str, Any]]:
    """실패 기여가 큰 상위 변수를 뽑는다.

    최적화 파트가 "어디부터 볼지" 정하는 입력으로 쓴다.

    Args:
        explanation: ``explain_frame`` 결과.
        die_id: 대상 다이.
        k: 개수.

    Returns:
        ``[{feature, shap_value_pp, feature_value, korean}]`` 리스트 (음수 기여 큰 순).
    """
    subset = explanation[(explanation[ID_COL] == die_id) & (explanation["shap_value_pp"] < 0)]
    subset = subset.sort_values("shap_value_pp").head(k)
    out = []
    for _, row in subset.iterrows():
        spec = SPEC_BY_NAME.get(row["feature"])
        out.append(
            {
                "feature": row["feature"],
                "shap_value_pp": float(row["shap_value_pp"]),
                "feature_value": float(row["feature_value"]),
                "korean": spec.korean if spec else row["feature"],
            }
        )
    return out
