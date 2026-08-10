"""평가 지표와 신뢰도 곡선.

.. warning::
   **accuracy 단독 보고 금지.** 성공률이 70%대이므로 "전부 성공"이라고만 찍어도
   정확도 70%가 나온다. PR-AUC / recall / macro-F1 을 함께 본다.

   또한 우리는 확률을 **수율(%)** 로 화면에 띄운다. "이 조건의 예상 수율 82%" 라고
   말하려면 그 82%가 실제로 82% 성공률을 뜻해야 한다. 그것을 보증하는 것이
   신뢰도 곡선(calibration curve)이며, 이 프로젝트에서는 선택이 아니라 필수다.

소유자: C(데이터·모델).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """홀드아웃 성능 지표 일괄 계산.

    Args:
        y_true: 실제 라벨 (0/1).
        y_prob: 예측 성공 확률.
        threshold: 판정 임계값.

    Returns:
        지표명 → 값 dict. ``brier`` 는 낮을수록 좋다(확률 정확도).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """신뢰도 곡선용 구간별 집계표.

    예측 확률을 등간격 구간으로 나누고, 구간별 평균 예측확률과 실제 성공률을 비교한다.
    두 값이 대각선 위에 있으면 "예측 82% = 실제 82%" 가 성립한다.

    Args:
        y_true: 실제 라벨.
        y_prob: 예측 확률.
        n_bins: 구간 수.

    Returns:
        컬럼 ``[bin_lower, bin_upper, n, mean_predicted, observed_rate, gap]`` DataFrame.
        표본이 없는 구간은 제외된다.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        observed = float(y_true[mask].mean())
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": n,
                "mean_predicted": mean_pred,
                "observed_rate": observed,
                "gap": observed - mean_pred,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """ECE — 구간별 |예측-실제| 를 표본 수로 가중 평균한 값.

    0에 가까울수록 확률이 잘 보정된 것이다. 발표 슬라이드에 한 숫자로 넣기 좋다.

    Args:
        table: ``calibration_table()`` 결과.

    Returns:
        ECE 값. 표가 비어 있으면 ``nan``.
    """
    if table.empty:
        return float("nan")
    weights = table["n"].to_numpy(dtype=float)
    return float(np.average(np.abs(table["gap"].to_numpy(dtype=float)), weights=weights))


def comparison_frame(results: dict[str, dict[str, float]], labels: dict[str, str] | None = None) -> pd.DataFrame:
    """모델별 지표 dict 를 비교표로 만든다.

    이 표가 "왜 이 모델인가"에 답하는 근거다. 발표자료에 그대로 붙인다.

    Args:
        results: ``{모델명: 지표dict}``.
        labels: 모델명 → 표시용 라벨.

    Returns:
        PR-AUC 내림차순으로 정렬된 DataFrame.
    """
    labels = labels or {}
    rows = []
    for name, metrics in results.items():
        row = {"model": name, "label": labels.get(name, name)}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "pr_auc" in df.columns:
        df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return df
