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

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    """OOF 예측으로 선택한 판정 임계값과 선택 근거."""

    threshold: float
    objective: str
    objective_value: float
    recall: float
    precision: float
    candidate_count: int
    feasible_candidate_count: int
    min_recall: float

    def to_dict(self) -> dict[str, float | int | str]:
        """JSON/메트릭 저장용 dict로 변환한다."""
        return asdict(self)


def select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_recall: float,
    objective: str = "f1_macro",
) -> ThresholdSelection:
    """PR 곡선 후보 중 recall 제약을 만족하며 macro-F1이 최대인 임계값을 고른다.

    이 함수에는 홀드아웃 예측을 넘기면 안 된다. 학습 세트의 out-of-fold 확률만
    사용해야 최종 홀드아웃 평가가 untouched 상태로 남는다. 동률이면 recall이 큰
    후보, 그마저 같으면 더 낮은 임계값을 택해 결과를 결정적으로 만든다.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("y_true와 y_prob은 길이가 같은 1차원 배열이어야 합니다.")
    if len(y_true) == 0 or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("임계값 선택에는 0/1 두 클래스가 모두 필요합니다.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob은 유한한 0~1 확률이어야 합니다.")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall은 0~1 범위여야 합니다.")
    if objective != "f1_macro":
        raise ValueError(f"지원하지 않는 임계값 objective: {objective}")

    # ``precision_recall_curve`` 가 반환하는 모든 고유 확률을 후보로 유지하되,
    # 후보마다 sklearn metric을 다시 계산하는 O(n_candidates * n_samples)
    # 루프는 피한다. 확률을 한 번 정렬한 뒤 각 threshold의 suffix 혼동행렬을
    # 벡터화하면 결과는 동일하면서 O(n log n)이다.
    _, _, raw_thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.asarray(raw_thresholds, dtype=float)
    order = np.argsort(y_prob, kind="mergesort")
    sorted_prob = y_prob[order]
    sorted_target = y_true[order]
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_target, dtype=np.int64)))
    starts = np.searchsorted(sorted_prob, thresholds, side="left")

    n_positive = int(sorted_target.sum())
    n_negative = int(len(sorted_target) - n_positive)
    true_positive = n_positive - positive_prefix[starts]
    predicted_positive = len(sorted_target) - starts
    false_positive = predicted_positive - true_positive
    false_negative = n_positive - true_positive
    true_negative = n_negative - false_positive

    recall_values = np.divide(
        true_positive,
        n_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=n_positive > 0,
    )
    precision_values = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_positive > 0,
    )
    positive_f1 = np.divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        out=np.zeros_like(true_positive, dtype=float),
        where=(2 * true_positive + false_positive + false_negative) > 0,
    )
    negative_f1 = np.divide(
        2 * true_negative,
        2 * true_negative + false_positive + false_negative,
        out=np.zeros_like(true_negative, dtype=float),
        where=(2 * true_negative + false_positive + false_negative) > 0,
    )
    macro_f1_values = (positive_f1 + negative_f1) / 2.0
    feasible = np.flatnonzero(recall_values + 1e-12 >= min_recall)

    if len(feasible) == 0:
        raise ValueError(f"recall >= {min_recall:.4f} 를 만족하는 PR 곡선 후보가 없습니다.")

    # 동률 규칙: macro-F1 최대 → recall 최대 → threshold 최소.
    best_index = max(
        feasible.tolist(),
        key=lambda index: (
            float(macro_f1_values[index]),
            float(recall_values[index]),
            -float(thresholds[index]),
        ),
    )
    return ThresholdSelection(
        threshold=float(thresholds[best_index]),
        objective=objective,
        objective_value=float(macro_f1_values[best_index]),
        recall=float(recall_values[best_index]),
        precision=float(precision_values[best_index]),
        candidate_count=int(len(thresholds)),
        feasible_candidate_count=int(len(feasible)),
        min_recall=float(min_recall),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
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


def save_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: str | Path,
    *,
    n_bins: int = 10,
    label: str = "model",
    dpi: int = 160,
) -> Path:
    """홀드아웃 신뢰도 곡선을 PNG로 저장한다."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="perfect calibration")
    if not table.empty:
        sizes = 35.0 + 165.0 * table["n"].to_numpy(dtype=float) / float(table["n"].max())
        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            color="#1565C0",
            linewidth=2,
            label=label,
        )
        ax.scatter(
            table["mean_predicted"],
            table["observed_rate"],
            s=sizes,
            color="#1565C0",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted success probability",
        ylabel="Observed success rate",
        title="Calibration curve (untouched holdout)",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def comparison_frame(
    results: dict[str, dict[str, float]],
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
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
