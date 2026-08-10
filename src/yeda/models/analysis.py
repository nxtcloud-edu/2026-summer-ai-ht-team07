"""저장된 모델의 수율 보정, 순열 어블레이션, 피처 스윕 분석.

이 모듈은 데이터 생성 규칙이나 공정 최적화 코드를 참조하지 않는다. 모든 결과는
저장된 모델의 predict_proba 출력에 대한 연관성·민감도 분석이며 인과효과가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from ..data.preprocess import make_split
from ..io_utils import load_config, resolve
from ..schema import CATEGORICAL, FEATURE_NAMES, SPEC_BY_NAME
from .registry import ModelBundle, load_bundle

CAUSALITY_WARNING = (
    "모델 민감도 분석이며 인과효과 또는 실제 공정 개선을 보장하지 않음."
)
FIGURE_WARNING = (
    "Model sensitivity only; associations are not causal effects or guaranteed gains."
)


@dataclass
class AnalysisResult:
    """분석표와 생성한 파일 경로."""

    yield_summary: pd.DataFrame
    permutation_summary: pd.DataFrame
    permutation_repeats: pd.DataFrame
    feature_sweep: pd.DataFrame
    feature_sweep_summary: pd.DataFrame
    paths: dict[str, Path]


def _validate_inputs(
    bundle: ModelBundle,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [name for name in bundle.feature_names if name not in X.columns]
    if missing:
        raise ValueError(f"분석 입력에 피처가 없습니다: {missing}")

    prepared = X.loc[:, bundle.feature_names].copy()
    if prepared.isna().any().any():
        raise ValueError("분석 입력은 학습 세트 대치값으로 결측 처리되어야 합니다.")

    labels = np.asarray(y_true, dtype=int).reshape(-1)
    if len(prepared) != len(labels) or len(labels) == 0:
        raise ValueError("X 와 y_true 는 같은 길이의 비어 있지 않은 데이터여야 합니다.")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("y_true 는 0과 1을 모두 포함해야 합니다.")
    return prepared, labels


def _predict_success(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(bundle.predict_proba(X), dtype=float).reshape(-1)
    if len(probability) != len(X):
        raise ValueError("predict_proba 결과 길이가 입력 행 수와 다릅니다.")
    if not np.isfinite(probability).all():
        raise ValueError("predict_proba 가 유한하지 않은 값을 반환했습니다.")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("predict_proba 결과는 0과 1 사이여야 합니다.")
    return probability


def summarize_batch_yield(
    bundle: ModelBundle,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """홀드아웃의 예상 수율과 실제 수율 차이를 한 행으로 요약한다."""
    prepared, labels = _validate_inputs(bundle, X, y_true)
    probability = _predict_success(bundle, prepared)
    predicted_yield = float(probability.mean())
    actual_yield = float(labels.mean())
    calibration_gap = predicted_yield - actual_yield
    return pd.DataFrame(
        [
            {
                "model": bundle.name,
                "n_holdout": len(labels),
                "predicted_yield": predicted_yield,
                "actual_yield": actual_yield,
                "calibration_gap": calibration_gap,
                "predicted_yield_pct": predicted_yield * 100.0,
                "actual_yield_pct": actual_yield * 100.0,
                "calibration_gap_pp": calibration_gap * 100.0,
                "absolute_calibration_gap_pp": abs(calibration_gap) * 100.0,
                "causal": False,
                "disclaimer": CAUSALITY_WARNING,
            }
        ]
    )


def permutation_ablation(
    bundle: ModelBundle,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    *,
    n_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """각 피처를 반복 셔플해 PR-AUC 저하와 Brier 증가를 측정한다."""
    if n_repeats < 2:
        raise ValueError("permutation_repeats 는 2 이상이어야 합니다.")
    prepared, labels = _validate_inputs(bundle, X, y_true)
    baseline_probability = _predict_success(bundle, prepared)
    baseline_pr_auc = float(average_precision_score(labels, baseline_probability))
    baseline_brier = float(brier_score_loss(labels, baseline_probability))

    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(prepared)) for _ in range(n_repeats)]
    repeat_rows: list[dict[str, float | int | str]] = []

    for feature in bundle.feature_names:
        spec = SPEC_BY_NAME[feature]
        original = prepared[feature].to_numpy(copy=True)
        for repeat, indices in enumerate(permutations):
            permuted = prepared.copy()
            permuted[feature] = original[indices]
            probability = _predict_success(bundle, permuted)
            permuted_pr_auc = float(average_precision_score(labels, probability))
            permuted_brier = float(brier_score_loss(labels, probability))
            repeat_rows.append(
                {
                    "feature": feature,
                    "korean": spec.korean,
                    "unit": spec.unit,
                    "adjustable": spec.adjustable,
                    "monotone": spec.monotone,
                    "repeat": repeat,
                    "seed": seed,
                    "baseline_pr_auc": baseline_pr_auc,
                    "permuted_pr_auc": permuted_pr_auc,
                    "pr_auc_drop": baseline_pr_auc - permuted_pr_auc,
                    "baseline_brier": baseline_brier,
                    "permuted_brier": permuted_brier,
                    "brier_increase": permuted_brier - baseline_brier,
                    "causal": False,
                    "disclaimer": CAUSALITY_WARNING,
                }
            )

    repeats = pd.DataFrame(repeat_rows)
    grouped = repeats.groupby("feature", sort=False)
    summary = grouped.agg(
        korean=("korean", "first"),
        unit=("unit", "first"),
        adjustable=("adjustable", "first"),
        monotone=("monotone", "first"),
        n_repeats=("repeat", "count"),
        baseline_pr_auc=("baseline_pr_auc", "first"),
        permuted_pr_auc_mean=("permuted_pr_auc", "mean"),
        permuted_pr_auc_std=("permuted_pr_auc", "std"),
        pr_auc_drop_mean=("pr_auc_drop", "mean"),
        pr_auc_drop_std=("pr_auc_drop", "std"),
        baseline_brier=("baseline_brier", "first"),
        permuted_brier_mean=("permuted_brier", "mean"),
        permuted_brier_std=("permuted_brier", "std"),
        brier_increase_mean=("brier_increase", "mean"),
        brier_increase_std=("brier_increase", "std"),
    ).reset_index()
    summary["causal"] = False
    summary["disclaimer"] = CAUSALITY_WARNING
    summary = summary.sort_values(
        ["pr_auc_drop_mean", "brier_increase_mean"],
        ascending=False,
    ).reset_index(drop=True)
    return summary, repeats


def _feature_grid(
    feature: str,
    observed: pd.Series,
    n_points: int,
    quantile_low: float,
    quantile_high: float,
) -> tuple[np.ndarray, float, float]:
    spec = SPEC_BY_NAME[feature]
    if feature in CATEGORICAL:
        levels = np.sort(pd.to_numeric(observed, errors="coerce").dropna().unique())
        levels = levels[(levels >= spec.low) & (levels <= spec.high)].astype(float)
        if len(levels) == 0:
            raise ValueError(f"관측된 범주 수준이 없습니다: {feature}")
        return levels, float(levels.min()), float(levels.max())
    if n_points < 2:
        raise ValueError("sweep_grid_points 는 2 이상이어야 합니다.")
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("sweep quantile 은 0 <= low < high <= 1 이어야 합니다.")

    numeric = pd.to_numeric(observed, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        raise ValueError(f"관측값이 없습니다: {feature}")
    q_low = max(float(spec.low), float(np.quantile(numeric, quantile_low)))
    q_high = min(float(spec.high), float(np.quantile(numeric, quantile_high)))
    # 5.8 / 0.1 이 내부적으로 57.999...가 되는 경우 floor가 5.7로
    # 한 칸 줄어들 수 있다. 분해능 단위로 스케일한 뒤 작은 허용오차를 줘서
    # 실제로 경계 위에 놓인 관측 분위수 끝점을 보존한다.
    snap_tolerance = 1e-9
    grid_low = (
        np.ceil(q_low / spec.resolution - snap_tolerance) * spec.resolution
    )
    grid_high = (
        np.floor(q_high / spec.resolution + snap_tolerance) * spec.resolution
    )
    if grid_low > grid_high:
        raise ValueError(f"관측 분위수 범위에서 grid 를 만들 수 없습니다: {feature}")

    raw = np.linspace(grid_low, grid_high, n_points)
    snapped = np.round(raw / spec.resolution) * spec.resolution
    grid = np.unique(np.round(np.clip(snapped, grid_low, grid_high), 10))
    return grid, q_low, q_high


def feature_value_sweep(
    bundle: ModelBundle,
    X: pd.DataFrame,
    *,
    n_points: int,
    quantile_low: float,
    quantile_high: float,
) -> pd.DataFrame:
    """한 피처씩 물리 범위에서 바꿔 평균 예측 수율 변화를 계산한다."""
    missing = [name for name in bundle.feature_names if name not in X.columns]
    if missing:
        raise ValueError(f"분석 입력에 피처가 없습니다: {missing}")
    prepared = X.loc[:, bundle.feature_names].copy()
    if prepared.empty or prepared.isna().any().any():
        raise ValueError("feature sweep 입력은 비어 있지 않고 결측이 없어야 합니다.")

    baseline_yield = float(_predict_success(bundle, prepared).mean())
    rows: list[dict[str, float | int | str]] = []
    for feature in bundle.feature_names:
        spec = SPEC_BY_NAME[feature]
        grid, q_low, q_high = _feature_grid(
            feature,
            prepared[feature],
            n_points,
            quantile_low,
            quantile_high,
        )
        for grid_index, value in enumerate(grid):
            swept = prepared.copy()
            swept[feature] = value
            predicted_yield = float(_predict_success(bundle, swept).mean())
            rows.append(
                {
                    "feature": feature,
                    "korean": spec.korean,
                    "unit": spec.unit,
                    "adjustable": spec.adjustable,
                    "monotone": spec.monotone,
                    "grid_index": grid_index,
                    "feature_value": float(value),
                    "q_low": q_low,
                    "q_high": q_high,
                    "n_holdout": len(prepared),
                    "baseline_predicted_yield": baseline_yield,
                    "predicted_yield": predicted_yield,
                    "predicted_yield_pct": predicted_yield * 100.0,
                    "delta_pp": (predicted_yield - baseline_yield) * 100.0,
                    "causal": False,
                    "disclaimer": CAUSALITY_WARNING,
                }
            )
    return pd.DataFrame(rows)


def summarize_feature_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """피처별 관측 범위 내 예측 수율 영향폭과 양 끝점을 한 행으로 요약한다."""
    rows: list[dict[str, float | int | bool | str]] = []
    for feature in FEATURE_NAMES:
        subset = sweep[sweep["feature"] == feature].sort_values("feature_value")
        if subset.empty:
            raise ValueError(f"feature sweep 결과가 없습니다: {feature}")

        minimum = subset.loc[subset["predicted_yield"].idxmin()]
        maximum = subset.loc[subset["predicted_yield"].idxmax()]
        low_endpoint = subset.iloc[0]
        high_endpoint = subset.iloc[-1]
        min_yield = float(minimum["predicted_yield"])
        max_yield = float(maximum["predicted_yield"])
        low_yield = float(low_endpoint["predicted_yield"])
        high_yield = float(high_endpoint["predicted_yield"])
        spec = SPEC_BY_NAME[feature]
        rows.append(
            {
                "feature": feature,
                "korean": spec.korean,
                "unit": spec.unit,
                "adjustable": spec.adjustable,
                "monotone": spec.monotone,
                "q_low": float(subset["q_low"].iloc[0]),
                "q_high": float(subset["q_high"].iloc[0]),
                "n_grid_points": len(subset),
                "baseline_predicted_yield": float(
                    subset["baseline_predicted_yield"].iloc[0]
                ),
                "min_predicted_yield": min_yield,
                "max_predicted_yield": max_yield,
                "effect_span_pp": (max_yield - min_yield) * 100.0,
                "min_predicted_yield_value": float(minimum["feature_value"]),
                "max_predicted_yield_value": float(maximum["feature_value"]),
                "low_endpoint_value": float(low_endpoint["feature_value"]),
                "low_endpoint_yield": low_yield,
                "high_endpoint_value": float(high_endpoint["feature_value"]),
                "high_endpoint_yield": high_yield,
                "endpoint_delta_pp": (high_yield - low_yield) * 100.0,
                "causal": False,
                "disclaimer": CAUSALITY_WARNING,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "effect_span_pp", ascending=False
    ).reset_index(drop=True)


def _save_csv(frame: pd.DataFrame, configured_path: str) -> Path:
    path = resolve(configured_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _save_yield_figure(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    row = summary.iloc[0]
    values = [row["actual_yield_pct"], row["predicted_yield_pct"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    bars = ax.bar(
        ["Actual holdout yield", "Mean predicted yield"],
        values,
        color=["#455A64", "#1565C0"],
        width=0.6,
    )
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Yield (%)")
    ax.set_title(
        "Holdout Yield Calibration\n"
        f"gap = {float(row['calibration_gap_pp']):+.2f} percentage points"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, FIGURE_WARNING, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_ablation_figure(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    ordered = summary.sort_values("pr_auc_drop_mean", ascending=True)
    features = ordered["feature"].tolist()
    y_pos = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.2), sharey=True)

    axes[0].barh(
        y_pos,
        ordered["pr_auc_drop_mean"],
        xerr=ordered["pr_auc_drop_std"],
        color="#1565C0",
        alpha=0.9,
        capsize=3,
    )
    axes[0].set_yticks(y_pos, labels=features)
    axes[0].set_xlabel("PR-AUC drop after permutation")
    axes[0].axvline(0.0, color="#444444", linewidth=0.8)
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(
        y_pos,
        ordered["brier_increase_mean"],
        xerr=ordered["brier_increase_std"],
        color="#EF6C00",
        alpha=0.9,
        capsize=3,
    )
    axes[1].set_xlabel("Brier score increase after permutation")
    axes[1].axvline(0.0, color="#444444", linewidth=0.8)
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Model-agnostic Permutation Ablation", fontsize=15)
    fig.text(0.5, 0.01, FIGURE_WARNING, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_sweep_figure(sweep: pd.DataFrame, path: Path, dpi: int) -> None:
    n_columns = 4
    n_rows = int(np.ceil(len(FEATURE_NAMES) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(16.0, 12.0))
    flat_axes = np.asarray(axes).reshape(-1)

    for axis, feature in zip(flat_axes, FEATURE_NAMES):
        subset = sweep[sweep["feature"] == feature]
        axis.plot(
            subset["feature_value"],
            subset["delta_pp"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            color="#1565C0",
        )
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_title(feature, fontsize=10)
        axis.grid(alpha=0.2)
        if feature in CATEGORICAL:
            axis.set_xticks(subset["feature_value"].to_numpy())

    for axis in flat_axes[len(FEATURE_NAMES):]:
        axis.set_visible(False)

    fig.suptitle("Feature Sweep: Mean Predicted Yield Change", fontsize=15)
    fig.supxlabel("Feature value", y=0.035)
    fig.supylabel("Predicted yield delta (percentage points)")
    fig.text(0.5, 0.005, FIGURE_WARNING, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.025, 0.07, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    config: dict[str, Any] | None = None,
    *,
    permutation_repeats: int | None = None,
    sweep_grid_points: int | None = None,
    seed: int | None = None,
) -> AnalysisResult:
    """저장된 본선 모델과 동일 홀드아웃으로 모든 분석 산출물을 생성한다."""
    cfg = config if config is not None else load_config("model")
    analysis_cfg = cfg.get("analysis")
    if not isinstance(analysis_cfg, dict):
        raise ValueError("configs/model.yaml 에 analysis 설정이 필요합니다.")
    outputs = analysis_cfg.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("analysis.outputs 설정이 필요합니다.")

    bundle = load_bundle(cfg)
    if bundle.name != cfg["primary_model"]:
        raise ValueError("저장 모델과 primary_model 설정이 다릅니다. make train 을 다시 실행하세요.")
    if list(bundle.feature_names) != list(FEATURE_NAMES):
        raise ValueError("저장 모델의 피처 순서가 현재 schema 와 다릅니다.")

    split = make_split(test_size=float(cfg["test_size"]), seed=int(cfg["seed"]))
    X_holdout = split.X_test.loc[:, bundle.feature_names]
    y_holdout = split.y_test

    repeat_count = int(
        permutation_repeats
        if permutation_repeats is not None
        else analysis_cfg["permutation_repeats"]
    )
    grid_points = int(
        sweep_grid_points
        if sweep_grid_points is not None
        else analysis_cfg["sweep_grid_points"]
    )
    analysis_seed = int(seed if seed is not None else analysis_cfg["seed"])

    yield_summary = summarize_batch_yield(bundle, X_holdout, y_holdout)
    permutation_summary, permutation_repeats_frame = permutation_ablation(
        bundle,
        X_holdout,
        y_holdout,
        n_repeats=repeat_count,
        seed=analysis_seed,
    )
    sweep = feature_value_sweep(
        bundle,
        X_holdout,
        n_points=grid_points,
        quantile_low=float(analysis_cfg["sweep_quantile_low"]),
        quantile_high=float(analysis_cfg["sweep_quantile_high"]),
    )
    sweep_summary = summarize_feature_sweep(sweep)

    paths = {
        "yield_summary_csv": _save_csv(
            yield_summary, outputs["yield_summary_csv"]
        ),
        "permutation_summary_csv": _save_csv(
            permutation_summary, outputs["permutation_summary_csv"]
        ),
        "permutation_repeats_csv": _save_csv(
            permutation_repeats_frame, outputs["permutation_repeats_csv"]
        ),
        "feature_sweep_csv": _save_csv(
            sweep, outputs["feature_sweep_csv"]
        ),
        "feature_sweep_summary_csv": _save_csv(
            sweep_summary, outputs["feature_sweep_summary_csv"]
        ),
        "yield_figure": resolve(outputs["yield_figure"]),
        "permutation_figure": resolve(outputs["permutation_figure"]),
        "feature_sweep_figure": resolve(outputs["feature_sweep_figure"]),
    }
    dpi = int(analysis_cfg["figure_dpi"])
    _save_yield_figure(yield_summary, paths["yield_figure"], dpi)
    _save_ablation_figure(
        permutation_summary, paths["permutation_figure"], dpi
    )
    _save_sweep_figure(sweep, paths["feature_sweep_figure"], dpi)

    return AnalysisResult(
        yield_summary=yield_summary,
        permutation_summary=permutation_summary,
        permutation_repeats=permutation_repeats_frame,
        feature_sweep=sweep,
        feature_sweep_summary=sweep_summary,
        paths=paths,
    )
