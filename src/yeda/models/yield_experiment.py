"""홀드아웃 cohort에서 4개 모델의 조건 추천을 비교하는 오프라인 실험.

모델 선택은 홀드아웃 PR-AUC와 Brier로 먼저 확정하고 단조 여부를 함께 보고한다.
각 모델의 자기 예측 optimized yield는 모델 선택에 사용하지 않는다. 조건 추천은
D의 공개 recommend 계약을 그대로 호출한다. 모든 추천 조건을 동결한 뒤에만
호출자가 전달한 oracle로 사후 감사한다.

이 모듈은 합성 physics/generator를 절대 import하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from ..data.preprocess import Split, make_split
from ..io_utils import load_config, resolve, save_json
from ..optimize.search import recommend
from ..schema import FEATURE_NAMES
from .evaluate import calibration_table, expected_calibration_error
from .registry import build_model

OracleFn = Callable[[pd.DataFrame], np.ndarray]

SELECTION_RULE = (
    "holdout_pr_auc_desc_then_brier_asc; " "optimized_predicted_yield_not_used"
)
ORACLE_ROLE = (
    "posthoc_only_after_model_selection_and_all_recommendations_are_frozen; "
    "not_used_for_training_candidate_generation_or_optimization"
)
DISCLAIMER = (
    "Synthetic post-hoc counterfactual audit on observed conditions; "
    "not causal and not a production guarantee."
)


@dataclass
class YieldExperimentResult:
    summary: pd.DataFrame
    cohort_rows: pd.DataFrame
    preset_rows: pd.DataFrame
    primary_presets: pd.DataFrame
    recommended_model: str
    paths: dict[str, Path]


def _predict_success(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(
        model.predict_proba(frame.loc[:, FEATURE_NAMES])[:, 1],
        dtype=float,
    )
    if not np.isfinite(probability).all():
        raise ValueError("model returned non-finite probabilities")
    return np.clip(probability, 0.0, 1.0)


def _fit_models(
    split: Split,
    model_config: dict[str, Any],
    *,
    calibration_bins: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    labels = split.y_test.to_numpy(dtype=int)
    for name, entry in model_config["models"].items():
        if not entry.get("enabled", True):
            continue
        model = build_model(name, model_config)
        model.fit(split.X_train, split.y_train)
        probability = _predict_success(model, split.X_test)
        table = calibration_table(labels, probability, n_bins=calibration_bins)
        rows.append(
            {
                "model": name,
                "label": entry.get("label", name),
                "holdout_pr_auc": float(average_precision_score(labels, probability)),
                "holdout_brier": float(brier_score_loss(labels, probability)),
                "holdout_ece": expected_calibration_error(table),
                "holdout_calibration_gap_pp": float(
                    (probability.mean() - labels.mean()) * 100.0
                ),
                "monotone_guardrail": bool(entry.get("apply_monotone", False)),
            }
        )
        models[name] = model
    if not models:
        raise ValueError("enabled model이 없습니다.")
    return models, pd.DataFrame(rows)


def _select_model_without_optimized_yield(
    metrics: pd.DataFrame,
) -> tuple[str, str]:
    ordered = metrics.sort_values(
        ["holdout_pr_auc", "holdout_brier", "model"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    metric_winner = str(ordered.iloc[0]["model"])
    return metric_winner, "highest holdout PR-AUC with Brier tie-break"


def _score_function(model: Any) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda frame: _predict_success(model, frame)


def _freeze_recommendations(
    models: dict[str, Any],
    conditions: list[tuple[str, dict[str, float], int | None]],
    optimize_config: dict[str, Any],
    *,
    group: str,
) -> list[dict[str, Any]]:
    """모든 모델 추천을 physics 없이 생성해 immutable record로 만든다."""
    frozen: list[dict[str, Any]] = []
    for model_name, model in models.items():
        score_fn = _score_function(model)
        for sample_id, current, actual_label in conditions:
            result = recommend(score_fn, current, optimize_config)
            changed = result.recommendations["feature"].tolist()
            frozen.append(
                {
                    "group": group,
                    "sample_id": sample_id,
                    "model": model_name,
                    "actual_label": actual_label,
                    "baseline_predicted": float(result.baseline_prob),
                    "optimized_predicted": float(result.optimized_prob),
                    "predicted_gain_pp": float(result.gain_pp),
                    "n_changed": len(changed),
                    "changed_features": "|".join(changed),
                    "n_evaluations": int(result.n_evaluations),
                    "current": dict(result.current),
                    "suggested": dict(result.suggested),
                }
            )
    return frozen


def _oracle_probabilities(
    oracle: OracleFn | None,
    frame: pd.DataFrame,
) -> np.ndarray:
    if oracle is None:
        return np.full(len(frame), np.nan, dtype=float)
    probability = np.asarray(
        oracle(frame.loc[:, FEATURE_NAMES]),
        dtype=float,
    ).reshape(-1)
    if len(probability) != len(frame):
        raise ValueError("oracle output length does not match input")
    if not np.isfinite(probability).all():
        raise ValueError("oracle returned non-finite probability")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("oracle probability must be in [0, 1]")
    return probability


def _posthoc_oracle_audit(
    frozen: list[dict[str, Any]],
    *,
    oracle: OracleFn | None,
    target_yield: float,
) -> pd.DataFrame:
    """동결된 추천만 batch oracle로 채점한다."""
    baseline_frame = pd.DataFrame(
        [row["current"] for row in frozen],
        columns=FEATURE_NAMES,
    )
    optimized_frame = pd.DataFrame(
        [row["suggested"] for row in frozen],
        columns=FEATURE_NAMES,
    )
    baseline_oracle = _oracle_probabilities(oracle, baseline_frame)
    optimized_oracle = _oracle_probabilities(oracle, optimized_frame)

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(frozen):
        baseline_value = float(baseline_oracle[index])
        optimized_value = float(optimized_oracle[index])
        oracle_available = np.isfinite(optimized_value)
        output_rows.append(
            {
                "group": row["group"],
                "sample_id": row["sample_id"],
                "model": row["model"],
                "actual_label": row["actual_label"],
                "baseline_predicted_yield": row["baseline_predicted"],
                "optimized_predicted_yield": row["optimized_predicted"],
                "predicted_gain_pp": row["predicted_gain_pp"],
                "baseline_oracle_yield": baseline_value,
                "optimized_oracle_yield": optimized_value,
                "oracle_gain_pp": (
                    (optimized_value - baseline_value) * 100.0
                    if oracle_available
                    else np.nan
                ),
                "true_improved": (
                    optimized_value > baseline_value + 1e-12
                    if oracle_available
                    else np.nan
                ),
                "true_worsened": (
                    optimized_value < baseline_value - 1e-12
                    if oracle_available
                    else np.nan
                ),
                "optimized_predicted_90pct_hit": (
                    row["optimized_predicted"] >= target_yield
                ),
                "optimized_oracle_90pct_hit": (
                    optimized_value >= target_yield if oracle_available else np.nan
                ),
                "optimized_prediction_oracle_error_pp": (
                    (row["optimized_predicted"] - optimized_value) * 100.0
                    if oracle_available
                    else np.nan
                ),
                "n_changed": row["n_changed"],
                "changed_features": row["changed_features"],
                "n_evaluations": row["n_evaluations"],
                "current_values": json.dumps(
                    row["current"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "suggested_values": json.dumps(
                    row["suggested"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "oracle_role": ORACLE_ROLE,
                "causal": False,
                "disclaimer": DISCLAIMER,
            }
        )
    return pd.DataFrame(output_rows)


def _paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    repeats: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _summarize_cohort(
    cohort: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    recommended_model: str,
    selection_reason: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bootstrap_repeats = int(experiment_config["bootstrap_repeats"])
    confidence_level = float(experiment_config["confidence_level"])
    base_seed = int(experiment_config["seed"])
    audit_cfg = experiment_config["posthoc_audit"]

    for model_index, (model_name, group) in enumerate(
        cohort.groupby("model", sort=False)
    ):
        metric = model_metrics[model_metrics["model"] == model_name].iloc[0]
        predicted_difference = group["predicted_gain_pp"].to_numpy(dtype=float)
        oracle_difference = group["oracle_gain_pp"].to_numpy(dtype=float)
        predicted_ci = _paired_bootstrap_interval(
            predicted_difference,
            repeats=bootstrap_repeats,
            confidence_level=confidence_level,
            seed=base_seed + model_index * 2,
        )
        oracle_ci = _paired_bootstrap_interval(
            oracle_difference,
            repeats=bootstrap_repeats,
            confidence_level=confidence_level,
            seed=base_seed + model_index * 2 + 1,
        )

        oracle_gain = float(group["oracle_gain_pp"].mean())
        true_improvement_rate = float(group["true_improved"].mean())
        true_worsening_rate = float(group["true_worsened"].mean())
        oracle_mae = float(group["optimized_prediction_oracle_error_pp"].abs().mean())
        posthoc_pass = (
            oracle_gain >= float(audit_cfg["min_oracle_gain_pp"])
            and true_improvement_rate >= float(audit_cfg["min_true_improvement_rate"])
            and oracle_mae <= float(audit_cfg["max_prediction_oracle_mae_pp"])
        )
        entry = model_config["models"][model_name]
        rows.append(
            {
                "model": model_name,
                "label": entry.get("label", model_name),
                "selected_model": model_name == recommended_model,
                "selection_reason": (
                    selection_reason if model_name == recommended_model else ""
                ),
                "monotone_guardrail": bool(entry.get("apply_monotone", False)),
                "holdout_pr_auc": float(metric["holdout_pr_auc"]),
                "holdout_brier": float(metric["holdout_brier"]),
                "holdout_ece": float(metric["holdout_ece"]),
                "holdout_calibration_gap_pp": float(
                    metric["holdout_calibration_gap_pp"]
                ),
                "cohort_size": len(group),
                "cohort_actual_yield": float(group["actual_label"].mean()),
                "baseline_predicted_yield": float(
                    group["baseline_predicted_yield"].mean()
                ),
                "optimized_predicted_yield": float(
                    group["optimized_predicted_yield"].mean()
                ),
                "predicted_improvement_pp": float(group["predicted_gain_pp"].mean()),
                "predicted_improvement_ci_low_pp": predicted_ci[0],
                "predicted_improvement_ci_high_pp": predicted_ci[1],
                "baseline_oracle_yield": float(group["baseline_oracle_yield"].mean()),
                "optimized_oracle_yield": float(group["optimized_oracle_yield"].mean()),
                "oracle_improvement_pp": oracle_gain,
                "oracle_improvement_ci_low_pp": oracle_ci[0],
                "oracle_improvement_ci_high_pp": oracle_ci[1],
                "optimized_predicted_90pct_row_rate": float(
                    group["optimized_predicted_90pct_hit"].mean()
                ),
                "optimized_oracle_90pct_row_rate": float(
                    group["optimized_oracle_90pct_hit"].mean()
                ),
                "true_improvement_rate": true_improvement_rate,
                "true_worsening_rate": true_worsening_rate,
                "optimized_prediction_oracle_bias_pp": float(
                    group["optimized_prediction_oracle_error_pp"].mean()
                ),
                "optimized_prediction_oracle_mae_pp": oracle_mae,
                "mean_changed_features": float(group["n_changed"].mean()),
                "posthoc_optimizer_audit_pass": posthoc_pass,
                "posthoc_audit_not_used_for_selection": True,
                "model_selection_rule": SELECTION_RULE,
                "oracle_role": ORACLE_ROLE,
                "causal": False,
                "disclaimer": DISCLAIMER,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["selected_model", "holdout_pr_auc"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def _save_csv(frame: pd.DataFrame, configured_path: str) -> Path:
    path = resolve(configured_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _save_figure(
    summary: pd.DataFrame,
    path: Path,
    *,
    target_yield: float,
    dpi: int,
) -> None:
    models = summary["model"].tolist()
    positions = np.arange(len(models), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.5))

    for axis, prefix, title in (
        (axes[0, 0], "predicted", "Model-predicted yield"),
        (axes[0, 1], "oracle", "Post-hoc synthetic oracle yield"),
    ):
        axis.bar(
            positions - width / 2,
            summary[f"baseline_{prefix}_yield"] * 100.0,
            width,
            label="baseline",
            color="#78909C",
        )
        axis.bar(
            positions + width / 2,
            summary[f"optimized_{prefix}_yield"] * 100.0,
            width,
            label="recommended conditions",
            color="#1565C0",
        )
        axis.axhline(
            target_yield * 100.0,
            color="#C62828",
            linestyle="--",
            label="90% target",
        )
        axis.set_title(title)
        axis.set_ylabel("Mean success probability (%)")
        axis.set_xticks(positions, labels=models, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)

    axes[1, 0].bar(
        positions - width / 2,
        summary["holdout_pr_auc"],
        width,
        label="PR-AUC",
        color="#2E7D32",
    )
    axes[1, 0].bar(
        positions + width / 2,
        summary["holdout_brier"],
        width,
        label="Brier (lower is better)",
        color="#EF6C00",
    )
    axes[1, 0].set_title("Held-out split predictive quality (used for selection)")
    axes[1, 0].set_xticks(positions, labels=models, rotation=20, ha="right")
    axes[1, 0].grid(axis="y", alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        positions - width / 2,
        summary["optimized_oracle_90pct_row_rate"] * 100.0,
        width,
        label="oracle p >= 90%",
        color="#6A1B9A",
    )
    axes[1, 1].bar(
        positions + width / 2,
        summary["true_improvement_rate"] * 100.0,
        width,
        label="oracle improved",
        color="#00838F",
    )
    axes[1, 1].set_title("Post-hoc optimizer audit")
    axes[1, 1].set_ylabel("Cohort rows (%)")
    axes[1, 1].set_xticks(positions, labels=models, rotation=20, ha="right")
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Model Selection and 90% Yield Target Audit", fontsize=15)
    fig.text(
        0.5,
        0.008,
        "Selection uses PR-AUC/Brier only; oracle is post-hoc only. Not causal.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_yield_experiment(
    model_config: dict[str, Any] | None = None,
    experiment_config: dict[str, Any] | None = None,
    *,
    optimize_config: dict[str, Any] | None = None,
    app_config: dict[str, Any] | None = None,
    oracle: OracleFn | None = None,
) -> YieldExperimentResult:
    """고정 holdout cohort와 3개 demo preset 실험을 실행한다."""
    model_cfg = model_config if model_config is not None else load_config("model")
    experiment_cfg = (
        experiment_config
        if experiment_config is not None
        else load_config("yield_experiment")
    )
    optimize_cfg = (
        optimize_config if optimize_config is not None else load_config("optimize")
    )
    app_cfg = app_config if app_config is not None else load_config("app")

    split = make_split(
        test_size=float(model_cfg["test_size"]),
        seed=int(model_cfg["seed"]),
    )
    models, metrics = _fit_models(
        split,
        model_cfg,
        calibration_bins=int(experiment_cfg["calibration_bins"]),
    )
    recommended_model, selection_reason = _select_model_without_optimized_yield(metrics)

    cohort_size = min(int(experiment_cfg["cohort_size"]), len(split.X_test))
    rng = np.random.default_rng(int(experiment_cfg["seed"]))
    positions = rng.choice(len(split.X_test), size=cohort_size, replace=False)
    cohort_conditions = [
        (
            f"holdout_{int(position):04d}",
            {
                feature: float(split.X_test.iloc[position][feature])
                for feature in FEATURE_NAMES
            },
            int(split.y_test.iloc[position]),
        )
        for position in positions
    ]
    presets = app_cfg["demo_presets"]
    preset_conditions = [
        (
            str(preset["name"]),
            {feature: float(preset["values"][feature]) for feature in FEATURE_NAMES},
            None,
        )
        for preset in presets
    ]

    # 이 두 단계가 끝날 때까지 oracle callback은 전달조차 하지 않는다.
    frozen_cohort = _freeze_recommendations(
        models,
        cohort_conditions,
        optimize_cfg,
        group="holdout_cohort",
    )
    frozen_presets = _freeze_recommendations(
        models,
        preset_conditions,
        optimize_cfg,
        group="demo_preset",
    )

    target = float(experiment_cfg["target_yield"])
    audited = _posthoc_oracle_audit(
        [*frozen_cohort, *frozen_presets],
        oracle=oracle,
        target_yield=target,
    )
    cohort_rows = audited[audited["group"] == "holdout_cohort"].reset_index(drop=True)
    preset_rows = audited[audited["group"] == "demo_preset"].reset_index(drop=True)
    preset_rows["selected_model"] = preset_rows["model"] == recommended_model
    primary_presets = preset_rows[preset_rows["selected_model"]].reset_index(drop=True)
    summary = _summarize_cohort(
        cohort_rows,
        metrics,
        model_cfg,
        experiment_cfg,
        recommended_model=recommended_model,
        selection_reason=selection_reason,
    )

    outputs = experiment_cfg["outputs"]
    paths = {
        "summary_csv": _save_csv(summary, outputs["summary_csv"]),
        "cohort_rows_csv": _save_csv(cohort_rows, outputs["cohort_rows_csv"]),
        "preset_comparison_csv": _save_csv(
            preset_rows, outputs["preset_comparison_csv"]
        ),
        "primary_presets_csv": _save_csv(
            primary_presets, outputs["primary_presets_csv"]
        ),
        "protocol_json": resolve(outputs["protocol_json"]),
        "figure": resolve(outputs["figure"]),
    }
    protocol = {
        "model_selection_rule": SELECTION_RULE,
        "recommended_model": recommended_model,
        "selection_reason": selection_reason,
        "cohort_source": (
            "seeded subset from the held-out test split after that split was used "
            "for model selection; not external validation"
        ),
        "cohort_size": cohort_size,
        "optimizer_contract": {
            "function": "yeda.optimize.search.recommend",
            "max_changed_features": int(
                optimize_cfg["constraints"]["max_changed_features"]
            ),
            "max_relative_move": float(
                optimize_cfg["constraints"]["max_relative_move"]
            ),
        },
        "hidden_physics_candidate_generation": False,
        "hidden_physics_optimizer_use": False,
        "oracle_role": ORACLE_ROLE,
        "oracle_used_for_model_selection": False,
        "target_yield": target,
        "theoretical_ood_upper_bound": (
            "excluded from deployable comparison; aggressive nine-variable "
            "oracle optimization would be an OOD theoretical upper bound only"
        ),
        "causal": False,
        "disclaimer": DISCLAIMER,
    }
    save_json(protocol, paths["protocol_json"])
    _save_figure(
        summary,
        paths["figure"],
        target_yield=target,
        dpi=int(experiment_cfg["figure_dpi"]),
    )
    return YieldExperimentResult(
        summary=summary,
        cohort_rows=cohort_rows,
        preset_rows=preset_rows,
        primary_presets=primary_presets,
        recommended_model=recommended_model,
        paths=paths,
    )
