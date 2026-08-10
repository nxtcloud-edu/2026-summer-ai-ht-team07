"""Model loading and inference service for API routers.

This module is the boundary between HTTP handlers and the training artifact.
It preserves the saved :class:`yeda.models.registry.ModelBundle` contract
(feature order, training-set imputation values and success probability) and
keeps the API process available in the expected pre-training state by falling
back to the existing UI mock implementation.

``get_shap_background`` returns model-ready rows only; importing and building
SHAP explainers remains the explainability module's responsibility.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        # ``make`` exports PYTHONPATH=src, but direct uvicorn/test imports do
        # not necessarily do so.  Joblib also needs ``yeda`` to be importable
        # in order to unpickle ModelBundle.
        sys.path.insert(0, str(import_root))

from yeda.data.preprocess import apply_imputer  # noqa: E402
from yeda.io_utils import load_config, resolve  # noqa: E402
from yeda.models.registry import ModelBundle, load_bundle  # noqa: E402
from yeda.schema import FEATURE_NAMES, SPEC_BY_NAME  # noqa: E402


DEFAULT_SHAP_BACKGROUND = PROJECT_ROOT / "artifacts" / "data" / "shap_background.csv"


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve environment/argument paths relative to the repository root."""
    return resolve(Path(value).expanduser())


def _schema_midpoints() -> dict[str, float]:
    """Safe imputation defaults used only before a model is available."""
    return {
        name: (float(SPEC_BY_NAME[name].low) + float(SPEC_BY_NAME[name].high)) / 2.0
        for name in FEATURE_NAMES
    }


class MLService:
    """Load a saved ModelBundle and expose prediction-oriented operations."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        background_path: str | os.PathLike[str] | None = None,
    ) -> None:
        configured_model = model_path or os.getenv("YEDA_MODEL_PATH")
        self.model_path: Path | None = (
            _resolve_path(configured_model) if configured_model else None
        )

        configured_background = (
            background_path
            or os.getenv("YEDA_SHAP_BACKGROUND_PATH")
            or os.getenv("YEDA_DATA_PATH")
        )
        self.background_path: Path | None = (
            _resolve_path(configured_background) if configured_background else None
        )

        self.bundle: ModelBundle | None = None
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.imputer_values: dict[str, float] = _schema_midpoints()
        self.model_name = "mock"
        self.is_loaded = False
        self.is_mock = True
        self.load_error: str | None = None
        self.load_model()

    def _configured_model_path(self) -> Path:
        """Return the canonical configured model path for status reporting."""
        if self.model_path is not None:
            return self.model_path
        config = load_config("model")
        return resolve(config["output"]["model_path"])

    @staticmethod
    def _validate_bundle(bundle: Any) -> ModelBundle:
        required = ("model", "name", "feature_names", "imputer_values", "predict_proba")
        missing = [name for name in required if not hasattr(bundle, name)]
        if missing:
            raise TypeError(f"invalid ModelBundle; missing: {', '.join(missing)}")

        names = list(bundle.feature_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("ModelBundle.feature_names must be non-empty and unique")
        if set(names) != set(FEATURE_NAMES):
            raise ValueError("ModelBundle features do not match schema.FEATURE_NAMES")
        if not callable(bundle.predict_proba):
            raise TypeError("ModelBundle.predict_proba is not callable")
        return bundle

    def load_model(self, model_path: str | os.PathLike[str] | None = None) -> bool:
        """(Re)load the real model, enabling mock mode on any load failure.

        Returning ``False`` is intentional for the normal pre-training state;
        routers can stay online and expose ``load_error`` through a health
        endpoint instead of failing during module import.
        """
        if model_path is not None:
            self.model_path = _resolve_path(model_path)

        self.bundle = None
        self.feature_names = list(FEATURE_NAMES)
        self.imputer_values = _schema_midpoints()
        self.model_name = "mock"
        self.is_loaded = False
        self.is_mock = True
        self.load_error = None

        try:
            if self.model_path is None:
                candidate = self._configured_model_path()
                bundle = load_bundle()
            else:
                candidate = self.model_path
                if not candidate.is_file():
                    raise FileNotFoundError(f"model artifact not found: {candidate}")
                bundle = joblib.load(candidate)

            validated = self._validate_bundle(bundle)
            self.model_path = candidate
            self.bundle = validated
            self.feature_names = list(validated.feature_names)
            self.imputer_values = {
                name: float(validated.imputer_values[name]) for name in self.feature_names
            }
            self.model_name = str(validated.name)
            self.is_loaded = True
            self.is_mock = False
            return True
        except Exception as exc:  # artifact absence/version mismatch must not stop the API
            try:
                self.model_path = self._configured_model_path()
            except Exception:
                pass
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def _to_frame(self, values: Any) -> pd.DataFrame:
        """Normalize supported input shapes without trusting dict insertion order."""
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
        elif isinstance(values, pd.Series):
            frame = values.to_frame().T
        elif isinstance(values, Mapping):
            frame = pd.DataFrame([dict(values)])
        elif isinstance(values, np.ndarray):
            array = values.reshape(1, -1) if values.ndim == 1 else values
            if array.ndim != 2 or array.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"array input must have {len(self.feature_names)} columns in model order"
                )
            frame = pd.DataFrame(array, columns=self.feature_names)
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) == 0:
                raise ValueError("prediction input must contain at least one row")
            if all(isinstance(row, Mapping) for row in values):
                frame = pd.DataFrame([dict(row) for row in values])
            else:
                if len(values) != len(self.feature_names):
                    raise ValueError(
                        f"ordered input must contain {len(self.feature_names)} values"
                    )
                frame = pd.DataFrame([list(values)], columns=self.feature_names)
        else:
            raise TypeError("input must be a mapping, Series, DataFrame, or ordered sequence")

        if frame.empty:
            raise ValueError("prediction input must contain at least one row")

        numeric = pd.DataFrame(index=frame.index)
        for name in self.feature_names:
            source = frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)
            numeric[name] = pd.to_numeric(source, errors="coerce")
        return apply_imputer(numeric, self.imputer_values)

    @staticmethod
    def _validate_probabilities(values: Any, n_rows: int) -> np.ndarray:
        probabilities = np.asarray(values, dtype=float).reshape(-1)
        if len(probabilities) != n_rows:
            raise ValueError(
                f"predict_proba returned {len(probabilities)} values for {n_rows} rows"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("predict_proba returned a non-finite success probability")
        return np.clip(probabilities, 0.0, 1.0)

    def predict_proba(self, values: Any) -> np.ndarray:
        """Return one pickup-success probability per input row (0..1)."""
        frame = self._to_frame(values)
        if self.is_mock:
            # Keep the API fallback numerically identical to the Streamlit
            # fallback rather than maintaining a second mock formula.
            from app.components.mock_backend import predict_proba as mock_predict_proba

            raw = mock_predict_proba(frame)
        else:
            assert self.bundle is not None
            raw = self.bundle.predict_proba(frame[self.feature_names])
        return self._validate_probabilities(raw, len(frame))

    def predict(self, values: Any) -> float | list[float]:
        """Return success probability; batches use a JSON-serializable list."""
        probabilities = self.predict_proba(values)
        if len(probabilities) == 1:
            return float(probabilities[0])
        return probabilities.astype(float).tolist()

    def _background_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        if self.background_path is not None:
            candidates.append(self.background_path)
        candidates.append(DEFAULT_SHAP_BACKGROUND)

        try:
            candidates.append(resolve(load_config("data_gen")["output_path"]))
        except Exception:
            pass

        unique: list[Path] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    def _read_background(self, path: Path, max_rows: int) -> pd.DataFrame | None:
        if not path.is_file():
            return None
        try:
            if path.suffix.lower() in {".parquet", ".pq"}:
                raw = pd.read_parquet(path, columns=self.feature_names).head(max_rows)
            else:
                raw = pd.read_csv(
                    path,
                    usecols=lambda column: column in self.feature_names,
                    nrows=max_rows,
                )
        except (OSError, ValueError, ImportError):
            return None
        if raw.empty or not any(name in raw.columns for name in self.feature_names):
            return None
        return self._to_frame(raw)

    def get_shap_background(
        self,
        n_samples: int = 200,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """Return deterministic, imputed background rows in bundle feature order.

        Discovery order is an explicit/env override, a preprocessed training
        artifact when present, then ``configs/data_gen.yaml:output_path``.
        If none exists, the bundle's training medians form a safe one-row
        background without importing SHAP or the data-generation physics.
        """
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")

        max_rows = max(n_samples, min(10_000, n_samples * 20))
        for path in self._background_candidates():
            background = self._read_background(path, max_rows=max_rows)
            if background is None:
                continue
            if len(background) > n_samples:
                background = background.sample(n=n_samples, random_state=random_state)
            return background[self.feature_names].reset_index(drop=True)

        median_row = {name: self.imputer_values[name] for name in self.feature_names}
        return pd.DataFrame([median_row], columns=self.feature_names)

    def status(self) -> dict[str, Any]:
        """Return JSON-safe state for ``/api/health`` and the frontend badge."""
        return {
            "is_loaded": self.is_loaded,
            "is_mock": self.is_mock,
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "load_error": self.load_error,
        }


# Process-wide default used by straightforward router imports.  The class is
# also public so tests can isolate model paths without mutating global state.
ml_service = MLService()
is_loaded = ml_service.is_loaded
is_mock = ml_service.is_mock


def reload_model(model_path: str | os.PathLike[str] | None = None) -> bool:
    """Reload the default service and refresh its module-level status flags."""
    loaded = ml_service.load_model(model_path)
    global is_loaded, is_mock
    is_loaded = ml_service.is_loaded
    is_mock = ml_service.is_mock
    return loaded


def predict(values: Any) -> float | list[float]:
    """Module-level success-probability API used by prediction routers."""
    return ml_service.predict(values)


def predict_proba(values: Any) -> np.ndarray:
    """Module-level batch probability API for explain/optimize consumers."""
    return ml_service.predict_proba(values)


def get_shap_background(
    n_samples: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """Module-level SHAP background-data API."""
    return ml_service.get_shap_background(n_samples, random_state)


def get_model_status() -> dict[str, Any]:
    """Return the default service's current state."""
    return ml_service.status()


__all__ = [
    "MLService",
    "get_model_status",
    "get_shap_background",
    "is_loaded",
    "is_mock",
    "ml_service",
    "predict",
    "predict_proba",
    "reload_model",
]
