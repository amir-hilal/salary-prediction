import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from config.settings import settings
from src.features.engineering import FEATURE_COLUMNS
from src.models.train import load_pipeline

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Salary prediction with a Q25–Q75 range from the leaf node's training data."""

    point_estimate: float
    range_low: float
    range_high: float


def _load_pipeline_from_registry() -> tuple[Pipeline, dict[int, tuple[float, float]]]:
    """Read models/registry/latest.json and load the referenced artifact."""
    registry_path = settings.models_registry_path / "latest.json"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No model registry found at {registry_path}. "
            "Run the training pipeline first."
        )
    entry = json.loads(registry_path.read_text())
    artifact_path = Path(entry["path"])
    logger.info(
        "predict | loading model '%s' from %s (trained %s)",
        entry["name"],
        artifact_path,
        entry["timestamp"],
    )
    return load_pipeline(artifact_path)


# Module-level singletons — loaded once on first import, reused for every call.
_pipeline: Pipeline | None = None
_leaf_ranges: dict[int, tuple[float, float]] | None = None


def _get_pipeline() -> tuple[Pipeline, dict[int, tuple[float, float]]]:
    global _pipeline, _leaf_ranges
    if _pipeline is None:
        _pipeline, _leaf_ranges = _load_pipeline_from_registry()
    return _pipeline, _leaf_ranges or {}


def predict(features: dict) -> PredictionResult:
    """Predict salary_in_usd with a Q25–Q75 range for a single candidate.

    Args:
        features: dict mapping each name in FEATURE_COLUMNS to its value.

    Returns:
        PredictionResult with point_estimate, range_low, and range_high in USD.

    Raises:
        ValueError: if any required feature column is missing from `features`.
    """
    missing = set(FEATURE_COLUMNS) - set(features.keys())
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    pipeline, leaf_ranges = _get_pipeline()

    point_estimate = float(pipeline.predict(row)[0])

    # Find the leaf this sample lands in and look up pre-computed salary range.
    X_transformed = pipeline[:-1].transform(row)
    leaf_id = int(pipeline.named_steps["model"].apply(X_transformed)[0])
    range_low, range_high = leaf_ranges.get(leaf_id, (point_estimate, point_estimate))

    logger.info(
        "predict | prediction_estimate=%.2f | range=(%.2f, %.2f)",
        point_estimate,
        range_low,
        range_high,
    )
    return PredictionResult(
        point_estimate=point_estimate,
        range_low=range_low,
        range_high=range_high,
    )
