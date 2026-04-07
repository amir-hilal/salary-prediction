import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from config.settings import settings
from src.features.engineering import FEATURE_COLUMNS
from src.models.train import load_pipeline

logger = logging.getLogger(__name__)


def _load_pipeline_from_registry() -> Pipeline:
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


# Module-level singleton — loaded once on first import, reused for every call.
_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline_from_registry()
    return _pipeline


def predict(features: dict) -> float:
    """Predict salary_in_usd for a single candidate.

    Args:
        features: dict mapping each name in FEATURE_COLUMNS to its value.

    Returns:
        Predicted salary in USD as a float.

    Raises:
        ValueError: if any required feature column is missing from `features`.
    """
    missing = set(FEATURE_COLUMNS) - set(features.keys())
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    pipeline = _get_pipeline()
    prediction: float = float(pipeline.predict(row)[0])

    logger.info("predict | features=%s | prediction=%.2f", features, prediction)
    return prediction
