import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
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


def _load_pipeline_from_supabase() -> tuple[Pipeline, dict[int, tuple[float, float]]]:
    """Download registry + artifact from Supabase Storage and load into memory.

    Expects two objects in the ``models`` bucket:
      - ``latest.json``   — registry metadata (path field used as object key)
      - ``<artifact>.joblib`` — the serialised pipeline

    Raises:
        RuntimeError: if the download fails or the bucket is unreachable.
    """
    from supabase import create_client  # noqa: PLC0415

    client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    bucket = settings.supabase_storage_bucket

    logger.info("predict | downloading registry from supabase storage (bucket=%s)", bucket)
    try:
        registry_bytes: bytes = client.storage.from_(bucket).download("latest.json")
        entry = json.loads(registry_bytes)

        # The artifact path stored in the registry is e.g.
        # "models/artifacts/decision_tree_20260407_172358.joblib".
        # In storage we keep just the filename.
        artifact_key = Path(entry["path"]).name
        logger.info(
            "predict | downloading artifact %s (trained %s)",
            artifact_key,
            entry["timestamp"],
        )
        artifact_bytes: bytes = client.storage.from_(bucket).download(artifact_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from Supabase Storage: {exc}") from exc

    artifact = joblib.load(io.BytesIO(artifact_bytes))
    if isinstance(artifact, dict):
        pipeline: Pipeline = artifact["pipeline"]
        leaf_ranges: dict[int, tuple[float, float]] = artifact["leaf_ranges"]
    else:
        pipeline = artifact
        leaf_ranges = {}
        logger.warning("predict | legacy artifact loaded from storage; leaf ranges unavailable")

    logger.info("predict | model loaded from supabase storage | artifact=%s", artifact_key)
    return pipeline, leaf_ranges


def _load_pipeline_from_registry() -> tuple[Pipeline, dict[int, tuple[float, float]]]:
    """Read models/registry/latest.json and load the referenced artifact from disk."""
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
        registry_path = settings.models_registry_path / "latest.json"
        if registry_path.exists():
            _pipeline, _leaf_ranges = _load_pipeline_from_registry()
        else:
            logger.info("predict | local registry not found — loading from Supabase Storage")
            _pipeline, _leaf_ranges = _load_pipeline_from_supabase()
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
