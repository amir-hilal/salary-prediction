import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config.settings import settings
from src.features.engineering import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R², and MAPE."""
    residuals = y_true.values - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true.values - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    # MAPE: exclude zero-target rows to avoid division by zero
    nonzero = y_true.values != 0
    mape = float(np.mean(np.abs(residuals[nonzero] / y_true.values[nonzero])) * 100)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
    logger.info("compute_metrics | %s", metrics)
    return metrics


def save_evaluation_plots(
    y_true: pd.Series,
    y_pred: np.ndarray,
    artifacts_dir: Path,
    timestamp: str,
) -> None:
    """Save residual plot and predicted-vs-actual scatter to artifacts_dir."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="steelblue")
    axes[0].axhline(0, color="crimson", linewidth=1, linestyle="--")
    axes[0].set_title("Residuals vs Predicted", fontweight="bold")
    axes[0].set_xlabel("Predicted salary (USD)")
    axes[0].set_ylabel("Residual (actual − predicted)")

    # Predicted vs actual
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
    axes[1].plot([min_val, max_val], [min_val, max_val], color="crimson",
                 linewidth=1, linestyle="--", label="Perfect prediction")
    axes[1].set_title("Predicted vs Actual", fontweight="bold")
    axes[1].set_xlabel("Actual salary (USD)")
    axes[1].set_ylabel("Predicted salary (USD)")
    axes[1].legend()

    plt.tight_layout()
    plot_path = artifacts_dir / f"eval_plots_{timestamp}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    logger.info("save_evaluation_plots | saved to %s", plot_path)


def save_registry_entry(
    artifact_path: Path,
    metrics: dict,
    best_params: dict,
    registry_dir: Path,
    timestamp: str,
) -> Path:
    """Write models/registry/latest.json with name, path, timestamp, and metrics.

    If an existing registry entry has a lower RMSE, we log a WARNING and still
    write the new entry — the caller decides whether to promote the artifact.
    """
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "latest.json"

    if registry_path.exists():
        existing = json.loads(registry_path.read_text())
        existing_rmse = existing.get("metrics", {}).get("rmse", float("inf"))
        if metrics["rmse"] > existing_rmse:
            logger.warning(
                "save_registry_entry | new RMSE %.2f is WORSE than existing %.2f — "
                "writing registry but review before promoting artifact",
                metrics["rmse"],
                existing_rmse,
            )

    entry = {
        "name": "decision_tree",
        "path": str(artifact_path),
        "timestamp": timestamp,
        "metrics": metrics,
        "best_params": best_params,
    }
    registry_path.write_text(json.dumps(entry, indent=2))
    logger.info("save_registry_entry | written to %s", registry_path)
    return registry_path


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_path: Path,
    best_params: dict,
) -> dict:
    """Run full evaluation: metrics + plots + registry entry. Returns the metrics dict."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = settings.models_artifacts_path
    registry_dir = settings.models_registry_path

    y_pred = pipeline.predict(X_test[FEATURE_COLUMNS])
    metrics = compute_metrics(y_test, y_pred)
    save_evaluation_plots(y_test, y_pred, artifacts_dir, timestamp)
    save_registry_entry(artifact_path, metrics, best_params, registry_dir, timestamp)

    return metrics
