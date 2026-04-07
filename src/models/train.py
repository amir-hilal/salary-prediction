import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor

from config.settings import settings
from src.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def _build_pipeline(random_state: int) -> Pipeline:
    """Build a Pipeline with RobustScaler + DecisionTreeRegressor.

    Scaling is a no-op for trees but keeps the pipeline compatible with
    linear baselines and ensures the scaler is persisted inside the artifact.
    """
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", DecisionTreeRegressor(random_state=random_state)),
    ])


def _param_grid() -> dict:
    """Return the hyperparameter search grid for GridSearchCV.

    Parameter names use the sklearn Pipeline convention: <step>__<param>.
    """
    return {
        "model__max_depth": settings.dt_max_depth_options,
        "model__min_samples_split": settings.dt_min_samples_split_options,
        "model__min_samples_leaf": settings.dt_min_samples_leaf_options,
    }


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, dict]:
    """Fit a Decision Tree pipeline with cross-validated hyperparameter search.

    Returns the best fitted pipeline and a dict of the best hyperparameters
    with their CV score.
    """
    missing = set(FEATURE_COLUMNS) - set(X_train.columns)
    if missing:
        raise ValueError(f"X_train is missing required feature columns: {missing}")

    pipeline = _build_pipeline(random_state=settings.random_state)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=_param_grid(),
        scoring="neg_root_mean_squared_error",
        cv=settings.dt_cv_folds,
        n_jobs=-1,
        refit=True,
    )

    logger.info("train | starting GridSearchCV | cv=%d", settings.dt_cv_folds)
    search.fit(X_train[FEATURE_COLUMNS], y_train)

    best_params = search.best_params_
    cv_rmse = -search.best_score_

    logger.info(
        "train | best_params=%s | cv_rmse=%.2f",
        best_params,
        cv_rmse,
    )

    return search.best_estimator_, {**best_params, "cv_rmse": cv_rmse}


def save_pipeline(pipeline: Pipeline, artifacts_dir: Path) -> Path:
    """Persist the fitted pipeline to models/artifacts/ with a timestamp suffix."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_path = artifacts_dir / f"decision_tree_{timestamp}.joblib"
    joblib.dump(pipeline, artifact_path)
    logger.info("save_pipeline | saved to %s", artifact_path)
    return artifact_path


def load_pipeline(path: Path) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    pipeline: Pipeline = joblib.load(path)
    logger.info("load_pipeline | loaded from %s", path)
    return pipeline
