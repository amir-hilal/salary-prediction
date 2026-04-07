import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
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


    # GridSearchCV is not a Decision Tree (DT). It is a hyperparameter optimization 
    # tool used to find the best settings for many different types of machine learning 
    # models, including Decision Trees.
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


def compute_leaf_ranges(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[int, tuple[float, float]]:
    """Compute Q25–Q75 of training targets for each decision tree leaf node.

    At prediction time, the leaf that a new sample lands in is looked up here
    to return a salary range that reflects real variation in that peer group.
    """
    dt = pipeline.named_steps["model"]
    # Apply every pipeline step except the final estimator so we get the same
    # representation the DT was fitted on.
    X_transformed = pipeline[:-1].transform(X_train[FEATURE_COLUMNS])
    leaf_ids = dt.apply(X_transformed)

    y_values = y_train.to_numpy()
    ranges: dict[int, tuple[float, float]] = {}
    for leaf_id in np.unique(leaf_ids):
        samples = y_values[leaf_ids == leaf_id]
        ranges[int(leaf_id)] = (
            float(np.percentile(samples, 25)),
            float(np.percentile(samples, 75)),
        )
    logger.info("compute_leaf_ranges | leaves=%d", len(ranges))
    return ranges


def save_pipeline(
    pipeline: Pipeline,
    leaf_ranges: dict[int, tuple[float, float]],
    artifacts_dir: Path,
) -> Path:
    """Persist the fitted pipeline and its leaf ranges to models/artifacts/."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_path = artifacts_dir / f"decision_tree_{timestamp}.joblib"
    joblib.dump({"pipeline": pipeline, "leaf_ranges": leaf_ranges}, artifact_path)
    logger.info("save_pipeline | saved to %s", artifact_path)
    return artifact_path


def load_pipeline(path: Path) -> tuple[Pipeline, dict[int, tuple[float, float]]]:
    """Load a previously saved pipeline and leaf ranges from disk."""
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        pipeline: Pipeline = artifact["pipeline"]
        leaf_ranges: dict[int, tuple[float, float]] = artifact["leaf_ranges"]
    else:
        # Legacy format — plain Pipeline saved without leaf ranges.
        pipeline = artifact
        leaf_ranges = {}
        logger.warning("load_pipeline | legacy artifact; salary ranges unavailable")
    logger.info("load_pipeline | loaded from %s", path)
    return pipeline, leaf_ranges


if __name__ == "__main__":
    import logging as _logging

    from src.data.cleaning import clean
    from src.data.ingestion import load_raw
    from src.data.preprocessing import split_and_scale
    from src.features.engineering import build_features
    from src.models.evaluate import evaluate

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("── Step 1: Load raw data")
    df = load_raw(settings.data_raw_path)

    print("── Step 2: Clean")
    df = clean(df, iqr_cap_factor=settings.iqr_cap_factor)

    print("── Step 3: Feature engineering")
    df = build_features(df)

    print("── Step 4: Train/test split")
    split, _ = split_and_scale(df, test_size=settings.test_size, random_state=settings.random_state)

    print("── Step 5: Train (GridSearchCV — this may take a minute)")
    pipeline, best_params = train(split.X_train, split.y_train)

    print("── Step 5b: Compute leaf ranges for salary range output")
    leaf_ranges = compute_leaf_ranges(pipeline, split.X_train, split.y_train)

    print("── Step 6: Save artifact")
    artifact_path = save_pipeline(pipeline, leaf_ranges, settings.models_artifacts_path)

    print("── Step 7: Evaluate on test set")
    metrics = evaluate(pipeline, split.X_test, split.y_test, artifact_path, best_params)

    print("\n✅ Training complete")
    print(f"   RMSE : ${metrics['rmse']:,.0f}")
    print(f"   MAE  : ${metrics['mae']:,.0f}")
    print(f"   R²   : {metrics['r2']:.3f}")
    print(f"   MAPE : {metrics['mape']:.1f}%")
    print(f"   Best params: {best_params}")
