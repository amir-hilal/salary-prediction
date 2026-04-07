import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

# Features scaled by RobustScaler — chosen because salary-related data has outliers.
# Tree-based models don't require scaling, but it costs nothing and keeps the
# pipeline compatible with linear baselines too.
_NUMERIC_FEATURES: list[str] = ["remote_ratio", "work_year"]


@dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[TrainTestSplit, RobustScaler]:
    """Split into train/test and fit a RobustScaler on X_train numeric features.

    The scaler is fit on training data only to prevent leakage.
    Returns the split and the fitted scaler (persist it alongside the model).
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = RobustScaler()
    X_train[_NUMERIC_FEATURES] = scaler.fit_transform(X_train[_NUMERIC_FEATURES])
    X_test[_NUMERIC_FEATURES] = scaler.transform(X_test[_NUMERIC_FEATURES])

    logger.info(
        "split_and_scale | X_train=%s X_test=%s | scaled_cols=%s",
        X_train.shape,
        X_test.shape,
        _NUMERIC_FEATURES,
    )

    return TrainTestSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test), scaler


def save_scaler(scaler: RobustScaler, path: Path) -> None:
    """Persist the fitted scaler so inference can apply the same transformation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("save_scaler | saved to %s", path)


def load_scaler(path: Path) -> RobustScaler:
    """Load a previously persisted scaler."""
    scaler: RobustScaler = joblib.load(path)
    logger.info("load_scaler | loaded from %s", path)
    return scaler
