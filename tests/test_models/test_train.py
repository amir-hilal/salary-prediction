import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.train import compute_leaf_ranges, load_pipeline, save_pipeline, train


# ── Synthetic data ────────────────────────────────────────────────────────────


def _make_synthetic_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Return a small DataFrame with valid integer-encoded features and a target."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "experience_level": rng.integers(0, 4, size=n),
        "employment_type":  rng.integers(0, 4, size=n),
        "remote_ratio":     rng.choice([0, 50, 100], size=n),
        "company_size":     rng.integers(0, 3, size=n),
        "work_year":        rng.integers(2020, 2025, size=n),
        "job_family":       rng.integers(0, 6, size=n),
        "location_region":  rng.integers(0, 4, size=n),
        "is_us_company":    rng.integers(0, 2, size=n),
        TARGET_COLUMN:      rng.integers(50_000, 200_000, size=n),
    })


# Train once per module to keep the suite fast (GridSearchCV on 80 rows).
@pytest.fixture(scope="module")
def trained() -> tuple[Pipeline, dict, dict, pd.DataFrame]:
    df = _make_synthetic_df()
    pipeline, best_params = train(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    leaf_ranges = compute_leaf_ranges(pipeline, df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    return pipeline, best_params, leaf_ranges, df


# ── train() ───────────────────────────────────────────────────────────────────


def test_train_returns_pipeline(trained: tuple) -> None:
    pipeline, _, _, _ = trained
    assert isinstance(pipeline, Pipeline)


def test_train_returns_best_params_dict(trained: tuple) -> None:
    _, best_params, _, _ = trained
    assert isinstance(best_params, dict)


def test_train_best_params_includes_cv_rmse(trained: tuple) -> None:
    _, best_params, _, _ = trained
    assert "cv_rmse" in best_params
    assert best_params["cv_rmse"] > 0


def test_train_best_params_includes_tree_hyperparams(trained: tuple) -> None:
    _, best_params, _, _ = trained
    assert "model__max_depth" in best_params
    assert "model__min_samples_split" in best_params
    assert "model__min_samples_leaf" in best_params


def test_train_pipeline_can_predict(trained: tuple) -> None:
    pipeline, _, _, df = trained
    predictions = pipeline.predict(df[FEATURE_COLUMNS])
    assert len(predictions) == len(df)


def test_train_predictions_are_positive(trained: tuple) -> None:
    pipeline, _, _, df = trained
    predictions = pipeline.predict(df[FEATURE_COLUMNS])
    assert all(p > 0 for p in predictions)


def test_train_raises_on_missing_feature_column() -> None:
    df = _make_synthetic_df()
    X_incomplete = df[FEATURE_COLUMNS].drop(columns=["experience_level"])
    with pytest.raises(ValueError, match="missing required feature columns"):
        train(X_incomplete, df[TARGET_COLUMN])


# ── compute_leaf_ranges ───────────────────────────────────────────────────────


def test_compute_leaf_ranges_returns_dict(trained: tuple) -> None:
    pipeline, _, leaf_ranges, df = trained
    assert isinstance(leaf_ranges, dict)


def test_compute_leaf_ranges_all_leaves_covered(trained: tuple) -> None:
    pipeline, _, leaf_ranges, df = trained
    dt = pipeline.named_steps["model"]
    X_transformed = pipeline[:-1].transform(df[FEATURE_COLUMNS])
    used_leaves = set(dt.apply(X_transformed).tolist())
    assert used_leaves <= set(leaf_ranges.keys())


def test_compute_leaf_ranges_q25_le_q75(trained: tuple) -> None:
    _, _, leaf_ranges, _ = trained
    for leaf_id, (low, high) in leaf_ranges.items():
        assert low <= high, f"leaf {leaf_id}: Q25={low} > Q75={high}"


# ── save_pipeline / load_pipeline ─────────────────────────────────────────────


def test_save_pipeline_creates_file(trained: tuple, tmp_path: pytest.TempPathFactory) -> None:
    pipeline, _, leaf_ranges, _ = trained
    artifact_path = save_pipeline(pipeline, leaf_ranges, tmp_path)
    assert artifact_path.exists()


def test_save_pipeline_uses_joblib_extension(trained: tuple, tmp_path: pytest.TempPathFactory) -> None:
    pipeline, _, leaf_ranges, _ = trained
    artifact_path = save_pipeline(pipeline, leaf_ranges, tmp_path)
    assert artifact_path.suffix == ".joblib"


def test_load_pipeline_roundtrip_produces_same_predictions(
    trained: tuple, tmp_path: pytest.TempPathFactory
) -> None:
    pipeline, _, leaf_ranges, df = trained
    artifact_path = save_pipeline(pipeline, leaf_ranges, tmp_path)
    loaded_pipeline, loaded_ranges = load_pipeline(artifact_path)

    original_pred = pipeline.predict(df[FEATURE_COLUMNS][:5])
    loaded_pred = loaded_pipeline.predict(df[FEATURE_COLUMNS][:5])
    assert list(original_pred) == pytest.approx(list(loaded_pred))


def test_load_pipeline_roundtrip_preserves_leaf_ranges(
    trained: tuple, tmp_path: pytest.TempPathFactory
) -> None:
    pipeline, _, leaf_ranges, _ = trained
    artifact_path = save_pipeline(pipeline, leaf_ranges, tmp_path)
    _, loaded_ranges = load_pipeline(artifact_path)
    assert loaded_ranges == leaf_ranges
