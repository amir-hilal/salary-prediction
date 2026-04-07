---
description: "Use when working on model training, evaluation, hyperparameter tuning, or serialisation. Covers scikit-learn pipelines, metric logging, artifact saving, and model registry in src/models/."
applyTo: "src/models/**,notebooks/04_model_training.ipynb"
---

# Model Training Instructions

## Training (`src/models/train.py`)
- Wrap preprocessing + model in a single `sklearn.pipeline.Pipeline` so inference requires the same object as training
- Use `cross_val_score` or `GridSearchCV` / `RandomizedSearchCV` for hyperparameter tuning
- Log all hyperparameters and CV scores at INFO level
- Accept config values (test split ratio, random seed, hyperparameter grid) from `config/settings.py` — no magic numbers in training code

## Evaluation (`src/models/evaluate.py`)
- Always report: RMSE, MAE, R², and MAPE
- Produce a residual plot and a predicted-vs-actual scatter; save them to `models/artifacts/`
- Persist a JSON evaluation report alongside the model file
- If the new model is worse than the previous registry entry on RMSE, log a WARNING and do not overwrite the artifact automatically

## Artifact Management
- Save models with `joblib.dump` to `models/artifacts/<model_name>_<timestamp>.joblib`
- Write a registry entry to `models/registry/latest.json`: `{name, path, timestamp, metrics}`
- Load the model in `src/models/predict.py` by reading `models/registry/latest.json` — no hardcoded paths
- The fitted scaler from preprocessing must be saved inside the same pipeline object

## Inference (`src/models/predict.py`)
- Expose a single `predict(features: dict) -> float` function — the API route calls only this
- Validate input features against `FEATURE_COLUMNS` before inference; raise `ValueError` on mismatch
- Load the pipeline once at module import time (module-level singleton) — not on every call

## General Rules
- Do not train in `src/api/` — the API only calls `predict.py`
- All randomness must be seeded via `settings.random_seed`
- Write tests in `tests/test_models/` using a small synthetic dataset — do not use real data in tests
