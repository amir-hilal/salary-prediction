---
description: "Use when working on data ingestion, EDA, cleaning, or feature engineering. Covers Kaggle download, pandas transformations, feature selection, and train/test split in src/data/ and src/features/."
applyTo: "src/data/**,src/features/**,notebooks/**"
---

# Data Pipeline Instructions

## Kaggle Ingestion (`src/data/ingestion.py`)
- Use the `kaggle` Python package or CLI; read credentials from environment variables `KAGGLE_USERNAME` / `KAGGLE_KEY` (never hardcode)
- Download to `data/raw/`; validate schema immediately after download using Pydantic models
- Log dataset shape and dtypes at INFO level after loading
- Raise a descriptive `ValueError` if required columns are missing

## EDA (`notebooks/eda.ipynb`)
- EDA lives in notebooks only — do not import notebook cells into `src/`
- Document every distribution finding as a markdown cell; justify every drop/keep decision
- Use `pandas-profiling` or manual describe/value_counts for initial overview

## Cleaning (`src/data/cleaning.py`)
- Handle missing values explicitly: drop, impute with median/mode, or flag — document the choice
- Cap numeric outliers using IQR; log the count of capped values
- Never mutate the raw DataFrame in place — always return a new DataFrame
- Preserve a record of dropped rows for auditability

## Feature Engineering (`src/features/engineering.py`)
- Create interaction terms and encoded categoricals here, not inside the model pipeline
- Use `sklearn.preprocessing` encoders to ensure train/test consistency
- Keep a `FEATURE_COLUMNS` list constant at the top of the file — used by both training and inference
- Log feature shapes before and after engineering

## Preprocessing (`src/data/preprocessing.py`)
- Perform scaling (StandardScaler / RobustScaler) and train/test split here
- Use `random_state=42` for reproducibility unless overridden by config
- Persist the fitted scaler alongside the model artifact so inference can reuse it
- Return a typed `TrainTestSplit` dataclass or named tuple — no bare tuples

## Notebook Naming
- Notebooks use simple, descriptive names with no numeric prefixes: `eda.ipynb`, `feature_engineering.ipynb`, `model_training.ipynb`
- Name reflects the stage, not an order number

## General Rules
- All functions accept and return `pd.DataFrame` — no implicit index reliance
- Log each pipeline stage with stage name, input shape, and output shape
- Write unit tests in `tests/test_data/` that exercise cleaning with deliberately dirty fixtures
