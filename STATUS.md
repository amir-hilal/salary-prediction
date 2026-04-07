# Project Status

> Update this file as work progresses. Mark items ✅ when done, 🚧 when in progress, ⬜ when not started.

---

## 1 · Data Pipeline

| File | Status | Notes |
|------|--------|-------|
| `notebooks/eda.ipynb` | ✅ | EDA complete — findings, correlation analysis, feature decisions documented |
| `config/settings.py` | ✅ | Pydantic BaseSettings — paths, test_size, random_state, iqr_cap_factor |
| `src/data/ingestion.py` | ✅ | load_raw() — schema validation, drops Unnamed: 0 |
| `src/data/cleaning.py` | ✅ | drop_leakage_columns(), cap_salary_outliers() |
| `src/features/engineering.py` | ✅ | job_family, location_region, is_us_company, ordinal encoding; FEATURE_COLUMNS defined |
| `src/data/preprocessing.py` | ✅ | split_and_scale() → TrainTestSplit + RobustScaler; save/load scaler |

---

## 2 · Model Training

| File | Status | Notes |
|------|--------|-------|
| `src/models/train.py` | ✅ | Decision Tree pipeline, GridSearchCV over max_depth / min_samples_split / min_samples_leaf |
| `src/models/evaluate.py` | ✅ | RMSE, MAE, R², MAPE; residual + predicted-vs-actual plots; registry entry with RMSE regression guard |
| `src/models/predict.py` | ✅ | predict(features) singleton loaded from registry/latest.json |
| `models/registry/latest.json` | ⬜ | Written after first training run |

---

## 3 · FastAPI

| File | Status | Notes |
|------|--------|-------|
| `src/api/schemas/salary.py` | ⬜ | PredictionRequest, PredictionResponse, ErrorResponse |
| `src/api/routes/prediction.py` | ⬜ | POST /predict, GET /health |
| `src/api/main.py` | ⬜ | App setup, lifespan, CORS, logging |

---

## 4 · LLM Integration

| File | Status | Notes |
|------|--------|-------|
| `src/llm/ollama_client.py` | ⬜ | httpx async client, generate(), OllamaError |
| `src/llm/narrative.py` | ⬜ | build_prompt(), parse_narrative(), NarrativeResult, ChartSpec |

---

## 5 · Visualizations

| File | Status | Notes |
|------|--------|-------|
| `src/visualizations/charts.py` | ⬜ | salary_histogram(), predicted_vs_actual_scatter(), feature_importance_bar() |

---

## 6 · Database (Supabase)

| File | Status | Notes |
|------|--------|-------|
| `deployment/scripts/setup_supabase.sql` | ⬜ | predictions + narratives tables, RLS policies, indexes |
| `src/database/client.py` | ⬜ | Supabase client singleton |
| `src/database/crud.py` | ⬜ | insert_prediction(), insert_narrative(), get_recent_predictions(), get_narrative_for_prediction() |

---

## 7 · Dashboard (Streamlit)

| File | Status | Notes |
|------|--------|-------|
| `dashboard/app.py` | ⬜ | Entry point, session_state init, Supabase client |
| `dashboard/components/filters.py` | ⬜ | render_sidebar_filters() → FilterState |
| `dashboard/components/charts.py` | ⬜ | Plotly wrappers consuming ChartSpec |
| `dashboard/pages/1_overview.py` | ⬜ | Salary landscape, auto-refresh |
| `dashboard/pages/2_predictions.py` | ⬜ | Prediction form, LLM narrative display |
| `dashboard/pages/3_insights.py` | ⬜ | Narrative list, comparative chart |

---

## 8 · Tests

| File | Status | Notes |
|------|--------|-------|
| `tests/test_data/test_ingestion.py` | ⬜ | load_raw() with dirty fixture |
| `tests/test_data/test_cleaning.py` | ⬜ | Outlier capping, leakage column removal |
| `tests/test_data/test_engineering.py` | ⬜ | job_family mapping, location_region mapping |
| `tests/test_models/test_train.py` | ⬜ | Training on synthetic data |
| `tests/test_api/test_prediction.py` | ⬜ | POST /predict happy path, 422, 500 |
| `tests/test_llm/test_narrative.py` | ⬜ | Mocked Ollama client, parse_narrative() |

---

## 9 · Deployment

| File | Status | Notes |
|------|--------|-------|
| `deployment/docker/Dockerfile.api` | ⬜ | |
| `deployment/docker/Dockerfile.dashboard` | ⬜ | |
| `deployment/docker/docker-compose.yml` | ⬜ | API + dashboard + Ollama |
| `.env.example` | ⬜ | All required env vars documented |

---

## Decisions Log

| Decision | Choice | Reason |
|----------|--------|--------|
| Target column | `salary_in_usd` | Comparable across currencies; `salary` + `salary_currency` are leakage |
| Dropped columns | `salary`, `salary_currency`, `employee_residence` | Leakage / redundant with `company_location` |
| Outlier treatment | IQR cap (1.5×) | Preserves records; extreme salaries are real but skew training |
| `job_title` encoding | keyword → `job_family` (6 groups) | High-cardinality nominal; grouping by function reduces sparsity |
| `company_location` encoding | map → `location_region` (4 regions) + `is_us_company` | Geography matters for salary; 80+ codes too sparse individually |
| Scaler | `RobustScaler` | Salary data has outliers; RobustScaler is IQR-based, more stable than StandardScaler |
| Model | Decision Tree Regressor | Project requirement; tree-based models handle our ordinal/label-encoded features well without scaling |
