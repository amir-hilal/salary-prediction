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
| `models/registry/latest.json` | ✅ | Written — RMSE=$42,564, R²=0.453, max_depth=5 |

---

## 3 · FastAPI

| File | Status | Notes |
|------|--------|-------|
| `src/api/schemas/salary.py` | ✅ | PredictionRequest (8 fields, validated), PredictionResponse, ErrorResponse |
| `src/api/routes/prediction.py` | ✅ | POST /api/v1/predict, GET /api/v1/health; BackgroundTask for DB insert |
| `src/api/main.py` | ✅ | Lifespan (warms model singleton), CORS, global 500 handler, logging |

### Sample `/predict` Request Body

```json
{
  "experience_level": 2,
  "employment_type": 3,
  "remote_ratio": 100,
  "company_size": 1,
  "work_year": 2024,
  "job_family": 2,
  "location_region": 3,
  "is_us_company": 1
}
```

| Field | Type | Values |
|-------|------|--------|
| `experience_level` | int | 0 = Entry-level · 1 = Mid-level · 2 = Senior · 3 = Executive |
| `employment_type` | int | 0 = Freelance · 1 = Part-time · 2 = Contract · 3 = Full-time |
| `remote_ratio` | int | 0 (on-site) · 50 (hybrid) · 100 (fully remote) |
| `company_size` | int | 0 = Small · 1 = Medium · 2 = Large |
| `work_year` | int | Calendar year, e.g. `2024` |
| `job_family` | int | 0 = Other · 1 = Analytics · 2 = Data Science · 3 = Data Engineering · 4 = ML/AI · 5 = Leadership |
| `location_region` | int | 0 = Rest of World · 1 = Asia Pacific · 2 = Europe · 3 = North America |
| `is_us_company` | int | 0 = Non-US company · 1 = US company |

> The body above represents a **Senior Data Scientist**, full-time, fully remote, **US-based medium company, North America**.

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
| `tests/test_data/test_cleaning.py` | ✅ | 12 tests — leakage drops, IQR cap, no-mutation, idempotency |
| `tests/test_data/test_engineering.py` | ✅ | 35 tests — job_family (11 titles), location_region (8 countries), ordinal encoding, build_features |
| `tests/test_models/test_train.py` | ✅ | 10 tests — pipeline shape, predict output, error on missing cols, save/load roundtrip |
| `tests/test_api/test_prediction.py` | ✅ | 14 tests — happy path, 4× 422 validation, 2× 500 safe message, health check |
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
