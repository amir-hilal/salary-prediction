# Salary Prediction

End-to-end ML system that predicts data-profession salaries, generates LLM-powered narrative insights, and surfaces everything on a live interactive dashboard.

| | URL |
|---|---|
| **Dashboard** | `https://salary-predic.streamlit.app/` |
| **API (v1)** | `https://underground-riva-hilalpines-4f2281c2.koyeb.app/api/v1/` |
| **API Docs** | `https://underground-riva-hilalpines-4f2281c2.koyeb.app/docs` |

> **v1** is the current and only API version. All endpoints live under `/api/v1/`.

---

## Table of Contents

1. [Local Setup](#local-setup)
2. [Supabase Setup](#supabase-setup)
3. [Running Locally](#running-locally)
4. [How It All Works](#how-it-all-works)
5. [Prediction Request Flow](#prediction-request-flow)
6. [Architecture Overview](#architecture-overview)
7. [Components](#components)
8. [Testing](#testing)
9. [Deployment](#deployment)

---

## Local Setup

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running locally
- A free [Supabase](https://supabase.com) account (for the database)
- Docker (optional — for containerised deployment only)

### Clone and install

```bash
git clone <repo-url>
cd salary-prediction

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Download the dataset

The raw data is the **Data Science Salaries** dataset from Kaggle:
https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries

Download the CSV and place it at `data/raw/ds_salaries.csv`. The file is already gitignored.

### Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values. The key variables:

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL (`https://xxx.supabase.co`) |
| `SUPABASE_ANON_KEY` | Publishable key (safe for client-side reads, respects RLS) |
| `SUPABASE_SERVICE_ROLE_KEY` | Secret key (server-side writes, bypasses RLS — keep it secret) |
| `OLLAMA_BASE_URL` | Ollama server address (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | LLM model name (default `phi4-mini`) |
| `API_BASE_URL` | Where the dashboard finds the API (default `http://localhost:8000`) |

### Pull the Ollama model

```bash
ollama pull phi4-mini
```

---

## Supabase Setup

1. Create a free project at [supabase.com](https://supabase.com).
2. In **Project Settings → API**, copy:
   - **Project URL** → `SUPABASE_URL`
   - **Publishable key** (`sb_publishable_...`) → `SUPABASE_ANON_KEY`
   - **Secret key** (`sb_secret_...`) → `SUPABASE_SERVICE_ROLE_KEY`
3. Open the **SQL Editor**, paste the contents of [`deployment/scripts/setup_supabase.sql`](deployment/scripts/setup_supabase.sql), and click **Run**.
4. Verify: go to **Database → Publications** and confirm `predictions` and `narratives` appear under `supabase_realtime`.

See [`docs/supabase-setup.md`](docs/supabase-setup.md) for the full step-by-step guide.

---

## Running Locally

Start two terminals:

```bash
# Terminal 1 — API
make api
# Runs uvicorn on http://localhost:8000
# Docs at http://localhost:8000/docs

# Terminal 2 — Dashboard
make dashboard
# Runs Streamlit on http://localhost:8501
```

Verify the API is up:

```bash
curl http://localhost:8000/api/v1/health
# → {"status": "ok", "model_version": "20260407_172358"}
```

### Makefile targets

| Target | What it does |
|---|---|
| `make pipeline` | Full data → feature → train pipeline |
| `make train` | Train the model only |
| `make api` | Start FastAPI dev server (port 8000) |
| `make dashboard` | Start Streamlit dashboard (port 8501) |
| `make test` | Run the full test suite |
| `make lint` | Run ruff + mypy |
| `make docker-up` | Spin up the full stack via Docker Compose |
| `make docker-down` | Stop the Docker Compose stack |
| `make clean` | Remove caches and generated artifacts |

---

## How It All Works

### EDA and decisions

- The dataset contains 607 salary records across data professions, with columns like `job_title`, `experience_level`, `company_location`, `salary_in_usd`, `remote_ratio`, etc.
- **Target column**: `salary_in_usd` — the only currency-normalised salary column. The raw `salary` and `salary_currency` columns are dropped as **data leakage** (they encode the same information in a currency-dependent way).
- **`employee_residence`** is dropped because it is redundant with `company_location` in this dataset and adds no predictive value.
- **Outlier treatment**: salary outliers are capped using the IQR method (1.5× factor). This preserves all records while limiting the influence of extreme values on training.
- **`job_title` → `job_family`**: the original column has 100+ unique titles. We map each title into one of 6 functional groups (Other, Analytics, Data Science, Data Engineering, ML/AI, Leadership) using keyword matching. This reduces sparsity without losing the signal.
- **`company_location` → `location_region` + `is_us_company`**: 80+ country codes are collapsed into 4 regions (Rest of World, Asia Pacific, Europe, North America), plus a binary US flag. Geography is a strong salary driver and the US alone accounts for the majority of records.
- **Scaler**: `RobustScaler` is used because it is based on percentiles (IQR) rather than mean/std, making it less sensitive to the outliers we just capped.

### Model training and evaluation

- **Model**: `DecisionTreeRegressor` (scikit-learn) — chosen because tree-based models handle ordinal and label-encoded features naturally, and the tree structure gives us free salary ranges via leaf nodes.
- **Hyperparameter search**: `GridSearchCV` over `max_depth`, `min_samples_split`, and `min_samples_leaf` with 5-fold cross-validation. The best configuration is `max_depth=5`, `min_samples_leaf=4`, `min_samples_split=10`.
- **Leaf ranges**: after training, every training sample is routed to its leaf via `apply()`. The **Q25 and Q75** of `salary_in_usd` within each leaf are computed and stored in a `leaf_ranges` dictionary. The pipeline and `leaf_ranges` are bundled together in a single `.joblib` artifact.
- **Evaluation metrics** (hold-out test set):
  - **RMSE**: $42,564 — the average magnitude of error, in dollars.
  - **MAE**: $31,480 — the median-like error; half of predictions are closer than this.
  - **R²**: 0.453 — the model explains ~45% of salary variance. The remaining 55% is driven by factors not in the dataset (negotiation, company brand, niche skills, etc.).
  - **MAPE**: 80.8% — high because of low-salary records where even a small absolute error is a large percentage. MAPE is misleading for this use case; RMSE and MAE are more reliable.
- **Key takeaway**: the model is useful for ballpark estimates and peer-group comparisons, not precise salary figures. This is why we always return a **range** alongside the point estimate, and the LLM narrative is required to disclose the model's error.

### Key terms

- **Point estimate**: the Decision Tree's single prediction — the mean salary of all training samples in the leaf node the input lands in.
- **Peer-group range (Q25–Q75)**: the interquartile range of actual training salaries in that same leaf. This is the "most peers with your profile earned between X and Y" range.
- **Leaf node**: the terminal node in a Decision Tree. Each leaf groups training samples that followed the same sequence of splits — effectively a peer group.
- **MAE (Mean Absolute Error)**: on average, the model's predictions are off by this amount. We inject this into the LLM prompt so the narrative can quote it honestly.

---

## Prediction Request Flow

This is what happens when a user submits a prediction from the Streamlit dashboard:

```
User fills form in Streamlit
        │
        ▼
POST /api/v1/predict  ──────────────────────────────────────┐
  │                                                         │
  │  1. Validate input (Pydantic schema)                    │
  │  2. Run input through the Decision Tree pipeline        │
  │     → point_estimate + leaf_id                          │
  │  3. Look up Q25–Q75 for that leaf → range_low/high      │
  │  4. INSERT prediction row into Supabase (awaited)       │
  │  5. Return response with prediction_id                  │
  │                                                         │
  ▼                                                         │
Dashboard receives prediction_id ◄──────────────────────────┘
        │
        ▼
GET /api/v1/predict/{id}/narrative  (SSE stream)
  │
  │  1. Fetch prediction row from Supabase (by prediction_id)
  │  2. Build LLM prompt (inject point estimate, range, MAE)
  │  3. Call Ollama with stream=True
  │  4. Yield each token as an SSE event: "data: <token>\n\n"
  │  5. Dashboard renders tokens word-by-word in real time
  │  6. After all tokens: parse the full text into structured
  │     NarrativeResult + ChartSpec, INSERT into Supabase
  │  7. Send "data: [DONE]\n\n" to close the stream
  │
  ▼
Dashboard fetches the parsed NarrativeResult from Supabase
  │
  │  Renders: summary, uncertainty disclosure, insights,
  │  recommendation, and the chart specified by ChartSpec
  ▼
Done — prediction + narrative stored in Supabase for the
Overview and Insights pages to display
```

The prediction is persisted **synchronously** before the API returns, so the narrative endpoint is guaranteed to find the row — no race conditions.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Streamlit Dashboard                          │
│               (Overview · Predictions · Insights)                    │
│                     reads via anon key (RLS)                         │
└──────────┬────────────────────────┬──────────────────────────────────┘
           │ POST /predict          │ GET /predict/{id}/narrative (SSE)
           ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          FastAPI  (v1)                               │
│                                                                      │
│  predict_salary()            stream_narrative()                      │
│   ├─ Decision Tree model      ├─ Fetch context from Supabase         │
│   ├─ Leaf range lookup        ├─ Build prompt                        │
│   └─ INSERT prediction ──┐    ├─ Stream tokens from Ollama (SSE)     │
│                          │    └─ Parse + INSERT narrative            │
└──────────────────────────┼──────────────┬────────────────────────────┘
                           │              │
                           ▼              ▼
          ┌──────────────────────┐   ┌─────────────────────┐
          │       Supabase       │   │    Ollama (local)   │
          │    (PostgreSQL)      │   │     phi4-mini LLM   │
          │                      │   │                     │
          │  predictions table   │   │  /api/generate      │
          │  narratives table    │   │  stream: true       │
          │  RLS + realtime      │   └─────────────────────┘
          └──────────────────────┘
```

---

## Components

### Data Pipeline (`src/data/`, `src/features/`, `notebooks/`)

- `src/data/ingestion.py` — loads `data/raw/ds_salaries.csv`, validates schema, drops auto-index column
- `src/data/cleaning.py` — drops leakage columns (`salary`, `salary_currency`, `employee_residence`), caps outliers via IQR
- `src/features/engineering.py` — maps `job_title` → `job_family`, `company_location` → `location_region` + `is_us_company`, ordinal-encodes `experience_level`, `employment_type`, `company_size`
- `src/data/preprocessing.py` — train/test split (80/20), `RobustScaler` fit on training set
- `notebooks/eda.ipynb` — exploratory analysis: distributions, correlations, feature decisions

### Model (`src/models/`)

- `train.py` — `DecisionTreeRegressor` + `GridSearchCV`, `compute_leaf_ranges()` bundles Q25–Q75 per leaf into the artifact
- `evaluate.py` — RMSE, MAE, R², MAPE on hold-out set; writes `models/registry/latest.json` with a regression guard (new model must beat or tie current RMSE)
- `predict.py` — `predict(features) → PredictionResult(point_estimate, range_low, range_high)`; loads the model as a singleton at startup

### FastAPI (`src/api/`)

- `main.py` — app setup, lifespan (warms model singleton), CORS, global 500 handler
- `routes/prediction.py`:
  - `POST /api/v1/predict` — validates input, runs inference, persists to Supabase, returns salary + range + prediction_id
  - `GET /api/v1/predict/{id}/narrative` — SSE stream of LLM tokens; parses and persists on completion
  - `GET /api/v1/health` — liveness check with model version
- `schemas/salary.py` — `PredictionRequest` (8 validated fields), `PredictionResponse`, `ErrorResponse`

### LLM Integration (`src/llm/`)

- `ollama_client.py` — async httpx client with two modes: `generate()` (blocking) and `generate_stream()` (token-by-token SSE)
- `narrative.py` — `build_prompt()` injects prediction context into a structured 6-section prompt; `parse_narrative()` extracts `NarrativeResult` + `ChartSpec` via regex; `generate_narrative_stream()` yields tokens and persists the parsed result on stream completion

### Database (`src/database/`)

- `client.py` — lazy singleton clients: async (`get_client()`, service-role key for writes) and sync (`get_anon_client()`, anon key for dashboard reads)
- `crud.py` — `insert_prediction()`, `insert_narrative()`, `get_recent_predictions()`, `get_narrative_for_prediction()`, `get_prediction_context_async()`
- Tables: `predictions` (features, salary, range, model version) and `narratives` (summary, uncertainty, insights, recommendation, chart_spec, raw LLM output)
- RLS policies: anon key can only read; service-role key can read/write

### Visualizations (`src/visualizations/`)

- `charts.py` — `salary_histogram()`, `predicted_vs_actual_scatter()`, `feature_importance_bar()`, `from_chart_spec()` dispatcher
- Used by both the LLM layer (chart spec generation) and the dashboard (rendering)

### Dashboard (`dashboard/`)

- `app.py` — Streamlit entry point, initialises Supabase anon client in session state
- `pages/overview.py` — salary landscape metrics, histogram, auto-refresh every 30s
- `pages/predictions.py` — input form → API call → salary metrics → SSE token-by-token narrative streaming → structured display + chart
- `pages/insights.py` — narrative list with sidebar filters (date, job family, location, experience), comparative charts
- `components/charts.py` — Plotly wrappers, `render_chart_from_spec()`
- `components/filters.py` — `FilterState` Pydantic model, `render_sidebar_filters()`

---

## Testing

99 tests across 5 test modules:

| Module | Tests | Covers |
|---|---|---|
| `test_data/test_cleaning.py` | 12 | Leakage drops, IQR cap, no-mutation, idempotency |
| `test_data/test_engineering.py` | 35 | job_family mapping (11 titles), location_region (8 countries), ordinal encoding |
| `test_models/test_train.py` | 14 | Pipeline shape, leaf ranges coverage, Q25 <= Q75, save/load roundtrip |
| `test_api/test_prediction.py` | 16 | Happy path, range fields, 422 validation, 500 safe message, health, SSE stream |
| `test_llm/test_narrative.py` | 22 | build_prompt, parse_narrative, generate_narrative, generate_narrative_stream |

Run all tests:

```bash
make test
```

Tests use mocked HTTP clients and Supabase — no real Ollama or database calls in CI.

---

## Deployment

### Docker

The API and dashboard are containerised separately. Docker Compose orchestrates both services plus Ollama:

```
deployment/
├── docker/
│   ├── Dockerfile.api           # FastAPI + uvicorn
│   ├── Dockerfile.dashboard     # Streamlit
│   └── docker-compose.yml       # API + dashboard + Ollama
└── scripts/
    └── setup_supabase.sql       # Run once against your Supabase project
```

```bash
# Start everything
make docker-up

# Stop
make docker-down
```

> Docker deployment files are scaffolded but not yet populated. They will be completed when the project moves to hosted infrastructure.

### Environment

All configuration is driven by environment variables via Pydantic `BaseSettings`. Copy `.env.example` to `.env` and fill in your values. Secrets are never hardcoded or committed — `.env` is gitignored.
