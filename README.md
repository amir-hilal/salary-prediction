# Salary Prediction — End-to-End ML Pipeline

A production-grade machine learning system that predicts salaries, generates LLM-powered narrative insights, and surfaces results on a live interactive dashboard.

---

## Architecture Overview

```
Kaggle Dataset
      │
      ▼
┌─────────────────────┐
│  Data Pipeline      │  ingestion → EDA → feature engineering → cleaning
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Model Training     │  train → evaluate → artifact registry
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FastAPI Endpoint   │  /predict  (deployed)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Ollama LLM         │  local data-analyst agent → narrative + chart
│  (data analyst)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Supabase           │  stores predictions, narratives, chart metadata
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Streamlit Dashboard│  live, interactive, user-facing
└─────────────────────┘
```

---

## Components

### 1. Data Pipeline (`src/data/`, `notebooks/`)
- **Source**: Kaggle dataset (loaded via `kaggle` CLI / API)
- **Stages**: raw ingestion → EDA → feature selection → cleaning → preprocessing
- **Notebooks**: numbered sequence (`01_` through `04_`) for reproducible exploration
- Processed data is persisted to `data/processed/` for training

### 2. Model Training (`src/models/`)
- Trains a regression model to predict salary
- Evaluation metrics tracked (RMSE, MAE, R²)
- Trained artifact serialized to `models/artifacts/`
- Registry entry written to `models/registry/`

### 3. FastAPI Prediction Endpoint (`src/api/`)
- `POST /predict` — accepts candidate features, returns predicted salary
- Pydantic schemas for strict input/output validation
- Deployed as a containerized service

### 4. LLM Narrative Layer (`src/llm/`)
- Calls a local **Ollama** model (e.g. `llama3`, `mistral`)
- Acts as a **data analyst**: reads prediction context and writes a human-readable narrative with insights
- Generates at least one visualization (salary distribution, feature importance, regional breakdown)
- Output: markdown narrative + chart spec/image

### 5. Database (`src/database/`)
- **Supabase** (PostgreSQL) stores:
  - Individual prediction records
  - LLM-generated narratives
  - Visualization metadata / chart data
- Real-time subscriptions power the live dashboard

### 6. Streamlit Dashboard (`dashboard/`)
- Multi-page app with live data from Supabase
- Pages: Overview · Predictions · Insights
- Interactive filters (location, role, experience, etc.)
- Embedded charts from the LLM visualization layer

---

## Repository Structure

```
salary-prediction/
├── .github/
│   ├── copilot-instructions.md        # Workspace-wide Copilot guidelines
│   ├── instructions/                  # Domain-specific instruction files
│   │   ├── data-pipeline.instructions.md
│   │   ├── model-training.instructions.md
│   │   ├── fastapi.instructions.md
│   │   ├── llm-integration.instructions.md
│   │   ├── supabase.instructions.md
│   │   └── streamlit-dashboard.instructions.md
│   └── workflows/                     # CI/CD (GitHub Actions)
│       ├── model-deploy.yml
│       └── api-deploy.yml
│
├── data/
│   ├── raw/                           # Downloaded Kaggle CSVs (gitignored)
│   ├── processed/                     # Cleaned, feature-engineered data
│   └── external/                      # Supplementary reference data
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingestion.py               # Kaggle API download & schema validation
│   │   ├── cleaning.py                # Missing values, outliers, type coercion
│   │   └── preprocessing.py           # Encoding, scaling, train/test split
│   ├── features/
│   │   └── engineering.py             # Feature creation & selection logic
│   ├── models/
│   │   ├── train.py                   # Training loop & hyperparameter tuning
│   │   ├── evaluate.py                # Metric computation & reporting
│   │   └── predict.py                 # Inference wrapper (used by API)
│   ├── api/
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── routes/prediction.py       # POST /predict route
│   │   └── schemas/salary.py          # Pydantic request/response models
│   ├── llm/
│   │   ├── ollama_client.py           # Ollama HTTP client wrapper
│   │   └── narrative.py               # Prompt construction & parsing
│   ├── visualizations/
│   │   └── charts.py                  # Plotly / Matplotlib chart generators
│   └── database/
│       ├── client.py                  # Supabase client initialisation
│       └── crud.py                    # Insert / query helpers
│
├── dashboard/
│   ├── app.py                         # Streamlit entry point
│   ├── pages/
│   │   ├── 1_overview.py
│   │   ├── 2_predictions.py
│   │   └── 3_insights.py
│   └── components/
│       ├── charts.py                  # Reusable chart components
│       └── filters.py                 # Sidebar filter widgets
│
├── models/
│   ├── artifacts/                     # Serialized model files (.pkl / .joblib)
│   └── registry/                      # Model metadata JSON files
│
├── config/
│   ├── settings.py                    # Pydantic BaseSettings (env-driven)
│   └── logging.yaml                   # Structured logging config
│
├── tests/
│   ├── test_data/                     # Data pipeline unit tests
│   ├── test_models/                   # Model training & inference tests
│   ├── test_api/                      # FastAPI route tests (httpx)
│   └── test_llm/                      # LLM client & narrative tests
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.dashboard
│   │   └── docker-compose.yml
│   └── scripts/
│       └── setup_supabase.sql         # Table DDL for Supabase
│
├── .env.example                       # Required environment variables
├── .gitignore
├── Makefile                           # Developer convenience commands
├── pyproject.toml
└── requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) installed and running locally
- Kaggle API credentials (`~/.kaggle/kaggle.json`)
- Supabase project (URL + anon key)
- Docker (for containerised deployment)

### Local Setup

```bash
# 1. Clone & enter the project
git clone <repo-url>
cd salary-prediction

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 5. Download the dataset
make data-download

# 6. Run the full pipeline
make pipeline

# 7. Start the API
make api

# 8. Start the dashboard
make dashboard
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make data-download` | Pull dataset from Kaggle |
| `make pipeline` | Clean → features → train → evaluate |
| `make api` | Start FastAPI dev server |
| `make dashboard` | Start Streamlit dashboard |
| `make test` | Run all tests |
| `make docker-up` | Spin up full stack via Docker Compose |
| `make lint` | Run ruff + mypy |

---

## Environment Variables

See [.env.example](.env.example) for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `KAGGLE_USERNAME` | Kaggle account username |
| `KAGGLE_KEY` | Kaggle API key |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anon/public key |
| `OLLAMA_BASE_URL` | Ollama server address (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Model name (e.g. `llama3`, `mistral`) |
| `API_BASE_URL` | FastAPI service URL (for dashboard to call) |

---

## Key Design Decisions

> These are initial decisions — subject to change.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML framework | scikit-learn | Broad algorithm support, fast iteration |
| API framework | FastAPI | Async, auto-docs, Pydantic integration |
| LLM runtime | Ollama (local) | No cloud cost, privacy, easy model swap |
| Database | Supabase | Postgres + realtime + REST out of the box |
| Dashboard | Streamlit | Fast iteration, Python-native |
| Containerisation | Docker Compose | Reproducible local + production parity |
| Config management | Pydantic BaseSettings | Env-var driven, type-safe |

---

## Data Flow (detailed)

```
1. kaggle datasets download  →  data/raw/
2. ingestion.py              →  schema validation, type enforcement
3. cleaning.py               →  nulls, outliers, duplicates
4. engineering.py            →  encode categoricals, create interaction terms
5. preprocessing.py          →  scale numerics, train/test split
6. train.py                  →  fit model, log metrics
7. evaluate.py               →  hold-out evaluation, save report
8. POST /predict             →  real-time inference via FastAPI
9. narrative.py              →  build Ollama prompt from prediction context
10. ollama_client.py         →  call local LLM, parse response
11. charts.py                →  generate Plotly/Matplotlib figure
12. crud.py                  →  INSERT into Supabase (prediction + narrative + chart)
13. Streamlit dashboard      →  SELECT from Supabase, render live
```

---

## Testing Strategy

- **Unit**: each `src/` module tested in isolation with mocked I/O
- **Integration**: API routes tested with `httpx.AsyncClient`
- **Contract**: Pydantic schemas validate all API boundaries
- Run with: `pytest tests/ -v`

---

## Deployment

The API and dashboard are deployed as separate Docker containers. See [deployment/docker/docker-compose.yml](deployment/docker/docker-compose.yml).

CI/CD pipelines (GitHub Actions) in `.github/workflows/`:
- `model-deploy.yml` — retrain and publish model artifact on data/code change
- `api-deploy.yml` — build and push API container on merge to main
