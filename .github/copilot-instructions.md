# Salary Prediction — Copilot Workspace Instructions

## Project Summary
End-to-end ML system: Kaggle data → scikit-learn model → FastAPI → Ollama LLM narrative → Supabase → Streamlit dashboard.

## Tech Stack
- **ML**: scikit-learn, pandas, numpy
- **API**: FastAPI + Pydantic v2, uvicorn
- **LLM**: Ollama (local), prompt via `src/llm/`
- **Database**: Supabase (PostgreSQL + realtime)
- **Dashboard**: Streamlit (multi-page)
- **Config**: Pydantic `BaseSettings` reading from `.env`
- **Testing**: pytest + httpx for async API tests
- **Linting**: ruff, mypy (strict)
- **Containers**: Docker Compose

## Code Style
- Python 3.12+; use type hints everywhere
- Pydantic v2 models for all data contracts (API schemas, config, DB records)
- No `print()` — use Python `logging` configured via `config/logging.yaml`
- Prefer explicit over implicit; avoid clever one-liners
- Functions stay small and single-purpose
- All secrets come from environment variables via `config/settings.py` — never hardcoded

## Architecture Conventions
- `src/` contains all importable Python modules
- `notebooks/` contains exploratory work only — no production logic
- Notebook names are simple and descriptive — no numeric prefixes (e.g. `eda.ipynb`, not `02_eda.ipynb`)
- Production logic is extracted from notebooks into `src/` modules
- The FastAPI app (`src/api/main.py`) is the sole inference entry point
- Supabase interactions go through `src/database/crud.py` only
- Ollama calls go through `src/llm/ollama_client.py` only

## Build & Test
```bash
make pipeline     # full data → train pipeline
make api          # start FastAPI dev server
make dashboard    # start Streamlit
make test         # pytest tests/
make lint         # ruff + mypy
```

## Conventions That Differ From Defaults
- Model artifacts live in `models/artifacts/` (not project root or `model/`)
- Config is loaded once at startup via `config/settings.py` — do not re-read `.env` files in business logic
- All database writes return the inserted record — callers should not query immediately after insert
- Visualizations are generated in `src/visualizations/charts.py` and consumed by both the LLM layer and the dashboard

## File-Specific Instructions
Additional domain instructions are in `.github/instructions/`. They are loaded on-demand when working on specific modules:
- `data-pipeline.instructions.md` — data ingestion, cleaning, feature engineering
- `model-training.instructions.md` — training, evaluation, serialisation
- `fastapi.instructions.md` — API routes, schemas, middleware
- `llm-integration.instructions.md` — Ollama client, prompt design, narrative parsing
- `supabase.instructions.md` — database schema, client setup, CRUD patterns
- `streamlit-dashboard.instructions.md` — multi-page app, realtime updates, components
