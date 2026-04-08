---
description: "Use when working on Supabase client setup, database schema, table operations, or CRUD functions. Covers the supabase-py client, table conventions, insert/query patterns, and realtime subscriptions in src/database/."
applyTo: "src/database/**,deployment/scripts/setup_supabase.sql"
---

# Supabase Instructions

## Client Setup (`src/database/client.py`)
- Initialise the `supabase-py` client using `SUPABASE_URL` and `SUPABASE_ANON_KEY` from `config/settings.py`
- Create a module-level singleton `get_client() -> Client` — call it lazily on first use
- Never expose the service-role key in client-side code; use it only in server-side scripts

## Database Schema
All tables live in the `public` schema. Core tables:

### `predictions`
| Column | Type | Notes |
|--------|------|-------|
| `id` | `uuid` | Primary key, `gen_random_uuid()` |
| `created_at` | `timestamptz` | `now()` default |
| `features` | `jsonb` | Raw input features |
| `predicted_salary` | `numeric(12,2)` | Model output |
| `model_version` | `text` | From registry |
| `currency` | `text` | Default `'USD'` |

### `narratives`
| Column | Type | Notes |
|--------|------|-------|
| `id` | `uuid` | Primary key |
| `prediction_id` | `uuid` | FK → `predictions.id` |
| `created_at` | `timestamptz` | |
| `summary` | `text` | LLM executive summary |
| `insights` | `jsonb` | Array of insight strings |
| `recommendation` | `text` | |
| `chart_spec` | `jsonb` | Parsed `ChartSpec` |
| `raw_response` | `text` | Full LLM output for debugging |

## CRUD Patterns (`src/database/crud.py`)
- Every write function returns the inserted record — do not query immediately after insert
- Use typed return values: define Pydantic models for each table row
- Function signatures:
  - `insert_prediction(features: dict, salary: float, model_version: str) -> PredictionRecord`
  - `insert_narrative(prediction_id: str, narrative: NarrativeResult) -> NarrativeRecord`
  - `get_recent_predictions(limit: int = 100) -> list[PredictionRecord]`
  - `get_narrative_for_prediction(prediction_id: str) -> NarrativeRecord | None`
- All functions are `async` — use `await` on the supabase-py async client calls

## Realtime (Dashboard)
- Subscribe to the `predictions` table channel for INSERT events in the Streamlit dashboard
- Use `supabase.channel().on("postgres_changes", ...).subscribe()` pattern
- Handle disconnection gracefully — reconnect with exponential back-off

## Security
- Enable Row Level Security (RLS) on all tables
- Write RLS policies in `deployment/scripts/setup_supabase.sql`
- The anon key allows read-only access; writes go through the API which uses the service-role key server-side

## General Rules
- Keep all Supabase logic inside `src/database/` — no direct Supabase imports elsewhere
- Write integration-style tests in `tests/test_data/` using a Supabase test project or mocks
- Index `predictions.created_at` and `narratives.prediction_id` for dashboard query performance
