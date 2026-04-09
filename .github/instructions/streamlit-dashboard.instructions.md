---
description: "Use when working on the Streamlit dashboard — app structure, multi-page layout, realtime Supabase updates, chart rendering, or interactive filter components in dashboard/."
applyTo: "dashboard/**"
---

# Streamlit Dashboard Instructions

## App Structure (`dashboard/app.py`)
- App entry point: minimal — sets page config, loads settings, initialises the Supabase client once
- Use `st.set_page_config(layout="wide")` for a spacious layout
- Navigation is handled automatically via the `pages/` folder (Streamlit MPA convention)
- Shared state (Supabase client, settings) lives in `st.session_state` — initialise once in `app.py`

## Pages

### `overview.py` — Salary Landscape
- High-level summary: total predictions today, average predicted salary, salary distribution histogram
- Pulls from `predictions` table (recent 500 rows)
- Refresh button + auto-refresh every 30 s using `st.rerun()` with `time.sleep`

### `predictions.py` — Prediction Explorer
- Interactive form: user inputs candidate features, calls the FastAPI `POST /predict` endpoint
- Displays the returned `predicted_salary` with confidence context
- Shows the LLM narrative (summary, insights, recommendation) fetched from `narratives` table
- Renders the chart specified in `chart_spec` using `src/visualizations/charts.py`

### `insights.py` — Narrative & Charts
- List view of recent narratives with expandable detail
- Sidebar filters: date range, job category, location, experience level
- Comparative chart: user's prediction vs. population distribution

## Components

### `dashboard/components/charts.py`
- Wrapper functions around Plotly Express for consistent styling
- Accept a `ChartSpec` Pydantic model; look up the relevant data from Supabase
- Functions: `salary_histogram()`, `predicted_vs_actual_scatter()`, `feature_importance_bar()`
- Use `st.plotly_chart(fig, use_container_width=True)` for all Plotly charts

### `dashboard/components/filters.py`
- `render_sidebar_filters() -> FilterState` — renders all sidebar widgets and returns a typed `FilterState`
- `FilterState` is a Pydantic model; pass it to query functions to build filtered Supabase queries

## Realtime Updates
- Subscribe to the `predictions` Supabase realtime channel on page load
- Use `st.session_state` to accumulate new rows pushed via realtime
- Call `st.rerun()` when new data arrives to refresh the displayed tables/charts

## API Calls (from Dashboard to FastAPI)
- Read `API_BASE_URL` from `config/settings.py` — never hardcode in dashboard pages
- Use `httpx` (sync) for the predict call from the form; show a spinner with `st.spinner()`
- Handle `422` (validation error) and `500` (server error) gracefully with `st.error()`

## General Rules
- No business logic in dashboard pages — only presentation and user interaction
- No direct Supabase imports in pages — go through `src/database/crud.py`
- No ML model imports in dashboard — all inference goes through the API
- Keep each page file under ~150 lines; extract reusable UI into `dashboard/components/`
- Do not use `st.experimental_*` APIs — prefer stable equivalents
