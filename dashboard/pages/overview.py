"""Page 1 — Salary Landscape.

High-level summary metrics and a salary distribution histogram pulled from
the most recent 500 predictions.  Auto-refreshes every 30 s.
"""

import time

import pandas as pd
import streamlit as st

from src.database.crud import get_recent_predictions
from dashboard.components.charts import (
    render_salary_histogram,
    render_salary_density_by_experience,
    render_salary_stacked_histogram_by_experience,
)

st.set_page_config(page_title="Salary Landscape", layout="wide")


def _load_predictions() -> list[dict]:
    try:
        records = get_recent_predictions(limit=500)
    except Exception as exc:
        st.error(f"Could not load predictions from Supabase: {exc}")
        return []
    return [r.model_dump() for r in records]


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Salary Landscape")
st.caption("See how recent predictions are distributed. Refresh to pull the latest data from Supabase.")

# Auto-refresh controls ---------------------------------------------------
col_refresh, col_interval = st.columns([1, 3])
with col_refresh:
    manual_refresh = st.button("Refresh now", icon=":material/refresh:")
with col_interval:
    auto_refresh = st.toggle("Auto-refresh every 30 s", value=True)

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------

if "overview_records" not in st.session_state or manual_refresh:
    with st.spinner("Loading predictions…"):
        st.session_state["overview_records"] = _load_predictions()

records: list[dict] = st.session_state["overview_records"]

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

if records:
    salaries = [r["predicted_salary"] for r in records]
    avg_salary = sum(salaries) / len(salaries)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total predictions", f"{len(records):,}")
    col2.metric("Average predicted salary", f"${avg_salary:,.0f}")
    col3.metric("Latest prediction", f"${salaries[0]:,.0f}")
else:
    st.info("No predictions yet.  Submit one on the Prediction Explorer page.")

# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

st.subheader("Predicted Salary Distribution")
render_salary_histogram(records)

# ---------------------------------------------------------------------------
# Raw table (collapsed by default)
# ---------------------------------------------------------------------------

with st.expander("Raw prediction data", expanded=False):
    st.dataframe(records, use_container_width=True)

# ---------------------------------------------------------------------------
# Training data density — where the model is most reliable
# ---------------------------------------------------------------------------


@st.cache_data
def _load_training_df() -> pd.DataFrame:
    from src.data.cleaning import cap_salary_outliers, drop_leakage_columns
    from src.data.ingestion import load_raw, load_raw_from_supabase
    from src.features.engineering import build_features

    from config.settings import settings

    if settings.environment == "production":
        df = load_raw_from_supabase()
    else:
        df = load_raw(settings.data_raw_path)

    df = drop_leakage_columns(df)
    df = cap_salary_outliers(df)
    df = build_features(df)
    return df


st.divider()
st.subheader("Training Data Density")
st.caption(
    "The model is most reliable in the $50k–$200k salary band where training data "
    "is densest — primarily Mid-level and Senior roles."
)

try:
    training_df = _load_training_df()
except Exception as exc:
    st.error(f"Could not load training data: {exc}")
    training_df = pd.DataFrame()

if not training_df.empty:
    col_violin, col_stacked = st.columns(2)
    with col_violin:
        render_salary_density_by_experience(training_df)
    with col_stacked:
        render_salary_stacked_histogram_by_experience(training_df)

# ---------------------------------------------------------------------------
# Auto-refresh loop
# ---------------------------------------------------------------------------

# Guard: only sleep when auto_refresh is on and no manual refresh just fired.
# Without the guard every rerun (including the one triggered by st.rerun())
# would re-enter the sleep and loop forever.
if auto_refresh and not manual_refresh:
    time.sleep(30)
    st.session_state["overview_records"] = _load_predictions()
    st.rerun()
