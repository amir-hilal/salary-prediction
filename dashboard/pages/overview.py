"""Page 1 — Salary Landscape.

High-level summary metrics and a salary distribution histogram pulled from
the most recent 500 predictions.  Auto-refreshes every 30 s.
"""

import time

import streamlit as st

from src.database.crud import get_recent_predictions
from dashboard.components.charts import render_salary_histogram

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
st.caption("Live view of the most recent 500 predictions from the API.")

# Auto-refresh controls ---------------------------------------------------
col_refresh, col_interval = st.columns([1, 3])
with col_refresh:
    manual_refresh = st.button("🔄 Refresh now")
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
# Auto-refresh loop
# ---------------------------------------------------------------------------

# Guard: only sleep when auto_refresh is on and no manual refresh just fired.
# Without the guard every rerun (including the one triggered by st.rerun())
# would re-enter the sleep and loop forever.
if auto_refresh and not manual_refresh:
    time.sleep(30)
    st.session_state["overview_records"] = _load_predictions()
    st.rerun()
