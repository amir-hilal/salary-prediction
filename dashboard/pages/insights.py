"""Page 3 — Narrative & Charts.

List view of recent AI-generated narratives with expandable detail.
The sidebar filters narrow the view by date, experience, region, and job family.
A comparative chart at the bottom shows the filtered distribution.
"""

from datetime import datetime

import streamlit as st

from src.database.crud import get_recent_narratives, get_recent_predictions
from dashboard.components.charts import render_salary_histogram, render_feature_importance
from dashboard.components.filters import FilterState, render_sidebar_filters

st.set_page_config(page_title="Narrative Insights", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

filters: FilterState = render_sidebar_filters()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Narrative Insights")
st.caption(
    "Browse AI-generated salary narratives.  "
    "Use the sidebar filters to narrow the view."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    narratives = get_recent_narratives(limit=50)
except Exception as exc:
    st.error(f"Could not load narratives from Supabase: {exc}")
    narratives = []

try:
    predictions_raw = get_recent_predictions(limit=500)
except Exception as exc:
    st.error(f"Could not load predictions from Supabase: {exc}")
    predictions_raw = []

all_records = [r.model_dump() for r in predictions_raw]


def _parse_date(value: datetime | str | None):
    """Return a date object regardless of whether value is a datetime or ISO string."""
    if value is None:
        return None
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    if isinstance(value, datetime):
        return value.date()
    return None


def _apply_filters(records: list[dict], f: FilterState) -> list[dict]:
    """Filter prediction records using the active FilterState."""
    out = records
    if f.date_from:
        out = [r for r in out if (d := _parse_date(r.get("created_at"))) and d >= f.date_from]
    if f.date_to:
        out = [r for r in out if (d := _parse_date(r.get("created_at"))) and d <= f.date_to]
    if f.experience_level is not None:
        out = [
            r for r in out
            if r.get("features", {}).get("experience_level") is not None
            and r.get("features", {}).get("experience_level") in f.experience_level
        ]
    if f.location_region is not None:
        out = [
            r for r in out
            if r.get("features", {}).get("location_region") is not None
            and r.get("features", {}).get("location_region") in f.location_region
        ]
    if f.job_family is not None:
        out = [
            r for r in out
            if r.get("features", {}).get("job_family") is not None
            and r.get("features", {}).get("job_family") in f.job_family
        ]
    return out


filtered_records = _apply_filters(all_records, filters)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)
col1.metric("Narratives available", len(narratives))
col2.metric("Filtered predictions", len(filtered_records))

st.divider()

# ---------------------------------------------------------------------------
# Narrative list
# ---------------------------------------------------------------------------

st.subheader("Recent Narratives")

if not narratives:
    st.info("No narratives yet.  Make a prediction on the Prediction Explorer page.")
else:
    for narrative in narratives:
        with st.expander(
            f"Prediction `{narrative.prediction_id[:8]}…`  —  {narrative.created_at.strftime('%Y-%m-%d %H:%M')}",
            expanded=False,
        ):
            st.markdown(f"**Summary**\n\n{narrative.summary}")
            st.markdown(f"**Uncertainty**\n\n{narrative.uncertainty}")

            if narrative.insights:
                st.markdown("**Key Insights**")
                for insight in narrative.insights:
                    st.markdown(f"- {insight}")

            st.markdown(f"**Recommendation**\n\n{narrative.recommendation}")

st.divider()

# ---------------------------------------------------------------------------
# Comparative chart — filtered distribution
# ---------------------------------------------------------------------------

st.subheader("Filtered Salary Distribution")
if filtered_records:
    render_salary_histogram(filtered_records)
else:
    st.info("No records match the current filters.")

# ---------------------------------------------------------------------------
# Feature importance reference chart
# ---------------------------------------------------------------------------

st.subheader("Feature Importance (Model Reference)")
render_feature_importance()
