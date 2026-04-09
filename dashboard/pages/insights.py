"""Page 3 — Narrative & Charts.

Four tabs:
  - Narratives: recent AI-generated narratives with expandable detail.
  - Key Drivers: salary by experience, region, and job family.
  - Patterns: salary by remote ratio, company size, year trend, and heatmap.
  - Usage: prediction volume over time plus usage metrics.

The sidebar filters apply to the narrative list only; all EDA charts use the
full training dataset.
"""

import logging

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    render_experience_region_heatmap,
    render_prediction_volume,
    render_salary_by_company_size,
    render_salary_by_experience,
    render_salary_by_job_family,
    render_salary_by_region,
    render_salary_by_remote_ratio,
    render_salary_trend,
)
from src.database.crud import get_recent_narratives, get_recent_predictions

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Narrative Insights", layout="wide")

# ---------------------------------------------------------------------------
# Training data loader (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def _load_training_df() -> pd.DataFrame:
    from src.data.cleaning import cap_salary_outliers, drop_leakage_columns
    from src.data.ingestion import load_raw
    from src.features.engineering import build_features

    from config.settings import settings
    df = load_raw(settings.data_raw_path)
    df = drop_leakage_columns(df)
    df = cap_salary_outliers(df)
    df = build_features(df)
    return df


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Salary Insights")
st.caption(
    "Explore salary trends, key drivers, and patterns from the training data. "
    "The Usage tab shows prediction activity and AI-generated narratives."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    narratives = get_recent_narratives(limit=50)
except Exception as exc:
    st.error(f"Could not load narratives from Supabase: {exc}")
    narratives = []
    logger.exception("Failed to load narratives")

try:
    predictions_raw = get_recent_predictions(limit=500)
except Exception as exc:
    st.error(f"Could not load predictions from Supabase: {exc}")
    predictions_raw = []
    logger.exception("Failed to load predictions")

all_records = [r.model_dump() for r in predictions_raw]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_drivers, tab_patterns, tab_usage = st.tabs(
    ["Key Drivers", "Patterns", "Usage"]
)

# ---------------------------------------------------------------------------
# Tab: Key Drivers
# ---------------------------------------------------------------------------

with tab_drivers:
    try:
        training_df = _load_training_df()
    except Exception as exc:
        st.error(f"Could not load training data: {exc}")
        logger.exception("Failed to load training DataFrame")
        training_df = pd.DataFrame()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Salary by Experience Level")
        render_salary_by_experience(training_df)

    with col_right:
        st.subheader("Salary by Region")
        render_salary_by_region(training_df)

    st.subheader("Median Salary by Job Family")
    render_salary_by_job_family(training_df)

# ---------------------------------------------------------------------------
# Tab: Patterns
# ---------------------------------------------------------------------------

with tab_patterns:
    if "training_df" not in dir():
        try:
            training_df = _load_training_df()
        except Exception as exc:
            st.error(f"Could not load training data: {exc}")
            logger.exception("Failed to load training DataFrame")
            training_df = pd.DataFrame()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Salary by Remote Ratio")
        render_salary_by_remote_ratio(training_df)

    with col_right:
        st.subheader("Salary by Company Size")
        render_salary_by_company_size(training_df)

    st.subheader("Salary Trend by Year")
    render_salary_trend(training_df)

    st.subheader("Median Salary: Experience × Region")
    render_experience_region_heatmap(training_df)

# ---------------------------------------------------------------------------
# Tab: Usage
# ---------------------------------------------------------------------------

with tab_usage:
    st.metric("Narratives available", len(narratives))

    st.subheader("Prediction Volume Over Time")
    render_prediction_volume(all_records)

    st.divider()
    st.subheader("Recent Narratives")

    if not narratives:
        st.info("No narratives yet. Make a prediction on the Prediction Explorer page.")
    else:
        for narrative in narratives:
            with st.expander(
                f"Prediction `{narrative.prediction_id[:8]}…`  —  "
                f"{narrative.created_at.strftime('%Y-%m-%d %H:%M')}",
                expanded=False,
            ):
                st.markdown(f"**Summary**\n\n{narrative.summary}")
                st.markdown(f"**Uncertainty**\n\n{narrative.uncertainty}")

                if narrative.insights:
                    st.markdown("**Key Insights**")
                    for insight in narrative.insights:
                        st.markdown(f"- {insight}")

                st.markdown(f"**Recommendation**\n\n{narrative.recommendation}")
