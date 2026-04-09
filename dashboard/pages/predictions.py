"""Page 2 — Prediction Explorer.

Users fill in candidate features, call the FastAPI /predict endpoint, and see
the returned salary estimate alongside the LLM-generated narrative and chart.
"""

import time

import httpx
import streamlit as st

from config.settings import settings
from src.database.crud import get_narrative_for_prediction
from dashboard.components.charts import render_chart_from_spec
from src.database.crud import get_recent_predictions
from src.llm.narrative import ChartSpec, NarrativeRecord

st.set_page_config(page_title="Prediction Explorer", layout="wide")

st.title("Prediction Explorer")
st.caption("Fill in the form below and click **Predict** to get a salary estimate.")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        experience_level = st.selectbox(
            "Experience level",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Entry-level", "Mid-level", "Senior", "Executive"][x],
        )
        employment_type = st.selectbox(
            "Employment type",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Part-time", "Freelance", "Contract", "Full-time"][x],
            index=3,
        )
        remote_ratio = st.selectbox(
            "Remote ratio",
            options=[0, 50, 100],
            format_func=lambda x: {0: "On-site (0%)", 50: "Hybrid (50%)", 100: "Remote (100%)"}[x],
            index=2,
        )
        company_size = st.selectbox(
            "Company size",
            options=[0, 1, 2],
            format_func=lambda x: ["Small", "Medium", "Large"][x],
            index=1,
        )

    with col2:
        work_year = st.number_input("Work year", min_value=2020, max_value=2030, value=2024)
        job_family = st.selectbox(
            "Job family",
            options=[0, 1, 2, 3, 4, 5],
            format_func=lambda x: [
                "Other", "Analytics", "Data Science",
                "Data Engineering", "ML / AI", "Leadership",
            ][x],
            index=2,
        )
        location_region = st.selectbox(
            "Location region",
            options=[0, 1, 2, 3],
            format_func=lambda x: [
                "Rest of World", "Asia Pacific", "Europe", "North America",
            ][x],
            index=3,
        )
        is_us_company = st.radio(
            "US-based company?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x else "No",
            horizontal=True,
        )

    submitted = st.form_submit_button("Predict", type="primary")

# ---------------------------------------------------------------------------
# Prediction call
# ---------------------------------------------------------------------------

if submitted:
    payload = {
        "experience_level": experience_level,
        "employment_type": employment_type,
        "remote_ratio": remote_ratio,
        "company_size": company_size,
        "work_year": int(work_year),
        "job_family": job_family,
        "location_region": location_region,
        "is_us_company": is_us_company,
    }

    with st.spinner("Calling the prediction API…"):
        try:
            response = httpx.post(
                f"{settings.api_base_url}/api/v1/predict",
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            try:
                result = response.json()
            except ValueError as json_exc:
                st.error(f"API returned invalid JSON: {json_exc}")
                st.stop()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422:
                st.error(f"Validation error — check your inputs. ({exc.response.text})")
            else:
                st.error(f"Server error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except httpx.RequestError as exc:
            st.error(f"Could not reach the API at {settings.api_base_url}. Is it running?  ({exc})")
            st.stop()

    if "salary" not in result or "prediction_id" not in result:
        st.error("API returned an unexpected response structure. Please try again.")
        st.stop()
    salary = result["salary"]
    if not isinstance(salary, dict) or not {"mean", "low", "high"}.issubset(salary):
        st.error("Salary data is incomplete. Please try again.")
        st.stop()
    prediction_id: str = result["prediction_id"]

    # -----------------------------------------------------------------------
    # Display salary result
    # -----------------------------------------------------------------------

    st.success("Prediction complete!")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted salary", f"${salary['mean']:,.0f}")
    m2.metric("Peer range (low)", f"${salary['low']:,.0f}")
    m3.metric("Peer range (high)", f"${salary['high']:,.0f}")
    st.caption(f"Model version: `{result['model_version']}` | Prediction ID: `{prediction_id}`")

    # -----------------------------------------------------------------------
    # LLM Narrative — poll until ready (max 90 s)
    # -----------------------------------------------------------------------

    st.subheader("AI Narrative")
    narrative_placeholder = st.empty()

    narrative = None
    for attempt in range(18):  # 18 × 5 s = 90 s max
        try:
            narrative = get_narrative_for_prediction(prediction_id)
        except Exception as exc:
            narrative_placeholder.error(f"Error fetching narrative: {exc}")
            break
        if narrative:
            break
        narrative_placeholder.info(
            f"Generating narrative… ({attempt * 5} s elapsed)"
        )
        time.sleep(5)

    if narrative is None:
        narrative_placeholder.warning(
            "The narrative is still being generated.  "
            "Refresh this page in a moment to see it."
        )
    else:
        narrative_placeholder.empty()

        with st.container():
            st.markdown(f"**Summary**\n\n{narrative.summary}")
            st.markdown(f"**Uncertainty**\n\n{narrative.uncertainty}")

            if narrative.insights:
                st.markdown("**Key Insights**")
                for insight in narrative.insights:
                    st.markdown(f"- {insight}")

            st.markdown(f"**Recommendation**\n\n{narrative.recommendation}")

        # -------------------------------------------------------------------
        # Chart from spec
        # -------------------------------------------------------------------

        st.subheader("Chart")
        recent_records = [r.model_dump() for r in get_recent_predictions(limit=200)]
        chart_spec = ChartSpec(**narrative.chart_spec) if isinstance(narrative.chart_spec, dict) else narrative.chart_spec
        render_chart_from_spec(
            chart_spec,
            recent_records,
            point_estimate=salary["mean"],
        )
