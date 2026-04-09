"""Salary Prediction Dashboard — Overview (landing page).

Streamlit multi-page app entry point.  Seeds ``st.session_state`` with the
Supabase anon client, renders the hero section, summary metrics, and a
predicted-salary histogram.

Uses st.navigation for explicit control over sidebar page labels and order.
"""

import time

import streamlit as st

from config.settings import settings
from src.database.client import get_anon_client
from src.database.crud import get_recent_predictions
from dashboard.components.charts import render_salary_histogram


def _init_session_state() -> None:
    """Seed session_state keys that must exist before any page renders."""
    if "supabase" not in st.session_state:
        st.session_state["supabase"] = get_anon_client()
    if "settings" not in st.session_state:
        st.session_state["settings"] = settings


st.set_page_config(
    page_title="Salary Prediction",
    page_icon=":material/work:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_session_state()


# ---------------------------------------------------------------------------
# Overview page (inline — not a separate file)
# ---------------------------------------------------------------------------

def _load_predictions() -> list[dict]:
    try:
        records = get_recent_predictions(limit=500)
    except Exception as exc:
        st.error(f"Could not load predictions from Supabase: {exc}")
        return []
    return [r.model_dump() for r in records]


def overview_page() -> None:
    """Render the Overview landing page."""

    # Hero section
    st.title("Salary Prediction for Data Professionals")
    st.markdown(
        "Get an instant salary estimate based on your experience, role, and location "
        "— powered by machine learning and explained by AI."
    )

    st.divider()

    # Feature cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(":material/target: Predict", anchor=False)
        st.markdown(
            "Enter your experience level, job family, region, and company details "
            "to get a salary estimate with a confidence range."
        )

    with c2:
        st.subheader(":material/psychology: Understand", anchor=False)
        st.markdown(
            "An AI-generated narrative explains **why** your prediction landed "
            "where it did — covering key factors, uncertainty, and next steps."
        )

    with c3:
        st.subheader(":material/bar_chart: Explore", anchor=False)
        st.markdown(
            "Browse historical predictions, compare salary distributions, and "
            "filter by experience, region, or job family."
        )

    st.divider()

    # CTA — centred button with shimmer animation
    st.markdown(
        """
<style>
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.cta-wrap {
  display: flex;
  justify-content: center;
  margin: 1.5rem 0;
}
.cta-wrap a {
  display: inline-block;
  padding: 0.75rem 2.5rem;
  border-radius: 8px;
  font-weight: 700;
  font-size: 1.1rem;
  color: #fff;
  text-decoration: none;
  background: linear-gradient(
    110deg,
    #4F8EF7 0%,
    #4F8EF7 40%,
    #7eb4ff 50%,
    #4F8EF7 60%,
    #4F8EF7 100%
  );
  background-size: 200% 100%;
  animation: shimmer 2.5s infinite linear;
  transition: filter 0.2s;
}
.cta-wrap a:hover {
  filter: brightness(1.12);
  color: #fff;
}
</style>
<div class="cta-wrap">
  <a href="/Reveal_Your_True_Salary" target="_self">Reveal Your True Salary →</a>
</div>
""",
        unsafe_allow_html=True,
    )

    # How it works
    st.markdown("#### How it works")
    st.markdown(
        "1. **Fill in your profile** — experience, role, company, and location.\n"
        "2. **Get an estimate** — a point prediction plus a Q25–Q75 peer-group range.\n"
        "3. **Read the narrative** — an AI explanation of the factors behind the number."
    )

    st.divider()

    # Salary Landscape — summary metrics + histogram
    st.subheader("Salary Landscape")
    st.caption("See how recent predictions are distributed. Refresh to pull the latest data from Supabase.")

    col_refresh, col_interval = st.columns([1, 3])
    with col_refresh:
        manual_refresh = st.button("Refresh now", icon=":material/refresh:")
    with col_interval:
        auto_refresh = st.toggle("Auto-refresh every 30 s", value=True)

    if "overview_records" not in st.session_state or manual_refresh:
        with st.spinner("Loading predictions…"):
            st.session_state["overview_records"] = _load_predictions()

    records: list[dict] = st.session_state["overview_records"]

    if records:
        salaries = [r["predicted_salary"] for r in records]
        avg_salary = sum(salaries) / len(salaries)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total predictions", f"{len(records):,}")
        col2.metric("Average predicted salary", f"${avg_salary:,.0f}")
        col3.metric("Latest prediction", f"${salaries[0]:,.0f}")
    else:
        st.info("No predictions yet. Head to **Reveal Your True Salary** to submit one.")

    st.subheader("Predicted Salary Distribution")
    render_salary_histogram(records)

    with st.expander("Raw prediction data", expanded=False):
        st.dataframe(records, use_container_width=True)

    st.divider()

    st.caption(
        "Trained on 607 data-professional salaries (2020–2022). "
        "Predictions are estimates, not guarantees."
    )

    # Auto-refresh loop
    if auto_refresh and not manual_refresh:
        time.sleep(30)
        st.session_state["overview_records"] = _load_predictions()
        st.rerun()


# ---------------------------------------------------------------------------
# Navigation — explicit sidebar labels and order
# ---------------------------------------------------------------------------

pg = st.navigation([
    st.Page(overview_page, title="Overview", icon=":material/home:"),
    st.Page("pages/reveal_your_true_salary.py", title="Reveal Your True Salary", icon=":material/target:"),
    st.Page("pages/insights.py", title="Insights", icon=":material/bar_chart:"),
])
pg.run()