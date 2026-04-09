"""Salary Prediction Dashboard — entry point.

Streamlit multi-page app.  This file sets global page config and seeds
``st.session_state`` with the Supabase anon client so all pages share a
single connection without re-creating it on every rerun.
"""

import streamlit as st

from config.settings import settings
from src.database.client import get_anon_client


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
# Hero section
# ---------------------------------------------------------------------------

st.title("Salary Prediction for Data Professionals")
st.markdown(
    "Get an instant salary estimate based on your experience, role, and location "
    "— powered by machine learning and explained by AI."
)

st.divider()

# ---------------------------------------------------------------------------
# Feature cards
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# CTA — centred button with shimmer animation
# ---------------------------------------------------------------------------

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
  <a href="/predictions" target="_self">Make a Prediction →</a>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

st.markdown("#### How it works")
st.markdown(
    "1. **Fill in your profile** — experience, role, company, and location.\n"
    "2. **Get an estimate** — a point prediction plus a Q25–Q75 peer-group range.\n"
    "3. **Read the narrative** — an AI explanation of the factors behind the number."
)

st.divider()

st.caption(
    "Trained on 607 data-professional salaries (2020–2022). "
    "Predictions are estimates, not guarantees."
)
