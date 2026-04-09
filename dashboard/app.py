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
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_session_state()

st.title("💼 Salary Prediction")
st.caption(
    "Explore predicted salaries, submit new predictions, and read AI-generated "
    "narratives.  Use the sidebar to navigate between pages."
)
