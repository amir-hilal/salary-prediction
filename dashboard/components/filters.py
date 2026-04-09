"""Sidebar filter widgets and typed FilterState model."""

from datetime import date, timedelta

import streamlit as st
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Typed state
# ---------------------------------------------------------------------------

class FilterState(BaseModel):
    """All active filter values from the sidebar.

    ``None`` means "no filter applied" for that dimension.
    """

    date_from: date | None = None
    date_to: date | None = None
    experience_level: list[int] | None = None  # 0–3
    location_region: list[int] | None = None   # 0–3
    job_family: list[int] | None = None         # 0–5


# ---------------------------------------------------------------------------
# Label maps — kept in sync with PredictionRequest schema
# ---------------------------------------------------------------------------

_EXPERIENCE_LABELS: dict[str, int] = {
    "Entry-level": 0,
    "Mid-level": 1,
    "Senior": 2,
    "Executive": 3,
}
_REGION_LABELS: dict[str, int] = {
    "Rest of World": 0,
    "Asia Pacific": 1,
    "Europe": 2,
    "North America": 3,
}
_JOB_FAMILY_LABELS: dict[str, int] = {
    "Other": 0,
    "Analytics": 1,
    "Data Science": 2,
    "Data Engineering": 3,
    "ML / AI": 4,
    "Leadership": 5,
}


def render_sidebar_filters() -> FilterState:
    """Render all sidebar filter widgets and return a typed FilterState.

    Must be called once per page render; the returned FilterState is passed
    to query functions to build filtered Supabase queries.
    """
    with st.sidebar:
        st.header("Filters")

        # Date range
        st.subheader("Date range")
        default_from = date.today() - timedelta(days=30)
        date_from: date = st.date_input("From", value=default_from)  # type: ignore[assignment]
        date_to: date = st.date_input("To", value=date.today())  # type: ignore[assignment]

        # Experience level
        st.subheader("Experience level")
        exp_selection: list[str] = st.multiselect(
            "Select levels",
            options=list(_EXPERIENCE_LABELS),
            default=list(_EXPERIENCE_LABELS),
            key="filter_experience",
        )
        experience_level = [_EXPERIENCE_LABELS[e] for e in exp_selection] or None

        # Location region
        st.subheader("Location region")
        region_selection: list[str] = st.multiselect(
            "Select regions",
            options=list(_REGION_LABELS),
            default=list(_REGION_LABELS),
            key="filter_region",
        )
        location_region = [_REGION_LABELS[r] for r in region_selection] or None

        # Job family
        st.subheader("Job family")
        family_selection: list[str] = st.multiselect(
            "Select families",
            options=list(_JOB_FAMILY_LABELS),
            default=list(_JOB_FAMILY_LABELS),
            key="filter_job_family",
        )
        job_family = [_JOB_FAMILY_LABELS[f] for f in family_selection] or None

    return FilterState(
        date_from=date_from,
        date_to=date_to,
        experience_level=experience_level,
        location_region=location_region,
        job_family=job_family,
    )
