"""Streamlit chart wrappers around src/visualizations/charts.py."""

import logging

import streamlit as st
from plotly.graph_objects import Figure

logger = logging.getLogger(__name__)

from src.llm.narrative import ChartSpec
from src.visualizations.charts import (
    feature_importance_bar,
    from_chart_spec,
    salary_histogram,
)


def render_salary_histogram(
    records: list[dict],
    *,
    spec: ChartSpec | None = None,
    point_estimate: float | None = None,
) -> None:
    """Render a salary histogram inside the current Streamlit column/container."""
    fig: Figure = salary_histogram(records, spec=spec, point_estimate=point_estimate)
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(
    importances: dict[str, float] | None = None,
    *,
    spec: ChartSpec | None = None,
) -> None:
    """Render a horizontal feature importance bar chart."""
    fig: Figure = feature_importance_bar(importances, spec=spec)
    st.plotly_chart(fig, use_container_width=True)


def render_chart_from_spec(
    spec: ChartSpec,
    records: list[dict],
    *,
    point_estimate: float | None = None,
    importances: dict[str, float] | None = None,
) -> None:
    """Dispatch to the right chart type and render it using the LLM ChartSpec."""
    try:
        fig: Figure = from_chart_spec(
            spec,
            records,
            point_estimate=point_estimate,
            importances=importances,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        logger.exception("render_chart_from_spec failed")
        st.error(f"Could not render chart: {exc}")
