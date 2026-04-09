"""Streamlit chart wrappers around src/visualizations/charts.py."""

import logging

import pandas as pd
import streamlit as st
from plotly.graph_objects import Figure

logger = logging.getLogger(__name__)

from src.llm.narrative import ChartSpec
from src.visualizations.charts import (
    feature_importance_bar,
    from_chart_spec,
    salary_histogram,
)
from src.visualizations.eda import (
    experience_region_heatmap,
    prediction_volume,
    salary_by_company_size,
    salary_by_experience,
    salary_by_job_family,
    salary_by_region,
    salary_by_remote_ratio,
    salary_density_by_experience,
    salary_stacked_histogram_by_experience,
    salary_trend,
    salary_us_vs_nonus,
)


def render_salary_histogram(
    records: list[dict],
    *,
    spec: ChartSpec | None = None,
    point_estimate: float | None = None,
) -> None:
    """Render a salary histogram inside the current Streamlit column/container."""
    fig: Figure = salary_histogram(records, spec=spec, point_estimate=point_estimate)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_histogram")


def render_feature_importance(
    importances: dict[str, float] | None = None,
    *,
    spec: ChartSpec | None = None,
) -> None:
    """Render a horizontal feature importance bar chart."""
    fig: Figure = feature_importance_bar(importances, spec=spec)
    st.plotly_chart(fig, use_container_width=True, key="chart_feature_importance")


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
        st.plotly_chart(fig, use_container_width=True, key="chart_from_spec")
    except Exception as exc:
        logger.exception("render_chart_from_spec failed")
        st.error(f"Could not render chart: {exc}")


# ---------------------------------------------------------------------------
# EDA chart wrappers — static training data
# ---------------------------------------------------------------------------

def render_salary_by_experience(df: pd.DataFrame) -> None:
    """Render salary distribution by experience level."""
    fig: Figure = salary_by_experience(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_by_experience")


def render_salary_by_region(df: pd.DataFrame) -> None:
    """Render salary distribution by region."""
    fig: Figure = salary_by_region(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_by_region")


def render_salary_by_job_family(df: pd.DataFrame) -> None:
    """Render median salary by job family."""
    fig: Figure = salary_by_job_family(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_by_job_family")


def render_salary_by_remote_ratio(df: pd.DataFrame) -> None:
    """Render salary distribution by remote work arrangement."""
    fig: Figure = salary_by_remote_ratio(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_by_remote_ratio")


def render_salary_trend(df: pd.DataFrame) -> None:
    """Render median salary trend by year."""
    fig: Figure = salary_trend(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_trend")


def render_salary_by_company_size(df: pd.DataFrame) -> None:
    """Render salary distribution by company size."""
    fig: Figure = salary_by_company_size(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_by_company_size")


def render_experience_region_heatmap(df: pd.DataFrame) -> None:
    """Render median salary heatmap by experience level × region."""
    fig: Figure = experience_region_heatmap(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_experience_region_heatmap")


def render_prediction_volume(records: list[dict]) -> None:
    """Render prediction count per calendar day."""
    fig: Figure = prediction_volume(records)
    st.plotly_chart(fig, use_container_width=True, key="chart_prediction_volume")


def render_salary_density_by_experience(df: pd.DataFrame) -> None:
    """Render violin plot of salary density per experience level."""
    fig: Figure = salary_density_by_experience(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_density_by_experience")


def render_salary_stacked_histogram_by_experience(df: pd.DataFrame) -> None:
    """Render stacked histogram of salary by experience level."""
    fig: Figure = salary_stacked_histogram_by_experience(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_stacked_histogram")


def render_salary_us_vs_nonus(df: pd.DataFrame) -> None:
    """Render overlapping histograms comparing US vs non-US company salaries."""
    fig: Figure = salary_us_vs_nonus(df)
    st.plotly_chart(fig, use_container_width=True, key="chart_salary_us_vs_nonus")

