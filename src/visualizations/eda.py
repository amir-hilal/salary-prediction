"""EDA chart functions for the salary prediction dashboard.

Each function accepts a prepared pandas DataFrame and returns a Plotly Figure.
Shared styling constants and _apply_defaults are imported from .charts.
"""

import logging
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from src.visualizations._labels import (
    _EXP_LABELS,
    _EXP_ORDER,
    _FAMILY_LABELS,
    _REGION_LABELS,
    _REGION_ORDER,
    _REMOTE_LABELS,
    _REMOTE_ORDER,
    _SIZE_LABELS,
    _SIZE_ORDER,
)
from src.visualizations.charts import _BRAND_COLOR, _apply_defaults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# salary_by_experience
# ---------------------------------------------------------------------------

def salary_by_experience(df: pd.DataFrame) -> Figure:
    """Box plot of salary distribution across experience levels."""
    if df.empty:
        logger.warning("salary_by_experience | empty DataFrame — returning empty figure")
        return go.Figure()

    plot_df = df.assign(experience_label=df["experience_level"].map(_EXP_LABELS))
    fig = px.box(
        plot_df,
        x="experience_label",
        y="salary_in_usd",
        color_discrete_sequence=[_BRAND_COLOR],
        category_orders={"experience_label": _EXP_ORDER},
        labels={"experience_label": "Experience Level", "salary_in_usd": "Salary (USD)"},
    )
    return _apply_defaults(fig, "Salary by Experience Level", "Experience Level", "Salary (USD)")


# ---------------------------------------------------------------------------
# salary_by_region
# ---------------------------------------------------------------------------

def salary_by_region(df: pd.DataFrame) -> Figure:
    """Box plot of salary by region, coloured by US vs non-US company."""
    if df.empty:
        logger.warning("salary_by_region | empty DataFrame — returning empty figure")
        return go.Figure()

    plot_df = df.assign(
        region_label=df["location_region"].map(_REGION_LABELS),
        is_us_label=df["is_us_company"].map({0: "Non-US company", 1: "US company"}),
    )
    fig = px.box(
        plot_df,
        x="region_label",
        y="salary_in_usd",
        color="is_us_label",
        labels={
            "region_label": "Region",
            "salary_in_usd": "Salary (USD)",
            "is_us_label": "",
        },
    )
    return _apply_defaults(fig, "Salary by Region", "Region", "Salary (USD)")


# ---------------------------------------------------------------------------
# salary_by_job_family
# ---------------------------------------------------------------------------

def salary_by_job_family(df: pd.DataFrame) -> Figure:
    """Horizontal bar chart of median salary per job family, sorted descending."""
    if df.empty:
        logger.warning("salary_by_job_family | empty DataFrame — returning empty figure")
        return go.Figure()

    plot_df = df.assign(family_label=df["job_family"].map(_FAMILY_LABELS))
    median_df = (
        plot_df.groupby("family_label")["salary_in_usd"]
        .median()
        .reset_index()
        .sort_values("salary_in_usd", ascending=True)
    )
    fig = px.bar(
        median_df,
        x="salary_in_usd",
        y="family_label",
        orientation="h",
        color_discrete_sequence=[_BRAND_COLOR],
        labels={"salary_in_usd": "Median Salary (USD)", "family_label": "Job Family"},
    )
    return _apply_defaults(
        fig, "Median Salary by Job Family", "Median Salary (USD)", "Job Family"
    )


# ---------------------------------------------------------------------------
# salary_by_remote_ratio
# ---------------------------------------------------------------------------

def salary_by_remote_ratio(df: pd.DataFrame) -> Figure:
    """Box plot of salary by remote work arrangement."""
    if df.empty:
        logger.warning("salary_by_remote_ratio | empty DataFrame — returning empty figure")
        return go.Figure()

    plot_df = df.assign(remote_label=df["remote_ratio"].map(_REMOTE_LABELS))
    fig = px.box(
        plot_df,
        x="remote_label",
        y="salary_in_usd",
        color_discrete_sequence=[_BRAND_COLOR],
        category_orders={"remote_label": _REMOTE_ORDER},
        labels={"remote_label": "Work Arrangement", "salary_in_usd": "Salary (USD)"},
    )
    return _apply_defaults(fig, "Salary by Remote Ratio", "Work Arrangement", "Salary (USD)")


# ---------------------------------------------------------------------------
# salary_trend
# ---------------------------------------------------------------------------

def salary_trend(df: pd.DataFrame) -> Figure:
    """Line chart of median salary per work_year with annotation at last data point."""
    if df.empty:
        logger.warning("salary_trend | empty DataFrame — returning empty figure")
        return go.Figure()

    trend_df = (
        df.groupby("work_year")["salary_in_usd"]
        .median()
        .reset_index()
        .sort_values("work_year")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_df["work_year"],
            y=trend_df["salary_in_usd"],
            mode="lines+markers",
            line={"color": _BRAND_COLOR},
            marker={"color": _BRAND_COLOR, "size": 8},
            name="Median Salary",
        )
    )

    last_year = trend_df["work_year"].iloc[-1]
    last_salary = trend_df["salary_in_usd"].iloc[-1]
    fig.add_annotation(
        x=last_year,
        y=last_salary,
        text="Training data ends here",
        showarrow=True,
        arrowhead=2,
        xanchor="left",
        yanchor="bottom",
        ax=20,
        ay=-30,
    )

    return _apply_defaults(fig, "Salary Trend by Year", "Year", "Median Salary (USD)")


# ---------------------------------------------------------------------------
# salary_by_company_size
# ---------------------------------------------------------------------------

def salary_by_company_size(df: pd.DataFrame) -> Figure:
    """Box plot of salary by company size."""
    if df.empty:
        logger.warning("salary_by_company_size | empty DataFrame — returning empty figure")
        return go.Figure()

    plot_df = df.assign(size_label=df["company_size"].map(_SIZE_LABELS))
    fig = px.box(
        plot_df,
        x="size_label",
        y="salary_in_usd",
        color_discrete_sequence=[_BRAND_COLOR],
        category_orders={"size_label": _SIZE_ORDER},
        labels={"size_label": "Company Size", "salary_in_usd": "Salary (USD)"},
    )
    return _apply_defaults(fig, "Salary by Company Size", "Company Size", "Salary (USD)")


# ---------------------------------------------------------------------------
# experience_region_heatmap
# ---------------------------------------------------------------------------

def experience_region_heatmap(df: pd.DataFrame) -> Figure:
    """Heatmap of median salary by experience level × region."""
    if df.empty:
        logger.warning("experience_region_heatmap | empty DataFrame — returning empty figure")
        return go.Figure()

    plot_df = df.assign(
        exp_label=df["experience_level"].map(_EXP_LABELS),
        region_label=df["location_region"].map(_REGION_LABELS),
    )

    pivot = (
        plot_df.groupby(["exp_label", "region_label"])["salary_in_usd"]
        .median()
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(index=_EXP_ORDER, columns=_REGION_ORDER, fill_value=0)

    z = pivot.values.tolist()
    text = [[f"${v:,.0f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=_REGION_ORDER,
            y=_EXP_ORDER,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zsmooth=False,
        )
    )
    return _apply_defaults(
        fig, "Median Salary: Experience × Region", "Region", "Experience"
    )


# ---------------------------------------------------------------------------
# prediction_volume
# ---------------------------------------------------------------------------

def prediction_volume(records: list[dict]) -> Figure:
    """Bar chart of prediction count per calendar day."""
    if not records:
        logger.warning("prediction_volume | no records — returning empty figure")
        return go.Figure()

    dates: list = []
    for r in records:
        created_at = r.get("created_at")
        if created_at is None:
            continue
        if isinstance(created_at, str):
            dates.append(datetime.fromisoformat(created_at).date())
        elif isinstance(created_at, datetime):
            dates.append(created_at.date())
        else:
            dates.append(created_at)

    if not dates:
        logger.warning("prediction_volume | no parseable dates — returning empty figure")
        return go.Figure()

    counts: Counter = Counter(dates)
    sorted_dates = sorted(counts.keys())

    fig = go.Figure(
        go.Bar(
            x=sorted_dates,
            y=[counts[d] for d in sorted_dates],
            marker_color=_BRAND_COLOR,
        )
    )
    return _apply_defaults(fig, "Prediction Volume Over Time", "Date", "Predictions")
