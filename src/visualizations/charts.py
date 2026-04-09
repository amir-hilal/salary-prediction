import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from src.llm.narrative import ChartSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

_BRAND_COLOR = "#4F8EF7"
_RANGE_COLOR = "#F7A24F"
_LAYOUT_DEFAULTS: dict = {
    "template": "plotly_white",
    "font": {"family": "Inter, sans-serif", "size": 13},
    "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
}


def _apply_defaults(fig: Figure, title: str, x_label: str, y_label: str) -> Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# salary_histogram
# ---------------------------------------------------------------------------

def salary_histogram(
    records: list[dict],
    *,
    spec: ChartSpec | None = None,
    point_estimate: float | None = None,
) -> Figure:
    """Histogram of predicted salaries from a list of prediction records.

    Args:
        records: list of dicts; each must have a ``predicted_salary`` key.
        spec: optional ChartSpec from the LLM — overrides title/axis labels.
        point_estimate: if provided, a vertical line marks this prediction on the chart.

    Returns:
        A Plotly Figure ready for ``st.plotly_chart()``.
    """
    if not records:
        logger.warning("salary_histogram | no records — returning empty figure")
        return go.Figure()

    df = pd.DataFrame(records)
    title = spec.title if spec else "Predicted Salary Distribution"
    x_label = spec.x_label if spec else "Predicted Salary (USD)"
    y_label = spec.y_label if spec else "Count"

    fig = px.histogram(
        df,
        x="predicted_salary",
        nbins=30,
        color_discrete_sequence=[_BRAND_COLOR],
        labels={"predicted_salary": x_label},
    )

    if point_estimate is not None:
        fig.add_vline(
            x=point_estimate,
            line_dash="dash",
            line_color=_RANGE_COLOR,
            annotation_text=f"Your prediction: ${point_estimate:,.0f}",
            annotation_position="top right",
        )

    return _apply_defaults(fig, title, x_label, y_label)


# ---------------------------------------------------------------------------
# predicted_vs_actual_scatter
# ---------------------------------------------------------------------------

def predicted_vs_actual_scatter(
    records: list[dict],
    *,
    spec: ChartSpec | None = None,
) -> Figure:
    """Scatter plot of predicted salary vs. a reference value in each record.

    Args:
        records: list of dicts with ``predicted_salary`` and ``salary_range_low`` /
                 ``salary_range_high`` keys (used as error bars representing the
                 peer-group IQR).
        spec: optional ChartSpec for title/axis overrides.

    Returns:
        A Plotly Figure ready for ``st.plotly_chart()``.
    """
    if not records:
        logger.warning("predicted_vs_actual_scatter | no records — returning empty figure")
        return go.Figure()

    df = pd.DataFrame(records)
    title = spec.title if spec else "Predicted Salary with Peer-Group Range"
    x_label = spec.x_label if spec else "Prediction Index"
    y_label = spec.y_label if spec else "Salary (USD)"

    # Build IQR error bars when range columns are present.
    error_minus = None
    error_plus = None
    if "salary_range_low" in df.columns and "salary_range_high" in df.columns:
        df["_low"] = df["salary_range_low"].fillna(df["predicted_salary"])
        df["_high"] = df["salary_range_high"].fillna(df["predicted_salary"])
        error_minus = df["predicted_salary"] - df["_low"]
        error_plus = df["_high"] - df["predicted_salary"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(df))),
            y=df["predicted_salary"],
            mode="markers",
            marker={"color": _BRAND_COLOR, "size": 8},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": error_plus.tolist() if error_plus is not None else [],
                "arrayminus": error_minus.tolist() if error_minus is not None else [],
                "color": _RANGE_COLOR,
                "thickness": 1.5,
            },
            name="Predicted ± IQR",
        )
    )

    return _apply_defaults(fig, title, x_label, y_label)


# ---------------------------------------------------------------------------
# feature_importance_bar
# ---------------------------------------------------------------------------

# Static feature importance proxy: fraction of salary variance attributable to
# each feature based on domain knowledge and EDA findings. Updated when a new
# model artifact is trained; used when real importances are unavailable.
_FEATURE_IMPORTANCE_FALLBACK: dict[str, float] = {
    "experience_level": 0.35,
    "is_us_company": 0.20,
    "location_region": 0.15,
    "job_family": 0.12,
    "remote_ratio": 0.08,
    "company_size": 0.05,
    "employment_type": 0.03,
    "work_year": 0.02,
}


def feature_importance_bar(
    importances: dict[str, float] | None = None,
    *,
    spec: ChartSpec | None = None,
) -> Figure:
    """Horizontal bar chart of feature importances.

    Args:
        importances: dict mapping feature name → importance score. Falls back
                     to domain-knowledge estimates if None or empty.
        spec: optional ChartSpec for title/axis overrides.

    Returns:
        A Plotly Figure ready for ``st.plotly_chart()``.
    """
    data = importances or _FEATURE_IMPORTANCE_FALLBACK
    title = spec.title if spec else "Feature Importance"
    x_label = spec.x_label if spec else "Importance Score"
    y_label = spec.y_label if spec else "Feature"

    df = (
        pd.DataFrame(list(data.items()), columns=["feature", "importance"])
        .sort_values("importance", ascending=True)
    )

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[_BRAND_COLOR],
        labels={"importance": x_label, "feature": y_label},
    )

    return _apply_defaults(fig, title, x_label, y_label)



# ---------------------------------------------------------------------------
# from_chart_spec  — dispatch helper used by the dashboard
# ---------------------------------------------------------------------------

def from_chart_spec(
    spec: ChartSpec,
    records: list[dict],
    *,
    point_estimate: float | None = None,
    importances: dict[str, float] | None = None,
) -> Figure:
    """Dispatch to the correct chart function based on ChartSpec.data_key.

    The dashboard calls this single function; it routes to the right chart
    based on the ``data_key`` the LLM included in its response.

    Args:
        spec: the parsed ChartSpec from the LLM narrative.
        records: recent prediction records from Supabase.
        point_estimate: marks the current prediction on histogram charts.
        importances: real feature importances from the model artifact, if available.

    Returns:
        A Plotly Figure.
    """
    key = spec.data_key.lower()

    if "feature" in key or "importance" in key:
        return feature_importance_bar(importances, spec=spec)

    if "scatter" in key or "actual" in key or "vs" in key:
        return predicted_vs_actual_scatter(records, spec=spec)

    # Default: histogram
    return salary_histogram(records, spec=spec, point_estimate=point_estimate)
