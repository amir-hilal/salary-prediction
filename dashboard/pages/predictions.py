"""Page 2 — Prediction Explorer.

Guided stepper form → FastAPI /predict call → salary metrics → SSE narrative
stream → structured display + chart.
"""

import logging

import httpx
import streamlit as st

from config.settings import settings
from dashboard.components.charts import render_chart_from_spec
from src.database.crud import get_narrative_for_prediction, get_recent_predictions
from src.llm.narrative import ChartSpec

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Prediction Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Label maps (code → human-readable)
# ---------------------------------------------------------------------------

_EXP_LABELS = {0: "Entry-level", 1: "Mid-level", 2: "Senior", 3: "Executive"}
_EMP_LABELS = {0: "Part-time", 1: "Freelance", 2: "Contract", 3: "Full-time"}
_REMOTE_LABELS = {0: "On-site", 50: "Hybrid", 100: "Remote"}
_SIZE_LABELS = {0: "Small", 1: "Medium", 2: "Large"}
_FAMILY_LABELS = {
    0: "Other", 1: "Analytics", 2: "Data Science",
    3: "Data Engineering", 4: "ML / AI", 5: "Leadership",
}
_REGION_LABELS = {0: "Rest of World", 1: "Asia Pacific", 2: "Europe", 3: "North America"}
_US_LABELS = {0: "No", 1: "Yes"}

# Pill badge colours (CSS background / border pairs for the profile summary)
_PILL_STYLES = [
    ("#e8f0fe", "#4F8EF7", "#1a56db"),  # blue
    ("#e6f4ea", "#34a853", "#1e7e34"),  # green
    ("#fef3e2", "#f7a24f", "#c47a1a"),  # orange
    ("#f3e8fd", "#9b59b6", "#6c3082"),  # violet
    ("#fce8e8", "#e74c3c", "#b71c1c"),  # red
    ("#eef0f2", "#8e99a4", "#5a6370"),  # gray
]

_TOTAL_STEPS = 4


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_stepper() -> None:
    defaults: dict = {
        "step": 1,
        "ss_experience_level": 2,
        "ss_employment_type": 3,
        "ss_job_family": 2,
        "ss_remote_ratio": 100,
        "ss_company_size": 1,
        "ss_location_region": 3,
        "ss_is_us_company": 1,
        "ss_work_year": 2026,
        # Result state
        "prediction_result": None,
        "prediction_payload": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_stepper()


def _reset_stepper() -> None:
    for key in [
        "step", "ss_experience_level", "ss_employment_type", "ss_job_family",
        "ss_remote_ratio", "ss_company_size", "ss_location_region",
        "ss_is_us_company", "ss_work_year", "prediction_result",
        "prediction_payload",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def _go_next() -> None:
    st.session_state.step = min(st.session_state.step + 1, _TOTAL_STEPS)


def _go_back() -> None:
    st.session_state.step = max(st.session_state.step - 1, 1)


def _profile_parts() -> list[str]:
    """Return the list of human-readable profile labels."""
    parts = [
        _EXP_LABELS[st.session_state.ss_experience_level],
        _EMP_LABELS[st.session_state.ss_employment_type],
        _REMOTE_LABELS[st.session_state.ss_remote_ratio],
        _FAMILY_LABELS[st.session_state.ss_job_family],
        _REGION_LABELS[st.session_state.ss_location_region],
        f"{_SIZE_LABELS[st.session_state.ss_company_size]} company",
    ]
    if st.session_state.ss_location_region == 3:  # North America
        parts.append("US company" if st.session_state.ss_is_us_company else "Non-US company")
    return parts


def _render_pills() -> None:
    """Render profile labels as styled pill badges using HTML."""
    parts = _profile_parts()
    spans = []
    for i, label in enumerate(parts):
        bg, border, text = _PILL_STYLES[i % len(_PILL_STYLES)]
        spans.append(
            f'<span style="display:inline-block;padding:4px 14px;margin:3px 4px;'
            f'border-radius:999px;border:1.5px solid {border};'
            f'background:{bg};color:{text};font-size:0.85rem;'
            f'font-weight:600;line-height:1.4;">{label}</span>'
        )
    st.markdown("".join(spans), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page header (only shown during stepper steps, not results)
# ---------------------------------------------------------------------------

# ===== RESULTS VIEW (step == 5) ===========================================

if st.session_state.step == 5:
    result = st.session_state.prediction_result
    if result is None:
        _reset_stepper()
        st.rerun()

    salary = result["salary"]
    prediction_id: str = result["prediction_id"]

    # -- Profile pills ------------------------------------------------------
    _render_pills()

    st.divider()

    # -- Horizontal salary range bar ----------------------------------------
    low, mean, high = salary["low"], salary["mean"], salary["high"]
    span = high - low if high != low else 1.0
    pct = ((mean - low) / span) * 100

    st.markdown(
        f"""
<div style="margin:1.5rem 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#888;margin-bottom:4px;">
    <span>Peer range (low)</span>
    <span>Predicted salary</span>
    <span>Peer range (high)</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-weight:700;font-size:1.15rem;margin-bottom:8px;">
    <span>${low:,.0f}</span>
    <span>${mean:,.0f}</span>
    <span>${high:,.0f}</span>
  </div>
  <div style="position:relative;height:12px;background:linear-gradient(90deg,#4F8EF7 0%,#F7A24F 100%);border-radius:6px;">
    <div style="
      position:absolute;
      left:{pct:.1f}%;
      top:50%;
      transform:translate(-50%,-50%);
      width:18px;height:18px;
      background:#fff;
      border:3px solid #4F8EF7;
      border-radius:50%;
    "></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # -- LLM Narrative — streamed via SSE -----------------------------------
    status = st.status("Generating narrative…", expanded=True)
    narrative_placeholder = status.empty()
    stream_error = False

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            with client.stream(
                "GET",
                f"{settings.api_base_url}/api/v1/predict/{prediction_id}/narrative",
            ) as stream_response:
                stream_response.raise_for_status()
                accumulated = ""
                for line in stream_response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    token = line[len("data: "):]
                    if token == "[DONE]":
                        break
                    if token.startswith("[ERROR]"):
                        narrative_placeholder.error(
                            f"Narrative generation failed: {token[len('[ERROR] '):]}"
                        )
                        stream_error = True
                        break
                    accumulated += token.replace("\\n", "\n")
                    narrative_placeholder.markdown(accumulated)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            narrative_placeholder.error("Prediction not found. Please try again.")
        else:
            narrative_placeholder.error(
                f"Narrative stream failed ({exc.response.status_code})."
            )
        stream_error = True
    except httpx.RequestError as exc:
        narrative_placeholder.error(f"Could not reach the API. ({exc})")
        stream_error = True

    if stream_error:
        status.update(label="Narrative failed", state="error", expanded=False)
    else:
        status.update(label="Narrative complete", state="complete", expanded=False)

    # -- Structured narrative (from Supabase) after stream ------------------
    if not stream_error:
        narrative = None
        try:
            narrative = get_narrative_for_prediction(prediction_id)
        except Exception as exc:
            logger.warning("Could not fetch narrative record: %s", exc)

        if narrative is not None:
            st.subheader("AI Narrative")

            st.markdown(f"**Summary**\n\n{narrative.summary}")
            st.markdown(f"**Uncertainty**\n\n{narrative.uncertainty}")

            if narrative.insights:
                st.markdown("**Key Insights**")
                for insight in narrative.insights:
                    st.markdown(f"- {insight}")

            st.markdown(f"**Recommendation**\n\n{narrative.recommendation}")

            st.divider()

            # -- Chart ------------------------------------------------------
            st.subheader("Salary Distribution")
            recent_records = [r.model_dump() for r in get_recent_predictions(limit=200)]
            chart_spec = (
                ChartSpec(**narrative.chart_spec)
                if isinstance(narrative.chart_spec, dict)
                else narrative.chart_spec
            )
            render_chart_from_spec(
                chart_spec, recent_records, point_estimate=salary["mean"],
            )

    # -- New prediction button + metadata footer -----------------------------
    st.divider()
    st.caption(
        f"Work year: {st.session_state.ss_work_year} · "
        f"Model: `{result['model_version']}` · "
        f"ID: `{prediction_id}`"
    )
    if st.button("New Prediction", type="primary"):
        _reset_stepper()
        st.rerun()

    st.stop()

# ===== STEPPER VIEW (steps 1–4) ===========================================

st.title("Prediction Explorer")
st.caption("Answer a few questions and get an AI-explained salary estimate.")

# Progress bar
st.progress(st.session_state.step / _TOTAL_STEPS)
st.markdown(
    f"**Step {st.session_state.step} of {_TOTAL_STEPS}** — "
    + ["About You", "Your Role", "Company & Location", "Review & Predict"][
        st.session_state.step - 1
    ]
)

# ---------------------------------------------------------------------------
# Step 1 — About You
# ---------------------------------------------------------------------------

if st.session_state.step == 1:
    st.session_state.ss_experience_level = st.selectbox(
        "Experience level",
        options=list(_EXP_LABELS.keys()),
        format_func=lambda x: _EXP_LABELS[x],
        index=list(_EXP_LABELS.keys()).index(st.session_state.ss_experience_level),
        key="w_experience_level",
    )
    st.session_state.ss_employment_type = st.selectbox(
        "Employment type",
        options=list(_EMP_LABELS.keys()),
        format_func=lambda x: _EMP_LABELS[x],
        index=list(_EMP_LABELS.keys()).index(st.session_state.ss_employment_type),
        key="w_employment_type",
    )

    st.button("Next →", on_click=_go_next, type="primary")

# ---------------------------------------------------------------------------
# Step 2 — Your Role
# ---------------------------------------------------------------------------

elif st.session_state.step == 2:
    st.session_state.ss_job_family = st.selectbox(
        "Job family",
        options=list(_FAMILY_LABELS.keys()),
        format_func=lambda x: _FAMILY_LABELS[x],
        index=list(_FAMILY_LABELS.keys()).index(st.session_state.ss_job_family),
        key="w_job_family",
    )
    st.session_state.ss_remote_ratio = st.selectbox(
        "Remote ratio",
        options=[0, 50, 100],
        format_func=lambda x: _REMOTE_LABELS[x],
        index=[0, 50, 100].index(st.session_state.ss_remote_ratio),
        key="w_remote_ratio",
    )

    c_back, c_next = st.columns(2)
    c_back.button("← Back", on_click=_go_back)
    c_next.button("Next →", on_click=_go_next, type="primary")

# ---------------------------------------------------------------------------
# Step 3 — Company & Location
# ---------------------------------------------------------------------------

elif st.session_state.step == 3:
    st.session_state.ss_company_size = st.selectbox(
        "Company size",
        options=list(_SIZE_LABELS.keys()),
        format_func=lambda x: _SIZE_LABELS[x],
        index=list(_SIZE_LABELS.keys()).index(st.session_state.ss_company_size),
        key="w_company_size",
    )
    st.session_state.ss_location_region = st.selectbox(
        "Location region",
        options=list(_REGION_LABELS.keys()),
        format_func=lambda x: _REGION_LABELS[x],
        index=list(_REGION_LABELS.keys()).index(st.session_state.ss_location_region),
        key="w_location_region",
    )

    # Conditional: show is_us_company only for North America (3)
    if st.session_state.ss_location_region == 3:
        st.session_state.ss_is_us_company = st.radio(
            "US-based company?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x else "No",
            index=[1, 0].index(st.session_state.ss_is_us_company),
            horizontal=True,
            key="w_is_us_company",
        )
    else:
        st.session_state.ss_is_us_company = 0

    c_back, c_next = st.columns(2)
    c_back.button("← Back", on_click=_go_back)
    c_next.button("Next →", on_click=_go_next, type="primary")

# ---------------------------------------------------------------------------
# Step 4 — Review & Predict
# ---------------------------------------------------------------------------

elif st.session_state.step == 4:
    st.markdown("**" + " · ".join(_profile_parts()) + "**")

    st.session_state.ss_work_year = st.number_input(
        "Work year",
        min_value=2020,
        max_value=2030,
        value=st.session_state.ss_work_year,
        key="w_work_year",
    )
    st.warning(
        "This model was trained on salary data from 2020, 2021, and 2022. "
        "Predictions for other years are extrapolated and may be less reliable."
    )

    c_back, c_predict = st.columns(2)
    c_back.button("← Back", on_click=_go_back)

    if c_predict.button("Predict", type="primary"):
        payload = {
            "experience_level": st.session_state.ss_experience_level,
            "employment_type": st.session_state.ss_employment_type,
            "remote_ratio": st.session_state.ss_remote_ratio,
            "company_size": st.session_state.ss_company_size,
            "work_year": int(st.session_state.ss_work_year),
            "job_family": st.session_state.ss_job_family,
            "location_region": st.session_state.ss_location_region,
            "is_us_company": st.session_state.ss_is_us_company,
        }

        with st.status("Predicting…", expanded=True) as status:
            try:
                response = httpx.post(
                    f"{settings.api_base_url}/api/v1/predict",
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 422:
                    st.error(f"Validation error — check your inputs. ({exc.response.text})")
                else:
                    st.error(f"Server error {exc.response.status_code}: {exc.response.text}")
                status.update(label="Prediction failed", state="error", expanded=False)
                st.stop()
            except httpx.RequestError as exc:
                st.error(f"Could not reach the API. Is it running? ({exc})")
                status.update(label="Prediction failed", state="error", expanded=False)
                st.stop()
            except ValueError as exc:
                st.error(f"API returned invalid JSON. ({exc})")
                status.update(label="Prediction failed", state="error", expanded=False)
                st.stop()

        if "salary" not in result or "prediction_id" not in result:
            st.error("Unexpected API response. Please try again.")
            st.stop()
        salary = result["salary"]
        if not isinstance(salary, dict) or not {"mean", "low", "high"}.issubset(salary):
            st.error("Salary data is incomplete. Please try again.")
            st.stop()

        st.session_state.prediction_result = result
        st.session_state.prediction_payload = payload
        st.session_state.step = 5
        st.rerun()

# ---------------------------------------------------------------------------
# Start-over button (visible on steps 2–4)
# ---------------------------------------------------------------------------

if st.session_state.step > 1:
    st.divider()
    if st.button("Start over"):
        _reset_stepper()
        st.rerun()
