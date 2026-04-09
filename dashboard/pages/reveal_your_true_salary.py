"""Page 2 — Prediction Explorer.

Guided stepper form → FastAPI /predict call → salary metrics → SSE narrative
stream → structured display + chart.
"""

import logging
import re

import httpx
import streamlit as st

from config.settings import settings
from dashboard.components.charts import render_chart_from_spec
from src.database.crud import get_narrative_for_prediction, get_recent_predictions
from src.llm.narrative import ChartSpec

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Reveal Your True Salary", layout="wide")

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

# Regex for on-the-fly stream formatting
_SECTION_HEAD_RE = re.compile(
    r"^\d+\.\s*(SUMMARY|UNCERTAINTY|INSIGHTS|COMPARISON|RECOMMENDATION)\s*$",
    re.MULTILINE,
)
_CHART_BLOCK_RE = re.compile(
    r"(?:^\d+\.\s*CHART\s*\n)?\[CHART\].*?(?:\[/CHART\]|$)",
    re.DOTALL | re.MULTILINE,
)
_SECTION_NICE = {
    "SUMMARY": "**Summary**",
    "UNCERTAINTY": "**Uncertainty**",
    "INSIGHTS": "**Key Insights**",
    "COMPARISON": "**Comparison**",
    "RECOMMENDATION": "**Recommendation**",
}


def _format_stream(raw: str) -> str:
    """Transform raw LLM output into clean markdown for live display."""
    text = _CHART_BLOCK_RE.sub("", raw)
    # Remove a bare "5. CHART" header if left over
    text = re.sub(r"^\d+\.\s*CHART\s*$", "", text, flags=re.MULTILINE)
    def _replace_header(m: re.Match) -> str:
        return _SECTION_NICE.get(m.group(1).upper(), m.group(0))
    text = _SECTION_HEAD_RE.sub(_replace_header, text)
    # Convert • bullets to markdown -
    text = re.sub(r"^\s*•\s*", "- ", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Shimmer skeleton + salary range bar HTML helpers
# ---------------------------------------------------------------------------

_SHIMMER_BAR_HTML = """
<style>
@keyframes sal-shimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.sal-shim {
  background: linear-gradient(110deg,
    rgba(150,150,150,0.15) 40%,
    rgba(150,150,150,0.35) 50%,
    rgba(150,150,150,0.15) 60%);
  background-size: 200% 100%;
  animation: sal-shimmer 1.5s infinite linear;
  border-radius: 4px;
}
</style>
<div style="margin:1.5rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:8px;">
    <div>
      <div class="sal-shim" style="width:75px;height:12px;margin-bottom:6px;"></div>
      <div class="sal-shim" style="width:95px;height:20px;margin-top:12px;"></div>
    </div>
    <div style="text-align:center;">
      <div class="sal-shim" style="width:85px;height:12px;margin:0 auto 6px;"></div>
      <div class="sal-shim" style="width:130px;height:30px;margin:0 auto;margin-top:12px;"></div>
    </div>
    <div style="text-align:right;">
      <div class="sal-shim" style="width:75px;height:12px;margin-bottom:6px;margin-left:auto;"></div>
      <div class="sal-shim" style="width:95px;height:20px;margin-left:auto;margin-top:12px;"></div>
    </div>
  </div>
  <div class="sal-shim" style="height:12px;border-radius:6px;margin-top:12px;"></div>
</div>
"""


def _salary_bar_html(salary: dict) -> str:
    """Return HTML for the horizontal salary range bar with a predicted-salary dot."""
    low, mean, high = salary["low"], salary["mean"], salary["high"]
    span = high - low if high != low else 1.0
    pct = ((mean - low) / span) * 100
    return f"""
<div style="margin:1.5rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:8px;">
    <div>
      <div style="font-size:0.8rem;color:#888;margin-bottom:2px;">Peer range (low)</div>
      <div style="font-weight:700;font-size:1.1rem;">${low:,.0f}</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.8rem;color:#888;margin-bottom:2px;">Predicted salary</div>
      <div style="font-weight:700;font-size:1.8rem;color:#4F8EF7;">${mean:,.0f}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.8rem;color:#888;margin-bottom:2px;">Peer range (high)</div>
      <div style="font-weight:700;font-size:1.1rem;">${high:,.0f}</div>
    </div>
  </div>
  <div style="position:relative;height:12px;background:linear-gradient(90deg,#4F8EF7 0%,#F7A24F 100%);border-radius:6px;">
    <div style="position:absolute;left:{pct:.1f}%;top:50%;transform:translate(-50%,-50%);
      width:18px;height:18px;background:#fff;border:3px solid #4F8EF7;border-radius:50%;"></div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_stepper() -> None:
    defaults: dict = {
        "step": 1,
        "ss_experience_level": None,
        "ss_employment_type": None,
        "ss_job_family": None,
        "ss_remote_ratio": None,
        "ss_company_size": None,
        "ss_location_region": None,
        "ss_is_us_company": None,
        "ss_work_year": 2026,
        # Result state
        "prediction_result": None,
        "prediction_payload": None,
        "narrative_done": False,
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
        "prediction_payload", "narrative_done",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def _go_next() -> None:
    st.session_state.step = min(st.session_state.step + 1, _TOTAL_STEPS)


def _go_back() -> None:
    st.session_state.step = max(st.session_state.step - 1, 1)


def _on_predict() -> None:
    """on_click callback for the Predict button — runs before the rerun renders."""
    st.session_state.prediction_payload = {
        "experience_level": st.session_state.ss_experience_level,
        "employment_type": st.session_state.ss_employment_type,
        "remote_ratio": st.session_state.ss_remote_ratio,
        "company_size": st.session_state.ss_company_size,
        "work_year": int(st.session_state.ss_work_year),
        "job_family": st.session_state.ss_job_family,
        "location_region": st.session_state.ss_location_region,
        "is_us_company": st.session_state.ss_is_us_company,
    }
    st.session_state.prediction_result = None
    st.session_state.narrative_done = False
    st.session_state.step = 5


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
    # -- Profile pills (always first) --------------------------------------
    _render_pills()
    st.divider()

    # Pre-allocate ALL page slots immediately to clear old step-4 content.
    salary_bar = st.empty()
    _div1 = st.empty()
    canvas = st.empty()
    footer_div = st.empty()
    footer_meta = st.empty()
    footer_btn = st.empty()

    # -- Salary range bar ---------------------------------------------------
    if st.session_state.prediction_result is None:
        # Show shimmer skeleton while the API call is in-flight
        salary_bar.markdown(_SHIMMER_BAR_HTML, unsafe_allow_html=True)

        payload = st.session_state.prediction_payload
        try:
            response = httpx.post(
                f"{settings.api_base_url}/api/v1/predict",
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as exc:
            salary_bar.empty()
            url = f"{settings.api_base_url}/api/v1/predict"
            if exc.response.status_code == 422:
                st.error(f"Validation error — check your inputs. ({exc.response.text})")
            else:
                st.error(
                    f"API error {exc.response.status_code} from `{url}`: "
                    f"{exc.response.text}"
                )
            st.stop()
        except httpx.RequestError as exc:
            salary_bar.empty()
            url = f"{settings.api_base_url}/api/v1/predict"
            st.error(
                f"Could not reach the API at `{url}`. "
                f"Is it running? ({exc})"
            )
            st.stop()
        except ValueError as exc:
            salary_bar.empty()
            st.error(f"API returned invalid JSON. ({exc})")
            st.stop()

        if "salary" not in result or "prediction_id" not in result:
            salary_bar.empty()
            st.error("Unexpected API response. Please try again.")
            st.stop()
        salary_data = result["salary"]
        if not isinstance(salary_data, dict) or not {"mean", "low", "high"}.issubset(salary_data):
            salary_bar.empty()
            st.error("Salary data is incomplete. Please try again.")
            st.stop()

        st.session_state.prediction_result = result

    result = st.session_state.prediction_result
    salary = result["salary"]
    prediction_id: str = result["prediction_id"]

    # Replace shimmer with real salary bar
    salary_bar.markdown(_salary_bar_html(salary), unsafe_allow_html=True)
    _div1.divider()

    # -- Narrative section (all inside the canvas) -------------------------
    if not st.session_state.get("narrative_done", False):
        # Stream SSE tokens into the canvas — formatted on-the-fly.
        with canvas.status("Thinking\u2026", expanded=True) as _status:
            _stream_md = st.empty()
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
                                stream_error = True
                                break
                            accumulated += token.replace("\\n", "\n")
                            _stream_md.markdown(_format_stream(accumulated))
            except httpx.HTTPStatusError:
                stream_error = True
            except httpx.RequestError:
                stream_error = True

            # Render chart inside the canvas after stream completes
            if not stream_error:
                narrative = None
                try:
                    narrative = get_narrative_for_prediction(prediction_id)
                except Exception as exc:
                    logger.warning("Could not fetch narrative record: %s", exc)

                if narrative is not None:
                    st.divider()
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

        if stream_error:
            canvas.status("Narrative failed", state="error", expanded=False)
            st.error("Could not generate narrative. Please try a new prediction.")
        else:
            canvas.status("Analysis done", state="complete", expanded=True)
            st.session_state.narrative_done = True
    else:
        # Subsequent reruns — narrative already generated, render inside canvas
        with canvas.status("Analysis done", state="complete", expanded=True):
            narrative = None
            try:
                narrative = get_narrative_for_prediction(prediction_id)
            except Exception as exc:
                logger.warning("Could not fetch narrative record: %s", exc)

            if narrative is not None:
                st.markdown(f"**Summary**\n\n{narrative.summary}")
                st.markdown(f"**Uncertainty**\n\n{narrative.uncertainty}")
                if narrative.insights:
                    st.markdown("**Key Insights**")
                    for insight in narrative.insights:
                        st.markdown(f"- {insight}")
                st.markdown(f"**Recommendation**\n\n{narrative.recommendation}")

                st.divider()
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

    # -- Footer: metadata + new prediction ----------------------------------
    footer_div.divider()
    footer_meta.caption(
        f"Work year: {st.session_state.ss_work_year} · "
        f"Model: `{result['model_version']}` · "
        f"ID: `{prediction_id}`"
    )
    if footer_btn.button("New Prediction", type="primary"):
        _reset_stepper()
        st.rerun()

    st.stop()

# ===== STEPPER VIEW (steps 1–4) ===========================================

if st.session_state.step >= 5:
    st.stop()

st.title("Reveal Your True Salary")
st.caption("Answer a few questions and discover what you should really be earning.")

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
        index=None if st.session_state.ss_experience_level is None else list(_EXP_LABELS.keys()).index(st.session_state.ss_experience_level),
        key="w_experience_level",
    )
    st.session_state.ss_employment_type = st.selectbox(
        "Employment type",
        options=list(_EMP_LABELS.keys()),
        format_func=lambda x: _EMP_LABELS[x],
        index=None if st.session_state.ss_employment_type is None else list(_EMP_LABELS.keys()).index(st.session_state.ss_employment_type),
        key="w_employment_type",
    )

    _step1_ready = st.session_state.ss_experience_level is not None and st.session_state.ss_employment_type is not None
    st.button("Next →", on_click=_go_next, type="primary", disabled=not _step1_ready)

# ---------------------------------------------------------------------------
# Step 2 — Your Role
# ---------------------------------------------------------------------------

elif st.session_state.step == 2:
    st.session_state.ss_job_family = st.selectbox(
        "Job family",
        options=list(_FAMILY_LABELS.keys()),
        format_func=lambda x: _FAMILY_LABELS[x],
        index=None if st.session_state.ss_job_family is None else list(_FAMILY_LABELS.keys()).index(st.session_state.ss_job_family),
        key="w_job_family",
    )
    st.session_state.ss_remote_ratio = st.selectbox(
        "Remote ratio",
        options=[0, 50, 100],
        format_func=lambda x: _REMOTE_LABELS[x],
        index=None if st.session_state.ss_remote_ratio is None else [0, 50, 100].index(st.session_state.ss_remote_ratio),
        key="w_remote_ratio",
    )

    _step2_ready = st.session_state.ss_job_family is not None and st.session_state.ss_remote_ratio is not None
    c_back, c_next = st.columns(2)
    c_back.button("← Back", on_click=_go_back)
    c_next.button("Next →", on_click=_go_next, type="primary", disabled=not _step2_ready)

# ---------------------------------------------------------------------------
# Step 3 — Company & Location
# ---------------------------------------------------------------------------

elif st.session_state.step == 3:
    st.session_state.ss_company_size = st.selectbox(
        "Company size",
        options=list(_SIZE_LABELS.keys()),
        format_func=lambda x: _SIZE_LABELS[x],
        index=None if st.session_state.ss_company_size is None else list(_SIZE_LABELS.keys()).index(st.session_state.ss_company_size),
        key="w_company_size",
    )
    st.session_state.ss_location_region = st.selectbox(
        "Location region",
        options=list(_REGION_LABELS.keys()),
        format_func=lambda x: _REGION_LABELS[x],
        index=None if st.session_state.ss_location_region is None else list(_REGION_LABELS.keys()).index(st.session_state.ss_location_region),
        key="w_location_region",
    )

    # Conditional: show is_us_company only for North America (3)
    if st.session_state.ss_location_region == 3:
        st.session_state.ss_is_us_company = st.radio(
            "US-based company?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x else "No",
            index=None if st.session_state.ss_is_us_company is None else [1, 0].index(st.session_state.ss_is_us_company),
            horizontal=True,
            key="w_is_us_company",
        )
    elif st.session_state.ss_location_region is not None:
        st.session_state.ss_is_us_company = 0

    _step3_ready = (
        st.session_state.ss_company_size is not None
        and st.session_state.ss_location_region is not None
        and (st.session_state.ss_location_region != 3 or st.session_state.ss_is_us_company is not None)
    )
    c_back, c_next = st.columns(2)
    c_back.button("← Back", on_click=_go_back)
    c_next.button("Next →", on_click=_go_next, type="primary", disabled=not _step3_ready)

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
    c_predict.button("Reveal Your True Salary", type="primary", on_click=_on_predict)

# ---------------------------------------------------------------------------
# Start-over button (visible on steps 2–4)
# ---------------------------------------------------------------------------

if st.session_state.step > 1:
    st.divider()
    if st.button("Start over"):
        _reset_stepper()
        st.rerun()
