import logging
import re
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from src.llm.ollama_client import OllamaError, generate, generate_stream

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ChartSpec(BaseModel):
    type: str
    title: str
    x_label: str
    y_label: str
    data_key: str


class NarrativeResult(BaseModel):
    summary: str
    uncertainty: str
    insights: list[str]
    recommendation: str
    chart_spec: ChartSpec


# ---------------------------------------------------------------------------
# Default fallback chart when the LLM omits or malforms the [CHART] block
# ---------------------------------------------------------------------------

_DEFAULT_CHART = ChartSpec(
    type="histogram",
    title="Salary Distribution",
    x_label="Salary (USD)",
    y_label="Count",
    data_key="salary_distribution",
)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# prediction_context keys the caller must supply:
#   point_estimate : float   — the model's point prediction in USD
#   range_low      : float   — Q25 of training salaries in the landing leaf
#   range_high     : float   — Q75 of training salaries in the landing leaf
#   currency       : str     — always "USD"
#   model_mae      : float   — MAE from models/registry/latest.json
#   features       : dict    — the raw encoded feature dict from the request

_SYSTEM_PROMPT = """\
You are a data analyst specialising in technology compensation.
Respond in plain English. Be precise — quote the numbers you are given; \
do not invent statistics.
Always include all six sections in this exact order:
1. SUMMARY
2. UNCERTAINTY
3. INSIGHTS
4. COMPARISON
5. CHART
6. RECOMMENDATION
""".strip()

_USER_TEMPLATE = """\
A candidate has been profiled with the following encoded features:
{features_block}

The salary model returned:
  - Point estimate : {currency} {point_estimate:,.0f}
  - Peer-group range (Q25–Q75) : {currency} {range_low:,.0f} – {currency} {range_high:,.0f}
  - Model typical error (MAE) : ± {currency} {model_mae:,.0f}

Write the six-section analysis. For section 2 (UNCERTAINTY) you MUST state:
  • the point estimate
  • the peer-group salary range
  • the model's typical error of ± {currency} {model_mae:,.0f}
Do not soften or omit these numbers.

For section 5 (CHART), use exactly this format:
[CHART]
type: bar | histogram | scatter | line
title: <chart title>
x_label: <axis label>
y_label: <axis label>
data_key: <supabase data key>
[/CHART]
""".strip()

# Few-shot example appended to the user message so the LLM learns the format.
_FEW_SHOT_EXAMPLE = """
---
EXAMPLE OUTPUT (do not copy numbers — use the real values above):

1. SUMMARY
Senior data scientists at US companies command a predicted salary of $145,000,
placing them in the upper tier of the global data science market. Full-time
employment and full remote work are positively associated with higher pay in
this profile.

2. UNCERTAINTY
The model predicts a salary of $145,000. Most peers with this exact profile
earned between $128,000 and $162,000 (peer-group Q25–Q75). The model carries
a typical absolute error of ± $31,500 (MAE), so the true salary could
reasonably fall anywhere from roughly $113,500 to $176,500.

3. INSIGHTS
• Experience level (Senior) is the strongest driver — senior roles earn ~40%
  more than mid-level peers with the same profile.
• US company flag adds approximately $25,000 compared to non-US equivalents.
• Full remote work correlates with higher pay in this dataset, likely because
  remote roles attract global talent competition.

4. COMPARISON
This estimate is above the dataset median of $115,000. North America accounts
for the top-paying region, and this profile lands squarely there.

5. CHART
[CHART]
type: bar
title: Salary by Experience Level
x_label: Experience Level
y_label: Average Salary (USD)
data_key: salary_by_experience
[/CHART]

6. RECOMMENDATION
To maximise compensation, the candidate should target senior IC or staff
engineer tracks at US-headquartered companies with established remote policies.
Specialising further in ML/AI (job family 4) would shift the predicted salary
upward by an estimated 10–15%.
---
""".strip()


def _features_block(features: dict) -> str:
    """Format the encoded feature dict as a readable bullet list."""
    labels = {
        "experience_level": {0: "Entry-level (EN)", 1: "Mid-level (MI)", 2: "Senior (SE)", 3: "Executive (EX)"},
        "employment_type": {0: "Freelance (FL)", 1: "Part-time (PT)", 2: "Contract (CT)", 3: "Full-time (FT)"},
        "remote_ratio": {0: "On-site (0%)", 50: "Hybrid (50%)", 100: "Fully remote (100%)"},
        "company_size": {0: "Small", 1: "Medium", 2: "Large"},
        "job_family": {0: "Other", 1: "Analytics", 2: "Data Science", 3: "Data Engineering", 4: "ML/AI", 5: "Leadership"},
        "location_region": {0: "Rest of World", 1: "Asia-Pacific", 2: "Western Europe", 3: "North America"},
        "is_us_company": {0: "Non-US company", 1: "US company"},
    }
    lines = []
    for key, value in features.items():
        mapping = labels.get(key)
        display = mapping[value] if mapping and value in mapping else str(value)  # type: ignore[index]
        lines.append(f"  • {key}: {display}")
    return "\n".join(lines)


def build_prompt(prediction_context: dict) -> str:
    """Assemble the full Ollama prompt from prediction context.

    Required keys in prediction_context:
        point_estimate, range_low, range_high, currency, model_mae, features
    """
    user_msg = _USER_TEMPLATE.format(
        features_block=_features_block(prediction_context["features"]),
        currency=prediction_context["currency"],
        point_estimate=prediction_context["point_estimate"],
        range_low=prediction_context["range_low"],
        range_high=prediction_context["range_high"],
        model_mae=prediction_context["model_mae"],
    )
    return f"{_SYSTEM_PROMPT}\n\n{user_msg}\n\n{_FEW_SHOT_EXAMPLE}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_CHART_BLOCK_RE = re.compile(
    r"\[CHART\](.*?)\[/CHART\]",
    re.DOTALL | re.IGNORECASE,
)
_CHART_FIELD_RE = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)
_SECTION_RE = re.compile(
    r"^\d+\.\s*(SUMMARY|UNCERTAINTY|INSIGHTS|COMPARISON|RECOMMENDATION)\s*\n(.*?)(?=\n\d+\.|$)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def _parse_chart_spec(raw: str) -> ChartSpec:
    match = _CHART_BLOCK_RE.search(raw)
    if not match:
        logger.warning("parse_narrative | [CHART] block missing — using default")
        return _DEFAULT_CHART
    fields = dict(_CHART_FIELD_RE.findall(match.group(1)))
    try:
        return ChartSpec(
            type=fields.get("type", _DEFAULT_CHART.type).split("|")[0].strip(),
            title=fields.get("title", _DEFAULT_CHART.title).strip(),
            x_label=fields.get("x_label", _DEFAULT_CHART.x_label).strip(),
            y_label=fields.get("y_label", _DEFAULT_CHART.y_label).strip(),
            data_key=fields.get("data_key", _DEFAULT_CHART.data_key).strip(),
        )
    except Exception:
        logger.warning("parse_narrative | malformed [CHART] block — using default")
        return _DEFAULT_CHART


def parse_narrative(raw: str) -> NarrativeResult:
    """Extract structured fields from the raw LLM response."""
    sections: dict[str, str] = {}
    for m in _SECTION_RE.finditer(raw):
        sections[m.group(1).upper()] = m.group(2).strip()

    insights_raw = sections.get("INSIGHTS", "")
    insights = [
        line.lstrip("•–- ").strip()
        for line in insights_raw.splitlines()
        if line.strip() and not line.strip().startswith(("#", "**"))
    ]

    return NarrativeResult(
        summary=sections.get("SUMMARY", raw[:300]),
        uncertainty=sections.get("UNCERTAINTY", ""),
        insights=insights or [insights_raw],
        recommendation=sections.get("RECOMMENDATION", ""),
        chart_spec=_parse_chart_spec(raw),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_narrative(prediction_context: dict) -> tuple[NarrativeResult, str]:
    """Build the prompt, call Ollama, parse and return the structured narrative.

    prediction_context must contain:
        point_estimate : float
        range_low      : float
        range_high     : float
        currency       : str  (e.g. "USD")
        model_mae      : float
        features       : dict  (encoded feature dict from the prediction request)

    Returns:
        A tuple of (NarrativeResult, raw_llm_response_str).

    Raises:
        OllamaError: if the Ollama call fails.
    """
    prompt = build_prompt(prediction_context)
    raw = await generate(prompt)
    logger.info(
        "generate_narrative | response_length=%d",
        len(raw),
    )
    return parse_narrative(raw), raw


async def generate_narrative_stream(
    prediction_context: dict,
) -> AsyncGenerator[str, None]:
    """Build the prompt, stream tokens from Ollama, and persist the result.

    Yields one token string at a time as they arrive from Ollama.  After the
    stream is exhausted the full text is parsed and persisted to Supabase.
    On ``OllamaError`` a single ``[ERROR]`` sentinel token is yielded instead
    of raising — callers must check for it.

    prediction_context must contain the same keys as ``generate_narrative``.

    Yields:
        Token strings from Ollama, then (internally) persists to Supabase.
    """
    # Import here to avoid circular dependency at module load time and to
    # match the pattern used elsewhere in the route layer.
    from src.database.crud import insert_narrative  # noqa: PLC0415

    # Validate required keys before touching Ollama — yield a sentinel and
    # return early so the SSE client sees a clean [ERROR] rather than a
    # mid-stream KeyError that closes the connection silently.
    required_keys = [
        "prediction_id", "point_estimate", "range_low",
        "range_high", "currency", "model_mae", "features",
    ]
    missing = [k for k in required_keys if k not in prediction_context]
    if missing:
        logger.error(
            "generate_narrative_stream | missing context keys: %s", missing
        )
        yield f"[ERROR] Internal error: prediction context incomplete."
        return

    prompt = build_prompt(prediction_context)
    full_text: list[str] = []

    try:
        async for token in generate_stream(prompt):
            full_text.append(token)
            yield token
    except OllamaError as exc:
        logger.warning("generate_narrative_stream | OllamaError: %s", exc)
        yield f"[ERROR] {exc}"
        return

    raw = "".join(full_text)
    logger.info("generate_narrative_stream | stream complete | length=%d", len(raw))

    # NOTE: Persistence happens after all tokens are yielded.  If it fails,
    # the client already received a complete narrative and [DONE] — there is
    # no way to signal the failure downstream.  Monitor ERROR logs for
    # "persist failed" entries to detect data-loss events.
    try:
        narrative = parse_narrative(raw)
        prediction_id: str = prediction_context["prediction_id"]
        await insert_narrative(
            prediction_id=prediction_id,
            narrative=narrative,
            raw_response=raw,
        )
        logger.info(
            "generate_narrative_stream | persisted | prediction_id=%s", prediction_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "generate_narrative_stream | persist failed (DATA LOSS RISK) | prediction_id=%s | error=%s",
            prediction_context.get("prediction_id", "UNKNOWN"),
            exc,
        )
