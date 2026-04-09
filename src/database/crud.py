import logging
import uuid as _uuid
from datetime import datetime

from pydantic import BaseModel

from src.database.client import get_anon_client, get_client
from src.llm.narrative import NarrativeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row models — returned by every write and read function
# ---------------------------------------------------------------------------

class PredictionRecord(BaseModel):
    id: str
    created_at: datetime
    features: dict
    predicted_salary: float
    salary_range_low: float | None
    salary_range_high: float | None
    model_version: str
    currency: str


class NarrativeRecord(BaseModel):
    id: str
    prediction_id: str
    created_at: datetime
    summary: str
    uncertainty: str
    insights: list[str]
    recommendation: str
    chart_spec: dict
    raw_response: str | None


# ---------------------------------------------------------------------------
# Writes (async — called from FastAPI route handlers)
# ---------------------------------------------------------------------------

async def insert_prediction(
    *,
    prediction_id: str,
    features: dict,
    predicted_salary: float,
    salary_range_low: float | None = None,
    salary_range_high: float | None = None,
    model_version: str,
    currency: str = "USD",
) -> PredictionRecord:
    """Insert a prediction row and return the inserted record.

    The caller supplies the UUID so it can be generated before the insert and
    used to link the narrative without an extra round-trip.

    Raises:
        ValueError: if prediction_id is not a valid UUID string.
    """
    try:
        _uuid.UUID(prediction_id)
    except ValueError as exc:
        raise ValueError(f"prediction_id must be a valid UUID; got {prediction_id!r}") from exc

    payload = {
        "id": prediction_id,
        "features": features,
        "predicted_salary": predicted_salary,
        "salary_range_low": salary_range_low,
        "salary_range_high": salary_range_high,
        "model_version": model_version,
        "currency": currency,
    }
    client = await get_client()
    response = await client.table("predictions").insert(payload).execute()
    row = response.data[0]
    logger.info("insert_prediction | id=%s | salary=%.2f", prediction_id, predicted_salary)
    return PredictionRecord(**row)


async def insert_narrative(
    *,
    prediction_id: str,
    narrative: NarrativeResult,
    raw_response: str | None = None,
) -> NarrativeRecord:
    """Insert or replace a narrative row linked to a prediction.

    Uses upsert on the unique prediction_id constraint so retries on timeout
    are idempotent rather than creating duplicate rows.
    """
    payload = {
        "prediction_id": prediction_id,
        "summary": narrative.summary,
        "uncertainty": narrative.uncertainty,
        "insights": narrative.insights,
        "recommendation": narrative.recommendation,
        "chart_spec": narrative.chart_spec.model_dump(),
        "raw_response": raw_response,
    }
    client = await get_client()
    response = (
        await client.table("narratives")
        .upsert(payload, on_conflict="prediction_id")
        .execute()
    )
    row = response.data[0]
    logger.info("insert_narrative | prediction_id=%s", prediction_id)
    return NarrativeRecord(**row)


# ---------------------------------------------------------------------------
# Reads (sync — called from Streamlit dashboard via anon key)
# ---------------------------------------------------------------------------

def get_recent_predictions(limit: int = 100) -> list[PredictionRecord]:
    """Return the most recent predictions ordered by created_at descending."""
    response = (
        get_anon_client()
        .table("predictions")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [PredictionRecord(**row) for row in response.data]


def get_narrative_for_prediction(prediction_id: str) -> NarrativeRecord | None:
    """Return the narrative for a given prediction, or None if not yet generated."""
    response = (
        get_anon_client()
        .table("narratives")
        .select("*")
        .eq("prediction_id", prediction_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    return NarrativeRecord(**response.data[0])

