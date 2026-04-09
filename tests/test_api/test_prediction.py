import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.models.predict import PredictionResult

_MOCK_RESULT = PredictionResult(point_estimate=125_000.0, range_low=110_000.0, range_high=140_000.0)

# ── Shared test payload ───────────────────────────────────────────────────────
# Represents: Senior Data Scientist, full-time, fully remote, US medium company.

VALID_PAYLOAD: dict = {
    "experience_level": 2,
    "employment_type": 3,
    "remote_ratio": 100,
    "company_size": 1,
    "work_year": 2024,
    "job_family": 2,
    "location_region": 3,
    "is_us_company": 1,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
async def client(mocker):
    # Prevent the lifespan from loading the real .joblib artifact during tests.
    mocker.patch("src.models.predict._get_pipeline")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── POST /api/v1/predict — happy path ─────────────────────────────────────────


async def test_predict_returns_200(client, mocker) -> None:
    mocker.patch("src.api.routes.prediction.predict", return_value=_MOCK_RESULT)
    response = await client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


async def test_predict_response_contains_salary(client, mocker) -> None:
    mocker.patch("src.api.routes.prediction.predict", return_value=_MOCK_RESULT)
    body = (await client.post("/api/v1/predict", json=VALID_PAYLOAD)).json()
    assert body["salary"]["mean"] == 125_000.0


async def test_predict_response_contains_range(client, mocker) -> None:
    mocker.patch("src.api.routes.prediction.predict", return_value=_MOCK_RESULT)
    body = (await client.post("/api/v1/predict", json=VALID_PAYLOAD)).json()
    assert body["salary"]["low"] == 110_000.0
    assert body["salary"]["high"] == 140_000.0


async def test_predict_response_range_low_le_high(client, mocker) -> None:
    mocker.patch("src.api.routes.prediction.predict", return_value=_MOCK_RESULT)
    body = (await client.post("/api/v1/predict", json=VALID_PAYLOAD)).json()
    assert body["salary"]["low"] <= body["salary"]["high"]


async def test_predict_response_currency_is_usd(client, mocker) -> None:
    mocker.patch("src.api.routes.prediction.predict", return_value=_MOCK_RESULT)
    body = (await client.post("/api/v1/predict", json=VALID_PAYLOAD)).json()
    assert body["salary"]["currency"] == "USD"


async def test_predict_response_has_prediction_id(client, mocker) -> None:
    mocker.patch("src.api.routes.prediction.predict", return_value=_MOCK_RESULT)
    body = (await client.post("/api/v1/predict", json=VALID_PAYLOAD)).json()
    assert "prediction_id" in body
    assert len(body["prediction_id"]) == 36  # standard UUID string length


async def test_predict_response_has_model_version(client, mocker) -> None:
    mocker.patch("src.api.routes.prediction.predict", return_value=_MOCK_RESULT)
    body = (await client.post("/api/v1/predict", json=VALID_PAYLOAD)).json()
    assert "model_version" in body


# ── POST /api/v1/predict — schema validation (422) ───────────────────────────


async def test_predict_422_on_out_of_range_experience_level(client) -> None:
    payload = {**VALID_PAYLOAD, "experience_level": 99}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


async def test_predict_422_on_invalid_remote_ratio(client) -> None:
    # Only 0, 50, 100 are valid — 75 is not in the Literal
    payload = {**VALID_PAYLOAD, "remote_ratio": 75}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


async def test_predict_422_on_missing_field(client) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "job_family"}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


async def test_predict_422_on_invalid_is_us_company(client) -> None:
    # Only 0 and 1 are valid
    payload = {**VALID_PAYLOAD, "is_us_company": 2}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


# ── POST /api/v1/predict — internal error (500) ───────────────────────────────


async def test_predict_500_on_unexpected_exception(client, mocker) -> None:
    mocker.patch(
        "src.api.routes.prediction.predict",
        side_effect=RuntimeError("unexpected crash"),
    )
    response = await client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 500


async def test_predict_500_response_has_safe_message(client, mocker) -> None:
    mocker.patch(
        "src.api.routes.prediction.predict",
        side_effect=RuntimeError("db connection lost"),
    )
    body = (await client.post("/api/v1/predict", json=VALID_PAYLOAD)).json()
    # FastAPI wraps HTTPException.detail under body["detail"]
    assert body["detail"]["code"] == "INTERNAL_ERROR"
    assert "db connection lost" not in body["detail"]["detail"]


# ── GET /api/v1/health ────────────────────────────────────────────────────────


async def test_health_returns_200(client) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200


async def test_health_status_is_ok(client) -> None:
    body = (await client.get("/api/v1/health")).json()
    assert body["status"] == "ok"


async def test_health_includes_model_version(client) -> None:
    body = (await client.get("/api/v1/health")).json()
    assert "model_version" in body


# ── GET /api/v1/predict/{id}/narrative — SSE streaming ───────────────────────

_VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"

_MOCK_CONTEXT = {
    "prediction_id": _VALID_UUID,
    "point_estimate": 125_000.0,
    "range_low": 110_000.0,
    "range_high": 140_000.0,
    "currency": "USD",
    "model_mae": 20_000.0,
    "features": {"experience_level": 2, "is_us_company": 1},
}


async def _mock_stream(*tokens: str):
    """Return an async generator that yields the given tokens."""
    for token in tokens:
        yield token


async def test_stream_narrative_400_on_invalid_uuid(client) -> None:
    response = await client.get("/api/v1/predict/not-a-uuid/narrative")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_UUID"


async def test_stream_narrative_404_on_missing_prediction(client, mocker) -> None:
    mocker.patch(
        "src.database.crud.get_prediction_context_async",
        return_value=None,
    )
    response = await client.get(f"/api/v1/predict/{_VALID_UUID}/narrative")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "NOT_FOUND"


async def test_stream_narrative_returns_200_and_event_stream_content_type(
    client, mocker
) -> None:
    mocker.patch(
        "src.database.crud.get_prediction_context_async",
        return_value=_MOCK_CONTEXT,
    )
    mocker.patch(
        "src.llm.narrative.generate_narrative_stream",
        return_value=_mock_stream("Hello"),
    )
    response = await client.get(f"/api/v1/predict/{_VALID_UUID}/narrative")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_stream_narrative_tokens_wrapped_in_sse_format(
    client, mocker
) -> None:
    mocker.patch(
        "src.database.crud.get_prediction_context_async",
        return_value=_MOCK_CONTEXT,
    )
    mocker.patch(
        "src.llm.narrative.generate_narrative_stream",
        return_value=_mock_stream("Token1", "Token2"),
    )
    response = await client.get(f"/api/v1/predict/{_VALID_UUID}/narrative")
    text = response.text
    assert "data: Token1\n\n" in text
    assert "data: Token2\n\n" in text


async def test_stream_narrative_ends_with_done_sentinel(client, mocker) -> None:
    mocker.patch(
        "src.database.crud.get_prediction_context_async",
        return_value=_MOCK_CONTEXT,
    )
    mocker.patch(
        "src.llm.narrative.generate_narrative_stream",
        return_value=_mock_stream("Hello"),
    )
    response = await client.get(f"/api/v1/predict/{_VALID_UUID}/narrative")
    # Strip trailing whitespace; last non-empty SSE event must be [DONE].
    events = [e for e in response.text.split("\n\n") if e.strip()]
    assert events[-1] == "data: [DONE]"
