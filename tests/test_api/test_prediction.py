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
