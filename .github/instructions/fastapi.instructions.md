---
description: "Use when writing FastAPI routes, Pydantic schemas, middleware, or API tests. Covers the POST /predict endpoint, schema design, error handling, and async test patterns in src/api/ and tests/test_api/."
applyTo: "src/api/**,tests/test_api/**"
---

# FastAPI Instructions

## App Setup (`src/api/main.py`)
- Create the `FastAPI` app with `title`, `version`, and `description` set
- Include all routers via `app.include_router()`
- Add CORS middleware only if the dashboard runs on a different origin
- Set up lifespan context (`@asynccontextmanager`) for startup/shutdown — load the model here, not at import time
- Configure structured logging from `config/logging.yaml` at app startup

## Schemas (`src/api/schemas/salary.py`)
- Use Pydantic v2 (`model_config = ConfigDict(...)`)
- `PredictionRequest`: all candidate features with field-level validation and examples
- `PredictionResponse`: `predicted_salary: float`, `currency: str`, `model_version: str`, `prediction_id: str` (UUID)
- `ErrorResponse`: `detail: str`, `code: str`
- Never expose internal model details in the response schema

## Routes (`src/api/routes/prediction.py`)
- `POST /predict` is the sole inference route
- Validate input with schema; return `422` automatically via Pydantic
- Call `src/models/predict.py::predict()` — never instantiate the model inside the route
- After prediction, call `src/database/crud.py` to persist the record (fire-and-forget or `BackgroundTask`)
- Return `PredictionResponse` with a freshly generated UUID for `prediction_id`
- Add a `GET /health` route returning `{"status": "ok", "model_version": "..."}` for uptime checks

## Error Handling
- Use `HTTPException` with structured `detail` dicts, not plain strings
- Add a global exception handler for unexpected errors: log at ERROR level, return `500` with safe message
- Never expose stack traces or internal paths to clients

## Testing (`tests/test_api/`)
- Use `httpx.AsyncClient` with `ASGITransport` — no live network calls
- Mock `src/models/predict.py::predict` and `src/database/crud.py::insert_prediction`
- Test the happy path, schema validation errors (`422`), and internal errors (`500`)
- Use `pytest-asyncio` with `asyncio_mode = "auto"`

## Security
- Validate all inputs at the schema level — Pydantic handles it; add custom validators only when needed
- Do not log raw request bodies in production — log `prediction_id` only
- API keys or auth headers if authentication is added must come from headers, validated in middleware
