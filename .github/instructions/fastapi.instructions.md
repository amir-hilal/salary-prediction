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
- `POST /predict` is the primary inference route
  - Validate input with schema; return `422` automatically via Pydantic
  - Call `src/models/predict.py::predict()` — never instantiate the model inside the route
  - After prediction, persist the record via `BackgroundTask` (task 1 only; narrative generation is no longer a background task here)
  - Return `PredictionResponse` with a freshly generated UUID for `prediction_id`
- `GET /predict/{prediction_id}/narrative` is the streaming narrative route
  - Validates `prediction_id` is a well-formed UUID; returns `400` if not
  - Fetches the prediction row from Supabase via `get_prediction_context(prediction_id)` to reconstruct context; returns `404` if not found
  - Returns `StreamingResponse(media_type="text/event-stream")` — each token wrapped as `f"data: {token}\n\n"` (SSE format)
  - Sends `data: [DONE]\n\n` as the final event to signal stream end
  - Persistence (parse + insert to Supabase) happens inside the generator after all tokens are yielded — callers do not need a separate persist step
  - On error, sends `data: [ERROR] ...\n\n` and closes; never raises an unhandled exception
- Add a `GET /health` route returning `{"status": "ok", "model_version": "..."}` for uptime checks

## Error Handling
- Use `HTTPException` with structured `detail` dicts, not plain strings
- Add a global exception handler for unexpected errors: log at ERROR level, return `500` with safe message
- Never expose stack traces or internal paths to clients

## Testing (`tests/test_api/`)
- Use `httpx.AsyncClient` with `ASGITransport` — no live network calls
- Mock `src/models/predict.py::predict` and `src/database/crud.py::insert_prediction`
- Test the happy path, schema validation errors (`422`), and internal errors (`500`)
- For the streaming narrative endpoint, mock `generate_narrative_stream` and collect the full SSE response body; assert `data: [DONE]` appears and tokens are present
- Use `pytest-asyncio` with `asyncio_mode = "auto"`

## Security
- Validate all inputs at the schema level — Pydantic handles it; add custom validators only when needed
- Do not log raw request bodies in production — log `prediction_id` only
- API keys or auth headers if authentication is added must come from headers, validated in middleware
