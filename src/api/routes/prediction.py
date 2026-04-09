import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.schemas.salary import ErrorResponse, PredictionRequest, PredictionResponse, SalaryDetail
from src.models.predict import predict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])


async def _persist_prediction(
    prediction_id: str,
    features: dict,
    predicted_salary: float,
    salary_range_low: float,
    salary_range_high: float,
    model_version: str,
) -> None:
    """Write the prediction record to Supabase (runs as a background task)."""
    try:
        from src.database.crud import insert_prediction  # noqa: PLC0415

        await insert_prediction(
            prediction_id=prediction_id,
            features=features,
            predicted_salary=predicted_salary,
            salary_range_low=salary_range_low,
            salary_range_high=salary_range_high,
            model_version=model_version,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "persist_prediction | insert failed | prediction_id=%s | error=%s",
            prediction_id,
            exc,
        )


async def _generate_and_persist_narrative(
    prediction_id: str,
    prediction_context: dict,
) -> None:
    """Call the LLM and write the narrative to Supabase (runs as a background task).

    Scheduled after _persist_prediction so the FK constraint is satisfied.
    Failures are logged and swallowed — a missing narrative never blocks the response.
    """
    try:
        from src.database.crud import insert_narrative  # noqa: PLC0415
        from src.llm.narrative import generate_narrative  # noqa: PLC0415

        narrative, raw = await generate_narrative(prediction_context)
        await insert_narrative(
            prediction_id=prediction_id,
            narrative=narrative,
            raw_response=raw,
        )
        logger.info(
            "generate_narrative | persisted | prediction_id=%s",
            prediction_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "generate_narrative | failed | prediction_id=%s | error=%s",
            prediction_id,
            exc,
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error or feature mismatch"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
    summary="Predict salary for a candidate profile",
)
async def predict_salary(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Return the predicted salary in USD for a single candidate profile.

    Input integers map to encoded feature values — see schema descriptions for
    the meaning of each value.
    """
    prediction_id = str(uuid.uuid4())
    features = payload.model_dump()

    try:
        result = predict(features)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"detail": str(exc), "code": "FEATURE_MISMATCH"},
        ) from exc
    except Exception as exc:
        logger.error(
            "predict_salary | unexpected error | prediction_id=%s | error=%s",
            prediction_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "detail": "An unexpected error occurred. Please try again later.",
                "code": "INTERNAL_ERROR",
            },
        ) from exc

    model_version: str = getattr(request.app.state, "model_version", "unknown")
    model_mae: float = getattr(request.app.state, "model_mae", 0.0)

    # Persist the prediction row synchronously so it exists before the
    # dashboard opens the narrative SSE stream — no race condition.
    await _persist_prediction(
        prediction_id,
        features,
        result.point_estimate,
        result.range_low,
        result.range_high,
        model_version,
    )

    logger.info(
        "predict_salary | prediction_id=%s | predicted_salary=%.2f",
        prediction_id,
        result.point_estimate,
    )

    return PredictionResponse(
        salary=SalaryDetail(
            mean=result.point_estimate,
            low=result.range_low,
            high=result.range_high,
            currency="USD",
        ),
        model_version=model_version,
        prediction_id=prediction_id,
    )


@router.get(
    "/health",
    summary="Health check",
)
async def health(request: Request) -> dict:
    """Return API liveness status and the currently loaded model version."""
    model_version: str = getattr(request.app.state, "model_version", "unknown")
    return {"status": "ok", "model_version": model_version}


@router.get(
    "/predict/{prediction_id}/narrative",
    summary="Stream the LLM narrative for a prediction",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid prediction_id format"},
        404: {"model": ErrorResponse, "description": "Prediction not found"},
    },
)
async def stream_narrative(prediction_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events endpoint that streams the LLM narrative token by token.

    The client reads ``data: <token>\\n\\n`` lines until it receives
    ``data: [DONE]\\n\\n``.  On error a ``data: [ERROR] ...\\n\\n`` line is
    sent and the stream closes.

    After all tokens are yielded the narrative is parsed and persisted to
    Supabase automatically — no separate persist step is needed.
    """
    # Validate UUID format.
    try:
        uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"detail": "prediction_id must be a valid UUID.", "code": "INVALID_UUID"},
        )

    # Fetch prediction row to reconstruct context.
    # Use the async version to avoid blocking the event loop.
    from src.database.crud import get_prediction_context_async  # noqa: PLC0415

    try:
        context = await get_prediction_context_async(prediction_id)
    except Exception as exc:
        logger.error("stream_narrative | context fetch failed | %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"detail": "Failed to fetch prediction context.", "code": "INTERNAL_ERROR"},
        ) from exc

    if context is None:
        raise HTTPException(
            status_code=404,
            detail={"detail": "Prediction not found.", "code": "NOT_FOUND"},
        )

    # Inject model_mae from app state (not stored in the DB row).
    context["model_mae"] = getattr(request.app.state, "model_mae", 0.0)

    from src.llm.narrative import generate_narrative_stream  # noqa: PLC0415

    async def _event_generator():
        try:
            async for token in generate_narrative_stream(context):
                # Escape newlines inside a token so each SSE message is one line.
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("stream_narrative | generator error | %s", exc)
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
