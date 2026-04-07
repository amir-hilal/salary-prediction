import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.api.schemas.salary import ErrorResponse, PredictionRequest, PredictionResponse
from src.models.predict import predict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])


def _persist_prediction(prediction_id: str, features: dict, predicted_salary: float) -> None:
    """Write the prediction record to Supabase (runs as a background task)."""
    try:
        from src.database.crud import insert_prediction  # noqa: PLC0415

        insert_prediction(
            prediction_id=prediction_id,
            features=features,
            predicted_salary=predicted_salary,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "persist_prediction | insert failed | prediction_id=%s | error=%s",
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
    background_tasks: BackgroundTasks,
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

    background_tasks.add_task(_persist_prediction, prediction_id, features, result.point_estimate)

    logger.info(
        "predict_salary | prediction_id=%s | predicted_salary=%.2f",
        prediction_id,
        result.point_estimate,
    )

    return PredictionResponse(
        predicted_salary=result.point_estimate,
        salary_range_low=result.range_low,
        salary_range_high=result.range_high,
        currency="USD",
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
