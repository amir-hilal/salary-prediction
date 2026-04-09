import json
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from src.api.routes import prediction

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Load logging config from config/logging.yaml if present and non-empty."""
    config_path = Path("config/logging.yaml")
    text = config_path.read_text() if config_path.exists() else ""
    if text.strip():
        import yaml  # PyYAML — required when logging.yaml is populated

        logging.config.dictConfig(yaml.safe_load(text))
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(name)s | %(message)s",
        )


def _read_model_version() -> str:
    """Return the timestamp of the currently registered model artifact."""
    registry_path = settings.models_registry_path / "latest.json"
    if registry_path.exists():
        entry = json.loads(registry_path.read_text())
        return str(entry.get("timestamp", "unknown"))
    return "unknown"


def _read_model_mae() -> float:
    """Return the MAE of the currently registered model artifact."""
    registry_path = settings.models_registry_path / "latest.json"
    if registry_path.exists():
        entry = json.loads(registry_path.read_text())
        return float(entry.get("metrics", {}).get("mae", 0.0))
    return 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _configure_logging()

    model_version = _read_model_version()
    app.state.model_version = model_version
    app.state.model_mae = _read_model_mae()

    # Eagerly warm the model singleton so the first request has no cold-start I/O.
    from src.models.predict import _get_pipeline  # noqa: PLC0415

    _get_pipeline()
    logger.info(
        "lifespan | startup complete | model_version=%s | model_mae=%.2f",
        model_version,
        app.state.model_mae,
    )

    yield

    logger.info("lifespan | shutdown")


app = FastAPI(
    title="Salary Prediction API",
    version="1.0.0",
    description=(
        "Predicts data-profession salaries using a Decision Tree model "
        "trained on the DS Salaries dataset."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to dashboard origin in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(prediction.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled exception | path=%s | error=%s",
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "code": "INTERNAL_ERROR",
        },
    )
