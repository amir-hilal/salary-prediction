import logging

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when the Ollama API returns an error or times out."""


async def generate(prompt: str) -> str:
    """Send a prompt to Ollama and return the generated text.

    Args:
        prompt: The full prompt string to send to the model.

    Returns:
        The model's response text.

    Raises:
        OllamaError: on HTTP errors, connection failures, or timeouts.
    """
    logger.debug(
        "ollama | model=%s prompt_length=%d",
        settings.ollama_model,
        len(prompt),
    )
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama request timed out after {settings.ollama_timeout}s."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Ollama returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {settings.ollama_base_url}."
        ) from exc

    return str(response.json().get("response", ""))
