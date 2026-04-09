import json
import logging
from collections.abc import AsyncGenerator

import httpx

from config.settings import settings
from src.llm.exceptions import LLMError

logger = logging.getLogger(__name__)


class OllamaError(LLMError):
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


async def generate_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Send a prompt to Ollama and yield response tokens as they arrive.

    Uses Ollama's streaming mode (``stream: True``). Each line from Ollama is a
    newline-delimited JSON object ``{"response": "token", "done": false}``.
    The generator yields the ``response`` field of each object and stops when
    ``done`` is ``True``.

    Args:
        prompt: The full prompt string to send to the model.

    Yields:
        One token string per chunk from Ollama.

    Raises:
        OllamaError: on HTTP errors, connection failures, or timeouts.
    """
    logger.debug(
        "ollama stream | model=%s prompt_length=%d",
        settings.ollama_model,
        len(prompt),
    )
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("ollama stream | skipping non-JSON line: %r", line)
                        continue
                    token: str = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama streaming request timed out after {settings.ollama_timeout}s."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Ollama returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {settings.ollama_base_url}."
        ) from exc
