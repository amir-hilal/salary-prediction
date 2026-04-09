"""Groq cloud LLM client — OpenAI-compatible chat completions via httpx.

Used in production instead of the local Ollama backend.  The two public
entry-points (``generate`` and ``generate_stream``) mirror the Ollama client
interface so callers can be swapped transparently.
"""

import json
import logging
from collections.abc import AsyncGenerator

import httpx

from config.settings import settings
from src.llm.exceptions import LLMError

logger = logging.getLogger(__name__)


class GroqError(LLMError):
    """Raised when the Groq API returns an error or times out."""


async def generate(prompt: str) -> str:
    """Send a prompt to Groq and return the generated text (non-streaming).

    The full prompt is sent as a single user message.

    Args:
        prompt: The full prompt string to send to the model.

    Returns:
        The model's response text.

    Raises:
        GroqError: on HTTP errors, connection failures, or timeouts.
    """
    logger.debug(
        "groq | model=%s prompt_length=%d",
        settings.groq_model,
        len(prompt),
    )
    payload = {
        "model": settings.groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.groq_timeout) as client:
            response = await client.post(
                f"{settings.groq_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise GroqError(
            f"Groq request timed out after {settings.groq_timeout}s."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise GroqError(
            f"Groq returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise GroqError(
            f"Could not connect to Groq at {settings.groq_base_url}."
        ) from exc

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", ""))


async def generate_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Send a prompt to Groq and yield response tokens as they arrive.

    Uses OpenAI-compatible streaming (``stream: true``).  Each SSE line is a
    JSON object with ``choices[0].delta.content``.  The generator stops when
    ``[DONE]`` is received.

    Args:
        prompt: The full prompt string to send to the model.

    Yields:
        One token string per chunk from Groq.

    Raises:
        GroqError: on HTTP errors, connection failures, or timeouts.
    """
    logger.debug(
        "groq stream | model=%s prompt_length=%d",
        settings.groq_model,
        len(prompt),
    )
    payload = {
        "model": settings.groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.groq_timeout) as client:
            async with client.stream(
                "POST",
                f"{settings.groq_base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("groq stream | skipping non-JSON line: %r", line)
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token: str = delta.get("content", "")
                    if token:
                        yield token
    except httpx.TimeoutException as exc:
        raise GroqError(
            f"Groq streaming request timed out after {settings.groq_timeout}s."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise GroqError(
            f"Groq returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise GroqError(
            f"Could not connect to Groq at {settings.groq_base_url}."
        ) from exc
