"""LLM backend dispatcher — picks Ollama or Groq based on ``settings.llm_provider``.

Import ``generate``, ``generate_stream``, and ``LLMError`` from this module
instead of importing a specific backend directly.
"""

from config.settings import settings
from src.llm.exceptions import LLMError

if settings.llm_provider == "groq":
    from src.llm.groq_client import generate, generate_stream  # noqa: F401
else:
    from src.llm.ollama_client import generate, generate_stream  # noqa: F401

__all__ = ["LLMError", "generate", "generate_stream"]
