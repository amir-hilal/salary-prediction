"""Shared exception hierarchy for LLM backends (Ollama, Groq, …)."""


class LLMError(Exception):
    """Base exception for all LLM provider errors."""
