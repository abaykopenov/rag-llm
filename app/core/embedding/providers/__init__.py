"""Конкретные реализации EmbeddingProvider."""
from app.core.embedding.providers.gemini import GeminiProvider
from app.core.embedding.providers.openai_compat import OpenAICompatProvider

__all__ = ["GeminiProvider", "OpenAICompatProvider"]
