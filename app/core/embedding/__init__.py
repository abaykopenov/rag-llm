"""
Embedding-провайдеры: Gemini, OpenAI-compat (OpenAI/Jina/vLLM/Ollama).

Выбор активного провайдера задаётся:
  - через env RAG_EMBEDDING_PROFILE (имя профиля в config/embedding_profiles.yml), либо
  - через плоские поля settings.embedding_* (backward-compat для старых .env).

См. app.core.embedder.Embedder — фасад, который использует провайдера.
"""
from app.core.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderInfo,
    TokenizerSpec,
)
from app.core.embedding.profiles import (
    EmbeddingProfile,
    load_profiles,
    resolve_active_profile,
)
from app.core.embedding.registry import PROVIDER_REGISTRY, create_provider

__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderInfo",
    "EmbeddingProfile",
    "TokenizerSpec",
    "PROVIDER_REGISTRY",
    "create_provider",
    "load_profiles",
    "resolve_active_profile",
]
