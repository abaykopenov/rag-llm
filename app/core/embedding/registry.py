"""
Реестр embedding-провайдеров: имя в YAML → класс.
"""
from __future__ import annotations

from app.core.embedding.base import EmbeddingProvider
from app.core.embedding.profiles import EmbeddingProfile
from app.core.embedding.providers.gemini import GeminiProvider
from app.core.embedding.providers.openai_compat import OpenAICompatProvider

PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "gemini": GeminiProvider,
    "openai_compat": OpenAICompatProvider,
}


def create_provider(profile: EmbeddingProfile) -> EmbeddingProvider:
    """Создать провайдера по профилю."""
    cls = PROVIDER_REGISTRY.get(profile.provider)
    if cls is None:
        raise ValueError(
            f"Неизвестный provider {profile.provider!r} в профиле {profile.name!r}. "
            f"Доступные: {list(PROVIDER_REGISTRY)}"
        )
    return cls(profile)
