"""
Embedder — тонкий фасад поверх активного EmbeddingProvider.

Активный провайдер определяется через профиль (YAML) или плоские env-поля
(legacy). См. app.core.embedding для реализаций провайдеров.

Публичный API (сохранён для обратной совместимости):
    await embedder.embed_texts(texts) -> list[list[float]]
    await embedder.embed_query(text)  -> list[float]
    await embedder.close()
    embedder.info                      -> EmbeddingProviderInfo
"""
from __future__ import annotations

import time
from typing import Optional

from app.config import settings
from app.core.embedding import (
    EmbeddingProvider,
    EmbeddingProviderInfo,
    create_provider,
    resolve_active_profile,
)
from app.utils.logging import get_logger

log = get_logger("embedder")


class Embedder:
    """Фасад над активным провайдером. Ленивая инициализация при первом вызове."""

    def __init__(self):
        self._provider: Optional[EmbeddingProvider] = None

    def _get_provider(self) -> EmbeddingProvider:
        if self._provider is None:
            profile = resolve_active_profile(settings)
            self._provider = create_provider(profile)
            log.info(
                "Embedding provider активирован: profile='{}', {}",
                profile.name, self._provider.info,
            )
        return self._provider

    @property
    def info(self) -> EmbeddingProviderInfo:
        """Паспорт активного провайдера (provider/model/dim)."""
        return self._get_provider().info

    @property
    def model(self) -> str:
        """Имя активной embedding-модели (для логов и monitoring endpoint)."""
        return self._get_provider().info.model

    @property
    def provider(self) -> str:
        """Имя активного провайдера."""
        return self._get_provider().info.provider

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        provider = self._get_provider()
        log.info(
            "Embedding запрос: {} текстов, {}",
            len(texts), provider.info,
        )
        start = time.perf_counter()
        all_embeddings = await provider.embed_texts(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        log.info(
            "Embedding готов: {} векторов, dim={}, {:.0f} мс",
            len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
            elapsed_ms,
        )
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        return await self._get_provider().embed_query(text)

    def get_tokenizer(self):
        """Токенизатор для HybridChunker (или None если не сконфигурирован)."""
        return self._get_provider().get_tokenizer()

    async def close(self) -> None:
        if self._provider is not None:
            await self._provider.close()
            self._provider = None


# Глобальный экземпляр — импортируется из всего приложения
embedder = Embedder()
