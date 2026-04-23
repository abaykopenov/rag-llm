"""
GeminiProvider — embeddings через нативный google-genai SDK.

Модели: text-embedding-004 (768 dim), gemini-embedding-001 (варьируется).
Бесплатный тариф: ~1500 RPM, без дневного лимита на embeddings в бете.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from app.core.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderInfo,
    TokenizerSpec,
)
from app.core.embedding.profiles import EmbeddingProfile
from app.utils.logging import get_logger

log = get_logger("embedding.gemini")

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


class GeminiProvider(EmbeddingProvider):
    """Native Gemini embeddings."""

    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile
        self.info = EmbeddingProviderInfo(
            provider="gemini",
            model=profile.model,
            dim=profile.dim,
        )
        self.max_input_tokens = profile.max_input_tokens
        self.tokenizer_spec = profile.tokenizer or TokenizerSpec()
        self._batch_size = profile.batch_size or 50
        self._client = None

    def _get_client(self):
        """Lazy init клиента google-genai."""
        if self._client is None:
            from google import genai

            api_key = self.profile.resolved_api_key()
            if not api_key:
                raise ValueError(
                    f"Gemini embedding provider '{self.profile.name}' требует API ключ. "
                    f"Задай env-переменную {self.profile.api_key_env or 'RAG_LLM_API_KEY'} "
                    f"(https://aistudio.google.com/apikey)"
                )
            self._client = genai.Client(api_key=api_key)
            log.info(
                "Gemini embedding client initialised (model={}, dim={})",
                self.info.model, self.info.dim,
            )
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        loop = asyncio.get_event_loop()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda b=batch: client.models.embed_content(
                            model=self.info.model,
                            contents=b,
                        ),
                    )
                    for emb in result.embeddings:
                        all_embeddings.append(list(emb.values))
                    break

                except Exception as e:
                    error_str = str(e)
                    if (
                        ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str)
                        and attempt < MAX_RETRIES
                    ):
                        wait = 30 * attempt
                        log.warning(
                            "Gemini embedding rate limit (попытка {}/{}). Ждём {} сек...",
                            attempt, MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                    elif attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning(
                            "Gemini embedding ошибка (попытка {}/{}): {}",
                            attempt, MAX_RETRIES, e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise

        return all_embeddings

    async def close(self) -> None:
        # У google-genai нет явного close; HTTP-сессии живут в SDK.
        self._client = None
