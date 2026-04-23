"""
OpenAICompatProvider — embeddings через OpenAI-совместимый API.

Работает с:
  - OpenAI (https://api.openai.com/v1)
  - Jina (https://api.jina.ai/v1)
  - vLLM (http://host:port/v1)
  - Ollama (http://host:11434/v1)
  - LM Studio, TGI, любой другой /embeddings endpoint в формате OpenAI.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from app.core.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderInfo,
    TokenizerSpec,
)
from app.core.embedding.profiles import EmbeddingProfile
from app.utils.logging import get_logger

log = get_logger("embedding.openai_compat")

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


class OpenAICompatProvider(EmbeddingProvider):
    """Embeddings через POST /embeddings (OpenAI-format)."""

    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile
        self.info = EmbeddingProviderInfo(
            provider="openai_compat",
            model=profile.model,
            dim=profile.dim,
        )
        self.max_input_tokens = profile.max_input_tokens
        self.tokenizer_spec = profile.tokenizer or TokenizerSpec()
        self._batch_size = profile.batch_size or 32
        self._base_url = (profile.base_url or "").rstrip("/")
        self._api_key = profile.resolved_api_key()
        self._client: Optional[httpx.AsyncClient] = None

        if not self._base_url:
            raise ValueError(
                f"OpenAICompat provider '{profile.name}' требует base_url в профиле"
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            # Ollama / vLLM могут работать без ключа, но некоторые требуют любой
            # не-пустой Bearer — отдаём, если задан.
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(120.0),
            )
            log.info(
                "OpenAI-compat embedding client initialised (model={}, base_url={})",
                self.info.model, self._base_url,
            )
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = await self._get_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            payload = {"model": self.info.model, "input": batch}

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await client.post("/embeddings", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    batch_embeddings = [item["embedding"] for item in sorted_data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except httpx.ConnectError as e:
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning(
                            "Embedding API недоступен (попытка {}/{}), повтор через {:.1f}с",
                            attempt, MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise ConnectionError(
                            f"Не удалось подключиться к Embedding API: {self._base_url}"
                        ) from e
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500 and attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning(
                            "Embedding API ошибка {}, повтор...",
                            e.response.status_code,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
                except httpx.TimeoutException:
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning("Embedding API timeout, повтор...")
                        await asyncio.sleep(delay)
                    else:
                        raise

        return all_embeddings

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
