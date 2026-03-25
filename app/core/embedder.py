"""
Embedding клиент — генерация векторов.
Поддерживает два режима:
  1. gemini — нативный Gemini SDK (google-genai)
  2. openai — OpenAI-совместимый API (vLLM, OpenAI, Jina, etc.)
"""

import asyncio
import time
from typing import Optional

import httpx

from app.config import settings
from app.utils.logging import get_logger

log = get_logger("embedder")

# Retry настройки
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # секунды


class Embedder:
    """Клиент для генерации embeddings."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.api_key = api_key or settings.embedding_api_key
        self.model = model or settings.embedding_model
        self.provider = settings.embedding_provider
        self._client: Optional[httpx.AsyncClient] = None
        self._genai_client = None

    # ─────────────────────────────────────────────────
    # Gemini Native SDK (рекомендуемый)
    # ─────────────────────────────────────────────────

    def _get_genai_client(self):
        """Lazy init клиента google-genai."""
        if self._genai_client is None:
            from google import genai
            api_key = self.api_key or settings.llm_api_key
            self._genai_client = genai.Client(api_key=api_key)
            log.info("Gemini embedding клиент инициализирован")
        return self._genai_client

    async def _embed_with_gemini(self, texts: list[str]) -> list[list[float]]:
        """Embedding через Gemini нативный SDK.

        Использует client.models.embed_content() — поддерживает batching.
        """
        client = self._get_genai_client()
        all_embeddings = []

        # Gemini поддерживает batch до ~100 текстов
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # Запускаем синхронный SDK в executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda b=batch: client.models.embed_content(
                            model=self.model,
                            contents=b,
                        )
                    )
                    # Извлекаем векторы
                    for emb in result.embeddings:
                        all_embeddings.append(list(emb.values))
                    break

                except Exception as e:
                    error_str = str(e)
                    if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < MAX_RETRIES:
                        wait = 30 * attempt
                        log.warning(
                            "Gemini embedding rate limit (попытка {}/{}). Ждём {} сек...",
                            attempt, MAX_RETRIES, wait
                        )
                        await asyncio.sleep(wait)
                    elif attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning("Gemini embedding ошибка (попытка {}/{}): {}", attempt, MAX_RETRIES, e)
                        await asyncio.sleep(delay)
                    else:
                        raise

        return all_embeddings

    # ─────────────────────────────────────────────────
    # OpenAI-compatible API (vLLM, OpenAI, Jina, etc.)
    # ─────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy init HTTP клиента."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0),
            )
        return self._client

    async def _embed_with_openai(self, texts: list[str]) -> list[list[float]]:
        """Embedding через OpenAI-совместимый API."""
        client = await self._get_client()
        all_embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "model": self.model,
                "input": batch,
            }

            last_error = None
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
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning(
                            "Embedding API недоступен (попытка {}/{}), повтор через {:.1f}с",
                            attempt, MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise ConnectionError(
                            f"Не удалось подключиться к Embedding API: {self.base_url}"
                        ) from e
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500 and attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning("Embedding API ошибка {}, повтор...", e.response.status_code)
                        await asyncio.sleep(delay)
                    else:
                        raise
                except httpx.TimeoutException as e:
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        log.warning("Embedding API timeout, повтор...")
                        await asyncio.sleep(delay)
                    else:
                        raise

        return all_embeddings

    # ─────────────────────────────────────────────────
    # Публичный интерфейс
    # ─────────────────────────────────────────────────

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить embeddings для списка текстов."""
        if not texts:
            return []

        log.info("Embedding запрос: {} текстов, provider={}, model={}",
                 len(texts), self.provider, self.model)
        start = time.perf_counter()

        if self.provider == "gemini":
            all_embeddings = await self._embed_with_gemini(texts)
        else:
            all_embeddings = await self._embed_with_openai(texts)

        elapsed_ms = (time.perf_counter() - start) * 1000

        log.info(
            "Embedding готов: {} векторов, dim={}, {:.0f} мс",
            len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
            elapsed_ms,
        )

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Получить embedding для одного текста (вопроса)."""
        result = await self.embed_texts([text])
        return result[0]

    async def close(self):
        """Закрыть HTTP клиент."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Глобальный экземпляр
embedder = Embedder()
