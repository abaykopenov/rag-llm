"""
Embedding клиент — генерация векторов через vLLM или OpenAI-совместимый API.
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
        self._client: Optional[httpx.AsyncClient] = None

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

    async def _request_with_retry(self, path: str, payload: dict) -> dict:
        """Выполнить HTTP запрос с retry и экспоненциальным backoff.

        Args:
            path: URL путь (например, "/embeddings")
            payload: JSON тело запроса

        Returns:
            Ответ API в формате dict

        Raises:
            ConnectionError: Если все попытки исчерпаны
        """
        client = await self._get_client()
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response.json()

            except httpx.ConnectError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        "Embedding API недоступен (попытка {}/{}), повтор через {:.1f}с: {}",
                        attempt, MAX_RETRIES, delay, self.base_url,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error("Embedding API недоступен после {} попыток: {}", MAX_RETRIES, self.base_url)
                    raise ConnectionError(
                        f"Не удалось подключиться к Embedding API по адресу {self.base_url} "
                        f"после {MAX_RETRIES} попыток. Убедитесь, что сервер запущен."
                    ) from e

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        "Embedding API ошибка {} (попытка {}/{}), повтор через {:.1f}с",
                        e.response.status_code, attempt, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = e
                else:
                    raise

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        "Embedding API timeout (попытка {}/{}), повтор через {:.1f}с",
                        attempt, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error("Embedding API timeout после {} попыток", MAX_RETRIES)
                    raise

        raise last_error  # На случай непредвиденного выхода из цикла

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить embeddings для списка текстов.

        Args:
            texts: Список текстов для embedding

        Returns:
            Список embedding-векторов
        """
        if not texts:
            return []

        log.info("Embedding запрос: {} текстов, model={}", len(texts), self.model)
        start = time.perf_counter()

        # Разбиваем на батчи (API может иметь лимит)
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            payload = {
                "model": self.model,
                "input": batch,
            }

            data = await self._request_with_retry("/embeddings", payload)

            # Сортируем по индексу (API может вернуть в другом порядке)
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            batch_embeddings = [item["embedding"] for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        elapsed_ms = (time.perf_counter() - start) * 1000

        log.info(
            "Embedding готов: {} векторов, dim={}, {:.0f} мс",
            len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
            elapsed_ms,
        )

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Получить embedding для одного текста (вопроса).

        Args:
            text: Текст вопроса

        Returns:
            Embedding вектор
        """
        result = await self.embed_texts([text])
        return result[0]

    async def close(self):
        """Закрыть HTTP клиент."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Глобальный экземпляр
embedder = Embedder()

