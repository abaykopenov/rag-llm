"""
Reranker — переранжирование чанков через cross-encoder.

После начального поиска (cosine similarity) reranker оценивает
каждую пару (вопрос, чанк) через cross-encoder модель,
что значительно повышает точность результатов.

Поддерживает два режима:
1. API — через vLLM/OpenAI-совместимый API (rerank endpoint)
2. Disabled — пропускает reranking (для тестов или если модели нет)
"""

import time
from typing import Optional

import httpx

from app.config import settings
from app.utils.logging import get_logger

log = get_logger("reranker")


class Reranker:
    """Cross-encoder reranker для переранжирования результатов поиска."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.enabled = settings.reranker_enabled
        self.base_url = (base_url or settings.reranker_base_url).rstrip("/")
        self.api_key = api_key or settings.reranker_api_key
        self.model = model or settings.reranker_model
        self.top_n = settings.reranker_top_n
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
                timeout=httpx.Timeout(60.0),
            )
        return self._client

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """Переранжировать чанки по релевантности к вопросу.

        Args:
            query: Вопрос пользователя
            chunks: Список чанков из начального поиска (каждый имеет 'text', 'id', 'score', etc.)
            top_n: Сколько лучших вернуть (default: settings.reranker_top_n)

        Returns:
            Переранжированный список чанков с обновлённым score
        """
        if not self.enabled or not chunks:
            return chunks

        top_n = top_n or self.top_n
        start = time.perf_counter()

        log.info(
            "Reranking: {} чанков → top-{}, model={}",
            len(chunks), top_n, self.model,
        )

        try:
            reranked = await self._rerank_api(query, chunks, top_n)
        except Exception as e:
            log.warning("Reranker API error, falling back to simple reranking: {}", e)
            reranked = await self.rerank_simple(query, chunks, top_n)

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("Reranking завершён: {} → {} чанков, {:.0f} мс", len(chunks), len(reranked), elapsed_ms)

        return reranked

    async def _rerank_api(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        """Reranking через API (Jina, Cohere, или vLLM reranker).

        Использует стандартный rerank API формат:
        POST /rerank {"query": "...", "documents": [...], "top_n": N}
        """
        client = await self._get_client()

        documents = [c.get("text", "") for c in chunks]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        response = await client.post("/rerank", json=payload)
        response.raise_for_status()
        data = response.json()

        # API возвращает: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        reranked = []
        for result in data.get("results", []):
            idx = result["index"]
            if idx < len(chunks):
                chunk = chunks[idx].copy()
                chunk["original_score"] = chunk.get("score", 0)
                chunk["score"] = round(result.get("relevance_score", 0), 4)
                chunk["rerank_score"] = chunk["score"]
                reranked.append(chunk)

        # Сортируем по rerank score
        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)

        return reranked[:top_n]

    async def rerank_simple(
        self,
        query: str,
        chunks: list[dict],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """Простой reranking без API — через embedding similarity (fallback).

        Используется когда reranker API недоступен.
        Переоценивает скоры на основе текстового совпадения ключевых слов.
        """
        top_n = top_n or self.top_n

        query_words = set(query.lower().split())

        scored = []
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            text_words = set(text.split())

            # Процент совпадения слов вопроса с текстом чанка
            overlap = len(query_words & text_words) / max(len(query_words), 1)

            # Комбинируем cosine score и keyword overlap
            original_score = chunk.get("score", 0)
            combined_score = 0.6 * original_score + 0.4 * overlap

            entry = chunk.copy()
            entry["original_score"] = original_score
            entry["score"] = round(combined_score, 4)
            scored.append(entry)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    async def close(self):
        """Закрыть HTTP клиент."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Глобальный экземпляр
reranker = Reranker()
