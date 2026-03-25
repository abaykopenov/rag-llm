"""
Retriever — поиск релевантных чанков по вопросу.

Поддерживает:
- Hybrid search (vector + keyword)
- Metadata фильтры (document, page, section, element_type)
- Reranking (переранжирование через cross-encoder)
- Parent-child (поиск по children, возврат parent чанков)
"""

import time
from typing import Optional

from app.config import settings
from app.core.embedder import embedder
from app.core.indexer import indexer
from app.core.reranker import reranker
from app.core.query_rewriter import query_rewriter, rrf_merge
from app.models.document import RetrievedChunk, ChunkMetadata
from app.utils.logging import get_logger

log = get_logger("retriever")


class Retriever:
    """Поиск релевантных чанков документа по вопросу."""

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        # Metadata фильтры
        document_id: str | None = None,
        element_type: str | None = None,
        section: str | None = None,
        # Hybrid search
        keywords: list[str] | None = None,
        # History для query rewriting
        history: list[dict] | None = None,
    ) -> list[RetrievedChunk]:
        """Найти релевантные чанки.

        Pipeline:
        1. Query Rewriting (переформулирует вопрос)
        2. Multi-Query (генерирует варианты запроса)
        3. Embed вопроса
        4. Cosine search в ChromaDB (с metadata фильтрами)
        5. Keyword search (если hybrid включён)
        6. Merge результатов (RRF — Reciprocal Rank Fusion)
        7. Reranking (если включён)
        8. Parent-child resolution
        9. Фильтрация по score threshold
        """
        top_k = top_k or settings.retrieval_top_k
        score_threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

        log.info("Retrieval: query='{}...', collection='{}', top_k={}", query[:50], collection, top_k)

        start = time.perf_counter()

        # --- 0. Query Rewriting ---
        search_query = query
        if settings.query_rewrite_enabled:
            try:
                search_query = await query_rewriter.rewrite(query, history=history)
            except Exception as e:
                log.warning("Query rewrite failed, using original: {}", e)

        # --- 1. Multi-Query или одиночный поиск ---
        where = self._build_where_filter(document_id, element_type)

        if settings.multi_query_enabled:
            results = await self._multi_query_search(
                search_query, collection, top_k, where, section, keywords,
            )
        else:
            # Одиночный поиск (как раньше)
            query_embedding = await embedder.embed_query(search_query)

            search_top_k = top_k * 3 if (settings.reranker_enabled or keywords) else top_k

            where_document = None
            if section:
                where_document = {"$contains": section}

            results = indexer.query(
                collection_name=collection,
                query_embedding=query_embedding,
                top_k=search_top_k,
                where=where,
                where_document=where_document,
            )

            # Hybrid search: keyword + vector merge
            if keywords:
                results = self._hybrid_merge(
                    vector_results=results,
                    query_keywords=keywords,
                    collection=collection,
                    top_k=search_top_k,
                    where=where,
                )

        # --- Reranking (если включён) ---
        if settings.reranker_enabled and results:
            results = await reranker.rerank(query, results, top_n=top_k)

        # --- Parent-child resolution ---
        if settings.parent_child_enabled:
            results = self._resolve_parents(results, collection)

        # --- Фильтруем по score и создаём RetrievedChunk ---
        chunks = []
        for item in results[:top_k]:
            score = item.get("score", 0)
            if score >= score_threshold:
                meta = item.get("metadata", {})
                chunks.append(RetrievedChunk(
                    id=item["id"],
                    text=item["text"],
                    score=score,
                    metadata=ChunkMetadata(
                        page=meta.get("page"),
                        section=meta.get("section") or None,
                        element_type=meta.get("element_type"),
                        char_count=meta.get("char_count", len(item["text"])),
                        parent_id=meta.get("parent_id"),
                        chunk_type=meta.get("chunk_type", "chunk"),
                    ),
                    original_score=item.get("original_score"),
                    rerank_score=item.get("rerank_score"),
                ))

        total_ms = (time.perf_counter() - start) * 1000
        log.info("Retrieval завершён: {} найдено, {} после фильтра, {:.0f}мс", len(results), len(chunks), total_ms)

        return chunks

    # ═══════════════════════════════════════════════════════
    # Multi-Query Search
    # ═══════════════════════════════════════════════════════

    async def _multi_query_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        where: dict | None,
        section: str | None,
        keywords: list[str] | None,
    ) -> list[dict]:
        """Multi-Query Retrieval: генерируем 3 варианта запроса, ищем по каждому, объединяем через RRF."""
        import asyncio

        queries = await query_rewriter.generate_multi_queries(query)
        search_top_k = top_k * 2

        where_document = {"$contains": section} if section else None

        # Параллельный поиск по всем вариантам
        async def search_one(q: str):
            q_embedding = await embedder.embed_query(q)
            return indexer.query(
                collection_name=collection,
                query_embedding=q_embedding,
                top_k=search_top_k,
                where=where,
                where_document=where_document,
            )

        all_results = await asyncio.gather(*[search_one(q) for q in queries])

        # RRF merge
        merged = rrf_merge(list(all_results))
        log.info("Multi-query: {} запросов, {} уникальных чанков", len(queries), len(merged))
        return merged

    # ═══════════════════════════════════════════════════════
    # Metadata фильтры
    # ═══════════════════════════════════════════════════════

    def _build_where_filter(
        self,
        document_id: str | None = None,
        element_type: str | None = None,
    ) -> dict | None:
        """Построить ChromaDB where-фильтр из параметров.

        Поддерживает:
        - document_id: точное совпадение
        - element_type: text, table, code, list, formula
        """
        conditions = []

        if document_id:
            conditions.append({"document_id": document_id})
        if element_type:
            conditions.append({"element_type": element_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ═══════════════════════════════════════════════════════
    # Hybrid Search
    # ═══════════════════════════════════════════════════════

    def _hybrid_merge(
        self,
        vector_results: list[dict],
        query_keywords: list[str],
        collection: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[dict]:
        """Объединить vector search и keyword search через RRF.

        Reciprocal Rank Fusion:
        rrf_score(d) = Σ 1/(k + rank_i(d))
        где k=60 (стандарт), rank_i — позиция в i-м списке.

        Это даёт стабильное объединение двух типов поиска
        без необходимости нормализации скоров.
        """
        RRF_K = 60  # Стандартный параметр RRF

        # Keyword search
        keyword_results = indexer.keyword_search(
            collection_name=collection,
            keywords=query_keywords,
            limit=top_k,
            where=where,
        )

        log.info(
            "Hybrid merge: {} vector + {} keyword results",
            len(vector_results), len(keyword_results),
        )

        # RRF scoring
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # Vector results — по position rank
        for rank, item in enumerate(vector_results):
            chunk_id = item["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)
            chunk_data[chunk_id] = item

        # Keyword results — по position rank
        for rank, item in enumerate(keyword_results):
            chunk_id = item["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)
            if chunk_id not in chunk_data:
                # Этот чанк нашёлся только через keywords — добавляем
                chunk_data[chunk_id] = {
                    "id": chunk_id,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "score": 0,  # Нет cosine score
                }

        # Сортируем по RRF score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        merged = []
        for chunk_id in sorted_ids[:top_k]:
            item = chunk_data[chunk_id]
            item["rrf_score"] = round(rrf_scores[chunk_id], 6)
            # Используем RRF score как основной, если нет cosine score
            if item.get("score", 0) == 0:
                item["score"] = item["rrf_score"]
            merged.append(item)

        return merged

    # ═══════════════════════════════════════════════════════
    # Parent-Child resolution
    # ═══════════════════════════════════════════════════════

    def _resolve_parents(self, results: list[dict], collection: str) -> list[dict]:
        """Заменить child-чанки на их parent'ов.

        Если найден child-чанк, ищем его parent по parent_id
        и подставляем parent текст (больше контекста для LLM).
        Дедуплицируем parent'ов — если несколько children от одного parent,
        берём parent один раз с лучшим score.
        """
        resolved = []
        seen_parents = set()

        for item in results:
            meta = item.get("metadata", {})
            chunk_type = meta.get("chunk_type", "chunk")
            parent_id = meta.get("parent_id")

            if chunk_type == "child" and parent_id:
                # Это child — пытаемся найти parent
                if parent_id in seen_parents:
                    continue  # Этот parent уже добавлен

                parent = indexer.get_chunk_by_id(collection, parent_id)
                if parent:
                    seen_parents.add(parent_id)
                    resolved.append({
                        "id": parent["id"],
                        "text": parent["text"],
                        "score": item["score"],  # Используем score child'а
                        "metadata": parent["metadata"],
                        "original_score": item.get("original_score"),
                        "rerank_score": item.get("rerank_score"),
                    })
                else:
                    resolved.append(item)
            else:
                if item["id"] not in seen_parents:
                    resolved.append(item)

        return resolved


# Глобальный экземпляр
retriever = Retriever()
