"""
Indexer — сохранение и управление чанками в ChromaDB.

Каждая коллекция стемплируется паспортом провайдера
(provider/model/dim) в её metadata при создании. При каждом
обращении (add/query) проверяется, что активный embedding-провайдер
совпадает с тем, что был при создании — иначе ChromaDB либо упадёт
на dim-mismatch, либо тихо сохранит мусор. Проверка даёт понятную ошибку.
"""

import time
from typing import Optional

import chromadb

from app.config import settings
from app.core.embedding.base import EmbeddingProviderInfo
from app.models.document import Chunk
from app.utils.logging import get_logger

log = get_logger("indexer")

# Ключи metadata, в которых хранится паспорт провайдера
_PROVIDER_META_KEYS = (
    "embedding_provider",
    "embedding_model",
    "embedding_dim",
)


class EmbeddingMismatchError(RuntimeError):
    """Активный провайдер не совпадает с тем, которым создана коллекция."""


class Indexer:
    """Управление векторным хранилищем ChromaDB."""

    def __init__(self):
        self._client: Optional[chromadb.ClientAPI] = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Получить клиент ChromaDB."""
        if self._client is None:
            if settings.chroma_host:
                # Client-server режим
                log.info("Подключение к ChromaDB: {}:{}", settings.chroma_host, settings.chroma_port)
                self._client = chromadb.HttpClient(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                )
            else:
                # Встроенный режим (persistent)
                log.info("ChromaDB в встроенном режиме: {}", settings.chroma_persist_dir)
                self._client = chromadb.PersistentClient(
                    path=settings.chroma_persist_dir,
                )
            log.info("ChromaDB подключён")
        return self._client

    def _active_provider_info(self) -> Optional[EmbeddingProviderInfo]:
        """Получить паспорт активного провайдера (ленивый импорт, чтобы избежать циклов).

        Возвращает None, если провайдер не может быть инициализирован (например,
        в operational-командах CLI, где провайдер может быть не нужен).
        """
        try:
            from app.core.embedder import embedder
            return embedder.info
        except Exception as e:
            log.debug("Не удалось получить активного провайдера: {}", e)
            return None

    def get_or_create_collection(
        self,
        name: str,
        expected_info: Optional[EmbeddingProviderInfo] = None,
        check: bool = True,
    ) -> chromadb.Collection:
        """Получить или создать коллекцию с проверкой embedding-провайдера.

        Args:
            name: Имя коллекции
            expected_info: Паспорт провайдера, который должен быть у коллекции.
                           None = взять у активного `embedder`.
            check: Если True — проверять совпадение с metadata коллекции.
                   Новые коллекции штампуются, legacy — штампуются с warning.

        Raises:
            EmbeddingMismatchError: Коллекция создана другим провайдером.
        """
        client = self._get_client()

        if expected_info is None and check:
            expected_info = self._active_provider_info()

        # Собираем metadata для get_or_create_collection.
        # ChromaDB НЕ перезапишет metadata существующей коллекции — это ок,
        # потому что ниже мы сами делаем merge + collection.modify().
        base_meta: dict = {"hnsw:space": "cosine"}
        if expected_info is not None:
            base_meta.update(expected_info.to_meta())

        coll = client.get_or_create_collection(name=name, metadata=base_meta)

        if not check or expected_info is None:
            return coll

        stored = EmbeddingProviderInfo.from_meta(coll.metadata or {})

        if stored is None:
            # Legacy-коллекция без паспорта — штампуем текущим провайдером
            log.warning(
                "Коллекция '{}' без embedding-паспорта — штампую: {}",
                name, expected_info,
            )
            new_meta = dict(coll.metadata or {})
            new_meta.update(expected_info.to_meta())
            try:
                coll.modify(metadata=new_meta)
            except Exception as e:
                log.warning("Не удалось записать metadata в '{}': {}", name, e)
            return coll

        if not stored.matches(expected_info):
            raise EmbeddingMismatchError(
                f"Коллекция '{name}' создана провайдером {stored}, но активный — "
                f"{expected_info}. Запусти `python -m app.cli reindex --collection {name} "
                f"--to-profile <имя>` или смени RAG_EMBEDDING_PROFILE."
            )

        return coll

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Добавить чанки с embeddings в коллекцию.

        Args:
            collection_name: Имя коллекции
            chunks: Список чанков
            embeddings: Список embedding-векторов

        Returns:
            Количество добавленных чанков
        """
        if not chunks:
            return 0

        log.info("Индексация: {} чанков в коллекцию '{}'", len(chunks), collection_name)
        start = time.perf_counter()

        collection = self.get_or_create_collection(collection_name)

        # Готовим данные для ChromaDB
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "filename": getattr(c.metadata, 'filename', '') or "",  # Добавляем filename
                "page": c.metadata.page or 0,
                "section": c.metadata.section or "",
                "element_type": c.metadata.element_type or "text",
                "char_count": c.metadata.char_count,
                "parent_id": c.metadata.parent_id or "",
                "chunk_type": c.metadata.chunk_type or "chunk",
            }
            for c in chunks
        ]

        # Добавляем батчами (ChromaDB имеет лимит)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("Индексация завершена: {} чанков, {:.0f} мс", len(chunks), elapsed_ms)

        # Инвалидируем BM25 — при следующем keyword-запросе пересоберётся
        if settings.bm25_enabled:
            self._invalidate_bm25(collection_name)

        return len(chunks)

    @staticmethod
    def _invalidate_bm25(collection_name: str) -> None:
        """Сбросить BM25-индекс после изменения коллекции."""
        try:
            from app.core.bm25_index import bm25_registry
            bm25_registry.invalidate(collection_name)
        except Exception as e:
            log.debug("BM25 invalidation failed: {}", e)

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """Поиск похожих чанков с опциональными фильтрами.

        Args:
            collection_name: Имя коллекции
            query_embedding: Embedding вектор запроса
            top_k: Количество результатов
            where: Фильтр по метаданным (ChromaDB where)
                   Пример: {"document_id": "abc"}, {"element_type": "table"}
            where_document: Фильтр по тексту документа (keyword search)
                   Пример: {"$contains": "выручка"}

        Returns:
            Список найденных чанков с scores
        """
        collection = self.get_or_create_collection(collection_name)

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_args["where"] = where
        if where_document:
            query_args["where_document"] = where_document

        results = collection.query(**query_args)

        # Преобразуем ChromaDB ответ в удобный формат
        items = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB возвращает distance (cosine), конвертируем в similarity
                distance = results["distances"][0][i]
                score = 1 - distance  # cosine distance → cosine similarity

                items.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i],
                })

        return items

    def keyword_search(
        self,
        collection_name: str,
        keywords: list[str],
        limit: int = 20,
        where: dict | None = None,
    ) -> list[dict]:
        """Поиск по ключевым словам.

        По умолчанию (settings.bm25_enabled=True) использует BM25 со стеммингом
        — поэтому "выручки" находит "выручка". Fallback на substring-match
        (старое поведение) — RAG_BM25_ENABLED=false.

        Args:
            collection_name: Имя коллекции
            keywords: Ключевые слова (объединяются в один BM25-запрос)
            limit: Максимум результатов
            where: Фильтр по метаданным. Для BM25 применяется клиентски
                   (см. bm25_index._match_where)

        Returns:
            Список {id, text, metadata, score, rank}
        """
        if not keywords:
            return []

        if settings.bm25_enabled:
            return self._keyword_search_bm25(collection_name, keywords, limit, where)
        return self._keyword_search_substring(collection_name, keywords, limit, where)

    def _keyword_search_bm25(
        self,
        collection_name: str,
        keywords: list[str],
        limit: int,
        where: dict | None,
    ) -> list[dict]:
        """BM25-поиск с ленивой сборкой индекса."""
        from app.core.bm25_index import bm25_registry

        # Не проверяем провайдера — BM25 не зависит от embedding-размерности
        chroma_coll = self.get_or_create_collection(collection_name, check=False)
        bm25 = bm25_registry.get(collection_name)
        bm25.ensure_ready(chroma_coll)

        query = " ".join(k for k in keywords if k and k.strip())
        return bm25.search(query, top_k=limit, where=where)

    def _keyword_search_substring(
        self,
        collection_name: str,
        keywords: list[str],
        limit: int,
        where: dict | None,
    ) -> list[dict]:
        """[LEGACY] Substring-match через ChromaDB where_document."""
        collection = self.get_or_create_collection(collection_name, check=False)

        all_results: dict[str, dict] = {}

        for keyword in keywords:
            try:
                get_args = {
                    "where_document": {"$contains": keyword},
                    "limit": limit,
                    "include": ["documents", "metadatas"],
                }
                if where:
                    get_args["where"] = where

                results = collection.get(**get_args)

                if results["ids"]:
                    for i, chunk_id in enumerate(results["ids"]):
                        if chunk_id not in all_results:
                            all_results[chunk_id] = {
                                "id": chunk_id,
                                "text": results["documents"][i] if results["documents"] else "",
                                "metadata": results["metadatas"][i] if results["metadatas"] else {},
                                "keyword_hits": 0,
                            }
                        all_results[chunk_id]["keyword_hits"] += 1

            except Exception as e:
                log.debug("Keyword search error for '{}': {}", keyword, e)

        items = list(all_results.values())
        items.sort(key=lambda x: x["keyword_hits"], reverse=True)
        return items[:limit]

    def get_chunks_by_document(
        self,
        collection_name: str,
        document_id: str,
        limit: int = 1000,
    ) -> list[dict]:
        """Получить все чанки документа из коллекции.

        Args:
            collection_name: Имя коллекции
            document_id: ID документа
            limit: Максимум чанков

        Returns:
            Список чанков с метаданными
        """
        collection = self.get_or_create_collection(collection_name)

        results = collection.get(
            where={"document_id": document_id},
            limit=limit,
            include=["documents", "metadatas"],
        )

        items = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                items.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": meta,
                })

        log.info("Чанки документа {}: найдено {} в '{}'", document_id[:8], len(items), collection_name)
        return items

    def get_chunk_by_id(
        self,
        collection_name: str,
        chunk_id: str,
        include_embedding: bool = False,
    ) -> dict | None:
        """Получить один чанк по ID.

        Args:
            collection_name: Имя коллекции
            chunk_id: ID чанка
            include_embedding: Включить embedding вектор в ответ

        Returns:
            Чанк с метаданными (и опционально embedding), или None
        """
        collection = self.get_or_create_collection(collection_name)

        include = ["documents", "metadatas"]
        if include_embedding:
            include.append("embeddings")

        try:
            results = collection.get(
                ids=[chunk_id],
                include=include,
            )
        except Exception as e:
            log.warning("Чанк '{}' не найден в '{}': {}", chunk_id, collection_name, e)
            return None

        if not results["ids"]:
            return None

        item = {
            "id": results["ids"][0],
            "text": results["documents"][0] if results["documents"] else "",
            "metadata": results["metadatas"][0] if results["metadatas"] else {},
        }

        if include_embedding and results.get("embeddings") is not None:
            item["embedding"] = results["embeddings"][0]

        return item

    def delete_collection(self, name: str):
        """Удалить коллекцию (и её BM25-индекс)."""
        client = self._get_client()
        try:
            client.delete_collection(name)
            log.info("Коллекция '{}' удалена", name)
        except Exception as e:
            log.warning("Не удалось удалить коллекцию '{}': {}", name, e)

        if settings.bm25_enabled:
            self._invalidate_bm25(name)

    def list_collections(self) -> list[dict]:
        """Список всех коллекций c паспортом провайдера (если есть)."""
        client = self._get_client()
        collections = client.list_collections()
        result = []
        for col_name in collections:
            try:
                # ChromaDB 1.0+ возвращает список имён строк
                if isinstance(col_name, str):
                    col = client.get_collection(col_name)
                else:
                    col = col_name
                name = col.name if hasattr(col, "name") else str(col_name)
                count = col.count()
                info = EmbeddingProviderInfo.from_meta(col.metadata or {})
                entry: dict = {"name": name, "chunks_count": count}
                if info:
                    entry["embedding_provider"] = info.provider
                    entry["embedding_model"] = info.model
                    entry["embedding_dim"] = info.dim
                result.append(entry)
            except Exception:
                name = col_name if isinstance(col_name, str) else str(col_name)
                result.append({"name": name, "chunks_count": 0})
        return result

    def iter_all_chunks(
        self,
        collection_name: str,
        page_size: int = 500,
    ):
        """Итератор по всем чанкам коллекции (без embeddings).

        Используется в CLI `reindex` для перегона данных между провайдерами.
        ChromaDB collection.get() поддерживает offset/limit начиная с 0.5.x.
        """
        # Не проверяем провайдера здесь: reindex должен уметь читать коллекции,
        # созданные любым провайдером (иначе невозможно с них мигрировать).
        coll = self.get_or_create_collection(collection_name, check=False)
        total = coll.count()
        offset = 0
        while offset < total:
            batch = coll.get(
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = batch.get("ids") or []
            if not ids:
                break
            for i, chunk_id in enumerate(ids):
                yield {
                    "id": chunk_id,
                    "text": (batch.get("documents") or [""] * len(ids))[i],
                    "metadata": (batch.get("metadatas") or [{}] * len(ids))[i] or {},
                }
            offset += len(ids)

    def add_raw(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        expected_info: Optional[EmbeddingProviderInfo] = None,
        batch_size: int = 100,
    ) -> int:
        """Низкоуровневое добавление (используется reindex).

        В отличие от add_chunks, принимает уже готовые id/text/metadata
        и явный expected_info — это нужно, чтобы при reindex штамповать
        целевую коллекцию правильным паспортом (а не активным провайдером,
        если они вдруг различаются).
        """
        if not ids:
            return 0
        coll = self.get_or_create_collection(
            collection_name, expected_info=expected_info, check=True,
        )
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            coll.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        if settings.bm25_enabled:
            self._invalidate_bm25(collection_name)

        return len(ids)


# Глобальный экземпляр
indexer = Indexer()
