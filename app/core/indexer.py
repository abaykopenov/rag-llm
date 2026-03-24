"""
Indexer — сохранение и управление чанками в ChromaDB.
"""

import time
from typing import Optional

import chromadb

from app.config import settings
from app.models.document import Chunk
from app.utils.logging import get_logger

log = get_logger("indexer")


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

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Получить или создать коллекцию."""
        client = self._get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

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

        return len(chunks)

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
        """Поиск по ключевым словам (для hybrid search).

        Args:
            collection_name: Имя коллекции
            keywords: Ключевые слова для поиска
            limit: Максимум результатов
            where: Опциональный фильтр по метаданным

        Returns:
            Список чанков, содержащих ключевые слова
        """
        collection = self.get_or_create_collection(collection_name)

        all_results = {}

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

        # Сортируем по количеству совпадений
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
        """Удалить коллекцию."""
        client = self._get_client()
        try:
            client.delete_collection(name)
            log.info("Коллекция '{}' удалена", name)
        except Exception as e:
            log.warning("Не удалось удалить коллекцию '{}': {}", name, e)

    def list_collections(self) -> list[dict]:
        """Список всех коллекций."""
        client = self._get_client()
        collections = client.list_collections()
        result = []
        for col_name in collections:
            try:
                # ChromaDB 1.0+ возвращает список имён строк
                if isinstance(col_name, str):
                    col = client.get_collection(col_name)
                    count = col.count()
                    result.append({"name": col_name, "chunks_count": count})
                else:
                    # Старые версии возвращают Collection объекты
                    count = col_name.count()
                    result.append({"name": col_name.name, "chunks_count": count})
            except Exception:
                result.append({"name": str(col_name), "chunks_count": 0})
        return result


# Глобальный экземпляр
indexer = Indexer()
