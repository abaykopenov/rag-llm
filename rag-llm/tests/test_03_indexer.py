"""
Тест 3: ChromaDB Indexer.
Проверяет индексацию, поиск, и Stage 2 методы (get_chunks_by_document, get_chunk_by_id).
Требует: chromadb (pip install chromadb)
Не требует: vLLM, Embedding API (используем фейковые embeddings)
"""

import shutil
import tempfile
from pathlib import Path

import pytest


def _get_test_chunks(n: int = 5, document_id: str = "test-doc-001"):
    """Создать тестовые чанки."""
    from app.models.document import Chunk, ChunkMetadata

    texts = [
        "RAG (Retrieval-Augmented Generation) объединяет поиск и генерацию текста для более точных ответов.",
        "ChromaDB — это векторная база данных, оптимизированная для хранения и поиска по embeddings.",
        "Docling от IBM — мощный парсер документов, поддерживающий PDF, DOCX, HTML и другие форматы.",
        "vLLM обеспечивает высокоскоростной инференс языковых моделей с оптимизацией памяти GPU.",
        "FastAPI — асинхронный web-фреймворк на Python, идеально подходящий для ML-сервисов.",
    ]

    chunks = []
    for i in range(min(n, len(texts))):
        chunks.append(Chunk(
            document_id=document_id,
            text=texts[i],
            metadata=ChunkMetadata(
                page=i + 1,
                section=f"Раздел {i + 1}",
                element_type="text",
                char_count=len(texts[i]),
            ),
        ))
    return chunks


def _generate_fake_embeddings(n: int, dim: int = 384) -> list[list[float]]:
    """Сгенерировать фейковые embeddings для тестирования."""
    import random
    random.seed(42)
    return [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]


# =================== Тесты Indexer ===================

def test_indexer_add_and_query():
    """Проверить индексацию и поиск чанков."""
    from app.core.indexer import Indexer

    # Используем отдельный экземпляр с temp dir
    tmp_dir = tempfile.mkdtemp()
    try:
        indexer = Indexer()
        # Monkey-patch для использования temporary PersistentClient
        import chromadb
        indexer._client = chromadb.PersistentClient(path=tmp_dir)

        chunks = _get_test_chunks(5)
        embeddings = _generate_fake_embeddings(5)

        # Индексация
        count = indexer.add_chunks("test-collection", chunks, embeddings)
        assert count == 5
        print(f"  ✅ Индексация: {count} чанков добавлено")

        # Поиск
        query_embedding = _generate_fake_embeddings(1)[0]
        results = indexer.query("test-collection", query_embedding, top_k=3)

        assert len(results) == 3
        assert all("id" in r for r in results)
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("metadata" in r for r in results)

        print(f"  ✅ Поиск: найдено {len(results)} чанков (запрошено top_k=3)")
        for r in results:
            print(f"     score={r['score']:.4f} | {r['text'][:50]}...")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_indexer_get_chunks_by_document():
    """Проверить Stage 2: получение чанков по document_id."""
    from app.core.indexer import Indexer
    import chromadb

    tmp_dir = tempfile.mkdtemp()
    try:
        indexer = Indexer()
        indexer._client = chromadb.PersistentClient(path=tmp_dir)

        # Добавляем чанки от двух документов
        chunks_a = _get_test_chunks(3, document_id="doc-AAA")
        chunks_b = _get_test_chunks(2, document_id="doc-BBB")
        em_a = _generate_fake_embeddings(3)
        em_b = _generate_fake_embeddings(2)

        indexer.add_chunks("col1", chunks_a, em_a)
        indexer.add_chunks("col1", chunks_b, em_b)

        # Получаем чанки doc-AAA
        result_a = indexer.get_chunks_by_document("col1", "doc-AAA")
        assert len(result_a) == 3
        assert all(r["metadata"]["document_id"] == "doc-AAA" for r in result_a)
        print(f"  ✅ get_chunks_by_document('doc-AAA'): {len(result_a)} чанков")

        # Получаем чанки doc-BBB
        result_b = indexer.get_chunks_by_document("col1", "doc-BBB")
        assert len(result_b) == 2
        print(f"  ✅ get_chunks_by_document('doc-BBB'): {len(result_b)} чанков")

        # Несуществующий документ
        result_none = indexer.get_chunks_by_document("col1", "doc-NONEXIST")
        assert len(result_none) == 0
        print(f"  ✅ get_chunks_by_document('doc-NONEXIST'): 0 чанков (ожидаемо)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_indexer_get_chunk_by_id():
    """Проверить Stage 2: получение одного чанка по ID."""
    from app.core.indexer import Indexer
    import chromadb

    tmp_dir = tempfile.mkdtemp()
    try:
        indexer = Indexer()
        indexer._client = chromadb.PersistentClient(path=tmp_dir)

        chunks = _get_test_chunks(3)
        embeddings = _generate_fake_embeddings(3)
        indexer.add_chunks("col-detail", chunks, embeddings)

        target_id = chunks[1].id

        # Без embedding
        result = indexer.get_chunk_by_id("col-detail", target_id)
        assert result is not None
        assert result["id"] == target_id
        assert result["text"] == chunks[1].text
        assert "embedding" not in result
        print(f"  ✅ get_chunk_by_id (без embedding): id={target_id[:12]}...")

        # С embedding
        result_with_emb = indexer.get_chunk_by_id("col-detail", target_id, include_embedding=True)
        assert result_with_emb is not None
        assert "embedding" in result_with_emb
        assert len(result_with_emb["embedding"]) == 384  # наша фейковая размерность
        print(f"  ✅ get_chunk_by_id (с embedding): {len(result_with_emb['embedding'])} dimensions")

        # Несуществующий чанк
        result_none = indexer.get_chunk_by_id("col-detail", "nonexistent-id-xyz")
        assert result_none is None
        print(f"  ✅ get_chunk_by_id('nonexistent'): None (ожидаемо)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_indexer_list_and_delete_collections():
    """Проверить список и удаление коллекций."""
    from app.core.indexer import Indexer
    import chromadb

    tmp_dir = tempfile.mkdtemp()
    try:
        indexer = Indexer()
        indexer._client = chromadb.PersistentClient(path=tmp_dir)

        # Создаём коллекции
        chunks = _get_test_chunks(2)
        embeddings = _generate_fake_embeddings(2)
        indexer.add_chunks("alpha", chunks, embeddings)
        indexer.add_chunks("beta", chunks, embeddings)

        # Список
        collections = indexer.list_collections()
        names = [c["name"] for c in collections]
        assert "alpha" in names
        assert "beta" in names
        print(f"  ✅ list_collections: {names}")

        # Удаление
        indexer.delete_collection("alpha")
        collections_after = indexer.list_collections()
        names_after = [c["name"] for c in collections_after]
        assert "alpha" not in names_after
        assert "beta" in names_after
        print(f"  ✅ delete_collection('alpha'): осталось {names_after}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("\n🧪 Тест 3: ChromaDB Indexer\n" + "=" * 50)

    print("\n--- Индексация и поиск ---")
    test_indexer_add_and_query()

    print("\n--- Stage 2: Чанки по документу ---")
    test_indexer_get_chunks_by_document()

    print("\n--- Stage 2: Чанк по ID ---")
    test_indexer_get_chunk_by_id()

    print("\n--- Коллекции ---")
    test_indexer_list_and_delete_collections()

    print("\n✅ Все тесты пройдены!\n")
