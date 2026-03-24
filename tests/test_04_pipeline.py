"""
Тест 4: Полный пайплайн — от файла до готовых чанков в ChromaDB.
Тестирует цепочку: Parse → Chunk → Index → Search (без LLM/Embedding API).
Использует фейковые embeddings.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest


def _create_rag_document(tmp_dir: str) -> Path:
    """Создать тестовый документ про RAG."""
    content = """# Полное руководство по RAG

## 1. Что такое RAG?

Retrieval-Augmented Generation (RAG) — это архитектурный паттерн, который объединяет 
поиск информации (retrieval) с генерацией текста (generation) для создания более 
точных и обоснованных ответов от языковых моделей.

RAG был предложен в 2020 году исследователями из Facebook AI Research (теперь Meta AI) 
в статье "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". 
С тех пор он стал стандартным подходом для enterprise-применений LLM.

## 2. Архитектура RAG-системы

### 2.1 Индексация документов
Процесс индексации включает следующие этапы:
- **Парсинг**: извлечение текста из различных форматов (PDF, DOCX, HTML)
- **Чанкинг**: разделение текста на фрагменты оптимального размера
- **Embedding**: преобразование текстовых фрагментов в числовые вектора
- **Хранение**: сохранение векторов в специализированной базе данных

### 2.2 Обработка запросов
При получении вопроса от пользователя:
1. Вопрос преобразуется в embedding вектор
2. Выполняется поиск ближайших соседей в векторной БД
3. Найденные фрагменты формируют контекст для LLM
4. LLM генерирует ответ на основе контекста и вопроса

## 3. Преимущества RAG

| Преимущество | Описание |
|---|---|
| Актуальность | Использует свежие данные, а не устаревшие знания модели |
| Точность | Ответы основаны на реальных документах |
| Прозрачность | Можно показать источники ответа |
| Масштабируемость | Можно добавлять новые документы без переобучения |

## 4. Технологический стек

### 4.1 vLLM
vLLM — это высокопроизводительный движок для инференса LLM. 
Он поддерживает модели Hugging Face и обеспечивает высокую скорость 
генерации благодаря PagedAttention и continuous batching.

### 4.2 ChromaDB
ChromaDB — открытая векторная база данных для AI-приложений. 
Она поддерживает различные метрики расстояния и легко интегрируется с Python.

### 4.3 Docling
Docling от IBM — продвинутый парсер документов, способный извлекать 
структурированный текст, таблицы, формулы из сложных PDF-файлов.

## 5. Заключение

RAG является ключевой технологией для построения надёжных AI-систем, 
способных работать с корпоративными данными. Комбинация поиска и генерации 
позволяет создавать точные, актуальные и проверяемые ответы.
"""
    file_path = Path(tmp_dir) / "rag_guide.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_full_pipeline_parse_chunk_index_search():
    """
    ПОЛНЫЙ ПАЙПЛАЙН: parse → chunk → index → search.
    Без внешних API (фейковые embeddings).
    """
    from app.core.parser import DocumentParser
    from app.core.chunker import DocumentChunker
    from app.core.indexer import Indexer
    from app.models.document import Document
    from app.utils.document_store import DocumentStore
    import chromadb
    import random
    import time

    random.seed(42)

    tmp_dir = tempfile.mkdtemp()
    chroma_dir = tempfile.mkdtemp()
    texts_dir = tempfile.mkdtemp()

    try:
        total_start = time.perf_counter()

        # Инициализация компонентов
        parser = DocumentParser()
        chunker = DocumentChunker()
        indexer = Indexer()
        indexer._client = chromadb.PersistentClient(path=chroma_dir)
        store = DocumentStore(storage_dir=tmp_dir)

        # Создаём документ
        file_path = _create_rag_document(tmp_dir)

        doc = Document(
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            collection="test-pipeline",
            status="processing",
        )

        # ====== STEP 1: PARSE ======
        print(f"\n  📄 Step 1: PARSE ({file_path.name})")
        parse_start = time.perf_counter()
        parse_result = parser.parse(file_path)
        parse_ms = (time.perf_counter() - parse_start) * 1000

        assert parse_result.full_text is not None
        assert len(parse_result.full_text) > 200
        doc.pages_count = parse_result.pages_count
        doc.raw_text = parse_result.full_text

        print(f"     ✅ Текст: {len(parse_result.full_text)} символов, "
              f"{parse_result.pages_count} стр., {parse_ms:.0f} мс")

        # Сохраняем текст (Stage 2)
        text_path = Path(texts_dir) / f"{doc.id}.md"
        text_path.write_text(parse_result.full_text, encoding="utf-8")
        print(f"     ✅ Текст сохранён: {text_path.name}")

        # ====== STEP 2: CHUNK ======
        print(f"\n  ✂️  Step 2: CHUNK")
        chunk_start = time.perf_counter()
        chunks = chunker.chunk(parse_result.docling_document, doc.id)
        chunk_ms = (time.perf_counter() - chunk_start) * 1000

        assert len(chunks) > 0
        doc.chunks_count = len(chunks)

        avg_len = sum(c.metadata.char_count for c in chunks) // len(chunks)
        print(f"     ✅ Чанки: {len(chunks)} шт., avg {avg_len} символов, {chunk_ms:.0f} мс")

        for i, c in enumerate(chunks[:3]):
            section = c.metadata.section or "—"
            print(f"     [{i+1}] page={c.metadata.page}, section='{section}', "
                  f"len={c.metadata.char_count}: \"{c.text[:60]}...\"")
        if len(chunks) > 3:
            print(f"     ... и ещё {len(chunks) - 3} чанков")

        # ====== STEP 3: EMBED (фейковый) ======
        print(f"\n  🔢 Step 3: EMBED (fake, {384}d vectors)")
        embed_start = time.perf_counter()
        fake_embeddings = [[random.uniform(-1, 1) for _ in range(384)] for _ in range(len(chunks))]
        embed_ms = (time.perf_counter() - embed_start) * 1000
        print(f"     ✅ Embeddings: {len(fake_embeddings)} × {384}d, {embed_ms:.1f} мс")

        # ====== STEP 4: INDEX ======
        print(f"\n  💾 Step 4: INDEX (ChromaDB)")
        index_start = time.perf_counter()
        count = indexer.add_chunks("test-pipeline", chunks, fake_embeddings)
        index_ms = (time.perf_counter() - index_start) * 1000
        assert count == len(chunks)
        print(f"     ✅ Индексировано: {count} чанков, {index_ms:.0f} мс")

        # ====== STEP 5: SEARCH ======
        print(f"\n  🔍 Step 5: SEARCH")
        query_embedding = [random.uniform(-1, 1) for _ in range(384)]
        search_start = time.perf_counter()
        results = indexer.query("test-pipeline", query_embedding, top_k=3)
        search_ms = (time.perf_counter() - search_start) * 1000

        assert len(results) == 3
        print(f"     ✅ Результаты: {len(results)} чанков, {search_ms:.1f} мс")
        for r in results:
            print(f"        score={r['score']:.4f} | \"{r['text'][:60]}...\"")

        # ====== STEP 6: STAGE 2 CHECKS ======
        print(f"\n  👁️  Step 6: STAGE 2 VISIBILITY")

        # Проверяем текст документа
        saved_text = text_path.read_text(encoding="utf-8")
        assert saved_text == parse_result.full_text
        print(f"     ✅ GET /documents/{{id}}/text: {len(saved_text)} символов")

        # Чанки документа
        doc_chunks = indexer.get_chunks_by_document("test-pipeline", doc.id)
        assert len(doc_chunks) == len(chunks)
        print(f"     ✅ GET /documents/{{id}}/chunks: {len(doc_chunks)} чанков")

        # Чанк по ID
        first_chunk = indexer.get_chunk_by_id("test-pipeline", chunks[0].id, include_embedding=True)
        assert first_chunk is not None
        assert first_chunk["text"] == chunks[0].text
        assert "embedding" in first_chunk
        print(f"     ✅ GET /chunks/{{id}}: текст + {len(first_chunk['embedding'])}d embedding")

        # ====== STEP 7: DOCUMENT STORE ======
        print(f"\n  📦 Step 7: DOCUMENT STORE (persistence)")
        doc.status = "ready"
        doc.processing_time_ms = (time.perf_counter() - total_start) * 1000
        store.save(doc)

        loaded = store.get(doc.id)
        assert loaded is not None
        assert loaded.filename == doc.filename
        assert loaded.status == "ready"
        assert loaded.chunks_count == len(chunks)
        print(f"     ✅ Документ сохранён: {loaded.filename}, "
              f"{loaded.chunks_count} чанков, статус={loaded.status}")

        # Проверяем коллекции
        collections = indexer.list_collections()
        col = next(c for c in collections if c["name"] == "test-pipeline")
        print(f"     ✅ Коллекция 'test-pipeline': {col['chunks_count']} чанков")

        total_ms = (time.perf_counter() - total_start) * 1000
        print(f"\n  ⏱️  Общее время: {total_ms:.0f} мс")
        print(f"\n{'=' * 50}")
        print(f"  🎉 ПОЛНЫЙ ПАЙПЛАЙН РАБОТАЕТ!\n")
        print(f"  Итого:  {file_path.name} → {parse_result.pages_count} стр. → "
              f"{len(chunks)} чанков → ChromaDB → поиск работает")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)
        shutil.rmtree(texts_dir, ignore_errors=True)


if __name__ == "__main__":
    print("\n🧪 Тест 4: ПОЛНЫЙ ПАЙПЛАЙН\n" + "=" * 50)
    test_full_pipeline_parse_chunk_index_search()
