"""
Тест 1: Модели данных и document_store.
Проверяет Pydantic модели и персистентное хранилище — не требует внешних сервисов.
"""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

import pytest


def test_document_model_creation():
    """Проверить создание Document с дефолтными значениями."""
    from app.models.document import Document

    doc = Document(filename="test.pdf", file_size=1024, collection="default")

    assert doc.filename == "test.pdf"
    assert doc.file_size == 1024
    assert doc.collection == "default"
    assert doc.status == "pending"
    assert doc.id  # UUID должен быть сгенерирован
    assert len(doc.id) > 0
    assert doc.created_at is not None
    # Проверяем timezone-aware
    assert doc.created_at.tzinfo is not None
    print(f"  ✅ Document создан: id={doc.id[:8]}..., created_at={doc.created_at}")


def test_chunk_model_creation():
    """Проверить создание Chunk с метаданными."""
    from app.models.document import Chunk, ChunkMetadata

    meta = ChunkMetadata(page=5, section="Глава 1", char_count=512)
    chunk = Chunk(document_id="test-doc-123", text="Пример текста чанка", metadata=meta)

    assert chunk.document_id == "test-doc-123"
    assert chunk.text == "Пример текста чанка"
    assert chunk.metadata.page == 5
    assert chunk.metadata.section == "Глава 1"
    assert chunk.metadata.char_count == 512
    assert chunk.id  # UUID
    print(f"  ✅ Chunk создан: id={chunk.id[:8]}..., page={meta.page}")


def test_document_store_save_and_get():
    """Проверить сохранение и загрузку из DocumentStore."""
    from app.models.document import Document
    from app.utils.document_store import DocumentStore

    # Используем временную директорию
    tmp_dir = tempfile.mkdtemp()
    try:
        store = DocumentStore(storage_dir=tmp_dir)

        doc = Document(
            filename="report.pdf",
            file_size=2048,
            collection="test-col",
            status="ready",
            pages_count=10,
            chunks_count=42,
        )

        # Сохраняем
        store.save(doc)

        # Получаем
        loaded = store.get(doc.id)
        assert loaded is not None
        assert loaded.filename == "report.pdf"
        assert loaded.collection == "test-col"
        assert loaded.pages_count == 10
        assert loaded.chunks_count == 42
        print(f"  ✅ Save + Get: документ '{loaded.filename}' сохранён и загружен")

        # Проверяем файл на диске
        json_path = Path(tmp_dir) / "documents.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert doc.id in data
        print(f"  ✅ JSON файл создан: {json_path.stat().st_size} байт")

        # Проверяем перезагрузку (симуляция рестарта)
        store2 = DocumentStore(storage_dir=tmp_dir)
        reloaded = store2.get(doc.id)
        assert reloaded is not None
        assert reloaded.filename == "report.pdf"
        print(f"  ✅ Persistence: данные сохранились после 'перезапуска'")

        # Удаление
        assert store2.delete(doc.id) is True
        assert store2.get(doc.id) is None
        print(f"  ✅ Delete: документ удалён")

    finally:
        shutil.rmtree(tmp_dir)


def test_document_store_collection_operations():
    """Проверить операции с коллекциями."""
    from app.models.document import Document
    from app.utils.document_store import DocumentStore

    tmp_dir = tempfile.mkdtemp()
    try:
        store = DocumentStore(storage_dir=tmp_dir)

        # Добавляем документы в разные коллекции
        for i in range(3):
            store.save(Document(filename=f"doc_{i}.pdf", file_size=100, collection="colA"))
        for i in range(2):
            store.save(Document(filename=f"doc_{i}.pdf", file_size=100, collection="colB"))

        assert store.count() == 5
        print(f"  ✅ Добавлено 5 документов в 2 коллекции")

        # Удаляем коллекцию
        removed = store.delete_by_collection("colA")
        assert removed == 3
        assert store.count() == 2
        print(f"  ✅ delete_by_collection('colA'): удалено {removed}, осталось {store.count()}")

        # Проверяем, что colB осталась
        remaining = store.get_all()
        assert all(d.collection == "colB" for d in remaining)
        print(f"  ✅ Оставшиеся документы: все из colB")

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    print("\n🧪 Тест 1: Модели данных и DocumentStore\n" + "=" * 50)
    test_document_model_creation()
    test_chunk_model_creation()
    test_document_store_save_and_get()
    test_document_store_collection_operations()
    print("\n✅ Все тесты пройдены!\n")
