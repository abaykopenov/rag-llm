"""
Персистентное хранилище документов на основе JSON-файла.
Заменяет in-memory dict, данные сохраняются между перезапусками.
"""

import json
import threading
from pathlib import Path
from typing import Optional

from app.models.document import Document
from app.utils.logging import get_logger

log = get_logger("store")


class DocumentStore:
    """Хранилище метаданных документов в JSON-файле.

    Потокобезопасное, сохраняет данные на диск после каждого изменения.
    Используется для метаданных документов (имя, статус, время обработки).
    Сами чанки и embeddings хранятся в ChromaDB.
    """

    def __init__(self, storage_dir: str = "./data"):
        self._storage_path = Path(storage_dir) / "documents.json"
        self._documents: dict[str, Document] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Загрузить документы из файла."""
        if self._storage_path.exists():
            try:
                raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
                for doc_id, doc_data in raw.items():
                    self._documents[doc_id] = Document.model_validate(doc_data)
                log.info("Загружено {} документов из {}", len(self._documents), self._storage_path)
            except Exception as e:
                log.warning("Не удалось загрузить документы из {}: {}", self._storage_path, e)
                self._documents = {}
        else:
            log.info("Файл хранилища не найден, начинаем с пустого: {}", self._storage_path)

    def _save(self):
        """Сохранить документы на диск."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for doc_id, doc in self._documents.items():
            # Исключаем raw_text — он может быть огромным
            data[doc_id] = doc.model_dump(mode="json", exclude={"raw_text"})
        self._storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save(self, doc: Document) -> None:
        """Сохранить или обновить документ."""
        with self._lock:
            self._documents[doc.id] = doc
            self._save()

    def get(self, doc_id: str) -> Optional[Document]:
        """Получить документ по ID."""
        return self._documents.get(doc_id)

    def get_all(self) -> list[Document]:
        """Получить все документы."""
        return list(self._documents.values())

    def delete(self, doc_id: str) -> bool:
        """Удалить документ по ID."""
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self._save()
                return True
            return False

    def delete_by_collection(self, collection: str) -> int:
        """Удалить все документы коллекции. Возвращает количество удалённых."""
        with self._lock:
            to_remove = [
                doc_id for doc_id, doc in self._documents.items()
                if doc.collection == collection
            ]
            for doc_id in to_remove:
                del self._documents[doc_id]
            if to_remove:
                self._save()
            return len(to_remove)

    def count(self) -> int:
        """Количество документов."""
        return len(self._documents)


# Глобальный экземпляр
document_store = DocumentStore()
