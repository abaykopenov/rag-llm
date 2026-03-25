"""
SessionStore — хранилище сессий диалога.

Аналогично DocumentStore: in-memory + JSON persistence.
"""

import json
from pathlib import Path
from typing import Optional

from app.models.session import Session
from app.utils.logging import get_logger

log = get_logger("session_store")

SESSIONS_FILE = Path("data/sessions.json")


class SessionStore:
    """Хранилище сессий с автосохранением в JSON."""

    def __init__(self, filepath: Path = SESSIONS_FILE):
        self._filepath = filepath
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self):
        """Загрузить сессии из файла."""
        if self._filepath.exists():
            try:
                data = json.loads(self._filepath.read_text(encoding="utf-8"))
                for item in data:
                    session = Session.model_validate(item)
                    self._sessions[session.id] = session
                log.info("Загружено {} сессий из {}", len(self._sessions), self._filepath)
            except Exception as e:
                log.warning("Ошибка загрузки сессий: {}", e)

    def _save(self):
        """Сохранить все сессии в файл."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [s.model_dump(mode="json") for s in self._sessions.values()]
        self._filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def create(self, collection: str = "default") -> Session:
        """Создать новую сессию."""
        session = Session(collection=collection)
        self._sessions[session.id] = session
        self._save()
        log.info("Сессия создана: {}", session.id)
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Получить сессию по ID."""
        return self._sessions.get(session_id)

    def save(self, session: Session):
        """Сохранить/обновить сессию."""
        self._sessions[session.id] = session
        self._save()

    def delete(self, session_id: str) -> bool:
        """Удалить сессию."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save()
            log.info("Сессия удалена: {}", session_id)
            return True
        return False

    def get_all(self) -> list[Session]:
        """Все сессии, отсортированные по дате обновления."""
        return sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )

    def count(self) -> int:
        return len(self._sessions)


# Глобальный экземпляр
session_store = SessionStore()
