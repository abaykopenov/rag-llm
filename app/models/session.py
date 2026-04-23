"""
Session — диалоговая сессия для чат-эндпоинтов (/api/chat, /api/sessions).

Модель была референсирована в app.utils.session_store и app.api.routes, но
исходный файл отсутствовал в репозитории — это исправление.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Одно сообщение в диалоге."""

    role: str                       # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    chunks_used: int = 0            # сколько чанков использовал ассистент
    model: str = ""
    tokens_used: int = 0

    def to_info_dict(self) -> dict:
        """Сериализация для SessionDetailResponse (ChatMessageInfo)."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "chunks_used": self.chunks_used,
            "model": self.model,
            "tokens_used": self.tokens_used,
        }


class Session(BaseModel):
    """Сессия диалога с историей сообщений."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    collection: str = "default"
    title: str = "Новая сессия"
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def add_message(
        self,
        role: str,
        content: str,
        *,
        chunks_used: int = 0,
        model: str = "",
        tokens_used: int = 0,
    ) -> ChatMessage:
        """Добавить сообщение в историю. Обновляет title по первому user-сообщению."""
        msg = ChatMessage(
            role=role,
            content=content,
            chunks_used=chunks_used,
            model=model,
            tokens_used=tokens_used,
        )
        self.messages.append(msg)
        self.updated_at = msg.timestamp

        # Ставим title из первого user-сообщения (первые ~60 символов)
        if role == "user" and self.title == "Новая сессия":
            self.title = content[:60].strip() or "Новая сессия"

        return msg

    def get_history(self, max_pairs: int = 5) -> list[dict]:
        """Последние N пар (user → assistant) в формате OpenAI messages.

        Возвращает список {"role": ..., "content": ...} для подстановки в
        generator.build_prompt(history=...).
        """
        if not self.messages:
            return []

        # Берём хвост истории ограниченного размера (2 * max_pairs сообщений)
        tail = self.messages[-(2 * max_pairs):]
        return [{"role": m.role, "content": m.content} for m in tail]
