"""
Модели данных: Document, Chunk, и связанные структуры.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
import uuid


class ChunkMetadata(BaseModel):
    """Метаданные чанка — откуда он взялся."""
    page: Optional[int] = Field(None, description="Номер страницы")
    section: Optional[str] = Field(None, description="Заголовок раздела")
    element_type: Optional[str] = Field(None, description="Тип элемента (text, table, list)")
    char_count: int = Field(0, description="Количество символов")
    # Parent-child
    parent_id: Optional[str] = Field(None, description="ID родительского чанка (для child)")
    chunk_type: str = Field("chunk", description="Тип: 'parent', 'child', или 'chunk' (обычный)")


class Chunk(BaseModel):
    """Один чанк документа."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID чанка")
    document_id: str = Field(..., description="ID документа-источника")
    text: str = Field(..., description="Текст чанка")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: Optional[list[float]] = Field(None, description="Embedding вектор", exclude=True)


class Document(BaseModel):
    """Загруженный документ."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID документа")
    filename: str = Field(..., description="Имя файла")
    file_size: int = Field(0, description="Размер файла в байтах")
    collection: str = Field("default", description="Название коллекции")
    pages_count: int = Field(0, description="Количество страниц")
    chunks_count: int = Field(0, description="Количество чанков")
    raw_text: Optional[str] = Field(None, description="Полный извлечённый текст")
    summary: Optional[str] = Field(None, description="Краткое содержание (auto-generated)")
    status: str = Field("pending", description="Статус: pending, processing, ready, error")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: Optional[float] = Field(None, description="Время обработки в мс")


class RetrievedChunk(BaseModel):
    """Чанк, найденный при retrieval."""
    id: str
    text: str
    score: float = Field(..., description="Релевантность (cosine similarity)")
    metadata: ChunkMetadata
    # Reranking
    original_score: Optional[float] = Field(None, description="Исходный cosine score (до reranking)")
    rerank_score: Optional[float] = Field(None, description="Score от cross-encoder reranker")

