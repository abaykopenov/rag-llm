"""
Pydantic-схемы для API запросов и ответов.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# === Запросы ===

class UploadResponse(BaseModel):
    """Ответ на загрузку документа."""
    document_id: str
    filename: str
    collection: str
    pages_count: int
    chunks_count: int
    status: str
    processing_time_ms: float
    summary: Optional[str] = Field(None, description="Краткое содержание (если LLM доступен)")


class QueryRequest(BaseModel):
    """Запрос на вопрос по документам."""
    question: str = Field(..., description="Вопрос пользователя", min_length=1)
    collection: str = Field(default="default", description="Коллекция для поиска")
    top_k: Optional[int] = Field(default=None, description="Количество чанков (по умолчанию из конфига)")
    temperature: Optional[float] = Field(default=None, description="Температура LLM", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, description="Макс. токенов в ответе")
    # Metadata фильтры
    document_id: Optional[str] = Field(default=None, description="Фильтр по ID документа")
    element_type: Optional[str] = Field(default=None, description="Фильтр: text, table, code, list, formula")
    section: Optional[str] = Field(default=None, description="Фильтр по секции (подстрока)")
    # Hybrid search
    keywords: Optional[list[str]] = Field(default=None, description="Ключевые слова для hybrid search")


class ChunkInfo(BaseModel):
    """Информация о чанке в ответе."""
    id: str
    text: str
    score: float
    page: Optional[int] = None
    section: Optional[str] = None


class QueryResponse(BaseModel):
    """Ответ на вопрос — с полной прозрачностью."""
    answer: str = Field(..., description="Ответ LLM")
    chunks_used: list[ChunkInfo] = Field(default_factory=list, description="Использованные чанки")
    prompt: str = Field(default="", description="Полный prompt, отправленный в LLM")
    model: str = Field(default="", description="Какая модель ответила")
    tokens_used: int = Field(default=0, description="Всего токенов")
    timing: dict = Field(default_factory=dict, description="Тайминги каждого этапа")


class CollectionInfo(BaseModel):
    """Информация о коллекции."""
    name: str
    chunks_count: int


class HealthResponse(BaseModel):
    """Health check ответ."""
    status: str
    version: str
    llm_provider: str
    embedding_provider: str
    collections: list[CollectionInfo] = []


class SystemStatsResponse(BaseModel):
    """Метрики системы."""
    hostname: str
    os: str
    cpu: dict
    memory: dict
    disk: dict
    gpu: dict


# === Stage 2: Visibility ===

class DocumentTextResponse(BaseModel):
    """Извлечённый текст документа."""
    document_id: str
    filename: str
    pages_count: int
    text_length: int = Field(..., description="Количество символов")
    text: str = Field(..., description="Полный извлечённый текст (markdown)")


class DocumentChunkDetail(BaseModel):
    """Детальная информация о чанке документа."""
    id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    element_type: Optional[str] = None
    char_count: int = 0


class DocumentChunksResponse(BaseModel):
    """Чанки документа с метаданными."""
    document_id: str
    filename: str
    collection: str
    chunks_count: int
    avg_chunk_length: int = Field(0, description="Средняя длина чанка в символах")
    chunks: list[DocumentChunkDetail]


class ChunkDetailResponse(BaseModel):
    """Детали одного чанка (включая embedding вектор)."""
    id: str
    text: str
    document_id: Optional[str] = None
    collection: str
    page: Optional[int] = None
    section: Optional[str] = None
    element_type: Optional[str] = None
    char_count: int = 0
    embedding: Optional[list[float]] = Field(None, description="Embedding вектор (если запрошен)")
    embedding_dimensions: Optional[int] = Field(None, description="Размерность embedding вектора")


# === Evaluation ===

class EvalSampleRequest(BaseModel):
    """Один пример для оценки RAG."""
    question: str = Field(..., description="Вопрос")
    ground_truth: Optional[str] = Field(None, description="Эталонный ответ (если есть)")


class EvalRequest(BaseModel):
    """Запрос на оценку качества RAG."""
    samples: list[EvalSampleRequest] = Field(..., description="Примеры для оценки")
    collection: str = Field(default="default", description="Коллекция для поиска")
    top_k: int = Field(default=5, description="Количество чанков для поиска")


class EvalMetricResult(BaseModel):
    """Метрики одного примера."""
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall: float
    answer_preview: str = Field("", description="Первые 200 символов ответа")


class EvalResponse(BaseModel):
    """Ответ с результатами оценки."""
    samples_count: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_overall: float
    grade: str = Field("", description="Буквенная оценка: A/B/C/D/F")
    results: list[EvalMetricResult]
    eval_time_ms: float
