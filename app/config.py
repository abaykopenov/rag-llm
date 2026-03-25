"""
RAG-LLM Configuration
Все настройки приложения в одном месте.
Можно переопределить через переменные окружения или .env файл.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Главный конфиг приложения."""

    # === Приложение ===
    app_name: str = "RAG-LLM"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    grpc_port: int = Field(default=50051, description="Порт gRPC сервера")
    debug: bool = False

    # === LLM Провайдер ===
    # Бесплатные варианты:
    #   Gemini:  base_url = https://generativelanguage.googleapis.com/v1beta/openai
    #            model = gemini-2.0-flash    (бесплатно, 15 RPM / 1M TPD)
    #   Groq:    base_url = https://api.groq.com/openai/v1
    #            model = llama-3.3-70b-versatile  (бесплатно, 30 RPM)
    #   OpenRouter: base_url = https://openrouter.ai/api/v1
    #            model = google/gemini-2.0-flash-exp:free
    llm_provider: str = Field(
        default="gemini",
        description="Провайдер LLM: gemini, groq, openrouter, vllm, openai, custom"
    )
    llm_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        description="Base URL для LLM API (OpenAI-compatible)"
    )
    llm_api_key: str = Field(
        default="",
        description="API ключ (Gemini: https://aistudio.google.com/apikey)"
    )
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Модель для генерации"
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Максимум токенов в ответе"
    )
    llm_temperature: float = Field(
        default=0.7,
        description="Температура генерации (0.0 - 1.0)"
    )

    # === Embedding Провайдер ===
    # Бесплатные варианты:
    #   Gemini:  base_url = https://generativelanguage.googleapis.com/v1beta/openai
    #            model = text-embedding-004   (бесплатно)
    #   Jina:    base_url = https://api.jina.ai/v1
    #            model = jina-embeddings-v3   (1M tokens free)
    embedding_provider: str = Field(
        default="gemini",
        description="Провайдер Embeddings: gemini, jina, vllm, openai, custom"
    )
    embedding_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        description="Base URL для Embedding API"
    )
    embedding_api_key: str = Field(
        default="",
        description="API ключ для Embedding (может совпадать с LLM)"
    )
    embedding_model: str = Field(
        default="text-embedding-004",
        description="Название embedding модели"
    )
    embedding_dimensions: int = Field(
        default=768,
        description="Размерность embedding вектора"
    )

    # === ChromaDB ===
    chroma_host: Optional[str] = Field(
        default=None,
        description="ChromaDB host (None = встроенный режим)"
    )
    chroma_port: int = Field(
        default=8400,
        description="ChromaDB port (для client-server режима)"
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="Директория для хранения данных ChromaDB"
    )

    # === Парсинг и Чанкинг ===
    parser_mode: str = Field(
        default="gemini",
        description="Режим парсинга: gemini (облако), docling (локально), vision (Vision LLM)"
    )
    vision_llm_base_url: str = Field(
        default="",
        description="Base URL для Vision LLM (для parser_mode=vision)"
    )
    vision_llm_model: str = Field(
        default="Qwen2-VL-72B-Instruct",
        description="Модель Vision LLM"
    )
    upload_dir: str = Field(
        default="./data/uploads",
        description="Директория для загруженных файлов"
    )
    texts_dir: str = Field(
        default="./data/texts",
        description="Директория для хранения извлечённых текстов"
    )
    chunk_max_tokens: int = Field(
        default=512,
        description="Максимальный размер чанка в токенах"
    )
    chunk_overlap_tokens: int = Field(
        default=64,
        description="Перекрытие чанков в токенах (0 = без overlap)"
    )

    # === Retrieval ===
    retrieval_top_k: int = Field(
        default=5,
        description="Количество чанков для поиска"
    )
    retrieval_score_threshold: float = Field(
        default=0.0,
        description="Минимальный score для включения чанка (0.0 = без фильтра)"
    )
    query_rewrite_enabled: bool = Field(
        default=True,
        description="Включить переформулирование запросов через LLM"
    )
    multi_query_enabled: bool = Field(
        default=False,
        description="Включить Multi-Query Retrieval (3 варианта запроса)"
    )

    # === Reranker ===
    # Jina Reranker: бесплатно (1M tokens/month)
    #   https://jina.ai/reranker/  → получить API key
    reranker_enabled: bool = Field(
        default=True,
        description="Включить reranking через cross-encoder"
    )
    reranker_base_url: str = Field(
        default="https://api.jina.ai/v1",
        description="Base URL для Reranker API (Jina бесплатный)"
    )
    reranker_api_key: str = Field(
        default="",
        description="API ключ (Jina: https://jina.ai/reranker/)"
    )
    reranker_model: str = Field(
        default="jina-reranker-v2-base-multilingual",
        description="Модель для reranking (Jina multilingual)"
    )
    reranker_top_n: int = Field(
        default=5,
        description="Количество чанков после reranking"
    )

    # === Parent-Child Chunks ===
    parent_child_enabled: bool = Field(
        default=True,
        description="Включить двухуровневые чанки (parent-child)"
    )
    parent_max_tokens: int = Field(
        default=1024,
        description="Максимальный размер parent чанка в токенах"
    )
    child_max_tokens: int = Field(
        default=256,
        description="Максимальный размер child чанка в токенах"
    )

    # === Prompt ===
    system_prompt: str = Field(
        default=(
            "Ты — полезный AI-ассистент. Отвечай на вопросы ТОЛЬКО на основе предоставленного контекста. "
            "Если в контексте нет ответа, скажи об этом честно. "
            "В конце ответа укажи источники в формате: 📎 Источники: [1] название файла, стр. N"
        ),
        description="Системный промпт для LLM"
    )

    model_config = {
        "env_prefix": "RAG_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Глобальный экземпляр настроек
settings = Settings()
