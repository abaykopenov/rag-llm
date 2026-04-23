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

    # === Логирование ===
    log_format: str = Field(
        default="text",
        description=(
            "Формат логов: 'text' (цветной, человекочитаемый) или 'json' "
            "(структурный, под Loki/ELK/Datadog)."
        ),
    )
    log_level: str = Field(
        default="INFO",
        description="Уровень: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_file_enabled: bool = Field(
        default=True,
        description="Писать ли логи в файл (./data/logs/rag-llm_<date>.log)",
    )
    log_file_dir: str = Field(
        default="./data/logs",
        description="Директория для файловых логов",
    )
    log_file_rotation: str = Field(
        default="10 MB",
        description="Ротация логов по размеру (формат loguru: '10 MB', '1 day')",
    )
    log_file_retention: str = Field(
        default="7 days",
        description="Сколько хранить ротированные файлы (формат loguru)",
    )

    # === Безопасность ===
    cors_allow_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description=(
            "Разрешённые CORS origins (через запятую). "
            "Используй '*' только для локальной разработки."
        ),
    )
    api_keys: str = Field(
        default="",
        description=(
            "Валидные ключи для header X-API-Key (через запятую). "
            "Пусто = аутентификация отключена (dev mode)."
        ),
    )
    max_upload_size_mb: int = Field(
        default=50,
        description="Максимальный размер загружаемого файла в МБ (0 = без лимита)",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

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
    llm_anthropic_cache_control: bool = Field(
        default=False,
        description=(
            "Размечать cache_control: ephemeral на системном промпте и блоке "
            "контекста в messages. Нужно ТОЛЬКО для Anthropic API (Claude). "
            "Для OpenAI/Gemini/vLLM кеш срабатывает автоматически — флаг "
            "оставь False."
        ),
    )
    llm_cache_min_tokens: int = Field(
        default=1024,
        description=(
            "Минимальная оценочная длина блока для маркировки cache_control "
            "(Anthropic требует ≥1024 токенов). ~4 символа на токен."
        ),
    )

    # === Embedding Провайдер ===
    # Основной способ конфигурации — профили в config/embedding_profiles.yml.
    # Выбор активного профиля — RAG_EMBEDDING_PROFILE=<имя>.
    # Плоские поля ниже (embedding_provider, embedding_base_url, ...) сохранены
    # для обратной совместимости: если RAG_EMBEDDING_PROFILE не задан и в YAML
    # нет профиля 'default', профиль будет собран из них.
    embedding_profile: str = Field(
        default="",
        description=(
            "Имя профиля из embedding_profiles.yml "
            "(например: gemini-free, vllm-bge, ollama-nomic, openai-large, jina-v3). "
            "Пусто = использовать профиль 'default' из YAML или fallback на плоские поля."
        ),
    )
    embedding_profiles_path: str = Field(
        default="config/embedding_profiles.yml",
        description="Путь к YAML-файлу с embedding-профилями",
    )

    # --- Legacy плоские поля (backward-compat) ---
    embedding_provider: str = Field(
        default="gemini",
        description="[legacy] Провайдер Embeddings: gemini, jina, vllm, openai, custom"
    )
    embedding_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        description="[legacy] Base URL для Embedding API"
    )
    embedding_api_key: str = Field(
        default="",
        description="[legacy] API ключ для Embedding (может совпадать с LLM)"
    )
    embedding_model: str = Field(
        default="text-embedding-004",
        description="[legacy] Название embedding модели"
    )
    embedding_dimensions: int = Field(
        default=768,
        description="[legacy] Размерность embedding вектора"
    )

    # === Tracing ===
    trace_db_path: str = Field(
        default="./data/traces.sqlite",
        description=(
            "Путь к SQLite-файлу с трейсами запросов. "
            "Ранее трейсы жили только в памяти (deque maxlen=100) и терялись "
            "при рестарте — теперь персистентны."
        ),
    )
    trace_memory_cache_size: int = Field(
        default=200,
        description=(
            "Сколько последних трейсов держать в памяти (LRU) для быстрого "
            "get_trace без обращения к SQLite"
        ),
    )
    trace_retention_days: int = Field(
        default=30,
        description=(
            "Сколько дней хранить завершённые трейсы в SQLite. 0 = вечно. "
            "Чистка запускается lazy раз в сутки при end_trace()."
        ),
    )

    # === BM25 keyword search ===
    bm25_enabled: bool = Field(
        default=True,
        description=(
            "Использовать BM25 для keyword-поиска. False = старый substring-match "
            "(оставлен как fallback / для сравнения качества)"
        ),
    )
    bm25_persist_dir: str = Field(
        default="./data/bm25",
        description="Директория для BM25-индексов (per collection)",
    )
    bm25_stemmer_lang: str = Field(
        default="russian",
        description=(
            "Язык стеммера для BM25 (snowballstemmer): russian, english, german, "
            "french, spanish, italian и др. Для RU+EN корпусов 'russian' — OK, "
            "английские слова останутся без стемминга, но будут матчиться точно."
        ),
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

    # --- Docling PDF pipeline ---
    docling_do_ocr: bool = Field(
        default=True,
        description="Включить OCR (для сканов). Требует EasyOCR — Docling скачает модели.",
    )
    docling_ocr_lang: str = Field(
        default="en,ru",
        description="Языки OCR через запятую (коды EasyOCR): en,ru,de,fr,...",
    )
    docling_do_table_structure: bool = Field(
        default=True,
        description="Включить TableFormer — восстановление структуры таблиц",
    )
    docling_table_mode: str = Field(
        default="accurate",
        description="Режим TableFormer: accurate (качество) или fast (скорость)",
    )
    docling_device: str = Field(
        default="auto",
        description="Устройство ускорения: auto, cpu, cuda, mps",
    )
    docling_num_threads: int = Field(
        default=4,
        description="Количество CPU-потоков для Docling",
    )
    docling_timeout_sec: float = Field(
        default=300.0,
        description="Максимум секунд на один документ (Docling абортит длинный парсинг)",
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
    chunk_output_format: str = Field(
        default="hybrid",
        description=(
            "Формат чанков: 'hybrid' (Docling HybridChunker + tokenizer активного "
            "embedding-провайдера, рекомендуется) или 'markdown' (legacy regex)"
        ),
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
