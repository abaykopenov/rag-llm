"""
RAG-LLM — Прозрачный RAG-as-a-service.

Точка входа FastAPI приложения.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.api.openai_compat import router as openai_router
from app.core.embedder import embedder
from app.core.llm_router import llm_router
from app.utils.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecyle: startup / shutdown."""
    log = get_logger("main")

    # Startup
    log.info("=== RAG-LLM {} запускается ===", settings.app_version)
    log.info("LLM: {} ({})", settings.llm_provider, settings.llm_base_url)
    log.info("Embedding: {} ({})", settings.embedding_provider, settings.embedding_base_url)
    log.info("Reranker: {} ({})", "ON" if settings.reranker_enabled else "OFF", settings.reranker_model)
    log.info("ChromaDB: {}", settings.chroma_persist_dir)
    log.info(
        "Security: CORS={}, API-key auth={}, max_upload={}MB",
        settings.cors_origins_list,
        "ON" if settings.api_keys_set else "OFF",
        settings.max_upload_size_mb,
    )

    # Создаём директории
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    # Запускаем gRPC сервер параллельно
    grpc_server = None
    try:
        from app.grpc_api.server import serve_grpc
        grpc_server = await serve_grpc()
        log.info("gRPC сервер запущен на порту 50051")
    except Exception as e:
        log.warning("gRPC сервер не запущен: {}", e)

    yield

    # Shutdown
    log.info("=== RAG-LLM останавливается ===")
    if grpc_server:
        await grpc_server.stop(5)
        log.info("gRPC сервер остановлен")
    await embedder.close()
    await llm_router.close()


# Настройка логирования
setup_logging(debug=settings.debug)

# Создание приложения
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Прозрачный RAG-as-a-service. "
        "Загружайте документы, задавайте вопросы, "
        "видьте каждый шаг пайплайна."
    ),
    lifespan=lifespan,
)

# CORS — список разрешённых origin'ов задаётся через RAG_CORS_ALLOW_ORIGINS.
# При allow_origins=["*"] credentials автоматически отключаются (требование спеки).
_cors_origins = settings.cors_origins_list
_allow_credentials = "*" not in _cors_origins
if "*" in _cors_origins:
    get_logger("main").warning(
        "CORS: origin='*' — credentials отключены. "
        "В production задай явный список через RAG_CORS_ALLOW_ORIGINS."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роуты.
# require_api_key уже встроен в router'ы через dependencies=[...]
app.include_router(router, prefix="/api")
app.include_router(openai_router)  # /v1/chat/completions


@app.get("/")
async def root():
    """Корневой endpoint (publicly accessible)."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": "/api",
        "openai_compat": "/v1/chat/completions",
    }


# ──────────────────────────────────────────────────────────────
# Unauthed liveness/readiness probes (для k8s/compose healthcheck).
# Не требуют X-API-Key — возвращают минимум информации.
# Для полного health-отчёта используй /api/health (требует ключ).
# ──────────────────────────────────────────────────────────────

@app.get("/health")
async def liveness():
    """Liveness probe: процесс жив, event loop отвечает."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness():
    """Readiness probe: можно ли отправлять трафик."""
    # Минимальная проверка: ChromaDB поднимается.
    try:
        from app.core.indexer import indexer
        indexer.list_collections()
        return {"status": "ready"}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"not ready: {e}")

