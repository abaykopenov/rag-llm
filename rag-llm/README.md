# 🔍 RAG-LLM

**Прозрачный RAG-as-a-service** — загружайте документы, задавайте вопросы, видьте каждый шаг пайплайна.

## ✨ Возможности

- 📄 **Парсинг документов** — PDF, DOCX, PPTX, HTML, Markdown (через Docling)
- 🧩 **Умный чанкинг** — Parent-Child chunks для точного поиска и богатого контекста
- 🔍 **Hybrid Search** — cosine similarity + keyword search (RRF merge)
- 🎯 **Reranker** — Jina cross-encoder переранжирование (включён по умолчанию)
- 🤖 **Multi-provider LLM** — Gemini, Groq, OpenRouter, vLLM, OpenAI
- 📊 **RAGAS Evaluation** — 4 метрики качества (Faithfulness, Answer Relevancy, Context Precision/Recall)
- 📝 **Auto-Summary** — автоматическая суммаризация при загрузке
- 🔗 **Dual API** — REST (FastAPI) + gRPC (с streaming и reflection)
- 🔎 **Трассировка** — trace_id для каждого запроса, пошаговые тайминги, системные метрики
- 🆓 **Бесплатные API** — работает из коробки с Gemini + Jina (бесплатные тарифы)

## 🚀 Быстрый старт

### 1. Установка

```bash
cd rag-llm
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Настройка API ключей

```bash
cp .env.example .env
```

Заполни API ключи в `.env`:
- **Gemini** (бесплатно): https://aistudio.google.com/apikey
- **Jina Reranker** (бесплатно): https://jina.ai/reranker/

### 3. Запуск

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Стартуют оба сервера:
- **REST API** → http://localhost:8000/docs (Swagger UI)
- **gRPC API** → `localhost:50051` (reflection enabled)

## 📡 API

### REST — загрузка документа
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@book.pdf" \
  -F "collection=my_books"
```

### REST — вопрос
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Что такое RAG?", "collection": "my_books"}'
```

### REST — трассировка
```bash
# Последние запросы с таймингами
curl http://localhost:8000/api/traces

# Детали конкретного запроса
curl http://localhost:8000/api/traces/tr_abc123

# Системные метрики
curl http://localhost:8000/api/system/stats
```

### gRPC
```bash
# Health check
grpcurl -plaintext localhost:50051 rag.RAGService/Health

# Query
grpcurl -plaintext -d '{"question": "Что такое RAG?"}' \
  localhost:50051 rag.RAGService/Query

# Streaming
grpcurl -plaintext -d '{"question": "Расскажи подробно"}' \
  localhost:50051 rag.RAGService/QueryStream
```

### Ответ включает

```json
{
  "answer": "RAG — это метод...",
  "chunks_used": [
    {"id": "chunk_042", "text": "...", "score": 0.92, "page": 15, "section": "Глава 3"}
  ],
  "prompt": "Используя следующий контекст...",
  "model": "gemini-2.0-flash",
  "timing": {
    "retrieval_ms": 23.1,
    "generation_ms": 2100.5,
    "total_ms": 2123.6,
    "trace_id": "tr_a3f8b2c1d4e5"
  }
}
```

## 🔌 Поддерживаемые провайдеры

### LLM
| Провайдер | Модель | Бесплатно |
|---|---|---|
| **Google Gemini** | `gemini-2.0-flash` | ✅ 15 RPM |
| **Groq** | `llama-3.3-70b-versatile` | ✅ 30 RPM |
| **OpenRouter** | `gemini-2.0-flash-exp:free` | ✅ |
| **vLLM** (локально) | Любая модель | ∞ |
| **OpenAI** | `gpt-4o-mini` | 💰 |

### Embeddings
| Провайдер | Модель | Размерность |
|---|---|---|
| **Google Gemini** | `text-embedding-004` | 768 |
| **Jina** | `jina-embeddings-v3` | 1024 |
| **vLLM** | `BAAI/bge-m3` | 1024 |

### Reranker
| Провайдер | Модель |
|---|---|
| **Jina** | `jina-reranker-v2-base-multilingual` |
| **vLLM** | Любой cross-encoder |

## 📁 Структура проекта

```
rag-llm/
├── app/
│   ├── main.py              # FastAPI + gRPC startup
│   ├── config.py            # Конфигурация (Pydantic Settings)
│   ├── api/
│   │   ├── routes.py        # REST endpoints (upload, query, traces, evaluate...)
│   │   └── schemas.py       # Pydantic request/response модели
│   ├── core/
│   │   ├── parser.py        # Docling парсинг документов
│   │   ├── chunker.py       # Parent-Child чанкинг
│   │   ├── embedder.py      # Embedding клиент (batch + retry)
│   │   ├── indexer.py       # ChromaDB индексация + keyword search
│   │   ├── retriever.py     # Hybrid search + metadata фильтры + reranking
│   │   ├── reranker.py      # Cross-encoder reranker (Jina API + fallback)
│   │   ├── llm_router.py    # Multi-provider LLM клиент
│   │   ├── generator.py     # Prompt assembly + LLM generation
│   │   └── summarizer.py    # Auto-summary
│   ├── grpc_api/
│   │   ├── server.py        # gRPC server (reflection, async)
│   │   ├── service.py       # 7 gRPC methods implementation
│   │   ├── rag_pb2.py       # Generated protobuf classes
│   │   └── rag_pb2_grpc.py  # Generated gRPC stubs
│   ├── evaluation/
│   │   └── evaluator.py     # RAGAS-style evaluation (LLM-as-judge)
│   ├── models/
│   │   └── document.py      # Document, Chunk, ChunkMetadata
│   └── utils/
│       ├── logging.py       # Loguru structured logging
│       ├── monitoring.py    # CPU/RAM/GPU метрики
│       ├── tracer.py        # Request tracing (trace_id, steps, timings)
│       └── document_store.py# Persistent document metadata (JSON)
├── proto/
│   └── rag.proto            # gRPC service definition
├── tests/
│   └── run_all.py           # 17 integration tests
├── docs/
│   ├── 01_PROJECT_DESCRIPTION.md
│   ├── 02_ROADMAP.md
│   └── 03_TECH_STACK.md
├── .env.example             # Template with all providers documented
├── requirements.txt
└── pyproject.toml
```

## 📊 API Endpoints

### REST API (`:8000`)
| Method | Endpoint | Описание |
|---|---|---|
| POST | `/api/upload` | Загрузка документа |
| POST | `/api/query` | Вопрос по документам |
| POST | `/api/evaluate` | RAGAS оценка качества |
| GET | `/api/documents/{id}/text` | Извлечённый текст |
| GET | `/api/documents/{id}/chunks` | Чанки документа |
| GET | `/api/documents/{id}/summary` | Краткое содержание |
| GET | `/api/chunks/{id}` | Детали чанка + embedding |
| GET | `/api/collections` | Список коллекций |
| GET | `/api/traces` | История запросов |
| GET | `/api/traces/{id}` | Детальная трассировка |
| GET | `/api/system/stats` | Системные метрики |
| GET | `/api/health` | Health check |

### gRPC API (`:50051`)
| Method | Type | Описание |
|---|---|---|
| `Upload` | Unary | Загрузка документа |
| `Query` | Unary | Вопрос по документам |
| `QueryStream` | Server streaming | Потоковый ответ |
| `Evaluate` | Unary | RAGAS оценка |
| `GetSummary` | Unary | Краткое содержание |
| `ListCollections` | Unary | Список коллекций |
| `Health` | Unary | Health check |

## 📚 Документация

- [Описание проекта](docs/01_PROJECT_DESCRIPTION.md)
- [Дорожная карта](docs/02_ROADMAP.md)
- [Технологический стек](docs/03_TECH_STACK.md)

## 📜 Лицензия

MIT
