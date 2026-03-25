# 🔍 RAG-LLM

**Universal RAG-as-a-Service** — загружайте документы, задавайте вопросы, интегрируйте в любой проект.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Возможности

| Категория | Функция |
|---|---|
| 📄 **Парсинг** | PDF, DOCX, PPTX, HTML (Docling / Gemini / Vision LLM) |
| 🔍 **Поиск** | Hybrid search + Jina reranker + query rewriting + multi-query |
| 🧩 **Чанкинг** | Parent-child chunks + 15% overlap |
| 💬 **Чат** | Conversational RAG с памятью сессий |
| 📡 **Streaming** | SSE real-time ответы |
| 🔌 **API** | REST + gRPC + OpenAI-compatible (`/v1/chat/completions`) |
| 🏗️ **Деплой** | Docker Compose для GPU серверов (vLLM) |
| 🤖 **Модели** | Groq, Gemini, OpenRouter, vLLM, OpenAI — переключение через `.env` |
| 📊 **Оценка** | RAGAS-style метрики (LLM-as-judge) |
| 🔎 **Трассировка** | trace_id, пошаговые тайминги, системные метрики |

## 🚀 Быстрый старт

```bash
git clone https://github.com/abaykopenov/rag-llm.git
cd rag-llm
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
cp .env.example .env           # Заполни API ключи
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Откройте:
- **Swagger UI** → http://localhost:8000/docs
- **gRPC** → `localhost:50051` (reflection enabled)

## 📡 API Endpoints

### REST API (`:8000`)

| Method | Endpoint | Описание |
|---|---|---|
| POST | `/api/upload` | Загрузка документа |
| POST | `/api/query` | Вопрос по документам |
| POST | `/api/chat` | Чат с памятью сессий |
| POST | `/api/chat/stream` | SSE streaming чат |
| POST | `/api/evaluate` | RAGAS оценка качества |
| GET | `/api/sessions` | Список сессий |
| GET | `/api/documents/{id}/text` | Извлечённый текст |
| GET | `/api/collections` | Список коллекций |
| GET | `/api/traces` | История запросов |
| GET | `/api/system/stats` | Системные метрики |

### OpenAI-Compatible API

Любой проект с поддержкой OpenAI SDK может подключиться:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")
response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Что такое RAG?"}],
)
print(response.choices[0].message.content)
```

Заголовки для управления RAG:
- `X-RAG-Collection: my_docs` — выбор коллекции
- `X-RAG-Top-K: 10` — количество чанков
- `X-RAG-Enabled: false` — отключить RAG

### gRPC API (`:50051`)

```bash
grpcurl -plaintext localhost:50051 rag.RAGService/Health
grpcurl -plaintext -d '{"question": "Что такое RAG?"}' localhost:50051 rag.RAGService/Query
```

## 🔌 Поддерживаемые провайдеры

| Компонент | Провайдеры |
|---|---|
| **LLM** | Gemini, Groq (бесплатно), OpenRouter, vLLM, OpenAI |
| **Embeddings** | Gemini (бесплатно), Jina, vLLM, Ollama |
| **Reranker** | Jina (бесплатно), vLLM |
| **Парсинг** | Gemini API, Docling (локально), Vision LLM |
| **Vector DB** | ChromaDB (встроенный или client-server) |

## 🏗️ Локальный деплой (GPU сервер)

Для полностью локальной работы без облачных API:

```bash
# На GPU сервере (120+ ГБ VRAM):
cd deploy/
docker compose up -d                        # LLM (Qwen2.5-72B)
docker compose --profile vision up -d       # + Vision (Qwen2-VL-72B)

# На рабочей машине:
cp .env.local .env
# Заменить GX10_IP на IP сервера
```

## 📁 Структура проекта

```
rag-llm/
├── app/
│   ├── main.py                 # FastAPI + gRPC startup + CORS
│   ├── config.py               # Конфигурация (Pydantic Settings)
│   ├── api/
│   │   ├── routes.py           # REST endpoints
│   │   ├── schemas.py          # Pydantic модели
│   │   └── openai_compat.py    # OpenAI-compatible /v1/ proxy
│   ├── core/
│   │   ├── parser.py           # Парсинг (Gemini / Docling / Vision)
│   │   ├── vision_parser.py    # Vision LLM парсинг (page-as-image)
│   │   ├── chunker.py          # Parent-child чанкинг + overlap
│   │   ├── embedder.py         # Embedding (Gemini / OpenAI-compat)
│   │   ├── indexer.py          # ChromaDB индексация
│   │   ├── retriever.py        # Hybrid search + multi-query + reranking
│   │   ├── query_rewriter.py   # LLM query rewriting + multi-query gen
│   │   ├── reranker.py         # Jina / vLLM cross-encoder
│   │   ├── llm_router.py       # Multi-provider LLM клиент
│   │   ├── generator.py        # Prompt assembly + generation
│   │   └── summarizer.py       # Auto-summary
│   ├── grpc_api/               # gRPC server + reflection
│   ├── evaluation/             # RAGAS-style evaluation
│   ├── models/                 # Document, Chunk, Session
│   └── utils/                  # Logging, monitoring, tracing, stores
├── deploy/
│   ├── docker-compose.yml      # vLLM + RAG API для GPU сервера
│   └── Dockerfile              # Контейнер RAG API
├── proto/rag.proto             # gRPC service definition
├── .env.example                # Шаблон конфигурации (облако)
├── .env.local                  # Шаблон конфигурации (локально)
└── requirements.txt
```

## ⚙️ Конфигурация

Все настройки через переменные окружения (`.env`):

```env
# LLM
RAG_LLM_PROVIDER=groq                          # gemini, groq, vllm, openai
RAG_LLM_BASE_URL=https://api.groq.com/openai/v1
RAG_LLM_MODEL=llama-3.3-70b-versatile

# Retrieval
RAG_QUERY_REWRITE_ENABLED=true                  # LLM переформулирует вопрос
RAG_MULTI_QUERY_ENABLED=false                   # 3 варианта запроса + RRF
RAG_PARENT_CHILD_ENABLED=true                   # Двухуровневые чанки

# Парсинг
RAG_PARSER_MODE=gemini                          # gemini, docling, vision
```

Полный список параметров: см. `app/config.py`

## 📜 Лицензия

MIT
