# 🗺️ RAG-LLM — Карта действий (Roadmap)

---

## Обзор стадий

```
Stage 1: Engine     → Рабочий RAG-пайплайн от PDF до ответа
Stage 2: Visibility → Показать внутренности пайплайна через API
Stage 3: Trace      → Пошаговое выполнение и метрики производительности
Stage 4: Control    → Менять параметры без перезапуска
Stage 5: Product    → Довести до удобного инструмента
```

---

## Stage 1 — Engine (Ядро) 🔧

> **Цель:** Рабочий RAG-пайплайн от PDF до ответа

### Задачи

| # | Задача | Детали |
|---|---|---|
| 1.1 | Структура проекта | Создать каталоги, `pyproject.toml`, конфиг |
| 1.2 | Модуль парсинга | Docling: PDF → структурированный текст |
| 1.3 | Модуль чанкинга | Docling HybridChunker → чанки с метаданными |
| 1.4 | Модуль embeddings | Запрос к vLLM `/v1/embeddings` → вектора |
| 1.5 | Модуль индексации | ChromaDB: сохранение чанков + векторов |
| 1.6 | Модуль retrieval | По вопросу → top-k чанков из ChromaDB |
| 1.7 | LLM Router | Единый клиент: vLLM / OpenAI / Anthropic / любой совместимый API |
| 1.8 | Модуль генерации | Сборка prompt + запрос через LLM Router |
| 1.9 | API endpoints | FastAPI: `/upload`, `/query`, `/collections`, `/health` |
| 1.10 | Тестирование | Загрузить реальный PDF, задать вопросы, проверить ответы |

### Структура проекта

```
rag-llm/
├── docs/                        # Документация проекта
│   ├── 01_PROJECT_DESCRIPTION.md
│   ├── 02_ROADMAP.md
│   └── 03_TECH_STACK.md
├── app/
│   ├── main.py              # FastAPI app, точка входа
│   ├── config.py             # Конфигурация (порты, модели, параметры)
│   ├── api/
│   │   ├── routes.py         # API endpoints
│   │   └── schemas.py        # Pydantic модели запрос/ответ
│   ├── core/
│   │   ├── parser.py         # Docling парсинг
│   │   ├── chunker.py        # Docling HybridChunker
│   │   ├── llm_router.py     # Роутер: vLLM / OpenAI / Anthropic / любой API
│   │   ├── embedder.py       # Embedding клиент (vLLM или API)
│   │   ├── indexer.py        # ChromaDB операции
│   │   ├── retriever.py      # Поиск по вопросу
│   │   └── generator.py      # Сборка prompt + LLM вызов
│   ├── models/
│   │   └── document.py       # Модели данных: Document, Chunk
│   └── utils/
│       ├── logging.py        # Loguru настройка
│       └── monitoring.py     # psutil/pynvml метрики
├── data/
│   ├── uploads/              # Загруженные PDF
│   └── chroma/               # Данные ChromaDB
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Результат стадии
✅ Можно загрузить PDF через API и получить ответ на вопрос по нему

---

## Stage 2 — Visibility (Прозрачность) 👁️ ✅

> **Цель:** Показать внутренности пайплайна через API

### Задачи

| # | Задача | Детали | Статус |
|---|---|---|---|
| 2.1 | Endpoint: извлечённый текст | `GET /documents/{id}/text` — что Docling вытащил | ✅ |
| 2.2 | Endpoint: чанки | `GET /documents/{id}/chunks` — все чанки с метаданными | ✅ |
| 2.3 | Расширить ответ query | В ответ `/query` добавить `chunks_used`, `scores`, `prompt` | ✅ |
| 2.4 | Endpoint: коллекции | `GET /collections` — список, статистика, количество чанков | ✅ |
| 2.5 | Endpoint: детали чанка | `GET /chunks/{id}` — текст, embedding вектор, метаданные | ✅ |

### Формат расширенного ответа

```json
{
  "answer": "RAG — это метод...",
  "chunks_used": [
    {
      "id": "chunk_042",
      "text": "Retrieval-Augmented Generation позволяет...",
      "score": 0.92,
      "page": 15,
      "section": "Глава 3: Методы"
    }
  ],
  "prompt": "Используя следующий контекст, ответь на вопрос...",
  "model": "Qwen2.5-7B-Instruct",
  "tokens_used": 1847
}
```

### Результат стадии
✅ В ответе API видно: какие чанки найдены, с какими scores, что ушло в prompt. Доступны endpoints для просмотра текста документа, чанков и деталей каждого чанка.

---

## Stage 3 — Trace (Трассировка) 📊 ✅

> **Цель:** Показать пошаговое выполнение и метрики производительности

### Задачи

| # | Задача | Детали | Статус |
|---|---|---|---|
| 3.1 | Структура trace | Каждый запрос получает `trace_id` | ✅ |
| 3.2 | Тайминги этапов | Время каждого шага: receive, retrieve, generate | ✅ |
| 3.3 | Мониторинг ресурсов | CPU, RAM, GPU на момент каждого запроса | ✅ |
| 3.4 | Endpoint: trace | `GET /traces/{id}` — полный лог запроса | ✅ |
| 3.5 | Endpoint: мониторинг | `GET /system/stats` — текущая нагрузка | ✅ |
| 3.6 | Endpoint: история | `GET /traces` — список последних запросов с метриками | ✅ |

### Реализация

**Файлы:**
- `app/utils/tracer.py` — Tracer с TraceStep/Trace dataclasses, хранит последние 100 трейсов
- `app/api/routes.py` — 3 новых endpoint: `/traces`, `/traces/{id}`, `/system/stats`

### Пример trace

```json
{
  "trace_id": "tr_abc123def456",
  "request_type": "query",
  "input_preview": "Что такое RAG?",
  "steps": [
    { "step": "receive_query", "time_ms": 0, "timestamp": "12:03:01.234", "question": "Что такое RAG?" },
    { "step": "retrieve_chunks", "time_ms": 23, "timestamp": "12:03:01.257", "chunks_found": 5, "top_k": 5 },
    { "step": "llm_generate", "time_ms": 2100, "timestamp": "12:03:03.357", "model": "gemini-2.0-flash", "total_tokens": 1847 },
    { "step": "total", "time_ms": 2123, "timestamp": "12:03:03.357" }
  ],
  "system": { "cpu_percent": 34, "ram_used_gb": 12.4, "gpu_util_percent": 62 },
  "total_ms": 2123,
  "status": "completed"
}
```

### Результат стадии
✅ Видно каждый шаг пайплайна с таймингами и нагрузкой. Можно диагностировать, где тормозит

---

## Stage 4 — Control (Управление) ⚙️

> **Цель:** Менять параметры без перезапуска

### Задачи

| # | Задача | Детали |
|---|---|---|
| 4.1 | Настройки чанкинга | `chunk_size`, `overlap` — менять через API |
| 4.2 | Настройки retrieval | `top_k`, `score_threshold` |
| 4.3 | Настройки генерации | `temperature`, `max_tokens`, `system_prompt` |
| 4.4 | Переиндексация | `POST /documents/{id}/reindex` — пересоздать чанки и embeddings |
| 4.5 | Сравнение стратегий | Один документ — разные параметры — сравнить ответы |

### Результат стадии
✅ Можно настраивать поведение RAG на лету и экспериментировать

---

## Stage 5 — Product (Продукт) 🚀

> **Цель:** Довести до удобного инструмента

### Задачи

| # | Задача | Детали |
|---|---|---|
| 5.1 | Web UI | React интерфейс: загрузка, чанки, вопросы, trace |
| 5.2 | Docker | Docker Compose: RAG-LLM + vLLM + ChromaDB |
| 5.3 | Доступ по сети | Привязка к IP, базовая аутентификация |
| 5.4 | Экспорт логов | Скачивание trace/логов в JSON/CSV |
| 5.5 | Мультиформат | DOCX, PPTX, HTML (Docling уже поддерживает) |
| 5.6 | Документация | README, API docs, примеры интеграции |

### Результат стадии
✅ Полноценный продукт, который можно развернуть на сервере и показать за 3 минуты

---

## Критерии готовности MVP (Stage 1 + 2 + 3)

MVP считается готовым, когда можно показать демо:

1. ✅ Загрузить PDF через API
2. ✅ Увидеть извлечённый текст
3. ✅ Увидеть чанки с метаданными
4. ✅ Задать вопрос и получить ответ
5. ✅ В ответе видны использованные чанки и scores
6. ✅ Виден prompt, отправленный в LLM
7. ✅ Видны тайминги каждого этапа
8. ✅ Видна нагрузка на систему
