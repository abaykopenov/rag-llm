"""
API endpoints для RAG-LLM.
"""

import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config import settings
from app.api.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    ChunkInfo,
    CollectionInfo,
    HealthResponse,
    SystemStatsResponse,
    # Stage 2
    DocumentTextResponse,
    DocumentChunkDetail,
    DocumentChunksResponse,
    ChunkDetailResponse,
    # Stage 3: Evaluation
    EvalRequest,
    EvalResponse,
    EvalMetricResult,
    # Stage 6: Chat
    ChatRequest,
    ChatResponse,
    ChatMessageInfo,
    SessionInfo,
    SessionDetailResponse,
)
from app.core.parser import parser
from app.core.chunker import chunker
from app.core.embedder import embedder
from app.core.indexer import indexer
from app.core.retriever import retriever
from app.core.generator import generator
from app.core.text_cleaner import clean_parsed_text, clean_chunk_text
from app.models.document import Document
from app.utils.logging import get_logger
from app.utils.monitoring import get_system_stats
from app.utils.document_store import document_store
from app.utils.session_store import session_store

log = get_logger("api")

router = APIRouter()


# === Вспомогательные функции ===

def _save_document_text(document_id: str, text: str) -> Path:
    """Сохранить извлечённый текст документа на диск."""
    texts_dir = Path(settings.texts_dir)
    texts_dir.mkdir(parents=True, exist_ok=True)
    text_path = texts_dir / f"{document_id}.md"
    text_path.write_text(text, encoding="utf-8")
    return text_path


def _load_document_text(document_id: str) -> str | None:
    """Загрузить сохранённый текст документа."""
    text_path = Path(settings.texts_dir) / f"{document_id}.md"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8")
    return None


# =====================================================
# Stage 1: Core endpoints
# =====================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
):
    """Загрузить документ (PDF, DOCX, и др.) в RAG систему.

    1. Валидирует формат файла
    2. Сохраняет файл на диск
    3. Парсит через Docling (асинхронно)
    4. Сохраняет извлечённый текст
    5. Нарезает на чанки
    6. Генерирует embeddings
    7. Сохраняет в ChromaDB
    """
    total_start = time.perf_counter()

    # Валидация формата файла
    try:
        parser.validate_file_extension(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    log.info("Загрузка файла: {} ({:.1f} MB)", file.filename, (file.size or 0) / (1024 * 1024))

    # Создаём документ
    doc = Document(
        filename=file.filename,
        file_size=file.size or 0,
        collection=collection,
        status="processing",
    )

    try:
        # 1. Сохраняем файл
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{doc.id}_{file.filename}"

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        log.info("Файл сохранён: {}", file_path)

        # 2. Парсим (асинхронно — не блокирует event loop)
        parse_result = await parser.parse_async(file_path)
        doc.pages_count = parse_result.pages_count

        # 3. Пост-обработка текста (удаление колонтитулов, номеров страниц, склейка строк)
        cleaned_text = clean_parsed_text(parse_result.full_text)
        doc.raw_text = cleaned_text

        # 4. Сохраняем очищенный текст на диск
        _save_document_text(doc.id, cleaned_text)
        log.info("Текст сохранён: {} символов", len(cleaned_text))

        # 5. Чанкинг
        chunks = chunker.chunk(parse_result.docling_document, doc.id)
        # Пост-обработка каждого чанка
        for c in chunks:
            c.text = clean_chunk_text(c.text)
        # Убираем пустые чанки после очистки
        chunks = [c for c in chunks if c.text and len(c.text.strip()) > 10]
        doc.chunks_count = len(chunks)

        if not chunks:
            raise ValueError("Документ не содержит текста для индексации")

        # 5. Embeddings
        texts = [c.text for c in chunks]
        embeddings = await embedder.embed_texts(texts)

        # Добавляем filename в метаданные чанков для поиска
        for c in chunks:
            c.metadata.filename = doc.filename

        # 6. Индексация в ChromaDB
        indexer.add_chunks(collection, chunks, embeddings)

        # Готово
        doc.status = "ready"
        doc.processing_time_ms = (time.perf_counter() - total_start) * 1000

        # Сохраняем в персистентное хранилище
        document_store.save(doc)

        # Auto-summary (не блокирует ошибки — если LLM недоступен, пропускаем)
        summary = None
        try:
            from app.core.summarizer import summarizer
            if doc.raw_text:
                summary = await summarizer.summarize(doc.raw_text, doc.filename)
                if summary:
                    doc.summary = summary
                    document_store.save(doc)
        except Exception as e:
            log.debug("Auto-summary пропущен (LLM недоступен): {}", e)

        log.info(
            "Документ готов: {} | {} страниц, {} чанков, {:.0f} мс",
            file.filename,
            doc.pages_count,
            doc.chunks_count,
            doc.processing_time_ms,
        )

        return UploadResponse(
            document_id=doc.id,
            filename=doc.filename,
            collection=doc.collection,
            pages_count=doc.pages_count,
            chunks_count=doc.chunks_count,
            status=doc.status,
            processing_time_ms=round(doc.processing_time_ms, 1),
            summary=summary,
        )

    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)
        document_store.save(doc)
        log.error("Ошибка обработки {}: {}", file.filename, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/summary")
async def get_document_summary(document_id: str):
    """Получить или сгенерировать краткое содержание документа."""
    doc = document_store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    # Если summary уже есть — возвращаем
    if doc.summary:
        return {"document_id": document_id, "summary": doc.summary, "cached": True}

    # Получаем текст: из памяти или с диска
    text = doc.raw_text or _load_document_text(document_id)
    if not text:
        raise HTTPException(status_code=400, detail="Текст документа не сохранён")

    try:
        from app.core.summarizer import summarizer
        summary = await summarizer.summarize(text, doc.filename)
        if summary:
            doc.summary = summary
            document_store.save(doc)
            return {"document_id": document_id, "summary": summary, "cached": False}
        else:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать summary")
    except ImportError:
        raise HTTPException(status_code=503, detail="LLM недоступен для суммаризации")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Задать вопрос по загруженным документам.

    1. Ищет релевантные чанки в коллекции
    2. Собирает prompt с контекстом
    3. Генерирует ответ через LLM
    4. Возвращает ответ + использованные чанки + prompt + тайминги + trace_id
    """
    from app.utils.tracer import tracer

    total_start = time.perf_counter()

    # Начинаем трассировку
    trace_id = tracer.start_trace("query", request.question)
    tracer.add_step(trace_id, "receive_query", 0, question=request.question[:80], collection=request.collection)

    log.info("Вопрос: '{}' (collection='{}', trace={})", request.question[:80], request.collection, trace_id)

    try:
        # 1. Retrieval
        retrieval_start = time.perf_counter()
        chunks = await retriever.retrieve(
            query=request.question,
            collection=request.collection,
            top_k=request.top_k,
            document_id=request.document_id,
            element_type=request.element_type,
            section=request.section,
            keywords=request.keywords,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        tracer.add_step(trace_id, "retrieve_chunks", retrieval_ms,
                        chunks_found=len(chunks),
                        top_k=request.top_k or settings.retrieval_top_k,
                        filters={
                            "document_id": request.document_id,
                            "element_type": request.element_type,
                            "keywords": request.keywords,
                        })

        if not chunks:
            tracer.add_step(trace_id, "no_results", 0)
            tracer.end_trace(trace_id, "completed")
            return QueryResponse(
                answer="В загруженных документах не найдено релевантной информации по вашему вопросу.",
                chunks_used=[],
                timing={"retrieval_ms": round(retrieval_ms, 1), "total_ms": round(retrieval_ms, 1), "trace_id": trace_id},
            )

        # 2. Генерация
        generation_start = time.perf_counter()
        result = await generator.generate(
            query=request.question,
            chunks=chunks,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000
        tracer.add_step(trace_id, "llm_generate", generation_ms,
                        model=result.model,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens)

        total_ms = (time.perf_counter() - total_start) * 1000
        tracer.end_trace(trace_id, "completed")

        # 3. Формируем ответ с полной прозрачностью
        chunks_info = [
            ChunkInfo(
                id=c.id,
                text=c.text,
                score=c.score,
                page=c.metadata.page,
                section=c.metadata.section,
            )
            for c in chunks
        ]

        return QueryResponse(
            answer=result.answer,
            chunks_used=chunks_info,
            prompt=result.prompt,
            model=result.model,
            tokens_used=result.total_tokens,
            timing={
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "llm_ms": round(result.llm_time_ms, 1),
                "total_ms": round(total_ms, 1),
                "trace_id": trace_id,
            },
        )

    except Exception as e:
        tracer.end_trace(trace_id, "error")
        raise


# ═══════════════════════════════════════════════════════════════
# Retrieve-only endpoint (без LLM генерации)
# ═══════════════════════════════════════════════════════════════

@router.post("/retrieve")
async def retrieve_chunks(request: QueryRequest):
    """Поиск релевантных чанков БЕЗ LLM-генерации.

    Полезно когда клиент (например SCORM-генератор) хочет:
    1. Получить только чанки (контекст)
    2. Самостоятельно построить промпт
    3. Вызвать LLM напрямую (Ollama/vLLM)

    Включает query rewrite если настроен.
    """
    total_start = time.perf_counter()

    log.info("Retrieve: '{}' (collection='{}', top_k={})",
             request.question[:80], request.collection, request.top_k)

    chunks = await retriever.retrieve(
        query=request.question,
        collection=request.collection,
        top_k=request.top_k,
        document_id=request.document_id,
        element_type=request.element_type,
        section=request.section,
        keywords=request.keywords,
    )

    retrieval_ms = (time.perf_counter() - total_start) * 1000

    chunks_info = [
        ChunkInfo(
            id=c.id,
            text=c.text,
            score=c.score,
            page=c.metadata.page,
            section=c.metadata.section,
        )
        for c in chunks
    ]

    return {
        "chunks": chunks_info,
        "total": len(chunks_info),
        "collection": request.collection,
        "query": request.question,
        "timing_ms": round(retrieval_ms, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Trace endpoints (Stage 3)
# ═══════════════════════════════════════════════════════════════

@router.get("/traces")
async def list_traces(limit: int = 20):
    """Список последних трассировок."""
    from app.utils.tracer import tracer
    return {
        "traces": tracer.get_recent_traces(limit),
        "stats": tracer.get_stats(),
    }


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Получить полную трассировку запроса."""
    from app.utils.tracer import tracer
    trace = tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace не найден")
    return trace


@router.get("/system/stats")
async def system_stats():
    """Текущая нагрузка на систему + статистика запросов."""
    from app.utils.tracer import tracer
    import platform

    stats = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "app_version": settings.app_version,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "reranker_enabled": settings.reranker_enabled,
        "trace_stats": tracer.get_stats(),
    }

    try:
        import psutil
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        stats["ram_total_gb"] = round(mem.total / (1024**3), 1)
        stats["ram_used_gb"] = round(mem.used / (1024**3), 1)
        stats["ram_percent"] = mem.percent
    except ImportError:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
        stats["gpu_util_percent"] = util.gpu
        stats["gpu_vram_total_gb"] = round(mem_info.total / (1024**3), 1)
        stats["gpu_vram_used_gb"] = round(mem_info.used / (1024**3), 1)
        pynvml.nvmlShutdown()
    except Exception:
        stats["gpu"] = "not available"

    return stats


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections():
    """Список всех коллекций с количеством чанков."""
    collections = indexer.list_collections()
    return [CollectionInfo(**c) for c in collections]


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Удалить коллекцию."""
    indexer.delete_collection(name)
    # Удаляем связанные документы из персистентного хранилища
    removed = document_store.delete_by_collection(name)
    log.info("Удалена коллекция '{}', связано документов: {}", name, removed)

    return {"status": "deleted", "collection": name, "documents_removed": removed}


@router.get("/documents")
async def list_documents():
    """Список загруженных документов."""
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "collection": doc.collection,
            "pages_count": doc.pages_count,
            "chunks_count": doc.chunks_count,
            "status": doc.status,
            "file_size": doc.file_size,
            "created_at": doc.created_at.isoformat(),
        }
        for doc in document_store.get_all()
    ]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния системы."""
    collections = indexer.list_collections()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        llm_provider=f"{settings.llm_provider} ({settings.llm_base_url})",
        embedding_provider=f"{settings.embedding_provider} ({settings.embedding_base_url})",
        collections=[CollectionInfo(**c) for c in collections],
    )


@router.get("/system/stats", response_model=SystemStatsResponse)
async def system_stats():
    """Текущая нагрузка системы: CPU, RAM, GPU."""
    stats = get_system_stats()
    return SystemStatsResponse(**stats)


# =====================================================
# Stage 2: Visibility endpoints
# =====================================================

@router.get("/documents/{document_id}/text", response_model=DocumentTextResponse)
async def get_document_text(document_id: str):
    """Получить извлечённый текст документа.

    Возвращает полный текст, который Docling извлёк из PDF/DOCX/и т.д.
    Текст в формате Markdown.
    """
    # Проверяем, что документ существует
    doc = document_store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Документ '{document_id}' не найден")

    # Загружаем текст с диска
    text = _load_document_text(document_id)
    if text is None:
        raise HTTPException(
            status_code=404,
            detail=f"Текст документа '{document_id}' не найден. "
                   f"Возможно, документ был загружен до включения сохранения текстов."
        )

    return DocumentTextResponse(
        document_id=doc.id,
        filename=doc.filename,
        pages_count=doc.pages_count,
        text_length=len(text),
        text=text,
    )


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(document_id: str):
    """Получить все чанки документа с метаданными.

    Показывает, на какие фрагменты был нарезан документ:
    текст каждого чанка, страница, раздел, тип элемента.
    """
    # Проверяем, что документ существует
    doc = document_store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Документ '{document_id}' не найден")

    # Получаем чанки из ChromaDB
    raw_chunks = indexer.get_chunks_by_document(doc.collection, document_id)

    chunks = []
    for c in raw_chunks:
        meta = c.get("metadata", {})
        section = meta.get("section", "") or None
        chunks.append(DocumentChunkDetail(
            id=c["id"],
            text=c["text"],
            page=meta.get("page") or None,
            section=section if section else None,
            element_type=meta.get("element_type"),
            char_count=meta.get("char_count", len(c["text"])),
        ))

    avg_len = 0
    if chunks:
        avg_len = sum(c.char_count for c in chunks) // len(chunks)

    return DocumentChunksResponse(
        document_id=doc.id,
        filename=doc.filename,
        collection=doc.collection,
        chunks_count=len(chunks),
        avg_chunk_length=avg_len,
        chunks=chunks,
    )


@router.get("/chunks/{chunk_id}", response_model=ChunkDetailResponse)
async def get_chunk_detail(
    chunk_id: str,
    collection: str = Query(default="default", description="Коллекция для поиска чанка"),
    include_embedding: bool = Query(default=False, description="Включить embedding вектор"),
):
    """Получить детали одного чанка.

    Возвращает текст, метаданные и (опционально) embedding вектор чанка.
    Полезно для отладки: понять, как выглядит конкретный фрагмент документа
    и его числовое представление.
    """
    # Ищем чанк в ChromaDB
    chunk = indexer.get_chunk_by_id(collection, chunk_id, include_embedding=include_embedding)

    if not chunk:
        raise HTTPException(status_code=404, detail=f"Чанк '{chunk_id}' не найден в коллекции '{collection}'")

    meta = chunk.get("metadata", {})
    section = meta.get("section", "") or None
    embedding = chunk.get("embedding")

    return ChunkDetailResponse(
        id=chunk["id"],
        text=chunk["text"],
        document_id=meta.get("document_id"),
        collection=collection,
        page=meta.get("page") or None,
        section=section if section else None,
        element_type=meta.get("element_type"),
        char_count=meta.get("char_count", len(chunk["text"])),
        embedding=embedding,
        embedding_dimensions=len(embedding) if embedding else None,
    )


# =====================================================
# Stage 3: Evaluation
# =====================================================

@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_rag(request: EvalRequest):
    """Оценить качество RAG-пайплайна (RAGAS-style).

    Для каждого вопроса:
    1. Ищем чанки в коллекции
    2. Генерируем ответ
    3. Оцениваем через LLM-as-judge:
       - Faithfulness: ответ основан на контексте?
       - Answer Relevancy: ответ релевантен вопросу?
       - Context Precision: найдены правильные чанки?
       - Context Recall: все ли нужные чанки найдены?
    """
    from app.evaluation.evaluator import evaluator, EvalSample

    samples = []
    for s in request.samples:
        # Прогоняем RAG pipeline
        chunks = await retriever.retrieve(
            query=s.question,
            collection=request.collection,
            top_k=request.top_k,
        )

        if chunks:
            result = await generator.generate(
                query=s.question,
                chunks=chunks,
            )
            answer = result.answer
        else:
            answer = "Не найдено релевантной информации."

        samples.append(EvalSample(
            question=s.question,
            answer=answer,
            contexts=[c.text for c in chunks],
            ground_truth=s.ground_truth,
        ))

    # Оцениваем
    report = await evaluator.evaluate_batch(samples)

    # Буквенная оценка
    avg = report.avg_overall
    if avg >= 0.9:
        grade = "A"
    elif avg >= 0.8:
        grade = "B"
    elif avg >= 0.7:
        grade = "C"
    elif avg >= 0.6:
        grade = "D"
    else:
        grade = "F"

    results = []
    for i, r in enumerate(report.results):
        results.append(EvalMetricResult(
            question=request.samples[i].question,
            faithfulness=round(r.faithfulness, 3),
            answer_relevancy=round(r.answer_relevancy, 3),
            context_precision=round(r.context_precision, 3),
            context_recall=round(r.context_recall, 3),
            overall=round(r.overall, 3),
            answer_preview=samples[i].answer[:200],
        ))

    return EvalResponse(
        samples_count=report.samples_count,
        avg_faithfulness=round(report.avg_faithfulness, 3),
        avg_answer_relevancy=round(report.avg_answer_relevancy, 3),
        avg_context_precision=round(report.avg_context_precision, 3),
        avg_context_recall=round(report.avg_context_recall, 3),
        avg_overall=round(report.avg_overall, 3),
        grade=grade,
        results=results,
        eval_time_ms=round(report.eval_time_ms, 1),
    )


# =====================================================
# Stage 6: Conversational RAG
# =====================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Чат с RAG — с историей диалога.

    1. Создаёт или загружает сессию
    2. Ищет релевантные чанки по вопросу
    3. Собирает prompt с историей + контекстом
    4. Генерирует ответ
    5. Сохраняет сообщения в сессию
    """
    from app.utils.tracer import tracer

    total_start = time.perf_counter()

    # 1. Сессия: загрузить или создать
    if request.session_id:
        session = session_store.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Сессия '{request.session_id}' не найдена")
    else:
        session = session_store.create(collection=request.collection)

    trace_id = tracer.start_trace("chat", request.message)
    log.info(
        "Chat: session={}, msg='{}' (collection='{}', history={})",
        session.id, request.message[:60], session.collection, session.message_count,
    )

    try:
        # 2. Retrieval
        retrieval_start = time.perf_counter()
        chunks = await retriever.retrieve(
            query=request.message,
            collection=session.collection,
            top_k=request.top_k,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # 3. История диалога
        history = session.get_history(max_pairs=5)

        # 4. Генерация с историей
        if chunks:
            result = await generator.generate(
                query=request.message,
                chunks=chunks,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                history=history,
            )
            answer = result.answer
            model = result.model
            tokens = result.total_tokens
        else:
            answer = "В загруженных документах не найдено релевантной информации по вашему вопросу."
            model = ""
            tokens = 0

        total_ms = (time.perf_counter() - total_start) * 1000

        # 5. Сохраняем сообщения в сессию
        session.add_message("user", request.message)
        session.add_message(
            "assistant", answer,
            chunks_used=len(chunks),
            model=model,
            tokens_used=tokens,
        )
        session_store.save(session)

        tracer.end_trace(trace_id, "completed")

        chunks_info = [
            ChunkInfo(
                id=c.id, text=c.text, score=c.score,
                page=c.metadata.page, section=c.metadata.section,
            )
            for c in chunks
        ]

        return ChatResponse(
            session_id=session.id,
            answer=answer,
            chunks_used=chunks_info,
            model=model,
            tokens_used=tokens,
            message_count=session.message_count,
            timing={
                "retrieval_ms": round(retrieval_ms, 1),
                "total_ms": round(total_ms, 1),
                "trace_id": trace_id,
            },
        )

    except Exception as e:
        tracer.end_trace(trace_id, "error")
        raise


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """Список всех сессий диалога."""
    return [
        SessionInfo(
            id=s.id,
            title=s.title,
            collection=s.collection,
            message_count=s.message_count,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in session_store.get_all()
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    """Получить полную историю сессии."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Сессия '{session_id}' не найдена")

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        collection=session.collection,
        messages=[
            ChatMessageInfo(
                role=m.role,
                content=m.content,
                timestamp=m.timestamp.isoformat(),
                chunks_used=m.chunks_used,
                model=m.model,
                tokens_used=m.tokens_used,
            )
            for m in session.messages
        ],
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Удалить сессию."""
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail=f"Сессия '{session_id}' не найдена")
    return {"status": "deleted", "session_id": session_id}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming чат с RAG — Server-Sent Events.

    Ответ приходит по частям в формате SSE:
    data: {"delta": "часть текста"}
    data: {"delta": "", "done": true, "session_id": "...", "message_count": 4}
    """
    import json as json_module
    from app.core.llm_router import llm_router

    # 1. Сессия
    if request.session_id:
        session = session_store.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Сессия '{request.session_id}' не найдена")
    else:
        session = session_store.create(collection=request.collection)

    # 2. Retrieval
    chunks = await retriever.retrieve(
        query=request.message,
        collection=session.collection,
        top_k=request.top_k,
    )

    # 3. Prompt с историей
    history = session.get_history(max_pairs=5)
    messages = generator.build_prompt(request.message, chunks, history=history)

    async def event_generator():
        full_answer = []
        try:
            async for delta in llm_router.generate_stream(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                full_answer.append(delta)
                yield f"data: {json_module.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

            # Финальный event
            answer = "".join(full_answer)
            session.add_message("user", request.message)
            session.add_message("assistant", answer, chunks_used=len(chunks))
            session_store.save(session)

            yield f"data: {json_module.dumps({'delta': '', 'done': True, 'session_id': session.id, 'message_count': session.message_count}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json_module.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
