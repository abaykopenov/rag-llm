"""
gRPC Service Implementation — реализация всех RPC методов.

Использует ту же бизнес-логику что и REST API,
но через gRPC протокол (быстрее, типизированнее).
"""

import asyncio
import time
import tempfile
from pathlib import Path

import grpc

from app.grpc_api import rag_pb2, rag_pb2_grpc
from app.config import settings
from app.core.parser import parser
from app.core.chunker import chunker
from app.core.embedder import embedder
from app.core.indexer import indexer
from app.core.retriever import retriever
from app.core.generator import generator
from app.models.document import Document
from app.utils.logging import get_logger
from app.utils.document_store import document_store

log = get_logger("grpc_service")


class RAGServicer(rag_pb2_grpc.RAGServiceServicer):
    """Имплементация RAG gRPC сервиса."""

    async def Upload(self, request, context):
        """Загрузка и обработка документа."""
        total_start = time.perf_counter()

        filename = request.filename or "document.pdf"
        collection = request.collection or "default"

        log.info("gRPC Upload: {} ({} bytes) → '{}'", filename, len(request.file_content), collection)

        try:
            # Сохраняем файл во временную директорию
            upload_dir = Path(settings.upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)

            doc = Document(filename=filename, collection=collection, file_size=len(request.file_content))
            doc.status = "processing"

            file_path = upload_dir / f"{doc.id}_{filename}"
            file_path.write_bytes(request.file_content)

            # Pipeline: parse → chunk → embed → index
            text, pages_count = await parser.parse(str(file_path))
            doc.raw_text = text
            doc.pages_count = pages_count

            # Сохраняем текст
            texts_dir = Path(settings.texts_dir)
            texts_dir.mkdir(parents=True, exist_ok=True)
            ext = Path(filename).suffix or ".md"
            text_path = texts_dir / f"{doc.id}{ext}"
            text_path.write_text(text, encoding="utf-8")

            # Chunk
            chunks = chunker.chunk(text, document_id=doc.id, source_filename=filename)
            doc.chunks_count = len(chunks)

            # Embed
            texts_to_embed = [c.text for c in chunks]
            embeddings = await embedder.embed_texts(texts_to_embed)

            for chunk_obj, emb in zip(chunks, embeddings):
                chunk_obj.embedding = emb

            # Index
            indexer.add_chunks(collection, chunks)

            doc.status = "ready"
            doc.processing_time_ms = (time.perf_counter() - total_start) * 1000
            document_store.save(doc)

            # Auto-summary
            summary = ""
            try:
                from app.core.summarizer import summarizer
                if doc.raw_text:
                    summary = await summarizer.summarize(doc.raw_text, doc.filename) or ""
                    if summary:
                        doc.summary = summary
                        document_store.save(doc)
            except Exception:
                pass

            return rag_pb2.UploadResponse(
                document_id=doc.id,
                filename=doc.filename,
                collection=doc.collection,
                pages_count=doc.pages_count,
                chunks_count=doc.chunks_count,
                status=doc.status,
                processing_time_ms=doc.processing_time_ms,
                summary=summary,
            )

        except Exception as e:
            log.error("gRPC Upload error: {}", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return rag_pb2.UploadResponse()

    async def Query(self, request, context):
        """Задать вопрос по документам."""
        total_start = time.perf_counter()
        question = request.question

        if not question:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("question is required")
            return rag_pb2.QueryResponse()

        collection = request.collection or "default"
        top_k = request.top_k or None
        temperature = request.temperature or None
        max_tokens = request.max_tokens or None

        log.info("gRPC Query: '{}' (collection='{}')", question[:80], collection)

        # Retrieval
        retrieval_start = time.perf_counter()
        chunks = await retriever.retrieve(
            query=question,
            collection=collection,
            top_k=top_k,
            document_id=request.document_id or None,
            element_type=request.element_type or None,
            section=request.section or None,
            keywords=list(request.keywords) if request.keywords else None,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if not chunks:
            return rag_pb2.QueryResponse(
                answer="В загруженных документах не найдено релевантной информации.",
                timing={"retrieval_ms": retrieval_ms},
            )

        # Generation
        generation_start = time.perf_counter()
        result = await generator.generate(
            query=question,
            chunks=chunks,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000

        chunks_info = [
            rag_pb2.ChunkInfo(
                id=c.id,
                text=c.text,
                score=c.score,
                page=c.metadata.page or 0,
                section=c.metadata.section or "",
            )
            for c in chunks
        ]

        return rag_pb2.QueryResponse(
            answer=result.answer,
            chunks_used=chunks_info,
            prompt=result.prompt,
            model=result.model,
            tokens_used=result.total_tokens,
            timing={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
        )

    async def QueryStream(self, request, context):
        """Streaming ответ — по кусочкам."""
        # Сначала делаем обычный запрос
        question = request.question
        collection = request.collection or "default"

        if not question:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("question is required")
            return

        # Retrieval
        chunks = await retriever.retrieve(
            query=question,
            collection=collection,
            top_k=request.top_k or None,
            document_id=request.document_id or None,
            element_type=request.element_type or None,
            section=request.section or None,
            keywords=list(request.keywords) if request.keywords else None,
        )

        if not chunks:
            yield rag_pb2.QueryChunk(
                text_delta="В загруженных документах не найдено релевантной информации.",
                is_final=True,
            )
            return

        # Генерируем ответ
        result = await generator.generate(
            query=question,
            chunks=chunks,
            temperature=request.temperature or None,
            max_tokens=request.max_tokens or None,
        )

        # Разбиваем ответ на чанки для streaming
        answer = result.answer
        chunk_size = 50  # символов на чанк
        chunks_info = [
            rag_pb2.ChunkInfo(
                id=c.id, text=c.text, score=c.score,
                page=c.metadata.page or 0,
                section=c.metadata.section or "",
            )
            for c in chunks
        ]

        for i in range(0, len(answer), chunk_size):
            text_part = answer[i:i + chunk_size]
            is_last = (i + chunk_size >= len(answer))

            yield rag_pb2.QueryChunk(
                text_delta=text_part,
                is_final=is_last,
                chunks_used=chunks_info if is_last else [],
                model=result.model if is_last else "",
            )
            await asyncio.sleep(0.01)  # Имитация streaming задержки

    async def Evaluate(self, request, context):
        """Оценка качества RAG (RAGAS-style)."""
        from app.evaluation.evaluator import evaluator, EvalSample

        collection = request.collection or "default"
        top_k = request.top_k or 5

        samples = []
        for s in request.samples:
            chunks = await retriever.retrieve(
                query=s.question,
                collection=collection,
                top_k=top_k,
            )

            if chunks:
                result = await generator.generate(query=s.question, chunks=chunks)
                answer = result.answer
            else:
                answer = "Не найдено релевантной информации."

            samples.append(EvalSample(
                question=s.question,
                answer=answer,
                contexts=[c.text for c in chunks],
                ground_truth=s.ground_truth or None,
            ))

        report = await evaluator.evaluate_batch(samples)

        avg = report.avg_overall
        grade = "A" if avg >= 0.9 else "B" if avg >= 0.8 else "C" if avg >= 0.7 else "D" if avg >= 0.6 else "F"

        results = [
            rag_pb2.EvalMetric(
                question=request.samples[i].question,
                faithfulness=r.faithfulness,
                answer_relevancy=r.answer_relevancy,
                context_precision=r.context_precision,
                context_recall=r.context_recall,
                overall=r.overall,
                answer_preview=samples[i].answer[:200],
            )
            for i, r in enumerate(report.results)
        ]

        return rag_pb2.EvalResponse(
            samples_count=report.samples_count,
            avg_faithfulness=report.avg_faithfulness,
            avg_answer_relevancy=report.avg_answer_relevancy,
            avg_context_precision=report.avg_context_precision,
            avg_context_recall=report.avg_context_recall,
            avg_overall=report.avg_overall,
            grade=grade,
            results=results,
            eval_time_ms=report.eval_time_ms,
        )

    async def GetSummary(self, request, context):
        """Получить краткое содержание документа."""
        doc = document_store.get(request.document_id)
        if not doc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Document not found")
            return rag_pb2.SummaryResponse()

        if doc.summary:
            return rag_pb2.SummaryResponse(
                document_id=request.document_id,
                summary=doc.summary,
                cached=True,
            )

        if not doc.raw_text:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Document text not saved")
            return rag_pb2.SummaryResponse()

        try:
            from app.core.summarizer import summarizer
            summary = await summarizer.summarize(doc.raw_text, doc.filename)
            if summary:
                doc.summary = summary
                document_store.save(doc)
                return rag_pb2.SummaryResponse(
                    document_id=request.document_id,
                    summary=summary,
                    cached=False,
                )
        except Exception as e:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(f"LLM unavailable: {e}")

        return rag_pb2.SummaryResponse()

    async def ListCollections(self, request, context):
        """Список коллекций."""
        collections = indexer.list_collections()
        items = []
        for name in collections:
            count = indexer.get_collection_count(name)
            items.append(rag_pb2.CollectionInfo(name=name, chunks_count=count))

        return rag_pb2.CollectionList(collections=items)

    async def Health(self, request, context):
        """Health check."""
        return rag_pb2.HealthResponse(
            status="ok",
            version=settings.app_version,
            llm_provider=settings.llm_provider,
            embedding_provider=settings.embedding_provider,
            reranker_enabled=settings.reranker_enabled,
        )
