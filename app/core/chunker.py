"""
Чанкинг документов — два режима:

1. 'markdown' (по умолчанию) — конвертирует документ в Markdown через Docling,
   затем разбивает по секциям (## заголовкам). Сохраняет таблицы как
   | col1 | col2 |, код как ```python, заголовки как ## Title.
   LLM обучены на Markdown — это оптимальный формат.

2. 'hybrid' — использует Docling HybridChunker + serialize().
   Таблицы преобразуются в плоский текст: "Выручка, Q4 = 15.2 млрд".
   Заголовки вставляются как хлебные крошки.

Оба режима:
- Определяют тип элемента (text, table, code, list, formula)
- Извлекают номера страниц из provenance
- Привязывают чанк к секции документа
"""

import re
import time

from app.config import settings
from app.models.document import Chunk, ChunkMetadata
from app.utils.logging import get_logger

log = get_logger("chunker")

# Приблизительный коэффициент: 1 токен ≈ 4 символа (для русского текста ~3)
CHARS_PER_TOKEN = 3


class DocumentChunker:
    """Чанкер документов с поддержкой Markdown и Hybrid режимов."""

    def chunk(
        self,
        docling_document,
        document_id: str,
        output_format: str = "markdown",
        include_headers: bool = True,
    ) -> list[Chunk]:
        """Нарезать документ на чанки.

        Args:
            docling_document: Объект DoclingDocument от парсера
            document_id: ID документа для привязки чанков
            output_format: Формат чанков:
                'markdown' — полный Markdown (## заголовки, | таблицы |, ```код```)
                'hybrid'   — Docling serialize (плоский текст + хлебные крошки)
            include_headers: (для hybrid) Включать заголовки в текст чанка

        Returns:
            Список чанков с метаданными
        """
        if output_format == "markdown":
            if settings.parent_child_enabled:
                return self._chunk_parent_child(docling_document, document_id)
            return self._chunk_markdown(docling_document, document_id)
        else:
            return self._chunk_hybrid(docling_document, document_id, include_headers)

    # ═══════════════════════════════════════════════════════
    # Parent-Child чанкинг
    # ═══════════════════════════════════════════════════════

    def _chunk_parent_child(self, docling_document, document_id: str) -> list[Chunk]:
        """Двухуровневый чанкинг: parent (большой, для LLM) + child (маленький, для поиска).

        Логика:
        1. Markdown → секции по ## (parent chunks, ~1024 токенов)
        2. Каждый parent → подчанки (children, ~256 токенов)
        3. Embed и ищем по children (точный поиск)
        4. При поиске возвращаем parent (богатый контекст)

        Возвращает все чанки (parents + children), indexer сохраняет оба типа.
        """
        log.info("Чанкинг (parent-child) документа {}", document_id)
        start = time.perf_counter()

        # 1. Получаем Markdown
        markdown_text = docling_document.export_to_markdown()
        if not markdown_text or not markdown_text.strip():
            log.warning("Документ не содержит текста!")
            return []

        # 2. Разбиваем на parent секции (большие, ~1024 токенов)
        parent_max_chars = settings.parent_max_tokens * CHARS_PER_TOKEN
        parent_sections = self._split_markdown_by_sections(markdown_text, parent_max_chars)

        # 3. Для каждого parent создаём children
        all_chunks = []
        child_max_chars = settings.child_max_tokens * CHARS_PER_TOKEN

        for section in parent_sections:
            parent_text = section["text"].strip()
            if not parent_text:
                continue

            # Создаём parent chunk
            element_type = self._detect_element_type(parent_text)
            parent_chunk = Chunk(
                document_id=document_id,
                text=parent_text,
                metadata=ChunkMetadata(
                    char_count=len(parent_text),
                    section=section.get("heading"),
                    element_type=element_type,
                    chunk_type="parent",
                ),
            )
            all_chunks.append(parent_chunk)

            # Разбиваем parent на children (если он достаточно длинный)
            if len(parent_text) > child_max_chars:
                child_texts = self._split_by_paragraphs(parent_text, child_max_chars)
                for child_text in child_texts:
                    child_text = child_text.strip()
                    if not child_text or len(child_text) < 30:
                        continue

                    child_element_type = self._detect_element_type(child_text)
                    child_chunk = Chunk(
                        document_id=document_id,
                        text=child_text,
                        metadata=ChunkMetadata(
                            char_count=len(child_text),
                            section=section.get("heading"),
                            element_type=child_element_type,
                            parent_id=parent_chunk.id,
                            chunk_type="child",
                        ),
                    )
                    all_chunks.append(child_chunk)
            else:
                # Parent достаточно короткий — он сам себе child
                # (будет найден как обычный чанк)
                pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        parents = [c for c in all_chunks if c.metadata.chunk_type == "parent"]
        children = [c for c in all_chunks if c.metadata.chunk_type == "child"]
        log.info(
            "Parent-child завершён: {} parents + {} children = {} всего, {:.0f} мс",
            len(parents), len(children), len(all_chunks), elapsed_ms,
        )

        return all_chunks

    # ═══════════════════════════════════════════════════════
    # Markdown чанкинг (обычный, без parent-child)
    # ═══════════════════════════════════════════════════════

    def _chunk_markdown(self, docling_document, document_id: str) -> list[Chunk]:
        """Конвертировать в Markdown и разбить по секциям."""
        log.info("Чанкинг (markdown) документа {}", document_id)
        start = time.perf_counter()

        # 1. Получаем полный Markdown от Docling
        markdown_text = docling_document.export_to_markdown()

        if not markdown_text or not markdown_text.strip():
            log.warning("Документ не содержит текста!")
            return []

        # 2. Разбиваем на секции по заголовкам ## (и ###)
        max_chars = settings.chunk_max_tokens * CHARS_PER_TOKEN
        sections = self._split_markdown_by_sections(markdown_text, max_chars)

        # 3. Создаём чанки
        chunks = []
        for section in sections:
            text = section["text"].strip()
            if not text:
                continue

            # Определяем тип элемента
            element_type = self._detect_element_type(text)

            meta = ChunkMetadata(
                char_count=len(text),
                section=section.get("heading"),
                element_type=element_type,
            )

            chunk = Chunk(
                document_id=document_id,
                text=text,
                metadata=meta,
            )
            chunks.append(chunk)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._log_stats(chunks, elapsed_ms, "markdown")
        return chunks

    def _split_markdown_by_sections(
        self, markdown_text: str, max_chars: int
    ) -> list[dict]:
        """Разбить Markdown по ## заголовкам с учётом max_chars.

        Если секция слишком длинная, разбивает её по подзаголовкам (###),
        а если всё ещё длинная — по абзацам (двойной перенос строки).
        """
        # Разбиваем по заголовкам уровня 2 (##)
        # Regex: ищем строки, начинающиеся с ##
        section_pattern = re.compile(r'^(#{1,3}\s+.+)$', re.MULTILINE)

        parts = []
        last_end = 0
        current_heading = None

        for match in section_pattern.finditer(markdown_text):
            # Сохраняем текст ДО этого заголовка
            if last_end < match.start():
                text_before = markdown_text[last_end:match.start()].strip()
                if text_before:
                    parts.append({
                        "heading": current_heading,
                        "text": (f"{current_heading}\n\n{text_before}" if current_heading
                                 else text_before),
                    })

            current_heading = match.group(1)
            last_end = match.end()

        # Последний кусок
        remaining = markdown_text[last_end:].strip()
        if remaining:
            parts.append({
                "heading": current_heading,
                "text": (f"{current_heading}\n\n{remaining}" if current_heading
                         else remaining),
            })

        # Объединяем слишком мелкие секции и разбиваем слишком крупные
        result = []
        for part in parts:
            text = part["text"]
            if len(text) <= max_chars:
                result.append(part)
            else:
                # Секция слишком длинная — разбиваем по абзацам
                sub_chunks = self._split_by_paragraphs(text, max_chars)
                for sc in sub_chunks:
                    result.append({
                        "heading": part["heading"],
                        "text": sc,
                    })

        # Объединяем слишком мелкие секции (< 50 символов) с предыдущей
        merged = []
        for part in result:
            if merged and len(part["text"]) < 50 and len(merged[-1]["text"]) + len(part["text"]) < max_chars:
                merged[-1]["text"] += "\n\n" + part["text"]
            else:
                merged.append(part)

        return merged

    def _split_by_paragraphs(self, text: str, max_chars: int) -> list[str]:
        """Разбить длинный текст по абзацам (\\n\\n) с overlap."""
        overlap_chars = settings.chunk_overlap_tokens * CHARS_PER_TOKEN
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_chars:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                # Overlap: берём последние N символов предыдущего чанка
                if overlap_chars > 0 and current:
                    overlap_text = current[-overlap_chars:]
                    current = overlap_text + "\n\n" + para
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    def _detect_element_type(self, text: str) -> str:
        """Определить основной тип контента в чанке."""
        has_table = bool(re.search(r'^\|.*\|.*\|', text, re.MULTILINE))
        has_code = '```' in text
        has_list = bool(re.search(r'^[-*]\s+', text, re.MULTILINE)) or \
                   bool(re.search(r'^\d+\.\s+', text, re.MULTILINE))
        has_formula = '$$' in text or '$' in text

        if has_table:
            return "table"
        if has_code:
            return "code"
        if has_list:
            return "list"
        if has_formula:
            return "formula"
        return "text"

    # ═══════════════════════════════════════════════════════
    # Hybrid чанкинг (старый режим)
    # ═══════════════════════════════════════════════════════

    def _chunk_hybrid(
        self, docling_document, document_id: str, include_headers: bool = True
    ) -> list[Chunk]:
        """Docling HybridChunker + serialize()."""
        log.info("Чанкинг (hybrid) документа {}", document_id)
        start = time.perf_counter()

        from docling.chunking import HybridChunker

        hybrid_chunker = HybridChunker(
            max_tokens=settings.chunk_max_tokens,
        )

        docling_chunks = list(hybrid_chunker.chunk(docling_document))

        chunks = []
        for i, dc in enumerate(docling_chunks):
            if include_headers:
                text = hybrid_chunker.serialize(dc)
            else:
                text = dc.text

            if not text or not text.strip():
                continue

            meta = ChunkMetadata(char_count=len(text))

            try:
                if hasattr(dc, "meta") and dc.meta:
                    if hasattr(dc.meta, "headings") and dc.meta.headings:
                        meta.section = " > ".join(dc.meta.headings)

                    if hasattr(dc.meta, "doc_items") and dc.meta.doc_items:
                        element_types = set()
                        for di in dc.meta.doc_items:
                            label = str(getattr(di, "label", "text"))
                            element_types.add(label)

                            if meta.page is None and hasattr(di, "prov"):
                                for prov_item in (di.prov or []):
                                    if hasattr(prov_item, "page_no") and prov_item.page_no is not None:
                                        meta.page = prov_item.page_no
                                        break

                        if "table" in element_types:
                            meta.element_type = "table"
                        elif "code" in element_types:
                            meta.element_type = "code"
                        elif "list_item" in element_types:
                            meta.element_type = "list"
                        elif any("formula" in et or "equation" in et for et in element_types):
                            meta.element_type = "formula"
                        else:
                            meta.element_type = "text"

                    if meta.page is None:
                        if hasattr(dc.meta, "page") and dc.meta.page is not None:
                            meta.page = dc.meta.page

            except Exception as e:
                log.debug("Ошибка извлечения метаданных чанка {}: {}", i, e)

            chunk = Chunk(
                document_id=document_id,
                text=text,
                metadata=meta,
            )
            chunks.append(chunk)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._log_stats(chunks, elapsed_ms, "hybrid")
        return chunks

    # ═══════════════════════════════════════════════════════
    # Общие утилиты
    # ═══════════════════════════════════════════════════════

    def _log_stats(self, chunks: list[Chunk], elapsed_ms: float, mode: str):
        """Логировать статистику чанкинга."""
        if chunks:
            avg_len = sum(c.metadata.char_count for c in chunks) // len(chunks)
            type_counts: dict[str, int] = {}
            for c in chunks:
                et = c.metadata.element_type or "text"
                type_counts[et] = type_counts.get(et, 0) + 1

            type_str = ", ".join(f"{t}={n}" for t, n in sorted(type_counts.items()))

            log.info(
                "Чанкинг ({}) завершён: {} чанков, avg {} символов, типы: [{}], {:.0f} мс",
                mode,
                len(chunks),
                avg_len,
                type_str,
                elapsed_ms,
            )
        else:
            log.warning("Чанкинг не создал ни одного чанка!")


# Глобальный экземпляр чанкера
chunker = DocumentChunker()

