"""
Чанкинг документов.

Режимы (по умолчанию — 'hybrid'):

1. 'hybrid' (рекомендуемый) — Docling HybridChunker с tokenizer'ом
   активного embedding-провайдера. Итоговый чанк = chunker.contextualize(),
   что добавляет heading breadcrumbs к тексту для embedding.
   Метаданные: page_no, headings, element_type берутся из chunk.meta —
   семантически, не через regex.

2. 'markdown' (legacy) — режет Markdown, полученный из Docling, по '##'
   заголовкам собственным regex-сплиттером. Работает без токенизатора,
   но теряет page_no и структурные labels. Сохранён ТОЛЬКО потому что
   на нём основан parent-child режим. См. TODO внутри.

Parent-child режим — пока работает поверх markdown-сплиттера (см. TODO).
Будет переведён на структуру DoclingDocument в этапе 4.
"""

import re
import time
from typing import Optional

from app.config import settings
from app.models.document import Chunk, ChunkMetadata
from app.utils.logging import get_logger

log = get_logger("chunker")

# Приблизительный коэффициент: 1 токен ≈ 3 символа (для русского текста).
# Используется ТОЛЬКО в legacy markdown-сплиттере; hybrid-режим использует
# реальный токенизатор активного embedding-провайдера.
CHARS_PER_TOKEN = 3

# Маппинг Docling doc_item.label → наш element_type.
# Проверяем в этом порядке: более специфичные сначала.
_LABEL_TO_ELEMENT_TYPE = {
    "table": "table",
    "code": "code",
    "code_block": "code",
    "formula": "formula",
    "equation": "formula",
    "list_item": "list",
    "figure": "figure",
    "picture": "figure",
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "header",
    "page_footer": "footer",
}


class DocumentChunker:
    """Чанкер документов: hybrid (Docling) как основной, markdown как legacy."""

    def chunk(
        self,
        docling_document,
        document_id: str,
        output_format: Optional[str] = None,
        include_headers: bool = True,
    ) -> list[Chunk]:
        """Нарезать документ на чанки.

        Args:
            docling_document: DoclingDocument от парсера
            document_id: ID документа для привязки чанков
            output_format: 'hybrid' (default) или 'markdown' (legacy).
                           None = из settings.chunk_output_format (default 'hybrid').
            include_headers: (для hybrid) Добавлять breadcrumbs заголовков в текст чанка

        Returns:
            Список чанков с метаданными
        """
        fmt = (output_format or getattr(settings, "chunk_output_format", "hybrid")).lower()

        # Parent-child пока использует markdown-сплиттер (TODO: перевести на DoclingDocument)
        if settings.parent_child_enabled:
            return self._chunk_parent_child(docling_document, document_id)

        if fmt == "hybrid":
            return self._chunk_hybrid(docling_document, document_id, include_headers)
        if fmt == "markdown":
            return self._chunk_markdown(docling_document, document_id)
        raise ValueError(f"Неизвестный output_format: {fmt}. Ожидается 'hybrid' или 'markdown'.")

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
    # LEGACY: Markdown-сплиттер по regex
    # ═══════════════════════════════════════════════════════
    # Оставлен ТОЛЬКО потому что _chunk_parent_child использует
    # _split_markdown_by_sections. После перевода parent-child на
    # структуру DoclingDocument (этап 4 плана) этот блок можно удалить.
    # Теряет: page_no, точные element_type из doc_items, настоящий
    # счёт токенов. Не используй для новых чанкингов.

    def _chunk_markdown(self, docling_document, document_id: str) -> list[Chunk]:
        """[LEGACY] Markdown через Docling → regex-сплит по ## заголовкам."""
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

    def _build_hybrid_chunker(self, max_tokens: Optional[int] = None):
        """Сконструировать HybridChunker с tokenizer активного провайдера.

        Если у активного embedding-профиля есть tokenizer_spec — используем его.
        Иначе падаем с понятной ошибкой: нельзя эффективно чанковать без
        tokenizer'а модели (границы не совпадут с окном embedding).

        max_tokens=None → возьмётся из tokenizer_spec.
        """
        from docling.chunking import HybridChunker

        # Ленивый импорт: embedder → провайдер → tokenizer
        from app.core.embedder import embedder

        tokenizer = embedder.get_tokenizer()
        if tokenizer is None:
            raise RuntimeError(
                "Активный embedding-профиль не сконфигурирован c tokenizer'ом. "
                "Задай его в config/embedding_profiles.yml (секция tokenizer: "
                "{type: hf|tiktoken, name: ...}). Пример для BGE-M3: "
                "name: BAAI/bge-m3. Без этого HybridChunker не знает реальных границ."
            )

        # Если явный max_tokens не задан — tokenizer принесёт свой
        if max_tokens is not None:
            tokenizer.max_tokens = max_tokens

        return HybridChunker(tokenizer=tokenizer, merge_peers=True)

    def _extract_rich_meta(self, dc) -> ChunkMetadata:
        """Извлечь page_no, headings, element_type из chunk.meta DoclingDocument."""
        meta = ChunkMetadata(char_count=0)  # char_count проставим снаружи

        try:
            if not getattr(dc, "meta", None):
                return meta

            # Heading breadcrumbs
            if getattr(dc.meta, "headings", None):
                meta.section = " > ".join(str(h) for h in dc.meta.headings)

            # Первый page_no из provenance любого doc_item
            doc_items = getattr(dc.meta, "doc_items", None) or []
            labels: list[str] = []
            for di in doc_items:
                label = str(getattr(di, "label", "") or "")
                if label:
                    labels.append(label.lower())

                if meta.page is None:
                    prov = getattr(di, "prov", None) or []
                    for p in prov:
                        page_no = getattr(p, "page_no", None)
                        if page_no is not None:
                            meta.page = int(page_no)
                            break

            # Element type: самый специфичный из doc_items (table > code > formula > list > text)
            priority = ("table", "code", "formula", "equation", "list_item",
                        "figure", "picture", "caption", "footnote")
            chosen = "text"
            for candidate in priority:
                if candidate in labels:
                    chosen = _LABEL_TO_ELEMENT_TYPE.get(candidate, "text")
                    break
            meta.element_type = chosen

            # Fallback на meta.page (если prov не дал)
            if meta.page is None:
                page = getattr(dc.meta, "page", None)
                if page is not None:
                    meta.page = int(page)

        except Exception as e:
            log.debug("Ошибка извлечения метаданных чанка: {}", e)

        return meta

    def _chunk_hybrid(
        self, docling_document, document_id: str, include_headers: bool = True,
    ) -> list[Chunk]:
        """Docling HybridChunker с токенизатором активного провайдера.

        Итоговый текст чанка = chunker.contextualize(dc) если include_headers,
        иначе dc.text. contextualize() добавляет breadcrumbs заголовков
        перед текстом — это рекомендуемый формат для embedding.
        """
        log.info("Чанкинг (hybrid) документа {}", document_id)
        start = time.perf_counter()

        hybrid_chunker = self._build_hybrid_chunker(
            max_tokens=settings.chunk_max_tokens,
        )

        docling_chunks = list(hybrid_chunker.chunk(docling_document))

        chunks = []
        for dc in docling_chunks:
            # Контекстуализация: добавляет heading-breadcrumbs к тексту
            if include_headers:
                try:
                    text = hybrid_chunker.contextualize(chunk=dc)
                except Exception:
                    # В некоторых версиях метод называется serialize()
                    text = hybrid_chunker.serialize(dc)
            else:
                text = dc.text

            if not text or not text.strip():
                continue

            meta = self._extract_rich_meta(dc)
            meta.char_count = len(text)

            chunks.append(Chunk(
                document_id=document_id,
                text=text,
                metadata=meta,
            ))

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

