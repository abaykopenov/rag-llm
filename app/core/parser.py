"""
Парсинг документов — два режима:
  1. gemini  — через Gemini API (облако, 0 RAM, отличное качество)
  2. docling — через Docling (локально, нужен PyTorch, ~2 ГБ RAM)

По умолчанию: gemini
"""

import asyncio
import base64
import time
from pathlib import Path
from typing import Optional

from app.config import settings
from app.utils.logging import get_logger

log = get_logger("parser")

# Поддерживаемые форматы файлов
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm",
    ".md", ".txt", ".csv", ".tsv",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}

# Форматы которые Gemini обрабатывает как документы (vision)
GEMINI_VISION_FORMATS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# MIME типы для Gemini File API
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
}


class ParseResult:
    """Результат парсинга документа."""

    def __init__(
        self,
        full_text: str,
        pages_count: int,
        parse_time_ms: float,
        parser_used: str = "unknown",
        docling_document=None,
    ):
        self.full_text = full_text
        self.pages_count = pages_count
        self.parse_time_ms = parse_time_ms
        self.parser_used = parser_used
        self.docling_document = docling_document


class DocumentParser:
    """Парсер документов с поддержкой нескольких бэкендов."""

    def __init__(self):
        self._docling_converter = None
        self._genai_client = None

    # ─────────────────────────────────────────────────
    # Gemini парсер (облако)
    # ─────────────────────────────────────────────────

    def _get_genai_client(self):
        """Lazy init клиента google-genai."""
        if self._genai_client is None:
            from google import genai
            api_key = settings.llm_api_key
            if not api_key:
                raise ValueError(
                    "Для Gemini-парсера нужен API ключ. "
                    "Установи RAG_LLM_API_KEY в .env "
                    "(https://aistudio.google.com/apikey)"
                )
            self._genai_client = genai.Client(api_key=api_key)
            log.info("Gemini клиент инициализирован")
        return self._genai_client

    def _parse_with_gemini(self, file_path: Path) -> ParseResult:
        """Парсинг через Gemini API.

        Для PDF: загружает файл через Files API, затем просит
        Gemini извлечь текст в формате Markdown с сохранением структуры.

        Для текстовых файлов: просто читает содержимое.
        """
        ext = file_path.suffix.lower()

        # Текстовые файлы читаем напрямую
        if ext in {".md", ".txt", ".csv", ".tsv", ".html", ".htm"}:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return ParseResult(
                full_text=text,
                pages_count=max(1, text.count("\n\n") // 2),
                parse_time_ms=0,
                parser_used="direct_read",
            )

        start = time.perf_counter()
        client = self._get_genai_client()
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        log.info("Gemini парсинг: {} ({:.1f} MB)", file_path.name, file_size_mb)

        # Загружаем файл через Files API
        mime_type = MIME_TYPES.get(ext, "application/octet-stream")
        uploaded_file = client.files.upload(
            file=file_path,
            config={"mime_type": mime_type},
        )

        log.info("Файл загружен в Gemini: {}", uploaded_file.name)

        # Промпт для извлечения текста
        prompt = (
            "Извлеки ВЕСЬ текст из этого документа в формате Markdown. "
            "Сохрани структуру: заголовки (# ## ###), списки, таблицы. "
            "Таблицы оформляй в формате Markdown таблиц. "
            "Не добавляй своих комментариев, только содержимое документа. "
            "Если есть формулы — используй LaTeX формат."
        )

        # Определяем модель (из конфига)
        model = settings.llm_model

        response = client.models.generate_content(
            model=model,
            contents=[uploaded_file, prompt],
        )

        full_text = response.text or ""
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Считаем примерное число страниц (258 токенов на страницу для PDF)
        pages_count = max(1, len(full_text) // 2000)

        # Удаляем загруженный файл из Gemini
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass  # Не критично, файл удалится автоматически через 48ч

        log.info(
            "Gemini парсинг завершён: ~{} стр, {} символов, {:.0f} мс",
            pages_count, len(full_text), elapsed_ms,
        )

        return ParseResult(
            full_text=full_text,
            pages_count=pages_count,
            parse_time_ms=elapsed_ms,
            parser_used="gemini",
        )

    # ─────────────────────────────────────────────────
    # Docling парсер (локальный)
    # ─────────────────────────────────────────────────

    def _get_docling_converter(self):
        """Lazy init конвертера Docling (тяжёлый импорт)."""
        if self._docling_converter is None:
            log.info("Инициализация Docling DocumentConverter...")
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()
            log.info("Docling готов")
        return self._docling_converter

    def _parse_with_docling(self, file_path: Path) -> ParseResult:
        """Парсинг через Docling (локально)."""
        start = time.perf_counter()

        converter = self._get_docling_converter()
        result = converter.convert(str(file_path))
        doc = result.document
        full_text = doc.export_to_markdown()

        pages_count = 0
        try:
            pages_count = doc.num_pages()
        except Exception:
            pages_count = full_text.count("\n\n") // 2 or 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        log.info(
            "Docling парсинг завершён: {} стр, {} символов, {:.0f} мс",
            pages_count, len(full_text), elapsed_ms,
        )

        return ParseResult(
            full_text=full_text,
            pages_count=pages_count,
            parse_time_ms=elapsed_ms,
            parser_used="docling",
            docling_document=doc,
        )

    # ─────────────────────────────────────────────────
    # Публичный интерфейс
    # ─────────────────────────────────────────────────

    @staticmethod
    def validate_file_extension(filename: str) -> None:
        """Проверить, что расширение файла поддерживается."""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Формат файла '{ext}' не поддерживается. "
                f"Допустимые форматы: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

    def parse(self, file_path: str | Path, mode: Optional[str] = None) -> ParseResult:
        """Распарсить документ.

        Args:
            file_path: Путь к файлу
            mode: Режим парсинга (gemini, docling). None = из конфига.

        Returns:
            ParseResult с текстом и метаданными
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        self.validate_file_extension(file_path.name)

        mode = mode or settings.parser_mode
        log.info("Парсинг: {} (mode={}, {:.1f} MB)",
                 file_path.name, mode, file_path.stat().st_size / (1024 * 1024))

        if mode == "gemini":
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return self._parse_with_gemini(file_path)
                except Exception as e:
                    error_str = str(e)
                    # Rate limit — ждём и повторяем
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        wait = 45 * (attempt + 1)  # 45s, 90s, 135s
                        log.warning(
                            "Gemini rate limit (попытка {}/{}). Ждём {} сек...",
                            attempt + 1, max_retries, wait
                        )
                        time.sleep(wait)
                        continue
                    # Другая ошибка — попробовать docling только если явно доступен
                    log.warning("Gemini парсинг не удался: {}", e)
                    try:
                        return self._parse_with_docling(file_path)
                    except Exception as docling_err:
                        log.error("Docling тоже не смог: {}", docling_err)
                        raise RuntimeError(
                            f"Парсинг не удался ни через Gemini ({e}), ни через Docling ({docling_err})"
                        )
            # Все попытки Gemini исчерпаны — rate limit
            raise RuntimeError(
                "Gemini API: дневной лимит исчерпан. Варианты:\n"
                "1. Подождите до сброса квоты (ежедневно)\n"
                "2. Создайте новый API ключ: https://aistudio.google.com/apikey\n"
                "3. Переключитесь на RAG_PARSER_MODE=docling (нужно 4+ ГБ RAM)"
            )
        elif mode == "docling":
            return self._parse_with_docling(file_path)
        elif mode == "vision":
            # Vision LLM парсинг — требует async
            raise NotImplementedError(
                "Vision LLM парсинг доступен только через parse_async(). "
                "Используйте: await parser.parse_async(file_path, mode='vision')"
            )
        else:
            raise ValueError(f"Неизвестный режим парсинга: {mode}. Допустимые: gemini, docling, vision")

    async def parse_async(self, file_path: str | Path, mode: Optional[str] = None) -> ParseResult:
        """Распарсить документ асинхронно."""
        mode = mode or settings.parser_mode

        # Vision LLM — нативно асинхронный
        if mode == "vision":
            return await self._parse_with_vision(file_path)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, file_path, mode)

    async def _parse_with_vision(self, file_path: str | Path) -> ParseResult:
        """Парсинг через Vision LLM (page-as-image)."""
        from app.core.vision_parser import vision_parser

        start = time.perf_counter()
        file_path = Path(file_path)

        try:
            full_text = await vision_parser.parse_pdf(str(file_path))
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Грубая оценка страниц по разделителям
            pages_count = full_text.count("<!-- Страница") or 1

            log.info(
                "Vision парсинг завершён: {} стр, {} символов, {:.0f} мс",
                pages_count, len(full_text), elapsed_ms,
            )

            return ParseResult(
                full_text=full_text,
                pages_count=pages_count,
                parse_time_ms=elapsed_ms,
                parser_used="vision",
            )
        except Exception as e:
            log.warning("Vision парсинг не удался: {}, пробуем Docling", e)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._parse_with_docling, file_path
            )


# Глобальный экземпляр парсера
parser = DocumentParser()
