"""
Парсинг документов через Docling (IBM).
Извлекает текст, таблицы, структуру из PDF и других форматов.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

log = get_logger("parser")

# Поддерживаемые форматы файлов
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm",
    ".md", ".txt", ".csv", ".tsv",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


class DocumentParser:
    """Парсер документов на базе Docling."""

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        """Lazy init конвертера Docling (тяжёлый импорт)."""
        if self._converter is None:
            log.info("Инициализация Docling DocumentConverter...")
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
            log.info("Docling готов")
        return self._converter

    @staticmethod
    def validate_file_extension(filename: str) -> None:
        """Проверить, что расширение файла поддерживается.

        Args:
            filename: Имя файла для проверки

        Raises:
            ValueError: Если формат не поддерживается
        """
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Формат файла '{ext}' не поддерживается. "
                f"Допустимые форматы: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

    def parse(self, file_path: str | Path) -> "ParseResult":
        """Распарсить документ (синхронно).

        Args:
            file_path: Путь к файлу (PDF, DOCX, PPTX, HTML, и др.)

        Returns:
            ParseResult с извлечённым текстом и метаданными
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        self.validate_file_extension(file_path.name)

        log.info("Парсинг файла: {} ({:.1f} MB)", file_path.name, file_path.stat().st_size / (1024 * 1024))

        start = time.perf_counter()

        converter = self._get_converter()
        result = converter.convert(str(file_path))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Извлекаем docling document
        doc = result.document

        # Получаем полный текст в формате markdown
        full_text = doc.export_to_markdown()

        # Считаем страницы
        pages_count = 0
        try:
            pages_count = doc.num_pages()
        except Exception:
            # Некоторые форматы не имеют страниц
            pages_count = full_text.count("\n\n") // 2 or 1

        log.info(
            "Парсинг завершён: {} страниц, {} символов, {:.0f} мс",
            pages_count,
            len(full_text),
            elapsed_ms,
        )

        return ParseResult(
            docling_document=doc,
            full_text=full_text,
            pages_count=pages_count,
            parse_time_ms=elapsed_ms,
        )

    async def parse_async(self, file_path: str | Path) -> "ParseResult":
        """Распарсить документ асинхронно (не блокирует event loop).

        Оборачивает синхронный parse() в run_in_executor,
        чтобы тяжёлый Docling не блокировал FastAPI при параллельных запросах.

        Args:
            file_path: Путь к файлу

        Returns:
            ParseResult с извлечённым текстом и метаданными
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, file_path)


class ParseResult:
    """Результат парсинга документа."""

    def __init__(
        self,
        docling_document,
        full_text: str,
        pages_count: int,
        parse_time_ms: float,
    ):
        self.docling_document = docling_document  # Docling DoclingDocument объект
        self.full_text = full_text
        self.pages_count = pages_count
        self.parse_time_ms = parse_time_ms


# Глобальный экземпляр парсера
parser = DocumentParser()

