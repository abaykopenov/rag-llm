"""
Auto-summarizer — генерация краткого содержания при загрузке документа.

При загрузке документа автоматически:
1. Берёт первые N символов документа (или весь, если маленький)
2. Отправляет в LLM с промптом на суммаризацию
3. Сохраняет summary в DocumentStore

Работает асинхронно — не блокирует загрузку.
"""

import time
from typing import Optional

from app.core.llm_router import llm_router
from app.utils.logging import get_logger

log = get_logger("summarizer")

# Максимум символов документа для суммаризации
# (чтобы не прокидывать огромный текст в LLM)
MAX_TEXT_CHARS = 8000


class Summarizer:
    """Генератор краткого содержания документов."""

    async def summarize(
        self,
        text: str,
        filename: str = "",
        max_text_chars: int = MAX_TEXT_CHARS,
    ) -> str:
        """Сгенерировать краткое содержание документа.

        Args:
            text: Полный текст документа
            filename: Имя файла (для контекста)
            max_text_chars: Максимум символов для отправки в LLM

        Returns:
            Краткое содержание (2-5 предложений)
        """
        if not text or not text.strip():
            return ""

        start = time.perf_counter()

        # Обрезаем текст если слишком длинный
        if len(text) > max_text_chars:
            truncated_text = text[:max_text_chars] + "\n\n[... текст обрезан ...]"
        else:
            truncated_text = text

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — ассистент для суммаризации документов. "
                    "Создай краткое содержание документа на 3-5 предложений. "
                    "Укажи: тему документа, ключевые разделы, основные выводы. "
                    "Пиши на том же языке, что и документ. "
                    "Не добавляй от себя — только из текста."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Документ: {filename}\n\n"
                    f"Текст:\n{truncated_text}\n\n"
                    f"Краткое содержание:"
                ),
            },
        ]

        try:
            response = await llm_router.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            summary = response.text.strip()

            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info(
                "Суммаризация '{}': {} символов текста → {} символов summary, {:.0f} мс",
                filename, len(text), len(summary), elapsed_ms,
            )

            return summary

        except Exception as e:
            log.error("Ошибка суммаризации '{}': {}", filename, e)
            return ""

    async def summarize_chunks(
        self,
        chunks_texts: list[str],
        filename: str = "",
    ) -> str:
        """Суммаризация на базе чанков (альтернатива — если текст не сохранён).

        Берёт первые несколько чанков и суммаризирует.
        """
        if not chunks_texts:
            return ""

        # Берём первые чанки до лимита
        combined = ""
        for chunk_text in chunks_texts:
            if len(combined) + len(chunk_text) > MAX_TEXT_CHARS:
                break
            combined += chunk_text + "\n\n---\n\n"

        return await self.summarize(combined, filename)


# Глобальный экземпляр
summarizer = Summarizer()
