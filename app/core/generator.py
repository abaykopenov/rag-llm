"""
Generator — сборка prompt и генерация ответа через LLM.
"""

import time

from app.config import settings
from app.core.llm_router import llm_router, LLMResponse
from app.models.document import RetrievedChunk
from app.utils.logging import get_logger

log = get_logger("generator")


class Generator:
    """Сборка prompt из контекста и генерация ответа."""

    def build_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """Собрать prompt для LLM.

        Args:
            query: Вопрос пользователя
            chunks: Найденные чанки
            history: Предыдущие сообщения [{"role": "user", "content": "..."}]

        Returns:
            Список messages для LLM API
        """
        # Формируем контекст из чанков
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = ""
            if chunk.metadata.section:
                source += f" | Раздел: {chunk.metadata.section}"
            if chunk.metadata.page:
                source += f" | Стр. {chunk.metadata.page}"

            context_parts.append(
                f"[Фрагмент {i}]{source}\n{chunk.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": settings.system_prompt,
            },
        ]

        # Добавляем историю диалога (если есть)
        if history:
            messages.extend(history)

        # Текущий вопрос с контекстом
        messages.append({
            "role": "user",
            "content": (
                f"Контекст из документов:\n\n{context}\n\n"
                f"---\n\n"
                f"Вопрос: {query}"
            ),
        })

        log.info(
            "Prompt собран: {} чанков, {} символов контекста, {} history msgs",
            len(chunks), len(context), len(history) if history else 0,
        )

        return messages

    async def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        temperature: float | None = None,
        max_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> "GenerationResult":
        """Сгенерировать ответ по вопросу с контекстом.

        Args:
            query: Вопрос пользователя
            chunks: Найденные чанки
            temperature: Температура генерации
            max_tokens: Максимум токенов
            history: История диалога для multi-turn

        Returns:
            GenerationResult с ответом и метаданными
        """
        start = time.perf_counter()

        # 1. Собираем prompt
        messages = self.build_prompt(query, chunks, history=history)

        # Сохраняем prompt для прозрачности
        full_prompt = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in messages
        )

        # 2. Отправляем в LLM
        llm_response: LLMResponse = await llm_router.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        total_ms = (time.perf_counter() - start) * 1000

        log.info("Генерация завершена: {:.0f} мс", total_ms)

        return GenerationResult(
            answer=llm_response.text,
            prompt=full_prompt,
            model=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
            cached_prompt_tokens=llm_response.cached_prompt_tokens,
            cache_creation_tokens=llm_response.cache_creation_tokens,
            generation_time_ms=total_ms,
            llm_time_ms=llm_response.generation_time_ms,
        )


class GenerationResult:
    """Результат генерации ответа."""

    def __init__(
        self,
        answer: str,
        prompt: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        generation_time_ms: float,
        llm_time_ms: float,
        cached_prompt_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ):
        self.answer = answer
        self.prompt = prompt
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.generation_time_ms = generation_time_ms
        self.llm_time_ms = llm_time_ms
        # Cache-метрики (0 если провайдер их не вернул или кеш не сработал).
        self.cached_prompt_tokens = cached_prompt_tokens
        self.cache_creation_tokens = cache_creation_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Доля токенов prompt'а, прочитанных из кеша (0.0–1.0)."""
        if not self.prompt_tokens:
            return 0.0
        return self.cached_prompt_tokens / self.prompt_tokens


# Глобальный экземпляр
generator = Generator()
