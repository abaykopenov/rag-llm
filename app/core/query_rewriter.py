"""
Query Rewriter — улучшение поисковых запросов через LLM.

Два режима:
1. rewrite() — переформулирует вопрос для лучшего поиска
2. generate_multi_queries() — генерирует 3 варианта запроса
"""

from app.core.llm_router import llm_router
from app.utils.logging import get_logger

log = get_logger("query_rewriter")

REWRITE_SYSTEM_PROMPT = """Ты — помощник для улучшения поисковых запросов в базе знаний.

Твоя задача: переформулировать вопрос пользователя так, чтобы он стал более конкретным и подходящим для поиска в документах.

Правила:
- Сохрани смысл вопроса
- Сделай его более конкретным и развёрнутым
- Если вопрос уже конкретный — верни его без изменений
- Если есть контекст из истории диалога — учти его
- Ответь ТОЛЬКО переформулированным вопросом, без пояснений"""

MULTI_QUERY_SYSTEM_PROMPT = """Ты — помощник для генерации поисковых запросов.

Задача: на основе вопроса пользователя сгенерируй ровно 3 разных варианта этого вопроса для поиска в базе знаний.

Правила:
- Каждый вариант должен искать ту же информацию, но другими словами
- Используй синонимы, перефразирование, разный уровень детализации
- Один вариант — более общий, один — более конкретный, один — с другой формулировкой
- Верни ровно 3 строки, каждый запрос на отдельной строке
- Без нумерации, без пояснений, только запросы"""


class QueryRewriter:
    """Улучшение поисковых запросов через LLM."""

    async def rewrite(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> str:
        """Переформулировать вопрос для более точного поиска.

        Args:
            query: Оригинальный вопрос
            history: История диалога для контекста

        Returns:
            Улучшенный вопрос
        """
        # Короткие и конкретные запросы не трогаем
        if len(query.split()) >= 8:
            log.debug("Запрос уже достаточно длинный, пропускаем rewrite")
            return query

        messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]

        # Добавляем контекст из истории
        if history:
            context = "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in history[-4:]
            )
            messages.append({
                "role": "user",
                "content": f"Контекст диалога:\n{context}\n\nПереформулируй этот вопрос: {query}",
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Переформулируй этот вопрос для поиска: {query}",
            })

        try:
            response = await llm_router.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=150,
            )
            rewritten = response.text.strip().strip('"').strip("'")

            if rewritten and len(rewritten) > 5:
                log.info("Query rewrite: '{}' → '{}'", query[:50], rewritten[:80])
                return rewritten
            return query

        except Exception as e:
            log.warning("Query rewrite failed: {}, using original", e)
            return query

    async def generate_multi_queries(self, query: str, n: int = 3) -> list[str]:
        """Сгенерировать несколько вариантов запроса для Multi-Query Retrieval.

        Args:
            query: Оригинальный вопрос
            n: Количество вариантов (по умолчанию 3)

        Returns:
            Список вариантов запроса (включая оригинальный)
        """
        messages = [
            {"role": "system", "content": MULTI_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        try:
            response = await llm_router.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )

            lines = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.text.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ][:n]

            # Всегда включаем оригинальный запрос
            queries = [query] + [q for q in lines if q != query]
            log.info("Multi-query: '{}' → {} вариантов", query[:50], len(queries))
            return queries

        except Exception as e:
            log.warning("Multi-query generation failed: {}", e)
            return [query]


def rrf_merge(results_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion для объединения результатов нескольких запросов.

    Args:
        results_lists: Список списков результатов от разных запросов
        k: Параметр RRF (стандарт: 60)

    Returns:
        Объединённый отсортированный список
    """
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for results in results_lists:
        for rank, item in enumerate(results):
            chunk_id = item["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = item

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        item = chunk_data[chunk_id].copy()
        item["score"] = round(rrf_scores[chunk_id], 6)
        merged.append(item)

    return merged


# Глобальный экземпляр
query_rewriter = QueryRewriter()
