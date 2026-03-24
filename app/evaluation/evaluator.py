"""
RAG Evaluation — оценка качества RAG-пайплайна.

Реализует упрощённые метрики в стиле RAGAS:
- Faithfulness: ответ основан на контексте?
- Answer Relevancy: ответ релевантен вопросу?
- Context Precision: найдены лучшие чанки?
- Context Recall: найдены все нужные чанки?

Не требует внешних зависимостей (ragas, datasets).
Использует LLM для оценки через промпты.
"""

import time
from typing import Optional
from dataclasses import dataclass, field

from app.core.llm_router import llm_router
from app.utils.logging import get_logger

log = get_logger("evaluator")


@dataclass
class EvalSample:
    """Один пример для оценки."""
    question: str
    answer: str  # Ответ RAG
    contexts: list[str]  # Найденные чанки
    ground_truth: Optional[str] = None  # Эталонный ответ (если есть)


@dataclass
class EvalResult:
    """Результат оценки одного примера."""
    faithfulness: float = 0.0  # 0-1: ответ основан на контексте
    answer_relevancy: float = 0.0  # 0-1: ответ релевантен вопросу
    context_precision: float = 0.0  # 0-1: контекст точный
    context_recall: float = 0.0  # 0-1: контекст полный (если есть ground_truth)
    overall: float = 0.0  # Средняя
    details: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """Отчёт по набору примеров."""
    samples_count: int = 0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_overall: float = 0.0
    results: list[EvalResult] = field(default_factory=list)
    eval_time_ms: float = 0.0


class RAGEvaluator:
    """Оценщик качества RAG-пайплайна."""

    async def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Оценить один пример.

        Args:
            sample: Вопрос + ответ + контексты + (опционально) эталон

        Returns:
            EvalResult с метриками 0-1
        """
        result = EvalResult()

        # 1. Faithfulness: ответ основан на контексте?
        result.faithfulness = await self._eval_faithfulness(
            sample.answer, sample.contexts
        )

        # 2. Answer Relevancy: ответ релевантен вопросу?
        result.answer_relevancy = await self._eval_answer_relevancy(
            sample.question, sample.answer
        )

        # 3. Context Precision: контексты содержат ответ?
        result.context_precision = await self._eval_context_precision(
            sample.question, sample.contexts
        )

        # 4. Context Recall: все ли нужные части есть? (если есть эталон)
        if sample.ground_truth:
            result.context_recall = await self._eval_context_recall(
                sample.ground_truth, sample.contexts
            )
        else:
            result.context_recall = result.context_precision  # Fallback

        # Overall
        scores = [result.faithfulness, result.answer_relevancy,
                  result.context_precision, result.context_recall]
        result.overall = sum(scores) / len(scores)

        return result

    async def evaluate_batch(self, samples: list[EvalSample]) -> EvalReport:
        """Оценить набор примеров.

        Args:
            samples: Список примеров для оценки

        Returns:
            EvalReport с агрегированными метриками
        """
        start = time.perf_counter()
        report = EvalReport(samples_count=len(samples))

        for i, sample in enumerate(samples):
            log.info("Оценка {}/{}: '{}'", i + 1, len(samples), sample.question[:50])
            try:
                result = await self.evaluate_sample(sample)
                report.results.append(result)
            except Exception as e:
                log.error("Ошибка оценки примера {}: {}", i + 1, e)
                report.results.append(EvalResult())

        # Агрегация
        if report.results:
            n = len(report.results)
            report.avg_faithfulness = sum(r.faithfulness for r in report.results) / n
            report.avg_answer_relevancy = sum(r.answer_relevancy for r in report.results) / n
            report.avg_context_precision = sum(r.context_precision for r in report.results) / n
            report.avg_context_recall = sum(r.context_recall for r in report.results) / n
            report.avg_overall = sum(r.overall for r in report.results) / n

        report.eval_time_ms = (time.perf_counter() - start) * 1000

        log.info(
            "Оценка завершена: {} примеров | "
            "faithfulness={:.2f} | relevancy={:.2f} | precision={:.2f} | recall={:.2f} | "
            "overall={:.2f} | {:.0f}мс",
            report.samples_count,
            report.avg_faithfulness,
            report.avg_answer_relevancy,
            report.avg_context_precision,
            report.avg_context_recall,
            report.avg_overall,
            report.eval_time_ms,
        )

        return report

    # ═══════════════════════════════════════════════════════
    # Метрики
    # ═══════════════════════════════════════════════════════

    async def _eval_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Faithfulness: все ли утверждения в ответе основаны на контексте?

        1.0 = все утверждения подтверждены контекстом
        0.0 = ответ выдуман (hallucination)
        """
        if not contexts or not answer.strip():
            return 0.0

        context = "\n---\n".join(contexts)

        prompt = [
            {
                "role": "system",
                "content": (
                    "Ты — строгий оценщик качества ответов. "
                    "Оцени, насколько ответ основан ТОЛЬКО на предоставленном контексте. "
                    "Отвечай ТОЛЬКО одним числом от 0.0 до 1.0 без пояснений."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Контекст:\n{context}\n\n"
                    f"Ответ:\n{answer}\n\n"
                    f"Какая доля утверждений в ответе подтверждена контекстом? "
                    f"Ответь числом от 0.0 до 1.0:"
                ),
            },
        ]

        return await self._get_score(prompt)

    async def _eval_answer_relevancy(self, question: str, answer: str) -> float:
        """Answer Relevancy: отвечает ли ответ на вопрос?

        1.0 = ответ полностью релевантен вопросу
        0.0 = ответ не по теме
        """
        if not answer.strip():
            return 0.0

        prompt = [
            {
                "role": "system",
                "content": (
                    "Ты — строгий оценщик релевантности. "
                    "Оцени, насколько ответ относится к вопросу. "
                    "Отвечай ТОЛЬКО одним числом от 0.0 до 1.0 без пояснений."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос:\n{question}\n\n"
                    f"Ответ:\n{answer}\n\n"
                    f"Насколько ответ релевантен вопросу? "
                    f"Ответь числом от 0.0 до 1.0:"
                ),
            },
        ]

        return await self._get_score(prompt)

    async def _eval_context_precision(self, question: str, contexts: list[str]) -> float:
        """Context Precision: содержат ли найденные чанки ответ на вопрос?

        1.0 = все чанки релевантны вопросу
        0.0 = чанки не содержат нужной информации
        """
        if not contexts:
            return 0.0

        context = "\n---\n".join(contexts)

        prompt = [
            {
                "role": "system",
                "content": (
                    "Ты — строгий оценщик качества поиска. "
                    "Оцени, содержат ли найденные фрагменты информацию для ответа на вопрос. "
                    "Отвечай ТОЛЬКО одним числом от 0.0 до 1.0 без пояснений."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос:\n{question}\n\n"
                    f"Найденные фрагменты:\n{context}\n\n"
                    f"Какая доля фрагментов содержит информацию для ответа? "
                    f"Ответь числом от 0.0 до 1.0:"
                ),
            },
        ]

        return await self._get_score(prompt)

    async def _eval_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """Context Recall: все ли части эталонного ответа есть в контексте?

        1.0 = контекст покрывает весь эталонный ответ
        0.0 = ничего из эталона не найдено
        """
        if not contexts or not ground_truth:
            return 0.0

        context = "\n---\n".join(contexts)

        prompt = [
            {
                "role": "system",
                "content": (
                    "Ты — строгий оценщик полноты поиска. "
                    "Оцени, покрывают ли найденные фрагменты информацию из эталонного ответа. "
                    "Отвечай ТОЛЬКО одним числом от 0.0 до 1.0 без пояснений."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Эталонный ответ:\n{ground_truth}\n\n"
                    f"Найденные фрагменты:\n{context}\n\n"
                    f"Какая доля информации из эталона покрыта фрагментами? "
                    f"Ответь числом от 0.0 до 1.0:"
                ),
            },
        ]

        return await self._get_score(prompt)

    async def _get_score(self, messages: list[dict]) -> float:
        """Получить числовой score от LLM."""
        try:
            response = await llm_router.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )

            text = response.text.strip()

            # Извлекаем число из ответа
            import re
            match = re.search(r'(0\.\d+|1\.0|0|1)', text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))

            log.warning("LLM вернул не число: '{}'", text)
            return 0.5  # Default если не удалось распарсить

        except Exception as e:
            log.error("Ошибка получения score: {}", e)
            return 0.5


# Глобальный экземпляр
evaluator = RAGEvaluator()
