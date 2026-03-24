# -*- coding: utf-8 -*-
"""Тест нового чанкера — с заголовками и типами элементов."""

import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DOC = """# Квартальный отчёт Q4 2024

## 1. Финансовые показатели

Выручка компании за Q4 2024 составила 15.2 млрд рублей, что на 23% выше
аналогичного периода прошлого года. Основной рост обеспечен сегментом
облачных сервисов (+45% г/г).

| Показатель | Q4 2023 | Q4 2024 | Изменение |
|---|---|---|---|
| Выручка | 12.4 млрд | 15.2 млрд | +23% |
| EBITDA | 3.1 млрд | 4.2 млрд | +35% |
| Чистая прибыль | 1.8 млрд | 2.5 млрд | +39% |
| Маржа EBITDA | 25% | 28% | +3 п.п. |

## 2. Технические метрики

### 2.1 Инфраструктура

Количество серверов увеличено до 12,000 (+30%).
Средний uptime составил 99.97%.

```python
# Пример мониторинга
def check_uptime(servers):
    total = len(servers)
    online = sum(1 for s in servers if s.is_online)
    return online / total * 100
```

### 2.2 ML Pipeline

Модели обучены на кластере из 64 GPU A100:
- **Latency p99**: 120ms → 85ms (-29%)
- **Throughput**: 1200 req/s → 2100 req/s (+75%)
- **Accuracy**: 94.2% → 96.8% (+2.6 п.п.)

Формула расчёта: $accuracy = TP + TN / (TP + TN + FP + FN)$

## 3. Планы на Q1 2025

1. Запуск новой embedding модели (1024d, multilingual)
2. Миграция на ChromaDB 2.0
3. Внедрение RAG pipeline для внутренней документации
4. Увеличение контекстного окна LLM до 128K токенов
"""


def main():
    from app.core.parser import DocumentParser
    from app.core.chunker import DocumentChunker

    # Парсим
    parser = DocumentParser()
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "report.md"
    path.write_text(DOC, encoding="utf-8")
    result = parser.parse(path)

    chunker_new = DocumentChunker()

    # ─── СРАВНЕНИЕ: с заголовками vs без ───
    print("=" * 70)
    print("  🆕 НОВЫЙ ЧАНКЕР: include_headers=True (заголовки в тексте)")
    print("=" * 70)

    chunks_with = chunker_new.chunk(result.docling_document, "doc-001", include_headers=True)

    total_chars_with = 0
    for i, c in enumerate(chunks_with):
        total_chars_with += len(c.text)
        section = c.metadata.section or "—"
        etype = c.metadata.element_type or "text"
        page = c.metadata.page or "—"

        print(f"\n  ┌─ Чанк #{i+1}  [{etype}]  page={page}")
        print(f"  │ Section: {section}")
        print(f"  │ Length: {c.metadata.char_count} символов")
        print(f"  │")

        lines = c.text.split("\n")
        for line in lines[:15]:
            print(f"  │  {line}")
        if len(lines) > 15:
            print(f"  │  ... ({len(lines) - 15} строк ещё)")
        print(f"  └{'─' * 55}")

    print(f"\n  📊 Суммарная длина: {total_chars_with} символов")
    print(f"  📊 Исходный текст:  {len(result.full_text)} символов")
    print(f"  📊 Покрытие: {total_chars_with / len(result.full_text) * 100:.1f}%")

    # ─── Без заголовков (старый режим) ───
    print(f"\n{'=' * 70}")
    print("  📦 СТАРЫЙ РЕЖИМ: include_headers=False (только тело)")
    print("=" * 70)

    chunks_without = chunker_new.chunk(result.docling_document, "doc-001", include_headers=False)

    total_chars_without = 0
    for i, c in enumerate(chunks_without):
        total_chars_without += len(c.text)
        first_line = c.text.split("\n")[0][:60]
        etype = c.metadata.element_type or "text"
        print(f"  [{etype:5s}] Чанк #{i+1}: \"{first_line}...\" ({c.metadata.char_count} сим)")

    # ─── Сравнение ───
    print(f"\n{'=' * 70}")
    print("  📊 СРАВНЕНИЕ")
    print("=" * 70)
    print(f"\n  {'':30s} │ С заголовками │ Без заголовков")
    print(f"  {'─'*30}─┼─{'─'*13}─┼─{'─'*14}")
    print(f"  {'Количество чанков':30s} │ {len(chunks_with):>13} │ {len(chunks_without):>14}")
    print(f"  {'Суммарная длина':30s} │ {total_chars_with:>13} │ {total_chars_without:>14}")
    print(f"  {'Покрытие текста':30s} │ {total_chars_with / len(result.full_text) * 100:>12.1f}% │ {total_chars_without / len(result.full_text) * 100:>13.1f}%")
    print(f"  {'Доп. символов (заголовки)':30s} │ {total_chars_with - total_chars_without:>+13} │ {'—':>14}")

    # ─── Что теперь получает LLM ───
    print(f"\n{'=' * 70}")
    print("  🤖 ЧТО ТЕПЕРЬ ПРИХОДИТ В LLM (пример чанка с таблицей)")
    print("=" * 70)

    for c in chunks_with:
        if c.metadata.element_type == "table":
            print(f"\n  Тип: {c.metadata.element_type}")
            print(f"  Секция: {c.metadata.section}")
            print(f"\n  ╔{'═' * 60}╗")
            for line in c.text.split("\n"):
                print(f"  ║ {line:<59}║")
            print(f"  ╚{'═' * 60}╝")
            break

    for c in chunks_with:
        if c.metadata.element_type == "code":
            print(f"\n  Тип: {c.metadata.element_type}")
            print(f"  Секция: {c.metadata.section}")
            print(f"\n  ╔{'═' * 60}╗")
            for line in c.text.split("\n"):
                print(f"  ║ {line:<59}║")
            print(f"  ╚{'═' * 60}╝")
            break


if __name__ == "__main__":
    main()
