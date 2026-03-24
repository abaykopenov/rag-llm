# -*- coding: utf-8 -*-
"""
Сравнение: текущий serialize() vs Markdown формат.
Что лучше для LLM?
"""

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

## 3. Планы на Q1 2025

1. Запуск новой embedding модели (1024d, multilingual)
2. Миграция на ChromaDB 2.0
3. Внедрение RAG pipeline для внутренней документации
4. Увеличение контекстного окна LLM до 128K токенов
"""


def main():
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker

    # Парсим
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "report.md"
    path.write_text(DOC, encoding="utf-8")
    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    # === 1. Текущий подход: serialize() ===
    print("=" * 70)
    print("  ВАРИАНТ 1: serialize() (текущий)")
    print("  Что сейчас приходит в LLM")
    print("=" * 70)

    chunker = HybridChunker(max_tokens=512)
    chunks = list(chunker.chunk(doc))

    for i, c in enumerate(chunks):
        text = chunker.serialize(c)
        print(f"\n  ─── Чанк #{i+1} ({len(text)} сим) ───")
        for line in text.split("\n"):
            print(f"  │ {line}")

    # === 2. Markdown: export_to_markdown() ===
    print(f"\n\n{'=' * 70}")
    print("  ВАРИАНТ 2: export_to_markdown()")
    print("  Полный Markdown документа")
    print("=" * 70)

    md = doc.export_to_markdown()
    print(f"\n  Длина: {len(md)} символов")
    print()
    for line in md.split("\n"):
        print(f"  │ {line}")

    # === 3. Сравнение конкретно таблицы ===
    print(f"\n\n{'=' * 70}")
    print("  🔍 СРАВНЕНИЕ: как таблица выглядит в каждом варианте")
    print("=" * 70)

    print("\n  serialize() — плоский текст:")
    print("  ─" * 35)
    for c in chunks:
        text = chunker.serialize(c)
        if "12.4" in text and "15.2" in text:
            for line in text.split("\n"):
                if "12.4" in line or "3.1" in line or "1.8" in line or "25%" in line or "Выручка" in line:
                    print(f"  │ {line}")
            break

    print("\n  export_to_markdown() — Markdown таблица:")
    print("  ─" * 35)
    for line in md.split("\n"):
        if "|" in line and ("Показатель" in line or "---" in line or "12.4" in line
                           or "3.1" in line or "1.8" in line or "25%" in line):
            print(f"  │ {line}")

    # === 4. Markdown → чанкинг по секциям ===
    print(f"\n\n{'=' * 70}")
    print("  ВАРИАНТ 3: Markdown → разбивка по ## заголовкам")
    print("  (Более простой и понятный подход)")
    print("=" * 70)

    # Простой markdown chunker по заголовкам
    sections = []
    current_section = []
    current_header = ""

    for line in md.split("\n"):
        if line.startswith("## "):
            if current_section:
                sections.append({
                    "header": current_header,
                    "text": "\n".join(current_section).strip(),
                })
            current_header = line
            current_section = [line]
        elif line.startswith("# ") and not line.startswith("## "):
            current_header = line
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append({
            "header": current_header,
            "text": "\n".join(current_section).strip(),
        })

    for i, sec in enumerate(sections):
        print(f"\n  ─── Секция #{i+1}: {sec['header']} ({len(sec['text'])} сим) ───")
        lines = sec["text"].split("\n")
        for line in lines[:12]:
            print(f"  │ {line}")
        if len(lines) > 12:
            print(f"  │ ... ещё {len(lines) - 12} строк")

    # === Итоговое сравнение ===
    print(f"\n\n{'=' * 70}")
    print("  📊 ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 70)

    total_serialize = sum(len(chunker.serialize(c)) for c in chunks)
    total_md = len(md)
    total_md_chunks = sum(len(s["text"]) for s in sections)

    print(f"""
  ┌──────────────────────┬────────────┬────────────┬──────────────────┐
  │ Метрика              │ serialize()│ Markdown   │ MD по секциям    │
  ├──────────────────────┼────────────┼────────────┼──────────────────┤
  │ Общая длина          │ {total_serialize:>10} │ {total_md:>10} │ {total_md_chunks:>16} │
  │ Количество чанков    │ {len(chunks):>10} │ {'1 (весь)':>10} │ {len(sections):>16} │
  │ Таблицы              │ {'плоский':>10} │ {'| col |':>10} │ {'| col |':>16} │
  │ Заголовки            │ {'текст':>10} │ {'## / ###':>10} │ {'## / ###':>16} │
  │ Код                  │ {'```':>10} │ {'```python':>10} │ {'```python':>16} │
  │ Списки               │ {'- / 1.':>10} │ {'- / 1.':>10} │ {'- / 1.':>16} │
  │ LLM-friendly         │ {'⭐⭐⭐':>10} │ {'⭐⭐⭐⭐⭐':>10} │ {'⭐⭐⭐⭐⭐':>16} │
  └──────────────────────┴────────────┴────────────┴──────────────────┘
""")

    print("  💡 ВЫВОД:")
    print("  Markdown формат ЗНАЧИТЕЛЬНО лучше для LLM:")
    print("  • Таблица: '| Выручка | 15.2 млрд |' vs 'Выручка, Q4 2024 = 15.2 млрд'")
    print("  • Заголовки: '## 1. Финансы' vs просто '1. Финансы'")
    print("  • Код: '```python\\ndef...' vs '```\\ndef...'")
    print("  • LLM обучены на миллионах Markdown-файлов — это родной формат!")


if __name__ == "__main__":
    main()
