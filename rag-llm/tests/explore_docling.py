# -*- coding: utf-8 -*-
"""Изучаем внутренности DoclingDocument — что доступно для чанкинга."""

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
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker

    # 1. Парсим
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "report.md"
    path.write_text(DOC, encoding="utf-8")

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    # 2. Изучаем DoclingDocument
    print("=" * 70)
    print("  📄 СТРУКТУРА DoclingDocument")
    print("=" * 70)

    print(f"\n  Тип: {type(doc).__name__}")
    print(f"\n  Доступные атрибуты:")
    for attr in sorted(dir(doc)):
        if not attr.startswith("_"):
            try:
                val = getattr(doc, attr)
                if not callable(val):
                    t = type(val).__name__
                    if isinstance(val, (str, int, float, bool)):
                        print(f"    .{attr} = {val!r}")
                    elif isinstance(val, (list, dict)):
                        print(f"    .{attr} ({t}, len={len(val)})")
                    else:
                        print(f"    .{attr} ({t})")
            except:
                pass

    # 3. Смотрим body items
    print(f"\n{'=' * 70}")
    print(f"  📝 BODY ITEMS (элементы документа)")
    print(f"{'=' * 70}")

    if hasattr(doc, 'body'):
        print(f"\n  Всего body items: {len(doc.body.children) if hasattr(doc.body, 'children') else '?'}")

    # Итерируем по items
    if hasattr(doc, 'iterate_items'):
        items = list(doc.iterate_items())
        print(f"  Всего iterate_items: {len(items)}")
        for i, (item, level) in enumerate(items):
            label = getattr(item, 'label', '?')
            text = ""
            if hasattr(item, 'text'):
                text = item.text[:80] if item.text else ""

            # Проверяем тип
            item_type = type(item).__name__

            prov = ""
            if hasattr(item, 'prov') and item.prov:
                for p in item.prov:
                    if hasattr(p, 'page_no'):
                        prov = f"page={p.page_no}"
                        break

            indent = "  " * level
            print(f"\n    {indent}[{i}] {item_type} label={label} {prov}")
            if text:
                print(f"    {indent}    text: \"{text}\"")

            # Для таблиц — показываем содержимое
            if 'table' in str(label).lower() or 'table' in item_type.lower():
                if hasattr(item, 'data'):
                    print(f"    {indent}    data: {type(item.data).__name__}")
                if hasattr(item, 'export_to_dataframe'):
                    try:
                        df = item.export_to_dataframe()
                        print(f"    {indent}    DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
                        print(f"    {indent}    Columns: {list(df.columns)}")
                        print(f"    {indent}    {df.to_string(index=False)[:200]}")
                    except:
                        pass
                if hasattr(item, 'export_to_markdown'):
                    try:
                        md = item.export_to_markdown()
                        print(f"    {indent}    Markdown table:")
                        for line in md.split("\n")[:6]:
                            print(f"    {indent}      {line}")
                    except:
                        pass

    # 4. Export форматы
    print(f"\n{'=' * 70}")
    print(f"  📤 ФОРМАТЫ ЭКСПОРТА")
    print(f"{'=' * 70}")

    for method in ['export_to_markdown', 'export_to_text', 'export_to_dict']:
        if hasattr(doc, method):
            try:
                content = getattr(doc, method)()
                if isinstance(content, str):
                    print(f"\n  .{method}() → {len(content)} символов")
                    print(f"    Первые 200 символов:")
                    print(f"    \"{content[:200]}\"")
                elif isinstance(content, dict):
                    print(f"\n  .{method}() → dict с ключами: {list(content.keys())[:10]}")
            except Exception as e:
                print(f"\n  .{method}() → ERROR: {e}")

    # 5. HybridChunker — что содержит каждый raw chunk
    print(f"\n{'=' * 70}")
    print(f"  ✂️  АНАЛИЗ RAW CHUNKS (HybridChunker)")
    print(f"{'=' * 70}")

    chunker = HybridChunker(max_tokens=512)
    raw_chunks = list(chunker.chunk(doc))

    print(f"\n  Всего чанков: {len(raw_chunks)}")

    for i, rc in enumerate(raw_chunks):
        print(f"\n  ┌─ Chunk #{i+1}")
        print(f"  │ type: {type(rc).__name__}")
        print(f"  │ text ({len(rc.text)} сим): \"{rc.text[:100]}...\"" if len(rc.text) > 100 else f"  │ text ({len(rc.text)} сим): \"{rc.text}\"")

        if hasattr(rc, 'meta') and rc.meta:
            print(f"  │ meta.headings: {rc.meta.headings}")
            if hasattr(rc.meta, 'doc_items'):
                for di in rc.meta.doc_items:
                    di_label = getattr(di, 'label', '?')
                    di_type = type(di).__name__
                    di_self_ref = getattr(di, 'self_ref', '?')
                    print(f"  │   doc_item: {di_type} label={di_label} ref={di_self_ref}")

        # Проверяем — есть ли enriched text (с заголовками)
        if hasattr(rc, 'enriched_text'):
            print(f"  │ enriched_text: \"{rc.enriched_text[:100]}...\"")

        print(f"  └{'─' * 50}")

    # 6. Попробуем HybridChunker с include_metadata
    print(f"\n{'=' * 70}")
    print(f"  🔧 CHUNKER PARAMS — что можно настроить")
    print(f"{'=' * 70}")

    import inspect
    sig = inspect.signature(HybridChunker.__init__)
    print(f"\n  HybridChunker.__init__ параметры:")
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
        print(f"    {name}: {param.annotation.__name__ if hasattr(param.annotation, '__name__') else param.annotation} = {default}")

    # Пробуем serialize с heading
    print(f"\n  Chunker.serialize() — с заголовками:")
    for i, rc in enumerate(raw_chunks[:3]):
        serialized = chunker.serialize(rc)
        print(f"\n  Chunk #{i+1} serialized ({len(serialized)} сим):")
        print(f"    \"{serialized[:150]}\"")

        # Сравниваем с rc.text
        if serialized != rc.text:
            print(f"    ⚡ serialized ≠ text! Разница: {len(serialized) - len(rc.text)} символов")
        else:
            print(f"    ✅ serialized == text")


if __name__ == "__main__":
    main()
