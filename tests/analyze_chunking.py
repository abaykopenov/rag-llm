# -*- coding: utf-8 -*-
"""
🔬 Глубокий анализ качества чанкинга Docling HybridChunker.

Проверяет:
1. Как разбивается текст на чанки — по каким границам
2. Сохраняются ли заголовки, списки, таблицы, код
3. Какие метаданные Docling извлекает (section, page)
4. Влияние max_tokens на качество (256 vs 512 vs 1024)
5. Потеряны ли части текста при чанкинге
6. Качество разбиения для разных типов контента
"""

import sys
import time
import tempfile
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")


# ══════════════════════════════════════════════════════════
# Тестовые документы
# ══════════════════════════════════════════════════════════

DOCS = {}

DOCS["scientific"] = """# Нейронные Сети в Обработке Естественного Языка

## 1. Введение

Обработка естественного языка (NLP) — это область искусственного интеллекта,
которая занимается взаимодействием между компьютерами и человеческим языком.
За последние годы нейронные сети произвели революцию в NLP, значительно улучшив
качество машинного перевода, генерации текста и анализа тональности.

Ключевым прорывом стала архитектура Transformer, предложенная в статье
"Attention Is All You Need" (Vaswani et al., 2017). Transformer полностью
заменил рекуррентные сети (RNN/LSTM) в задачах NLP благодаря механизму
самовнимания (self-attention), который позволяет модели учитывать
взаимосвязи между всеми словами в предложении одновременно.

## 2. Архитектура Transformer

### 2.1 Механизм самовнимания

Самовнимание вычисляется по формуле:
$$Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V$$

Где Q (Query), K (Key), V (Value) — это линейные проекции входных
эмбеддингов. Параметр d_k — размерность ключей.

Многоголовочное внимание (Multi-Head Attention) применяет несколько
независимых слоёв внимания параллельно, что позволяет модели
захватывать различные типы зависимостей в тексте.

### 2.2 Позиционное кодирование

Поскольку Transformer не обрабатывает последовательность рекурсивно,
ему нужен способ учитывать позицию токенов. Для этого используются
позиционные эмбеддинги — синусоидальные функции разных частот,
которые добавляются к входным эмбеддингам.

### 2.3 Энкодер и Декодер

Transformer состоит из стека энкодеров и декодеров:
- **Энкодер** обрабатывает входную последовательность и создаёт контекстные представления
- **Декодер** генерирует выходную последовательность, обращаясь к представлениям энкодера
- Каждый слой содержит: Multi-Head Attention, Layer Normalization, Feed-Forward Network

## 3. Сравнение моделей

| Модель | Параметры | Контекст | Год | Тип |
|--------|-----------|----------|-----|-----|
| BERT | 340M | 512 | 2018 | Энкодер |
| GPT-2 | 1.5B | 1024 | 2019 | Декодер |
| T5 | 11B | 512 | 2019 | Энк-Дек |
| GPT-3 | 175B | 2048 | 2020 | Декодер |
| GPT-4 | ~1.8T | 128K | 2023 | Декодер |
| Llama 3 | 405B | 128K | 2024 | Декодер |

## 4. Практическое применение

### 4.1 RAG (Retrieval-Augmented Generation)

RAG объединяет поиск по базе знаний с генерацией текста:

1. Пользователь задаёт вопрос
2. Вопрос преобразуется в вектор через embedding модель
3. В векторной БД находятся ближайшие фрагменты документов
4. Найденные фрагменты и вопрос подаются в LLM
5. LLM генерирует ответ, опираясь на реальные данные

Преимущества RAG:
- Актуальная информация без переобучения
- Снижение галлюцинаций
- Прозрачность источников

### 4.2 Fine-tuning

Fine-tuning — это дообучение предобученной модели на специфических данных.
Методы: LoRA, QLoRA, полное дообучение. LoRA добавляет небольшие адаптеры
к весам модели, что требует значительно меньше памяти GPU.

## 5. Заключение

Нейронные сети, и в особенности архитектура Transformer, радикально
изменили область NLP. Современные LLM способны понимать контекст,
генерировать связный текст и решать сложные языковые задачи.
Дальнейшее развитие идёт в направлении увеличения эффективности,
мультимодальности и специализации моделей.
"""

DOCS["mixed_content"] = """# Руководство по развёртыванию vLLM

## Системные требования

Минимальные требования для запуска vLLM:
- GPU: NVIDIA с поддержкой CUDA 11.8+ (минимум RTX 3060 12GB)
- RAM: 32GB
- CPU: 8 ядер
- Disk: 100GB SSD (для весов моделей)

## Установка

```bash
pip install vllm
# или для разработки:
git clone https://github.com/vllm-project/vllm
cd vllm
pip install -e .
```

## Запуск сервера

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Что такое RAG?"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

## Конфигурация Docker

```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --tensor-parallel-size 1
      --max-model-len 4096
      --gpu-memory-utilization 0.9
```

## Метрики производительности

| Метрика | Значение | Описание |
|---------|----------|----------|
| Throughput | 30-50 tok/s | Скорость генерации |
| TTFT | 50-200ms | Time to First Token |
| Latency p99 | <500ms | 99-й перцентиль задержки |
| GPU Util | 85-95% | Загрузка GPU |

## Частые ошибки

**CUDA Out of Memory:**
Решение: уменьшить `--max-model-len` или включить `--quantization awq`

**Timeout при загрузке модели:**
Решение: увеличить `--download-dir` на быстрый SSD и проверить скорость сети

**Низкий throughput:**
Решение: включить `--enable-chunked-prefill` и увеличить `--max-num-batched-tokens`
"""

DOCS["short_paragraphs"] = """# FAQ

## Что такое embedding?
Embedding — это числовое представление текста в виде вектора фиксированной размерности.

## Зачем нужны вектора?
Вектора позволяют измерять семантическое сходство между текстами через расстояние.

## Что такое cosine similarity?
Косинусное сходство — мера похожести двух векторов: cos(θ) = (A·B) / (|A|·|B|).

## Какую размерность embedding использовать?
- 384d — быстро, достаточно для простых задач
- 768d — баланс скорость/качество
- 1024d — высокое качество для сложных задач
- 4096d — максимальное качество, но медленно

## Какой chunk size выбрать?
Маленькие чанки (128-256 токенов) — точный поиск, но теряется контекст.
Большие чанки (512-1024 токенов) — больше контекста, но менее точный поиск.
Оптимум обычно 256-512 токенов.
"""


# ══════════════════════════════════════════════════════════
# Утилиты анализа
# ══════════════════════════════════════════════════════════

def parse_document(text: str, filename: str = "test.md"):
    """Парсинг через Docling."""
    from app.core.parser import DocumentParser
    parser = DocumentParser()
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / filename
    path.write_text(text, encoding="utf-8")
    result = parser.parse(path)
    return result


def chunk_with_settings(docling_doc, doc_id: str, max_tokens: int):
    """Чанкинг с конкретным max_tokens."""
    from docling.chunking import HybridChunker
    from app.models.document import Chunk, ChunkMetadata

    chunker = HybridChunker(max_tokens=max_tokens)
    docling_chunks = list(chunker.chunk(docling_doc))

    chunks = []
    raw_chunks = []
    for dc in docling_chunks:
        text = dc.text
        if not text or not text.strip():
            continue

        meta = ChunkMetadata(char_count=len(text))
        try:
            if hasattr(dc, "meta") and dc.meta:
                if hasattr(dc.meta, "headings") and dc.meta.headings:
                    meta.section = " > ".join(dc.meta.headings)
                if hasattr(dc.meta, "page") and dc.meta.page is not None:
                    meta.page = dc.meta.page
        except Exception:
            pass

        chunks.append(Chunk(document_id=doc_id, text=text, metadata=meta))
        raw_chunks.append(dc)

    return chunks, raw_chunks


def print_separator(char="─", length=80):
    print(char * length)


def print_header(title):
    print(f"\n{'═' * 80}")
    print(f"  {title}")
    print(f"{'═' * 80}")


# ══════════════════════════════════════════════════════════
# Анализ 1: Детальный разбор чанков
# ══════════════════════════════════════════════════════════

def analyze_chunk_details(doc_name: str, text: str):
    """Подробный анализ каждого чанка."""
    print_header(f"📋 ДЕТАЛЬНЫЙ РАЗБОР: {doc_name}")

    parse_result = parse_document(text)
    chunks, raw_chunks = chunk_with_settings(parse_result.docling_document, "analysis", 512)

    print(f"\n  Исходный текст: {len(text)} символов")
    print(f"  После парсинга: {len(parse_result.full_text)} символов")
    print(f"  Количество чанков: {len(chunks)}")
    print(f"  max_tokens: 512")

    total_chunk_chars = sum(len(c.text) for c in chunks)
    coverage = total_chunk_chars / len(parse_result.full_text) * 100 if parse_result.full_text else 0

    print(f"  Суммарная длина чанков: {total_chunk_chars} символов")
    print(f"  Покрытие текста: {coverage:.1f}%")

    print(f"\n{'─' * 80}")
    for i, chunk in enumerate(chunks):
        section = chunk.metadata.section or "—"
        text_preview = chunk.text.replace("\n", " ↵ ")

        print(f"\n  ┌─ Чанк #{i+1}")
        print(f"  │ Section: {section}")
        print(f"  │ Length:  {chunk.metadata.char_count} символов (~{chunk.metadata.char_count // 4} токенов)")
        print(f"  │ Page:    {chunk.metadata.page or '—'}")

        # Показываем первые и последние строки
        lines = chunk.text.strip().split("\n")
        if len(lines) <= 6:
            for line in lines:
                print(f"  │   {line}")
        else:
            for line in lines[:3]:
                print(f"  │   {line}")
            print(f"  │   ... ({len(lines) - 6} строк пропущено) ...")
            for line in lines[-3:]:
                print(f"  │   {line}")

        print(f"  └{'─' * 60}")

    return chunks, parse_result


# ══════════════════════════════════════════════════════════
# Анализ 2: Влияние max_tokens
# ══════════════════════════════════════════════════════════

def analyze_max_tokens_impact(doc_name: str, text: str):
    """Как max_tokens влияет на количество и размер чанков."""
    print_header(f"📐 ВЛИЯНИЕ max_tokens: {doc_name}")

    parse_result = parse_document(text)

    configs = [128, 256, 512, 1024]
    results = {}

    print(f"\n  {'max_tokens':>10} │ {'Чанков':>6} │ {'Avg (сим)':>9} │ {'Min':>5} │ {'Max':>5} │ {'Медиана':>7} │ {'Покрытие':>8}")
    print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*8}")

    for max_t in configs:
        chunks, _ = chunk_with_settings(parse_result.docling_document, "test", max_t)

        if not chunks:
            print(f"  {max_t:>10} │ {'0':>6} │ {'—':>9} │ {'—':>5} │ {'—':>5} │ {'—':>7} │ {'—':>8}")
            continue

        lengths = sorted([c.metadata.char_count for c in chunks])
        avg_len = sum(lengths) // len(lengths)
        min_len = lengths[0]
        max_len = lengths[-1]
        median_len = lengths[len(lengths) // 2]
        total = sum(lengths)
        coverage = total / len(parse_result.full_text) * 100 if parse_result.full_text else 0

        results[max_t] = {
            "count": len(chunks),
            "avg": avg_len,
            "min": min_len,
            "max": max_len,
            "median": median_len,
            "coverage": coverage,
            "chunks": chunks,
        }

        print(f"  {max_t:>10} │ {len(chunks):>6} │ {avg_len:>9} │ {min_len:>5} │ {max_len:>5} │ {median_len:>7} │ {coverage:>7.1f}%")

    return results


# ══════════════════════════════════════════════════════════
# Анализ 3: Потери текста
# ══════════════════════════════════════════════════════════

def analyze_text_loss(doc_name: str, text: str):
    """Какой текст теряется при чанкинге?"""
    print_header(f"🔍 АНАЛИЗ ПОТЕРЬ: {doc_name}")

    parse_result = parse_document(text)
    chunks, _ = chunk_with_settings(parse_result.docling_document, "test", 512)

    original = parse_result.full_text
    reconstructed = "\n".join(c.text for c in chunks)

    # Разбиваем на слова для сравнения
    original_words = set(original.lower().split())
    chunk_words = set(reconstructed.lower().split())

    missing_words = original_words - chunk_words
    extra_words = chunk_words - original_words

    print(f"\n  Исходных слов: {len(original_words)}")
    print(f"  Слов в чанках: {len(chunk_words)}")
    print(f"  Потеряно слов: {len(missing_words)}")
    print(f"  Лишних слов:   {len(extra_words)}")
    print(f"  Процент сохранения: {len(chunk_words & original_words) / len(original_words) * 100:.1f}%")

    if missing_words:
        # Показываем что потеряли (не стоп-слова)
        significant_missing = [w for w in sorted(missing_words) if len(w) > 3][:20]
        if significant_missing:
            print(f"\n  Потерянные значимые слова (первые 20):")
            print(f"  {', '.join(significant_missing)}")

    # Проверяем конкретные элементы
    print(f"\n  Проверка ключевых элементов:")

    checks = {
        "Заголовки (## ...)": any("##" in word for word in original.split("\n") if word.strip().startswith("##")),
        "Списки (- ...)": "- " in original,
        "Нумерация (1. ...)": "1. " in original,
        "Таблицы (| ... |)": "|" in original,
        "Код (```)": "```" in original,
        "Формулы ($$)": "$$" in original,
        "Жирный (**...**)": "**" in original,
    }

    for element, present_in_original in checks.items():
        if present_in_original:
            present_in_chunks = False
            elem_marker = element.split("(")[1].split(")")[0].split(" ")[0]
            for c in chunks:
                if elem_marker.strip("`*$-").lower() in c.text.lower() or elem_marker in c.text:
                    present_in_chunks = True
                    break

            status = "✅ сохранён" if present_in_chunks else "⚠️  потенциально утерян"
            print(f"    {element}: {status}")
        else:
            print(f"    {element}: — (нет в оригинале)")


# ══════════════════════════════════════════════════════════
# Анализ 4: Качество границ
# ══════════════════════════════════════════════════════════

def analyze_chunk_boundaries(doc_name: str, text: str):
    """Насколько хорошо Docling выбирает границы чанков."""
    print_header(f"✂️  ГРАНИЦЫ ЧАНКОВ: {doc_name}")

    parse_result = parse_document(text)
    chunks, _ = chunk_with_settings(parse_result.docling_document, "test", 512)

    good_starts = 0
    good_ends = 0
    mid_sentence_cuts = 0

    print()
    for i, chunk in enumerate(chunks):
        text_stripped = chunk.text.strip()
        first_line = text_stripped.split("\n")[0][:60]
        last_line = text_stripped.split("\n")[-1][-60:]

        # Хорошее начало: с заголовка, с начала предложения (заглавная), с маркера списка
        good_start = (
            text_stripped[0].isupper()
            or text_stripped.startswith(("#", "-", "*", "1", "|", "```", "$$"))
            or text_stripped.startswith(("**",))
        )
        if good_start:
            good_starts += 1

        # Хороший конец: точка, вопросительный/восклицательный знак, закрытие кода
        good_end = text_stripped[-1] in ".!?:;)" or text_stripped.endswith(("```", "$$", "|"))
        if good_end:
            good_ends += 1

        # Разрез посреди предложения
        if not good_start and i > 0:
            mid_sentence_cuts += 1

        start_icon = "✅" if good_start else "⚠️"
        end_icon = "✅" if good_end else "⚠️"

        print(f"  Чанк #{i+1}:")
        print(f"    {start_icon} Начало: \"{first_line}...\"")
        print(f"    {end_icon} Конец:  \"...{last_line}\"")

    print(f"\n  Итого:")
    print(f"    Хорошее начало: {good_starts}/{len(chunks)} ({good_starts/len(chunks)*100:.0f}%)")
    print(f"    Хороший конец:  {good_ends}/{len(chunks)} ({good_ends/len(chunks)*100:.0f}%)")
    print(f"    Разрезы посреди предложения: {mid_sentence_cuts}")


# ══════════════════════════════════════════════════════════
# Анализ 5: Сохранение секций
# ══════════════════════════════════════════════════════════

def analyze_section_preservation(doc_name: str, text: str):
    """Как Docling сохраняет секции/разделы при чанкинге."""
    print_header(f"📑 СЕКЦИИ И МЕТАДАННЫЕ: {doc_name}")

    parse_result = parse_document(text)
    chunks, raw_chunks = chunk_with_settings(parse_result.docling_document, "test", 512)

    print(f"\n  Количество чанков: {len(chunks)}")

    # Группируем чанки по секциям
    section_map = Counter()
    for c in chunks:
        section = c.metadata.section or "(без секции)"
        section_map[section] += 1

    print(f"\n  Распределение по секциям:")
    print(f"  {'─' * 60}")
    for section, count in section_map.most_common():
        bar = "█" * count
        print(f"    {section[:50]:<50} │ {count} │ {bar}")

    # Проверяем raw chunk метаданные
    print(f"\n  Raw Docling chunk атрибуты (первый чанк):")
    if raw_chunks:
        dc = raw_chunks[0]
        for attr in ['text', 'meta', 'path']:
            if hasattr(dc, attr):
                val = getattr(dc, attr)
                if attr == 'text':
                    val = val[:80] + "..." if len(val) > 80 else val
                print(f"    .{attr} = {val}")

        if hasattr(dc, 'meta') and dc.meta:
            meta = dc.meta
            for attr in dir(meta):
                if not attr.startswith('_'):
                    try:
                        val = getattr(meta, attr)
                        if not callable(val):
                            print(f"    .meta.{attr} = {val}")
                    except:
                        pass


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║          🔬 ГЛУБОКИЙ АНАЛИЗ КАЧЕСТВА ЧАНКИНГА DOCLING                       ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    doc_pairs = [
        ("Научная статья (NLP/Transformers)", "scientific"),
        ("Техническая документация (vLLM)", "mixed_content"),
        ("Короткие параграфы (FAQ)", "short_paragraphs"),
    ]

    for name, key in doc_pairs:
        text = DOCS[key]

        # 1. Детальный разбор
        chunks, parse_result = analyze_chunk_details(name, text)

        # 2. Границы чанков
        analyze_chunk_boundaries(name, text)

        # 3. Потери текста
        analyze_text_loss(name, text)

        # 4. Секции и метаданные
        analyze_section_preservation(name, text)

    # 5. Сравнение max_tokens по всем документам
    print_header("📊 СВОДКА: ВЛИЯНИЕ max_tokens НА ВСЕ ДОКУМЕНТЫ")
    for name, key in doc_pairs:
        text = DOCS[key]
        print(f"\n  📄 {name}:")
        analyze_max_tokens_impact(name, text)

    # Финальная оценка
    print_header("🏆 ИТОГОВАЯ ОЦЕНКА DOCLING HybridChunker")
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │ Критерий                     │ Оценка │ Комментарий         │
  ├──────────────────────────────┼────────┼─────────────────────┤
  │ Сохранение структуры         │  ???   │ Выводы по секциям   │
  │ Качество границ              │  ???   │ Начало/конец чанков │
  │ Полнота текста               │  ???   │ % потерь            │
  │ Метаданные (section/page)    │  ???   │ Есть ли данные      │
  │ Обработка таблиц             │  ???   │ Цельность таблиц    │
  │ Обработка кода               │  ???   │ Цельность блоков    │
  └──────────────────────────────┴────────┴─────────────────────┘

  (Оценки заполнятся после анализа результатов выше)
    """)


if __name__ == "__main__":
    main()
