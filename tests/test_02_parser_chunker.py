"""
Тест 2: Парсер и Чанкер.
Проверяет Docling парсинг PDF и нарезку на чанки.
Требует: Docling (pip install docling)
"""

import asyncio
import tempfile
import shutil
from pathlib import Path

import pytest


def _create_test_txt_file(tmp_dir: str) -> Path:
    """Создать простой текстовый тестовый файл."""
    content = """# Тестовый документ для RAG-LLM

## Глава 1: Введение в RAG

Retrieval-Augmented Generation (RAG) — это метод, который расширяет возможности 
языковых моделей путём добавления внешних источников знаний. Вместо того чтобы 
полагаться исключительно на информацию, усвоенную во время обучения, модель 
получает доступ к актуальным документам для генерации более точных ответов.

Основные компоненты RAG-системы:
1. Парсинг документов — извлечение текста из PDF, DOCX и других форматов
2. Чанкинг — нарезка текста на фрагменты оптимального размера
3. Эмбеддинг — преобразование текста в числовые вектора
4. Индексация — сохранение векторов в специализированной базе данных
5. Поиск — нахождение наиболее релевантных фрагментов
6. Генерация — формирование ответа на основе найденного контекста

## Глава 2: Преимущества RAG

RAG решает ключевые проблемы LLM:
- Устаревание знаний — модель может использовать актуальную информацию
- Галлюцинации — ответы основаны на реальных документах
- Прозрачность — можно увидеть, какие источники использовались

## Глава 3: Архитектура

Типичная архитектура RAG состоит из двух фаз:

### Фаза индексации
Документы загружаются, парсятся, нарезаются на чанки, преобразуются 
в эмбеддинги и сохраняются в векторную базу данных.

### Фаза поиска и генерации
При получении вопроса, система находит наиболее релевантные чанки 
и передаёт их вместе с вопросом в языковую модель для генерации ответа.
"""
    file_path = Path(tmp_dir) / "test_document.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _create_test_html_file(tmp_dir: str) -> Path:
    """Создать HTML тестовый файл."""
    content = """<!DOCTYPE html>
<html>
<head><title>RAG-LLM Test</title></head>
<body>
<h1>Vector Databases</h1>
<p>Векторные базы данных хранят числовые представления (embeddings) текстовых данных 
и обеспечивают быстрый поиск по семантическому сходству.</p>

<h2>ChromaDB</h2>
<p>ChromaDB — это open-source векторная база данных, оптимизированная для 
AI-приложений. Она поддерживает различные метрики расстояния: 
cosine, euclidean, inner product.</p>

<h2>Преимущества ChromaDB</h2>
<ul>
<li>Простая интеграция с Python</li>
<li>Поддержка метаданных</li>
<li>Встроенный и client-server режимы</li>
<li>Автоматическое сохранение данных</li>
</ul>

<table>
<tr><th>Метрика</th><th>Описание</th></tr>
<tr><td>Cosine</td><td>Косинусное расстояние</td></tr>
<tr><td>L2</td><td>Евклидово расстояние</td></tr>
</table>
</body>
</html>"""
    file_path = Path(tmp_dir) / "test_vectors.html"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# =================== Тесты Парсера ===================

def test_file_extension_validation():
    """Проверить валидацию расширений файлов."""
    from app.core.parser import DocumentParser, ALLOWED_EXTENSIONS

    parser = DocumentParser()

    # Допустимые
    for ext in [".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".png"]:
        parser.validate_file_extension(f"test{ext}")  # Не должно бросить исключение

    print(f"  ✅ Допустимые форматы ({len(ALLOWED_EXTENSIONS)} шт.) прошли валидацию")

    # Недопустимые
    for ext in [".exe", ".zip", ".py", ".js", ".dll"]:
        try:
            parser.validate_file_extension(f"test{ext}")
            assert False, f"Должен был бросить ValueError для {ext}"
        except ValueError as e:
            assert "не поддерживается" in str(e)

    print(f"  ✅ Недопустимые форматы (.exe, .zip, .py, .js, .dll) отклонены")


def test_parse_markdown_file():
    """Проверить парсинг Markdown файла через Docling."""
    from app.core.parser import DocumentParser

    parser = DocumentParser()
    tmp_dir = tempfile.mkdtemp()

    try:
        file_path = _create_test_txt_file(tmp_dir)

        result = parser.parse(file_path)

        assert result.full_text is not None
        assert len(result.full_text) > 100
        assert result.parse_time_ms > 0
        assert result.pages_count >= 0  # Markdown не имеет "страниц"
        assert result.docling_document is not None

        print(f"  ✅ Markdown парсинг: {len(result.full_text)} символов, "
              f"{result.pages_count} стр., {result.parse_time_ms:.0f} мс")
        print(f"     Первые 100 символов: {result.full_text[:100]}...")

    finally:
        shutil.rmtree(tmp_dir)


def test_parse_html_file():
    """Проверить парсинг HTML файла через Docling."""
    from app.core.parser import DocumentParser

    parser = DocumentParser()
    tmp_dir = tempfile.mkdtemp()

    try:
        file_path = _create_test_html_file(tmp_dir)

        result = parser.parse(file_path)

        assert result.full_text is not None
        assert len(result.full_text) > 50
        assert "ChromaDB" in result.full_text or "chromadb" in result.full_text.lower()
        assert result.docling_document is not None

        print(f"  ✅ HTML парсинг: {len(result.full_text)} символов, "
              f"{result.pages_count} стр., {result.parse_time_ms:.0f} мс")

    finally:
        shutil.rmtree(tmp_dir)


def test_parse_async():
    """Проверить асинхронный парсинг."""
    from app.core.parser import DocumentParser

    parser = DocumentParser()
    tmp_dir = tempfile.mkdtemp()

    try:
        file_path = _create_test_txt_file(tmp_dir)

        result = asyncio.run(parser.parse_async(file_path))

        assert result.full_text is not None
        assert len(result.full_text) > 100
        print(f"  ✅ Async парсинг: {len(result.full_text)} символов, {result.parse_time_ms:.0f} мс")

    finally:
        shutil.rmtree(tmp_dir)


def test_parse_nonexistent_file():
    """Проверить ошибку при несуществующем файле."""
    from app.core.parser import DocumentParser

    parser = DocumentParser()

    try:
        parser.parse("/nonexistent/path/file.pdf")
        assert False, "Должен бросить FileNotFoundError"
    except FileNotFoundError:
        print(f"  ✅ FileNotFoundError для несуществующего файла")


def test_parse_invalid_extension():
    """Проверить ошибку при неподдерживаемом формате."""
    from app.core.parser import DocumentParser

    parser = DocumentParser()
    tmp_dir = tempfile.mkdtemp()

    try:
        bad_file = Path(tmp_dir) / "script.exe"
        bad_file.write_text("not a document", encoding="utf-8")

        try:
            parser.parse(bad_file)
            assert False, "Должен бросить ValueError"
        except ValueError as e:
            assert "не поддерживается" in str(e)
            print(f"  ✅ ValueError для .exe: {e}")

    finally:
        shutil.rmtree(tmp_dir)


# =================== Тесты Чанкера ===================

def test_chunker():
    """Проверить нарезку документа на чанки."""
    from app.core.parser import DocumentParser
    from app.core.chunker import DocumentChunker

    parser = DocumentParser()
    chunker = DocumentChunker()
    tmp_dir = tempfile.mkdtemp()

    try:
        file_path = _create_test_txt_file(tmp_dir)
        parse_result = parser.parse(file_path)

        chunks = chunker.chunk(parse_result.docling_document, "test-doc-id-123")

        assert len(chunks) > 0
        assert all(c.text.strip() for c in chunks)
        assert all(c.document_id == "test-doc-id-123" for c in chunks)
        assert all(c.metadata.char_count > 0 for c in chunks)
        assert all(c.id for c in chunks)  # UUID

        avg_len = sum(c.metadata.char_count for c in chunks) // len(chunks)
        total_chars = sum(c.metadata.char_count for c in chunks)

        print(f"  ✅ Чанкинг: {len(chunks)} чанков")
        print(f"     Средняя длина: {avg_len} символов")
        print(f"     Всего символов: {total_chars}")
        print(f"     Пример чанка #1 ({chunks[0].metadata.char_count} симв.):")
        print(f"       \"{chunks[0].text[:80]}...\"")

        # Проверяем что чанки не пустые и уникальные
        texts = set(c.text for c in chunks)
        assert len(texts) == len(chunks), "Есть дублирующиеся чанки!"
        print(f"  ✅ Все {len(chunks)} чанков уникальны")

    finally:
        shutil.rmtree(tmp_dir)


def test_chunker_with_html():
    """Проверить чанкинг HTML документа."""
    from app.core.parser import DocumentParser
    from app.core.chunker import DocumentChunker

    parser = DocumentParser()
    chunker = DocumentChunker()
    tmp_dir = tempfile.mkdtemp()

    try:
        file_path = _create_test_html_file(tmp_dir)
        parse_result = parser.parse(file_path)

        chunks = chunker.chunk(parse_result.docling_document, "html-doc-id")

        assert len(chunks) > 0
        print(f"  ✅ HTML чанкинг: {len(chunks)} чанков из HTML файла")

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    print("\n🧪 Тест 2: Парсер и Чанкер\n" + "=" * 50)

    print("\n--- Валидация расширений ---")
    test_file_extension_validation()

    print("\n--- Парсинг Markdown ---")
    test_parse_markdown_file()

    print("\n--- Парсинг HTML ---")
    test_parse_html_file()

    print("\n--- Async парсинг ---")
    test_parse_async()

    print("\n--- Ошибки парсинга ---")
    test_parse_nonexistent_file()
    test_parse_invalid_extension()

    print("\n--- Чанкинг Markdown ---")
    test_chunker()

    print("\n--- Чанкинг HTML ---")
    test_chunker_with_html()

    print("\n✅ Все тесты пройдены!\n")
