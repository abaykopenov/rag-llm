"""
🧪 RAG-LLM Test Runner
Запускает все тесты последовательно с красивым выводом.

Использование:
  python tests/run_all.py
"""

import sys
import time
import traceback


def run_test(name: str, func):
    """Запустить тест с красивым выводом."""
    print(f"\n{'═' * 60}")
    print(f"  🧪 {name}")
    print(f"{'═' * 60}")

    start = time.perf_counter()
    try:
        func()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n  ✅ PASSED ({elapsed:.0f} мс)")
        return True
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n  ❌ FAILED ({elapsed:.0f} мс)")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + " RAG-LLM Test Suite".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    results = []
    total_start = time.perf_counter()

    # ============ Test 1: Модели данных ============
    from tests.test_01_models import (
        test_document_model_creation,
        test_chunk_model_creation,
        test_document_store_save_and_get,
        test_document_store_collection_operations,
    )

    results.append(("Document model", run_test("Document model", test_document_model_creation)))
    results.append(("Chunk model", run_test("Chunk model", test_chunk_model_creation)))
    results.append(("DocumentStore CRUD", run_test("DocumentStore CRUD", test_document_store_save_and_get)))
    results.append(("DocumentStore collections", run_test("DocumentStore collections", test_document_store_collection_operations)))

    # ============ Test 2: Парсер и Чанкер ============
    from tests.test_02_parser_chunker import (
        test_file_extension_validation,
        test_parse_markdown_file,
        test_parse_html_file,
        test_parse_async,
        test_parse_nonexistent_file,
        test_parse_invalid_extension,
        test_chunker,
        test_chunker_with_html,
    )

    results.append(("File extension validation", run_test("File extension validation", test_file_extension_validation)))
    results.append(("Parse Markdown", run_test("Parse Markdown", test_parse_markdown_file)))
    results.append(("Parse HTML", run_test("Parse HTML", test_parse_html_file)))
    results.append(("Parse async", run_test("Parse async", test_parse_async)))
    results.append(("Parse nonexistent file", run_test("Parse nonexistent file", test_parse_nonexistent_file)))
    results.append(("Parse invalid extension", run_test("Parse invalid extension", test_parse_invalid_extension)))
    results.append(("Chunker (Markdown)", run_test("Chunker (Markdown)", test_chunker)))
    results.append(("Chunker (HTML)", run_test("Chunker (HTML)", test_chunker_with_html)))

    # ============ Test 3: Indexer ============
    from tests.test_03_indexer import (
        test_indexer_add_and_query,
        test_indexer_get_chunks_by_document,
        test_indexer_get_chunk_by_id,
        test_indexer_list_and_delete_collections,
    )

    results.append(("Indexer add & query", run_test("Indexer add & query", test_indexer_add_and_query)))
    results.append(("Indexer get_chunks_by_document", run_test("Indexer get_chunks_by_document", test_indexer_get_chunks_by_document)))
    results.append(("Indexer get_chunk_by_id", run_test("Indexer get_chunk_by_id", test_indexer_get_chunk_by_id)))
    results.append(("Indexer collections", run_test("Indexer collections", test_indexer_list_and_delete_collections)))

    # ============ Test 4: Full Pipeline ============
    from tests.test_04_pipeline import test_full_pipeline_parse_chunk_index_search

    results.append(("FULL PIPELINE", run_test("FULL PIPELINE", test_full_pipeline_parse_chunk_index_search)))

    # ============ Итоги ============
    total_ms = (time.perf_counter() - total_start) * 1000
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " РЕЗУЛЬТАТЫ".center(58) + "║")
    print("╠" + "═" * 58 + "╣")

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        line = f"  {status}  {name}"
        print(f"║{line:<58}║")

    print("╠" + "═" * 58 + "╣")
    summary = f"  {passed} passed, {failed} failed | {total_ms:.0f} мс"
    print(f"║{summary:<58}║")
    print("╚" + "═" * 58 + "╝")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
