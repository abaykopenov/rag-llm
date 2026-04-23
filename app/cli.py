"""
CLI-утилиты RAG-LLM.

Запуск:
    python -m app.cli reindex --collection docs --to-profile vllm-bge
    python -m app.cli collections
    python -m app.cli profiles
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import settings
from app.core.embedding import create_provider, load_profiles
from app.core.indexer import indexer
from app.utils.logging import get_logger, setup_logging

log = get_logger("cli")


# ═══════════════════════════════════════════════════════════════════
# Команды
# ═══════════════════════════════════════════════════════════════════

def cmd_profiles(_args) -> int:
    """Показать список доступных embedding-профилей."""
    path = Path(settings.embedding_profiles_path)
    if not path.exists():
        print(f"Файл профилей не найден: {path}")
        return 1

    profiles = load_profiles(path)
    active = (settings.embedding_profile or "").strip() or "(не задан)"
    print(f"Активный профиль: {active}")
    print(f"Файл: {path}\n")

    for name, p in profiles.items():
        marker = " ← активный" if name == settings.embedding_profile else ""
        tok = p.tokenizer.name if p.tokenizer and p.tokenizer.name else "(нет)"
        print(f"  {name}{marker}")
        print(f"    provider: {p.provider}")
        print(f"    model:    {p.model}")
        print(f"    dim:      {p.dim}")
        print(f"    base_url: {p.base_url or '-'}")
        print(f"    tokenizer: {tok}")
        print()
    return 0


def cmd_collections(_args) -> int:
    """Показать список коллекций с паспортом провайдера."""
    collections = indexer.list_collections()
    if not collections:
        print("Коллекций нет.")
        return 0

    for c in collections:
        name = c["name"]
        count = c.get("chunks_count", 0)
        prov = c.get("embedding_provider")
        model = c.get("embedding_model")
        dim = c.get("embedding_dim")
        stamp = f"{prov}/{model}@dim={dim}" if prov else "(паспорт отсутствует — legacy)"
        print(f"  {name:30s}  chunks={count:<6d}  {stamp}")
    return 0


async def cmd_reindex(args) -> int:
    """Пересобрать embeddings для коллекции новым провайдером.

    План:
      1. Загружаем target-профиль, создаём провайдера.
      2. Читаем все чанки из исходной коллекции (текст + metadata).
      3. Гоним их через target.embed_texts() батчами.
      4. Пишем в новую коллекцию со штампом target-паспорта.
      5. По флагу --delete-source удаляем старую.
    """
    src = args.collection
    target_profile_name = args.to_profile
    target_name = args.target_name or f"{src}_{target_profile_name}"
    if target_name == src:
        print("--target-name не может совпадать с --collection", file=sys.stderr)
        return 2

    # 1. Загрузить target профиль
    profiles_path = Path(settings.embedding_profiles_path)
    if not profiles_path.exists():
        print(f"Файл профилей не найден: {profiles_path}", file=sys.stderr)
        return 1
    profiles = load_profiles(profiles_path)
    if target_profile_name not in profiles:
        print(
            f"Профиль '{target_profile_name}' не найден. "
            f"Доступные: {list(profiles)}",
            file=sys.stderr,
        )
        return 1

    target_profile = profiles[target_profile_name]
    target_provider = create_provider(target_profile)
    target_info = target_provider.info
    log.info("Target: profile='{}', {}", target_profile_name, target_info)

    # 2. Проверить, что исходная коллекция существует
    existing = {c["name"] for c in indexer.list_collections()}
    if src not in existing:
        print(f"Коллекция '{src}' не найдена. Существующие: {sorted(existing)}",
              file=sys.stderr)
        return 1
    if target_name in existing:
        print(f"Целевая коллекция '{target_name}' уже существует — "
              f"укажи другой --target-name или удали её вручную.",
              file=sys.stderr)
        return 1

    # 3. Подсчитать чанки
    src_coll = indexer.get_or_create_collection(src, check=False)
    total = src_coll.count()
    if total == 0:
        print(f"Коллекция '{src}' пустая — нечего переиндексировать.")
        return 0

    print(f"Переиндексация '{src}' → '{target_name}'")
    print(f"  Чанков:   {total}")
    print(f"  Target:   {target_info}")
    print(f"  Batch:    {args.batch_size}")
    if args.dry_run:
        print("  DRY-RUN: ничего не пишем.")
        return 0

    # 4. Итерируем, эмбедим, пишем
    buffer_ids: list[str] = []
    buffer_docs: list[str] = []
    buffer_meta: list[dict] = []
    written = 0
    failures = 0

    async def flush():
        nonlocal written, failures
        if not buffer_ids:
            return
        try:
            embeddings = await target_provider.embed_texts(buffer_docs)
        except Exception as e:
            failures += len(buffer_ids)
            log.error("Ошибка embedding батча ({} чанков): {}", len(buffer_ids), e)
            buffer_ids.clear()
            buffer_docs.clear()
            buffer_meta.clear()
            return

        indexer.add_raw(
            collection_name=target_name,
            ids=list(buffer_ids),
            documents=list(buffer_docs),
            embeddings=embeddings,
            metadatas=list(buffer_meta),
            expected_info=target_info,
        )
        written += len(buffer_ids)
        print(f"  ... {written}/{total} записано", flush=True)
        buffer_ids.clear()
        buffer_docs.clear()
        buffer_meta.clear()

    for chunk in indexer.iter_all_chunks(src):
        buffer_ids.append(chunk["id"])
        buffer_docs.append(chunk["text"])
        buffer_meta.append(chunk["metadata"] or {})
        if len(buffer_ids) >= args.batch_size:
            await flush()

    await flush()

    print(f"Готово: записано {written}/{total}, ошибок {failures}")

    # 5. Очистка (опционально)
    if args.delete_source and failures == 0 and written == total:
        print(f"Удаляю исходную коллекцию '{src}' (как просил --delete-source)...")
        indexer.delete_collection(src)
    elif args.delete_source:
        print("⚠ --delete-source проигнорирован: были ошибки или расхождение.",
              file=sys.stderr)

    await target_provider.close()
    return 0 if failures == 0 else 1


# ═══════════════════════════════════════════════════════════════════
# Точка входа
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Утилиты RAG-LLM (reindex, list collections, list profiles).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profiles", help="Список доступных embedding-профилей")
    sub.add_parser("collections", help="Список коллекций ChromaDB")

    r = sub.add_parser(
        "reindex",
        help="Пересобрать embeddings коллекции под новый профиль",
        description=(
            "Читает чанки из --collection (без embeddings), эмбедит их "
            "через --to-profile и пишет в новую коллекцию. "
            "Исходная коллекция НЕ трогается (если не задан --delete-source)."
        ),
    )
    r.add_argument("--collection", required=True, help="Исходная коллекция")
    r.add_argument("--to-profile", required=True, help="Имя профиля из YAML")
    r.add_argument(
        "--target-name",
        default=None,
        help="Имя новой коллекции (по умолчанию: {collection}_{profile})",
    )
    r.add_argument(
        "--batch-size", type=int, default=64,
        help="Размер батча при embedding (default: 64)",
    )
    r.add_argument(
        "--delete-source", action="store_true",
        help="Удалить исходную коллекцию после успеха (по умолчанию НЕ удаляется)",
    )
    r.add_argument(
        "--dry-run", action="store_true",
        help="Показать что будет сделано, но не выполнять",
    )
    return parser


def main() -> int:
    setup_logging(debug=settings.debug)
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "profiles":
        return cmd_profiles(args)
    if args.command == "collections":
        return cmd_collections(args)
    if args.command == "reindex":
        return asyncio.run(cmd_reindex(args))

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
