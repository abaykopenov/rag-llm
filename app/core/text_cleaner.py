"""
Пост-обработка текста после парсинга PDF/DOCX.

Решает типичные проблемы Docling и PDF-парсеров:
1. Повторяющиеся колонтитулы (headers/footers)
2. Номера страниц в тексте
3. Разорванные абзацы (одно предложение разбито на несколько строк)
4. <!-- image --> плейсхолдеры (оставляем один, убираем дубли)
5. Множественные пустые строки
"""

import re
from collections import Counter
from app.utils.logging import get_logger

log = get_logger("text_cleaner")


def clean_parsed_text(text: str) -> str:
    """Полная пост-обработка текста документа после парсинга.

    Применяется к полному тексту ДО чанкинга.
    """
    if not text or not text.strip():
        return text

    original_len = len(text)

    text = _remove_repeated_headers(text)
    text = _remove_page_numbers(text)
    text = _merge_broken_lines(text)
    text = _collapse_image_placeholders(text)
    text = _collapse_blank_lines(text)

    cleaned_len = len(text)
    removed = original_len - cleaned_len
    if removed > 0:
        log.info(
            "Текст очищен: {} → {} символов (−{}, −{:.1f}%)",
            original_len, cleaned_len, removed,
            removed / original_len * 100 if original_len else 0,
        )

    return text


def clean_chunk_text(text: str) -> str:
    """Лёгкая очистка текста отдельного чанка.

    Применяется к каждому чанку ПОСЛЕ чанкинга.
    """
    if not text or not text.strip():
        return text

    text = _remove_page_numbers(text)
    text = _collapse_image_placeholders(text)
    text = _merge_broken_lines(text)
    text = _collapse_blank_lines(text)

    return text.strip()


# ── Internal helpers ────────────────────────────────────────────────


def _remove_repeated_headers(text: str, min_repeats: int = 3) -> str:
    """Удаляет строки, которые повторяются ≥ min_repeats раз (колонтитулы).

    Типичный пример: заголовок документа повторяется на каждой странице.
    Пропускает короткие строки (<30 символов) — они могут быть легитимными.
    """
    lines = text.split("\n")
    # Считаем только длинные строки (>30 символов после strip)
    stripped = [ln.strip() for ln in lines]
    counts = Counter(s for s in stripped if len(s) >= 30)

    repeated = {s for s, n in counts.items() if n >= min_repeats}
    if not repeated:
        return text

    for r in repeated:
        log.debug("Удалён колонтитул ({} повторов): '{}'", counts[r], r[:60])

    result = []
    for ln, s in zip(lines, stripped):
        if s in repeated:
            continue
        result.append(ln)

    return "\n".join(result)


def _remove_page_numbers(text: str) -> str:
    """Удаляет автономные номера страниц (строки, содержащие только число).

    Пример: строка "4" или "  12  " между абзацами.
    """
    lines = text.split("\n")
    result = []
    for ln in lines:
        stripped = ln.strip()
        # Только числа 1-9999, стоящие отдельной строкой
        if stripped and re.fullmatch(r"\d{1,4}", stripped):
            continue
        result.append(ln)
    return "\n".join(result)


def _merge_broken_lines(text: str) -> str:
    """Склеивает строки, разорванные PDF-парсером.

    Определяет разрыв: строка A заканчивается на букву/запятую/дефис,
    а строка B начинается со строчной буквы или продолжает предложение.
    Не трогает markdown-заголовки, списки, блоки кода.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text

    result = [lines[0]]

    for i in range(1, len(lines)):
        prev = result[-1]
        curr = lines[i]
        prev_stripped = prev.rstrip()
        curr_stripped = curr.strip()

        # Не склеиваем, если:
        if not prev_stripped or not curr_stripped:
            result.append(curr)
            continue

        # - Markdown заголовки (## ...)
        if curr_stripped.startswith("#"):
            result.append(curr)
            continue

        # - Списки (- , * , 1. , ❼)
        if re.match(r"^(\s*[-*❼•]|\s*\d+[\.\)])\s", curr):
            result.append(curr)
            continue

        # - Блоки кода
        if curr_stripped.startswith("```"):
            result.append(curr)
            continue

        # - Таблицы markdown
        if curr_stripped.startswith("|"):
            result.append(curr)
            continue

        # Условие склейки: предыдущая строка обрывается "на полуслове"
        # (заканчивается на букву, запятую, дефис, НЕ на точку/двоеточие и т.д.)
        ends_mid_sentence = bool(
            re.search(r"[а-яёa-z,\-–—]$", prev_stripped, re.IGNORECASE)
        )
        # Текущая строка начинается со строчной буквы или продолжает предложение
        starts_continuation = bool(
            re.match(r"^[а-яёa-z]", curr_stripped)
        )

        if ends_mid_sentence and starts_continuation:
            # Склеиваем с пробелом
            result[-1] = prev_stripped + " " + curr_stripped
        else:
            result.append(curr)

    return "\n".join(result)


def _collapse_image_placeholders(text: str) -> str:
    """Сокращает повторяющиеся <!-- image --> до одного на блок."""
    # Заменяем несколько подряд <!-- image --> (с пустыми строками между) на один
    text = re.sub(
        r"(<!-- image -->\s*\n\s*)+<!-- image -->",
        "<!-- image -->",
        text,
    )
    return text


def _collapse_blank_lines(text: str) -> str:
    """Схлопывает 3+ пустых строки подряд в 2."""
    return re.sub(r"\n{4,}", "\n\n\n", text)
