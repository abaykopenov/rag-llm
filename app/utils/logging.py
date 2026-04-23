"""
Loguru logging configuration.

Два режима:
  - text (default): цветной, человекочитаемый, для dev
  - json: одна строка = одна JSON-запись, для Loki/ELK/Datadog

JSON-запись всегда содержит: ts, level, module, msg. Опционально:
  - trace_id (из app.utils.trace_context — автоматом пробрасывается в логи
    внутри обработчика запроса)
  - exception (текст traceback'а, если лог был внутри except: ...)
  - любые extra-поля, переданные через log.info("msg", key=value)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from app.utils.trace_context import get_trace_id

# ─────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────

_TEXT_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{extra[module]: <10}</cyan> | "
    "{message}"
)


def _json_sink(message) -> None:
    """Loguru sink: принимает отформатированное сообщение и пишет JSON в stderr.

    Использование: logger.add(_json_sink, format=_json_format_callable).
    """
    # message уже содержит JSON-строку (сформирована в _json_format_callable)
    print(message, end="", file=sys.stderr, flush=True)


def _json_format_callable(record) -> str:
    """Собрать JSON-строку из loguru record.

    Формат возвращает строку-шаблон для loguru — но нам нужна полная
    кастомизация, поэтому мы подменяем сообщение целиком через "{extra[_json]}".
    Loguru подставит наш заранее подготовленный JSON.
    """
    payload: dict[str, Any] = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["extra"].get("module") or record["name"],
        "msg": record["message"],
    }

    # trace_id из contextvar (если есть активный запрос)
    tid = get_trace_id()
    if tid:
        payload["trace_id"] = tid

    # Любые пользовательские extra-поля (кроме технических)
    for k, v in record["extra"].items():
        if k in ("module", "_json"):
            continue
        # Сериализуемые только — чтобы не падать на кастомных объектах
        try:
            json.dumps(v, ensure_ascii=False, default=str)
            payload[k] = v
        except (TypeError, ValueError):
            payload[k] = repr(v)

    # Exception traceback (если лог был внутри except) — нормализованный формат
    exc = record.get("exception")
    if exc:
        import traceback as tb_mod
        try:
            payload["exception"] = {
                "type": exc.type.__name__ if exc.type else None,
                "value": str(exc.value) if exc.value else None,
                "traceback": "".join(
                    tb_mod.format_tb(exc.traceback)
                ) if exc.traceback else None,
            }
        except Exception:
            payload["exception"] = repr(exc)

    record["extra"]["_json"] = json.dumps(
        payload, ensure_ascii=False, default=str,
    )
    # Возвращаем шаблон для loguru: просто вывести готовый JSON + перевод строки
    return "{extra[_json]}\n"


# ─────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────

def setup_logging(debug: bool = False) -> None:
    """Настройка логирования.

    Читает конфиг из app.config.settings. `debug=True` из main.py форсит
    DEBUG-уровень независимо от settings.log_level.
    """
    # Импортируем тут, а не на уровне модуля, чтобы избежать циклического
    # импорта (config нередко импортирует utils/logging при инициализации).
    from app.config import settings

    logger.remove()

    level = "DEBUG" if debug else settings.log_level.upper()
    fmt = settings.log_format.lower()

    if fmt == "json":
        logger.add(
            _json_sink,
            format=_json_format_callable,
            level=level,
            # loguru сам не делает цветную раскраску в callable-форматтере
        )
    else:
        logger.add(
            sys.stderr,
            format=_TEXT_FORMAT,
            level=level,
            colorize=True,
        )

    # Файловый лог — тот же формат, без colorize
    if settings.log_file_enabled:
        log_dir = Path(settings.log_file_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(log_dir / "rag-llm_{time:YYYY-MM-DD}.log")

        if fmt == "json":
            logger.add(
                file_path,
                format=_json_format_callable,
                level=level,
                rotation=settings.log_file_rotation,
                retention=settings.log_file_retention,
                encoding="utf-8",
            )
        else:
            logger.add(
                file_path,
                format=_TEXT_FORMAT,
                level=level,
                rotation=settings.log_file_rotation,
                retention=settings.log_file_retention,
                encoding="utf-8",
            )


def get_logger(module: str):
    """Получить логгер для модуля.

    Использование:
        log = get_logger("parser")
        log.info("Документ загружен: {}", filename)

    trace_id проставляется автоматически через contextvar — его не нужно
    передавать руками в каждый лог-вызов.
    """
    return logger.bind(module=module)
