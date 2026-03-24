"""
Loguru logging configuration.
Структурное логирование каждого этапа пайплайна.
"""

import sys
from loguru import logger
from pathlib import Path


def setup_logging(debug: bool = False) -> None:
    """Настройка логирования."""

    # Убираем дефолтный handler
    logger.remove()

    # Формат лога
    log_format = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{extra[module]: <10}</cyan> | "
        "{message}"
    )

    # Консольный вывод
    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if debug else "INFO",
        colorize=True,
    )

    # Файловый лог (ротация каждые 10 MB)
    log_dir = Path("./data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_dir / "rag-llm_{time:YYYY-MM-DD}.log"),
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )


def get_logger(module: str):
    """Получить логгер для модуля.

    Использование:
        log = get_logger("parser")
        log.info("Документ загружен: {}", filename)
    """
    return logger.bind(module=module)
