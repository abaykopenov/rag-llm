"""
Утилиты безопасности: санитизация имён файлов, API-key auth, ограничение размера аплоада.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePath
from typing import Optional

from fastapi import Header, HTTPException, UploadFile, status

from app.config import settings
from app.utils.logging import get_logger

log = get_logger("security")

# Максимальная длина имени файла после санитизации (символов).
_MAX_FILENAME_LEN = 200


def sanitize_filename(filename: str) -> str:
    """Привести имя файла к безопасному виду.

    Гарантии:
      - Нет сепараторов путей (/, \\) — `PurePath(name).name` отрезает директории.
      - Нет "..", нулевых байт, control-chars.
      - Только буквы (в т.ч. кириллица), цифры, `.`, `-`, `_`, пробел, `()`.
      - Непустое имя (fallback: 'unnamed').
      - Длина ≤ _MAX_FILENAME_LEN (с сохранением расширения).
    """
    if not filename:
        return "unnamed"

    # 1. Берём только базовое имя — отбрасываем любые директории
    name = PurePath(filename).name

    # 2. Убираем null-байты и control-chars
    name = name.replace("\x00", "")
    name = "".join(ch for ch in name if ch.isprintable() or ch.isspace())

    # 3. Прячем попытки path traversal
    name = name.replace("..", "_")

    # 4. Оставляем только безопасные символы (unicode-letters + digits + . - _ пробел)
    name = re.sub(r"[^\w.\-\s()]", "_", name, flags=re.UNICODE)

    # 5. Тримим ведущие/замыкающие точки и пробелы (Windows issues)
    name = name.strip(". ")

    if not name:
        return "unnamed"

    # 6. Обрезаем длину с сохранением расширения
    if len(name) > _MAX_FILENAME_LEN:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) < 10:
            allowed = _MAX_FILENAME_LEN - len(ext) - 1
            name = stem[:allowed] + "." + ext
        else:
            name = name[:_MAX_FILENAME_LEN]

    return name


async def save_upload_with_size_limit(
    upload: UploadFile,
    dest: Path,
    max_bytes: int,
) -> int:
    """Сохранить UploadFile на диск, проверяя размер потоком.

    Читает чанками, прерывает запись и удаляет файл, если размер превышен.
    Так избегаем записи мегабайтов на диск перед проверкой.

    Args:
        upload: FastAPI UploadFile
        dest: Куда писать
        max_bytes: Лимит в байтах (0 или отрицательный = без лимита)

    Returns:
        Сколько байт записано

    Raises:
        HTTPException(413): Если размер превышен
    """
    bytes_written = 0
    chunk_size = 64 * 1024  # 64 KB

    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            bytes_written += len(chunk)
            if 0 < max_bytes < bytes_written:
                # Откатываем — не оставляем недозалитый файл
                try:
                    f.close()
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"Файл превышает лимит {max_bytes // (1024 * 1024)} МБ "
                        f"(RAG_MAX_UPLOAD_SIZE_MB)."
                    ),
                )
            f.write(chunk)

    return bytes_written


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency: валидация ключа из header X-API-Key.

    Если в конфиге `api_keys` пустой — пропускает всех (dev mode).
    Иначе ключ должен быть в множестве валидных.
    """
    valid = settings.api_keys_set
    if not valid:
        return  # auth отключена

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if x_api_key not in valid:
        log.warning("Отклонён запрос с невалидным API-ключом (префикс={}...)",
                    x_api_key[:6] if x_api_key else "")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
