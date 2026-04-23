"""
Contextvar для trace_id, чтобы коррелировать логи и трейсер внутри запроса.

Логер достаёт текущий trace_id из этой переменной и добавляет его в JSON-запись
(если включён log_format=json). Тресер выставляет его при start_trace() и
очищает в end_trace(). Это позволяет грепать логи одного запроса без ручной
прокидки trace_id через все функции.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

# Не типизируем через Generic — хочется видеть None в дефолте
_current_trace_id: ContextVar[Optional[str]] = ContextVar(
    "trace_id", default=None,
)


def set_trace_id(trace_id: Optional[str]) -> object:
    """Установить trace_id для текущего async-контекста.

    Возвращает token, через который можно откатить обратно (reset_trace_id).
    """
    return _current_trace_id.set(trace_id)


def get_trace_id() -> Optional[str]:
    """Текущий trace_id или None, если не установлен."""
    return _current_trace_id.get()


def reset_trace_id(token: object) -> None:
    """Откатить значение (обычно вызывается в finally)."""
    try:
        _current_trace_id.reset(token)  # type: ignore[arg-type]
    except (ValueError, LookupError):
        # token невалидный — просто игнорим
        pass


def clear_trace_id() -> None:
    """Сбросить в None без token'а."""
    _current_trace_id.set(None)
