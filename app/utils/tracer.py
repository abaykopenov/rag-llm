"""
Trace — пошаговая трассировка запросов.

Каждый запрос получает trace_id и записывает каждый шаг
пайплайна: время, входные/выходные данные, системные метрики.

Хранит последние N трейсов в памяти для просмотра через API.
"""

import time
import uuid
import platform
from datetime import datetime, timezone
from typing import Optional
from collections import deque
from dataclasses import dataclass, field

from app.utils.logging import get_logger

log = get_logger("tracer")

# Максимум трейсов в памяти
MAX_TRACES = 100


@dataclass
class TraceStep:
    """Один шаг в трассировке."""
    step: str
    time_ms: float = 0.0
    timestamp: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class Trace:
    """Полная трассировка одного запроса."""
    trace_id: str = ""
    request_type: str = ""          # "query", "upload", "evaluate"
    input_preview: str = ""          # Первые символы вопроса/файла
    steps: list = field(default_factory=list)
    system_snapshot: dict = field(default_factory=dict)
    started_at: str = ""
    total_ms: float = 0.0
    status: str = "running"          # running, completed, error

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "request_type": self.request_type,
            "input_preview": self.input_preview,
            "steps": [
                {
                    "step": s.step,
                    "time_ms": round(s.time_ms, 1),
                    "timestamp": s.timestamp,
                    **s.details,
                }
                for s in self.steps
            ],
            "system": self.system_snapshot,
            "started_at": self.started_at,
            "total_ms": round(self.total_ms, 1),
            "status": self.status,
        }


class Tracer:
    """Трассировщик пайплайна — записывает каждый шаг."""

    def __init__(self):
        self._traces: deque[Trace] = deque(maxlen=MAX_TRACES)
        self._active: dict[str, tuple[Trace, float]] = {}  # trace_id -> (trace, start_time)

    def start_trace(self, request_type: str, input_preview: str = "") -> str:
        """Начать новую трассировку.

        Args:
            request_type: Тип запроса (query, upload, evaluate)
            input_preview: Превью входных данных (первые символы вопроса)

        Returns:
            trace_id
        """
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        trace = Trace(
            trace_id=trace_id,
            request_type=request_type,
            input_preview=input_preview[:100],
            started_at=now.isoformat(),
        )

        self._active[trace_id] = (trace, time.perf_counter())

        log.debug("Trace started: {} ({})", trace_id, request_type)
        return trace_id

    def add_step(
        self,
        trace_id: str,
        step_name: str,
        time_ms: float = 0.0,
        **details,
    ):
        """Добавить шаг к трассировке.

        Args:
            trace_id: ID трассировки
            step_name: Название шага (embed_query, retrieve_chunks, etc.)
            time_ms: Время выполнения в миллисекундах
            **details: Дополнительные данные (chunks_found, tokens, etc.)
        """
        if trace_id not in self._active:
            return

        trace, _ = self._active[trace_id]
        now = datetime.now(timezone.utc)

        step = TraceStep(
            step=step_name,
            time_ms=time_ms,
            timestamp=now.strftime("%H:%M:%S.%f")[:-3],
            details=details,
        )
        trace.steps.append(step)

    def end_trace(self, trace_id: str, status: str = "completed"):
        """Завершить трассировку.

        Args:
            trace_id: ID трассировки
            status: Статус (completed, error)
        """
        if trace_id not in self._active:
            return

        trace, start_time = self._active.pop(trace_id)
        trace.total_ms = (time.perf_counter() - start_time) * 1000
        trace.status = status
        trace.system_snapshot = self._get_system_snapshot()

        # Финальный шаг total
        trace.steps.append(TraceStep(
            step="total",
            time_ms=trace.total_ms,
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
        ))

        self._traces.append(trace)

        log.debug(
            "Trace completed: {} ({}) {:.0f}ms, {} steps",
            trace_id, trace.request_type, trace.total_ms, len(trace.steps),
        )

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Получить трассировку по ID."""
        # Сначала ищем в завершённых
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace.to_dict()

        # Потом в активных
        if trace_id in self._active:
            trace, _ = self._active[trace_id]
            return trace.to_dict()

        return None

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Получить последние трассировки."""
        traces = list(self._traces)
        traces.reverse()  # Новые первыми
        return [t.to_dict() for t in traces[:limit]]

    def get_stats(self) -> dict:
        """Общая статистика по трассировкам."""
        all_traces = list(self._traces)
        if not all_traces:
            return {
                "total_traces": 0,
                "active_traces": len(self._active),
            }

        times = [t.total_ms for t in all_traces if t.total_ms > 0]
        by_type: dict[str, list[float]] = {}
        for t in all_traces:
            by_type.setdefault(t.request_type, []).append(t.total_ms)

        return {
            "total_traces": len(all_traces),
            "active_traces": len(self._active),
            "avg_time_ms": round(sum(times) / len(times), 1) if times else 0,
            "max_time_ms": round(max(times), 1) if times else 0,
            "min_time_ms": round(min(times), 1) if times else 0,
            "by_type": {
                k: {
                    "count": len(v),
                    "avg_ms": round(sum(v) / len(v), 1),
                }
                for k, v in by_type.items()
            },
            "error_count": sum(1 for t in all_traces if t.status == "error"),
        }

    def _get_system_snapshot(self) -> dict:
        """Снимок состояния системы."""
        snapshot = {
            "platform": platform.system(),
            "python": platform.python_version(),
        }

        try:
            import psutil
            snapshot["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            snapshot["ram_total_gb"] = round(mem.total / (1024**3), 1)
            snapshot["ram_used_gb"] = round(mem.used / (1024**3), 1)
            snapshot["ram_percent"] = mem.percent
        except ImportError:
            pass

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snapshot["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
            snapshot["gpu_util_percent"] = util.gpu
            snapshot["gpu_vram_total_gb"] = round(mem_info.total / (1024**3), 1)
            snapshot["gpu_vram_used_gb"] = round(mem_info.used / (1024**3), 1)
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return snapshot


# Глобальный экземпляр
tracer = Tracer()
