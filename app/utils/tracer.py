"""
Trace — пошаговая трассировка запросов с persistence в SQLite.

Каждый запрос получает trace_id и записывает каждый шаг пайплайна: время,
входные/выходные данные, системные метрики. Активные трейсы живут в памяти
(пока запрос выполняется), завершённые сохраняются в SQLite и кешируются
в ограниченном in-memory LRU для быстрого get_trace().

Публичный API сохранён (start_trace, add_step, end_trace, get_trace,
get_recent_traces, get_stats) — код retriever'а / routes не переписывается.

Дополнительно:
  - start_trace() устанавливает trace_id в contextvar, end_trace() сбрасывает.
    Это значит, что логи внутри обработчика запроса автоматически получают
    `trace_id` в JSON-формате (см. app.utils.logging).
"""
from __future__ import annotations

import json
import platform
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from app.config import settings
from app.utils.logging import get_logger
from app.utils.trace_context import set_trace_id, reset_trace_id, clear_trace_id

log = get_logger("tracer")


# ─────────────────────────────────────────────────
# Data types (остались обратно-совместимыми)
# ─────────────────────────────────────────────────

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
    request_type: str = ""           # "query", "upload", "evaluate", ...
    input_preview: str = ""           # Первые символы вопроса/файла
    steps: list = field(default_factory=list)
    system_snapshot: dict = field(default_factory=dict)
    started_at: str = ""
    total_ms: float = 0.0
    status: str = "running"           # running, completed, error

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


# ─────────────────────────────────────────────────
# SQLite storage
# ─────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id       TEXT PRIMARY KEY,
    request_type   TEXT NOT NULL,
    input_preview  TEXT,
    started_at     TEXT NOT NULL,
    total_ms       REAL DEFAULT 0,
    status         TEXT NOT NULL,
    system_json    TEXT,
    steps_json     TEXT,
    created_epoch  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
CREATE INDEX IF NOT EXISTS idx_traces_type ON traces(request_type);
"""


class _TraceDB:
    """Лёгкая обёртка над sqlite3 с потокобезопасными write'ами.

    SQLite в режиме WAL + check_same_thread=False + один write-lock даёт
    простую и достаточно быструю запись для нашего RPS (~10-100 req/s max).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._last_cleanup_epoch: float = 0.0

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
                timeout=5.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            log.info("Trace DB готова: {}", self.path)
        return self._conn

    def upsert(self, trace: Trace) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO traces (
                    trace_id, request_type, input_preview, started_at,
                    total_ms, status, system_json, steps_json, created_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    request_type=excluded.request_type,
                    input_preview=excluded.input_preview,
                    started_at=excluded.started_at,
                    total_ms=excluded.total_ms,
                    status=excluded.status,
                    system_json=excluded.system_json,
                    steps_json=excluded.steps_json
                """,
                (
                    trace.trace_id,
                    trace.request_type,
                    trace.input_preview,
                    trace.started_at,
                    trace.total_ms,
                    trace.status,
                    json.dumps(trace.system_snapshot, ensure_ascii=False, default=str),
                    json.dumps(
                        [
                            {
                                "step": s.step,
                                "time_ms": s.time_ms,
                                "timestamp": s.timestamp,
                                "details": s.details,
                            }
                            for s in trace.steps
                        ],
                        ensure_ascii=False, default=str,
                    ),
                    time.time(),
                ),
            )
            conn.commit()

    def get(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT request_type, input_preview, started_at, total_ms, "
                "status, system_json, steps_json FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        if not row:
            return None
        return _row_to_trace(trace_id, row)

    def recent(self, limit: int = 20) -> list[Trace]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT trace_id, request_type, input_preview, started_at, "
                "total_ms, status, system_json, steps_json "
                "FROM traces ORDER BY created_epoch DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_trace(r[0], r[1:]) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            conn = self._get_conn()
            total = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE status != 'running'"
            ).fetchone()[0]
            if total == 0:
                return {"total_traces": 0}

            avg_row = conn.execute(
                "SELECT AVG(total_ms), MIN(total_ms), MAX(total_ms) "
                "FROM traces WHERE total_ms > 0"
            ).fetchone()
            by_type_rows = conn.execute(
                "SELECT request_type, COUNT(*), AVG(total_ms) "
                "FROM traces WHERE total_ms > 0 GROUP BY request_type"
            ).fetchall()
            err_count = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE status = 'error'"
            ).fetchone()[0]

        return {
            "total_traces": total,
            "avg_time_ms": round(avg_row[0] or 0, 1),
            "min_time_ms": round(avg_row[1] or 0, 1),
            "max_time_ms": round(avg_row[2] or 0, 1),
            "by_type": {
                rt: {"count": cnt, "avg_ms": round(avg or 0, 1)}
                for rt, cnt, avg in by_type_rows
            },
            "error_count": err_count,
        }

    def cleanup_if_due(self, retention_days: int) -> None:
        """Лениво чистим старые трейсы (не чаще раза в сутки)."""
        if retention_days <= 0:
            return
        now = time.time()
        if (now - self._last_cleanup_epoch) < 86400:
            return
        cutoff = now - retention_days * 86400
        with self._lock:
            conn = self._get_conn()
            res = conn.execute(
                "DELETE FROM traces WHERE created_epoch < ?",
                (cutoff,),
            )
            conn.commit()
            if res.rowcount:
                log.info("Trace DB: удалено {} старых трейсов", res.rowcount)
        self._last_cleanup_epoch = now


def _row_to_trace(trace_id: str, row: tuple) -> Trace:
    """row = (request_type, input_preview, started_at, total_ms, status,
              system_json, steps_json)"""
    request_type, input_preview, started_at, total_ms, status, sys_json, steps_json = row
    trace = Trace(
        trace_id=trace_id,
        request_type=request_type or "",
        input_preview=input_preview or "",
        started_at=started_at or "",
        total_ms=float(total_ms or 0),
        status=status or "completed",
    )
    try:
        trace.system_snapshot = json.loads(sys_json) if sys_json else {}
    except Exception:
        trace.system_snapshot = {}
    try:
        steps_raw = json.loads(steps_json) if steps_json else []
        trace.steps = [
            TraceStep(
                step=s.get("step", ""),
                time_ms=float(s.get("time_ms", 0)),
                timestamp=s.get("timestamp", ""),
                details=s.get("details") or {},
            )
            for s in steps_raw
        ]
    except Exception:
        trace.steps = []
    return trace


# ─────────────────────────────────────────────────
# Tracer
# ─────────────────────────────────────────────────

class Tracer:
    """Трассировщик пайплайна с persistence в SQLite.

    Активные трейсы — в памяти (_active). Завершённые сохраняются в SQLite
    и добавляются в in-memory LRU (_cache) размером trace_memory_cache_size
    для быстрого get_trace без запроса в БД.
    """

    def __init__(self):
        self._active: dict[str, tuple[Trace, float, object]] = {}
        # trace_id -> (trace, start_perf_counter, contextvar_token)
        self._cache: "OrderedDict[str, Trace]" = OrderedDict()
        self._db = _TraceDB(settings.trace_db_path)
        self._cache_size = settings.trace_memory_cache_size

    # ─── lifecycle ───

    def start_trace(self, request_type: str, input_preview: str = "") -> str:
        """Начать новую трассировку.

        Дополнительно устанавливает trace_id в contextvar — все логи внутри
        этого async-контекста будут получать его автоматически.
        """
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        trace = Trace(
            trace_id=trace_id,
            request_type=request_type,
            input_preview=(input_preview or "")[:100],
            started_at=now.isoformat(),
        )
        token = set_trace_id(trace_id)
        self._active[trace_id] = (trace, time.perf_counter(), token)

        log.debug("Trace started: {} ({})", trace_id, request_type)
        return trace_id

    def add_step(
        self,
        trace_id: str,
        step_name: str,
        time_ms: float = 0.0,
        **details,
    ) -> None:
        if trace_id not in self._active:
            return

        trace, _, _ = self._active[trace_id]
        now = datetime.now(timezone.utc)

        trace.steps.append(TraceStep(
            step=step_name,
            time_ms=time_ms,
            timestamp=now.strftime("%H:%M:%S.%f")[:-3],
            details=details,
        ))

    def end_trace(self, trace_id: str, status: str = "completed") -> None:
        entry = self._active.pop(trace_id, None)
        if entry is None:
            return
        trace, start_perf, token = entry
        trace.total_ms = (time.perf_counter() - start_perf) * 1000
        trace.status = status
        trace.system_snapshot = self._get_system_snapshot()
        trace.steps.append(TraceStep(
            step="total",
            time_ms=trace.total_ms,
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
        ))

        # Persist + cache
        try:
            self._db.upsert(trace)
        except Exception as e:
            log.warning("Trace persist failed ({}): {}", trace_id, e)

        self._cache[trace_id] = trace
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # drop oldest

        # Периодическая чистка старых
        try:
            self._db.cleanup_if_due(settings.trace_retention_days)
        except Exception as e:
            log.debug("Trace cleanup failed: {}", e)

        # Сбросить contextvar — иначе trace_id протечёт в следующие запросы
        # того же worker'а (в реальности FastAPI запросы изолированы
        # контекстом, но в тестах/background task'ах это критично)
        reset_trace_id(token)

        log.debug(
            "Trace completed: {} ({}) {:.0f}ms, {} steps",
            trace_id, trace.request_type, trace.total_ms, len(trace.steps),
        )

    # ─── read ───

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Получить трейс по ID (cache → active → SQLite)."""
        if trace_id in self._cache:
            self._cache.move_to_end(trace_id)
            return self._cache[trace_id].to_dict()

        if trace_id in self._active:
            trace, _, _ = self._active[trace_id]
            return trace.to_dict()

        trace = self._db.get(trace_id)
        if trace:
            self._cache[trace_id] = trace
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return trace.to_dict()

        return None

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Последние трейсы (SQLite, по времени создания DESC)."""
        traces = self._db.recent(limit=limit)
        return [t.to_dict() for t in traces]

    def get_stats(self) -> dict:
        """Общая статистика по трейсам."""
        stats = self._db.stats()
        stats["active_traces"] = len(self._active)
        return stats

    # ─── system ───

    def _get_system_snapshot(self) -> dict:
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
