"""
Мониторинг системных ресурсов: CPU, RAM, GPU.
"""

import psutil
import platform
import shutil
from typing import Any

from app.utils.logging import get_logger

log = get_logger("monitor")


def get_system_stats() -> dict[str, Any]:
    """Собрать метрики системы: CPU, RAM, диск, GPU."""

    # CPU
    cpu_percent_per_core = psutil.cpu_percent(interval=0.3, percpu=True)
    cpu_avg = round(sum(cpu_percent_per_core) / len(cpu_percent_per_core), 1) if cpu_percent_per_core else 0

    # RAM
    mem = psutil.virtual_memory()

    # Диск
    disk = shutil.disk_usage(".")

    stats = {
        "hostname": platform.node(),
        "os": platform.system(),
        "cpu": {
            "percent": cpu_avg,
            "per_core": cpu_percent_per_core,
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        },
        "memory": {
            "percent": mem.percent,
            "used_gb": round(mem.used / (1024 ** 3), 1),
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "available_gb": round(mem.available / (1024 ** 3), 1),
        },
        "disk": {
            "percent": round(disk.used / disk.total * 100, 1),
            "used_gb": round(disk.used / (1024 ** 3), 1),
            "total_gb": round(disk.total / (1024 ** 3), 1),
        },
        "gpu": _get_gpu_stats(),
    }

    return stats


def _get_gpu_stats() -> dict[str, Any]:
    """Получить метрики GPU через pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            try:
                power = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            except Exception:
                power = None

            gpus.append({
                "index": i,
                "name": name,
                "gpu_percent": util.gpu,
                "memory_percent": round(mem_info.used / mem_info.total * 100, 1),
                "memory_used_gb": round(mem_info.used / (1024 ** 3), 2),
                "memory_total_gb": round(mem_info.total / (1024 ** 3), 2),
                "temperature_c": temp,
                "power_w": power,
            })

        pynvml.nvmlShutdown()

        return {
            "available": True,
            "count": device_count,
            "devices": gpus,
        }

    except Exception:
        return {"available": False, "count": 0, "devices": []}
