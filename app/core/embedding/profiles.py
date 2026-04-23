"""
Загрузка и разрешение embedding-профилей из YAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from app.core.embedding.base import TokenizerSpec
from app.utils.logging import get_logger

log = get_logger("embedding.profiles")


@dataclass
class EmbeddingProfile:
    """Распарсенный профиль из embedding_profiles.yml."""

    name: str
    provider: str                  # "gemini" | "openai_compat"
    model: str
    dim: int
    max_input_tokens: int = 2048
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    batch_size: int = 32
    tokenizer: Optional[TokenizerSpec] = None
    # Дополнительные опции, специфичные для провайдера, можно положить сюда
    extra: dict = field(default_factory=dict)

    def resolved_api_key(self) -> str:
        """Итоговый ключ: приоритет у api_key_env, иначе api_key, иначе пусто."""
        if self.api_key_env:
            env_val = os.getenv(self.api_key_env, "").strip()
            if env_val:
                return env_val
        if self.api_key:
            return self.api_key
        return ""


def load_profiles(path: str | Path) -> dict[str, EmbeddingProfile]:
    """Загрузить все профили из YAML-файла.

    Args:
        path: Путь к YAML-файлу

    Returns:
        Словарь {имя_профиля: EmbeddingProfile}

    Raises:
        FileNotFoundError: если файл не существует
        ValueError: если YAML невалидный или не содержит 'profiles'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding profiles file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError(
            f"{path}: expected top-level key 'profiles' with a mapping of profile names"
        )

    result: dict[str, EmbeddingProfile] = {}
    for name, cfg in raw_profiles.items():
        if not isinstance(cfg, dict):
            log.warning("Profile '{}' is not a mapping, skipping", name)
            continue

        cfg = dict(cfg)  # копия, чтобы .pop() не портил исходник
        tokenizer_raw = cfg.pop("tokenizer", None)
        tok_spec: Optional[TokenizerSpec] = None
        if tokenizer_raw:
            if not isinstance(tokenizer_raw, dict):
                log.warning(
                    "Profile '{}': tokenizer must be a mapping, got {}",
                    name, type(tokenizer_raw).__name__,
                )
            else:
                tok_spec = TokenizerSpec(
                    type=str(tokenizer_raw.get("type", "none")),
                    name=str(tokenizer_raw.get("name", "")),
                    max_tokens=int(tokenizer_raw.get("max_tokens", 512)),
                )

        try:
            result[name] = EmbeddingProfile(
                name=name,
                provider=str(cfg.pop("provider")),
                model=str(cfg.pop("model")),
                dim=int(cfg.pop("dim")),
                max_input_tokens=int(cfg.pop("max_input_tokens", 2048)),
                base_url=cfg.pop("base_url", None),
                api_key=cfg.pop("api_key", None),
                api_key_env=cfg.pop("api_key_env", None),
                batch_size=int(cfg.pop("batch_size", 32)),
                tokenizer=tok_spec,
                extra=cfg,  # всё остальное — в extra
            )
        except KeyError as e:
            raise ValueError(
                f"Profile '{name}' missing required key: {e}"
            ) from e

    log.info("Загружено {} embedding-профилей: {}", len(result), list(result))
    return result


def build_profile_from_flat_settings(settings) -> EmbeddingProfile:
    """Fallback: собрать профиль из старых плоских env-полей settings.

    Используется, если не указан RAG_EMBEDDING_PROFILE и/или нет YAML-файла.
    Сохраняет обратную совместимость со старыми .env.
    """
    # Старый settings.embedding_provider = "gemini" | "jina" | "vllm" | "openai" | ...
    # Новый provider = "gemini" | "openai_compat"
    old = (settings.embedding_provider or "gemini").lower()
    provider = "gemini" if old == "gemini" else "openai_compat"

    return EmbeddingProfile(
        name="_from_env",
        provider=provider,
        model=settings.embedding_model,
        dim=settings.embedding_dimensions,
        base_url=settings.embedding_base_url or None,
        api_key=settings.embedding_api_key or None,
        batch_size=32,
        tokenizer=None,  # без профиля — без токенизатора; HybridChunker упадёт, но embed работает
    )


def resolve_active_profile(settings) -> EmbeddingProfile:
    """Определить активный профиль по текущей конфигурации.

    Порядок разрешения:
      1. Если RAG_EMBEDDING_PROFILE задан и YAML существует — берём оттуда.
      2. Если RAG_EMBEDDING_PROFILE задан, но профиля нет — ошибка.
      3. Если не задан, но YAML существует и в нём есть профиль с именем 'default' —
         берём его.
      4. Иначе — собираем профиль из плоских env-полей (backward-compat).
    """
    profiles_path = Path(settings.embedding_profiles_path)
    active_name = (settings.embedding_profile or "").strip()

    # Путь 1-2: явное имя профиля
    if active_name:
        if not profiles_path.exists():
            raise FileNotFoundError(
                f"RAG_EMBEDDING_PROFILE={active_name!r}, но файл профилей не найден: "
                f"{profiles_path}. Создай его или сбрось переменную."
            )
        profiles = load_profiles(profiles_path)
        if active_name not in profiles:
            raise ValueError(
                f"Профиль {active_name!r} не найден. Доступные: {list(profiles)}"
            )
        log.info("Активный embedding-профиль: '{}'", active_name)
        return profiles[active_name]

    # Путь 3: неявный 'default' в YAML
    if profiles_path.exists():
        profiles = load_profiles(profiles_path)
        if "default" in profiles:
            log.info("Активный embedding-профиль: 'default' (из YAML)")
            return profiles["default"]

    # Путь 4: fallback на плоский .env
    log.warning(
        "RAG_EMBEDDING_PROFILE не задан и YAML-профилей нет — "
        "собираю профиль из плоских env-переменных (tokenizer не будет подгружен)"
    )
    return build_profile_from_flat_settings(settings)
