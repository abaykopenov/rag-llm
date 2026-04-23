"""
Базовые типы для embedding-провайдеров.

EmbeddingProvider — ABC, который должны реализовать все провайдеры.
EmbeddingProviderInfo — паспорт провайдера (имя, модель, dim) для штамповки в metadata коллекции ChromaDB.
TokenizerSpec — спецификация токенизатора для HybridChunker (Docling).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class EmbeddingProviderInfo:
    """Идентифицирует, какой провайдер и какой моделью построил вектора в коллекции.

    Записывается в metadata коллекции ChromaDB при создании. При каждом
    обращении к коллекции проверяется, что активный провайдер совпадает —
    иначе размерность векторов не совпадёт и поиск сломается.
    """
    provider: str       # "gemini" | "openai_compat"
    model: str          # "text-embedding-004" | "BAAI/bge-m3" | ...
    dim: int            # размерность вектора

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    def to_meta(self) -> dict[str, str | int]:
        """Сериализация в metadata-ключи коллекции ChromaDB."""
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingProviderInfo":
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            dim=int(data["dim"]),
        )

    @classmethod
    def from_meta(cls, meta: dict) -> Optional["EmbeddingProviderInfo"]:
        """Прочитать из metadata коллекции (если все ключи есть), иначе None."""
        if not meta:
            return None
        try:
            return cls(
                provider=str(meta["embedding_provider"]),
                model=str(meta["embedding_model"]),
                dim=int(meta["embedding_dim"]),
            )
        except (KeyError, ValueError, TypeError):
            return None

    def matches(self, other: "EmbeddingProviderInfo") -> bool:
        return (
            self.provider == other.provider
            and self.model == other.model
            and self.dim == other.dim
        )

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}@dim={self.dim}"


@dataclass
class TokenizerSpec:
    """Как получить токенизатор для HybridChunker.

    type:
      "hf"        — HuggingFace AutoTokenizer (для BGE/Jina/Nomic/proxy-MPNet)
      "tiktoken"  — OpenAI tiktoken (cl100k_base и пр.)
      "none"      — токенизатор не сконфигурирован; HybridChunker упадёт,
                    если его попросить — это предупреждение, что провайдер
                    пока неполный.
    """
    type: str = "none"
    name: str = ""
    max_tokens: int = 512


class EmbeddingProvider(ABC):
    """Общий интерфейс для всех embedding-провайдеров."""

    info: EmbeddingProviderInfo
    max_input_tokens: int
    tokenizer_spec: TokenizerSpec

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить embeddings для батча текстов."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Получить embedding для одного текста.

        Провайдер может переопределить это, если у модели есть отдельный
        query-prefix (Jina, E5) или endpoint (Cohere input_type=search_query).
        """
        result = await self.embed_texts([text])
        return result[0]

    def get_tokenizer(self):
        """Вернуть Docling BaseTokenizer для HybridChunker.

        Ленивый импорт — чтобы не тянуть transformers/tiktoken без нужды.
        Возвращает None если токенизатор не сконфигурирован (type=none).
        """
        spec = self.tokenizer_spec
        if spec is None or spec.type == "none":
            return None

        if spec.type == "hf":
            return _load_hf_tokenizer(spec)
        if spec.type == "tiktoken":
            return _load_tiktoken_tokenizer(spec)

        raise ValueError(f"Unknown tokenizer type: {spec.type}")

    async def close(self) -> None:  # pragma: no cover — override if нужен cleanup
        """Освободить ресурсы (HTTP-клиенты и т.п.)."""
        return None


def _load_hf_tokenizer(spec: TokenizerSpec):
    """Ленивый импорт docling_core + transformers."""
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from transformers import AutoTokenizer

    return HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(spec.name),
        max_tokens=spec.max_tokens,
    )


def _load_tiktoken_tokenizer(spec: TokenizerSpec):
    """Ленивый импорт docling_core OpenAITokenizer."""
    # Docling Core предоставляет OpenAITokenizer в последних версиях;
    # если его нет — падаем с понятным сообщением.
    try:
        from docling_core.transforms.chunker.tokenizer.openai import (
            OpenAITokenizer,
        )
    except ImportError as e:
        raise ImportError(
            "tiktoken-токенизатор требует docling_core>=2.x с поддержкой "
            "OpenAITokenizer. Обнови docling или выбери type=hf."
        ) from e

    import tiktoken

    return OpenAITokenizer(
        tokenizer=tiktoken.get_encoding(spec.name),
        max_tokens=spec.max_tokens,
    )
