"""
BM25 keyword-search для hybrid retrieval.

Задумано заменой substring-match в indexer.keyword_search(). Даёт:
- Правильный скоринг (BM25Okapi, с IDF и длиной документа).
- Стемминг (выручки → выручка → общий корень "выручк").
- Токенизацию с стоп-словами.

Ограничение Snowball Russian: стеммер по правилам, не по словарю — некоторые
пары слов ("прибыль"/"прибыли") получат разные основы. Большинство падежей
работает корректно. Если нужно качественное покрытие русского — поставь
`pymorphy3` и подмени стеммер в _get_stemmer(): реализуй обёртку над
`MorphAnalyzer().parse(w)[0].normal_form`. Текущий Snowball-путь — без тяжёлых
зависимостей.

Архитектура:
- BM25Collection — per-collection индекс с persistent save/load.
- BM25Registry — singleton, кэширует загруженные коллекции в памяти.

Жизненный цикл индекса:
- Build: автоматически при первом search() для коллекции, если нет
  сохранённого индекса. Читаем чанки из ChromaDB, токенизируем, сохраняем.
- Invalidate: вызывается из indexer.add_chunks / delete_collection.
  Удаляет файлы с диска и сбрасывает кэш — следующий search пересоберёт.
- Инкрементальный append библиотекой bm25s не поддерживается, поэтому
  мы делаем полный rebuild. На 100K чанков это ~1–2 секунды.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import bm25s
import snowballstemmer

from app.config import settings
from app.utils.logging import get_logger

if TYPE_CHECKING:
    import chromadb

log = get_logger("bm25")

# Расширение, куда bm25s сохраняет индекс — проверяем для load()
_INDEX_MARKER = "params.index.json"  # создаётся bm25s.save()
_META_FILENAME = "_rag_meta.json"    # наши метаданные (ids, texts, metas)


class BM25Collection:
    """BM25-индекс одной коллекции ChromaDB с сохранением на диск."""

    def __init__(
        self,
        name: str,
        persist_dir: Path | str,
        stemmer_lang: str = "russian",
    ):
        self.name = name
        self.persist_path = Path(persist_dir) / name
        self.stemmer_lang = stemmer_lang
        # Lazy-создаётся при индексации — переиспользуется между запросами
        self._stemmer = None
        self._retriever: Optional[bm25s.BM25] = None
        self._ids: list[str] = []
        # Храним тексты+метаданные в индексе, чтобы не ходить в Chroma
        # на каждый keyword-запрос.
        self._texts: dict[str, str] = {}
        self._metas: dict[str, dict] = {}

    # ─────────────────────────────────────────────────
    # Stemmer (lazy)
    # ─────────────────────────────────────────────────

    def _get_stemmer(self):
        if self._stemmer is None:
            try:
                self._stemmer = snowballstemmer.stemmer(self.stemmer_lang)
            except KeyError:
                log.warning(
                    "Стеммер '{}' не найден, используем английский по умолчанию",
                    self.stemmer_lang,
                )
                self._stemmer = snowballstemmer.stemmer("english")
        return self._stemmer

    # ─────────────────────────────────────────────────
    # Build / save / load
    # ─────────────────────────────────────────────────

    def build_from_chroma(self, chroma_collection) -> int:
        """Прочитать все чанки из ChromaDB и построить BM25-индекс."""
        start = time.perf_counter()
        result = chroma_collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or [""] * len(ids)
        metas = result.get("metadatas") or [{}] * len(ids)

        if not ids:
            log.info("BM25: коллекция '{}' пустая, индекс не строится", self.name)
            self._retriever = None
            self._ids = []
            self._texts = {}
            self._metas = {}
            return 0

        self._ids = list(ids)
        self._texts = dict(zip(self._ids, docs))
        self._metas = {cid: (m or {}) for cid, m in zip(self._ids, metas)}

        stemmer = self._get_stemmer()
        tokens = bm25s.tokenize(docs, stemmer=stemmer, show_progress=False)

        self._retriever = bm25s.BM25()
        self._retriever.index(tokens, show_progress=False)

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "BM25 индекс построен: collection='{}', docs={}, stemmer={}, {:.0f}мс",
            self.name, len(ids), self.stemmer_lang, elapsed_ms,
        )
        return len(ids)

    def save(self) -> None:
        """Сохранить индекс на диск."""
        if self._retriever is None:
            return
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(self.persist_path))
        meta_path = self.persist_path / _META_FILENAME
        meta_path.write_text(
            json.dumps(
                {
                    "ids": self._ids,
                    "texts": self._texts,
                    "metas": self._metas,
                    "stemmer_lang": self.stemmer_lang,
                    "version": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """Попытаться загрузить индекс с диска. True если успешно."""
        meta_path = self.persist_path / _META_FILENAME
        marker = self.persist_path / _INDEX_MARKER
        if not (meta_path.exists() and marker.exists()):
            return False

        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            self._ids = data["ids"]
            self._texts = data["texts"]
            self._metas = data["metas"]
            # Не доверяем сохранённому stemmer_lang если в конфиге поменяли
            if data.get("stemmer_lang") != self.stemmer_lang:
                log.info(
                    "BM25: язык стеммера коллекции '{}' сменился ({} -> {}), "
                    "индекс будет пересобран",
                    self.name, data.get("stemmer_lang"), self.stemmer_lang,
                )
                return False
            self._retriever = bm25s.BM25.load(str(self.persist_path))
            return True
        except Exception as e:
            log.warning(
                "BM25 load failed для '{}': {}; индекс будет пересобран",
                self.name, e,
            )
            return False

    def invalidate(self) -> None:
        """Сбросить in-memory кэш и удалить файлы с диска."""
        self._retriever = None
        self._ids = []
        self._texts = {}
        self._metas = {}
        if self.persist_path.exists():
            try:
                shutil.rmtree(self.persist_path)
            except OSError as e:
                log.warning("BM25: не удалось удалить {}: {}", self.persist_path, e)

    def ensure_ready(self, chroma_collection) -> None:
        """Индекс загружен или построен. Если нет — строим и сохраняем."""
        if self._retriever is not None:
            return
        if self.load():
            return
        self.build_from_chroma(chroma_collection)
        self.save()

    # ─────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Найти top_k документов по BM25.

        Args:
            query: Строка запроса (ключевые слова через пробел).
            top_k: Сколько результатов вернуть ПОСЛЕ фильтра `where`.
            where: ChromaDB-style фильтр по metadata (поддерживаются
                   exact-match + $and + $or).
        """
        if self._retriever is None or not self._ids:
            return []
        if not query or not query.strip():
            return []

        stemmer = self._get_stemmer()
        query_tokens = bm25s.tokenize(query, stemmer=stemmer, show_progress=False)

        # Overfetch для пост-фильтрации по where
        raw_k = min(top_k * 3 if where else top_k, len(self._ids))
        raw_k = max(1, raw_k)
        results, scores = self._retriever.retrieve(
            query_tokens, k=raw_k, show_progress=False,
        )

        out: list[dict] = []
        for rank, (doc_idx, score) in enumerate(
            zip(results[0].tolist(), scores[0].tolist())
        ):
            chunk_id = self._ids[doc_idx]
            meta = self._metas.get(chunk_id, {})
            if where and not _match_where(meta, where):
                continue
            out.append({
                "id": chunk_id,
                "text": self._texts.get(chunk_id, ""),
                "metadata": meta,
                "score": float(score),
                "rank": len(out),
            })
            if len(out) >= top_k:
                break
        return out


# ─────────────────────────────────────────────────
# `where`-matcher (клиентская фильтрация по metadata)
# ─────────────────────────────────────────────────

def _match_where(metadata: dict, where: dict) -> bool:
    """Проверить, удовлетворяет ли metadata ChromaDB-style фильтру.

    Поддерживается:
      - Exact match: {"document_id": "abc"} → metadata["document_id"] == "abc"
      - $and: {"$and": [cond1, cond2]}
      - $or:  {"$or": [cond1, cond2]}
      - $in:  {"field": {"$in": [v1, v2]}}
      - $eq:  {"field": {"$eq": v}}
      - $ne:  {"field": {"$ne": v}}
    """
    if not where:
        return True

    if "$and" in where:
        return all(_match_where(metadata, cond) for cond in where["$and"])
    if "$or" in where:
        return any(_match_where(metadata, cond) for cond in where["$or"])

    for key, expected in where.items():
        if key.startswith("$"):
            continue  # уже обработано выше, защитный тормоз
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif "$eq" in expected:
                if actual != expected["$eq"]:
                    return False
            elif "$ne" in expected:
                if actual == expected["$ne"]:
                    return False
            else:
                # Неизвестный оператор — консервативно False
                return False
        else:
            if actual != expected:
                return False
    return True


# ─────────────────────────────────────────────────
# Singleton registry
# ─────────────────────────────────────────────────

class BM25Registry:
    """Менеджер BM25-индексов всех коллекций."""

    def __init__(self, persist_dir: str, stemmer_lang: str):
        self.persist_dir = Path(persist_dir)
        self.stemmer_lang = stemmer_lang
        self._cache: dict[str, BM25Collection] = {}

    def get(self, name: str) -> BM25Collection:
        if name not in self._cache:
            self._cache[name] = BM25Collection(
                name=name,
                persist_dir=self.persist_dir,
                stemmer_lang=self.stemmer_lang,
            )
        return self._cache[name]

    def invalidate(self, name: str) -> None:
        """Сбросить индекс одной коллекции (вызывается из indexer.add/delete)."""
        if name in self._cache:
            self._cache[name].invalidate()
            del self._cache[name]
        else:
            # Коллекцию ещё не загружали, но файлы могут быть
            path = self.persist_dir / name
            if path.exists():
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    log.warning("BM25 invalidate: не удалён {}: {}", path, e)


bm25_registry = BM25Registry(
    persist_dir=settings.bm25_persist_dir,
    stemmer_lang=settings.bm25_stemmer_lang,
)
