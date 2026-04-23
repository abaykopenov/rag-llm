"""
LLM Router — единый клиент для работы с любым LLM провайдером.
Поддерживает vLLM, OpenAI, Anthropic, Groq, и любой OpenAI-совместимый API.

Prompt caching:
  У большинства провайдеров кеш префикса — автоматический на стороне сервера:
    - OpenAI: с Oct 2024, для prompt >=1024 токенов, скидка 50% на cached.
    - Gemini 2.5+: implicit caching, скидка ~75% (через OpenAI-compat эндпоинт
      тоже работает).
    - vLLM: нужно запустить с --enable-prefix-caching на сервере.
    - Groq: частичное автоматическое кеширование.
    - Anthropic: требует явный cache_control в сообщениях
      (включается settings.llm_anthropic_cache_control).

  Что для этого нужно на клиенте:
    1. Держать system prompt и структуру prefix'а стабильными — не добавлять
       в них timestamps, счётчики, UUID'ы. В текущем коде префикс стабилен:
       system = settings.system_prompt (статичный string), history = тело
       сессии, user = контекст + вопрос (вопрос в конце, контекст кешируется
       при повторе той же выборки чанков).
    2. Парсить cached_tokens из usage — для телеметрии и валидации экономии.
       Делается в _parse_cache_stats() ниже.
"""

import asyncio
import time
from typing import Optional

import httpx

from app.config import settings
from app.utils.logging import get_logger

log = get_logger("llm_router")

# Retry настройки
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # секунды


def _annotate_for_anthropic_cache(
    messages: list[dict],
    min_chars_for_cache: int = 4 * 1024,
) -> list[dict]:
    """Разметить системный промпт и контекст cache_control для Anthropic.

    Anthropic принимает messages с content = list of blocks. На стабильный
    длинный блок ставим cache_control: ephemeral — Anthropic будет кешировать
    этот префикс (5-минутный TTL).

    Стратегия:
      - system сообщение с длинным content → маркируем его cache_control
      - последнее user сообщение, если оно выглядит как "Контекст:...Вопрос:..."
        → режем на 2 блока: контекст (cache_control) + вопрос (без cache_control)
      - остальные сообщения остаются как есть

    Не-Anthropic провайдерам эта разметка не повредит: OpenAI-compat endpoints
    игнорируют `cache_control` поле. Но лучше её применять ТОЛЬКО когда
    settings.llm_anthropic_cache_control=True — см. вызов в generate().
    """
    if not messages:
        return messages

    result: list[dict] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        role = msg.get("role", "")

        # Content уже в формате list-of-blocks — не трогаем
        if isinstance(content, list):
            result.append(msg)
            continue

        if not isinstance(content, str):
            result.append(msg)
            continue

        # 1. Системный промпт: маркируем если достаточно длинный
        if role == "system" and len(content) >= min_chars_for_cache:
            result.append({
                "role": role,
                "content": [{
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }],
            })
            continue

        # 2. Последний user-блок: если есть маркер "Вопрос:" — режем
        is_last_user = (
            role == "user"
            and i == len(messages) - 1
            and "Вопрос:" in content
        )
        if is_last_user:
            # Режем по последнему вхождению "Вопрос:" — всё до него это контекст
            idx = content.rfind("Вопрос:")
            context_block = content[:idx].rstrip()
            question_block = content[idx:]

            if len(context_block) >= min_chars_for_cache:
                result.append({
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": context_block,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": question_block},
                    ],
                })
                continue

        # По умолчанию — без изменений
        result.append(msg)

    return result


def _parse_cache_stats(usage: dict) -> tuple[int, int]:
    """Извлечь из usage-блока cached и cache_creation токены.

    Покрывает форматы:
      - OpenAI:    usage.prompt_tokens_details.cached_tokens
      - Anthropic: usage.cache_read_input_tokens,
                   usage.cache_creation_input_tokens
      - Gemini (native): cached_content_token_count — через OpenAI-compat
        эндпоинт редко пробрасывается, но проверим.

    Returns:
        (cached_prompt_tokens, cache_creation_tokens)
    """
    if not isinstance(usage, dict):
        return 0, 0

    cached = 0
    creation = 0

    # OpenAI: prompt_tokens_details.cached_tokens
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)

    # Anthropic: cache_read_input_tokens / cache_creation_input_tokens
    # (могут быть пробрасены через OpenAI-compat прокси)
    if not cached:
        cached = int(usage.get("cache_read_input_tokens", 0) or 0)
    creation = int(usage.get("cache_creation_input_tokens", 0) or 0)

    # Gemini native: cached_content_token_count (на всякий случай)
    if not cached:
        cached = int(usage.get("cached_content_token_count", 0) or 0)

    return cached, creation


class LLMRouter:
    """Роутер запросов к LLM провайдерам.

    Все провайдеры используют OpenAI-совместимый протокол.
    Меняется только base_url и api_key.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy init HTTP клиента."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(300.0),  # LLM может думать долго
            )
        return self._client

    async def generate(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "LLMResponse":
        """Отправить запрос к LLM и получить ответ.

        Args:
            messages: Список сообщений [{"role": "system", "content": "..."}, ...]
            model: Модель (если отличается от настроек)
            temperature: Температура генерации
            max_tokens: Максимум токенов в ответе

        Returns:
            LLMResponse с текстом ответа и метаданными
        """
        client = await self._get_client()

        # Anthropic-style cache_control — применяется, только если включено в
        # конфиге. Для OpenAI/Gemini/vLLM это поле бессмысленно; при отправке
        # большинство прокси его игнорируют, но чтобы не спотыкаться — не шлём.
        if settings.llm_anthropic_cache_control:
            min_chars = settings.llm_cache_min_tokens * 4  # ~4 символа/токен
            messages = _annotate_for_anthropic_cache(messages, min_chars)

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }

        log.info(
            "LLM запрос: model={}, messages={}, temp={}{}",
            payload["model"],
            len(messages),
            payload["temperature"],
            " [cache_control=on]" if settings.llm_anthropic_cache_control else "",
        )

        start = time.perf_counter()
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                break  # Успех — выходим из retry-цикла

            except httpx.ConnectError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        "LLM недоступен (попытка {}/{}), повтор через {:.1f}с: {}",
                        attempt, MAX_RETRIES, delay, self.base_url,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error("LLM недоступен после {} попыток: {}", MAX_RETRIES, self.base_url)
                    raise ConnectionError(
                        f"Не удалось подключиться к LLM по адресу {self.base_url} "
                        f"после {MAX_RETRIES} попыток. Убедитесь, что сервер запущен."
                    ) from e

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500 and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        "LLM ошибка {} (попытка {}/{}), повтор через {:.1f}с",
                        e.response.status_code, attempt, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error("LLM HTTP ошибка: {} {}", e.response.status_code, e.response.text[:500])
                    raise

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        "LLM timeout (попытка {}/{}), повтор через {:.1f}с",
                        attempt, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error("LLM timeout после {} попыток", MAX_RETRIES)
                    raise
        else:
            # Все попытки исчерпаны без break
            raise last_error  # type: ignore

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Извлекаем ответ
        choice = data["choices"][0]
        answer = choice["message"]["content"]

        usage = data.get("usage", {}) or {}
        cached, cache_creation = _parse_cache_stats(usage)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)

        if cached:
            hit_ratio = (cached / prompt_tokens * 100) if prompt_tokens else 0.0
            log.info(
                "LLM ответ: {} токенов, {:.0f} мс, cache hit: {}/{} ({:.1f}%)",
                usage.get("completion_tokens", "?"),
                elapsed_ms,
                cached, prompt_tokens, hit_ratio,
            )
        else:
            log.info(
                "LLM ответ: {} токенов, {:.0f} мс{}",
                usage.get("completion_tokens", "?"),
                elapsed_ms,
                f", cache creation: {cache_creation}" if cache_creation else "",
            )

        return LLMResponse(
            text=answer,
            model=data.get("model", payload["model"]),
            prompt_tokens=prompt_tokens,
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            generation_time_ms=elapsed_ms,
            cached_prompt_tokens=cached,
            cache_creation_tokens=cache_creation,
        )

    async def generate_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Streaming генерация — yields текст по кусочкам.

        Yields:
            str: Очередной фрагмент текста
        """
        client = await self._get_client()

        if settings.llm_anthropic_cache_control:
            min_chars = settings.llm_cache_min_tokens * 4
            messages = _annotate_for_anthropic_cache(messages, min_chars)

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": True,
        }

        log.info(
            "LLM stream запрос: model={}, messages={}",
            payload["model"], len(messages),
        )

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]  # убираем "data: "
                if data_str.strip() == "[DONE]":
                    break

                import json
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError):
                    continue

    async def close(self):
        """Закрыть HTTP клиент."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class LLMResponse:
    """Ответ от LLM провайдера."""

    def __init__(
        self,
        text: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        generation_time_ms: float,
        cached_prompt_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ):
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.generation_time_ms = generation_time_ms
        # Cache-метрики (0 если провайдер их не вернул или кеш не сработал).
        # OpenAI/Gemini: cached_prompt_tokens = сколько токенов пришли из кеша.
        # Anthropic: cache_creation_tokens = сколько токенов пошли В кеш (первый
        # запрос после которого последующие будут читаться из кеша).
        self.cached_prompt_tokens = cached_prompt_tokens
        self.cache_creation_tokens = cache_creation_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Доля токенов, прочитанных из кеша (0.0–1.0)."""
        if not self.prompt_tokens:
            return 0.0
        return self.cached_prompt_tokens / self.prompt_tokens


# Глобальный экземпляр роутера
llm_router = LLMRouter()

