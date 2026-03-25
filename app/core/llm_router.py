"""
LLM Router — единый клиент для работы с любым LLM провайдером.
Поддерживает vLLM, OpenAI, Anthropic, Groq, и любой OpenAI-совместимый API.
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

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }

        log.info(
            "LLM запрос: model={}, messages={}, temp={}",
            payload["model"],
            len(messages),
            payload["temperature"],
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

        usage = data.get("usage", {})

        log.info(
            "LLM ответ: {} токенов, {:.0f} мс",
            usage.get("completion_tokens", "?"),
            elapsed_ms,
        )

        return LLMResponse(
            text=answer,
            model=data.get("model", payload["model"]),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            generation_time_ms=elapsed_ms,
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
    ):
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.generation_time_ms = generation_time_ms


# Глобальный экземпляр роутера
llm_router = LLMRouter()

