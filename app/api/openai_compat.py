"""
OpenAI-compatible API — универсальный RAG-as-a-Service.

Любой клиент подключается как к OpenAI:
  client = OpenAI(base_url="http://host:8000/v1", api_key="any")
  response = client.chat.completions.create(model="any", messages=[...])

RAG-LLM автоматически:
1. Берёт последнее user-сообщение
2. Ищет релевантные чанки
3. Формирует prompt с контекстом
4. Возвращает ответ в формате OpenAI
"""

import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.core.retriever import retriever
from app.core.generator import generator
from app.core.llm_router import llm_router
from app.utils.logging import get_logger
from app.utils.security import require_api_key

log = get_logger("openai_compat")

# OpenAI-compat endpoints тоже под X-API-Key, если в конфиге задан ключ.
router = APIRouter(
    prefix="/v1",
    tags=["OpenAI Compatible"],
    dependencies=[Depends(require_api_key)],
)


# === Pydantic Models (OpenAI format) ===

class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str = Field("auto", description="Модель (игнорируется, используется из конфига)")
    messages: list[OAIMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


class OAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OAIChoice(BaseModel):
    index: int = 0
    message: OAIMessage
    finish_reason: str = "stop"


class OAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage


# === Endpoints ===

@router.post("/chat/completions")
async def chat_completions(
    request: OAIChatRequest,
    x_rag_collection: str = Header("default", alias="X-RAG-Collection"),
    x_rag_top_k: int = Header(5, alias="X-RAG-Top-K"),
    x_rag_enabled: bool = Header(True, alias="X-RAG-Enabled"),
):
    """OpenAI-совместимый chat completions с автоматическим RAG.

    Headers:
        X-RAG-Collection: Коллекция для поиска (default: "default")
        X-RAG-Top-K: Количество чанков (default: 5)
        X-RAG-Enabled: Включить RAG (default: true). Если false — проксирует напрямую.
    """
    # Ищем последнее user-сообщение для RAG
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        # Нет user-сообщения — проксируем как есть
        x_rag_enabled = False

    # RAG: ищем контекст
    chunks = []
    if x_rag_enabled and user_message:
        try:
            chunks = await retriever.retrieve(
                query=user_message,
                collection=x_rag_collection,
                top_k=x_rag_top_k,
            )
        except Exception as e:
            log.warning("RAG retrieval failed, proceeding without: {}", e)

    # Формируем messages с RAG контекстом
    if chunks:
        # Берём историю (все сообщения кроме последнего user)
        history = [
            {"role": m.role, "content": m.content}
            for m in request.messages[:-1]
        ]
        messages = generator.build_prompt(user_message, chunks, history=history)
    else:
        # Без RAG — отправляем messages как есть
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Streaming
    if request.stream:
        return await _stream_response(request_id, messages, request.temperature, request.max_tokens)

    # Non-streaming
    start = time.perf_counter()
    llm_response = await llm_router.generate(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    log.info(
        "OpenAI compat: {} чанков RAG, {} tokens, {:.0f}ms",
        len(chunks), llm_response.total_tokens,
        (time.perf_counter() - start) * 1000,
    )

    return OAIChatResponse(
        id=request_id,
        created=int(time.time()),
        model=llm_response.model,
        choices=[
            OAIChoice(
                message=OAIMessage(role="assistant", content=llm_response.text),
            )
        ],
        usage=OAIUsage(
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        ),
    )


async def _stream_response(request_id: str, messages: list[dict], temperature, max_tokens):
    """SSE streaming в формате OpenAI."""

    async def event_generator():
        async for delta in llm_router.generate_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": settings.llm_model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Финальный chunk
        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": settings.llm_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/models")
async def list_models():
    """OpenAI-совместимый список моделей."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.llm_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": settings.llm_provider,
            }
        ],
    }
