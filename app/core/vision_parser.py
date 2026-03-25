"""
Vision Parser — парсинг документов через Vision LLM.

Каждая страница PDF рендерится как PNG, затем отправляется в Vision LLM,
который "видит" страницу как человек и извлекает:
- Текст (с сохранением структуры)
- Таблицы (в Markdown формате)
- Описание изображений и диаграмм
- Формулы (в LaTeX)

Аналог Gemini API парсинга, но работает с любым Vision LLM
(Qwen2-VL, InternVL, LLaVA и др.) через OpenAI-compatible API.
"""

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings
from app.utils.logging import get_logger

log = get_logger("vision_parser")

# Системный промпт для Vision LLM
VISION_PARSE_PROMPT = """Ты — эксперт по извлечению текста из документов.

Задача: извлеки ВСЕ содержимое этой страницы документа в формате Markdown.

Правила:
1. **Текст**: сохрани структуру (заголовки → ## или ###, абзацы, списки)
2. **Таблицы**: преобразуй в Markdown-таблицы (| col1 | col2 |)
3. **Изображения/Диаграммы**: опиши содержимое в формате [Рис: описание того, что изображено]
4. **Формулы**: запиши в LaTeX формате ($формула$)
5. **Нумерация**: сохрани нумерацию списков и страниц
6. **Не добавляй** ничего от себя, только то что есть на странице
7. **Язык**: сохрани оригинальный язык документа

Верни ТОЛЬКО Markdown-текст страницы, без пояснений."""


class VisionParser:
    """Парсинг документов через Vision LLM (page-as-image подход)."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init HTTP клиента."""
        if self._client is None or self._client.is_closed:
            base_url = settings.vision_llm_base_url or settings.llm_base_url
            api_key = settings.llm_api_key

            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    async def parse_pdf(self, file_path: str, max_concurrent: int = 3) -> str:
        """Распарсить PDF через Vision LLM.

        Args:
            file_path: Путь к PDF файлу
            max_concurrent: Максимум параллельных запросов

        Returns:
            Полный Markdown текст документа
        """
        start = time.perf_counter()
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        log.info("Vision parsing: {}", path.name)

        # 1. Рендерим PDF в картинки
        images = self._render_pdf(file_path)
        log.info("Отрендерено {} страниц", len(images))

        if not images:
            log.warning("Не удалось отрендерить PDF")
            return ""

        # 2. Обрабатываем каждую страницу через Vision LLM
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_page(page_num: int, image_bytes: bytes) -> str:
            async with semaphore:
                return await self._parse_page(page_num, image_bytes)

        tasks = [
            process_page(i + 1, img)
            for i, img in enumerate(images)
        ]
        page_texts = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Объединяем результаты
        markdown_parts = []
        for i, result in enumerate(page_texts):
            if isinstance(result, Exception):
                log.warning("Ошибка на странице {}: {}", i + 1, result)
                markdown_parts.append(f"\n\n<!-- Ошибка на странице {i + 1} -->\n\n")
            elif result:
                markdown_parts.append(result)

        full_text = "\n\n---\n\n".join(markdown_parts)

        elapsed = time.perf_counter() - start
        log.info(
            "Vision parsing завершён: {} страниц, {} символов, {:.1f} сек",
            len(images), len(full_text), elapsed,
        )

        return full_text

    def _render_pdf(self, file_path: str) -> list[bytes]:
        """Рендер PDF в PNG-картинки.

        Пытается использовать PyMuPDF (fitz), затем pdf2image.
        """
        # Способ 1: PyMuPDF (быстрый, без poppler)
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            images = []
            for page in doc:
                # Рендерим с DPI=200 (баланс качество/скорость)
                pix = page.get_pixmap(dpi=200)
                images.append(pix.tobytes("png"))

            doc.close()
            log.info("PDF отрендерен через PyMuPDF: {} страниц", len(images))
            return images

        except ImportError:
            log.debug("PyMuPDF не установлен, пробуем pdf2image")

        # Способ 2: pdf2image (нужен poppler)
        try:
            from pdf2image import convert_from_path

            pil_images = convert_from_path(file_path, dpi=200)
            images = []
            for img in pil_images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images.append(buf.getvalue())

            log.info("PDF отрендерен через pdf2image: {} страниц", len(images))
            return images

        except ImportError:
            log.error("Ни PyMuPDF ни pdf2image не установлены!")
            log.error("Установите: pip install PyMuPDF  или  pip install pdf2image")
            return []
        except Exception as e:
            log.error("Ошибка рендера PDF: {}", e)
            return []

    async def _parse_page(self, page_num: int, image_bytes: bytes) -> str:
        """Обработать одну страницу через Vision LLM.

        Args:
            page_num: Номер страницы
            image_bytes: PNG-картинка страницы

        Returns:
            Markdown-текст страницы
        """
        client = self._get_client()
        model = settings.vision_llm_model

        # Кодируем картинку в base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PARSE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            text = data["choices"][0]["message"]["content"]

            log.debug("Страница {} обработана: {} символов", page_num, len(text))
            return f"<!-- Страница {page_num} -->\n\n{text}"

        except httpx.TimeoutException:
            log.warning("Timeout на странице {}", page_num)
            return ""
        except Exception as e:
            log.error("Ошибка Vision LLM на странице {}: {}", page_num, e)
            return ""

    async def parse_image(self, file_path: str) -> str:
        """Распарсить одиночное изображение через Vision LLM."""
        path = Path(file_path)
        with open(path, "rb") as f:
            image_bytes = f.read()

        return await self._parse_page(1, image_bytes)

    async def close(self):
        """Закрыть HTTP клиент."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Глобальный экземпляр
vision_parser = VisionParser()
