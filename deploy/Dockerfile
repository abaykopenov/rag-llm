# RAG-LLM API Сервер
FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY app/ ./app/
COPY proto/ ./proto/
COPY data/ ./data/

# Создаём директории
RUN mkdir -p data/uploads data/texts data/chroma data/sessions

EXPOSE 8000 50051

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
