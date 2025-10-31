# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps (optional): for llama.cpp client, curl
RUN apt-get update && apt-get install -y --no-install-recommends     ca-certificates curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY prompts ./prompts
COPY samples ./samples

ENV PORT=8009
EXPOSE 8009

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8009"]
