FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /app/shared

COPY src/eva-researcher /app/eva-researcher
RUN pip install --no-cache-dir /app/eva-researcher

WORKDIR /app/eva-researcher

EXPOSE 9300

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9300/health || exit 1

CMD ["uvicorn", "eva_researcher.main:app", "--host", "0.0.0.0", "--port", "9300"]
