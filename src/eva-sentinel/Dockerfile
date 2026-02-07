FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /app/shared

COPY src/eva-sentinel /app/eva-sentinel
RUN pip install --no-cache-dir /app/eva-sentinel

WORKDIR /app/eva-sentinel

EXPOSE 8200

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8200/health || exit 1

CMD ["uvicorn", "eva_sentinel.main:app", "--host", "0.0.0.0", "--port", "8200"]
