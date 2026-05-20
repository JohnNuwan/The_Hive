FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
RUN pip install --no-cache-dir /app/shared

COPY src/eva-hermes /app/eva-hermes
RUN pip install --no-cache-dir /app/eva-hermes

WORKDIR /app/eva-hermes

EXPOSE 9500

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9500/health || exit 1

CMD ["uvicorn", "eva_hermes.main:app", "--host", "0.0.0.0", "--port", "9500"]
