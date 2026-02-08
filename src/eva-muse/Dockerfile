FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /app/shared

COPY src/eva-muse /app/eva-muse
RUN pip install --no-cache-dir /app/eva-muse

WORKDIR /app/eva-muse

EXPOSE 9100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9100/health || exit 1

CMD ["uvicorn", "eva_muse.main:app", "--host", "0.0.0.0", "--port", "9100"]
