FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /app/shared

COPY src/eva-wraith /app/eva-wraith
RUN pip install --no-cache-dir /app/eva-wraith

WORKDIR /app/eva-wraith

EXPOSE 9400

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9400/health || exit 1

CMD ["uvicorn", "eva_wraith.main:app", "--host", "0.0.0.0", "--port", "9400"]
