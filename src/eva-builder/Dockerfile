FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /app/shared

COPY src/eva-builder /app/eva-builder
RUN pip install --no-cache-dir /app/eva-builder

WORKDIR /app/eva-builder

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

CMD ["uvicorn", "eva_builder.main:app", "--host", "0.0.0.0", "--port", "9000"]
