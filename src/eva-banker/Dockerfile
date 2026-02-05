FROM python:3.11-slim

WORKDIR /app

# Note: MT5 ne fonctionne que sur Windows
# Ce Dockerfile est pour le mode mock/paper trading

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
RUN pip install -e /app/shared

COPY src/eva-banker /app/eva-banker
RUN pip install -e /app/eva-banker

WORKDIR /app/eva-banker

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

CMD ["uvicorn", "eva_banker.main:app", "--host", "0.0.0.0", "--port", "8100"]
