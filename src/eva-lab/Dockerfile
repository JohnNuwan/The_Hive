FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système pour JAX (si nécessaire)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/shared /app/shared
COPY src/eva-lab /app/eva-lab

RUN pip install --no-cache-dir /app/shared
RUN pip install --no-cache-dir /app/eva-lab

# Exposition du port API
EXPOSE 8600

CMD ["uvicorn", "eva_lab.main:app", "--host", "0.0.0.0", "--port", "8600"]
