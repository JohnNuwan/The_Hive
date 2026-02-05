FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système pour JAX (si nécessaire)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

# Exposition du port API
EXPOSE 8500

CMD ["uvicorn", "eva_lab.main:app", "--host", "0.0.0.0", "--port", "8500"]
