FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances partagées
COPY src/shared /app/shared
COPY src/openclaw /app/openclaw
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir aiohttp \
    && pip install --no-cache-dir /app/shared \
    && pip install --no-cache-dir /app/openclaw

# Copier eva-substrate
COPY src/eva-substrate /app/eva-substrate
RUN pip install --no-cache-dir /app/eva-substrate

WORKDIR /app/eva-substrate

# Port exposé
EXPOSE 8400

# Commande de démarrage
CMD ["uvicorn", "eva_substrate.main:app", "--host", "0.0.0.0", "--port", "8400"]
