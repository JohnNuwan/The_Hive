FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances partagées
COPY src/shared /app/shared
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /app/shared

# Copier eva-compliance
COPY src/eva-compliance /app/eva-compliance
RUN pip install --no-cache-dir /app/eva-compliance

WORKDIR /app/eva-compliance

# Port exposé
EXPOSE 8300

# Commande de démarrage
CMD ["uvicorn", "eva_compliance.main:app", "--host", "0.0.0.0", "--port", "8300"]
