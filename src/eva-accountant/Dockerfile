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

# Copier eva-accountant
COPY src/eva-accountant /app/eva-accountant
RUN pip install --no-cache-dir /app/eva-accountant

WORKDIR /app/eva-accountant

# Port exposé
EXPOSE 8500

# Commande de démarrage
CMD ["uvicorn", "eva_accountant.main:app", "--host", "0.0.0.0", "--port", "8500"]
