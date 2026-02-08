# EVA Core with Integrated Ollama & CUDA support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# 1. Installer les dépendances système (Python, Curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# 2. Installer Ollama nativement
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Copier les dépendances partagées
COPY src/shared /app/shared
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir /app/shared

# 4. Copier eva-core
COPY src/eva-core /app/eva-core
RUN pip3 install --no-cache-dir /app/eva-core

WORKDIR /app/eva-core

# Port exposé (FastAPI)
EXPOSE 8000
# Port exposé (Ollama)
EXPOSE 11434

# 5. Script de démarrage double (Ollama + Uvicorn)
RUN echo '#!/bin/bash\nollama serve & \nsleep 5\nuvicorn eva_core.main:app --host 0.0.0.0 --port 8000' > /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
