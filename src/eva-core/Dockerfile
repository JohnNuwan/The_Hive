# EVA Core with Integrated Ollama & CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# 1. Installer les dépendances système (Curl, zstd)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Fix pip
RUN pip install --upgrade pip setuptools wheel

# 2. Installer Ollama nativement
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Copier les dépendances partagées
COPY src/shared /app/shared
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install --no-cache-dir /app/shared

# 4. Copier eva-core et installer (avec dépendances explicites car pip est capricieux)
COPY src/eva-core /app/eva-core
RUN python3 -m pip install --no-cache-dir \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "langgraph>=0.0.20" \
    "langchain>=0.1.0" \
    "langchain-community>=0.0.10" \
    "langchain-ollama>=0.1.0" \
    "qdrant-client>=1.7.0" \
    "httpx>=0.26.0" \
    "python-multipart>=0.0.6" \
    "psutil>=5.9.0" \
    "docker>=7.0.0"
RUN python3 -m pip install --no-cache-dir /app/eva-core

WORKDIR /app/eva-core

# Port exposé (FastAPI)
EXPOSE 8000
# Port exposé (Ollama)
EXPOSE 11434

# 5. Script de démarrage double (Ollama + Uvicorn)
RUN echo '#!/bin/bash\nollama serve & \nsleep 10\necho "--- PIP LIST ---"\npython3 -m pip list\necho "--- STARTING UVICORN ---"\nexport PYTHONPATH=$PYTHONPATH:/app/eva-core\npython3 -m uvicorn eva_core.main:app --host 0.0.0.0 --port 8000' > /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
