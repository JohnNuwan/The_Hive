# EVA Core with Integrated Ollama & CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

# 1. Installer les dépendances système (Curl, zstd, build-essential pour compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    zstd \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Fix pip and protobuf versions for gRPC compatibility (Standardized lock)
RUN pip install --upgrade pip setuptools wheel && \
    pip uninstall -y protobuf google-protobuf grpcio grpcio-tools && \
    pip install "protobuf==4.25.3" "grpcio==1.62.2" "grpcio-tools==1.62.2"

# 2. Installer Ollama nativement (Cache Restore)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Copier les dépendances partagées
COPY src/shared /app/shared
RUN touch /app/shared/shared/proto/__init__.py && \
    python3 -m grpc_tools.protoc -I/app/shared/shared/proto \
            --python_out=/app/shared/shared/proto \
            --grpc_python_out=/app/shared/shared/proto \
            /app/shared/shared/proto/swarm.proto && \
    sed -i 's/import swarm_pb2 as swarm__pb2/from . import swarm_pb2 as swarm__pb2/' /app/shared/shared/proto/swarm_pb2_grpc.py
RUN python3 -m pip install --no-cache-dir /app/shared

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
EXPOSE 8080
# Port exposé (Ollama)
EXPOSE 11434

# 5. Script de démarrage double (Ollama + Uvicorn)
COPY src/eva-core/start.sh /app/start.sh
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
