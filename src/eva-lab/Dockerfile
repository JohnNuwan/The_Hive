FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /app

# Installe JAX avec CUDA en verrouillant les versions critiques.
RUN pip install --no-cache-dir --upgrade \
    jax==0.4.23 \
    jaxlib==0.4.23+cuda11.cudnn86 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    tensorflow-probability==0.23.0 \
    dm-haiku==0.0.11 \
    optax==0.1.7 \
    chex==0.1.85 \
    distrax==0.1.5 \
    "scipy<1.13" \
    gymnasium

# Copie puis installe la bibliotheque partagee.
COPY src/shared /app/shared
RUN python3 -m pip install grpcio-tools && \
    python3 -m grpc_tools.protoc -I/app/shared/shared/proto \
    --python_out=/app/shared/shared/proto \
    --grpc_python_out=/app/shared/shared/proto \
    /app/shared/shared/proto/swarm.proto
RUN pip install --no-cache-dir /app/shared

# Copie puis installe eva-lab avec toutes ses dependances declarees.
COPY src/eva-lab /app/eva-lab
RUN pip install --no-cache-dir /app/eva-lab

WORKDIR /app/eva-lab

EXPOSE 8600

CMD ["uvicorn", "eva_lab.main:app", "--host", "0.0.0.0", "--port", "8600"]
