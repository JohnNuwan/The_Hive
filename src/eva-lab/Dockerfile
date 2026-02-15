FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install JAX with CUDA support
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install additional research dependencies
RUN pip install --no-cache-dir \
    tensorflow-probability \
    dm-haiku \
    optax \
    chex \
    distrax \
    gymnasium

# Copy shared library
COPY src/shared /app/shared
RUN python3 -m pip install grpcio-tools && \
    python3 -m grpc_tools.protoc -I/app/shared/shared/proto \
            --python_out=/app/shared/shared/proto \
            --grpc_python_out=/app/shared/shared/proto \
            /app/shared/shared/proto/swarm.proto
RUN pip install --no-cache-dir /app/shared

# Copy eva-lab
COPY src/eva-lab /app/eva-lab
RUN pip install --no-cache-dir fastapi uvicorn

WORKDIR /app/eva-lab

EXPOSE 8600

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8600"]
