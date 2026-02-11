# vLLM Service Dockerfile — EAGLE-3 Speculative Decoding
FROM vllm/vllm-openai:latest

# Enhance with quantization + EAGLE-3 speculative decoding support
RUN pip install bitsandbytes accelerate

# Expose the API port
EXPOSE 8000

# Health check for container readiness
HEALTHCHECK --interval=30s --timeout=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint is handled by the base image, arguments passed via docker-compose
