# vLLM Service Dockerfile
FROM vllm/vllm-openai:latest

# Enhance with bitsandbytes for 4-bit quantization support (AWQ/GPTQ)
RUN pip install bitsandbytes accelerate

# Expose the API port
EXPOSE 8000

# Entrypoint is handled by the base image, arguments passed via docker-compose
