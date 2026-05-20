
from pathlib import Path
compose = Path('/home/aza/The_Hive/docker-compose.yml')
text = compose.read_text(encoding='utf-8')
if '\n  hermes:\n' not in text:
    block = """  hermes:
    build:
      context: .
      dockerfile: src/eva-hermes/Dockerfile
    ports:
      - 9500:9500
    environment:
      - LLM_BACKEND=${HIVE_LLM_BACKEND:-vllm}
      - VLLM_HOST=${HIVE_VLLM_HOST:-vllm}
      - VLLM_PORT=${HIVE_VLLM_PORT:-8000}
      - OLLAMA_HOST=${HIVE_OLLAMA_HOST:-host.docker.internal}
      - OLLAMA_PORT=${HIVE_OLLAMA_PORT:-11434}
      - COUNCIL_VLLM_HOST=${HIVE_VLLM_HOST:-vllm}
      - COUNCIL_VLLM_PORT=${HIVE_VLLM_PORT:-8000}
      - HERMES_EXPERT_COORDINATOR_ROLE=${HERMES_EXPERT_COORDINATOR_ROLE:-general}
      - HERMES_EXPERT_TECHNICAL_ROLE=${HERMES_EXPERT_TECHNICAL_ROLE:-research}
      - HERMES_EXPERT_TRADING_ROLE=${HERMES_EXPERT_TRADING_ROLE:-banker}
      - HERMES_EXPERT_MACRO_NEWS_ROLE=${HERMES_EXPERT_MACRO_NEWS_ROLE:-research}
      - HERMES_EXPERT_DEVELOPMENT_ROLE=${HERMES_EXPERT_DEVELOPMENT_ROLE:-code}
      - HERMES_EXPERT_DEVELOPMENT_BACKEND=${HERMES_EXPERT_DEVELOPMENT_BACKEND:-ollama}
    extra_hosts:
      - host.docker.internal:host-gateway
    networks:
      - hive-net
    deploy:
      replicas: 1
      restart_policy:
        condition: any
"""
    text = text.replace('\n  nexus:\n', '\n' + block + '  nexus:\n', 1)
    compose.write_text(text, encoding='utf-8')
nginx = Path('/home/aza/The_Hive/src/eva-nexus/nginx.conf')
nginx_text = nginx.read_text(encoding='utf-8')
if 'location /api/hermes/' not in nginx_text:
    marker = '    location = /api/banker/health {'
    block = """    location /api/hermes/ {
        proxy_pass http://host.docker.internal:9500/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

"""
    nginx_text = nginx_text.replace(marker, block + marker, 1)
    nginx.write_text(nginx_text, encoding='utf-8')
print('ok')
