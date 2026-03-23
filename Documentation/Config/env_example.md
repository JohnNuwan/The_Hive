# Variables d'Environnement - THE HIVE

> **Version**: 1.0.0  
> **Usage**: Copier ce fichier vers `.env` et remplir les valeurs

---

##  Instructions

1. Copiez ce fichier : `cp env_example.md .env`
2. Remplissez toutes les valeurs `CHANGE_ME`
3. Ne commitez JAMAIS le fichier `.env` (ajouté à `.gitignore`)
4. Les secrets sensibles doivent être stockés dans The Vault (YubiKey)

---

##  Network & General

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
ENVIRONMENT=production  # development | staging | production
LOG_LEVEL=INFO          # DEBUG | INFO | WARNING | ERROR
TIMEZONE=Europe/Paris

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
# Internal network (vmbr1)
INTERNAL_NETWORK=10.0.1.0/24
CORE_VM_IP=10.0.1.100
TRADING_VM_IP=10.0.1.200
SENTINEL_VM_IP=10.0.1.150

# Tailscale (secure remote access)
TAILSCALE_AUTH_KEY=tskey-auth-CHANGE_ME
TAILSCALE_HOSTNAME=the-hive
```

---

##  EVA Core

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# EVA CORE API
# ═══════════════════════════════════════════════════════════════════════════════
CORE_API_HOST=0.0.0.0
CORE_API_PORT=8000
CORE_API_WORKERS=4
CORE_API_SECRET_KEY=CHANGE_ME_GENERATE_WITH_openssl_rand_hex_32

# JWT Authentication
JWT_SECRET_KEY=CHANGE_ME_GENERATE_WITH_openssl_rand_hex_64
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# ═══════════════════════════════════════════════════════════════════════════════
# LLM SERVER (vLLM)
# ═══════════════════════════════════════════════════════════════════════════════
VLLM_HOST=localhost
VLLM_PORT=8080
VLLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
VLLM_QUANTIZATION=awq  # awq | gptq | none
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=8192
VLLM_TENSOR_PARALLEL_SIZE=1

# ═══════════════════════════════════════════════════════════════════════════════
# REDIS
# ═══════════════════════════════════════════════════════════════════════════════
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=CHANGE_ME
REDIS_DB=0
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_TIMEOUT=5

# ═══════════════════════════════════════════════════════════════════════════════
# QDRANT (Vector Database)
# ═══════════════════════════════════════════════════════════════════════════════
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=CHANGE_ME
QDRANT_COLLECTION_CONVERSATIONS=conversations
QDRANT_COLLECTION_OSINT=osint
QDRANT_COLLECTION_TRADING=trading_knowledge
QDRANT_COLLECTION_DOCUMENTS=documents
```

---

##  The Banker (Trading)

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# BANKER API
# ═══════════════════════════════════════════════════════════════════════════════
BANKER_API_HOST=0.0.0.0
BANKER_API_PORT=8100
BANKER_API_WORKERS=2

# ═══════════════════════════════════════════════════════════════════════════════
# METATRADER 5
# ═══════════════════════════════════════════════════════════════════════════════
# Master Account (Primary)
MT5_MASTER_LOGIN=CHANGE_ME
MT5_MASTER_PASSWORD=CHANGE_ME
MT5_MASTER_SERVER=FTMO-Demo

# Magic Number for identifying EVA's trades
MT5_MAGIC_NUMBER=12345

# ═══════════════════════════════════════════════════════════════════════════════
# TIMESCALEDB
# ═══════════════════════════════════════════════════════════════════════════════
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5432
TIMESCALE_DB=thehive
TIMESCALE_USER=eva
TIMESCALE_PASSWORD=CHANGE_ME
TIMESCALE_SSLMODE=prefer

# Connection string
DATABASE_URL=postgresql://eva:CHANGE_ME@localhost:5432/thehive

# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT (Law 2 - Constitution)
# ═══════════════════════════════════════════════════════════════════════════════
# These are HARDCODED in Constitution but can be referenced here
RISK_MAX_DAILY_DRAWDOWN_PERCENT=4.0
RISK_MAX_TOTAL_DRAWDOWN_PERCENT=8.0
RISK_MAX_SINGLE_TRADE_PERCENT=1.0
RISK_MAX_OPEN_POSITIONS=3
RISK_NEWS_FILTER_MINUTES=30
RISK_ANTI_TILT_LOSSES=2
RISK_ANTI_TILT_DURATION_HOURS=24

# ═══════════════════════════════════════════════════════════════════════════════
# ECONOMIC CALENDAR (News Filter)
# ═══════════════════════════════════════════════════════════════════════════════
FOREX_FACTORY_API_URL=https://nfs.faireconomy.media/ff_calendar_thisweek.json
```

---

##  The Sentinel (Security)

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# SENTINEL API
# ═══════════════════════════════════════════════════════════════════════════════
SENTINEL_API_HOST=0.0.0.0
SENTINEL_API_PORT=8200

# ═══════════════════════════════════════════════════════════════════════════════
# WAZUH SIEM
# ═══════════════════════════════════════════════════════════════════════════════
WAZUH_MANAGER_IP=10.0.1.150
WAZUH_API_PORT=55000
WAZUH_API_USER=wazuh
WAZUH_API_PASSWORD=CHANGE_ME

# ═══════════════════════════════════════════════════════════════════════════════
# THREAT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
ABUSEIPDB_API_KEY=CHANGE_ME
VIRUSTOTAL_API_KEY=CHANGE_ME
SHODAN_API_KEY=CHANGE_ME
```

---

##  Notifications

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# DISCORD
# ═══════════════════════════════════════════════════════════════════════════════
DISCORD_WEBHOOK_ALERTS=https://discord.com/api/webhooks/CHANGE_ME
DISCORD_WEBHOOK_TRADES=https://discord.com/api/webhooks/CHANGE_ME
DISCORD_WEBHOOK_SYSTEM=https://discord.com/api/webhooks/CHANGE_ME

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM (Optional)
# ═══════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN=CHANGE_ME
TELEGRAM_CHAT_ID=CHANGE_ME
```

---

##  Hardware Security

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# THE TABLET (USB Key with Laws)
# ═══════════════════════════════════════════════════════════════════════════════
TABLET_MOUNT_PATH=/mnt/THE_LAW
TABLET_CONSTITUTION_FILE=/mnt/THE_LAW/Constitution.toml
TABLET_KERNEL_HASH_FILE=/mnt/THE_LAW/kernel.sha512

# ═══════════════════════════════════════════════════════════════════════════════
# THE VAULT (YubiKey HSM)
# ═══════════════════════════════════════════════════════════════════════════════
VAULT_YUBIKEY_SLOT=9a
VAULT_PIV_PIN=CHANGE_ME  # 6-8 digits
# Note: Actual secrets are stored ON the YubiKey, not in .env

# ═══════════════════════════════════════════════════════════════════════════════
# THE WATCHDOG (ESP32)
# ═══════════════════════════════════════════════════════════════════════════════
WATCHDOG_USB_PORT=/dev/ttyUSB0
WATCHDOG_BAUD_RATE=115200
WATCHDOG_HEARTBEAT_INTERVAL_SECONDS=10
WATCHDOG_TIMEOUT_SECONDS=60
```

---

##  Development

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT ONLY (Never use in production)
# ═══════════════════════════════════════════════════════════════════════════════
DEBUG=false
TESTING=false
MOCK_MT5=false
PAPER_TRADING=false
DISABLE_KERNEL_VALIDATION=false  # NEVER set to true in prod!

# ═══════════════════════════════════════════════════════════════════════════════
# CI/CD
# ═══════════════════════════════════════════════════════════════════════════════
GIT_REPO=git@github.com:yourusername/the-hive.git
GITHUB_TOKEN=ghp_CHANGE_ME
```

---

##  Monitoring & Observability

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# PROMETHEUS
# ═══════════════════════════════════════════════════════════════════════════════
PROMETHEUS_PORT=9090
PROMETHEUS_RETENTION_DAYS=30

# ═══════════════════════════════════════════════════════════════════════════════
# GRAFANA
# ═══════════════════════════════════════════════════════════════════════════════
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=CHANGE_ME
```

---

##  Security Notes

> [!CAUTION]
> **Ne JAMAIS** :
> - Committer `.env` dans Git
> - Partager les mots de passe MT5 en clair
> - Utiliser `DISABLE_KERNEL_VALIDATION=true` en production
> - Stocker les clés API sensibles ailleurs que dans The Vault

> [!TIP]
> Utilisez `gpg` pour chiffrer votre fichier `.env` de backup :
> ```bash
> gpg -c .env  # Crée .env.gpg chiffré
> ```
