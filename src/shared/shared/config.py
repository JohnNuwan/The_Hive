"""
Configuration Centralisée - THE HIVE
Utilise pydantic-settings pour validation et chargement .env
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration globale THE HIVE"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERAL
    # ═══════════════════════════════════════════════════════════════════════════
    app_name: str = "THE HIVE"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    timezone: str = "Europe/Paris"

    # ═══════════════════════════════════════════════════════════════════════════
    # EVA CORE API
    # ═══════════════════════════════════════════════════════════════════════════
    core_api_host: str = "0.0.0.0"
    core_api_port: int = 8000
    core_api_workers: int = 4
    internal_secret_key: SecretStr
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    prompt_master_templates_dir: str = "Documentation/Biblio_IA"

    # ═══════════════════════════════════════════════════════════════════════════
    # LLM SERVER
    # ═══════════════════════════════════════════════════════════════════════════
    vllm_host: str = "vllm-server"
    vllm_port: int = 8000
    vllm_model: str = "google/gemma-3-4b-it"
    llm_backend: Literal["ollama", "vllm"] = "vllm"
    # Alternative: Ollama
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "llama3:8b"
    use_ollama: bool = True  # True pour dev, False pour prod (vLLM)

    # EAGLE-3 Speculative Decoding (latence ÷3)
    eagle_enabled: bool = True
    eagle_draft_model: str = "yuhuili/EAGLE3-Gemma3-4B-IT"
    eagle_num_speculative_tokens: int = 5


    # ═══════════════════════════════════════════════════════════════════════════
    # THE COUNCIL (Model Swapping)
    # ═══════════════════════════════════════════════════════════════════════════
    council_model_general: str = "llama3.2:1b"
    council_model_research: str = "qwen2.5:3b"
    council_model_banker: str = "qwen2.5-coder:3b"

    # ═══════════════════════════════════════════════════════════════════════════
    # REDIS
    # ═══════════════════════════════════════════════════════════════════════════
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: SecretStr = Field(default=SecretStr(""))
    redis_db: int = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # QDRANT
    # ═══════════════════════════════════════════════════════════════════════════
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: SecretStr = Field(default=SecretStr(""))
    qdrant_collection_conversations: str = "conversations"
    qdrant_collection_documents: str = "documents"

    # ═══════════════════════════════════════════════════════════════════════════
    # NEO4J (Graph Memory)
    # ═══════════════════════════════════════════════════════════════════════════
    neo4j_host: str = "localhost"
    neo4j_port: int = 7687
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = Field(default=SecretStr("devpassword"))

    # ═══════════════════════════════════════════════════════════════════════════
    # TIMESCALEDB
    # ═══════════════════════════════════════════════════════════════════════════
    timescale_host: str = "localhost"
    timescale_port: int = 5432
    timescale_db: str = "thehive"
    timescale_user: str = "eva"
    timescale_password: SecretStr = Field(default=SecretStr(""))

    @property
    def database_url(self) -> str:
        """URL de connexion PostgreSQL/TimescaleDB"""
        password = self.timescale_password.get_secret_value()
        return (
            f"postgresql://{self.timescale_user}:{password}"
            f"@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"
        )

    @property
    def redis_url(self) -> str:
        """URL de connexion Redis"""
        password = self.redis_password.get_secret_value()
        if password:
            return f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ═══════════════════════════════════════════════════════════════════════════
    # TRADING (Banker)
    # ═══════════════════════════════════════════════════════════════════════════
    banker_api_host: str = "localhost"
    banker_api_port: int = 8100
    mt5_magic_number: int = 12345
    mock_mt5: bool = True  # True pour dev sans MT5 réel
    paper_trading: bool = True
    mt5_login: int = 0
    mt5_password: SecretStr = Field(default=SecretStr(""))
    mt5_server: str = ""

    # Constitution Loi 2 - Limites de risque
    risk_max_daily_drawdown_percent: float = 4.0
    risk_max_total_drawdown_percent: float = 8.0
    risk_max_single_trade_percent: float = 1.0
    risk_max_open_positions: int = 3
    risk_anti_tilt_losses: int = 2
    risk_anti_tilt_duration_hours: int = 24
    risk_news_filter_minutes: int = 30

    # ═══════════════════════════════════════════════════════════════════════════
    # SECURITY & SUPPORT SERVICES
    # ═══════════════════════════════════════════════════════════════════════════
    sentinel_api_host: str = "localhost"
    sentinel_api_port: int = 8200
    compliance_api_port: int = 8300
    substrate_api_port: int = 8400
    accountant_api_port: int = 8500
    lab_api_port: int = 8600
    rwa_api_port: int = 8700
    kernel_api_port: int = 8800
    shadow_api_port: int = 8900
    builder_api_port: int = 9000
    nervous_api_port: int = 9090
    muse_api_port: int = 9100
    sage_api_port: int = 9200
    researcher_api_port: int = 9300
    wraith_api_port: int = 9400

    # ═══════════════════════════════════════════════════════════════════════════
    # EVA LAB (Feature Flags — Sprint 5)
    # ═══════════════════════════════════════════════════════════════════════════
    enable_dreamer_training: bool = False  # RTX 3090 only — active DreamerV3 training
    enable_shadow_learning: bool = True    # Collecte passive des données pour DreamerV3
    shadow_learning_buffer_size: int = 10000  # Nombre max de transitions en mémoire
    shadow_learning_flush_interval: int = 300  # Flush sur disque toutes les N secondes

    # Constitution Loi 0 - Seuils température
    gpu_temp_warning: float = 80.0
    gpu_temp_critical: float = 90.0

    # ═══════════════════════════════════════════════════════════════════════════
    # NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_alerts: str = ""
    discord_webhook_trades: str = ""

    # ═══════════════════════════════════════════════════════════════════════════
    # COMFYUI (Image/Video Generation — GPU Server)
    # ═══════════════════════════════════════════════════════════════════════════
    comfyui_host: str = "192.168.1.5"
    comfyui_port: int = 8188

    # ═══════════════════════════════════════════════════════════════════════════
    # CIVITAI
    # ═══════════════════════════════════════════════════════════════════════════
    civitai_api_key: str = ""



@lru_cache
def get_settings() -> Settings:
    """Retourne une instance cachée des settings"""
    return Settings()
