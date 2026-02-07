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
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    timezone: str = "Europe/Paris"

    # ═══════════════════════════════════════════════════════════════════════════
    # EVA CORE API
    # ═══════════════════════════════════════════════════════════════════════════
    core_api_host: str = "0.0.0.0"
    core_api_port: int = 8000
    core_api_workers: int = 4
    jwt_secret_key: SecretStr = Field(default=SecretStr("dev-secret-change-in-prod"))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # ═══════════════════════════════════════════════════════════════════════════
    # LLM SERVER
    # ═══════════════════════════════════════════════════════════════════════════
    vllm_host: str = "localhost"
    vllm_port: int = 8080
    vllm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Alternative: Ollama
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "qwen2.5:7b"
    use_ollama: bool = True  # True pour dev, False pour prod (vLLM)

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
    banker_api_port: int = 8100
    mt5_magic_number: int = 12345
    mock_mt5: bool = True  # True pour dev sans MT5 réel
    paper_trading: bool = True

    # Constitution Loi 2 - Limites de risque
    risk_max_daily_drawdown_percent: float = 4.0
    risk_max_total_drawdown_percent: float = 8.0
    risk_max_single_trade_percent: float = 1.0
    risk_max_open_positions: int = 3
    risk_anti_tilt_losses: int = 2
    risk_anti_tilt_duration_hours: int = 24
    risk_news_filter_minutes: int = 30

    # ═══════════════════════════════════════════════════════════════════════════
    # SECURITY (Sentinel)
    # ═══════════════════════════════════════════════════════════════════════════
    sentinel_api_port: int = 8200
    compliance_api_port: int = 8300
    substrate_api_port: int = 8400
    accountant_api_port: int = 8500

    # Constitution Loi 0 - Seuils température
    gpu_temp_warning: float = 80.0
    gpu_temp_critical: float = 90.0

    # ═══════════════════════════════════════════════════════════════════════════
    # NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    discord_webhook_alerts: str = ""
    discord_webhook_trades: str = ""


@lru_cache
def get_settings() -> Settings:
    """Retourne une instance cachée des settings"""
    return Settings()
