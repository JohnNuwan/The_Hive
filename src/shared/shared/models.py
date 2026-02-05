"""
Modèles Pydantic Partagés - THE HIVE
Basé sur Documentation/Models/python_models.py
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class TradeAction(str, Enum):
    """Action de trading"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Type d'ordre"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSource(str, Enum):
    """Source de l'ordre"""
    VOICE = "VOICE"
    CHAT = "CHAT"
    API = "API"
    STRATEGY = "STRATEGY"
    COPY = "COPY"


class IntentType(str, Enum):
    """Types d'intent pour classification"""
    TRADING_ORDER = "TRADING_ORDER"
    POSITION_STATUS = "POSITION_STATUS"
    RISK_INQUIRY = "RISK_INQUIRY"
    GENERAL_CHAT = "GENERAL_CHAT"
    MEMORY_RECALL = "MEMORY_RECALL"
    OSINT_REQUEST = "OSINT_REQUEST"
    SYSTEM_COMMAND = "SYSTEM_COMMAND"
    SECURITY_ALERT = "SECURITY_ALERT"


class MessageRole(str, Enum):
    """Rôle du message dans la conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SecuritySeverity(str, Enum):
    """Niveau de sévérité sécurité"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessageType(str, Enum):
    """Type de message inter-agents"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ALERT = "alert"


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class TradeOrder(BaseModel):
    """Ordre de trading"""
    id: UUID = Field(default_factory=uuid4)
    symbol: str = Field(..., description="Symbole (ex: XAUUSD)")
    action: TradeAction
    volume: Decimal = Field(..., gt=0, le=10)
    stop_loss_price: Decimal | None = Field(None, description="Prix Stop Loss (obligatoire ROE)")
    take_profit_price: Decimal | None = None
    order_type: OrderType = OrderType.MARKET
    source: OrderSource = OrderSource.CHAT
    account_id: UUID | None = None
    magic_number: int = 12345
    comment: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {Decimal: str}


class Position(BaseModel):
    """Position ouverte MT5"""
    ticket: int
    symbol: str
    action: TradeAction
    volume: Decimal
    open_price: Decimal
    current_price: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    profit: Decimal
    swap: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    magic_number: int = 0
    open_time: datetime

    @property
    def pnl_pips(self) -> float:
        """Calcul P&L en pips (approximatif)"""
        diff = float(self.current_price - self.open_price)
        if self.action == TradeAction.SELL:
            diff = -diff
        # Simplification: 1 pip = 0.0001 pour forex, 0.1 pour or
        pip_size = 0.1 if "XAU" in self.symbol else 0.0001
        return diff / pip_size


class RiskStatus(BaseModel):
    """État actuel des risques (Loi 2)"""
    account_id: UUID
    timestamp: datetime = Field(default_factory=datetime.now)
    daily_drawdown_percent: Decimal = Decimal("0")
    total_drawdown_percent: Decimal = Decimal("0")
    open_positions_count: int = 0
    anti_tilt_active: bool = False
    anti_tilt_expires_at: datetime | None = None
    news_filter_active: bool = False
    trading_allowed: bool = True

    def check_trading_allowed(
        self,
        max_daily_dd: Decimal = Decimal("4.0"),
        max_total_dd: Decimal = Decimal("8.0"),
        max_positions: int = 3,
    ) -> tuple[bool, str | None]:
        """Vérifie si le trading est autorisé selon Constitution Loi 2"""
        if self.anti_tilt_active:
            return False, "ANTI_TILT_ACTIVE"
        if self.news_filter_active:
            return False, "NEWS_FILTER_ACTIVE"
        if self.daily_drawdown_percent >= max_daily_dd:
            return False, "DAILY_DRAWDOWN_LIMIT"
        if self.total_drawdown_percent >= max_total_dd:
            return False, "TOTAL_DRAWDOWN_LIMIT"
        if self.open_positions_count >= max_positions:
            return False, "MAX_POSITIONS_REACHED"
        return True, None


class AccountBalance(BaseModel):
    """Solde du compte MT5"""
    login: int
    server: str
    balance: Decimal
    equity: Decimal
    margin: Decimal = Decimal("0")
    free_margin: Decimal = Decimal("0")
    margin_level: float | None = None
    currency: str = "USD"
    leverage: int = 100
    timestamp: datetime = Field(default_factory=datetime.now)


class PropFirmAccount(BaseModel):
    """Compte Prop Firm (FTMO, FundedNext, etc.)"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    login: int
    server: str
    broker: str
    phase: str = Field(..., description="challenge|verification|funded")
    initial_balance: Decimal
    current_balance: Decimal
    max_daily_loss_percent: Decimal = Decimal("4.0")
    max_total_loss_percent: Decimal = Decimal("8.0")
    profit_target_percent: Decimal | None = None
    copy_enabled: bool = True
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMUNICATION MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class AgentMessage(BaseModel):
    """Message inter-agents via Redis Pub/Sub"""
    id: UUID = Field(default_factory=uuid4)
    type: AgentMessageType
    source_agent: str
    target_agent: str
    action: str
    payload: dict[str, Any] = {}
    correlation_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    ttl_seconds: int = 30

    def to_redis_channel(self) -> str:
        """Génère le nom du channel Redis"""
        return f"eva.{self.target_agent}.{self.type.value}s"


class ChatMessage(BaseModel):
    """Message de conversation"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = {}


class Intent(BaseModel):
    """Intent classifié par le routeur"""
    intent_type: IntentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: dict[str, Any] = {}
    target_expert: str = "core"
    raw_text: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class SecurityEvent(BaseModel):
    """Événement de sécurité"""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    source_ip: str | None = None
    target_service: str | None = None
    severity: SecuritySeverity
    description: str = ""
    details: dict[str, Any] = {}
    action_taken: str | None = None
    resolved: bool = False


class AuditRecord(BaseModel):
    """Enregistrement d'audit (Black Box - Loi 3)"""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    agent: str
    action: str
    details: dict[str, Any] = {}
    user_id: str | None = None
    session_id: UUID | None = None
    previous_hash: str = ""
    record_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Calcule le hash SHA-256 de l'enregistrement"""
        import hashlib
        import json

        data = {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "action": self.action,
            "details": self.details,
            "previous_hash": previous_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class GPUMetrics(BaseModel):
    """Métriques GPU (Loi 0 - monitoring température)"""
    name: str = "NVIDIA GeForce RTX 3090"
    temperature_celsius: float
    utilization_percent: float
    memory_used_mb: int
    memory_total_mb: int = 24576
    power_draw_watts: float = 0.0
    fan_speed_percent: int = 0

    def is_overheating(self, threshold: float = 90.0) -> bool:
        """Vérifie si le GPU surchauffe (Loi 0)"""
        return self.temperature_celsius >= threshold


class HardwareMetrics(BaseModel):
    """Métriques hardware système"""
    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_percent: float
    cpu_freq_mhz: float = 0.0
    ram_used_gb: float
    ram_total_gb: float
    swap_used_gb: float = 0.0
    disk_used_percent: float = 0.0
    gpu: GPUMetrics | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class HiveError(BaseModel):
    """Erreur standardisée THE HIVE"""
    code: str
    message: str
    severity: SecuritySeverity = SecuritySeverity.MEDIUM
    category: str = "GENERAL"
    recoverable: bool = True
    details: dict[str, Any] = {}
    constitution_reference: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
