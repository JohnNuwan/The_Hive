"""
THE HIVE - Shared Python Models (Pydantic V2)
═══════════════════════════════════════════════════════════════════════════════
Modèles de données partagés entre tous les agents Python.
Ces modèles définissent les contrats de données pour la communication inter-agents.

Usage:
    from eva_shared.models import TradeOrder, RiskCheck, AgentMessage
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class TradeAction(str, Enum):
    """Type d'action de trading."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Type d'ordre."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    """Statut d'un ordre."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSource(str, Enum):
    """Source de l'ordre."""
    MANUAL = "manual"
    ALGO = "algo"
    COPIER = "copier"


class AgentType(str, Enum):
    """Types d'experts E.V.A."""
    CORE = "core"
    BANKER = "banker"
    SHADOW = "shadow"
    BUILDER = "builder"
    SENTINEL = "sentinel"
    MUSE = "muse"
    SAGE = "sage"
    WRAITH = "wraith"
    RESEARCHER = "researcher"
    ADVOCATE = "advocate"
    SOVEREIGN = "sovereign"


class IntentType(str, Enum):
    """Types d'intentions utilisateur."""
    TRADING_ORDER = "TRADING_ORDER"
    TRADING_QUERY = "TRADING_QUERY"
    OSINT_SEARCH = "OSINT_SEARCH"
    SYSTEM_COMMAND = "SYSTEM_COMMAND"
    CREATIVE_REQUEST = "CREATIVE_REQUEST"
    HEALTH_QUERY = "HEALTH_QUERY"
    SECURITY_ALERT = "SECURITY_ALERT"
    GENERAL_CHAT = "GENERAL_CHAT"


class Severity(str, Enum):
    """Niveaux de sévérité pour alertes/logs."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MoodLevel(str, Enum):
    """État d'humeur de l'utilisateur."""
    FOCUSED = "focused"
    RELAXED = "relaxed"
    STRESSED = "stressed"
    UNKNOWN = "unknown"


class StressLevel(str, Enum):
    """Niveau de stress (Loi 1 - Santé)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TradeOrder(BaseModel):
    """
    Ordre de trading.
    Respecte les contraintes de la Loi 2 (Protection du Capital).
    """
    id: UUID = Field(default_factory=uuid4)
    symbol: str = Field(..., min_length=1, max_length=20, examples=["XAUUSD", "EURUSD"])
    action: TradeAction
    volume: Decimal = Field(..., gt=0, le=10, description="Taille du lot (max 10)")
    
    # Stop Loss OBLIGATOIRE (ROE Trading)
    stop_loss_price: Decimal = Field(..., gt=0, description="Prix du SL - OBLIGATOIRE")
    take_profit_price: Optional[Decimal] = Field(None, gt=0)
    
    entry_price: Optional[Decimal] = Field(None, description="Prix limite (None = Market)")
    order_type: OrderType = OrderType.MARKET
    
    magic_number: int = Field(default=12345, description="ID de stratégie")
    comment: Optional[str] = Field(None, max_length=100)
    source: OrderSource = OrderSource.MANUAL
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('volume')
    @classmethod
    def validate_volume_precision(cls, v: Decimal) -> Decimal:
        """Le volume doit avoir max 2 décimales."""
        if v.as_tuple().exponent < -2:
            raise ValueError("Volume precision must be max 2 decimal places")
        return v


class TradeOrderResult(BaseModel):
    """Résultat d'exécution d'un ordre."""
    order_id: UUID
    ticket: int = Field(..., description="Numéro de ticket MT5")
    symbol: str
    action: TradeAction
    volume: Decimal
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Optional[Decimal]
    
    status: OrderStatus
    execution_time_ms: int
    slippage_points: int = 0
    
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None


class Position(BaseModel):
    """Position ouverte sur un compte."""
    ticket: int
    symbol: str
    action: TradeAction
    volume: Decimal
    open_price: Decimal
    current_price: Decimal
    stop_loss: Decimal
    take_profit: Optional[Decimal]
    
    profit: Decimal = Field(description="P&L en devise du compte")
    profit_pips: Decimal = Field(description="P&L en pips")
    swap: Decimal = Decimal("0")
    
    open_time: datetime
    magic_number: int
    comment: Optional[str] = None


class AccountBalance(BaseModel):
    """État du compte de trading."""
    login: int
    server: str
    currency: str = "USD"
    
    balance: Decimal
    equity: Decimal
    margin: Decimal
    free_margin: Decimal
    margin_level_percent: Optional[Decimal] = None
    
    profit: Decimal = Field(description="P&L positions ouvertes")
    
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RiskCalculationRequest(BaseModel):
    """Demande de calcul de risque."""
    symbol: str
    stop_loss_distance_pips: Decimal = Field(..., gt=0)
    risk_percent: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.1"), le=Decimal("1.0"))
    account_id: Optional[UUID] = None
    
    @field_validator('risk_percent')
    @classmethod
    def validate_max_risk(cls, v: Decimal) -> Decimal:
        """Loi 2: Max 1% de risque par trade."""
        if v > Decimal("1.0"):
            raise ValueError("Risk cannot exceed 1% per trade (Constitution Law 2)")
        return v


class RiskCalculationResult(BaseModel):
    """Résultat du calcul de risque."""
    recommended_lot_size: Decimal
    max_allowed_lot_size: Decimal
    risk_amount: Decimal = Field(description="Montant risqué en devise")
    account_equity: Decimal
    stop_loss_pips: Decimal
    tick_value: Decimal
    
    warnings: list[str] = Field(default_factory=list)


class RiskStatus(BaseModel):
    """
    État actuel du risque d'un compte.
    Vérifie les contraintes de la Loi 2.
    """
    account_id: UUID
    
    # Drawdown (Loi 2: Max 4% journalier, 8% total)
    daily_drawdown_percent: Decimal = Field(ge=0)
    daily_drawdown_limit: Decimal = Field(default=Decimal("4.0"))
    total_drawdown_percent: Decimal = Field(ge=0)
    total_drawdown_limit: Decimal = Field(default=Decimal("8.0"))
    
    # Anti-Tilt
    anti_tilt_active: bool = False
    anti_tilt_expires_at: Optional[datetime] = None
    consecutive_losses: int = 0
    
    # Décision
    trading_allowed: bool = True
    restrictions: list[str] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def check_trading_allowed(self) -> 'RiskStatus':
        """Calcule si le trading est autorisé selon la Constitution."""
        restrictions = []
        
        if self.daily_drawdown_percent >= self.daily_drawdown_limit - Decimal("0.05"):
            restrictions.append("Daily drawdown limit reached (Law 2)")
            
        if self.total_drawdown_percent >= self.total_drawdown_limit - Decimal("0.05"):
            restrictions.append("Total drawdown limit reached (Law 2)")
            
        if self.anti_tilt_active:
            restrictions.append("Anti-tilt active (2 consecutive losses)")
        
        self.restrictions = restrictions
        self.trading_allowed = len(restrictions) == 0
        return self


class RiskRejection(BaseModel):
    """Rejet d'un ordre par le système de risque."""
    rejected: bool = True
    reason: str
    details: str
    constitution_reference: str = Field(examples=["Loi 2 - Protection du Capital"])


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT COMMUNICATION MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentMessage(BaseModel):
    """
    Message standardisé pour la communication inter-agents.
    Utilisé via Redis Pub/Sub.
    """
    id: UUID = Field(default_factory=uuid4)
    source_agent: AgentType
    target_agent: AgentType
    
    action: str = Field(..., description="Action à effectuer")
    payload: dict[str, Any] = Field(default_factory=dict)
    
    priority: int = Field(default=3, ge=1, le=5, description="1=max, 5=min")
    timeout_seconds: int = Field(default=30, le=300)
    
    # Tracing
    correlation_id: Optional[UUID] = Field(default=None, description="Pour lier les messages")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Response tracking
    requires_response: bool = True
    response_topic: Optional[str] = None


class AgentResponse(BaseModel):
    """Réponse d'un agent à une requête."""
    request_id: UUID
    source_agent: AgentType
    
    success: bool
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentStatus(BaseModel):
    """État d'un agent."""
    agent: AgentType
    status: str = Field(pattern="^(active|idle|busy|offline|error)$")
    
    last_active: datetime
    vram_usage_mb: Optional[int] = None
    queue_depth: int = 0
    
    current_task: Optional[str] = None
    uptime_seconds: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT & CONVERSATION MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class UserContext(BaseModel):
    """
    Contexte utilisateur pour personnalisation et Loi 1.
    """
    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Device
    device: str = Field(default="desktop", pattern="^(mobile|desktop|halo)$")
    
    # Mood & Health (Loi 1 - Épanouissement)
    mood: MoodLevel = MoodLevel.UNKNOWN
    sleep_hours: Optional[float] = Field(None, ge=0, le=24)
    stress_level: StressLevel = StressLevel.LOW
    heart_rate_bpm: Optional[int] = Field(None, ge=30, le=220)


class Intent(BaseModel):
    """Intent détecté dans un message utilisateur."""
    type: IntentType
    confidence: float = Field(..., ge=0, le=1)
    entities: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Message de conversation."""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    
    intent: Optional[Intent] = None
    routed_to: Optional[AgentType] = None
    
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[int] = None
    
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Requête de chat entrante."""
    content: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[UUID] = None
    context: Optional[UserContext] = None


class ChatResponse(BaseModel):
    """Réponse de chat."""
    message_id: UUID
    session_id: UUID
    content: str
    
    intent: Optional[Intent] = None
    routed_to: AgentType = AgentType.CORE
    
    processing_time_ms: int
    tokens_used: int
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityEvent(BaseModel):
    """Événement de sécurité détecté par The Sentinel."""
    id: UUID = Field(default_factory=uuid4)
    event_type: str
    severity: Severity
    
    source_ip: Optional[str] = None
    target_service: Optional[str] = None
    
    description: str
    action_taken: str
    
    raw_data: Optional[dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AuditRecord(BaseModel):
    """
    Enregistrement d'audit (Black Box).
    Immutable - utilisé pour la traçabilité.
    """
    event_type: str
    actor: str  # admin, eva_core, banker, kernel
    action: str
    target: Optional[str] = None
    
    old_value: Optional[dict[str, Any]] = None
    new_value: Optional[dict[str, Any]] = None
    
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SystemHealth(BaseModel):
    """État de santé du système."""
    status: str = Field(..., pattern="^(healthy|degraded|critical)$")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    components: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="État de chaque composant (llm_server, database, redis, gpu)"
    )


class HardwareMetrics(BaseModel):
    """Métriques hardware collectées par The Keeper."""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # GPU (RTX 3090)
    gpu_temp_celsius: float = Field(..., ge=0, le=150)
    gpu_power_watts: float = Field(..., ge=0)
    gpu_vram_used_mb: int = Field(..., ge=0)
    gpu_vram_total_mb: int = Field(default=24576)  # 24GB
    gpu_utilization_percent: float = Field(..., ge=0, le=100)
    
    # CPU
    cpu_load_percent: float = Field(..., ge=0, le=100)
    
    # RAM
    ram_used_mb: int
    ram_total_mb: int
    
    # Disk
    disk_used_gb: float
    disk_total_gb: float
    
    @property
    def gpu_temp_critical(self) -> bool:
        """Loi 0: Température critique si > 90°C."""
        return self.gpu_temp_celsius > 90


# ═══════════════════════════════════════════════════════════════════════════════
# PROP FIRM / HYDRA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PropFirmAccount(BaseModel):
    """Compte Prop Firm pour le système Hydra."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    broker: str = Field(..., examples=["FTMO", "FundedNext", "The5ers"])
    
    login: int
    server: str
    
    phase: str = Field(..., pattern="^(challenge|verification|funded)$")
    status: str = Field(default="active", pattern="^(active|suspended|closed)$")
    
    initial_balance: Decimal
    current_balance: Decimal
    currency: str = "USD"
    
    is_master: bool = False
    copy_enabled: bool = True
    
    created_at: datetime = Field(default_factory=datetime.now)


class CopyTradeRequest(BaseModel):
    """Demande de réplication d'un trade sur tous les comptes Hydra."""
    order: TradeOrder
    target_accounts: list[UUID] = Field(default_factory=list, description="Tous si vide")
    scaling_mode: str = Field(default="proportional", pattern="^(fixed|proportional)$")


class CopyTradeResult(BaseModel):
    """Résultat de la réplication d'un trade."""
    source_ticket: int
    copies: list[dict[str, Any]]
    total_latency_ms: int
    success_count: int
    failure_count: int
