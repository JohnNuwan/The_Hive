"""
Shared - Modèles et Utilitaires Partagés THE HIVE
"""

from shared.models import (
    # Enums
    TradeAction,
    OrderType,
    OrderSource,
    IntentType,
    MessageRole,
    SecuritySeverity,
    # Trading
    TradeOrder,
    Position,
    RiskStatus,
    AccountBalance,
    PropFirmAccount,
    # Communication
    AgentMessage,
    AgentMessageType,
    # Chat
    ChatMessage,
    Intent,
    # Sécurité
    SecurityEvent,
    AuditRecord,
    SecuritySeverity,
    # Agences / Système
    AgentStatus,
    HardwareMetrics,
    GPUMetrics,
)
from shared.math_ops import symlog, inv_symlog
from shared.config import Settings, get_settings

__all__ = [
    # Enums
    "TradeAction",
    "OrderType",
    "OrderSource",
    "IntentType",
    "MessageRole",
    "SecuritySeverity",
    # Trading
    "TradeOrder",
    "Position",
    "RiskStatus",
    "AccountBalance",
    "PropFirmAccount",
    # Communication
    "AgentMessage",
    "AgentMessageType",
    # Chat
    "ChatMessage",
    "Intent",
    # Sécurité
    "SecurityEvent",
    "AuditRecord",
    # Système
    "HardwareMetrics",
    "GPUMetrics",
    "AgentStatus",
    # Config
    "Settings",
    "get_settings",
    # Math
    "symlog",
    "inv_symlog",
    "calculate_var",
    "calculate_cvar",
]
