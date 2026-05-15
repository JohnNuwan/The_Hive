"""Exports partages de THE HIVE avec chargement paresseux.

Ce module evite de charger les dependances lourdes comme `torch` quand un
outil leger, par exemple l'agent follower client, a uniquement besoin des
modeles Pydantic.
"""

from __future__ import annotations

from typing import Any

_MODEL_EXPORTS = {
    "AccountBalance",
    "AgentMessage",
    "AgentMessageType",
    "AgentStatus",
    "AuditRecord",
    "BaseHealthResponse",
    "ChatMessage",
    "ConnectorMode",
    "EventEnvelope",
    "ExecutionEventEnvelope",
    "GPUMetrics",
    "HardwareMetrics",
    "Intent",
    "IntentType",
    "InvestmentThesisEnvelope",
    "MarketContextEnvelope",
    "MessageRole",
    "OrderSource",
    "OrderType",
    "Position",
    "PromotionReportEnvelope",
    "PropFirmAccount",
    "RiskStatus",
    "RuntimeMode",
    "SecurityEvent",
    "SecuritySeverity",
    "TradeAction",
    "TradeOrder",
    "TradingContextEnvelope",
    "TradingDecisionEnvelope",
    "TrainingRunEnvelope",
}

_MATH_EXPORTS = {"calculate_cvar", "calculate_var", "inv_symlog", "symlog"}
_CONFIG_EXPORTS = {"Settings", "get_settings"}
_TELEMETRY_EXPORTS = {"Telemetry"}
_CIRCUIT_BREAKER_EXPORTS = {"CircuitBreaker", "CircuitBreakerOpenError"}
_GRPC_EXPORTS = {"SwarmGRPCClient"}

__all__ = sorted(
    _MODEL_EXPORTS
    | _MATH_EXPORTS
    | _CONFIG_EXPORTS
    | _TELEMETRY_EXPORTS
    | _CIRCUIT_BREAKER_EXPORTS
    | _GRPC_EXPORTS
)


def __getattr__(name: str) -> Any:
    """Charge un export partage uniquement quand il est demande.

    Args:
        name (str): Nom d'export recherche.

    Returns:
        Any: Objet exporte par le sous-module correspondant.

    Raises:
        AttributeError: Si l'export n'existe pas.
    """

    if name in _MODEL_EXPORTS:
        from shared import models

        return getattr(models, name)
    if name in _MATH_EXPORTS:
        from shared import math_ops

        return getattr(math_ops, name)
    if name in _CONFIG_EXPORTS:
        from shared import config

        return getattr(config, name)
    if name in _TELEMETRY_EXPORTS:
        from shared import telemetry

        return getattr(telemetry, name)
    if name in _CIRCUIT_BREAKER_EXPORTS:
        from shared import circuit_breaker

        return getattr(circuit_breaker, name)
    if name in _GRPC_EXPORTS:
        from shared import grpc_client

        return getattr(grpc_client, name)
    raise AttributeError(f"Export shared inconnu: {name}")
