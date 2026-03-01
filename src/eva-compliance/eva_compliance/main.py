"""
EVA Compliance — Agent Juridique & Fiscal de THE HIVE.

Ce module implémente le « Keeper » (Expert L) du système d'experts.
Il est responsable de :
- L'écoute des trades profitables via Redis Pub/Sub.
- Le provisionnement automatique des taxes (URSSAF, 25% BNC).
- La gestion du compte escrow (fonds bloqués pour l'État).
- L'exposition de l'identité juridique de l'entité.
- La simulation fiscale prospective.
- L'historique détaillé des provisions et alertes compliance.

Architecture :
    - Passif : écoute les événements du Banker et provisionne.
    - Aucune action de trading, uniquement comptable.
    - Persistance sur disque (escrow_ledger.json).
"""

import logging
import asyncio
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from eva_compliance.legal_wrapper import LegalWrapper
from eva_compliance.tax_manager import TaxManager
from shared.redis_client import init_redis, get_redis_client
from shared.auth_middleware import InternalAuthMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════


class TaxSimulation(BaseModel):
    """Requête de simulation fiscale."""
    annual_revenue: float = Field(..., gt=0, description="Chiffre d'affaires annuel brut en EUR")
    regime: str = Field(default="micro_bnc", description="Régime: micro_bnc, reel_simplifie")
    activity: str = Field(default="trading", description="Activité: trading, saas, consulting")


class ComplianceAlert(BaseModel):
    """Alerte de conformité."""
    severity: str = Field(description="Sévérité: info, warning, critical")
    category: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ProvisionEntry(BaseModel):
    """Entrée de provision fiscale."""
    trade_id: str | None = None
    gross_profit: float
    tax_rate: float
    tax_amount: float
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application Compliance."""
    logger.info("⚖️ Démarrage EVA Compliance (Le Keeper)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Services
    app.state.legal = LegalWrapper()
    app.state.tax_manager = TaxManager()
    app.state.provision_history: deque[dict[str, Any]] = deque(maxlen=500)
    app.state.compliance_alerts: deque[dict[str, Any]] = deque(maxlen=200)

    # Tâches de fond
    asyncio.create_task(trade_listener(app.state.tax_manager))
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ EVA Compliance actif et à l'écoute du Banker")
    yield
    logger.info("🛑 Arrêt EVA Compliance")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA Compliance (Keeper) API",
    description="Agent Juridique, Fiscal & Compliance - THE HIVE",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(InternalAuthMiddleware)


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def trade_listener(tax_manager: TaxManager):
    """Écoute les trades profitables et provisionne les taxes."""
    redis = get_redis_client()

    async def handle_trade(channel: str, message: dict):
        logger.info(f"⚖️ Signal de profit reçu: {message}")
        result = tax_manager.process_trade_result(message)
        logger.info(f"📝 Résultat provision: {result.get('message', result.get('status'))}")

        # Enregistrer dans l'historique
        app.state.provision_history.append({
            "trade_data": message,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

    try:
        await redis.subscribe(["eva.compliance.trades"], handle_trade)
        await redis.listen()
    except Exception as e:
        logger.error(f"Erreur listener trades: {e}")


async def hard_heartbeat():
    """Signal haute fréquence pour l'Orchestrateur Core."""
    redis = get_redis_client()
    while True:
        try:
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "keeper",
            }
            await redis.cache_set("eva.keeper.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    """Vérifie la santé du module Compliance / Keeper."""
    return {
        "status": "online",
        "service": "compliance",
        "provisions_count": len(app.state.provision_history),
        "active_alerts": len(app.state.compliance_alerts),
    }


@app.get("/ledger", tags=["Fiscal"])
async def get_ledger():
    """Récupère l'état du compte escrow (fonds bloqués pour l'URSSAF)."""
    tax_manager: TaxManager = app.state.tax_manager
    return tax_manager.get_escrow_status()


@app.get("/identity", tags=["Juridique"])
async def get_identity():
    """Retourne l'identité juridique publique de l'entité (SIRET, propriétaire)."""
    legal: LegalWrapper = app.state.legal
    return legal.get_public_identity()


@app.post("/simulate", tags=["Fiscal"])
async def simulate_tax(simulation: TaxSimulation):
    """
    Simulation fiscale prospective.

    Calcule les taxes et cotisations prévisionnelles selon le régime
    fiscal et l'activité déclarée.
    """
    revenue = simulation.annual_revenue

    # Taux URSSAF Auto-Entrepreneur BNC
    rates = {
        "micro_bnc": {
            "urssaf": 0.214,
            "cfe_estimate": 710,
            "abattement": 0.34,
            "ir_base": 0.66,
        },
        "reel_simplifie": {
            "urssaf": 0.214,
            "cfe_estimate": 710,
            "abattement": 0.0,
            "ir_base": 1.0,
        },
    }

    regime = rates.get(simulation.regime, rates["micro_bnc"])

    urssaf = round(revenue * regime["urssaf"], 2)
    cfe = regime["cfe_estimate"]
    taxable_income = round(revenue * regime["ir_base"], 2)

    # IR simulé (barème 2026 simplifié)
    ir = 0.0
    if taxable_income > 11294:
        ir += min(taxable_income - 11294, 28797 - 11294) * 0.11
    if taxable_income > 28797:
        ir += min(taxable_income - 28797, 82341 - 28797) * 0.30
    if taxable_income > 82341:
        ir += min(taxable_income - 82341, 177106 - 82341) * 0.41
    if taxable_income > 177106:
        ir += (taxable_income - 177106) * 0.45
    ir = round(ir, 2)

    total_taxes = round(urssaf + cfe + ir, 2)
    net_income = round(revenue - total_taxes, 2)
    effective_rate = round(total_taxes / revenue * 100, 2) if revenue > 0 else 0

    return {
        "regime": simulation.regime,
        "activity": simulation.activity,
        "gross_revenue": revenue,
        "breakdown": {
            "urssaf": urssaf,
            "cfe": cfe,
            "impot_revenu": ir,
            "total_taxes": total_taxes,
        },
        "net_income": net_income,
        "effective_tax_rate": effective_rate,
        "provision_monthly": round(total_taxes / 12, 2),
    }


@app.get("/history", tags=["Fiscal"])
async def get_provision_history(limit: int = Query(default=50, ge=1, le=500)):
    """Historique détaillé des provisions fiscales."""
    history = list(app.state.provision_history)[-limit:]
    return {"provisions": history, "total": len(app.state.provision_history)}


@app.get("/report/urssaf", tags=["Fiscal"])
async def get_urssaf_report():
    """
    Rapport URSSAF formaté pour déclaration trimestrielle.

    Calcule le CA par trimestre et les cotisations dues.
    """
    tax_manager: TaxManager = app.state.tax_manager
    escrow = tax_manager.get_escrow_status()

    total_profit = escrow.get("total_escrow", 0) / 0.25 if escrow.get("total_escrow") else 0
    quarterly = round(total_profit / 4, 2)

    return {
        "period": f"T1 {datetime.now().year}",
        "gross_revenue_quarter": quarterly,
        "cotisations_urssaf": round(quarterly * 0.214, 2),
        "total_provisions": escrow.get("total_escrow", 0),
        "transactions_count": escrow.get("transactions_count", 0),
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/alerts", tags=["Compliance"])
async def get_compliance_alerts(limit: int = Query(default=50, ge=1, le=200)):
    """Alertes de conformité (seuils URSSAF, rappels déclaratifs)."""
    # Génération dynamique d'alertes
    alerts = list(app.state.compliance_alerts)[-limit:]

    # Ajouter des alertes contextuelles
    now = datetime.now()
    if now.month in (1, 4, 7, 10) and now.day <= 15:
        alerts.append({
            "severity": "warning",
            "category": "declaration",
            "message": f"📅 Déclaration URSSAF trimestrielle à effectuer avant le 15/{now.month:02d}/{now.year}",
            "timestamp": now.isoformat(),
        })

    return {"alerts": alerts, "total": len(alerts)}
