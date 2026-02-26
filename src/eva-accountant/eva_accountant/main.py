"""
EVA Accountant — L'Auditeur Financier de THE HIVE.

Ce module gère la comptabilité opérationnelle de l'entité :
- Suivi du ROI net (profits bruts - taxes - dépenses).
- Enregistrement des dépenses d'exploitation (API, infra, électricité).
- Synchronisation avec le Keeper (Compliance) pour les données fiscales.
- Persistance sur disque via un fichier ledger JSON.

Architecture :
    - Passif : agrège les données financières et expose des rapports.
    - Se synchronise avec Compliance pour les provisions fiscales.
    - Heartbeat vers le Core pour la découverte des agents.
"""

import logging
import asyncio
import json
import os
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared.redis_client import init_redis, get_redis_client
from shared.auth_middleware import InternalAuthMiddleware
from shared import BaseHealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin du fichier de persistance des données financières
LEDGER_FILE = "ledger.json"


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════


class OperatingExpense(BaseModel):
    """
    Représente une dépense d'exploitation.

    Attributes:
        description: Description humaine de la dépense.
        amount: Montant en euros.
        category: Catégorie (infrastructure, api, software, electricity).
        timestamp: Date/heure de l'enregistrement.
    """
    description: str
    amount: float
    category: str  # infrastructure, api, software, electricity
    timestamp: datetime = Field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application Accountant.

    Charge le ledger depuis le disque, initialise Redis et
    démarre le heartbeat de présence.

    Args:
        app (FastAPI): L'instance de l'application en cours.

    Yields:
        None: Rend la main une fois l'initialisation terminée.
    """
    logger.info("💰 Démarrage EVA Accountant (L'Auditeur)...")

    # Redis — tolérant aux pannes au démarrage
    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # Charger les données persistantes
    load_ledger()

    # Heartbeat
    asyncio.create_task(hard_heartbeat())

    logger.info("✅ EVA Accountant prêt")
    yield

    # Sauvegarder à l'arrêt
    save_ledger()
    logger.info("🛑 Arrêt EVA Accountant")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA Accountant API",
    description="L'Auditeur Financier - THE HIVE",
    version="0.1.0",
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
# ÉTAT FINANCIER & PERSISTANCE
# ═══════════════════════════════════════════════════════════════════════════════


# État financier global en mémoire
financial_state: dict = {
    "gross_profit": 0.0,
    "tax_provision": 0.0,
    "operating_expenses": 0.0,
    "net_roi": 0.0,
    "expenses_detail": [],
}


def save_ledger():
    """Sauvegarde l'état financier sur disque (fichier JSON)."""
    try:
        with open(LEDGER_FILE, "w", encoding="utf-8") as f:
            json.dump(financial_state, f, indent=4, default=str)
        logger.info("💾 Ledger sauvegardé")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde ledger: {e}")


def load_ledger():
    """
    Charge le ledger depuis le disque.

    Si le fichier n'existe pas, démarre avec un état vierge.
    """
    global financial_state
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                financial_state.update(data)
            logger.info("📂 Ledger chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur chargement ledger: {e}")
    else:
        logger.info("🆕 Aucun ledger trouvé, démarrage à zéro.")


# ═══════════════════════════════════════════════════════════════════════════════
# TÂCHES DE FOND
# ═══════════════════════════════════════════════════════════════════════════════


async def hard_heartbeat():
    """
    Signal de présence pour l'Orchestrateur Core.

    Inclut le ROI net courant dans le payload pour le monitoring.
    """
    redis = get_redis_client()
    while True:
        try:
            payload = {
                "status": "online",
                "ts": datetime.now().timestamp(),
                "expert": "accountant",
                "net_roi": financial_state["net_roi"],
            }
            await redis.cache_set("eva.accountant.status", payload, ttl_seconds=10)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", response_model=BaseHealthResponse, tags=["Système"])
async def health():
    """Vérifie la santé du module Accountant."""
    return BaseHealthResponse(status="online")


@app.get("/report", tags=["Comptabilité"])
async def get_report():
    """
    Bilan financier consolidé.

    Returns:
        dict: Résumé (brut, taxes, dépenses, net) + détail des dépenses.
    """
    return {
        "summary": {
            "gross": financial_state["gross_profit"],
            "tax": financial_state["tax_provision"],
            "expenses": financial_state["operating_expenses"],
            "net": financial_state["net_roi"],
        },
        "expenses": financial_state["expenses_detail"],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/expense", tags=["Comptabilité"])
async def register_expense(expense: OperatingExpense):
    """
    Enregistre une nouvelle dépense d'exploitation.

    Recalcule automatiquement le ROI net après enregistrement.

    Args:
        expense (OperatingExpense): Détails de la dépense.

    Returns:
        dict: Statut et nouveau ROI net.
    """
    financial_state["operating_expenses"] += expense.amount
    financial_state["expenses_detail"].append(expense.model_dump())

    # Recalcul ROI
    financial_state["net_roi"] = (
        financial_state["gross_profit"]
        - financial_state["tax_provision"]
        - financial_state["operating_expenses"]
    )

    save_ledger()
    logger.info(f"💸 Dépense enregistrée : {expense.description} ({expense.amount} €)")
    return {"status": "recorded", "new_net_roi": financial_state["net_roi"]}


class PnLReport(BaseModel):
    """Bilan quotidien ou PnL envoyé par le Banker"""
    timestamp: datetime = Field(default_factory=datetime.now)
    symbol: str = "GLOBAL"
    profit_loss: float
    balance: float
    equity: float

@app.post("/pnl", tags=["Trading"])
async def receive_pnl(report: PnLReport):
    """
    Reçoit les mises à jour de PnL du Banker (E.V.A).
    Vérifie le Max Drawdown Journalier (ex: 4%).
    """
    financial_state["gross_profit"] += report.profit_loss
    
    # 1. Update High Water Mark & Current Drawdown
    if report.equity > financial_state.get("high_water_mark", 0):
        financial_state["high_water_mark"] = report.equity
        financial_state["current_drawdown_pct"] = 0.0
    else:
        if financial_state.get("high_water_mark", 0) > 0:
            dd_amount = financial_state["high_water_mark"] - report.equity
            financial_state["current_drawdown_pct"] = (dd_amount / financial_state["high_water_mark"]) * 100
            
    # Max Drawdown record
    if financial_state.get("current_drawdown_pct", 0) > financial_state.get("max_drawdown_pct", 0):
        financial_state["max_drawdown_pct"] = financial_state["current_drawdown_pct"]

    # 2. Daily Drawdown Limit Check (Default 4.0%)
    MAX_DAILY_DD = float(os.getenv("MAX_DD_DAILY", "4.0"))
    
    if financial_state.get("current_drawdown_pct", 0) >= MAX_DAILY_DD:
        logger.error(f"🚨 HARD LIMIT BREACH! Drawdown: {financial_state['current_drawdown_pct']:.2f}% (Limit: {MAX_DAILY_DD}%)")
        redis = get_redis_client()
        if redis:
            payload = {
                "command": "GLOBAL_STOP",
                "issuer": "accountant",
                "reason": f"Max Daily Drawdown Reached: {financial_state['current_drawdown_pct']:.2f}%",
                "timestamp": datetime.now().isoformat()
            }
            # Broadcast on the swarm command channel
            await redis.publish("eva.all.swarm_command", payload)
            logger.critical("🛑 SENT GLOBAL_STOP TO ALL EXPERTS")

    # Recalcul ROI Net
    financial_state["net_roi"] = (
        financial_state["gross_profit"]
        - financial_state["tax_provision"]
        - financial_state["operating_expenses"]
    )

    save_ledger()
    return {"status": "pnl_recorded", "drawdown_pct": financial_state.get("current_drawdown_pct", 0)}

@app.post("/sync-ledger", tags=["Comptabilité"])
async def sync_with_compliance(data: dict):
    """
    Synchronise les données avec le Keeper (Compliance).

    Met à jour les profits bruts et les provisions fiscales, puis
    recalcule le ROI net.

    Args:
        data (dict): Données du Compliance (total_profit, total_tax).

    Returns:
        dict: Statut de synchronisation et ROI net actualisé.
    """
    # Si le profit vient du compliance, on ne l'écrase pas violemment, on synchronise
    # Mais par sécurité pour ce POC, on garde gross_profit localement maître s'il est plus grand
    compliance_profit = data.get("total_profit", 0.0)
    if compliance_profit > financial_state["gross_profit"]:
        financial_state["gross_profit"] = compliance_profit
        
    financial_state["tax_provision"] = data.get("total_tax", 0.0)

    # Recalcul ROI
    financial_state["net_roi"] = (
        financial_state["gross_profit"]
        - financial_state["tax_provision"]
        - financial_state["operating_expenses"]
    )

    save_ledger()
    return {"status": "synchronized", "net_roi": financial_state["net_roi"]}
