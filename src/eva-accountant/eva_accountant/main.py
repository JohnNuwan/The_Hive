"""
EVA Accountant — L'Auditeur Financier de THE HIVE.

Ce module gère la comptabilité opérationnelle de l'entité :
- Suivi du ROI net (profits bruts - taxes - dépenses).
- Enregistrement des dépenses d'exploitation (API, infra, électricité).
- Synchronisation avec le Keeper (Compliance) pour les taxes.
- Réception des PnL du Banker avec contrôle de drawdown.
- Projections financières et prévisions.
- Export des données (CSV/JSON).
- Dashboard financier consolidé.

Architecture :
    - Passif : écoute les mises à jour PnL du Banker.
    - Persistance sur disque (accountant_ledger.json).
    - Heartbeat vers le Core avec ROI net.
"""

import asyncio
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from shared.redis_client import init_redis, get_redis_client
from shared.auth_middleware import InternalAuthMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════


class OperatingExpense(BaseModel):
    """Représente une dépense d'exploitation."""
    description: str
    amount: float
    category: str = Field(description="Catégorie: infrastructure, api, software, electricity, other")
    timestamp: datetime = Field(default_factory=datetime.now)


class PnLReport(BaseModel):
    """Bilan quotidien ou PnL envoyé par le Banker."""
    timestamp: datetime = Field(default_factory=datetime.now)
    symbol: str = "GLOBAL"
    profit_loss: float
    balance: float
    equity: float


class ProjectionRequest(BaseModel):
    """Requête de projection financière."""
    months: int = Field(default=3, ge=1, le=24, description="Nombre de mois à projeter")
    growth_rate: float = Field(default=0.0, ge=-1.0, le=10.0, description="Taux de croissance mensuel estimé")
    include_expenses: bool = True


class CurrencyConversion(BaseModel):
    """Conversion de devises simplifiée."""
    amount: float = Field(..., gt=0)
    from_currency: str = Field(default="EUR")
    to_currency: str = Field(default="USD")


# ═══════════════════════════════════════════════════════════════════════════════
# VARIABLES D'ÉTAT
# ═══════════════════════════════════════════════════════════════════════════════

LEDGER_FILE = "accountant_ledger.json"
DAILY_DRAWDOWN_LIMIT = 0.04  # 4% max drawdown / jour

ledger = {
    "gross_profit": 0.0,
    "total_taxes": 0.0,
    "total_expenses": 0.0,
    "net_roi": 0.0,
    "expenses": [],
    "pnl_history": [],
    "daily_start_balance": None,
    "currency": "EUR",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTANCE
# ═══════════════════════════════════════════════════════════════════════════════


def save_ledger():
    """Sauvegarde l'état financier sur disque."""
    try:
        with open(LEDGER_FILE, "w") as f:
            json.dump(ledger, f, indent=4, default=str)
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde ledger: {e}")


def load_ledger():
    """Charge le ledger depuis le disque."""
    global ledger
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r") as f:
                loaded = json.load(f)
                ledger.update(loaded)
                logger.info("📂 Ledger Accountant chargé")
        except Exception as e:
            logger.error(f"❌ Erreur chargement ledger: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("📊 Démarrage EVA Accountant (L'Auditeur)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    load_ledger()

    app.state.monthly_summaries: deque[dict[str, Any]] = deque(maxlen=24)
    app.state.daily_snapshots: deque[dict[str, Any]] = deque(maxlen=365)

    asyncio.create_task(hard_heartbeat())
    logger.info("✅ EVA Accountant en veille comptable")
    yield
    save_ledger()
    logger.info("🛑 Arrêt EVA Accountant")


async def hard_heartbeat():
    """Signal de présence avec ROI net courant."""
    try:
        redis = get_redis_client()
    except Exception:
        redis = None
    while True:
        try:
            if redis:
                payload = {
                    "status": "online",
                    "ts": datetime.now().timestamp(),
                    "expert": "accountant",
                    "net_roi": ledger["net_roi"],
                }
                await redis.cache_set("eva.accountant.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="EVA Accountant API",
    description="L'Auditeur Financier - THE HIVE",
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
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", tags=["Système"])
async def health():
    return {
        "status": "online",
        "service": "accountant",
        "net_roi": ledger["net_roi"],
    }


@app.get("/report", tags=["Bilan"])
async def get_report():
    """Bilan financier consolidé."""
    return {
        "gross_profit": ledger["gross_profit"],
        "total_taxes": ledger["total_taxes"],
        "total_expenses": ledger["total_expenses"],
        "net_roi": ledger["net_roi"],
        "expense_count": len(ledger["expenses"]),
        "pnl_entries": len(ledger["pnl_history"]),
        "currency": ledger["currency"],
        "generated_at": datetime.now().isoformat(),
    }


@app.post("/expense", tags=["Dépenses"])
async def register_expense(expense: OperatingExpense):
    """Enregistre une dépense d'exploitation et recalcule le ROI net."""
    entry = expense.model_dump(mode="json")
    ledger["expenses"].append(entry)
    ledger["total_expenses"] += expense.amount
    ledger["net_roi"] = ledger["gross_profit"] - ledger["total_taxes"] - ledger["total_expenses"]
    save_ledger()
    logger.info(f"💸 Dépense: {expense.description} — {expense.amount}€ ({expense.category})")
    return {"status": "registered", "net_roi": ledger["net_roi"]}


@app.get("/expenses", tags=["Dépenses"])
async def list_expenses(category: str = Query(default=""), limit: int = Query(default=50, ge=1, le=500)):
    """Liste les dépenses d'exploitation, filtrable par catégorie."""
    expenses = ledger["expenses"]
    if category:
        expenses = [e for e in expenses if e.get("category") == category]
    return {"expenses": expenses[-limit:], "total": len(expenses)}


@app.post("/pnl", tags=["PnL"])
async def receive_pnl(report: PnLReport):
    """
    Reçoit les mises à jour de PnL du Banker.
    Vérifie le Max Drawdown Journalier (4%).
    """
    entry = report.model_dump(mode="json")
    ledger["pnl_history"].append(entry)

    # Mise à jour du profit brut
    if report.profit_loss > 0:
        ledger["gross_profit"] += report.profit_loss

    # Check drawdown journalier
    alert = None
    if ledger["daily_start_balance"] is None:
        ledger["daily_start_balance"] = report.balance
    else:
        start = ledger["daily_start_balance"]
        if start > 0:
            drawdown = (start - report.equity) / start
            if drawdown >= DAILY_DRAWDOWN_LIMIT:
                alert = {
                    "type": "DRAWDOWN_LIMIT",
                    "drawdown_percent": round(drawdown * 100, 2),
                    "limit_percent": DAILY_DRAWDOWN_LIMIT * 100,
                    "message": f"🚨 DRAWDOWN {drawdown*100:.1f}% ≥ {DAILY_DRAWDOWN_LIMIT*100}% — KILL SWITCH recommandé",
                }
                logger.warning(alert["message"])

    ledger["net_roi"] = ledger["gross_profit"] - ledger["total_taxes"] - ledger["total_expenses"]
    save_ledger()

    response = {"status": "received", "net_roi": ledger["net_roi"]}
    if alert:
        response["alert"] = alert
    return response


@app.post("/sync", tags=["Bilan"])
async def sync_with_compliance(data: dict):
    """
    Synchronise les données avec le Keeper (Compliance).

    Met à jour les profits bruts et provisions fiscales.
    """
    if "total_profit" in data:
        ledger["gross_profit"] = data["total_profit"]
    if "total_tax" in data:
        ledger["total_taxes"] = data["total_tax"]
    ledger["net_roi"] = ledger["gross_profit"] - ledger["total_taxes"] - ledger["total_expenses"]
    save_ledger()
    return {"status": "synced", "net_roi": ledger["net_roi"]}


# ─── PROJECTIONS ──────────────────────────────────────────────────────────────


@app.post("/projections", tags=["Projections"])
async def get_projections(request: ProjectionRequest):
    """
    Projections financières sur N mois.

    Basées sur le ROI actuel, le taux de croissance estimé,
    et les dépenses récurrentes.
    """
    current_roi = ledger["net_roi"]
    monthly_expenses = ledger["total_expenses"] / max(1, len(set(
        e.get("timestamp", "")[:7] for e in ledger["expenses"]
    ))) if ledger["expenses"] else 0

    projections = []
    cumulated = current_roi
    for month in range(1, request.months + 1):
        growth = cumulated * request.growth_rate if request.growth_rate else 0
        expenses = monthly_expenses if request.include_expenses else 0
        cumulated += growth - expenses
        projections.append({
            "month": month,
            "projected_roi": round(cumulated, 2),
            "growth": round(growth, 2),
            "expenses": round(expenses, 2),
        })

    return {
        "current_roi": current_roi,
        "growth_rate": request.growth_rate,
        "monthly_expenses": round(monthly_expenses, 2),
        "projections": projections,
    }


# ─── EXPORT ───────────────────────────────────────────────────────────────────


@app.get("/export", tags=["Export"])
async def export_data(format: str = Query(default="json", description="Format: json, csv")):
    """Exporte les données comptables en JSON ou CSV."""
    if format == "csv":
        lines = ["date,type,amount,category,description"]
        for e in ledger["expenses"]:
            lines.append(f"{e.get('timestamp','')},expense,{e.get('amount',0)},{e.get('category','')},{e.get('description','')}")
        for p in ledger["pnl_history"]:
            lines.append(f"{p.get('timestamp','')},pnl,{p.get('profit_loss',0)},{p.get('symbol','')},PnL Update")
        csv_content = "\n".join(lines)
        return JSONResponse(
            content={"format": "csv", "data": csv_content, "rows": len(lines) - 1},
            headers={"Content-Type": "application/json"},
        )

    return {
        "format": "json",
        "data": {
            "summary": {
                "gross_profit": ledger["gross_profit"],
                "total_taxes": ledger["total_taxes"],
                "total_expenses": ledger["total_expenses"],
                "net_roi": ledger["net_roi"],
            },
            "expenses": ledger["expenses"],
            "pnl_history": ledger["pnl_history"],
        },
        "exported_at": datetime.now().isoformat(),
    }


# ─── DASHBOARD ────────────────────────────────────────────────────────────────


@app.get("/dashboard", tags=["Dashboard"])
async def get_dashboard():
    """Tableau de bord financier consolidé."""
    # Répartition des dépenses par catégorie
    expense_by_category: dict[str, float] = {}
    for e in ledger["expenses"]:
        cat = e.get("category", "other")
        expense_by_category[cat] = expense_by_category.get(cat, 0) + e.get("amount", 0)

    # PnL tendance (dernières entrées)
    recent_pnl = ledger["pnl_history"][-10:]

    return {
        "summary": {
            "gross_profit": ledger["gross_profit"],
            "total_taxes": ledger["total_taxes"],
            "total_expenses": ledger["total_expenses"],
            "net_roi": ledger["net_roi"],
            "currency": ledger["currency"],
        },
        "expenses_by_category": expense_by_category,
        "recent_pnl": recent_pnl,
        "expense_count": len(ledger["expenses"]),
        "pnl_count": len(ledger["pnl_history"]),
    }


# ─── DEVISE ───────────────────────────────────────────────────────────────────


@app.post("/currency/convert", tags=["Devises"])
async def convert_currency(conversion: CurrencyConversion):
    """
    Conversion de devises simplifiée (taux fixes).

    En production, utiliser un service comme ExchangeRate-API.
    """
    rates = {
        ("EUR", "USD"): 1.09,
        ("USD", "EUR"): 0.917,
        ("EUR", "GBP"): 0.855,
        ("GBP", "EUR"): 1.17,
        ("EUR", "CHF"): 0.96,
        ("CHF", "EUR"): 1.04,
    }

    key = (conversion.from_currency.upper(), conversion.to_currency.upper())
    rate = rates.get(key)

    if rate is None:
        return {"status": "error", "message": f"Paire {key} non supportée. Paires: {list(rates.keys())}"}

    converted = round(conversion.amount * rate, 2)
    return {
        "original": conversion.amount,
        "from": conversion.from_currency,
        "to": conversion.to_currency,
        "rate": rate,
        "converted": converted,
        "mode": "FIXED_RATE",
    }
