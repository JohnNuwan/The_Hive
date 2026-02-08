import logging
import asyncio
import json
import os
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from shared.redis_client import init_redis, get_redis_client
from shared.auth_middleware import InternalAuthMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LEDGER_FILE = "ledger.json"

class OperatingExpense(BaseModel):
    description: str
    amount: float
    category: str # infrastructure, api, software, electricity
    timestamp: datetime = datetime.now()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie Accountant"""
    logger.info("💰 Démarrage EVA Accountant (L'Auditeur)...")
    await init_redis()
    
    # Charger les données persistantes
    load_ledger()
    
    # Heartbeat
    asyncio.create_task(hard_heartbeat())
    
    logger.info("✅ EVA Accountant prêt")
    yield
    logger.info("🛑 Arrêt EVA Accountant")

app = FastAPI(title="EVA Accountant", lifespan=lifespan)
app.add_middleware(InternalAuthMiddleware)

# État financier global
financial_state = {
    "gross_profit": 0.0,
    "tax_provision": 0.0,
    "operating_expenses": 0.0,
    "net_roi": 0.0,
    "expenses_detail": []
}

def save_ledger():
    """Sauvegarde l'état financier sur disque"""
    try:
        with open(LEDGER_FILE, "w") as f:
            json.dump(financial_state, f, indent=4, default=str)
        logger.info("💾 Ledger sauvegardé")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde ledger: {e}")

def load_ledger():
    """Charge le ledger depuis le disque"""
    global financial_state
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r") as f:
                data = json.load(f)
                financial_state.update(data)
            logger.info("📂 Ledger chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur chargement ledger: {e}")
    else:
        logger.info("🆕 Aucun ledger trouvé, démarrage à zéro.")

async def hard_heartbeat():
    """Signal de présence pour l'orchestrateur Hive"""
    redis = get_redis_client()
    while True:
        payload = {
            "status": "online", 
            "ts": datetime.now().timestamp(), 
            "expert": "accountant",
            "net_roi": financial_state["net_roi"]
        }
        await redis.cache_set("eva.accountant.status", payload, ttl_seconds=10)
        await asyncio.sleep(2.0)

@app.get("/health")
async def health():
    return {"status": "online", "service": "accountant"}

@app.get("/report")
async def get_report():
    """Bilan financier consolidé"""
    return {
        "summary": {
            "gross": financial_state["gross_profit"],
            "tax": financial_state["tax_provision"],
            "expenses": financial_state["operating_expenses"],
            "net": financial_state["net_roi"]
        },
        "expenses": financial_state["expenses_detail"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/expense")
async def register_expense(expense: OperatingExpense):
    """Enregistre une nouvelle dépense (ex: coût API, électricité)"""
    financial_state["operating_expenses"] += expense.amount
    financial_state["expenses_detail"].append(expense.dict())
    
    # Recalcul ROI
    financial_state["net_roi"] = financial_state["gross_profit"] - financial_state["tax_provision"] - financial_state["operating_expenses"]
    
    save_ledger()
    logger.info(f"💸 Dépense enregistrée : {expense.description} ({expense.amount} €)")
    return {"status": "recorded", "new_net_roi": financial_state["net_roi"]}

@app.post("/sync-ledger")
async def sync_with_compliance(data: dict):
    """Synchronise les données avec le Juriste (Compliance)"""
    financial_state["gross_profit"] = data.get("total_profit", 0.0)
    financial_state["tax_provision"] = data.get("total_tax", 0.0)
    
    # Recalcul ROI
    financial_state["net_roi"] = financial_state["gross_profit"] - financial_state["tax_provision"] - financial_state["operating_expenses"]
    
    save_ledger()
    return {"status": "synchronized", "net_roi": financial_state["net_roi"]}
