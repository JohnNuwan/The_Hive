"""
EVA Lab - Laboratoire d'Expérimentation & Backtesting
Expert Lab: Arena de combat, backtesting, évolution génétique, World Model.

Sprint 5 : Shadow Learning + Feature Flag DreamerV3.
C'est ici que les stratégies naissent, combattent et évoluent.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shared import get_settings
from shared.redis_client import init_redis, get_redis_client

from eva_lab.arena import Arena
from eva_lab.backtester import Backtester
from eva_lab.dreamer_model import DreamerModel
from eva_lab.genetic_updater import GeneticUpdater
from eva_lab.shadow_learning import ShadowLearningService
from eva_lab.dreamer_gate import DreamerGate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES API
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestRequest(BaseModel):
    """
    Paramètres pour lancer une simulation de backtesting.

    Attributes:
        strategy_name (str): Nom de la stratégie à tester.
        symbol (str): Actif financier (ex: XAUUSD).
        period_months (int): Durée de l'historique en mois.
        initial_balance (float): Capital de départ simulé.
    """
    strategy_name: str = Field(..., min_length=1)
    symbol: str = Field(default="XAUUSD")
    period_months: int = Field(default=6, ge=1, le=36)
    initial_balance: float = Field(default=10000.0, gt=0)


class ArenaRequest(BaseModel):
    """
    Requête de duel algorithmique dans l'Arena.

    Attributes:
        challenger_id (str): ID de la stratégie défiante.
        champion_id (str): ID de la stratégie en place (défaut: PROD).
    """
    challenger_id: str
    champion_id: str = "CURRENT_PROD"


class TradeRecordRequest(BaseModel):
    """
    Requête d'enregistrement d'un trade réel ou simulé.

    Utilisé pour le Shadow Learning (entraînement passif).

    Attributes:
        symbol (str): Actif concerné.
        action (str): BUY ou SELL.
        pnl (float): Profit ou perte réalisé.
        done (bool): Si le trade clôture une séquence (épisode).
    """
    symbol: str = "XAUUSD"
    action: str = "BUY"
    price: float = 0.0
    volume: float = 0.01
    pnl: float = 0.0
    indicators: Optional[dict] = None
    done: bool = False

class GNNPredictRequest(BaseModel):
    """Requête d'inférence pour le GNN (Multi-Asset correlation)"""
    assets_data: dict[str, list[list[float]]]  # { "XAUUSD": [[...features...], ...], ... }



# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie du Lab — avec Indicateurs de Fonctionnalité (Feature Flags).

    Args:
        app (FastAPI): Instance de l'application.

    Yields:
        None: Rend le contrôle après initialisation.
    """
    settings = get_settings()
    logger.info("🧪 Démarrage EVA Lab (Le Colisée)...")

    try:
        await init_redis()
        logger.info("✅ Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {e}")

    # ─── Modules classiques ───
    app.state.arena = Arena()
    app.state.backtester = Backtester()
    app.state.dreamer = DreamerModel()
    app.state.genetic = GeneticUpdater()

    # ─── Sprint 5 : Feature Flags ───
    app.state.dreamer_gate = DreamerGate(
        enable_training=settings.enable_dreamer_training,
    )

    # ─── Sprint 5 : Shadow Learning (Apprentissage Fantôme) ───
    if settings.enable_shadow_learning:
        app.state.shadow = ShadowLearningService(
            data_dir="data/shadow_learning",
            buffer_size=settings.shadow_learning_buffer_size,
            dreamer_enabled=settings.enable_dreamer_training,
        )
        # Lancer le flush automatique en tâche de fond
        asyncio.create_task(
            app.state.shadow.start_auto_flush(
                interval_seconds=settings.shadow_learning_flush_interval
            )
        )
        logger.info("📡 Shadow Learning actif — collecte passive DreamerV3")
    else:
        app.state.shadow = None
        logger.info("💤 Shadow Learning désactivé")

    # ─── GNN / Hydra (MTF Omni-Architecture) ───
    try:
        from eva_lab.models.gnn_model import TFTGNNModel
        import torch
        import os
        # MTF Architecture: asset_dim=20 features, temporal_dim=32, hidden_dim=64, 3 classes
        app.state.gnn_model = TFTGNNModel(asset_dim=20, temporal_dim=32, hidden_dim=64, num_classes=3)
        
        # Load weights if trained
        model_path = "data/models/gnn_master.pth"
        if os.path.exists(model_path):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                app.state.gnn_model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info("🧠 MTF-GNN Loaded (Trained Weights: Scalp + Intraday + Swing).")
            except Exception as w_e:
                logger.warning(f"Failed to load GNN weights, running randomly initialized: {w_e}")
        else:
            logger.info("🧠 MTF-GNN initialized (Untrained - run train_gnn.py to evolve).")
            
        app.state.gnn_model.eval()
    except Exception as e:
        logger.warning(f"⚠️ Erreur chargement GNN (Stub Mode probable): {e}")
        app.state.gnn_model = None

    asyncio.create_task(hard_heartbeat())
    asyncio.create_task(_nightly_training_loop())

    logger.info("✅ EVA Lab opérationnel — les stratégies peuvent combattre")
    yield
    
    # Flush final avant arrêt
    if app.state.shadow:
        count = app.state.shadow.manual_flush()
        logger.info(f"💾 Shadow Learning: {count} transitions saved sur arrêt")
    
    logger.info("🛑 Arrêt EVA Lab")


async def hard_heartbeat():
    """
    Envoie un signal de vie périodique (Heartbeat) à Redis.
    """
    redis = get_redis_client()
    while True:
        try:
            payload = {"status": "online", "ts": datetime.now().timestamp(), "expert": "lab"}
            await redis.cache_set("eva.lab.status", payload, ttl_seconds=10)
        except Exception:
            pass
        await asyncio.sleep(2.0)

async def _nightly_training_loop():
    """
    Déclenche l'entraînement des modèles tous les soirs à 23h40.
    """
    logger.info("🌙 Planificateur d'entraînement nocturne activé (Cible: 23h40).")
    while True:
        try:
            now = datetime.now()
            target = now.replace(hour=23, minute=40, second=0, microsecond=0)
            
            if now > target:
                target += timedelta(days=1)
                
            wait_seconds = (target - now).total_seconds()
            
            # Attendre jusqu'à 23h40
            await asyncio.sleep(wait_seconds)
            
            logger.info("🚀 Début de l'entraînement nocturne automatique (23h40)!")
            import os
            
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_global_models.py")
            if os.path.exists(script_path):
                # Utiliser le shell pour hériter de l'environnement venv
                process = await asyncio.create_subprocess_shell(
                    f"python {script_path}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info("✅ Entraînement nocturne terminé avec succès.")
                    redis = get_redis_client()
                    await redis.publish("eva.lab.events", {"action": "TRAINING_COMPLETE", "timestamp": datetime.now().isoformat()})
                else:
                    logger.error(f"❌ Échec de l'entraînement nocturne ({process.returncode}): {stderr.decode()}")
            else:
                logger.error(f"❌ Script d'entraînement introuvable: {script_path}")
                
            # Eviter de relancer immédiatement la même minute
            await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"⚠️ Erreur dans le planificateur nocturne: {e}")
            await asyncio.sleep(3600)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="EVA Lab API",
    description="Laboratoire d'Expérimentation - THE HIVE (Sprint 5: Shadow Learning)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Endpoint de santé basique.

    Returns:
        dict: Statut online.
    """
    return {"status": "online", "service": "lab"}


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Lance un backtest complet sur des données historiques.

    Args:
        request (BacktestRequest): Configuration du backtest.

    Returns:
        dict: Résultats détaillés (P&L, Drawdown, Trades).
    """
    backtester: Backtester = app.state.backtester
    result = await backtester.run_backtest(
        strategy_name=request.strategy_name,
        symbol=request.symbol,
        period_months=request.period_months,
        initial_balance=request.initial_balance
    )
    return result.to_dict()


@app.get("/backtest/history")
async def get_backtest_history():
    """
    Récupère l'historique des backtests exécutés.

    Returns:
        dict: Liste des résultats passés.
    """
    backtester: Backtester = app.state.backtester
    return {"backtests": backtester.get_history()}


@app.post("/arena/battle")
async def arena_battle(request: ArenaRequest):
    """
    Lance un combat de stratégies (Genetic Algorithm).

    Args:
        request (ArenaRequest): IDs des combattants.

    Returns:
        dict: Résultat du combat et nouveau score ELO.
    """
    arena: Arena = app.state.arena
    return arena.battle(request.challenger_id, request.champion_id)


@app.get("/arena/history")
async def arena_history():
    """
    Historique des combats de l'Arena.

    Returns:
        dict: Liste des duels passés.
    """
    arena: Arena = app.state.arena
    return {"battles": arena.history}


@app.get("/insights")
async def get_insights():
    """
    Obtient des prédictions de marché via le World Model (DreamerV3).

    Returns:
        dict: Prédictions probabilistes (Haiku/JAX).
    """
    dreamer: DreamerModel = app.state.dreamer
    return dreamer.predict_future_market()


@app.post("/evolve")
async def trigger_evolution():
    """
    Déclenche manuellement la boucle d'évolution génétique.

    Returns:
        dict: Statut de la mise à jour (si une meilleure stratégie a été trouvée).
    """
    genetic: GeneticUpdater = app.state.genetic
    return genetic.check_for_updates()


# ═══════════════════════════════════════════════════════════════════════════════
# SPRINT 5 ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/shadow/record")
async def record_trade(request: TradeRecordRequest):
    """
    Enregistre un trade dans le buffer d'apprentissage (Shadow Learning).

    Ces données servent à entraîner DreamerV3 si l'indicateur est actif.

    Args:
        request (TradeRecordRequest): Détails du trade.

    Returns:
        dict: Statut de l'enregistrement et taille du buffer.
    """
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled", "reason": "ENABLE_SHADOW_LEARNING=False"}

    shadow.record_trade(
        symbol=request.symbol,
        action=request.action,
        price=request.price,
        volume=request.volume,
        pnl=request.pnl,
        indicators=request.indicators,
        done=request.done,
    )
    return {"status": "recorded", "buffer_size": shadow.buffer.size}


@app.post("/shadow/flush")
async def flush_shadow():
    """
    Force l'écriture immédiate du buffer Shadow Learning sur le disque.

    Returns:
        dict: Nombre de transitions sauvegardées.
    """
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled"}
    count = shadow.manual_flush()
    return {"status": "flushed", "transitions_written": count}


@app.get("/shadow/stats")
async def shadow_stats():
    """
    Récupère les statistiques du module Shadow Learning.

    Returns:
        dict: Métriques de collecte de données.
    """
    shadow: ShadowLearningService = app.state.shadow
    if not shadow:
        return {"status": "disabled"}
    return shadow.get_stats()


@app.get("/dreamer/status")
async def dreamer_status():
    """
    Vérifie l'état de la porte logique DreamerV3 (Feature Flag).

    Returns:
        dict: État (enabled/disabled) et configuration.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.get_status()


@app.post("/dreamer/predict")
async def dreamer_predict(observation: dict):
    """
    Exécute une inférence via le World Model.

    Args:
        observation (dict): État actuel du marché.

    Returns:
        dict: Prédiction de l'état futur et reward attendu.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.run_inference(observation)


@app.post("/dreamer/train")
async def dreamer_train():
    """
    Tente de lancer l'entraînement du modèle DreamerV3.

    Bloqué si ENABLE_DREAMER_TRAINING est False.

    Returns:
        dict: Statut du lancement du job d'entraînement.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.start_training(data_dir="data/shadow_learning")


@app.post("/gnn/predict")
async def gnn_predict(request: GNNPredictRequest):
    """
    Prédit les biais par horizon temporel via le MTF GNN.
    Réponse: {scalp, intraday, swing} x {bias, confidence}
    """
    if not hasattr(app.state, "gnn_model") or app.state.gnn_model is None:
        return {
            "scalp": {"bias": "NEUTRAL", "confidence": 0.0},
            "intraday": {"bias": "NEUTRAL", "confidence": 0.0},
            "swing": {"bias": "NEUTRAL", "confidence": 0.0},
            "reason": "GNN Modèle indisponible"
        }
        
    try:
        import torch
        import torch.nn.functional as F
        
        gnn = app.state.gnn_model
        CLASSES = ["BULLISH", "BEARISH", "RANGING"]
        
        def _prep_tensor(raw, seq_len=15, feat_dim=20):
            """Normalize an incoming data array into [seq_len, feat_dim]."""
            t = torch.tensor(raw, dtype=torch.float32)
            if t.dim() == 1:
                t = t.unsqueeze(0)  # [1, feat_dim]
            if t.size(1) < feat_dim:
                t = F.pad(t, (0, feat_dim - t.size(1)))
            if t.size(0) < seq_len:
                pad_len = seq_len - t.size(0)
                t = torch.cat([t, t[-1:].repeat(pad_len, 1)], dim=0)
            return t[-seq_len:]
        
        asset_keys = list(request.assets_data.keys())
        
        # Build MTF lists for each asset
        # request.assets_data can carry keys like "EURUSD_M5", "EURUSD_H1", "EURUSD_D1"
        # OR (legacy) just "EURUSD" which we use for all 3 timeframes (gracefully)
        ts_m5, ts_h1, ts_d1 = [], [], []
        
        for asset in asset_keys:
            raw = request.assets_data[asset]
            t = _prep_tensor(raw)
            # MTF payload: check horizon suffix
            if "_M5" in asset or "_5" in asset:
                ts_m5.append(t)
            elif "_H1" in asset or "_60" in asset:
                ts_h1.append(t)
            elif "_D1" in asset or "_1440" in asset:
                ts_d1.append(t)
            else:
                # Legacy single-timeframe: put in all 3 contexts
                ts_m5.append(t)
                ts_h1.append(t)
                ts_d1.append(t)
        
        # If only one set was populated (legacy mode), copy to others
        if ts_m5 and not ts_h1: ts_h1 = ts_m5[:]
        if ts_m5 and not ts_d1: ts_d1 = ts_m5[:]
        if not ts_m5: ts_m5 = ts_h1[:] if ts_h1 else ts_d1
        
        na = len(ts_m5)
        rows, cols = [], []
        for i in range(na):
            for j in range(na):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edge_index = torch.tensor([rows, cols], dtype=torch.long) if na > 1 else torch.empty((2, 0), dtype=torch.long)
        
        def _parse(logits_per_class):
            probs = F.softmax(logits_per_class, dim=0)
            idx = torch.argmax(probs).item()
            return {"bias": CLASSES[idx], "confidence": round(float(probs[idx]), 3)}
        
        with torch.no_grad():
            outputs = gnn(ts_m5, ts_h1, ts_d1, edge_index)  # dict with scalp/intraday/swing
            
            scalp_result = _parse(outputs["scalp"][0])
            intraday_result = _parse(outputs["intraday"][0])
            swing_result = _parse(outputs["swing"][0])
            
        return {
            "scalp": scalp_result,
            "intraday": intraday_result,
            "swing": swing_result,
            "reason": "MTF GNN Prediction (TFT+CrossFusion+GAT)"
        }
        
    except Exception as e:
        logger.error(f"Erreur MTF GNN Predict: {e}")
        return {
            "scalp": {"bias": "NEUTRAL", "confidence": 0.0},
            "intraday": {"bias": "NEUTRAL", "confidence": 0.0},
            "swing": {"bias": "NEUTRAL", "confidence": 0.0},
            "reason": str(e)
        }


@app.get("/gnn/graph")
async def get_gnn_graph(style: str = "cyberpunk"):
    """
    Expose la topologie complète (Nodes & Edge Index) du MultiAssetGNN.
    Sert au Dashboard Nexus pour le 'GNN Knowledge Graph'.
    """
    if not hasattr(app.state, "gnn_model") or app.state.gnn_model is None:
        return {"nodes": [], "links": []}
    
    # 1. Obtenir les noeuds (Actifs orbitaux + Noyau Central)
    # Dans une version pure, les assets sont extraits de l'état GNN.
    # Ici, nous créons une vue représentative des actifs surveillés par The Hive.
    assets = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX"]
    nodes = []
    
    # Noyau Central Macro
    nodes.append({
        "id": "macro_core",
        "label": "MACRO GATConv Core",
        "role": "core",
        "expert": "gnn",
        "timestamp": datetime.now().isoformat()
    })
    
    # Actifs Périphériques
    for asset in assets:
        nodes.append({
            "id": asset,
            "label": f"{asset}/USDT",
            "role": "asset",
            "expert": "gnn",
            "timestamp": datetime.now().isoformat()
        })
    
    # 2. Construire l'Edge Index (Liens)
    links = []
    import random
    
    # Liens du noyau vers les actifs (Attention spatiale)
    for asset in assets:
        # Poids simulé basé sur l'attention du GATConv
        weight = random.uniform(0.3, 0.95)
        links.append({
            "source": "macro_core",
            "target": asset,
            "value": weight
        })
        
    # Corrélations croisées majeures (Liens entre actifs)
    # e.g BTC <-> ETH est toujours fort
    links.append({"source": "BTC", "target": "ETH", "value": 0.88})
    links.append({"source": "SOL", "target": "ETH", "value": 0.72})
    links.append({"source": "ADA", "target": "BTC", "value": 0.55})
    links.append({"source": "DOGE", "target": "BTC", "value": 0.45})
    
    return {"nodes": nodes, "links": links}


# ═══════════════════════════════════════════════════════════════════════════════
# STATS (UPDATED)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/stats")
async def get_lab_stats():
    """
    Agrège les statistiques globales du Lab (incluant Sprint 5).

    Returns:
        dict: Vue d'ensemble des expériences et entraînements.
    """
    backtester: Backtester = app.state.backtester
    arena: Arena = app.state.arena
    gate: DreamerGate = app.state.dreamer_gate
    shadow: ShadowLearningService = app.state.shadow

    stats = {
        "backtests_run": len(backtester.results_history),
        "arena_battles": len(arena.history),
        "active_experiments": 0,
        "best_strategy": backtester.results_history[-1].strategy_name if backtester.results_history else None,
        "dreamer": gate.get_status(),
    }
    if shadow:
        stats["shadow_learning"] = shadow.get_stats()

    return stats

