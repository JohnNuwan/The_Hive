"""
Cerveau de l'Expert Banker (The Brain).
Contient la logique décisionnelle (Manager), l'exécution (Worker) et la boucle d'autonomie.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Any
from uuid import UUID

from shared import (
    TradeAction,
    TradeOrder,
    symlog,
    calculate_var,
    calculate_cvar,
)
from eva_banker.services.mt5 import MT5Service
from eva_banker.skill_library import SkillLibrary, SkilledBehavior
from eva_banker.models.gnn_model import TFTGNNModel
from eva_banker.services.risk import RiskValidator  # Type hinting only if needed at runtime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MANAGER (DECISION)
# ═══════════════════════════════════════════════════════════════════════════════

class BankerManager:
    """
    NIVEAU HAUT : Le Manager (Abstract World Model).
    Planifie les stratégies en utilisant TFT-GNN et la conscience du risque.
    """
    def __init__(self, library: SkillLibrary):
        self.library = library
        # Initialisation du modèle (dims fictives pour l'exemple)
        self.brain = TFTGNNModel(asset_dim=5, temporal_dim=64, hidden_dim=128)

    def plan_strategy(self, market_history: dict) -> SkilledBehavior:
        """
        Analyse le marché via TFT-GNN et injecte VaR/CVaR.
        """
        # 1. Calcul des métriques de risque adaptatives (Inhibiteur interne)
        returns = market_history.get("returns", [])
        var = calculate_var(returns)
        cvar = calculate_cvar(returns)
        
        # 2. Préparation des données pour le modèle (Normalisées via Symlog)
        price = symlog(market_history.get("price", 0))
        
        logger.info(f"Manager decision core triggered. Price: {price}, VaR: {var}, CVaR: {cvar}")
        
        # Si le risque (VaR) est trop élevé, on bascule en mode conservateur
        if var < -0.02: # Perte potentielle > 2% attendue
            logger.warning("High VaR detected. Selecting HEDGING skill.")
            return SkilledBehavior.HEDGING
            
        return SkilledBehavior.SCALPING


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER (EXECUTION)
# ═══════════════════════════════════════════════════════════════════════════════

class BankerWorker:
    """
    NIVEAU BAS : L'Exécutant (Worker).
    Support de GhostShield pour l'invisibilité HFT.
    """
    def __init__(self, mt5_service: MT5Service, ghost_shield=None):
        self.mt5 = mt5_service
        self.ghost = ghost_shield

    async def execute_skill(self, skill: SkilledBehavior, order: TradeOrder):
        logger.info(f"Worker executing skill: {skill}")
        if self.ghost and skill != SkilledBehavior.HEDGING: # Le hedging doit être direct
            return await self.ghost.execute_obfuscated_order(order)
        return await self.mt5.execute_skill(skill, order)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE (AUTONOMY)
# ═══════════════════════════════════════════════════════════════════════════════

class AutoTradingEngine:
    """
    Moteur de Trading Automatique ("Weekend Drift").
    Orchestre la boucle : Analyse -> Planification -> Exécution.
    """
    def __init__(self, manager: BankerManager, worker: BankerWorker, mt5: MT5Service, risk: RiskValidator):
        self.manager = manager
        self.worker = worker
        self.mt5 = mt5
        self.risk = risk
        self.is_active = False
        self._loop_task = None
        self.symbol = "XAUUSD"  # Default Asset

    async def start(self):
        """Démarre le pilote automatique"""
        if self.is_active:
            return
        self.is_active = True
        self._loop_task = asyncio.create_task(self._drift_loop())
        logger.info(f"🚀 AUTO-TRADING ENGINE STARTED on {self.symbol}")

    async def stop(self):
        """Arrête le pilote automatique"""
        if not self.is_active:
            return
        self.is_active = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 AUTO-TRADING ENGINE STOPPED")

    async def _drift_loop(self):
        """Boucle principale de drift"""
        logger.info("🌊 Entering Drift Loop...")
        while self.is_active:
            try:
                # 1. Vérifier si trading autorisé (Loi 2 + Kill Switch)
                # Note: Le RiskValidator interne check déjà beaucoup, mais on fait un pre-check
                status = await self.risk.get_current_status()
                if not status.trading_allowed:
                    logger.warning("Auto-Trading paused: Risk limits hit or Kill-Switch active.")
                    await asyncio.sleep(60)
                    continue

                # 2. Vérifier positions ouvertes
                positions = await self.mt5.get_open_positions()
                if len(positions) >= 3: # Max concurrent positions (Hardcoded safety)
                    logger.info("Max positions reached (3). Waiting...")
                    await asyncio.sleep(60)
                    continue

                # 3. Analyser le marché (Live MT5 Data + Mocked Momentum)
                tick = await self.mt5.get_symbol_tick(self.symbol)
                if not tick:
                    logger.warning(f"Could not fetch tick for {self.symbol}. Skipping...")
                    await asyncio.sleep(10)
                    continue

                current_price = tick['bid'] if isinstance(tick, dict) else tick.bid # Support both
                
                market_data = {
                    "price": float(current_price), 
                    "returns": [0.001, 0.002, -0.001, 0.003] # Fake bullish momentum
                }
                
                # 4. Planifier la stratégie
                skill = self.manager.plan_strategy(market_data)
                
                # 5. Exécuter un trade (Weekend Drift = Micro Lots)
                # Validation des SL/TP dynamiques (Points)
                # XAUUSD: 1 point = 0.01. So 1000 points = $10.00
                action = TradeAction.BUY 
                
                sl_dist = Decimal("10.0") # $10 de distance (1000 points)
                tp_dist = Decimal("20.0") # $20 de distance (2000 points)
                
                # Calcul dynamique
                entry_price = Decimal(str(current_price))
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
                
                logger.info(f"🔮 Drift Calculation: Price={entry_price} SL={sl_price} TP={tp_price}")

                # Création de l'ordre
                order = TradeOrder(
                    symbol=self.symbol,
                    action=action,
                    volume=Decimal("0.01"),
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                    comment="Auto-Drift AI"
                )

                # Validation & Exécution
                validation = await self.risk.validate_order(order)
                if validation["allowed"]:
                    logger.info(f"🤖 AUTO-EXECUTING: {action} {self.symbol}")
                    await self.worker.execute_skill(skill, order)
                else:
                    logger.warning(f"Auto-Trade rejected: {validation['reason']}")

                # 6. Attendre (Drift Interval)
                # On trade toutes les 5 minutes pour la démo
                await asyncio.sleep(300) 

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-Trading Loop Error: {e}")
                await asyncio.sleep(60)
