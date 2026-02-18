"""
Cerveau de l'Expert Banker (The Brain).
Contient la logique décisionnelle (Manager), l'exécution (Worker) et la boucle d'autonomie.
"""

import asyncio
import logging
from decimal import Decimal
from uuid import UUID
from datetime import datetime
import os
import aiohttp

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
from eva_banker.strategist import Strategist

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
        
        # logger.info(f"Manager decision core triggered. Price: {price}, VaR: {var}, CVaR: {cvar}")
        
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
        # The Hive Mind: Multi-Asset Symbols
        self.symbols = ["XAUUSD", "EURUSD", "BTCUSD", "US30.cash"]
        self.latest_decisions = {} # Stores latest analysis per symbol
        
        # Sprint 7: The Cortex
        self.cortex = Strategist(mt5_service=mt5)
        
        # Sprint 8.5: Telegram Notification
        from shared.telegram_client import TelegramClient
        self.telegram = TelegramClient()

    async def start(self):
        """Démarre le pilote automatique"""
        if self.is_active:
            return
        self.is_active = True
        self._loop_task = asyncio.create_task(self._drift_loop())
        logger.info(f"🚀 AUTO-TRADING ENGINE STARTED on {self.symbols}")
        self.telegram.send_sync(f"🐝 **THE HIVE IS AWAKE**\nMonitoring: {self.symbols}\nRisk: {self.risk.max_risk_per_trade * 100}%")

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
        """Boucle principale de drift (Multi-Asset)"""
        logger.info("🌊 Entering Drift Loop (The Hive Mind)...")
        while self.is_active:
            try:
                # 1. Vérifier si trading autorisé (Loi 2 + Kill Switch)
                status = await self.risk.get_current_status()
                if not status.trading_allowed:
                    logger.warning("Auto-Trading paused: Risk limits hit or Kill-Switch active.")
                    await asyncio.sleep(60)
                    continue

                # 2. Vérifier positions ouvertes (Global Limit)
                # 2. Vérifier positions ouvertes (Global Limit)
                positions = await self.mt5.get_open_positions()
                
                # THE SHEPHERD (MANAGEMENT) 🐑
                # Loop through open positions to secure profits (Break-Even & Trailing)
                for pos in positions:
                    try:
                        # Skip if already secured/managed or recently opened (avoid noise)
                        if (datetime.now() - pos.open_time).total_seconds() < 60: continue
                        
                        current_price = float(pos.current_price)
                        open_price = float(pos.open_price)
                        sl = float(pos.stop_loss) if pos.stop_loss else 0.0
                        
                        # Calculate profit in points
                        if pos.action == TradeAction.BUY:
                            profit_points = current_price - open_price
                            # Break-Even Logic (Secure after ~150 points / 15 pips)
                            if profit_points > 1.5 and (sl == 0.0 or sl < open_price):
                                new_sl = open_price + 0.1 # Secure small profit
                                await self.mt5.modify_position(pos.ticket, sl=new_sl, tp=0.0)
                                msg = f"🛡️ **Shepherd**: Secured BUY {pos.symbol} at Break-Even ({new_sl})"
                                logger.info(msg)
                                self.telegram.send_sync(msg)
                            
                            # Trailing Stop (Follow if Profit > 300 points)
                            elif profit_points > 3.0:
                                trailing_sl = current_price - 1.0 # Trail by 1.0 (approx 10 pips)
                                if trailing_sl > sl:
                                    await self.mt5.modify_position(pos.ticket, sl=trailing_sl, tp=0.0)
                                    msg = f"🐑 **Shepherd**: Trailing BUY {pos.symbol} to {trailing_sl}"
                                    logger.debug(msg)
                                    if profit_points % 5.0 < 0.5: # Notify occasionally to avoid spam
                                        self.telegram.send_sync(msg)

                        elif pos.action == TradeAction.SELL:
                            profit_points = open_price - current_price
                            # Break-Even
                            if profit_points > 1.5 and (sl == 0.0 or sl > open_price):
                                new_sl = open_price - 0.1
                                await self.mt5.modify_position(pos.ticket, sl=new_sl, tp=0.0)
                                msg = f"🛡️ **Shepherd**: Secured SELL {pos.symbol} at Break-Even ({new_sl})"
                                logger.info(msg)
                                self.telegram.send_sync(msg)
                                
                            # Trailing
                            elif profit_points > 3.0:
                                trailing_sl = current_price + 1.0
                                if sl == 0.0 or trailing_sl < sl:
                                    await self.mt5.modify_position(pos.ticket, sl=trailing_sl, tp=0.0)
                                    msg = f"🐑 **Shepherd**: Trailing SELL {pos.symbol} to {trailing_sl}"
                                    logger.debug(msg)
                                    if profit_points % 5.0 < 0.5:
                                        self.telegram.send_sync(msg)

                    except Exception as e_shepherd:
                        logger.error(f"Shepherd Error on {pos.ticket}: {e_shepherd}")

                if len(positions) >= 5: # Increased global limit for multi-asset
                    logger.info("Max global positions reached (5). Waiting...")
                    await asyncio.sleep(60)
                    continue

                # 3. Iterate over symbols (The Hive Mind)
                for symbol in self.symbols:
                    if not self.is_active: break
                    
                    try:
                        # 0. CORTEX STRATEGY (Macro) - The Conscious Mind
                        # Refresh strategy every 15 minutes (900s)
                        last_strat = self.cortex.latest_strategy.get(symbol, {})
                        last_time = last_strat.get("timestamp")
                        
                        bias = "NEUTRAL"
                        should_refresh = not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() > 900
                        
                        if should_refresh:
                            # Non-blocking async call? No, we need the bias.
                            # But we don't want to block other symbols too long.
                            # For now, we await. It takes ~5-10s per symbol every 15m. Acceptable.
                            try:
                                strategy = await self.cortex.analyze_market_context(symbol)
                                bias = strategy.get("bias", "NEUTRAL")
                            except Exception as e_cortex:
                                logger.error(f"🧠 Cortex Error: {e_cortex}")
                        else:
                            bias = last_strat.get("bias", "NEUTRAL")

                        # A. Market Data
                        tick = await self.mt5.get_symbol_tick(symbol)
                        if not tick or (isinstance(tick, dict) and "bid" not in tick):
                            # logger.debug(f"Skipping {symbol}: No Tick Data")
                            continue
                            
                        current_price = tick['bid'] if isinstance(tick, dict) else tick.bid
                        
                        # B. AI Analysis (Dreamer V3)
                        candles = await self.mt5.get_recent_candles(symbol, count=100)
                        
                        # Indicators
                        from shared.indicators import IndicatorFactory
                        features = {}
                        rsi_val = 50.0
                        
                        if candles and len(candles) >= 50:
                            closes = [c["close"] for c in candles]
                            highs = [c["high"] for c in candles]
                            lows = [c["low"] for c in candles]
                            volumes = [c["tick_volume"] for c in candles]
                            
                            
                            rsi_val = IndicatorFactory.rsi(closes, 14).iloc[-1]
                            macd_data = IndicatorFactory.macd(closes)
                            bb_data = IndicatorFactory.bollinger_bands(closes)
                            atr_val = IndicatorFactory.atr(highs, lows, closes, 14).iloc[-1]
                            fib_levels = IndicatorFactory.get_fibonacci_levels(highs, lows, 100)
                            rvol = IndicatorFactory.relative_volume(volumes, 20).iloc[-1]
                            cycles = IndicatorFactory.detect_cycles(closes)
                            
                            features = {
                                "RSI": rsi_val,
                                "MACD_Hist": macd_data["histogram"].iloc[-1],
                                "BB_Pct": bb_data["pct_b"].iloc[-1],
                                "ATR": atr_val,
                                "RVOL": rvol,
                                "Cycle_High": cycles["bars_since_high"],
                                "Cycle_Low": cycles["bars_since_low"],
                                "Fib_0": fib_levels.get("fib_0", 0.0),
                            }
                            # Add full fib levels for vector mapping
                            for k, v in fib_levels.items():
                                features[k] = v
                                
                            logger.debug(f"👁️ {symbol} Vision: RSI={rsi_val:.1f} MACD={features['MACD_Hist']:.5f}")
                        else:
                            features = {"RSI": 50.0}

                        # C. Dreamer Inference
                        observation = {
                            "price": float(current_price),
                            "indicators": features
                        }
                        
                        action = None
                        comment = "Hold"
                        
                        try:
                            from shared.internal_auth import InternalAuth
                            # DYNAMIC URL: Use 'localhost' for native mode, 'lab' for docker
                            lab_host = os.getenv("LAB_HOST", "localhost")
                            lab_url = f"http://{lab_host}:8600/dreamer/predict"
                            token = InternalAuth.generate_token("banker")
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.post(lab_url, json=observation, headers={"X-Hive-Internal-Token": token}, timeout=2.0) as resp:
                                    if resp.status == 200:
                                        result = await resp.json()
                                        mz_action = result.get("action", 0)
                                        mz_value = result.get("value", 0.0)
                                        dreamer_comment = f"Dreamer V3 (v={mz_value:.2f})"
                                        
                                        if mz_action == 1:
                                            action = TradeAction.BUY
                                            comment = f"{dreamer_comment} -> BUY"
                                        elif mz_action == 2:
                                            action = TradeAction.SELL
                                            comment = f"{dreamer_comment} -> SELL"
                                    else:
                                        # Failover RSI
                                        if rsi_val < 30: action = TradeAction.BUY
                                        elif rsi_val > 70: action = TradeAction.SELL
                                        comment = f"Fallback RSI ({rsi_val:.1f})"
                        except Exception:
                            # Silent failover
                            if rsi_val < 30: action = TradeAction.BUY
                            elif rsi_val > 70: action = TradeAction.SELL
                            comment = "Fallback (Error)"

                            comment = "Fallback (Error)"

                        # CORTEX FILTER (The Conscious Check)
                        if action == TradeAction.BUY and bias == "BEARISH":
                            msg = f"🙅 Cortex VETO: Blocking BUY on {symbol} (Trend is BEARISH)"
                            logger.info(msg)
                            self.telegram.send_sync(msg)
                            action = None
                            comment = "Blocked by Cortex (Bearish Trend)"
                        elif action == TradeAction.SELL and bias == "BULLISH":
                            msg = f"🙅 Cortex VETO: Blocking SELL on {symbol} (Trend is BULLISH)"
                            logger.info(msg)
                            self.telegram.send_sync(msg)
                            action = None
                            comment = "Blocked by Cortex (Bullish Trend)"

                        # FORCE LOGGING for user visibility
                        logger.info(f"🧠 Analysis {symbol}: Price={current_price:.2f} RSI={rsi_val:.1f} -> Action={action} ({comment}) [Bias: {bias}]")

                        # Store Decision State
                        self.latest_decisions[symbol] = {
                            "price": float(current_price),
                            "rsi": rsi_val,
                            "macd": features.get("MACD_Hist", 0.0),
                            "action": str(action) if action else "WAIT",
                            "comment": comment,
                            "timestamp": datetime.now().isoformat()
                        }

                        if action is None:
                            continue

                        # D. Execution
                        skill = self.manager.plan_strategy({"price": float(current_price), "indicators": {"RSI": rsi_val}})
                        
                        atr = features.get("ATR", 0.0)
                        if atr > 0:
                            sl_dist = Decimal(str(atr * 1.5))
                            tp_dist = Decimal("0.0") # Let profits run (Shepherd Mode)
                        else:
                            sl_dist = Decimal("10.0") if "USD" in symbol else Decimal("0.0050")
                            tp_dist = Decimal("0.0") # Let profits run (Shepherd Mode)
                            
                        entry_price = Decimal(str(current_price))
                        sl_price = entry_price - sl_dist if action == TradeAction.BUY else entry_price + sl_dist
                        tp_price = Decimal("0.0") # No TP
                        
                        order = TradeOrder(
                            symbol=symbol,
                            action=action,
                            volume=Decimal("0.01"),
                            stop_loss_price=sl_price,
                            take_profit_price=tp_price,
                            comment=comment
                        )
                        
                        validation = await self.risk.validate_order(order)
                        if validation["allowed"]:
                            logger.info(f"🤖 EXEC {symbol}: {action} | {comment}")
                            result = await self.worker.execute_skill(skill, order)
                            if result.get("success"):
                                # Notify Telegram
                                self.telegram.send_sync(f"🚀 **OPEN EXECUTION**\n{symbol} {action.value} @ {entry_price}\nSL: {sl_price}\nMethod: {comment}")
                                asyncio.create_task(self._record_learning_experience(order, result))
                        else:
                            logger.warning(f"Rejected {symbol}: {validation['reason']}")
                            self.telegram.send_sync(f"⚠️ **REJECTED** {symbol}: {validation['reason']}")
                            
                        # Small delay between symbols
                        await asyncio.sleep(1.0)
                        
                    except Exception as e_sym:
                        logger.error(f"Error processing {symbol}: {e_sym}")
                        continue

                # 4. Wait (Drift Interval)
                await asyncio.sleep(300) 

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-Trading Loop Error: {e}")
                await asyncio.sleep(60)

    async def _record_learning_experience(self, order: TradeOrder, result: dict):
        """Envoie les données du trade au Lab pour Shadow Learning (DreamerV3)"""
        try:
            import aiohttp
            from shared.internal_auth import InternalAuth
            
            # DYNAMIC URL: Use 'localhost' for native mode, 'lab' for docker
            lab_host = os.getenv("LAB_HOST", "localhost")
            lab_url = f"http://{lab_host}:8600/shadow/record"
            
            payload = {
                "symbol": order.symbol,
                "action": order.action.value,
                "price": float(order.stop_loss_price) + 10.0, # Approx entry price if not in result
                "volume": float(order.volume),
                "pnl": 0.0, # PnL inconnu à l'ouverture
                "indicators": {"strategy": "drift_v1", "reason": order.comment},
                "done": False
            }
            
            token = InternalAuth.generate_token("banker")
            headers = {
                "X-Hive-Internal-Token": token
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(lab_url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        logger.info(f"🧠 Shadow Learning: Trade recorded in Lab (Ticket {result.get('ticket')})")
                    else:
                        logger.warning(f"Shadow Learning failed: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send shadow learning data: {e}")
