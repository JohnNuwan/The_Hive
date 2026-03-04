"""
Cerveau de l'Expert Banker (The Brain).
Contient la logique décisionnelle (Manager), l'exécution (Worker) et la boucle d'autonomie.
"""

import asyncio
import logging
from decimal import Decimal
from uuid import UUID
from datetime import datetime, timedelta
import os
import aiohttp
import random
import uuid

from shared.redis_client import get_redis_client

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
from eva_banker.nemesis import get_nemesis_system # Import the Nemesis System
from eva_banker.services.news_filter import NewsFilterService # Import News Filter

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
        self._daily_report_task = None
        # The Hive Mind: Multi-Asset Symbols (Sprint 15: Global Universe)
        from shared import get_settings
        self.symbols = get_settings().banker_symbols
        self.latest_decisions = {} # Stores latest analysis per symbol
        
        # Sprint 7: The Cortex
        self.cortex = Strategist(mt5_service=mt5)
        
        # Sprint 8.5: Telegram Notification
        from shared.telegram_client import TelegramClient
        self.telegram = TelegramClient()
        
        # Sprint 9: Close Detection & Anti-Spam
        self._known_tickets = set()         # Tickets currently open (for close detection)
        self._last_veto_sent = {}           # symbol -> datetime (anti-spam)
        self._trade_open_info = {}          # ticket -> {symbol, action, entry_price, open_time, comment}
        
        # Sprint 10: News Filter 📰
        self.news = NewsFilterService(filter_minutes=30)
        self._news_task = None

    async def start(self):
        """Démarre le pilote automatique"""
        if self.is_active:
            return
        self.is_active = True
        # --- NEW (Sprint 12): State Sync on startup ---
        await self._sync_open_positions()
        
        self._loop_task = asyncio.create_task(self._drift_loop())
        self._daily_report_task = asyncio.create_task(self._half_day_report_loop())
        self._news_task = asyncio.create_task(self.news.start_monitoring())
        logger.info(f"🚀 AUTO-TRADING ENGINE STARTED on {self.symbols}")
        self.telegram.send_sync(
            f"🐝 *THE HIVE IS AWAKE*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Symbols: {', '.join(self.symbols)}\n"
            f"⚙️ Risk: {self.risk.max_risk_per_trade * 100}%\n"
            f"🕐 {datetime.now().strftime('%H:%M UTC+1')}"
        )

    async def stop(self):
        """Arrête le pilote automatique"""
        if not self.is_active:
            return
        self.is_active = False
        for task in [self._loop_task, self._daily_report_task, self._news_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("🛑 AUTO-TRADING ENGINE STOPPED")

    # ═══════════════════════════════════════════════════════════════════════════
    # TELEGRAM FORMATTERS (Sprint 9)
    # ═══════════════════════════════════════════════════════════════════════════

    def _fmt_open_msg(self, symbol: str, action: str, entry_price: float, sl_price: float,
                      rsi: float, atr: float, vwap: float, adx: float, cortex_bias: str, gnn_bias: str, 
                      comment: str, indicators: dict = None) -> str:
        """Formate un message d'ouverture riche avec indicateurs avancés."""
        sl_dist = abs(entry_price - sl_price)
        emoji = "🟢" if action == "BUY" else "🔴"
        
        # Format Indicators (Safe Get)
        indicators = indicators or {}
        macd = indicators.get("MACD_Hist", 0.0)
        bb_pct = indicators.get("BB_Pct", 0.5)
        rvol = indicators.get("RVOL", 1.0)
        sup = indicators.get("sr_sup", 0.0)
        res = indicators.get("sr_res", 0.0)
        
        # Visual MACD
        macd_icon = "📈" if macd > 0 else "📉"
        
        return (
            f"⚡ *E.V.A | New Position (M1/M15)*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 *Asset:* {symbol}\n"
            f"🔹 *Action:* {emoji} {action}\n"
            f"🔹 *Entry:* {entry_price:.5f}\n"
            f"🛡️ *S/L:* {sl_price:.5f} ({sl_dist:.2f} pts)\n\n"
            
            f"📊 *Markets & Signals:*\n"
            f"  • RSI: {rsi:.1f} | ADX: {adx:.1f}\n"
            f"  • VWAP: {vwap:.2f}\n"
            f"  • MACD: {macd_icon} {macd:.4f}\n"
            f"  • Vol: {rvol:.1f}x (Relative)\n"
            f"  • BB Position: {bb_pct*100:.1f}%\n"
            f"  • S/R: {sup:.2f} / {res:.2f}\n\n"
            
            f"🧠 *AI Reasoning:*\n"
            f"  • Cortex: {cortex_bias}\n"
            f"  • GNN (Proxmox): {gnn_bias}\n"
            f"  • *Logic:* {comment}\n\n"
            
            f"⏳ {datetime.now().strftime('%H:%M')} | The Hive"
        )

    def _fmt_close_msg(self, symbol: str, action: str, entry_price: float, exit_price: float,
                       profit: float, duration_min: int, reason: str = "SL/TP Hit") -> str:
        """Formate un message de fermeture riche."""
        pips = exit_price - entry_price
        if action == "SELL":
            pips = -pips
        # Normalize pips based on asset type
        pip_size = 0.1 if "XAU" in symbol else (1.0 if "US30" in symbol or "BTC" in symbol else 0.0001)
        pips_display = pips / pip_size
        
        emoji = "✅" if profit >= 0 else "❌"
        pnl_sign = "+" if profit >= 0 else ""
        
        # Duration formatting
        if duration_min >= 60:
            dur_str = f"{duration_min // 60}h{duration_min % 60:02d}m"
        else:
            dur_str = f"{duration_min}min"
        
        return (
            f"⚡ *E.V.A | Trade Closed*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 *Asset:* {symbol}\n"
            f"🔹 *Action:* {action}\n"
            f"🔹 *Result:* {emoji} {pnl_sign}{pips_display:.1f} pips\n\n"
            f"💰 *Financials:*\n"
            f"  • Entry: {entry_price:.5f}\n"
            f"  • Exit: {exit_price:.5f}\n"
            f"  • P&L: {pnl_sign}${profit:.2f}\n"
            f"  • Duration: {dur_str}\n\n"
            f"🏷️ *Reason:* {reason}\n"
            f"⏳ {datetime.now().strftime('%H:%M')} | The Hive"
        )

    def _fmt_shepherd_msg(self, symbol: str, action: str, event: str, 
                          new_sl: float, profit_pips: float) -> str:
        """Formate un message Shepherd enrichi."""
        return (
            f"🛡️ *E.V.A Shepherd | {event}*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 *Asset:* {symbol} {action}\n"
            f"🔹 *New S/L:* {new_sl:.5f}\n"
            f"🔹 *Secured:* +{profit_pips:.1f} pips"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # CLOSE DETECTION (Sprint 9)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _detect_closed_positions(self, current_positions: list):
        """Détecte les positions fermées et envoie une notification."""
        if current_positions is None:
            # Glitch in MT5 retrieval, abort detection safely
            return
            
        current_tickets = {pos.ticket for pos in current_positions}
        
        # Find tickets that disappeared (= closed)
        closed_tickets = self._known_tickets - current_tickets
        
        for ticket in closed_tickets:
            info = self._trade_open_info.get(ticket, {})
            if not info:
                continue
            
            # Try to get the actual close info from MT5 deal history
            try:
                from_dt = info.get("open_time", datetime.now() - timedelta(days=1))
                to_dt = datetime.now() + timedelta(days=1) # Deal with server timezone ahead of local
                deals = await self.mt5.get_deal_history(from_dt, to_dt)
                
                # Find the closing deal for this position
                close_deal = None
                for deal in deals:
                    if deal.get("position_id") == ticket or deal.get("magic") == 12345:
                        if deal.get("symbol") == info.get("symbol"):
                            close_deal = deal
                            break
                
                if close_deal:
                    profit = close_deal["profit"] + close_deal.get("swap", 0) + close_deal.get("commission", 0)
                    exit_price = close_deal["price"]
                    duration = (close_deal["time"] - info["open_time"]).total_seconds() / 60
                    reason = close_deal.get("comment", "SL/TP Hit") or "SL/TP Hit"
                else:
                    # Fallback: no deal found, use stored info
                    profit = 0.0
                    exit_price = info.get("entry_price", 0.0)
                    duration = (datetime.now() - info["open_time"]).total_seconds() / 60
                    reason = "Fermé (détails indisponibles)"
                
                msg = self._fmt_close_msg(
                    symbol=info["symbol"],
                    action=info["action"],
                    entry_price=info["entry_price"],
                    exit_price=exit_price,
                    profit=profit,
                    duration_min=int(duration),
                    reason=reason
                )
                self.telegram.send_sync(msg)
                logger.info(f"📤 Close notification sent for {info['symbol']} #{ticket} (P&L: ${profit:.2f})")
                
                # 🧠 FEEDBACK LOOP: Send real P&L to Lab for micro-training
                asyncio.create_task(self._send_pnl_feedback(
                    symbol=info["symbol"],
                    action=info["action"],
                    price=exit_price,
                    pnl=profit,
                ))
                
                # 🛡️ ANTI-TILT LOOP: Report losses to Nemesis for Self-Healing
                if profit < 0:
                    asyncio.create_task(get_nemesis_system().report_loss(
                        trade_id=str(ticket),
                        loss_amount=abs(profit),
                        market_context={"symbol": info["symbol"], "action": info["action"], "volatility": 0, "news_event": False, "trend_reversal": False}
                    ))
                
                # 💰 ACCOUNTANT LOOP: Send financial event for Drawdown validation
                asyncio.create_task(self._send_pnl_to_accountant(
                    symbol=info["symbol"],
                    profit=profit
                ))

                # 📸 VIRALIZATION LOOP: Send winning trade to Muse Media Factory (Port 8601)
                if profit >= 0.5:
                    asyncio.create_task(self._viralize_trade(
                        symbol=info["symbol"],
                        action=info["action"],
                        pnl=profit
                    ))
            except Exception as e:
                logger.error(f"Error processing closed ticket #{ticket}: {e}")
                # Cleanup if it failed mid-way
                self._trade_open_info.pop(ticket, None)
        
        # Update known tickets to current state
        self._known_tickets = current_tickets

    async def _sync_open_positions(self):
        """Peuple l'état au démarrage avec les positions existantes sur MT5 (Sprint 12)."""
        logger.info("🔄 Syncing existing positions from MT5 state...")
        try:
            positions = await self.mt5.get_open_positions()
            if positions is not None:
                for pos in positions:
                    self._known_tickets.add(pos.ticket)
                    self._trade_open_info[pos.ticket] = {
                        "symbol": pos.symbol,
                        "action": pos.action.value if hasattr(pos.action, 'value') else str(pos.action),
                        "entry_price": float(pos.open_price),
                        "open_time": pos.open_time,
                    }
                logger.info(f"✅ Synced {len(positions)} existing positions.")
        except Exception as e:
            logger.error(f"Failed to startup-sync positions: {e}")

    async def _viralize_trade(self, symbol: str, action: str, pnl: float):
        """Notifie l'agent The Muse pour générer une image virale d'un gain."""
        try:
            payload = {
                "symbol": symbol,
                "action": action,
                "pnl": pnl
            }
            # Muse run par défaut sur le port 9100 selon le docker-compose
            muse_url = f"http://{self.mt5.settings.api_host}:9100/viralize/trade"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(muse_url, json=payload, timeout=60) as resp:
                    if resp.status == 200:
                        logger.info(f"✨ Trade Viralization Success for {symbol}")
                    else:
                        logger.warning(f"⚠️ Muse Viralization Failed: {resp.status} - {await resp.text()}")
        except Exception as e:
            logger.error(f"Error calling Muse for viralization: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # DAILY REPORT (Sprint 9)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _half_day_report_loop(self):
        """Envoie un rapport récapitulatif toutes les demi-journées (midi et minuit)."""
        while self.is_active:
            try:
                now = datetime.now()
                # Determine next target: 11:55 or 23:55 (just before half-day ends)
                target1 = now.replace(hour=11, minute=55, second=0, microsecond=0)
                target2 = now.replace(hour=23, minute=55, second=0, microsecond=0)
                
                if now < target1:
                    next_report = target1
                elif now < target2:
                    next_report = target2
                else:
                    next_report = target1 + timedelta(days=1)
                
                wait_seconds = (next_report - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                if not self.is_active:
                    break
                
                await self._send_half_day_report()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Half-day report error: {e}")
                await asyncio.sleep(3600)

    async def _send_half_day_report(self):
        """Génère et envoie le rapport de la demi-journée."""
        try:
            now = datetime.now()
            # Define period:
            if now.hour < 15:
                period_name = "Matinée"
                period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                period_name = "Après-Midi"
                period_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
            
            period_end = now
            
            # Get deals from the period
            deals = await self.mt5.get_deal_history(period_start, period_end)
            
            # Get account info
            summary = await self.mt5.get_account_summary()
            
            total_trades = len(deals)
            wins = sum(1 for d in deals if d["profit"] > 0)
            losses = sum(1 for d in deals if d["profit"] < 0)
            total_pnl = sum(d["profit"] + d.get("swap", 0) + d.get("commission", 0) for d in deals)
            
            best_trade = max(deals, key=lambda d: d["profit"]) if deals and any(d["profit"] > 0 for d in deals) else None
            worst_trade = min(deals, key=lambda d: d["profit"]) if deals and any(d["profit"] < 0 for d in deals) else None
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            balance = summary.get("balance", 0)
            pnl_pct = (total_pnl / balance * 100) if balance > 0 else 0
            pnl_sign = "+" if total_pnl >= 0 else ""
            
            best_str = f"{best_trade['symbol']} +${best_trade['profit']:.2f}" if best_trade and best_trade["profit"] > 0 else "N/A"
            worst_str = f"{worst_trade['symbol']} ${worst_trade['profit']:.2f}" if worst_trade and worst_trade["profit"] < 0 else "N/A"
            
            # Additional Context for Report
            nemesis_str = "Actif" if self.risk._is_anti_tilt_active() else "Inactif"
            dd_pct = getattr(self.risk, "_get_daily_drawdown_percent", lambda: 0.0)()
            
            msg = (
                f"📈 *E.V.A | Bilan {period_name}*\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"📆 Date: {now.strftime('%d/%m/%Y %H:%M')}\n\n"
                f"📊 *Performances*\n"
                f"  • P&L: {pnl_sign}${total_pnl:.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                f"  • Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)\n"
                f"  • Balance: ${balance:,.2f}\n"
                f"  • Drawdown Journée: {dd_pct}%\n\n"
                f"🏆 *Top / Flop*\n"
                f"  • Best: {best_str}\n"
                f"  • Worst: {worst_str}\n\n"
                f"🛡️ *Sécurité*\n"
                f"  • Marge Libre: ${summary.get('margin_free', 0):,.2f}\n"
                f"  • Nemesis (Anti-Tilt): {nemesis_str}\n\n"
                f"🧠 _The Hive continuously learning._"
            )
            
            self.telegram.send_sync(msg)
        except Exception as e:
            logger.error(f"Error generating half-day report: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # ACCOUNTANT & LAB INTEGRATION (REST API)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _send_pnl_to_accountant(self, symbol: str, profit: float):
        """Envoie le résultat financier à l'Accountant (Port 8500) pour le suivi de la Drawdown"""
        try:
            import aiohttp
            import os
            # Use LAB_HOST as a fallback since Accountant runs on the same Proxmox server
            accountant_host = os.getenv("ACCOUNTANT_HOST", os.getenv("LAB_HOST", "localhost"))
            url = f"http://{accountant_host}:8500/pnl"
            
            # Re-fetch latest balance for true equity tracking
            summary = await self.mt5.get_account_summary()
            balance = float(summary.get("balance", 100000.0))
            equity = float(summary.get("equity", balance))
            
            payload = {
                "symbol": symbol,
                "profit_loss": profit,
                "balance": balance,
                "equity": equity
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                     if resp.status == 200:
                         logger.debug(f"P&L of ${profit:.2f} properly accounted.")
                     else:
                         logger.warning(f"Accountant returned HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"Failed to reach Accountant: {e}")

    async def _send_pnl_feedback(self, symbol: str, action: str, price: float, pnl: float):
        """Envoie le P&L réel d'une transaction fermée au Lab pour Shadow Learning"""
        try:
            import aiohttp
            import os
            lab_host = os.getenv("LAB_HOST", "localhost")
            url = f"http://{lab_host}:8600/shadow/record"
            
            payload = {
                "symbol": symbol,
                "action": action,
                "price": price,
                "volume": 0.01,
                "pnl": pnl,
                "done": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5.0) as resp:
                     if resp.status == 200:
                         logger.debug(f"P&L feedback for {symbol} sent to Lab.")
                     else:
                         logger.warning(f"Lab returned HTTP {resp.status} for P&L feedback.")
        except Exception as e:
            logger.warning(f"Failed to send P&L feedback to Lab: {e}")
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN DRIFT LOOP
    # ═══════════════════════════════════════════════════════════════════════════

    async def _drift_loop(self):
        """Boucle principale de drift (Multi-Asset)"""
        logger.info("🌊 Entering Drift Loop (The Hive Mind)...")
        while self.is_active:
            try:
                # 1. Vérifier si trading autorisé (Loi 2 + Kill Switch)
                status = await self.risk.get_current_status()
                nemesis = get_nemesis_system()
                
                # Fetch Balance and update RiskValidator
                summary = await self.mt5.get_account_summary()
                balance = Decimal(str(summary.get("balance", 100000)))
                self.risk.update_account_balance(balance)
                
                if not status.trading_allowed:
                    logger.warning("Auto-Trading paused: Risk limits hit or Kill-Switch active.")
                    await asyncio.sleep(60)
                    continue
                    
                if nemesis.should_block_trading():
                    logger.warning("Auto-Trading paused: Nemesis Meditation Phase Active (Consecutive Losses).")
                    await asyncio.sleep(60)
                    continue

                if nemesis.should_block_trading():
                    logger.warning("Auto-Trading paused: Nemesis Meditation Phase Active (Consecutive Losses).")
                    await asyncio.sleep(60)
                    continue

                # 2. Vérifier positions ouvertes (Global Limit)
                positions = await self.mt5.get_open_positions()
                
                # CLOSE DETECTION (Sprint 9)
                await self._detect_closed_positions(positions)
                
                # THE SHEPHERD (MANAGEMENT) 🐑
                # Loop through open positions to secure profits (Break-Even & Trailing)
                for pos in positions:
                    try:
                        # Skip if recently opened (avoid noise)
                        if (datetime.now() - pos.open_time).total_seconds() < 60: continue
                        
                        current_price = float(pos.current_price)
                        open_price = float(pos.open_price)
                        sl = float(pos.stop_loss) if pos.stop_loss else 0.0
                        
                        # Dynamic trailing thresholds based on asset class (FX vs Gold/Indices)
                        is_high_vol = "XAU" in pos.symbol or "BTC" in pos.symbol or "US30" in pos.symbol
                        be_threshold = 2.5 if is_high_vol else 0.0015 # Room to breathe
                        trail_activation = 5.0 if is_high_vol else 0.0030
                        trail_distance = 2.0 if is_high_vol else 0.0010
                        
                        if pos.action == TradeAction.BUY:
                            profit = current_price - open_price
                            # Break-Even Logic
                            if profit > be_threshold and (sl == 0.0 or sl < open_price):
                                new_sl = open_price + (0.1 if is_high_vol else 0.0001) # Secure small profit
                                await self.mt5.modify_position(pos.ticket, sl=new_sl, tp=0.0)
                                msg = self._fmt_shepherd_msg(pos.symbol, "BUY", "SECURED", new_sl, profit)
                                logger.info(msg)
                                self.telegram.send_sync(msg)
                            
                            # Trailing Stop 
                            elif profit > trail_activation:
                                trailing_sl = current_price - trail_distance
                                if trailing_sl > sl:
                                    await self.mt5.modify_position(pos.ticket, sl=trailing_sl, tp=0.0)
                                    msg = self._fmt_shepherd_msg(pos.symbol, "BUY", "TRAILING", trailing_sl, profit)
                                    logger.debug(msg)

                        elif pos.action == TradeAction.SELL:
                            profit = open_price - current_price
                            # Break-Even
                            if profit > be_threshold and (sl == 0.0 or sl > open_price):
                                new_sl = open_price - (0.1 if is_high_vol else 0.0001)
                                await self.mt5.modify_position(pos.ticket, sl=new_sl, tp=0.0)
                                msg = self._fmt_shepherd_msg(pos.symbol, "SELL", "SECURED", new_sl, profit)
                                logger.info(msg)
                                self.telegram.send_sync(msg)
                                
                            # Trailing
                            elif profit > trail_activation:
                                trailing_sl = current_price + trail_distance
                                if sl == 0.0 or trailing_sl < sl:
                                    await self.mt5.modify_position(pos.ticket, sl=trailing_sl, tp=0.0)
                                    msg = self._fmt_shepherd_msg(pos.symbol, "SELL", "TRAILING", trailing_sl, profit)
                                    logger.debug(msg)

                    except Exception as e_shepherd:
                        logger.error(f"Shepherd Error on {pos.ticket}: {e_shepherd}")

                if len(positions) >= 12: # Increased global limit for multi-asset (Sprint 15)
                    logger.info("Max global positions reached (12). Waiting...")
                    await asyncio.sleep(60)
                    continue

                # 3. Iterate over symbols (The Hive Mind)
                for symbol in self.symbols:
                    if not self.is_active: break
                    
                    try:
                        # NEW (Sprint 10) : Night Session Filter (Rollover Trap)
                        if not self.risk.is_within_trading_session(symbol):
                            continue
                            
                        # NEW: Localized News Tracking (Sprint 11 P3)
                        if getattr(self, "news", None) and self.news.should_block_trading(symbol):
                            continue
                            
                        # 3. Get Context from Cortex (Sprint 10)
                        last_strat = self.latest_decisions.get(symbol, {})
                        last_time = last_strat.get("timestamp")
                        
                        bias = "NEUTRAL"
                        gnn_bias = "N/A" # Default if not refreshed
                        should_refresh = not last_time or (datetime.now() - datetime.fromisoformat(last_time)).total_seconds() > 900
                        
                        if should_refresh:
                            try:
                                strat_result = await self.cortex.analyze_market_context(symbol)
                                self.latest_decisions[symbol] = strat_result
                                last_strat = strat_result  # Keep memory pointer fresh for Telegram output
                                bias = strat_result.get("bias", "NEUTRAL")
                                
                                # --- NEW (Sprint 12): Signal Reversal Logic ---
                                # Close positions of opposite direction if bias is strong
                                asyncio.create_task(self._handle_reversal(symbol, bias))
                                
                                # 4. Neural Analysis (GNN / Dreamer)
                                gnn_bias = strat_result.get("gnn_bias", "UNKNOWN")
                            except Exception as e_cortex:
                                logger.error(f"🧠 Cortex Error: {e_cortex}")
                        else:
                            bias = last_strat.get("bias", "NEUTRAL")
                            gnn_bias = last_strat.get("gnn_bias", "N/A")

                        # A. Market Data
                        tick = await self.mt5.get_symbol_tick(symbol)
                        if not tick or (isinstance(tick, dict) and "bid" not in tick):
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
                                "Fib_236": fib_levels.get("fib_236", 0.0),
                                "Fib_382": fib_levels.get("fib_382", 0.0),
                                "Fib_500": fib_levels.get("fib_500", 0.0),
                                "Fib_618": fib_levels.get("fib_618", 0.0),
                                "Fib_100": fib_levels.get("fib_100", 0.0)
                            }
                            
                            # ----- EXTENDED FEATURES (For future AI V4 training & Shadow Learning) -----
                            vwap_val = IndicatorFactory.vwap(highs, lows, closes, volumes).iloc[-1]
                            obv_val = IndicatorFactory.obv(closes, volumes).iloc[-1]
                            momentum_val = IndicatorFactory.momentum(closes, 10).iloc[-1]
                            trix_val = IndicatorFactory.trix(closes, 15).iloc[-1]
                            stoch_data = IndicatorFactory.stochastic(highs, lows, closes)
                            cci_val = IndicatorFactory.cci(highs, lows, closes).iloc[-1]
                            adx_data = IndicatorFactory.adx(highs, lows, closes)
                            ichimoku_data = IndicatorFactory.ichimoku(highs, lows, closes)
                            trendlines_val = IndicatorFactory.trendlines(closes).iloc[-1]
                            sr_data = IndicatorFactory.support_resistance(highs, lows, closes)
                            gann_data = IndicatorFactory.gann_angles(highs, lows, 100)

                            extended_features = {
                                "vwap": vwap_val,
                                "obv": obv_val,
                                "momentum": momentum_val,
                                "trix": trix_val,
                                "stoch_k": stoch_data["percent_k"].iloc[-1],
                                "stoch_d": stoch_data["percent_d"].iloc[-1],
                                "cci": cci_val,
                                "adx": adx_data["adx"].iloc[-1],
                                "adx_plus_di": adx_data["plus_di"].iloc[-1],
                                "adx_minus_di": adx_data["minus_di"].iloc[-1],
                                "ichi_tenkan": ichimoku_data["tenkan_sen"].iloc[-1],
                                "ichi_kijun": ichimoku_data["kijun_sen"].iloc[-1],
                                "ichi_senkou_a": ichimoku_data["senkou_span_a"].iloc[-1],
                                "ichi_senkou_b": ichimoku_data["senkou_span_b"].iloc[-1],
                                "trendline_slope": trendlines_val,
                                "sr_res": sr_data["nearest_resistance"],
                                "sr_sup": sr_data["nearest_support"],
                                "fib_786": fib_levels.get("fib_786", 0.0),
                                "fib_ext_1618": fib_levels.get("fib_ext_1618", 0.0),
                                "fib_ext_2618": fib_levels.get("fib_ext_2618", 0.0),
                                "gann_1x1": gann_data["gann_1x1"],
                                "gann_1x2": gann_data["gann_1x2"],
                                "gann_2x1": gann_data["gann_2x1"]
                            }
                                
                            # --- FORMATTED LOGGING ---
                            from colorama import Fore, Style
                            sym_color = Fore.CYAN if "XAU" in symbol else (Fore.YELLOW if "BTC" in symbol else Fore.WHITE)
                            bias_color = Fore.GREEN if bias == "BULLISH" else (Fore.RED if bias == "BEARISH" else Fore.LIGHTBLACK_EX)
                            
                            logger.info(
                                f"🧠 {sym_color}{symbol:<8}{Style.RESET_ALL} | "
                                f"Price: {current_price:<9.2f} | "
                                f"RSI: {rsi_val:<4.1f} | "
                                f"ADX: {adx_data['adx'].iloc[-1]:<4.1f} | VWAP: {vwap_val:<9.2f} | "
                                f"Cortex: {bias_color}[{bias}]{Style.RESET_ALL}"
                            )
                        else:
                            features = {"RSI": 50.0}
                            extended_features = {}

                        # C. Dreamer Inference
                        observation = {
                            "price": float(current_price),
                            "indicators": features
                        }
                        
                        action = None
                        comment = "Hold"
                        
                        try:
                            from shared.internal_auth import InternalAuth
                            lab_host = os.getenv("LAB_HOST", "localhost")
                            lab_url = f"http://{lab_host}:8600/dreamer/predict"
                            token = InternalAuth.generate_token("banker")
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.post(lab_url, json=observation, headers={"X-Hive-Internal-Token": token}, timeout=5.0) as resp:
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
                                            
                                        # Strict neural-network enforcement in production.
                                        # No Epsilon-Greedy, no Bias fallback loops allowing random entries.
                                    else:
                                        logger.error(f"Dreamer Inference failed: HTTP {resp.status}")
                                        action = None
                                        comment = f"Lab error (HTTP {resp.status})"
                                        
                        except Exception as e_lab:
                            # If connection to Lab completely fails, DO NOT trade. HOLD explicitly.
                            logger.error(f"Dreamer Inference failed for {symbol}: {e_lab.__class__.__name__} - {e_lab}. Holding.")
                            action = None
                            comment = "Error connecting to Lab"

                        if action == TradeAction.BUY and bias == "BEARISH":
                            logger.info(f"🙅 Cortex VETO: Blocking BUY on {symbol} (Trend is BEARISH on M15)")
                            action = None
                            comment = "Blocked by Cortex (Bearish Trend on M15)"
                        elif action == TradeAction.SELL and bias == "BULLISH":
                            logger.info(f"🙅 Cortex VETO: Blocking SELL on {symbol} (Trend is BULLISH on M15)")
                            action = None
                            comment = "Blocked by Cortex (Bullish Trend on M15)"

                        # FORCE LOGGING for user visibility
                        log_msg = f"[M1/M15] {symbol}: Price={current_price:.2f} RSI={rsi_val:.1f} -> Action={action} ({comment}) [Context: {bias}]"
                        logger.info(f"🧠 {log_msg}")

                        # PUBLISH TO AGENT FEED (UI)
                        redis = get_redis_client()
                        await redis.publish("eva.banker.feed", {
                            "id": str(uuid.uuid4()),
                            "source_agent": "Banker",
                            "action": f"Analyse {symbol}: Price={current_price:.2f} | RSI={rsi_val:.1f} -> {action or 'Hold'} ({comment})",
                            "timestamp": datetime.now().isoformat(),
                            "type": "request" if action is None else "event"
                        })

                        # Store Decision State
                        decision_state = self.latest_decisions.get(symbol, {})
                        decision_state.update({
                            "price": float(current_price),
                            "rsi": rsi_val,
                            "macd": features.get("MACD_Hist", 0.0),
                            "vwap": float(vwap_val),
                            "adx": float(adx_data["adx"].iloc[-1]),
                            "action": str(action) if action else "WAIT",
                            "comment": comment,
                            "timestamp": datetime.now().isoformat()
                        })
                        self.latest_decisions[symbol] = decision_state

                        if action is None:
                            continue

                        # D. Execution
                        skill = self.manager.plan_strategy({"price": float(current_price), "indicators": {"RSI": rsi_val}})
                        
                        atr = features.get("ATR", 0.0)
                        is_high_vol = "XAU" in symbol or "BTC" in symbol or "US30" in symbol
                        
                        if atr > 0:
                            # Multiply ATR to give the algorithm breathing room.
                            # Usually SL = ATR * 2.5 is minimum for trend following.
                            sl_dist = Decimal(str(atr * 3.0)) 
                            tp_dist = Decimal("0.0") # Let profits run (Shepherd Mode)
                        else:
                            # Realistic baseline stop loss distances if ATR fails
                            # e.g Gold ($10), Indices ($30), Forex (20 pips)
                            if "XAU" in symbol: sl_dist = Decimal("8.0")
                            elif "US30" in symbol or "BTC" in symbol: sl_dist = Decimal("30.0")
                            else: sl_dist = Decimal("0.0020")
                            tp_dist = Decimal("0.0") # Let profits run
                            
                        entry_price = Decimal(str(current_price))
                        sl_price = entry_price - sl_dist if action == TradeAction.BUY else entry_price + sl_dist
                        tp_price = Decimal("0.0") # No TP
                        
                        # Dynamic Volume Calculation (Sprint 10)
                        balance = self.risk._account_balance
                        risk_pct = self.risk.max_risk_per_trade
                        
                        dynamic_vol = self.risk.calculate_lot_size(
                            balance=balance,
                            risk_percent=risk_pct,
                            sl_distance=sl_dist,
                            symbol=symbol
                        )
                        
                        # Safety Caps
                        final_vol = min(0.10, max(0.01, dynamic_vol))
                        
                        safe_comment = comment[:30] if comment else ""
                        order = TradeOrder(
                            symbol=symbol,
                            action=action,
                            volume=Decimal(str(final_vol)),
                            stop_loss_price=sl_price,
                            take_profit_price=tp_price,
                            comment=safe_comment
                        )
                        if action:
                            # --- NEW (Sprint 13): Margin Pre-check ---
                            margin_required = await self.mt5.get_margin_required(symbol, action, final_vol) # Use final_vol here
                            account = await self.mt5.get_account_summary()
                            
                            if margin_required is not None and account:
                                free_margin = float(account.get("free_margin", 0.0))
                                if free_margin < margin_required:
                                    logger.error(f"❌ MARGIN VETO: {symbol} {action} requires ${margin_required:.2f}, but only ${free_margin:.2f} free.")
                                    self.telegram.send_sync(f"⚠️ *MARGIN VETO* | {symbol} {action} blocked\nNeed: ${margin_required:.2f} / Free: ${free_margin:.2f}")
                                    action = None
                                    comment = "Insufficient Margin"
                            
                        # If action was vetoed by margin check, skip execution
                        if action is None:
                            continue

                        validation = await self.risk.validate_order(order)
                        if validation["allowed"]:
                            logger.info(f"🤖 EXEC {symbol}: {action} | {comment}")
                            result = await self.worker.execute_skill(skill, order)
                            if result.get("success"):
                                # --- NEW (Sprint 11): LLM Micro-Reasoning ---
                                combined_indicators = {**features, **extended_features}
                                reasoning = await self.cortex.get_micro_reasoning(
                                    symbol=symbol, 
                                    action=action.value, 
                                    indicators=combined_indicators
                                )
                                
                                # Rich Telegram OPEN notification (Sprint 9/11)
                                open_msg = self._fmt_open_msg(
                                    symbol=symbol,
                                    action=action.value,
                                    entry_price=float(entry_price),
                                    sl_price=float(sl_price),
                                    rsi=rsi_val,
                                    atr=atr,
                                    vwap=extended_features.get("vwap", float(current_price)),
                                    adx=extended_features.get("adx", 0.0),
                                    cortex_bias=last_strat.get("cortex_bias", "UNKNOWN"),
                                    gnn_bias=last_strat.get("gnn_bias", "UNKNOWN"),
                                    comment=reasoning, # Replace technical comment with LLM Reasoning
                                    indicators=combined_indicators
                                )
                                self.telegram.send_sync(open_msg)
                                
                                # Track this position for close detection
                                ticket = result.get("ticket", 0)
                                if ticket:
                                    self._trade_open_info[ticket] = {
                                        "symbol": symbol,
                                        "action": action.value,
                                        "entry_price": float(entry_price),
                                        "open_time": datetime.now(),
                                    }
                                    self._known_tickets.add(ticket)
                                
                                asyncio.create_task(self._record_learning_experience(order, result, features, extended_features, float(current_price)))
                            else:
                                # ═══ ORDER FAILED — LOG IT ═══
                                fail_msg = result.get("message", "Unknown error")
                                fail_code = result.get("retcode", "?")
                                logger.error(f"❌ ORDER FAILED {symbol} {action}: {fail_msg} (retcode={fail_code})")
                                self.telegram.send_sync(
                                    f"❌ *ORDER FAILED* | {symbol} {action.value}\n"
                                    f"Reason: {fail_msg}\n"
                                    f"SL: {float(sl_price):.2f} | Vol: {float(order.volume)}"
                                )
                        else:
                            logger.warning(f"Rejected {symbol}: {validation['reason']}")
                            
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

    async def _record_learning_experience(self, order: TradeOrder, result: dict, features: dict, extended_features: dict, current_price: float):
        """Envoie les données du trade au Lab pour Shadow Learning (DreamerV3)"""
        try:
            import aiohttp
            from shared.internal_auth import InternalAuth
            
            lab_host = os.getenv("LAB_HOST", "localhost")
            lab_url = f"http://{lab_host}:8600/shadow/record"
            
            # Fuse core features and extended features for the background DB payload
            db_indicators = {**features, **extended_features}
            
            payload = {
                "symbol": order.symbol,
                "action": order.action.name,
                "price": float(order.stop_loss_price) if order.stop_loss_price else 0.0,
                "volume": float(order.volume),
                "pnl": 0.0, 
                "indicators": db_indicators,
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

    async def _send_pnl_feedback(self, symbol: str, action: str, price: float, pnl: float):
        """Envoie le P&L réel au Lab pour micro-training (Sprint 9.5)."""
        try:
            from shared.internal_auth import InternalAuth
            
            lab_host = os.getenv("LAB_HOST", "localhost")
            lab_url = f"http://{lab_host}:8600/shadow/feedback"
            
            payload = {
                "symbol": symbol,
                "action": action,
                "price": price,
                "pnl": pnl,
                "indicators": {"price_norm": price / 3000.0},
                "done": True
            }
            
            token = InternalAuth.generate_token("banker")
            headers = {"X-Hive-Internal-Token": token}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(lab_url, json=payload, headers=headers, timeout=5.0) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"🧠 Shadow Feedback: {symbol} P&L=${pnl:.2f} → Lab trained (loss={result.get('wm_loss', '?')})")
                    else:
                        logger.warning(f"Shadow Feedback failed: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send P&L feedback: {e}")

    async def _handle_reversal(self, symbol: str, bias: str):
        """Ferme les positions opposées au nouveau biais (Sprint 12)."""
        if bias not in ["BULLISH", "BEARISH"]:
            return
            
        for ticket, info in list(self._trade_open_info.items()):
            if info["symbol"] == symbol:
                should_close = False
                if bias == "BEARISH" and info["action"] == "BUY":
                    should_close = True
                elif bias == "BULLISH" and info["action"] == "SELL":
                    should_close = True
                
                if should_close:
                    logger.warning(f"🔄 Reversal {symbol}: Closing opposite {info['action']} #{ticket}")
                    await self.mt5.close_position(ticket)
                    # Note: notification de fermeture sera envoyée par le loop principal au prochain cycle
