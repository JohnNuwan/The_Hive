"""
Sprint 9.5 — Retrain DreamerV3 from Real MT5 History
Pulls closed trades from MT5 and sends them as feedback to Lab for retraining.

Usage:
    python scripts/retrain_from_history.py [--days 30]
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "eva-banker"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shared"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrain Dreamer from MT5 history")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to pull")
    args = parser.parse_args()

    # 1. Connect to MT5 and pull deal history
    logger.info(f"📊 Pulling last {args.days} days of MT5 trade history...")
    
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        return

    if not mt5.initialize():
        logger.error("❌ MT5 initialization failed")
        return

    from_dt = datetime.now() - timedelta(days=args.days)
    to_dt = datetime.now()
    
    deals = mt5.history_deals_get(from_dt, to_dt)
    if deals is None or len(deals) == 0:
        logger.warning("⚠️ No deals found in the specified period")
        mt5.shutdown()
        return
    
    # 2. Filter for closing deals (DEAL_ENTRY_OUT = 1)
    close_deals = []
    for deal in deals:
        if deal.entry == 1 and deal.symbol:  # DEAL_ENTRY_OUT
            close_deals.append({
                "ticket": deal.ticket,
                "position_id": deal.position_id,
                "symbol": deal.symbol,
                "type": "BUY" if deal.type == 0 else "SELL",
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "swap": deal.swap,
                "commission": deal.commission,
                "time": datetime.fromtimestamp(deal.time),
                "comment": deal.comment,
                "magic": deal.magic,
            })
    
    mt5.shutdown()
    
    logger.info(f"📈 Found {len(close_deals)} closed trades")
    
    if not close_deals:
        logger.info("Nothing to retrain on.")
        return
    
    # 3. Stats
    total_pnl = sum(d["profit"] + d.get("swap", 0) + d.get("commission", 0) for d in close_deals)
    wins = sum(1 for d in close_deals if d["profit"] > 0)
    losses = sum(1 for d in close_deals if d["profit"] < 0)
    
    logger.info(f"💰 Total P&L: ${total_pnl:.2f}")
    logger.info(f"✅ Wins: {wins} | ❌ Losses: {losses} | Win Rate: {wins/(wins+losses)*100:.0f}%")
    
    # 4. Send each trade as feedback to Lab
    import aiohttp
    
    lab_host = os.getenv("LAB_HOST", "localhost")
    lab_url = f"http://{lab_host}:8600/shadow/feedback"
    
    try:
        from shared.internal_auth import InternalAuth
        token = InternalAuth.generate_token("banker")
        headers = {"X-Hive-Internal-Token": token}
    except ImportError:
        headers = {}
    
    sent = 0
    errors = 0
    
    async with aiohttp.ClientSession() as session:
        for i, deal in enumerate(close_deals):
            pnl = deal["profit"] + deal.get("swap", 0) + deal.get("commission", 0)
            
            payload = {
                "symbol": deal["symbol"],
                "action": deal["type"],
                "price": deal["price"],
                "pnl": pnl,
                "indicators": {
                    "price_norm": deal["price"] / 3000.0,
                },
                "done": True
            }
            
            try:
                async with session.post(lab_url, json=payload, headers=headers, timeout=5.0) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        sent += 1
                        emoji = "✅" if pnl > 0 else "❌"
                        if (i + 1) % 10 == 0 or i == 0:
                            logger.info(
                                f"  [{i+1}/{len(close_deals)}] {emoji} {deal['symbol']} {deal['type']} "
                                f"P&L=${pnl:+.2f} → WM Loss={result.get('wm_loss', '?')}"
                            )
                    else:
                        errors += 1
                        logger.warning(f"  [{i+1}] Failed: HTTP {resp.status}")
            except Exception as e:
                errors += 1
                logger.error(f"  [{i+1}] Error: {e}")
            
            # Small delay to avoid overwhelming Lab
            await asyncio.sleep(0.1)
    
    logger.info(f"\n🏁 Retraining Complete!")
    logger.info(f"   Sent: {sent} | Errors: {errors}")
    logger.info(f"   The model has now learned from {sent} real trades.")
    logger.info(f"   Restart the Lab (docker compose restart lab) for fresh state.")


if __name__ == "__main__":
    asyncio.run(main())
