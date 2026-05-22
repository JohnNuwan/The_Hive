#!/usr/bin/env python3
"""
THE HIVE — Hermes Loss Auditor
This script analyzes the latest daily live trade review positions log,
isolates the 3 worst losing trades, requests an audit from Hermes LLM expert (trading)
on port 9500 to evaluate the risk and compliance (FTMO/FTUK rules),
creates a detailed markdown report, and sends a Telegram alert.
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from pathlib import Path

# Setup paths
WORKDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKDIR / "src" / "shared"))
sys.path.append(str(WORKDIR / "src" / "eva-lab"))

try:
    from shared.telegram_client import TelegramClient
except ImportError:
    # Fallback to direct requests if path configuration is weird on host
    class TelegramClient:
        def __init__(self):
            self.token = os.getenv("TELEGRAM_BOT_TOKEN")
            self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
            self.topic_id = os.getenv("TELEGRAM_TOPIC_ID")
            self.enabled = bool(self.token and self.chat_id)
        
        def send_sync(self, message: str) -> None:
            if not self.enabled:
                return
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": True,
            }
            if self.topic_id:
                payload["message_thread_id"] = self.topic_id
            requests.post(url, json=payload, timeout=10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hermes_loss_auditor")

REVIEWS_DIR = WORKDIR / "data" / "live_trade_reviews"
REPORTS_DIR = WORKDIR / "data" / "reports"

def find_latest_review_file() -> Path | None:
    """Finds the most recent live trade review JSONL file."""
    if not REVIEWS_DIR.exists():
        logger.warning(f"Reviews directory does not exist: {REVIEWS_DIR}")
        return None
    files = sorted(REVIEWS_DIR.glob("live_trade_review_*.jsonl"), key=lambda p: p.name, reverse=True)
    if not files:
        logger.warning("No live trade reviews found.")
        return None
    return files[0]

def analyze_losses(file_path: Path) -> list[dict]:
    """Reads the JSONL and returns the worst 3 losing trades (pnl < 0)."""
    trades = []
    logger.info(f"Reading trades from {file_path.name}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    trade = json.loads(line)
                    if float(trade.get("pnl", 0.0)) < 0.0:
                        trades.append(trade)
                except Exception as e:
                    logger.warning(f"Error parsing trade line: {e}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

    # Sort by pnl ascending (most negative first)
    trades.sort(key=lambda t: float(t.get("pnl", 0.0)))
    return trades[:3]

def query_hermes_expert(trade: dict) -> str:
    """Queries the Hermes LLM trading expert for risk, FTMO/FTUK compliance, and technical analysis."""
    url = "http://192.168.1.6:9500/chat"
    
    # Extract trade details
    symbol = trade.get("symbol", "UNKNOWN")
    action = trade.get("action", "UNKNOWN")
    pnl = trade.get("pnl", 0.0)
    closed_at = trade.get("closed_at", "UNKNOWN")
    metadata = trade.get("metadata", {})
    entry_price = metadata.get("entry_price", 0.0)
    exit_price = metadata.get("exit_price", 0.0)
    entry_time = metadata.get("entry_time", "UNKNOWN")
    volume = metadata.get("volume", 0.0)
    cortex_bias = metadata.get("cortex_bias", "UNKNOWN")
    gnn_bias = metadata.get("gnn_bias", "UNKNOWN")
    indicators = metadata.get("indicators", {})
    rsi = indicators.get("rsi", "n/a")
    adx = indicators.get("adx", "n/a")
    vwap = indicators.get("vwap", "n/a")
    ema_200 = indicators.get("ema_200", "n/a")
    atr = indicators.get("atr", "n/a")
    close_reason = trade.get("close_reason", "UNKNOWN")
    
    prompt = f"""
Analyze this losing trade from THE HIVE automated fleet:
- Ticket/ID: {trade.get('ticket', 'n/a')}
- Symbol: {symbol}
- Action: {action} (Volume: {volume} lots)
- PNL: {pnl} EUR
- Timing: Entered at {entry_time}, Closed at {closed_at} ({trade.get('duration_minutes', 0.0):.2f} mins)
- Entry Price: {entry_price}, Exit Price: {exit_price}
- Veto/Bias context: Cortex bias = {cortex_bias}, GNN bias = {gnn_bias}
- Technical Indicators at entry/close: RSI={rsi}, ADX={adx}, ATR={atr}, VWAP={vwap}, EMA-200={ema_200}
- Close Reason: {close_reason}

Please provide an expert trading diagnostic:
1. Compare this to standard FTMO/FTUK rules (drawdown controls, leverage risks, single trade risk limit of 1%).
2. Detail the technical reason for the loss (e.g. entry against trend, spread expansion, premature stop, or lag in exiting).
3. Suggest a concrete correction (e.g. adjust stop loss, improve GNN/Cortex veto rules, avoid trading during specific regimes).
    """

    payload = {
        "message": prompt.strip(),
        "expert": "trading",
        "system_prompt": "You are Hermes, the master AI compliance officer and expert quantitative trading risk auditor. Analyze HIVE trade failures under strict FTMO/FTUK conditions.",
        "temperature": 0.2,
        "max_tokens": 1000
    }

    try:
        logger.info(f"Querying Hermes trading expert for {symbol} trade ({pnl} EUR)...")
        response = requests.post(url, json=payload, timeout=45)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response content from Hermes.")
        else:
            return f"Error from Hermes API (Status {response.status_code}): {response.text}"
    except Exception as exc:
        logger.error(f"Failed to query Hermes expert: {exc}")
        return f"Failed to reach Hermes expert: {exc}"

def main():
    latest_file = find_latest_review_file()
    if not latest_file:
        logger.error("No live trade review file found. Aborting audit.")
        return
        
    worst_trades = analyze_losses(latest_file)
    if not worst_trades:
        logger.info("No losing trades found in the latest review file. Excellent job!")
        return
        
    logger.info(f"Found {len(worst_trades)} losing trades to audit.")
    
    audits = []
    for trade in worst_trades:
        diagnosis = query_hermes_expert(trade)
        audits.append((trade, diagnosis))
        
    # Create markdown report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "hermes_loss_audit_latest.md"
    
    report_content = []
    report_content.append(f"# 🛡️ Hermes Loss Audit Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Source file: `{latest_file.name}`\n")
    report_content.append("This report audits the worst daily losing positions against FTMO/FTUK compliance rules and provides technical diagnostics from the Hermes LLM Trading Expert.\n")
    
    telegram_lines = []
    telegram_lines.append(f"🚨 *HERMES LOSS AUDIT* ({datetime.now().strftime('%d/%m/%Y')})")
    telegram_lines.append(f"Analyzed {latest_file.name}\n")
    
    for i, (trade, diagnosis) in enumerate(audits, start=1):
        symbol = trade.get("symbol")
        pnl = trade.get("pnl")
        action = trade.get("action")
        ticket = trade.get("ticket")
        duration = trade.get("duration_minutes", 0.0)
        close_reason = trade.get("close_reason")
        
        # Markdown Report section
        report_content.append(f"## ❌ Trade #{i}: {symbol} {action} (Ticket: {ticket})")
        report_content.append(f"- **PNL**: **{pnl} EUR**")
        report_content.append(f"- **Duration**: {duration:.1f} mins")
        report_content.append(f"- **Exit Reason**: `{close_reason}`")
        report_content.append("\n### 🧠 Hermes Diagnosis:")
        report_content.append(diagnosis)
        report_content.append("\n" + "—" * 40 + "\n")
        
        # Telegram Summary section
        short_diag = diagnosis.split("\n")[0] if diagnosis else "No summary available."
        # If the first line is headers/intro, try getting a longer snippet
        if len(short_diag) < 15 and len(diagnosis.split("\n")) > 1:
            short_diag = diagnosis.split("\n")[1]
        
        telegram_lines.append(f"*{i}. {symbol} {action}* (Ticket {ticket})")
        telegram_lines.append(f" 💸 PNL: *{pnl} EUR* | Exit: `{close_reason}`")
        telegram_lines.append(f" 🧠 *Diagnosis*: {short_diag[:140]}...")
        telegram_lines.append("")
        
    report_content_str = "\n".join(report_content)
    try:
        report_path.write_text(report_content_str, encoding="utf-8")
        logger.info(f"Markdown report written to {report_path}")
    except Exception as e:
        logger.error(f"Failed to write markdown report: {e}")
        
    # Send telegram alert
    telegram_lines.append(f"📝 Full audit saved in `data/reports/hermes_loss_audit_latest.md` on the server.")
    telegram_msg = "\n".join(telegram_lines)
    
    try:
        logger.info("Sending Telegram alert...")
        TelegramClient().send_sync(telegram_msg)
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")

if __name__ == "__main__":
    main()
