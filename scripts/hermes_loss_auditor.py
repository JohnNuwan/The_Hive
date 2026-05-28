#!/usr/bin/env python3
"""THE HIVE — Auditeur de Pertes Hermes (Loss Auditor).

Ce script analyse le journal des revues de transactions réelles de la journée,
isole les 3 pires transactions perdantes, interroge l'expert de trading
Hermes (LLM sur port 9500) pour un audit de conformité aux règles FTMO/FTUK,
génère un rapport détaillé en Markdown et envoie une alerte de synthèse.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import requests
from datetime import datetime
from pathlib import Path

# Résolution des chemins et injection du PYTHONPATH
WORKDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKDIR / "src" / "shared"))
sys.path.append(str(WORKDIR / "src" / "eva-lab"))

try:
    from shared.telegram_client import TelegramClient
except ImportError:
    # Client de secours robuste si la structure des imports est altérée hors conteneur
    class TelegramClient:
        """Client Telegram & Discord de secours minimal."""

        def __init__(self) -> None:
            self.token = os.getenv("TELEGRAM_BOT_TOKEN")
            self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
            self.topic_id = os.getenv("TELEGRAM_TOPIC_ID")
            self.enabled = bool(self.token and self.chat_id)
            
            # Initialisation de Discord pour le secours
            try:
                from shared.discord_client import DiscordClient
                self.discord = DiscordClient()
            except Exception:
                self.discord = None
        
        def send_sync(self, message: str) -> None:
            """Envoie de manière synchrone le message sur Telegram et Discord.

            Args:
                message (str): Texte brut à transmettre.
            """
            if self.discord:
                try:
                    self.discord.send_sync(message)
                except Exception:
                    pass

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
            try:
                requests.post(url, json=payload, timeout=10)
            except Exception:
                pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hermes_loss_auditor")

REVIEWS_DIR = WORKDIR / "data" / "live_trade_reviews"
REPORTS_DIR = WORKDIR / "data" / "reports"


def find_latest_review_file() -> Path | None:
    """Recherche le journal de revue de transactions réelles le plus récent.

    Returns:
        Path | None: Chemin absolu du fichier .jsonl ou None si introuvable.
    """
    if not REVIEWS_DIR.exists():
        logger.warning("Le répertoire des revues n'existe pas : %s", REVIEWS_DIR)
        return None
    files = sorted(REVIEWS_DIR.glob("live_trade_review_*.jsonl"), key=lambda p: p.name, reverse=True)
    if not files:
        logger.warning("Aucun journal de revue trouvé.")
        return None
    return files[0]


def analyze_losses(file_path: Path) -> list[dict]:
    """Analyse le fichier journal et isole les 3 pires transactions perdantes.

    Args:
        file_path (Path): Chemin absolu du fichier journal .jsonl.

    Returns:
        list[dict]: Liste des 3 pires transactions avec PNL négatif.
    """
    trades = []
    logger.info("Lecture des transactions depuis %s...", file_path.name)
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
                    logger.warning("Erreur de parsing sur une ligne de transaction : %s", e)
    except Exception as e:
        logger.error("Erreur de lecture du fichier %s : %s", file_path, e)
        return []

    # Tri par PNL croissant (la plus grande perte en premier)
    trades.sort(key=lambda t: float(t.get("pnl", 0.0)))
    return trades[:3]


def query_hermes_expert(trade: dict) -> str:
    """Interroge l'expert quantitatif Hermes pour obtenir un diagnostic de risque.

    Args:
        trade (dict): Dictionnaire de la transaction perdante.

    Returns:
        str: Diagnostic rédigé par l'IA Hermes.
    """
    url = "http://192.168.1.6:9500/chat"
    
    # Extraction des métadonnées et indicateurs de la transaction
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
        logger.info("Interrogation de l'expert Hermes pour %s (%s EUR)...", symbol, pnl)
        response = requests.post(url, json=payload, timeout=45)
        if response.status_code == 200:
            result = response.json()
            return result.get("message", result.get("response", "No response content from Hermes."))
        else:
            return f"Error from Hermes API (Status {response.status_code}): {response.text}"
    except Exception as exc:
        logger.error("Échec de l'interrogation Hermes : %s", exc)
        return f"Failed to reach Hermes expert: {exc}"


def main() -> None:
    """Routine principale d'analyse et d'audit Hermes."""
    latest_file = find_latest_review_file()
    if not latest_file:
        logger.error("Aucun fichier de revue trouvé. Audit annulé.")
        return
        
    worst_trades = analyze_losses(latest_file)
    if not worst_trades:
        logger.info("Aucune transaction perdante trouvée aujourd'hui. Excellent travail !")
        return
        
    logger.info("Trouvé %d transaction(s) perdante(s) à auditer.", len(worst_trades))
    
    audits = []
    for trade in worst_trades:
        diagnosis = query_hermes_expert(trade)
        audits.append((trade, diagnosis))
        
    # Création du rapport en Markdown
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "hermes_loss_audit_latest.md"
    
    report_content = []
    report_content.append(f"# 🛡️ Rapport d'Audit Hermes — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Fichier source : `{latest_file.name}`\n")
    report_content.append("Ce document audite les pires pertes de la journée face aux règles FTMO/FTUK.\n")
    
    telegram_lines = []
    telegram_lines.append(f"🚨 *AUDIT DE PERTE HERMES* ({datetime.now().strftime('%d/%m/%Y')})")
    telegram_lines.append(f"Fichier : `{latest_file.name}`\n")
    
    for i, (trade, diagnosis) in enumerate(audits, start=1):
        symbol = trade.get("symbol")
        pnl = trade.get("pnl")
        action = trade.get("action")
        ticket = trade.get("ticket")
        duration = trade.get("duration_minutes", 0.0)
        close_reason = trade.get("close_reason")
        
        # Section rapport Markdown
        report_content.append(f"## ❌ Transaction #{i}: {symbol} {action} (Ticket: {ticket})")
        report_content.append(f"- **PNL**: **{pnl} EUR**")
        report_content.append(f"- **Durée**: {duration:.1f} mins")
        report_content.append(f"- **Raison de clôture**: `{close_reason}`")
        report_content.append("\n### 🧠 Diagnostic d'Hermes :")
        report_content.append(diagnosis)
        report_content.append("\n" + "—" * 40 + "\n")
        
        # Section résumé synthétique
        short_diag = diagnosis.split("\n")[0] if diagnosis else "Aucun résumé."
        if len(short_diag) < 15 and len(diagnosis.split("\n")) > 1:
            short_diag = diagnosis.split("\n")[1]
        
        telegram_lines.append(f"*{i}. {symbol} {action}* (Ticket {ticket})")
        telegram_lines.append(f" 💸 PNL: *{pnl} EUR* | Clôture : `{close_reason}`")
        telegram_lines.append(f" 🧠 *Diagnostic* : {short_diag[:140]}...")
        telegram_lines.append("")
        
    report_content_str = "\n".join(report_content)
    try:
        report_path.write_text(report_content_str, encoding="utf-8")
        logger.info("Rapport Markdown écrit avec succès dans %s", report_path)
    except Exception as e:
        logger.error("Échec de l'écriture du rapport Markdown : %s", e)
        
    # Envoi de la notification
    telegram_lines.append("📝 Audit complet sauvegardé dans `data/reports/hermes_loss_audit_latest.md` sur le serveur.")
    telegram_msg = "\n".join(telegram_lines)
    
    try:
        logger.info("Envoi de la notification de synthèse...")
        TelegramClient().send_sync(telegram_msg)
    except Exception as e:
        logger.error("Échec de l'envoi de la notification : %s", e)


if __name__ == "__main__":
    main()
