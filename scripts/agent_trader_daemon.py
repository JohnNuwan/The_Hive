#!/usr/bin/env python3
"""THE HIVE - Head Trader Agent (LLM Swarm)

Ce daemon surveille la Memoire Partagee (Redis) pour recuperer les rapports
des Agents Analystes (Chartist, Technical). Il les soumet ensuite au LLM
configure (via OpenRouter ou vLLM) qui prend une decision d'execution finale.
Si un ordre valide est decide, il est transmis au Master Banker.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import redis
except ImportError:
    print("Veuillez installer redis: pip install redis")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HeadTrader")

BANKER_URL = "http://127.0.0.1:8100/orders"
REDIS_HOST = "192.168.1.6"
REDIS_PASSWORD = "devpassword"

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")

def get_analyst_reports(r: redis.Redis) -> dict[str, str]:
    """Recupere tous les rapports d'analyse publies par la Swarm."""
    reports = {}
    for key in r.scan_iter("hive:agent:chartist:report:*"):
        symbol = key.decode().split(":")[4]
        tf = key.decode().split(":")[5]
        report = r.get(key)
        if report:
            reports[f"{symbol}_{tf}"] = report.decode("utf-8")
    return reports

def ask_head_trader(symbol: str, analysis: str) -> dict | None:
    """Demande au LLM de prendre une decision de trading basee sur l'analyse."""
    if not LLM_API_KEY:
        logger.error("OPENROUTER_API_KEY non definie dans l'environnement.")
        return None

    prompt = f"""
Tu es l'Agent Head Trader de THE HIVE. Tu executes des ordres sur le Forex/Crypto en fonction des analyses fournies par ton equipe d'analystes.
Voici le dernier rapport d'analyse pour {symbol} :

{analysis}

INSTRUCTIONS STRICTES:
1. Analyse le rapport et decide si tu dois prendre position (BUY, SELL) ou attendre (WAIT).
2. Si tu decides de trader, tu DOIS fournir un Stop Loss (SL) valide et logique selon l'analyse (indispensable) et un Take Profit (TP).
3. Tu DOIS repondre UNIQUEMENT au format JSON valide, sans aucun texte autour, selon ce schema :
{{
    "action": "BUY" ou "SELL" ou "WAIT",
    "stop_loss": 1.2345,
    "take_profit": 1.2500,
    "confidence": 80,
    "reason": "Explication courte de ta decision"
}}
"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert AI Head Trader. Always output strict JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        logger.info("Soumission de l'analyse a %s pour decision...", LLM_MODEL)
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        
        content = resp.json()["choices"][0]["message"]["content"]
        # Nettoyage au cas ou le modele ajoute des blocs markdown
        content = content.replace("```json", "").replace("```", "").strip()
        
        decision = json.loads(content)
        return decision
        
    except Exception as e:
        logger.error("Erreur d'interrogation LLM: %s", e)
        return None

def execute_order(symbol: str, decision: dict) -> None:
    """Envoie l'ordre au Master Banker via l'API REST."""
    action = decision.get("action", "WAIT")
    if action == "WAIT":
        logger.info("Decision pour %s: WAIT. Raison: %s", symbol, decision.get("reason"))
        return

    payload = {
        "symbol": symbol.split("_")[0], # Ex: EURUSD_H4 -> EURUSD
        "action": action,
        "volume": 0.01, # Volume par defaut pour les tests
        "stop_loss": decision.get("stop_loss"),
        "take_profit": decision.get("take_profit"),
        "comment": "Agent LLM: " + str(decision.get("reason", ""))[:20],
        "source": "manual" # Utilisation d'une source acceptee
    }

    try:
        logger.info("Envoi de l'ordre au Banker : %s", payload)
        resp = requests.post(BANKER_URL, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Ordre execute avec succes ! Reponse: %s", resp.json())
    except Exception as e:
        logger.error("Erreur execution Banker: %s", e)

def main() -> None:
    parser = argparse.ArgumentParser(description="Head Trader Agent Swarm")
    parser.add_argument("--interval", type=int, default=3600, help="Intervalle de verification en secondes (defaut 1h)")
    args = parser.parse_args()

    r = redis.Redis(host=REDIS_HOST, port=6379, db=0, password=REDIS_PASSWORD)

    logger.info("=== Lancement du Head Trader Agent ===")
    logger.info("LLM Model : %s", LLM_MODEL)
    
    while True:
        try:
            reports = get_analyst_reports(r)
            if not reports:
                logger.info("Aucun rapport d'analyse trouve dans la Memoire Partagee.")
            else:
                for symbol, report in reports.items():
                    logger.info("Analyse du rapport pour %s...", symbol)
                    decision = ask_head_trader(symbol, report)
                    
                    if decision:
                        logger.info("Decision de l'Agent: %s (Confiance: %s%%)", decision.get("action"), decision.get("confidence"))
                        execute_order(symbol, decision)
                    
                    # On supprime le rapport pour ne pas le retraiter en boucle
                    # Dans une version avancee, on utiliserait un timestamp
                    r.delete(f"hive:agent:chartist:report:{symbol.split('_')[0]}:{symbol.split('_')[1]}")
                    
        except Exception as e:
            logger.error("Boucle principale erreur: %s", e)
            
        logger.info("Attente de %s secondes avant le prochain cycle...", args.interval)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
