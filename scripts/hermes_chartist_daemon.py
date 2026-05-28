#!/usr/bin/env python3
"""THE HIVE — Daemon de Briefing Chartist et d'Analyse Technique Proactive (Hermes).

Ce script se connecte à MetaTrader 5, extrait l'historique des prix glissants,
calcule des indicateurs géométriques et chartistes évolués (Fibonacci, points pivots,
pentes de tendance, cycles), interroge l'expert Hermes pour une synthèse cognitive
et distribue le rapport sous forme d'Embed soigné dans le salon Discord `#analyse-technique`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

# Chargement de dotenv si disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Résolution des chemins et injection du PYTHONPATH
WORKDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKDIR / "src" / "shared"))

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from shared.discord_client import DiscordClient
except ImportError:
    class DiscordClient:
        """Client Discord de secours minimal."""
        def __init__(self) -> None:
            self.enabled = False
        def send_sync(self, message: str, category: str | None = None) -> None:
            print(f"[Discord Secours - Catégorie {category}]: {message}")

try:
    from shared.indicators import IndicatorFactory
except ImportError:
    # Factory de secours locale si l'import échoue hors dépôt
    class IndicatorFactory:
        """Calculateurs d'indicateurs de secours."""
        @staticmethod
        def support_resistance(highs, lows, closes, window=20) -> Dict[str, float]:
            return {
                "nearest_resistance": float(closes.max()),
                "nearest_support": float(closes.min()),
                "dist_to_res": 0.0,
                "dist_to_sup": 0.0
            }
        @staticmethod
        def get_fibonacci_levels(highs, lows, period=100) -> Dict[str, float]:
            h, l = float(highs.max()), float(lows.min())
            d = h - l
            return {
                "fib_0": l, "fib_236": l + d*0.236, "fib_382": l + d*0.382,
                "fib_500": l + d*0.5, "fib_618": l + d*0.618, "fib_786": l + d*0.786, "fib_100": h
            }
        @staticmethod
        def trendlines(closes, window=20) -> pd.Series:
            return pd.Series([0.0] * len(closes))
        @staticmethod
        def detect_cycles(closes) -> Dict[str, int]:
            return {"bars_since_high": 10, "bars_since_low": 10}
        @staticmethod
        def rsi(prices, period=14) -> pd.Series:
            return pd.Series([50.0] * len(prices))
        @staticmethod
        def adx(highs, lows, closes, period=14) -> Dict[str, pd.Series]:
            return {"adx": pd.Series([25.0] * len(closes)), "plus_di": pd.Series([20.0] * len(closes)), "minus_di": pd.Series([20.0] * len(closes))}
        @staticmethod
        def atr(highs, lows, closes, period=14) -> pd.Series:
            return pd.Series([1.0] * len(closes))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hermes_chartist_daemon")


def initialize_mt5() -> bool:
    """Initialise le terminal MetaTrader 5 avec les variables d'environnement.

    Returns:
        bool: True si l'initialisation a réussi, False sinon.
    """
    if mt5 is None:
        logger.warning("Le package MetaTrader5 n'est pas disponible dans cet environnement.")
        return False

    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")

    if login == 0:
        logger.warning("Identifiants MT5 non configurés dans le .env.")
        return False

    logger.info("Connexion au serveur MetaTrader 5 (%s)...", server)
    if not mt5.initialize(login=login, password=password, server=server):
        logger.error("Échec de l'initialisation de MT5: %s", mt5.last_error())
        return False

    logger.info("✅ Connexion MT5 établie avec succès pour le compte %d.", login)
    return True


def fetch_ohlc_data(symbol: str, timeframe_str: str, bars_count: int) -> pd.DataFrame | None:
    """Récupère l'historique récent des bougies depuis MT5 pour un symbole donné.

    Args:
        symbol (str): Le symbole de l'actif (ex: 'XAUUSD').
        timeframe_str (str): Le timeframe cible (M5, H1, H4, D1).
        bars_count (int): Nombre de bougies à récupérer.

    Returns:
        pd.DataFrame | None: DataFrame normalisé des prix ou None en cas d'erreur.
    """
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1 if mt5 else 1,
        "M5": mt5.TIMEFRAME_M5 if mt5 else 5,
        "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
        "H1": mt5.TIMEFRAME_H1 if mt5 else 60,
        "H4": mt5.TIMEFRAME_H4 if mt5 else 240,
        "D1": mt5.TIMEFRAME_D1 if mt5 else 1440,
    }

    tf = timeframe_map.get(timeframe_str.upper())
    if tf is None:
        logger.error("Timeframe non supporté : %s", timeframe_str)
        return None

    # Sélection active du symbole sur le Market Watch
    if not mt5.symbol_select(symbol, True):
        logger.warning("Impossible de sélectionner le symbole %s. Tentative de récupération best-effort.", symbol)

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars_count)
    if rates is None or len(rates) == 0:
        logger.error("Impossible de récupérer les bougies pour %s (%s). Erreur : %s", symbol, timeframe_str, mt5.last_error())
        return None

    # Normalisation dans un DataFrame Pandas
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def generate_mock_data(symbol: str, count: int = 200) -> pd.DataFrame:
    """Génère un jeu de données synthétique pour les tests hors connexion MT5.

    Args:
        symbol (str): Nom de l'actif.
        count (int): Nombre de lignes à générer.

    Returns:
        pd.DataFrame: DataFrame normalisé de bougies synthétiques.
    """
    logger.info("Génération de données synthétiques de test pour %s...", symbol)
    base_price = 2000.0 if "XAU" in symbol else 18000.0 if "US100" in symbol else 65000.0
    
    # Génération d'une marche aléatoire réaliste
    np_random = pd.Series(pd.np.random.normal(0.0005, 0.005, count) if hasattr(pd, "np") else [0.0] * count)
    closes = base_price * (1 + np_random.cumsum())
    opens = closes.shift(1).fillna(base_price)
    highs = closes.combine(opens, max) * 1.002
    lows = closes.combine(opens, min) * 0.998
    
    df = pd.DataFrame({
        "time": pd.date_range(end=datetime.now(), periods=count, freq="4h"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": [1000] * count,
        "real_volume": [1000] * count,
        "spread": [20] * count
    })
    return df


def calculate_chartist_coordinates(df: pd.DataFrame) -> dict[str, Any]:
    """Calcule toutes les coordonnées géométriques et chartistes glissantes.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les bougies (open, high, low, close).

    Returns:
        dict[str, Any]: Un dictionnaire de coordonnées formaté.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # 1. Supports & Résistances pivots
    sup_res = IndicatorFactory.support_resistance(high, low, close, window=20)

    # 2. Niveaux de retracement Fibonacci
    fibs = IndicatorFactory.get_fibonacci_levels(high, low, period=100)

    # 3. Pente linéaire de tendance (régression sur 20 bougies)
    slopes = IndicatorFactory.trendlines(close, window=20)
    last_slope = float(slopes.iloc[-1])

    # 4. Cycles de Donchian
    cycles = IndicatorFactory.detect_cycles(close)

    # 5. Oscillateurs de momentum et volatilité
    rsi_series = IndicatorFactory.rsi(close, 14)
    last_rsi = float(rsi_series.iloc[-1])

    adx_dict = IndicatorFactory.adx(high, low, close, 14)
    last_adx = float(adx_dict["adx"].iloc[-1])
    last_plus_di = float(adx_dict["plus_di"].iloc[-1])
    last_minus_di = float(adx_dict["minus_di"].iloc[-1])

    atr_series = IndicatorFactory.atr(high, low, close, 14)
    last_atr = float(atr_series.iloc[-1])
    last_close = float(close.iloc[-1])

    return {
        "close": last_close,
        "support": sup_res["nearest_resistance"], # Pivots
        "resistance": sup_res["nearest_support"],
        "dist_to_res": sup_res["dist_to_res"],
        "dist_to_sup": sup_res["dist_to_sup"],
        "fibs": fibs,
        "trend_slope": last_slope,
        "rsi": last_rsi,
        "adx": last_adx,
        "plus_di": last_plus_di,
        "minus_di": last_minus_di,
        "atr": last_atr,
        "atr_pct": (last_atr / last_close) * 100.0,
        "bars_since_high": cycles["bars_since_high"],
        "bars_since_low": cycles["bars_since_low"],
    }


def generate_chart_image(df: pd.DataFrame, coords: dict[str, Any], symbol: str, timeframe: str) -> bytes | None:
    """Génère une image de graphique technique haut de gamme avec supports, résistances et Fibonacci.

    Args:
        df (pd.DataFrame): Données historiques OHLC.
        coords (dict[str, Any]): Coordonnées géométriques calculées.
        symbol (str): Symbole de l'actif.
        timeframe (str): Unité de temps.

    Returns:
        bytes | None: Données binaires du graphique PNG, ou None en cas d'erreur.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        import matplotlib.dates as mdates

        # Configuration du style sombre premium (Style The Hive)
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f12")
        ax.set_facecolor("#15151c")

        # Sélection des 100 dernières bougies pour la lisibilité
        plot_df = df.iloc[-100:].copy()
        
        # Tracé des prix de clôture (Ligne néon brillante)
        x = plot_df["time"]
        y = plot_df["close"].astype(float)
        
        # Effet de lueur (glow effect)
        ax.plot(x, y, color="#00d2ff", alpha=0.2, linewidth=5)
        ax.plot(x, y, color="#00d2ff", linewidth=1.5, label="Prix de Clôture")

        # Tracé du Support Majeur Pivot (qui correspond à la résistance pivot en calcul)
        ax.axhline(coords["support"], color="#ff4757", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"Résistance Pivot ({coords['support']:.2f})")
        
        # Tracé de la Résistance Majeure Pivot (qui correspond au support pivot en calcul)
        ax.axhline(coords["resistance"], color="#2ed573", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"Support Pivot ({coords['resistance']:.2f})")

        # Tracé du Golden Pocket de Fibonacci (61.8%)
        fib618 = coords["fibs"]["fib_618"]
        ax.axhline(fib618, color="#ffa502", linestyle="-.", linewidth=1.5, alpha=0.9,
                   label=f"Fib 61.8% Golden Pocket ({fib618:.2f})")

        # Tracé des autres retracements clés de Fibonacci
        fibs_to_plot = {
            "23.6%": coords["fibs"]["fib_236"],
            "38.2%": coords["fibs"]["fib_382"],
            "50.0%": coords["fibs"]["fib_500"],
            "78.6%": coords["fibs"]["fib_786"],
        }
        for label, val in fibs_to_plot.items():
            ax.axhline(val, color="#7f8c8d", linestyle=":", linewidth=0.8, alpha=0.5)
            # Ajout d'étiquettes de texte discrètes à droite
            ax.text(x.iloc[-1], val, f" Fib {label}", color="#7f8c8d", fontsize=8, va="center")

        # Mise en valeur du prix de clôture actuel (Point brillant)
        last_x = x.iloc[-1]
        last_y = y.iloc[-1]
        ax.plot(last_x, last_y, marker="o", color="#39ff14", markersize=7, label=f"Dernier : {last_y:.2f}")

        # Titres et étiquettes
        ax.set_title(f"STRUCTURE TECHNIQUE DE PRÉCISION — {symbol} [{timeframe.upper()}]",
                     color="#ffffff", fontsize=11, fontweight="bold", pad=15)
        
        # Formatage des axes (affichage simple du temps)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        fig.autofmt_xdate()
        
        # Discrétion des bordures (spines)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#2f3542")
        ax.spines["left"].set_color("#2f3542")
        
        # Grille
        ax.grid(True, linestyle=":", alpha=0.15, color="#7f8c8d")
        ax.tick_params(colors="#a4b0be", labelsize=8)

        # Légende premium
        ax.legend(facecolor="#0f0f12", edgecolor="#2f3542", loc="upper left", fontsize=8)

        # Sauvegarde en mémoire
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f0f12")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
        
    except Exception as e:
        logger.error("Échec de génération du graphique matplotlib pour %s : %s", symbol, e)
        return None


def query_hermes_chartist(symbol: str, timeframe: str, coords: dict[str, Any]) -> str:
    """Interroge l'expert Hermes pour obtenir la synthèse chartiste.

    Args:
        symbol (str): Actif concerné.
        timeframe (str): Horizon temporel.
        coords (dict[str, Any]): Coordonnées mathématiques.

    Returns:
        str: Bulletin rédigé par Hermes.
    """
    prompt = f"""
Vous êtes Hermes, le Maître Chartist de THE HIVE et expert en analyse technique quantitative de haut niveau.
Analysez la structure géométrique et le comportement des prix pour {symbol} en unité de temps {timeframe} :

[Coordonnées de Marché Actuelles]
- Prix de Clôture Actuel : {coords['close']:.2f}
- Résistance Majeure de Pivot : {coords['support']:.2f} (Distance: {coords['dist_to_res']:.2f})
- Support Majeur de Pivot : {coords['resistance']:.2f} (Distance: {coords['dist_to_sup']:.2f})
- Niveaux Clés de Retracement de Fibonacci (Swing récent sur 100 bougies) :
  * 0% (Bas Local) : {coords['fibs']['fib_0']:.2f}
  * 23.6% : {coords['fibs']['fib_236']:.2f}
  * 38.2% : {coords['fibs']['fib_382']:.2f}
  * 50.0% (Zone Neutre) : {coords['fibs']['fib_500']:.2f}
  * 61.8% (Golden Pocket / Zone de Confluence) : {coords['fibs']['fib_618']:.2f}
  * 78.6% : {coords['fibs']['fib_786']:.2f}
  * 100% (Haut Local) : {coords['fibs']['fib_100']:.2f}
- Dynamique de Tendance :
  * Pente de Régression Linéaire (20p) : {coords['trend_slope']:.6f}
  * Force ADX : {coords['adx']:.2f} (Di+ = {coords['plus_di']:.2f}, Di- = {coords['minus_di']:.2f})
  * Volatilité ATR : {coords['atr']:.2f} ({coords['atr_pct']:.2f}% du prix)
- Oscillateur de Momentum :
  * RSI (14p) : {coords['rsi']:.2f}
- Analyse Cyclique (Donchian) :
  * Bougies écoulées depuis le plus haut local : {coords['bars_since_high']}
  * Bougies écoulées depuis le plus bas local : {coords['bars_since_low']}

Veuillez rédiger un bulletin chartiste extrêmement rigoureux, structuré et professionnel (en Français) :
1. **Analyse de Structure & Tendance** : Évaluez la tendance de fond (haussière, baissière, range), sa force (ADX) et son orientation.
2. **Confluences et Zones Clés de Réaction** : Détaillez les opportunités d'achat ou de vente à proximité des supports/résistances ou du Golden Pocket de Fibonacci (61.8%). Précisez si nous approchons d'une zone de breakout ou de rejet.
3. **Verdict Chartist & Biais Tactique** : Donnez le biais global (HAUSSIER / BAISSIER / NEUTRE) et un indice de confiance (Faible / Modéré / Élevé).
"""

    hermes_url = "http://192.168.1.6:9500/chat"
    payload = {
        "message": prompt.strip(),
        "expert": "trading",
        "system_prompt": "You are Hermes, the master AI chief chartist and technical analysis officer. Analyze indicators and structural patterns for the trading fleet.",
        "temperature": 0.3,
        "max_tokens": 1000
    }

    try:
        logger.info("Interrogation de l'expert Hermes pour %s (%s)...", symbol, timeframe)
        response = requests.post(hermes_url, json=payload, timeout=45)
        if response.status_code == 200:
            result = response.json()
            return result.get("message", result.get("response", "Aucun contenu reçu de Hermes."))
    except Exception as exc:
        logger.warning("Échec de connexion directe à Hermes (port 9500) : %s. Tentative de secours vLLM...", exc)

    # Secours vers vLLM Direct (port 8000)
    vllm_url = "http://192.168.1.6:8000/v1/chat/completions"
    vllm_model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    vllm_payload = {
        "model": vllm_model,
        "messages": [
            {"role": "system", "content": "Vous êtes Hermes, le Maître Chartist de THE HIVE. Rédigez des bulletins d'analyse technique rigoureux et clairs en Français."},
            {"role": "user", "content": prompt.strip()}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        logger.info("Interrogation du serveur vLLM de secours (%s)...", vllm_model)
        response = requests.post(vllm_url, json=vllm_payload, timeout=45)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return content
    except Exception as exc:
        logger.error("Échec de toutes les méthodes d'inférence LLM : %s", exc)
        
    # Fallback ultime purement algorithmique
    return f"""### 📊 ANALYSE TECHNIQUE AUTOMATIQUE (FALLBACK) — {symbol} [{timeframe}]
*   **Prix Actuel** : {coords['close']:.2f}
*   **Structure** : Résistance à {coords['support']:.2f} | Support à {coords['resistance']:.2f}.
*   **Fibonacci 61.8%** : {coords['fibs']['fib_618']:.2f}.
*   **Dynamique** : RSI à {coords['rsi']:.2f} (Régime : {'Suracheté' if coords['rsi'] > 70 else 'Surtendu' if coords['rsi'] < 30 else 'Neutre'}). ADX à {coords['adx']:.2f} (Tendance {'forte' if coords['adx'] > 25 else 'faible'}).
*   *Note : L'expert Hermes n'a pas pu être joint pour une synthèse rédigée.*"""


def main() -> None:
    """Routine principale d'exécution du Daemon Chartist."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Daemon d'Analyse Chartist Proactive — THE HIVE.")
    parser.add_argument("--symbols", default="XAUUSD,US100.cash,BTCUSD", help="Symboles séparés par des virgules.")
    parser.add_argument("--timeframe", default="H4", help="Timeframe (M5, H1, H4, D1).")
    parser.add_argument("--bars", type=int, default=200, help="Nombre de bougies d'historique.")
    parser.add_argument("--dry-run", action="store_true", help="Calcule uniquement sans envoyer d'alertes.")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    logger.info("Démarrage du briefing chartist pour les symboles : %s", symbols)

    mt5_active = initialize_mt5()

    reports = []
    for symbol in symbols:
        df = None
        if mt5_active:
            df = fetch_ohlc_data(symbol, args.timeframe, args.bars)

        if df is None:
            # Fallback sur données synthétiques pour dry-run/secours
            df = generate_mock_data(symbol, args.bars)

        try:
            coords = calculate_chartist_coordinates(df)
            briefing = query_hermes_chartist(symbol, args.timeframe, coords)
            
            # Génération du graphique technique
            chart_bytes = generate_chart_image(df, coords, symbol, args.timeframe)
            
            # Mise en page du bulletin technique pour le symbole
            symbol_report = f"### 📊 BRIEFING CHARTISTE — {symbol} [{args.timeframe.upper()}]\n"
            symbol_report += f"**Cours de clôture : {coords['close']:.2f}**\n\n"
            symbol_report += briefing
            reports.append((symbol_report, chart_bytes))
            
            print(f"\n--- RAPPORT {symbol} ---\n{symbol_report}\n")
        except Exception as exc:
            logger.error("Erreur de traitement pour le symbole %s : %s", symbol, exc, exc_info=True)

    if mt5_active:
        mt5.shutdown()
        logger.info("MT5 déconnecté.")

    if args.dry_run:
        logger.info("Exécution en mode simulation terminée. Aucun message envoyé.")
        return

    # Envoi individuel par symbole à Discord (évite la limite de taille d'Embed de 4096 caractères)
    if reports:
        import time
        logger.info("Transmission des bulletins chartistes avec graphiques vers Discord...")
        for report, chart in reports:
            try:
                if chart:
                    logger.info("Envoi du rapport avec graphique...")
                    DiscordClient()._send_photo_sync_internal(chart, report, category="analyse_technique")
                else:
                    logger.info("Envoi du rapport texte seul (secours)...")
                    DiscordClient().send_sync(report, category="analyse_technique")
                time.sleep(0.5)
            except Exception as exc:
                logger.error("Impossible de transmettre le rapport à Discord : %s", exc)
        logger.info("✅ Transmission Discord terminée.")


if __name__ == "__main__":
    main()
