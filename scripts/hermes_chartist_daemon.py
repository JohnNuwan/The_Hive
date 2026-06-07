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

    import os
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    llm_model = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
    
    if openrouter_api_key:
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": "Vous etes Hermes, le Maitre Chartist de THE HIVE. Redigez des bulletins d'analyse technique rigoureux et clairs en Francais."},
                {"role": "user", "content": prompt.strip()}
            ],
            "temperature": 0.3
        }
        
        try:
            logger.info("Interrogation OpenRouter (%s) pour l'analyse chartiste de %s...", llm_model, symbol)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return content
            else:
                logger.warning("Erreur OpenRouter: %s", response.text)
        except Exception as exc:
            logger.error("Echec OpenRouter : %s", exc)
    else:
        logger.warning("Pas de OPENROUTER_API_KEY trouvee. Impossible de lancer l'analyse I.A.")

    # Fallback ultime purement algorithmique
    return f"""### 📊 ANALYSE TECHNIQUE AUTOMATIQUE (FALLBACK) — {symbol} [{timeframe}]
*   **Prix Actuel** : {coords['close']:.2f}
*   **Structure** : Résistance à {coords['support']:.2f} | Support à {coords['resistance']:.2f}.
*   **Fibonacci 61.8%** : {coords['fibs']['fib_618']:.2f}.
*   **Dynamique** : RSI à {coords['rsi']:.2f} (Régime : {'Suracheté' if coords['rsi'] > 70 else 'Surtendu' if coords['rsi'] < 30 else 'Neutre'}). ADX à {coords['adx']:.2f} (Tendance {'forte' if coords['adx'] > 25 else 'faible'}).
*   *Note : L'expert Hermes n'a pas pu être joint pour une synthèse rédigée.*"""


def clean_pdf_text(text: str) -> str:
    """Nettoie le texte des caractères non-compatibles avec la police standard de FPDF."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("œ", "oe").replace("Œ", "Oe")
    text = text.replace("€", "EUR").replace("…", "...")
    
    # Table de correspondance minimale pour les emojis courants dans les briefings
    emoji_map = {
        "📊": "[Anal.]", "🚨": "[Alert]", "⚡": "[Signal]", "🕒": "[Time]",
        "📈": "[Bull]", "📉": "[Bear]", "📋": "[Audit]", "🌿": "[EVA]",
        "✅": "[OK]", "❌": "[Err]", "ℹ️": "[Info]", "⭐": "[*]", "✨": "[*]"
    }
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    
    # Encodage/décodage latin-1 pour éliminer tout autre caractère non supporté par la police Helvetica de base
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(reports: list, timeframe: str) -> bytes:
    """Génère un document PDF unifié et professionnel contenant tous les briefings et graphiques.

    Args:
        reports (list): Liste de tuples (symbol, briefing, chart_bytes, coords).
        timeframe (str): Horizon de temps des briefings.

    Returns:
        bytes: Données binaires du PDF.
    """
    from fpdf import FPDF
    import tempfile
    import os

    class ChartistPDF(FPDF):
        def header(self):
            # En-tête de page premium
            self.set_fill_color(21, 21, 28) # #15151c (sombre premium)
            self.rect(0, 0, 210, 25, "F")
            
            self.set_text_color(255, 255, 255)
            self.set_font("helvetica", "B", 12)
            self.set_xy(10, 5)
            self.cell(0, 8, "THE HIVE - RAPPORT TECHNIQUE HERMES", new_x="LMARGIN", new_y="NEXT", align="L")
            
            self.set_font("helvetica", "I", 9)
            self.set_text_color(164, 176, 190)
            self.set_x(10)
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
            self.cell(0, 5, f"Rapport Chartist Multi-Actifs - TF: {timeframe.upper()} - Edite le {now_str}", new_x="LMARGIN", new_y="NEXT", align="L")
            
            # Ligne de séparation bleue néon (#54a0ff)
            self.set_draw_color(84, 160, 255)
            self.set_line_width(0.8)
            self.line(0, 25, 210, 25)
            self.ln(15)

        def footer(self):
            # Pied de page discret
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.set_text_color(127, 140, 141)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = ChartistPDF()
    pdf.alias_nb_pages()
    
    # Page de garde / Cover Page
    pdf.add_page()
    
    # Titre de la page de garde
    pdf.ln(20)
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(47, 53, 66) # Gris ardoise
    pdf.cell(0, 15, "BULLETIN CHARTISTE", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 15, "HERMES COGNITIVE", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.ln(10)
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 10, f"Analyse multi-devises et indices en horizon {timeframe.upper()}", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.ln(20)
    # Liste des actifs inclus
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(84, 160, 255) # Bleu néon
    pdf.cell(0, 10, "Actifs analyses dans ce rapport :", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(47, 53, 66)
    for symbol, _, _, coords in reports:
        pdf.cell(0, 8, f"- {symbol} (Dernier cours : {coords['close']:.2f})", new_x="LMARGIN", new_y="NEXT", align="C")
        
    pdf.ln(25)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(255, 71, 87) # Rouge alerte
    disclaimer_text = (
        "Disclaimer : Ce document est genere automatiquement par l'intelligence artificielle Hermes "
        "a titre purement informatif. Il ne constitue pas un conseil en investissement ni une incitation "
        "a negocier sur les marches financiers."
    )
    pdf.multi_cell(0, 5, disclaimer_text, align="C")

    # Une page par symbole
    for symbol, report_text, chart, coords in reports:
        pdf.add_page()
        
        # Titre du symbole
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(84, 160, 255) # Bleu
        pdf.cell(0, 10, f"Rapport Technique : {symbol}", new_x="LMARGIN", new_y="NEXT", align="L")
        pdf.ln(2)
        
        # Tableau récapitulatif des indicateurs clés
        pdf.set_font("helvetica", "B", 9)
        pdf.set_fill_color(241, 242, 246)
        pdf.set_text_color(47, 53, 66)
        
        headers = ["Metrique", "Valeur", "Metrique", "Valeur"]
        widths = [45, 45, 45, 45]
        
        # Ligne d'en-tête du tableau
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, h, border=1, align="C", fill=True)
        pdf.ln()
        
        # Lignes de données
        pdf.set_font("helvetica", "", 9)
        data_rows = [
            ("Cours de Cloture", f"{coords['close']:.4f}", "Momentum RSI (14)", f"{coords['rsi']:.2f}"),
            ("Resistance Pivot", f"{coords['support']:.4f}", "Force ADX", f"{coords['adx']:.2f}"),
            ("Support Pivot", f"{coords['resistance']:.4f}", "Volatilite ATR", f"{coords['atr']:.4f} ({coords['atr_pct']:.2f}%)"),
            ("Fibonacci 61.8%", f"{coords['fibs']['fib_618']:.4f}", "Pente Tendance (20p)", f"{coords['trend_slope']:.6f}"),
        ]
        
        for row in data_rows:
            pdf.cell(widths[0], 6, row[0], border=1, align="L")
            pdf.cell(widths[1], 6, row[1], border=1, align="C")
            pdf.cell(widths[2], 6, row[2], border=1, align="L")
            pdf.cell(widths[3], 6, row[3], border=1, align="C")
            pdf.ln()
            
        pdf.ln(5)
        
        # Corps du Briefing
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(47, 53, 66)
        pdf.cell(0, 8, "Synthese Cognitive d'Hermes :", new_x="LMARGIN", new_y="NEXT", align="L")
        
        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(47, 53, 66)
        
        # Nettoyage et découpage du briefing
        cleaned_briefing = clean_pdf_text(report_text)
        # Supprime le titre s'il est répété
        lines = cleaned_briefing.split("\n")
        filtered_lines = []
        for line in lines:
            line_strip = line.strip()
            # Enlever les balises markdown de titre pour les ré-afficher proprement ou les ignorer
            if line_strip.startswith("###") or line_strip.startswith("##"):
                clean_line = line_strip.lstrip("#").strip()
                filtered_lines.append(f"\n{clean_line.upper()} :")
            elif line_strip.startswith("*") or line_strip.startswith("-"):
                clean_line = line_strip.lstrip("*-").strip()
                filtered_lines.append(f"  - {clean_line}")
            else:
                filtered_lines.append(line_strip)
                
        reconstructed_text = "\n".join(filtered_lines).strip()
        
        # Remplacement des ** pour que ça ne fasse pas tâche dans le PDF
        reconstructed_text = reconstructed_text.replace("**", "")
        pdf.multi_cell(0, 4.5, reconstructed_text)
        
        # Insérer l'image du graphique technique
        if chart:
            pdf.ln(6)
            # Sauvegarder dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(chart)
                temp_path = f.name
            
            # Positionner l'image de façon centrée
            current_y = pdf.get_y()
            if current_y > 180:
                # Si pas assez de place pour l'image, on crée une nouvelle page (sans ré-en-tête de symbol)
                pdf.add_page()
                current_y = pdf.get_y()
                
            pdf.image(temp_path, x=25, y=current_y, w=160, h=80)
            os.unlink(temp_path)
            
    return pdf.output()


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
            reports.append((symbol, briefing, chart_bytes, coords))
            
            # --- START LLM SWARM MEMORY LOGIC ---
            try:
                import redis
                # Pousse le rapport dans la memoire partagee (Redis) pour que l'Agent Trader le lise.
                r = redis.Redis(host='192.168.1.6', port=6379, db=0, password='devpassword')
                redis_key = f"hive:agent:chartist:report:{symbol}:{args.timeframe.upper()}"
                r.set(redis_key, symbol_report)
                logger.info("Rapport publiÃ© dans la mÃ©moire Redis : %s", redis_key)
            except Exception as e:
                logger.error("Erreur de sauvegarde Swarm Memoire: %s", e)
            # --- END LLM SWARM MEMORY LOGIC ---

            # --- START HIPPORAG 2 GRAPH MEMORY INGESTION ---
            if not args.dry_run:
                try:
                    import asyncio
                    from shared.memory_bridge import get_memory_bridge
                    
                    logger.info("Ingestion du rapport dans la memoire de graphe HippoRAG 2...")
                    metadata = {
                        "source": "hermes_chartist",
                        "symbol": symbol,
                        "timeframe": args.timeframe.upper(),
                        "close": float(coords["close"]),
                        "rsi": float(coords["rsi"]),
                        "adx": float(coords["adx"]),
                        "support": float(coords["support"]),
                        "resistance": float(coords["resistance"])
                    }
                    
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    
                    if loop and loop.is_running():
                        loop.create_task(
                            get_memory_bridge().add(
                                content=symbol_report,
                                user_id="hermes_chartist",
                                metadata=metadata
                            )
                        )
                    else:
                        asyncio.run(
                            get_memory_bridge().add(
                                content=symbol_report,
                                user_id="hermes_chartist",
                                metadata=metadata
                            )
                        )
                    logger.info("✅ Ingestion HippoRAG 2 terminee pour %s.", symbol)
                except Exception as e:
                    logger.error("Erreur d'ingestion HippoRAG 2: %s", e)
            # --- END HIPPORAG 2 GRAPH MEMORY INGESTION ---
            
            print(f"\n--- RAPPORT {symbol} ---\n{symbol_report}\n")
        except Exception as exc:
            logger.error("Erreur de traitement pour le symbole %s : %s", symbol, exc, exc_info=True)

    if mt5_active:
        mt5.shutdown()
        logger.info("MT5 déconnecté.")

    if args.dry_run:
        logger.info("Exécution en mode simulation terminée. Aucun message envoyé.")
        return

    # Compilation et envoi du rapport PDF global unifié
    if reports:
        try:
            logger.info("Compilation du rapport PDF global...")
            pdf_bytes = generate_pdf_report(reports, args.timeframe)
            
            # Nom de fichier daté et propre
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"Rapport_Chartist_{args.timeframe.upper()}_{date_str}.pdf"
            
            logger.info("Envoi du PDF a Discord : %s", filename)
            symbols_str = ", ".join([r[0] for r in reports])
            discord_msg = (
                f"📊 **Bulletin Chartist Hermes [{args.timeframe.upper()}]**\n"
                f"Consultez l'analyse technique detaillee pour les actifs suivants : **{symbols_str}**.\n"
                f"*Genere a {datetime.now().strftime('%H:%M:%S')}*"
            )
            DiscordClient().send_file_sync(
                file_bytes=pdf_bytes,
                filename=filename,
                content=discord_msg,
                category="analyse_technique"
            )
            logger.info("✅ Transmission PDF Discord terminee.")
        except Exception as exc:
            logger.error("Impossible de compiler ou de transmettre le rapport PDF a Discord : %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
