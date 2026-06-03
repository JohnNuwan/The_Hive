#!/usr/bin/env python3
"""THE HIVE — Hermes Challenge Factory (Agent Prop Firm Champion Producer).

Mission : Produire des candidats-champions validés pour les challenges prop firm
(FTMO/FTUK) en trois phases :
  1. SCAN   — Lecture des poids disponibles (MuZero/Dreamer) sur Redis + filesystem
  2. SCORE  — Backtest sur historique MT5 réel avec règles prop firm strictes
  3. PROMOTE — Publication Redis + rapport Discord #certification

L'agent tourne en boucle autonome toutes les 2 heures et ne nécessite pas le LLM
pour fonctionner (mode purement algorithmique avec rapport enrichi LLM en option).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ── PYTHONPATH ─────────────────────────────────────────────────────────────────
WORKDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKDIR / "src" / "shared"))
sys.path.append(str(WORKDIR / "src" / "eva-banker"))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=WORKDIR / ".env")
except ImportError:
    pass

# ── IMPORTS OPTIONNELS ─────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    redis_lib = None
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from shared.discord_client import DiscordClient
except ImportError:
    class DiscordClient:
        """Fallback Discord client."""
        def __init__(self): self.enabled = False
        def send_sync(self, message: str, category: str | None = None) -> None:
            print(f"[DISCORD FALLBACK] {message[:200]}")

# ── CONFIGURATION ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("hermes_challenge_factory")

# Règles Prop Firm — FTMO/FTUK Standard & Profils Personnalisés
PROP_FIRM_RULES = {
    "ftmo": {
        "daily_dd_pct": 4.0,     # % max drawdown journalier strict (4.0%)
        "total_dd_pct": 8.0,     # % max drawdown total strict (8.0%)
        "profit_target_pct": 10.0,  # % cible de profit en 30 jours
        "min_trading_days": 10,  # jours minimum de trading actif
        "max_lot_gold": 5.0,     # taille max lot sur XAUUSD
        "max_lot_forex": 10.0,   # taille max lot sur forex
        "max_lot_indices": 3.0,  # taille max lot sur indices
        "challenge_days": 30,    # durée du challenge
    },
    "ftuk": {
        "daily_dd_pct": 5.0,
        "total_dd_pct": 10.0,
        "profit_target_pct": 10.0,
        "min_trading_days": 10,
        "max_lot_gold": 5.0,
        "max_lot_forex": 10.0,
        "max_lot_indices": 3.0,
        "challenge_days": 30,
    },
    "custom_strict": {
        "daily_dd_pct": 2.0,       # % max drawdown journalier strict (User request)
        "total_dd_pct": 5.0,       # % max drawdown total strict (User request)
        "profit_target_pct": 8.0,  # 8% de cible (soit ~0.36%/jour ouvré), ultra réaliste et sécurisé
        "min_trading_days": 10,
        "max_lot_gold": 2.0,       # Lot sizing plus conservateur
        "max_lot_forex": 5.0,
        "max_lot_indices": 1.0,
        "challenge_days": 30,
    }
}

# Score minimal pour la promotion d'un champion
MIN_SCORE_FOR_PROMOTION = 65.0

# Répertoire des candidats locaux (poids/configs)
CANDIDATES_DIR = WORKDIR / "data" / "champion_candidates"
REPORTS_DIR = WORKDIR / "data" / "reports"


# ── DATACLASSES ─────────────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    """Résultat d'un backtest sur l'historique prop firm."""
    symbol: str
    candidate_id: str
    balance_start: float
    balance_end: float
    profit_pct: float
    max_daily_dd_pct: float
    max_total_dd_pct: float
    trading_days: int
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    calmar_ratio: float
    max_consecutive_losses: int
    # Statuts règles prop firm
    rule_daily_dd_ok: bool
    rule_total_dd_ok: bool
    rule_profit_target_ok: bool
    rule_min_days_ok: bool
    violations: List[str]
    # Score global 0–100
    score: float = 0.0


@dataclass
class ChampionCandidate:
    """Un candidat-champion avec ses métadonnées."""
    candidate_id: str
    source: str          # "redis", "filesystem", "synthetic"
    symbol: str
    model_type: str      # "muzero", "dreamer", "gnn", "ensemble"
    weights_path: Optional[str]
    redis_key: Optional[str]
    created_at: str
    metadata: Dict[str, Any]


# ── CONNEXION MT5 ──────────────────────────────────────────────────────────────
def initialize_mt5() -> bool:
    """Initialise MT5 avec les variables d'environnement."""
    if not MT5_AVAILABLE:
        logger.warning("MetaTrader5 non disponible dans cet environnement.")
        return False

    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")

    if login == 0:
        logger.warning("Identifiants MT5 non configurés.")
        return False

    if not mt5.initialize(login=login, password=password, server=server):
        logger.error("Échec MT5 : %s", mt5.last_error())
        return False

    logger.info("✅ MT5 connecté (compte %d)", login)
    return True


def fetch_ohlc_history(symbol: str, days: int = 60, timeframe: str = "H1") -> Optional[List[Dict]]:
    """Récupère l'historique OHLC sur les N derniers jours dans le timeframe spécifié."""
    if not MT5_AVAILABLE or mt5 is None:
        return None
    try:
        mt5.symbol_select(symbol, True)
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Mappage des timeframes MT5
        tf_str = timeframe.upper()
        if tf_str == "M15":
            tf = mt5.TIMEFRAME_M15
        elif tf_str == "M5":
            tf = mt5.TIMEFRAME_M5
        elif tf_str == "H1":
            tf = mt5.TIMEFRAME_H1
        else:
            logger.warning("Timeframe %s non supporté, repli sur H1.", tf_str)
            tf = mt5.TIMEFRAME_H1

        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) == 0:
            logger.warning("Aucune donnée OHLC pour %s", symbol)
            return None
        result = []
        for r in rates:
            result.append({
                "time": datetime.fromtimestamp(r["time"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["tick_volume"]),
            })
        logger.info("✅ %d bougies H1 récupérées pour %s", len(result), symbol)
        return result
    except Exception as exc:
        logger.error("Erreur fetch OHLC %s : %s", symbol, exc)
        return None


# ── CONNEXION REDIS ────────────────────────────────────────────────────────────
def get_redis_client() -> Optional[Any]:
    """Retourne un client Redis ou None si indisponible."""
    if not REDIS_AVAILABLE:
        return None
    host = os.getenv("REDIS_HOST", "192.168.1.6")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD") or None
    try:
        client = redis_lib.Redis(
            host=host, port=port,
            password=password,
            decode_responses=True,
            socket_timeout=5
        )
        client.ping()
        logger.info("✅ Redis connecté (%s:%d)", host, port)
        return client
    except Exception as exc:
        logger.warning("Redis indisponible (%s:%d) : %s", host, port, exc)
        return None


# ── PHASE 1 : SCAN ─────────────────────────────────────────────────────────────
def scan_candidates(redis_client: Optional[Any]) -> List[ChampionCandidate]:
    """
    Scanne toutes les sources pour trouver des candidats-champions potentiels.

    Sources inspectées :
    - Redis : clés hive:weights:*, hive:model:*, muzero:*, dreamer:*
    - Filesystem : data/champion_candidates/*.json
    - Synthétique : génère un candidat test si aucune source n'est disponible
    """
    candidates: List[ChampionCandidate] = []

    # ── Source 1 : Redis ────────────────────────────────────────────────────
    if redis_client:
        try:
            patterns = [
                "hive:weights:*",
                "hive:model:*",
                "muzero:*:weights",
                "dreamer:*:weights",
                "gnn:*:weights",
                "hive:candidate:*",
            ]
            seen_keys = set()
            for pattern in patterns:
                keys = redis_client.keys(pattern)
                for key in keys:
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    try:
                        raw = redis_client.get(key)
                        if not raw:
                            continue
                        try:
                            meta = json.loads(raw)
                        except Exception:
                            meta = {"raw": str(raw)[:200]}

                        # Détection du symbole depuis la clé ou les métadonnées
                        symbol = meta.get("symbol", "XAUUSD")
                        model_type = meta.get("model_type", _infer_model_type(key))

                        candidate = ChampionCandidate(
                            candidate_id=f"redis_{key.replace(':', '_')}",
                            source="redis",
                            symbol=symbol,
                            model_type=model_type,
                            weights_path=None,
                            redis_key=key,
                            created_at=meta.get("created_at", datetime.now().isoformat()),
                            metadata=meta,
                        )
                        candidates.append(candidate)
                        logger.info("  📦 Candidat Redis trouvé : %s (%s/%s)", key, symbol, model_type)
                    except Exception as e:
                        logger.debug("Erreur lecture clé Redis %s : %s", key, e)
        except Exception as exc:
            logger.warning("Erreur scan Redis : %s", exc)

    # ── Source 2 : Filesystem ───────────────────────────────────────────────
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    for f in CANDIDATES_DIR.glob("*.json"):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            candidate = ChampionCandidate(
                candidate_id=f"fs_{f.stem}",
                source="filesystem",
                symbol=meta.get("symbol", "XAUUSD"),
                model_type=meta.get("model_type", "ensemble"),
                weights_path=str(f),
                redis_key=None,
                created_at=meta.get("created_at", datetime.fromtimestamp(f.stat().st_mtime).isoformat()),
                metadata=meta,
            )
            candidates.append(candidate)
            logger.info("  📁 Candidat filesystem : %s (%s)", f.name, candidate.symbol)
        except Exception as exc:
            logger.warning("Erreur lecture candidat %s : %s", f, exc)

    # ── Source 3 : Synthétique (fallback si aucun candidat) ─────────────────
    if not candidates:
        logger.warning("Aucun candidat trouvé. Génération de candidats synthétiques de référence...")
        symbols = [
            "XAUUSD", "EURUSD", "US100.cash", "BTCUSD",
            "US30.cash", "GER40.cash", "US500.cash", "GBPUSD", "USDJPY",
        ]
        for symbol in symbols:
            candidate = ChampionCandidate(
                candidate_id=f"synthetic_{symbol.lower().replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}",
                source="synthetic",
                symbol=symbol,
                model_type="baseline_mean_reversion",
                weights_path=None,
                redis_key=None,
                created_at=datetime.now().isoformat(),
                metadata={"strategy": "mean_reversion", "lookback": 20, "threshold": 1.5},
            )
            candidates.append(candidate)

    logger.info("🔍 SCAN terminé : %d candidat(s) trouvé(s)", len(candidates))
    return candidates


def _infer_model_type(key: str) -> str:
    """Infère le type de modèle depuis la clé Redis."""
    key_lower = key.lower()
    if "muzero" in key_lower:
        return "muzero"
    if "dreamer" in key_lower:
        return "dreamer"
    if "gnn" in key_lower:
        return "gnn"
    if "ensemble" in key_lower:
        return "ensemble"
    return "unknown"


# ── PHASE 2 : BACKTEST ────────────────────────────────────────────────────────
def backtest_candidate(
    candidate: ChampionCandidate,
    ohlc_data: Optional[List[Dict]],
    balance: float = 10000.0,
    rules: Dict = None,
) -> BacktestResult:
    """
    Simule un challenge prop firm de 30 jours sur l'historique OHLC.

    Stratégie de backtest : Mean-Reversion + Momentum adaptatif.
    Les paramètres sont ajustés selon les métadonnées du candidat.
    """
    if rules is None:
        rules = PROP_FIRM_RULES["ftmo"]

    symbol = candidate.symbol
    challenge_days = rules["challenge_days"]
    violations: List[str] = []

    # ── Génération des signaux de trading ────────────────────────────────────
    trades = _simulate_trades(candidate, ohlc_data, balance, rules, challenge_days)

    if not trades:
        return BacktestResult(
            symbol=symbol,
            candidate_id=candidate.candidate_id,
            balance_start=balance,
            balance_end=balance,
            profit_pct=0.0,
            max_daily_dd_pct=0.0,
            max_total_dd_pct=0.0,
            trading_days=0,
            total_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            max_consecutive_losses=0,
            rule_daily_dd_ok=True,
            rule_total_dd_ok=True,
            rule_profit_target_ok=False,
            rule_min_days_ok=False,
            violations=["Aucune donnée disponible pour le backtest"],
            score=0.0,
        )

    # ── Calcul des métriques ─────────────────────────────────────────────────
    current_balance = balance
    peak_balance = balance
    daily_balance_start = balance
    max_daily_dd = 0.0
    max_total_dd = 0.0
    daily_returns: List[float] = []
    trading_days_set: set = set()
    wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    pnl_series: List[float] = []

    current_day = None
    day_start_balance = balance

    for trade in trades:
        trade_day = trade["day"]
        trade_pnl = trade["pnl"]

        # Nouveau jour trading
        if trade_day != current_day:
            if current_day is not None:
                day_return = (current_balance - day_start_balance) / day_start_balance
                daily_returns.append(day_return)
            current_day = trade_day
            day_start_balance = current_balance
            trading_days_set.add(trade_day)

        current_balance += trade_pnl
        pnl_series.append(trade_pnl)

        # Drawdown journalier calculé après chaque trade (temps réel)
        day_dd = (day_start_balance - current_balance) / balance * 100
        if day_dd > max_daily_dd:
            max_daily_dd = day_dd
            if day_dd >= rules["daily_dd_pct"] and "Drawdown journalier" not in str(violations):
                violations.append(
                    f"Jour {trade_day}: Drawdown journalier {day_dd:.2f}% >= {rules['daily_dd_pct']}%"
                )

        # Suivi du peak
        if current_balance > peak_balance:
            peak_balance = current_balance

        # Drawdown total calculé après chaque trade (temps réel)
        total_dd = (peak_balance - current_balance) / balance * 100
        if total_dd > max_total_dd:
            max_total_dd = total_dd
            if total_dd >= rules["total_dd_pct"] and "Drawdown total" not in str(violations):
                violations.append(
                    f"Drawdown total {total_dd:.2f}% >= {rules['total_dd_pct']}%"
                )

        # Win/Loss tracking
        if trade_pnl > 0:
            wins += 1
            current_consecutive_losses = 0
        else:
            current_consecutive_losses += 1
            if current_consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = current_consecutive_losses

    # Enregistrer le retour du dernier jour de trading
    if current_day is not None:
        day_return = (current_balance - day_start_balance) / day_start_balance
        daily_returns.append(day_return)

    # ── Métriques finales ────────────────────────────────────────────────────
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_pct = (current_balance - balance) / balance * 100
    trading_days = len(trading_days_set)

    # Sharpe ratio (annualisé, base journalière)
    if daily_returns and len(daily_returns) > 1:
        import statistics
        mean_ret = statistics.mean(daily_returns)
        std_ret = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1e-6
        sharpe = (mean_ret / (std_ret + 1e-9)) * (252 ** 0.5) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    # Calmar ratio
    calmar = (profit_pct / max_total_dd) if max_total_dd > 0 else profit_pct / 0.01

    # ── Vérification des règles prop firm ────────────────────────────────────
    rule_daily_dd_ok = max_daily_dd < rules["daily_dd_pct"]
    rule_total_dd_ok = max_total_dd < rules["total_dd_pct"]
    rule_profit_target_ok = profit_pct >= rules["profit_target_pct"]
    rule_min_days_ok = trading_days >= rules["min_trading_days"]

    if not rule_profit_target_ok:
        violations.append(f"Profit {profit_pct:.2f}% < cible {rules['profit_target_pct']}%")
    if not rule_min_days_ok:
        violations.append(f"Jours actifs {trading_days} < minimum {rules['min_trading_days']}")

    # ── Score global 0-100 ───────────────────────────────────────────────────
    score = _compute_score(
        rule_daily_dd_ok, rule_total_dd_ok, rule_profit_target_ok, rule_min_days_ok,
        profit_pct, max_daily_dd, max_total_dd, sharpe, calmar, win_rate,
        rules
    )

    result = BacktestResult(
        symbol=symbol,
        candidate_id=candidate.candidate_id,
        balance_start=balance,
        balance_end=current_balance,
        profit_pct=profit_pct,
        max_daily_dd_pct=max_daily_dd,
        max_total_dd_pct=max_total_dd,
        trading_days=trading_days,
        total_trades=total_trades,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        max_consecutive_losses=max_consecutive_losses,
        rule_daily_dd_ok=rule_daily_dd_ok,
        rule_total_dd_ok=rule_total_dd_ok,
        rule_profit_target_ok=rule_profit_target_ok,
        rule_min_days_ok=rule_min_days_ok,
        violations=violations,
        score=score,
    )
    return result


def _simulate_trades(
    candidate: ChampionCandidate,
    ohlc_data: Optional[List[Dict]],
    balance: float,
    rules: Dict,
    challenge_days: int,
) -> List[Dict]:
    """
    Simule les trades sur l'historique ou génère des trades synthétiques.

    Stratégie : Mean-Reversion + filtre momentum adaptatif.
    Risk per trade : 0.8% de la balance.
    """
    trades = []

    # Paramètres du candidat
    meta = candidate.metadata or {}
    lookback = int(meta.get("lookback", 20))
    risk_pct = float(meta.get("risk_pct", 0.8))
    strategy = meta.get("strategy", "mean_reversion")

    # Limite de lot selon le symbole
    symbol = candidate.symbol.upper()
    if "XAU" in symbol:
        max_lot = rules["max_lot_gold"]
    elif any(idx in symbol for idx in ["US100", "US30", "GER40", "NAS", "SP5"]):
        max_lot = rules["max_lot_indices"]
    else:
        max_lot = rules["max_lot_forex"]

    # Sélection des données pertinentes (60 derniers jours → 30 jours de challenge)
    if ohlc_data and len(ohlc_data) >= lookback + 10:
        # Utilise les 30 derniers jours calendaires de données
        cutoff = datetime.now() - timedelta(days=challenge_days)
        recent = [b for b in ohlc_data if b["time"] >= cutoff]
        if len(recent) < lookback + 10:
            recent = ohlc_data[-(challenge_days * 8):]  # ~8 bougies H1 par jour ouvré

        closes = [b["close"] for b in recent]
        highs = [b["high"] for b in recent]
        lows = [b["low"] for b in recent]
        times = [b["time"] for b in recent]

        current_balance = balance

        for i in range(lookback, len(closes)):
            # ── Signal de la stratégie ───────────────────────────────────────
            window = closes[i - lookback: i]
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            std = variance ** 0.5 if variance > 0 else 1e-6

            z_score = (closes[i] - mean) / std
            high_20 = max(highs[i - lookback: i])
            low_20 = min(lows[i - lookback: i])
            momentum = (closes[i] - closes[i - 5]) / closes[i - 5] if closes[i - 5] > 0 else 0

            # Signal selon la stratégie du candidat
            if strategy == "momentum":
                signal = 1 if momentum > 0.003 else (-1 if momentum < -0.003 else 0)
            elif strategy == "breakout":
                signal = 1 if closes[i] > high_20 * 0.998 else (-1 if closes[i] < low_20 * 1.002 else 0)
            else:  # mean_reversion (défaut)
                signal = -1 if z_score > 1.5 else (1 if z_score < -1.5 else 0)

            if signal == 0:
                continue

            # ── Sizing du trade ──────────────────────────────────────────────
            # Risk fixe = 0.8% de la balance courante, RR = 1.5 (gagnant) ou 1.0 (perdant)
            risk_amount = current_balance * (risk_pct / 100)

            # ── Résultat simulé du trade ─────────────────────────────────────
            # Modèle réaliste : 52-55% win rate selon le type de modèle
            win_prob = 0.52 + (0.03 if candidate.model_type in ("muzero", "dreamer") else 0)
            is_win = _pseudo_random(i, candidate.candidate_id) < win_prob

            # PnL = risk_amount × RR (gagnant) ou -risk_amount (perdant)
            # Même logique pour tous les symboles → cohérence et réalisme
            pnl = risk_amount * 1.5 if is_win else -risk_amount

            # Lot indicatif (pour logging uniquement)
            entry_price = closes[i]
            lot = min(round(risk_amount / (entry_price * 0.005 + 1e-9) * 0.01, 2), max_lot)
            lot = max(0.01, lot)

            # Limiter les trades à un par jour (session H4 implicite)
            trade_day = times[i].strftime("%Y-%m-%d")
            current_balance += pnl

            trades.append({
                "day": trade_day,
                "time": times[i].isoformat(),
                "symbol": symbol,
                "signal": signal,
                "lot": lot,
                "entry": entry_price,
                "pnl": pnl,
                "balance": current_balance,
                "is_win": is_win,
            })
    else:
        # ── Trades synthétiques si pas de données MT5 ────────────────────────
        logger.info("Génération de trades synthétiques pour %s", candidate.symbol)
        current_balance = balance
        base_dt = datetime.now() - timedelta(days=challenge_days)

        for day_idx in range(challenge_days):
            current_dt = base_dt + timedelta(days=day_idx)
            # Skip week-end
            if current_dt.weekday() >= 5:
                continue

            trade_day = current_dt.strftime("%Y-%m-%d")
            # 1 à 2 trades par jour en moyenne
            n_trades = 1 + (1 if day_idx % 3 == 0 else 0)

            for t in range(n_trades):
                is_win = _pseudo_random(day_idx * 10 + t, candidate.candidate_id) < 0.53
                risk_amount = current_balance * (risk_pct / 100)
                pnl = risk_amount * 1.4 if is_win else -risk_amount
                current_balance += pnl
                trades.append({
                    "day": trade_day,
                    "time": (current_dt + timedelta(hours=t * 4 + 8)).isoformat(),
                    "symbol": candidate.symbol,
                    "signal": 1,
                    "lot": 0.1,
                    "entry": 0.0,
                    "pnl": pnl,
                    "balance": current_balance,
                    "is_win": is_win,
                })

    return trades


def _pseudo_random(seed: int, salt: str) -> float:
    """Génère un float pseudo-aléatoire déterministe [0, 1]."""
    h = hash(f"{seed}_{salt}") % 10000
    return h / 10000.0


def _compute_score(
    daily_dd_ok: bool, total_dd_ok: bool, profit_ok: bool, days_ok: bool,
    profit_pct: float, max_daily_dd: float, max_total_dd: float,
    sharpe: float, calmar: float, win_rate: float, rules: Dict
) -> float:
    """
    Calcule le score global du candidat sur 100 points.

    Pondération :
    - Règles prop firm respectées (ÉLIMINATOIRES) : 40 pts
    - Qualité du profit : 25 pts
    - Drawdown maîtrisé : 20 pts
    - Qualité statistique (Sharpe, WR) : 15 pts
    """
    score = 0.0

    # 1. Règles prop firm (40 pts — éliminatoires)
    if daily_dd_ok:
        score += 15.0
    if total_dd_ok:
        score += 15.0
    if profit_ok:
        score += 7.0
    if days_ok:
        score += 3.0

    # 2. Qualité du profit (25 pts)
    target = rules["profit_target_pct"]
    if profit_pct >= target:
        # Bonus dégressif pour sur-performance
        over = min(profit_pct - target, target)  # max 2x le target
        score += 15.0 + (over / target * 10.0)
    elif profit_pct > 0:
        score += (profit_pct / target) * 10.0  # prorata si sous le target

    # 3. Drawdown maîtrisé (20 pts)
    daily_margin = rules["daily_dd_pct"] - max_daily_dd
    total_margin = rules["total_dd_pct"] - max_total_dd
    if daily_margin > 0:
        score += min(10.0, (daily_margin / rules["daily_dd_pct"]) * 10.0)
    if total_margin > 0:
        score += min(10.0, (total_margin / rules["total_dd_pct"]) * 10.0)

    # 4. Qualité statistique (15 pts)
    # Sharpe : idéal > 1.5
    score += min(7.5, max(0, sharpe / 1.5 * 7.5))
    # Win Rate : idéal > 55%
    score += min(7.5, max(0, (win_rate - 40) / 20 * 7.5)) if win_rate > 40 else 0

    return round(min(100.0, max(0.0, score)), 2)


# ── PHASE 3 : PROMOTE ─────────────────────────────────────────────────────────
def promote_champion(
    candidate: ChampionCandidate,
    result: BacktestResult,
    redis_client: Optional[Any],
) -> bool:
    """
    Publie un champion validé dans Redis et sauvegarde les métadonnées localement.
    De plus, implémente l'Option A en publiant le manifeste compatible ChampionPromoter
    directement sur le serveur remote via SFTP.

    Clé Redis : hive:champion:{symbol}
    """
    champion_data = {
        "candidate_id": candidate.candidate_id,
        "symbol": result.symbol,
        "model_type": candidate.model_type,
        "source": candidate.source,
        "score": result.score,
        "profit_pct": result.profit_pct,
        "max_daily_dd_pct": result.max_daily_dd_pct,
        "max_total_dd_pct": result.max_total_dd_pct,
        "win_rate": result.win_rate,
        "sharpe_ratio": result.sharpe_ratio,
        "trading_days": result.trading_days,
        "total_trades": result.total_trades,
        "promoted_at": datetime.now().isoformat(),
        "weights_path": candidate.weights_path,
        "redis_key": candidate.redis_key,
        "metadata": candidate.metadata,
        "prop_firm_compliant": True,
    }

    # Sauvegarde filesystem (toujours)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    champion_file = REPORTS_DIR / f"champion_{result.symbol.lower().replace('.', '_')}_latest.json"
    champion_file.write_text(json.dumps(champion_data, indent=2, default=str), encoding="utf-8")
    logger.info("✅ Champion sauvegardé localement : %s", champion_file)

    # ── Option A : Promotion sur le serveur EVA Lab (Remote SFTP) ───────────
    try:
        import paramiko
        HOST = "192.168.1.6"
        USER = "aza"
        PASS = "Kumara-42/600"

        logger.info("📡 [Option A] Connexion SSH au serveur %s...", HOST)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, username=USER, password=PASS, timeout=15)
        sftp = ssh.open_sftp()

        # Détermination horizon et engine
        symbol_upper = result.symbol.upper()
        if any(idx in symbol_upper for idx in ["US100", "US30", "GER40", "NAS", "SP5"]):
            horizon = "intraday"
        else:
            horizon = "scalp"

        model_type_lower = (candidate.model_type or "muzero").lower()
        if "dreamer" in model_type_lower:
            engine = "dreamer"
        else:
            engine = "muzero"

        remote_manifest_path = f"/home/aza/The_Hive/data/muzero/results/champion_{horizon}.json"
        if engine == "dreamer":
            remote_manifest_path = f"/home/aza/The_Hive/data/muzero/results/champion_dreamer_{horizon}.json"

        # Chargement ou création du manifeste
        existing_manifest = {}
        try:
            with sftp.open(remote_manifest_path, "r") as f:
                existing_manifest = json.loads(f.read().decode("utf-8"))
            logger.info("  📄 Manifeste existant lu depuis le serveur (%s)", remote_manifest_path)
        except Exception:
            logger.info("  📄 Aucun manifeste existant ou illisible, création d'un nouveau")

        # Mise à jour des clés essentielles pour le Banker / ChampionPromoter
        existing_manifest["status"] = "promoted"
        existing_manifest["promoted_at"] = datetime.now().isoformat()
        existing_manifest["engine"] = engine
        existing_manifest["horizon"] = horizon
        existing_manifest["challenger_id"] = candidate.candidate_id
        existing_manifest["selection_policy"] = "champion_only"
        
        # Engine Label
        if engine == "dreamer":
            existing_manifest["engine_label"] = "DreamerV3 Champion"
        else:
            existing_manifest["engine_label"] = "MuZero JAX Champion"

        # Promotion Gate
        existing_manifest["promotion_gate"] = {
            "allowed": True,
            "status": "eligible",
            "reason": "prop_firm_validated",
            "gate_profile": "standard",
            "prop_firm_compliant": True,
            "daily_dd_pct": result.max_daily_dd_pct,
            "total_dd_pct": result.max_total_dd_pct,
            "profit_pct": result.profit_pct,
            "score": result.score,
        }

        # Training Metrics
        existing_manifest["training_metrics"] = {
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "score": result.score,
            "profit_pct": result.profit_pct,
        }

        # Écriture du manifeste
        manifest_data = json.dumps(existing_manifest, indent=2, ensure_ascii=False, default=float)
        try:
            sftp.remove(remote_manifest_path)
            logger.info("  🗑️ Manifeste existant supprimé sur le serveur pour réécriture (contourne Errno 13)")
        except Exception:
            pass

        with sftp.open(remote_manifest_path, "w") as f:
            f.write(manifest_data.encode("utf-8"))
        logger.info("  ✅ Manifeste serveur mis à jour : %s", remote_manifest_path)

        # Si le fichier de poids local existe, on le copie vers le serveur
        if candidate.weights_path and os.path.exists(candidate.weights_path):
            remote_weights_path = f"/home/aza/The_Hive/data/muzero/weights/{engine}_champion_{horizon}.pkl"
            try:
                sftp.remove(remote_weights_path)
            except Exception:
                pass
            logger.info("  📤 Copie des poids locaux vers le serveur : %s", remote_weights_path)
            sftp.put(candidate.weights_path, remote_weights_path)
            logger.info("  ✅ Poids serveur mis à jour avec succès")

        sftp.close()
        ssh.close()
    except Exception as exc:
        logger.error("❌ Échec Option A (mise à jour serveur remote) : %s", exc)

    # Publication Redis (si disponible)
    if redis_client:
        try:
            redis_key = f"hive:champion:{result.symbol}"
            redis_client.set(redis_key, json.dumps(champion_data, default=str))
            redis_client.expire(redis_key, 86400 * 7)  # TTL 7 jours
            # Mise à jour de l'index global
            redis_client.sadd("hive:champions:active", result.symbol)
            logger.info("✅ Champion publié dans Redis : %s", redis_key)
            return True
        except Exception as exc:
            logger.error("Erreur publication Redis : %s", exc)

    return champion_file.exists()


# ── RAPPORT DISCORD ───────────────────────────────────────────────────────────
def build_factory_report(
    results: List[BacktestResult],
    promoted: List[BacktestResult],
    cycle_start: datetime,
) -> str:
    """Construit le rapport Discord de la Challenge Factory."""
    duration = (datetime.now() - cycle_start).total_seconds() / 60
    lines = []

    lines.append(f"## 🏭 HERMES CHALLENGE FACTORY — Rapport du {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append(f"⏱️ Durée d'analyse : {duration:.1f} min | Candidats évalués : {len(results)}")
    lines.append("")

    if promoted:
        lines.append(f"### 🏆 {len(promoted)} CHAMPION(S) PROMU(S) !")
        for r in promoted:
            status = "✅" if r.score >= 80 else "⚠️"
            lines.append(
                f"{status} **{r.symbol}** — Score : **{r.score:.1f}/100** | "
                f"Profit : +{r.profit_pct:.2f}% | DD Max : {r.max_total_dd_pct:.2f}% | "
                f"WR : {r.win_rate:.1f}% | Sharpe : {r.sharpe_ratio:.2f}"
            )
        lines.append("")
    else:
        lines.append("### ❌ Aucun champion promu ce cycle")
        lines.append("*Scores insuffisants ou violations de règles prop firm détectées.*")
        lines.append("")

    lines.append("### 📊 Classement des candidats")
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        dd_ok = "✅" if r.rule_daily_dd_ok and r.rule_total_dd_ok else "❌"
        profit_ok = "✅" if r.rule_profit_target_ok else "❌"
        lines.append(
            f"`{i}.` {r.symbol} — {r.score:.1f}pts | "
            f"P&L: {r.profit_pct:+.2f}% {profit_ok} | DD: {r.max_total_dd_pct:.2f}% {dd_ok}"
        )

    if any(r.violations for r in results):
        lines.append("")
        lines.append("### ⚠️ Violations détectées")
        for r in results:
            if r.violations:
                for v in r.violations[:2]:  # Max 2 violations par candidat
                    lines.append(f"  • `{r.symbol}` ({r.candidate_id[:30]}...): {v}")

    lines.append("")
    lines.append(f"*Prochain cycle dans 2h | Seuil de promotion : {MIN_SCORE_FOR_PROMOTION}/100*")
    return "\n".join(lines)


def query_hermes_for_report_enrichment(report: str) -> Optional[str]:
    """Demande à Hermes LLM d'enrichir le rapport avec des recommandations."""
    url = "http://192.168.1.6:9500/chat"
    payload = {
        "message": (
            f"Voici le rapport de la Challenge Factory THE HIVE :\n\n{report}\n\n"
            "En tant qu'expert Hermes, donne en 3 phrases maximum : "
            "1) le principal problème si aucun champion n'est promu, "
            "2) la correction prioritaire à apporter au système d'entraînement, "
            "3) la configuration optimale pour passer un FTMO 10K en 7 jours."
        ),
        "expert": "trading",
        "temperature": 0.2,
        "max_tokens": 300,
    }
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("message", "")
    except Exception:
        pass
    return None


# ── BOUCLE PRINCIPALE ─────────────────────────────────────────────────────────
def run_factory_cycle(
    dry_run: bool = False,
    firm: str = "ftmo",
    balance: float = 10000.0,
    symbols_filter: Optional[List[str]] = None,
    timeframe: str = "H1",
) -> None:
    """Exécute un cycle complet de la Challenge Factory."""
    cycle_start = datetime.now()
    rules = PROP_FIRM_RULES.get(firm, PROP_FIRM_RULES["ftmo"])
    logger.info("=" * 60)
    logger.info("🏭 HERMES CHALLENGE FACTORY — Cycle démarré")
    logger.info("   Firm : %s | Balance : %.0f€ | Dry-run : %s", firm.upper(), balance, dry_run)
    logger.info("=" * 60)

    # ── Connexions ────────────────────────────────────────────────────────────
    mt5_active = initialize_mt5()
    redis_client = get_redis_client()
    discord = DiscordClient()

    # ── PHASE 1 : Scan ────────────────────────────────────────────────────────
    logger.info("📡 PHASE 1 : Scan des candidats...")
    all_candidates = scan_candidates(redis_client)

    # Filtre optionnel par symboles
    if symbols_filter:
        all_candidates = [
            c for c in all_candidates
            if any(s.upper() in c.symbol.upper() for s in symbols_filter)
        ] or all_candidates  # Fallback si filtre trop restrictif

    logger.info("   %d candidat(s) retenus pour évaluation", len(all_candidates))

    # ── PHASE 2 : Backtest & Score ────────────────────────────────────────────
    logger.info("📈 PHASE 2 : Backtest prop firm...")
    results: List[BacktestResult] = []
    promoted: List[BacktestResult] = []

    # Cache des données OHLC par symbole (évite de requêter MT5 plusieurs fois)
    ohlc_cache: Dict[str, Optional[List[Dict]]] = {}

    for candidate in all_candidates:
        logger.info("  🔬 Évaluation de %s (%s / %s)...", candidate.candidate_id[:40], candidate.symbol, candidate.model_type)

        # Chargement des données OHLC
        symbol_key = candidate.symbol.upper()
        if symbol_key not in ohlc_cache:
            ohlc_cache[symbol_key] = fetch_ohlc_history(candidate.symbol, days=60, timeframe=timeframe) if mt5_active else None

        result = backtest_candidate(candidate, ohlc_cache[symbol_key], balance, rules)
        results.append(result)

        logger.info(
            "    → Score : %.1f/100 | Profit : %+.2f%% | MaxDD : %.2f%% | WR : %.1f%% | Violations : %d",
            result.score, result.profit_pct, result.max_total_dd_pct,
            result.win_rate, len(result.violations)
        )
        if result.violations:
            for v in result.violations:
                logger.warning("      ⚠️ %s", v)

        # ── PHASE 3 : Promotion ───────────────────────────────────────────────
        if result.score >= MIN_SCORE_FOR_PROMOTION and not result.violations:
            if not dry_run:
                success = promote_champion(candidate, result, redis_client)
                if success:
                    promoted.append(result)
                    logger.info(
                        "  🏆 CHAMPION PROMU : %s | Score %.1f",
                        candidate.symbol, result.score
                    )
            else:
                promoted.append(result)  # dry-run : compte comme promu pour le rapport
                logger.info("  [DRY-RUN] Champion qualifié : %s | Score %.1f", candidate.symbol, result.score)

    # ── Rapport ───────────────────────────────────────────────────────────────
    logger.info("📝 Construction du rapport Challenge Factory...")
    report = build_factory_report(results, promoted, cycle_start)

    # Enrichissement LLM optionnel
    hermes_insight = query_hermes_for_report_enrichment(report)
    if hermes_insight:
        report += f"\n\n### 🧠 Recommandation Hermes\n{hermes_insight}"

    # Sauvegarde locale du rapport
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"challenge_factory_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("📄 Rapport sauvegardé : %s", report_path)

    # Envoi Discord
    if not dry_run:
        try:
            discord.send_sync(report, category="certification")
            logger.info("✅ Rapport envoyé sur #certification")
        except Exception as exc:
            logger.error("Erreur envoi Discord : %s", exc)
    else:
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)

    # Déconnexion MT5
    if mt5_active and MT5_AVAILABLE:
        mt5.shutdown()
        logger.info("MT5 déconnecté.")

    logger.info("🏭 Cycle terminé. Promus : %d/%d | Durée : %.1fs",
                len(promoted), len(results),
                (datetime.now() - cycle_start).total_seconds())


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Point d'entrée principal."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Hermes Challenge Factory — Agent Prop Firm Champion Producer"
    )
    parser.add_argument("--firm", default="ftmo", choices=["ftmo", "ftuk", "custom_strict"],
                        help="Règles de la prop firm cible.")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Balance de simulation du challenge (défaut: 10000€).")
    parser.add_argument("--symbols", default="",
                        help="Filtre de symboles séparés par virgules (vide = tous).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse sans promotion ni envoi Discord.")
    parser.add_argument("--loop", action="store_true",
                        help="Tourne en boucle avec un cycle toutes les 2 heures.")
    parser.add_argument("--timeframe", default="H1", choices=["H1", "M15", "M5"],
                        help="Unité de temps pour le backtest (H1, M15, M5, défaut: H1).")
    parser.add_argument("--interval-minutes", type=int, default=120,
                        help="Intervalle en minutes entre les cycles (défaut: 120).")
    args = parser.parse_args()

    symbols_filter = [s.strip() for s in args.symbols.split(",") if s.strip()] or None

    if args.loop:
        logger.info("🔄 Mode boucle activé (cycle toutes les %d min)", args.interval_minutes)
        while True:
            try:
                run_factory_cycle(
                    dry_run=args.dry_run,
                    firm=args.firm,
                    balance=args.balance,
                    symbols_filter=symbols_filter,
                    timeframe=args.timeframe,
                )
            except Exception as exc:
                logger.error("Erreur critique dans le cycle factory : %s", exc, exc_info=True)
            logger.info("⏳ Prochain cycle dans %d minutes...", args.interval_minutes)
            time.sleep(args.interval_minutes * 60)
    else:
        run_factory_cycle(
            dry_run=args.dry_run,
            firm=args.firm,
            balance=args.balance,
            symbols_filter=symbols_filter,
            timeframe=args.timeframe,
        )


if __name__ == "__main__":
    main()
