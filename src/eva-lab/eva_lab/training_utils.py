"""Utilitaires communs pour les entrainements EVA Lab."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.indicators import IndicatorFactory

logger = logging.getLogger(__name__)

GNN_ASSET_DIM = 20
GNN_TEMPORAL_DIM = 64
GNN_HIDDEN_DIM = 128
GNN_NUM_CLASSES = 3

MTF_HORIZONS: dict[str, dict[str, Any]] = {
    "M5": {
        "minutes": 5,
        "count": 2000,
        "seq_len": 20,
        "future": 12,
        "strategy": "scalp",
    },
    "H1": {
        "minutes": 60,
        "count": 2000,
        "seq_len": 20,
        "future": 24,
        "strategy": "intraday",
    },
    "D1": {
        "minutes": 1440,
        "count": 1000,
        "seq_len": 15,
        "future": 7,
        "strategy": "swing",
    },
}

HORIZON_TO_TIMEFRAME = {
    "scalp": "M5",
    "intraday": "H1",
    "swing": "D1",
}


def get_history_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Retourne le dossier des historiques utilise pour l'entrainement.

    Args:
        data_dir (str | os.PathLike[str] | None): Dossier explicite a utiliser.

    Returns:
        Path: Chemin du dossier d'historique.
    """
    explicit = data_dir or os.getenv("TRAINING_DATA_DIR") or Path("data") / "history"
    return Path(explicit)


def parse_history_filename(path: Path) -> tuple[str, str] | None:
    """Extrait le symbole et le timeframe depuis un nom de fichier historique.

    Args:
        path (Path): Fichier CSV a parser.

    Returns:
        tuple[str, str] | None: Couple ``(symbol, timeframe)`` si le format est valide.
    """
    stem = path.stem
    if "_" not in stem:
        return None
    symbol, timeframe = stem.rsplit("_", 1)
    if not symbol or not timeframe:
        return None
    return symbol, timeframe.upper()


def discover_history_inventory(
    data_dir: str | os.PathLike[str] | None = None,
) -> dict[str, set[str]]:
    """Construit l'inventaire des historiques disponibles.

    Args:
        data_dir (str | os.PathLike[str] | None): Dossier des CSV.

    Returns:
        dict[str, set[str]]: Mapping ``symbole -> timeframes disponibles``.
    """
    history_dir = get_history_dir(data_dir)
    inventory: dict[str, set[str]] = {}
    if not history_dir.exists():
        logger.warning("Dossier historique introuvable: %s", history_dir)
        return inventory

    for file_path in sorted(history_dir.glob("*.csv")):
        parsed = parse_history_filename(file_path)
        if parsed is None:
            continue
        symbol, timeframe = parsed
        inventory.setdefault(symbol, set()).add(timeframe)
    return inventory


def _can_build_timeframe(available: set[str], timeframe: str) -> bool:
    """Indique si un timeframe peut etre reconstruit depuis les sources disponibles.

    Args:
        available (set[str]): Timeframes presents pour un symbole.
        timeframe (str): Timeframe cible.

    Returns:
        bool: ``True`` si le timeframe peut etre charge ou resample.
    """
    if timeframe in available:
        return True
    if timeframe == "H1":
        return "M5" in available
    if timeframe == "D1":
        return "H1" in available or "M5" in available
    return False


def resolve_training_symbols(
    data_dir: str | os.PathLike[str] | None = None,
    required_timeframes: set[str] | None = None,
    max_symbols: int = 0,
) -> list[str]:
    """Retourne l'univers d'actifs effectivement entrainable.

    Args:
        data_dir (str | os.PathLike[str] | None): Dossier des CSV.
        required_timeframes (set[str] | None): Timeframes necessaires.
        max_symbols (int): Nombre max de symboles retournes. ``0`` desactive la limite.

    Returns:
        list[str]: Liste triee de symboles exploitables.
    """
    manual_symbols = os.getenv("TRAINING_SYMBOLS", "").strip()
    if manual_symbols:
        symbols = [item.strip() for item in manual_symbols.split(",") if item.strip()]
        return list(dict.fromkeys(symbols))

    required = required_timeframes or set()
    inventory = discover_history_inventory(data_dir)
    symbols: list[str] = []
    for symbol, available in inventory.items():
        if all(_can_build_timeframe(available, timeframe) for timeframe in required):
            symbols.append(symbol)

    preferred_prefix = {
        "BTCUSD": 0,
        "ETHUSD": 1,
        "XAUUSD": 2,
        "EURUSD": 3,
        "GBPUSD": 4,
        "USDJPY": 5,
        "US30.cash": 6,
    }
    symbols.sort(key=lambda item: (preferred_prefix.get(item, 999), item))
    if max_symbols > 0:
        return symbols[:max_symbols]
    return symbols


def load_history_frame(
    symbol: str,
    timeframe: str,
    data_dir: str | os.PathLike[str] | None = None,
) -> pd.DataFrame | None:
    """Charge un historique de marche ou le reconstruit par resampling.

    Args:
        symbol (str): Symbole a charger.
        timeframe (str): Timeframe cible (ex: ``M5``, ``H1``, ``D1``).
        data_dir (str | os.PathLike[str] | None): Dossier des CSV.

    Returns:
        pd.DataFrame | None: Donnees OHLCV indexees par date, ou ``None`` si indisponible.
    """
    history_dir = get_history_dir(data_dir)
    direct_path = history_dir / f"{symbol}_{timeframe}.csv"
    if direct_path.exists():
        return _read_history_frame(direct_path)

    if timeframe == "H1":
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1H")

    if timeframe == "D1":
        source_path = history_dir / f"{symbol}_H1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1D")
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1D")

    return None


def _read_history_frame(path: Path) -> pd.DataFrame:
    """Charge et normalise un CSV d'historique.

    Args:
        path (Path): Chemin du fichier CSV.

    Returns:
        pd.DataFrame: Donnees OHLCV triees et indexees sur ``time``.
    """
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"], utc=False)
    frame = frame.sort_values("time").drop_duplicates(subset=["time"])
    frame = frame.set_index("time")
    for column in ["tick_volume", "spread", "real_volume"]:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Reconstruit un timeframe superieur depuis un timeframe plus fin.

    Args:
        frame (pd.DataFrame): Donnees OHLCV source.
        rule (str): Regle Pandas de resampling (ex: ``1H``, ``1D``).

    Returns:
        pd.DataFrame: Donnees OHLCV resamplees.
    """
    aggregated = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "max",
            "real_volume": "sum",
        }
    )
    return aggregated.dropna(subset=["open", "high", "low", "close"])


def build_muzero_market_data(frame: pd.DataFrame) -> np.ndarray:
    """Construit la matrice d'observation MuZero a partir d'un historique reel.

    Args:
        frame (pd.DataFrame): Historique OHLCV indexe par date.

    Returns:
        np.ndarray: Matrice ``[pas_de_temps, 26]`` compatible avec l'environnement.
    """
    enriched = frame.copy()
    close = enriched["close"]
    high = enriched["high"]
    low = enriched["low"]
    volume = enriched["tick_volume"]

    enriched["ema_200"] = close.ewm(span=200, adjust=False).mean()
    enriched["rsi"] = IndicatorFactory.rsi(close, 14)

    macd = IndicatorFactory.macd(close)
    enriched["macd_hist"] = macd["histogram"]
    enriched["vwap"] = IndicatorFactory.vwap(high, low, close, volume)
    enriched["obv"] = IndicatorFactory.obv(close, volume)
    enriched["momentum"] = IndicatorFactory.momentum(close)
    enriched["trix"] = IndicatorFactory.trix(close)

    stochastic = IndicatorFactory.stochastic(high, low, close)
    enriched["stoch_k"] = stochastic["percent_k"]
    enriched["stoch_d"] = stochastic["percent_d"]

    enriched["cci"] = IndicatorFactory.cci(high, low, close)
    adx = IndicatorFactory.adx(high, low, close)
    enriched["adx"] = adx["adx"]
    enriched["adx_plus_di"] = adx["plus_di"]
    enriched["adx_minus_di"] = adx["minus_di"]

    ichimoku = IndicatorFactory.ichimoku(high, low, close)
    enriched["ichi_tenkan"] = ichimoku["tenkan_sen"]
    enriched["ichi_kijun"] = ichimoku["kijun_sen"]
    enriched["ichi_senkou_a"] = ichimoku["senkou_span_a"]
    enriched["ichi_senkou_b"] = ichimoku["senkou_span_b"]

    enriched["atr"] = IndicatorFactory.atr(high, low, close, 14)
    bollinger = IndicatorFactory.bollinger_bands(close)
    enriched["bb_pct"] = bollinger["pct_b"]
    enriched["spread_norm"] = enriched["spread"].astype(float) / close.replace(0, np.nan)
    enriched["return_1"] = close.pct_change().replace([np.inf, -np.inf], 0.0)

    enriched = enriched.ffill().bfill().fillna(0.0)

    ordered_columns = [
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "ema_200",
        "rsi",
        "macd_hist",
        "vwap",
        "obv",
        "momentum",
        "trix",
        "stoch_k",
        "stoch_d",
        "cci",
        "adx",
        "adx_plus_di",
        "adx_minus_di",
        "ichi_tenkan",
        "ichi_kijun",
        "ichi_senkou_a",
        "ichi_senkou_b",
        "atr",
        "bb_pct",
        "spread_norm",
        "return_1",
    ]
    matrix = enriched[ordered_columns].astype(np.float32).to_numpy()
    return matrix


def get_gnn_model_kwargs() -> dict[str, int]:
    """Retourne la configuration unique du modele GNN.

    Returns:
        dict[str, int]: Parametres communs du modele.
    """
    return {
        "asset_dim": GNN_ASSET_DIM,
        "temporal_dim": GNN_TEMPORAL_DIM,
        "hidden_dim": GNN_HIDDEN_DIM,
        "num_classes": GNN_NUM_CLASSES,
    }


def get_horizon_timeframe(horizon: str) -> str:
    """Mappe un horizon strategique vers son timeframe principal.

    Args:
        horizon (str): Horizon ``scalp``, ``intraday`` ou ``swing``.

    Returns:
        str: Timeframe associe.
    """
    return HORIZON_TO_TIMEFRAME.get(horizon.lower(), "H1")


def build_inventory_report(data_dir: str | os.PathLike[str] | None = None) -> dict[str, list[str]]:
    """Construit un rapport simple des historiques disponibles.

    Args:
        data_dir (str | os.PathLike[str] | None): Dossier des CSV.

    Returns:
        dict[str, list[str]]: Inventaire trie exploitable pour les logs et rapports.
    """
    inventory = discover_history_inventory(data_dir)
    return {symbol: sorted(timeframes) for symbol, timeframes in sorted(inventory.items())}
