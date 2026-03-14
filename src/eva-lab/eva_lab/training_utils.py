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
FOREX_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "CNH",
    "EUR",
    "GBP",
    "JPY",
    "NZD",
    "USD",
}
CRYPTO_BASES = {
    "AAVE",
    "AAV",
    "ADA",
    "ALGO",
    "AVAX",
    "BNB",
    "BTC",
    "DOGE",
    "DOT",
    "ETH",
    "LINK",
    "LTC",
    "SOL",
    "UNI",
    "XRP",
}
CRYPTO_QUOTES = ("USDT", "USDC", "USD", "EUR", "BTC", "ETH")
METAL_CODES = ("XAU", "XAG", "XPT", "XPD")
INDEX_TOKENS = (
    ".CASH",
    "US30",
    "US100",
    "US500",
    "GER40",
    "UK100",
    "NAS100",
    "SPX500",
    "USTEC",
)
CORE_FAMILIES = ("crypto", "forex", "index_cfd", "metal")
SECONDARY_FAMILIES = ("cfd_other", "equity_cfd", "unknown")


def _env_int(name: str, default: int) -> int:
    """Lit un entier depuis l'environnement avec repli robuste.

    Args:
        name (str): Nom de la variable a lire.
        default (int): Valeur de repli.

    Returns:
        int: Valeur entiere exploitable.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Valeur entiere invalide pour %s=%s. Repli=%s.", name, raw_value, default)
        return default

MTF_HORIZONS: dict[str, dict[str, Any]] = {
    "M5": {
        "minutes": 5,
        "count": _env_int("TRAINING_HISTORY_M5_BARS", 12000),
        "seq_len": 20,
        "future": 12,
        "strategy": "scalp",
    },
    "H1": {
        "minutes": 60,
        "count": _env_int("TRAINING_HISTORY_H1_BARS", 12000),
        "seq_len": 20,
        "future": 24,
        "strategy": "intraday",
    },
    "D1": {
        "minutes": 1440,
        "count": _env_int("TRAINING_HISTORY_D1_BARS", 1800),
        "seq_len": 15,
        "future": 7,
        "strategy": "swing",
    },
}

SUPPORTED_TIMEFRAMES: dict[str, dict[str, Any]] = {
    "M1": {
        "minutes": 1,
        "count": _env_int("TRAINING_HISTORY_M1_BARS", 30000),
    },
    "M5": {
        "minutes": 5,
        "count": _env_int("TRAINING_HISTORY_M5_BARS", 12000),
    },
    "M15": {
        "minutes": 15,
        "count": _env_int("TRAINING_HISTORY_M15_BARS", 15000),
    },
    "H1": {
        "minutes": 60,
        "count": _env_int("TRAINING_HISTORY_H1_BARS", 12000),
    },
    "D1": {
        "minutes": 1440,
        "count": _env_int("TRAINING_HISTORY_D1_BARS", 1800),
    },
    "W1": {
        "minutes": 10080,
        "count": _env_int("TRAINING_HISTORY_W1_BARS", 520),
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
    if timeframe == "M5":
        return "M1" in available
    if timeframe == "M15":
        return "M5" in available or "M1" in available
    if timeframe == "H1":
        return "M15" in available or "M5" in available or "M1" in available
    if timeframe == "D1":
        return "H1" in available or "M15" in available or "M5" in available or "M1" in available
    if timeframe == "W1":
        return (
            "D1" in available
            or "H1" in available
            or "M15" in available
            or "M5" in available
            or "M1" in available
        )
    return False


def classify_training_symbol(symbol: str) -> str:
    """Classe un symbole d'entrainement dans une famille de marche.

    Args:
        symbol (str): Symbole brut issu des historiques.

    Returns:
        str: Famille retenue pour l'equilibrage de l'univers.
    """
    symbol_upper = symbol.upper()
    alnum_symbol = "".join(char for char in symbol_upper if char.isalnum())

    if any(alnum_symbol.startswith(code) for code in METAL_CODES):
        return "metal"

    if _looks_like_crypto_symbol(alnum_symbol):
        return "crypto"

    if _looks_like_forex_symbol(alnum_symbol):
        return "forex"

    if any(token in symbol_upper for token in INDEX_TOKENS):
        return "index_cfd"

    if symbol_upper.endswith(".CASH"):
        return "equity_cfd"

    if any(char.isdigit() for char in symbol_upper):
        return "cfd_other"

    if 1 <= len(alnum_symbol) <= 6 and alnum_symbol.isalpha():
        return "equity_cfd"

    return "unknown"


def _looks_like_crypto_symbol(symbol: str) -> bool:
    """Retourne ``True`` si le symbole ressemble a une paire crypto.

    Args:
        symbol (str): Symbole normalise.

    Returns:
        bool: ``True`` si une structure crypto probable est detectee.
    """
    for quote in CRYPTO_QUOTES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            if base in CRYPTO_BASES and base not in FOREX_CODES:
                return True
    return False


def _looks_like_forex_symbol(symbol: str) -> bool:
    """Retourne ``True`` si le symbole ressemble a une paire Forex.

    Args:
        symbol (str): Symbole normalise.

    Returns:
        bool: ``True`` si le motif correspond a une paire Forex.
    """
    if len(symbol) < 6:
        return False
    base = symbol[:3]
    quote = symbol[3:6]
    if base in METAL_CODES:
        return False
    return base in FOREX_CODES and quote in FOREX_CODES


def _symbol_priority(symbol: str) -> tuple[int, str]:
    """Retourne une priorite stable pour les symboles preferes.

    Args:
        symbol (str): Symbole a ordonner.

    Returns:
        tuple[int, str]: Cle d'ordre stable.
    """
    preferred_prefix = {
        "BTCUSD": 0,
        "ETHUSD": 1,
        "EURUSD": 2,
        "GBPUSD": 3,
        "USDJPY": 4,
        "XAUUSD": 5,
        "US30.CASH": 6,
        "US100.CASH": 7,
        "US500.CASH": 8,
        "GER40.CASH": 9,
        "UK100.CASH": 10,
        "XAGUSD": 11,
    }
    return preferred_prefix.get(symbol.upper(), 999), symbol


def _sort_symbols_by_family(symbols: list[str]) -> dict[str, list[str]]:
    """Trie les symboles par famille d'actifs.

    Args:
        symbols (list[str]): Symboles candidats.

    Returns:
        dict[str, list[str]]: Mapping ``famille -> symboles tries``.
    """
    families: dict[str, list[str]] = {family: [] for family in (*CORE_FAMILIES, *SECONDARY_FAMILIES)}
    for symbol in symbols:
        family = classify_training_symbol(symbol)
        families.setdefault(family, []).append(symbol)

    for family_symbols in families.values():
        family_symbols.sort(key=_symbol_priority)
    return families


def _pick_balanced_symbols(symbols: list[str], max_symbols: int) -> list[str]:
    """Construit un univers equilibre entre familles d'actifs.

    Args:
        symbols (list[str]): Symboles entrainables.
        max_symbols (int): Nombre maximal de symboles a retenir.

    Returns:
        list[str]: Univers equilibre et dedoublonne.
    """
    families = _sort_symbols_by_family(symbols)
    selected: list[str] = []

    while len(selected) < max_symbols:
        added = False
        for family in CORE_FAMILIES:
            family_symbols = families.get(family, [])
            if family_symbols:
                selected.append(family_symbols.pop(0))
                added = True
                if len(selected) >= max_symbols:
                    return selected

        if not added:
            break

    while len(selected) < max_symbols:
        added = False
        for family in SECONDARY_FAMILIES:
            family_symbols = families.get(family, [])
            if family_symbols:
                selected.append(family_symbols.pop(0))
                added = True
                if len(selected) >= max_symbols:
                    return selected

        if not added:
            break

    return selected


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

    symbols.sort(key=lambda item: (classify_training_symbol(item), _symbol_priority(item)))
    if max_symbols > 0:
        return _pick_balanced_symbols(symbols, max_symbols)
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

    if timeframe == "M5":
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "5min")

    if timeframe == "M15":
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "15min")
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "15min")

    if timeframe == "H1":
        source_path = history_dir / f"{symbol}_M15.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1H")
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1H")
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1H")

    if timeframe == "D1":
        source_path = history_dir / f"{symbol}_H1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1D")
        source_path = history_dir / f"{symbol}_M15.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1D")
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1D")
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1D")

    if timeframe == "W1":
        source_path = history_dir / f"{symbol}_D1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1W")
        source_path = history_dir / f"{symbol}_H1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1W")
        source_path = history_dir / f"{symbol}_M15.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1W")
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1W")
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            return _resample_ohlcv(_read_history_frame(source_path), "1W")

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


def get_timeframe_history_bars(
    timeframe: str,
    env_prefix: str = "TRAINING_HISTORY",
    fallback: int | None = None,
) -> int:
    """Retourne le budget de bougies a consommer pour un timeframe.

    Args:
        timeframe (str): Timeframe cible (ex: ``M5``, ``H1``, ``D1``).
        env_prefix (str): Prefixe des variables d'environnement.
        fallback (int | None): Valeur de repli si le timeframe est inconnu.

    Returns:
        int: Nombre de bougies a charger.
    """
    timeframe_key = timeframe.upper()
    default_value = fallback if fallback is not None else int(SUPPORTED_TIMEFRAMES.get(timeframe_key, {}).get("count", 0))
    env_name = f"{env_prefix.upper()}_{timeframe_key}_BARS"
    return _env_int(env_name, default_value)


def get_horizon_history_bars(
    horizon: str,
    env_prefix: str = "TRAINING_HISTORY",
    fallback: int | None = None,
) -> int:
    """Retourne le budget de bougies a utiliser pour un horizon strategique.

    Args:
        horizon (str): Horizon ``scalp``, ``intraday`` ou ``swing``.
        env_prefix (str): Prefixe des variables d'environnement.
        fallback (int | None): Valeur de repli si necessaire.

    Returns:
        int: Nombre de bougies a charger pour l'horizon.
    """
    timeframe = get_horizon_timeframe(horizon)
    return get_timeframe_history_bars(timeframe, env_prefix=env_prefix, fallback=fallback)


def build_inventory_report(data_dir: str | os.PathLike[str] | None = None) -> dict[str, list[str]]:
    """Construit un rapport simple des historiques disponibles.

    Args:
        data_dir (str | os.PathLike[str] | None): Dossier des CSV.

    Returns:
        dict[str, list[str]]: Inventaire trie exploitable pour les logs et rapports.
    """
    inventory = discover_history_inventory(data_dir)
    return {symbol: sorted(timeframes) for symbol, timeframes in sorted(inventory.items())}
