"""Utilitaires communs pour les entrainements EVA Lab."""

from __future__ import annotations

import hashlib
import json
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


def _env_float(name: str, default: float) -> float:
    """Lit un flottant depuis l'environnement avec repli robuste.

    Args:
        name (str): Nom de la variable a lire.
        default (float): Valeur de repli.

    Returns:
        float: Valeur flottante exploitable.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Valeur flottante invalide pour %s=%s. Repli=%s.", name, raw_value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Lit un booleen depuis l'environnement avec repli robuste.

    Args:
        name (str): Nom de la variable a lire.
        default (bool): Valeur de repli.

    Returns:
        bool: Valeur booleenne exploitable.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Valeur booleenne invalide pour %s=%s. Repli=%s.", name, raw_value, default)
    return default


def parse_symbol_csv(raw_value: str | None) -> list[str]:
    """Normalise une liste CSV de symboles.

    Args:
        raw_value (str | None): Valeur brute issue d'une variable d'environnement.

    Returns:
        list[str]: Liste dedoublonnee et nettoyee.
    """
    if not raw_value:
        return []

    symbols: list[str] = []
    seen: set[str] = set()
    for item in str(raw_value).split(","):
        symbol = item.strip()
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    return symbols


def resolve_symbol_overrides(
    env_names: list[str] | tuple[str, ...] | None,
) -> tuple[list[str], str | None]:
    """Retourne la premiere surcharge de symboles disponible.

    Args:
        env_names (list[str] | tuple[str, ...] | None): Variables a tester,
            dans l'ordre de priorite.

    Returns:
        tuple[list[str], str | None]: Symboles retenus et nom de la variable
        d'environnement source.
    """
    for env_name in env_names or ():
        raw_value = os.getenv(env_name)
        symbols = parse_symbol_csv(raw_value)
        if symbols:
            return symbols, env_name
    return [], None

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

MODEL_FAMILY_SYMBOLS: dict[str, list[str]] = {
    "fx": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"],
    "indices": ["US30.cash", "US500.cash", "GER40.cash"],
    "metals": ["XAUUSD", "XAGUSD"],
    "crypto": ["BTCUSD", "ETHUSD", "BNBUSD"],
}

MODEL_TRAINING_PLAN: list[str] = [
    "scalp_fx",
    "scalp_indices",
    "scalp_metals",
    "intraday_fx",
    "intraday_indices",
    "swing_fx",
    "swing_indices",
    "scalp_crypto",
    "intraday_metals",
    "swing_metals",
    "intraday_crypto",
    "swing_crypto",
]

FAMILY_ALIASES: dict[str, str] = {
    "forex": "fx",
    "fx": "fx",
    "index": "indices",
    "indices": "indices",
    "index_cfd": "indices",
    "metal": "metals",
    "metals": "metals",
    "crypto": "crypto",
}

FEATURE_PROFILE_MATRIX: dict[str, dict[str, dict[str, Any]]] = {
    "scalp": {
        "fx": {
            "profile_name": "scalp_fx_v1",
            "feature_version": "v1",
            "entry_features": ["EMA_200", "VWAP", "ADX", "RSI", "ATR", "session_phase"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "relative_volume",
            ],
        },
        "indices": {
            "profile_name": "scalp_indices_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum", "relative_volume"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "bb_width",
                "relative_volume",
            ],
        },
        "metals": {
            "profile_name": "scalp_metals_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "BB_Pct", "ATR", "RSI"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "bb_width",
            ],
        },
        "crypto": {
            "profile_name": "scalp_crypto_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum", "relative_volume"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "relative_volume",
            ],
        },
    },
    "intraday": {
        "fx": {
            "profile_name": "intraday_fx_v1",
            "feature_version": "v1",
            "entry_features": ["EMA_200", "VWAP", "ADX", "RSI", "ATR", "session_phase"],
            "audit_features": ["price_vs_vwap", "obv_slope", "atr_pct", "adx_regime", "relative_volume"],
        },
        "indices": {
            "profile_name": "intraday_indices_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum", "relative_volume"],
            "audit_features": ["price_vs_vwap", "obv_slope", "vwap_distance_zscore", "adx_regime", "bb_width"],
        },
        "metals": {
            "profile_name": "intraday_metals_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "BB_Pct", "ATR", "RSI"],
            "audit_features": ["price_vs_vwap", "obv_slope", "vwap_distance_zscore", "atr_pct", "bb_width"],
        },
        "crypto": {
            "profile_name": "intraday_crypto_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum", "relative_volume"],
            "audit_features": ["price_vs_vwap", "obv_slope", "obv_divergence", "atr_pct", "relative_volume"],
        },
    },
    "swing": {
        "fx": {
            "profile_name": "swing_fx_v1",
            "feature_version": "v1",
            "entry_features": ["EMA_200", "ADX", "RSI", "ATR", "session_phase"],
            "audit_features": ["price_vs_vwap", "obv_slope", "atr_pct", "adx_regime", "relative_volume"],
        },
        "indices": {
            "profile_name": "swing_indices_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum"],
            "audit_features": ["price_vs_vwap", "obv_slope", "vwap_distance_zscore", "atr_pct", "bb_width"],
        },
        "metals": {
            "profile_name": "swing_metals_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "BB_Pct", "ATR", "RSI"],
            "audit_features": ["price_vs_vwap", "obv_slope", "vwap_distance_zscore", "bb_width", "atr_pct"],
        },
        "crypto": {
            "profile_name": "swing_crypto_v1",
            "feature_version": "v1",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum"],
            "audit_features": ["price_vs_vwap", "obv_slope", "obv_divergence", "atr_pct", "relative_volume"],
        },
    },
}

POSITION_MECHANICS_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "scalp": {
        "fx": {
            "profile_name": "scalp_fx_v1",
            "entry_filter": {
                "ema_mode": "strict",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "min_adx": 18.0,
                "trend_adx": 23.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 80,
                "stale_penalty": 0.75,
                "trend_penalty": 0.35,
                "range_penalty": 0.05,
            },
            "pyramiding_policy": {
                "max_additions": 1,
                "min_profit_to_add": 0.0015,
                "reward_bonus": 0.08,
            },
            "split_policy": {
                "max_splits": 2,
                "min_trade_return": 0.008,
                "slbe_after_split": True,
            },
            "slbe_policy": {
                "activation_return": 0.0045,
                "bonus": 6.0,
            },
            "close_policy": {
                "winner_threshold": 0.012,
                "strong_winner_threshold": 0.02,
                "tp_like_threshold": 0.01,
            },
        },
        "indices": {
            "profile_name": "scalp_indices_v1",
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": True,
                "require_obv_confirmation": True,
                "min_adx": 16.0,
                "trend_adx": 21.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 90,
                "stale_penalty": 0.60,
                "trend_penalty": 0.28,
                "range_penalty": 0.04,
            },
            "pyramiding_policy": {
                "max_additions": 2,
                "min_profit_to_add": 0.0012,
                "reward_bonus": 0.10,
            },
            "split_policy": {
                "max_splits": 3,
                "min_trade_return": 0.007,
                "slbe_after_split": True,
            },
            "slbe_policy": {
                "activation_return": 0.0040,
                "bonus": 6.0,
            },
            "close_policy": {
                "winner_threshold": 0.010,
                "strong_winner_threshold": 0.018,
                "tp_like_threshold": 0.009,
            },
        },
        "metals": {
            "profile_name": "scalp_metals_v1",
            "entry_filter": {
                "ema_mode": "relaxed",
                "require_vwap_alignment": True,
                "require_obv_confirmation": True,
                "min_adx": 15.0,
                "trend_adx": 20.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 90,
                "stale_penalty": 0.55,
                "trend_penalty": 0.24,
                "range_penalty": 0.04,
            },
            "pyramiding_policy": {
                "max_additions": 2,
                "min_profit_to_add": 0.0010,
                "reward_bonus": 0.10,
            },
            "split_policy": {
                "max_splits": 3,
                "min_trade_return": 0.006,
                "slbe_after_split": True,
            },
            "slbe_policy": {
                "activation_return": 0.0038,
                "bonus": 6.0,
            },
            "close_policy": {
                "winner_threshold": 0.009,
                "strong_winner_threshold": 0.016,
                "tp_like_threshold": 0.008,
            },
        },
        "crypto": {
            "profile_name": "scalp_crypto_v1",
            "entry_filter": {
                "ema_mode": "relaxed",
                "require_vwap_alignment": False,
                "require_obv_confirmation": True,
                "min_adx": 14.0,
                "trend_adx": 18.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 100,
                "stale_penalty": 0.50,
                "trend_penalty": 0.20,
                "range_penalty": 0.03,
            },
            "pyramiding_policy": {
                "max_additions": 2,
                "min_profit_to_add": 0.0015,
                "reward_bonus": 0.08,
            },
            "split_policy": {
                "max_splits": 2,
                "min_trade_return": 0.008,
                "slbe_after_split": True,
            },
            "slbe_policy": {
                "activation_return": 0.0050,
                "bonus": 5.0,
            },
            "close_policy": {
                "winner_threshold": 0.012,
                "strong_winner_threshold": 0.022,
                "tp_like_threshold": 0.010,
            },
        },
    },
}
POSITION_MECHANICS_PROFILES["intraday"] = {
    family: json.loads(json.dumps(profile))
    for family, profile in POSITION_MECHANICS_PROFILES["scalp"].items()
}
POSITION_MECHANICS_PROFILES["swing"] = {
    family: json.loads(json.dumps(profile))
    for family, profile in POSITION_MECHANICS_PROFILES["scalp"].items()
}
for family_name, family_profile in POSITION_MECHANICS_PROFILES["intraday"].items():
    family_profile["profile_name"] = f"intraday_{family_name}_v1"
    family_profile["entry_filter"]["min_adx"] = max(
        12.0,
        float(family_profile["entry_filter"]["min_adx"]) - 2.0,
    )
    family_profile["entry_filter"]["trend_adx"] = max(
        16.0,
        float(family_profile["entry_filter"]["trend_adx"]) - 2.0,
    )
    family_profile["hold_policy"]["stale_penalty_after_steps"] = int(
        family_profile["hold_policy"]["stale_penalty_after_steps"]
    ) + 40
    family_profile["hold_policy"]["trend_penalty"] = max(
        0.15,
        float(family_profile["hold_policy"]["trend_penalty"]) - 0.08,
    )
    family_profile["close_policy"]["winner_threshold"] = max(
        0.012,
        float(family_profile["close_policy"]["winner_threshold"]) + 0.004,
    )
    family_profile["close_policy"]["strong_winner_threshold"] = max(
        0.020,
        float(family_profile["close_policy"]["strong_winner_threshold"]) + 0.006,
    )

for family_name, family_profile in POSITION_MECHANICS_PROFILES["swing"].items():
    family_profile["profile_name"] = f"swing_{family_name}_v1"
    family_profile["entry_filter"]["min_adx"] = max(
        10.0,
        float(family_profile["entry_filter"]["min_adx"]) - 4.0,
    )
    family_profile["entry_filter"]["trend_adx"] = max(
        14.0,
        float(family_profile["entry_filter"]["trend_adx"]) - 4.0,
    )
    family_profile["hold_policy"]["stale_penalty_after_steps"] = int(
        family_profile["hold_policy"]["stale_penalty_after_steps"]
    ) + 80
    family_profile["hold_policy"]["trend_penalty"] = max(
        0.10,
        float(family_profile["hold_policy"]["trend_penalty"]) - 0.14,
    )
    family_profile["close_policy"]["winner_threshold"] = max(
        0.016,
        float(family_profile["close_policy"]["winner_threshold"]) + 0.007,
    )
    family_profile["close_policy"]["strong_winner_threshold"] = max(
        0.028,
        float(family_profile["close_policy"]["strong_winner_threshold"]) + 0.010,
    )


def get_model_family_symbols(family: str) -> list[str]:
    """Retourne les symboles figes d'une famille de modeles.

    Args:
        family (str): Famille ciblee.

    Returns:
        list[str]: Liste de symboles ordonnee et stable.
    """
    return list(MODEL_FAMILY_SYMBOLS.get(normalize_model_family(family), []))


def get_model_training_plan() -> list[str]:
    """Retourne l'ordre officiel de relance des familles de modeles.

    Returns:
        list[str]: Plan d'entrainement stable pour l'usine V2.
    """
    return list(MODEL_TRAINING_PLAN)


def normalize_model_family(family: str | None) -> str:
    """Normalise une famille d'actifs vers le vocabulaire officiel V2.

    Args:
        family (str | None): Famille brute a normaliser.

    Returns:
        str: Famille normalisee (`fx`, `indices`, `metals`, `crypto`) ou `mixed`.
    """
    normalized = str(family or "").strip().lower()
    if not normalized:
        return "mixed"
    return FAMILY_ALIASES.get(normalized, normalized)


def resolve_model_family(symbol: str | None = None, family: str | None = None) -> str:
    """Determine la famille officielle d'un symbole ou d'une configuration.

    Args:
        symbol (str | None): Symbole de marche si disponible.
        family (str | None): Famille explicite si deja connue.

    Returns:
        str: Famille normalisee utilisable par l'usine de modeles.
    """
    if family:
        return normalize_model_family(family)
    if not symbol:
        return "mixed"
    return normalize_model_family(classify_training_symbol(symbol))


def infer_family_from_symbols(symbols: list[str], family: str | None = None) -> str:
    """Deduit une famille globale a partir d'un univers de symboles.

    Args:
        symbols (list[str]): Symboles du run.
        family (str | None): Famille explicite prioritaire si fournie.

    Returns:
        str: Famille unique si stable, sinon `mixed`.
    """
    if family:
        return resolve_model_family(family=family)
    normalized_families = {
        resolve_model_family(symbol=symbol)
        for symbol in symbols
        if str(symbol or "").strip()
    }
    if len(normalized_families) == 1:
        return next(iter(normalized_families))
    return "mixed"


def _extract_profile_version(profile_name: str | None, default: str = "v1") -> str:
    """Extrait une version de profil depuis son nom logique.

    Args:
        profile_name (str | None): Nom complet du profil.
        default (str): Version de repli si aucune suffixe n'est detecte.

    Returns:
        str: Version normalisee du profil.
    """
    normalized = str(profile_name or "").strip().lower()
    if not normalized:
        return default
    tokens = normalized.split("_")
    if tokens and tokens[-1].startswith("v"):
        return tokens[-1]
    return default


def _upgrade_scalp_feature_profile_v2(family: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Applique le profil V3 des features pour `scalp`.

    Args:
        family (str): Famille d'actifs ciblee.
        profile (dict[str, Any]): Profil de base resolu depuis la matrice V1.

    Returns:
        dict[str, Any]: Profil enrichi et versionne pour la V3.
    """
    upgraded = dict(profile)
    upgraded["feature_version"] = "v2"

    family_overrides: dict[str, dict[str, Any]] = {
        "fx": {
            "profile_name": "scalp_fx_v2",
            "entry_features": ["EMA_200", "VWAP", "ADX", "RSI", "ATR", "session_phase"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "bb_width",
                "relative_volume",
                "session_phase",
            ],
        },
        "indices": {
            "profile_name": "scalp_indices_v2",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum", "relative_volume"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "bb_width",
                "relative_volume",
                "session_phase",
            ],
        },
        "metals": {
            "profile_name": "scalp_metals_v2",
            "entry_features": ["VWAP", "OBV", "ADX", "BB_Pct", "ATR", "RSI"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "bb_width",
                "relative_volume",
            ],
        },
        "crypto": {
            "profile_name": "scalp_crypto_v2",
            "entry_features": ["VWAP", "OBV", "ADX", "ATR", "Momentum", "relative_volume"],
            "audit_features": [
                "price_vs_vwap",
                "obv_slope",
                "obv_divergence",
                "vwap_distance_zscore",
                "atr_pct",
                "adx_regime",
                "bb_width",
                "relative_volume",
            ],
        },
    }

    upgraded.update(family_overrides.get(family, family_overrides["fx"]))
    return upgraded


def _upgrade_scalp_position_mechanics_profile_v2(
    family: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Applique le profil V3 de mecanique de position pour `scalp`.

    Args:
        family (str): Famille d'actifs ciblee.
        profile (dict[str, Any]): Profil de base V1.

    Returns:
        dict[str, Any]: Profil V3 enrichi de politiques d'activite et de recompense.
    """
    upgraded = json.loads(json.dumps(profile))

    family_overrides: dict[str, dict[str, Any]] = {
        "fx": {
            "profile_name": "scalp_fx_v2",
            "entry_filter": {
                "ema_mode": "moderate",
                "require_vwap_alignment": True,
                "require_obv_confirmation": False,
                "allow_trend_fallback": False,
                "min_adx": 13.5,
                "trend_adx": 18.5,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 48,
                "stale_penalty": 1.10,
                "trend_penalty": 0.58,
                "range_penalty": 0.08,
            },
            "pyramiding_policy": {
                "max_additions": 2,
                "min_profit_to_add": 0.0006,
                "reward_bonus": 0.16,
            },
            "split_policy": {
                "max_splits": 3,
                "min_trade_return": 0.0038,
                "min_realized_pct": 0.025,
                "slbe_after_split": True,
                "failure_penalty": 0.85,
            },
            "slbe_policy": {
                "activation_return": 0.0024,
                "bonus": 7.5,
                "exit_bonus": 1.75,
            },
            "close_policy": {
                "winner_threshold": 0.0055,
                "strong_winner_threshold": 0.0095,
                "tp_like_threshold": 0.0042,
            },
            "activity_policy": {
                "min_entries": 2,
                "inactive_episode_penalty": 16.0,
                "insufficient_entries_penalty": 8.0,
            },
            "directional_policy": {
                "min_entry_share": 0.20,
                "max_directional_imbalance": 0.60,
                "imbalance_penalty": 8.0,
            },
            "reward_policy": {
                "realized_reward_multiplier": 1.35,
                "close_realized_bonus_multiplier": 1.75,
                "split_realized_bonus_multiplier": 1.20,
                "hold_drag_penalty_multiplier": 0.35,
                "pyramid_failure_penalty": 0.18,
                "pyramid_negative_exit_penalty": 0.45,
            },
        },
        "indices": {
            "profile_name": "scalp_indices_v2",
            "entry_filter": {
                "ema_mode": "relaxed",
                "require_vwap_alignment": False,
                "require_obv_confirmation": False,
                "allow_trend_fallback": True,
                "min_adx": 10.5,
                "trend_adx": 15.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 42,
                "stale_penalty": 1.25,
                "trend_penalty": 0.64,
                "range_penalty": 0.08,
            },
            "pyramiding_policy": {
                "max_additions": 3,
                "min_profit_to_add": 0.0005,
                "reward_bonus": 0.18,
            },
            "split_policy": {
                "max_splits": 4,
                "min_trade_return": 0.0035,
                "min_realized_pct": 0.020,
                "slbe_after_split": True,
                "failure_penalty": 0.80,
            },
            "slbe_policy": {
                "activation_return": 0.0020,
                "bonus": 7.0,
                "exit_bonus": 1.50,
            },
            "close_policy": {
                "winner_threshold": 0.0048,
                "strong_winner_threshold": 0.0088,
                "tp_like_threshold": 0.0040,
            },
            "activity_policy": {
                "min_entries": 1,
                "inactive_episode_penalty": 18.0,
                "insufficient_entries_penalty": 9.0,
            },
            "directional_policy": {
                "min_entry_share": 0.18,
                "max_directional_imbalance": 0.65,
                "imbalance_penalty": 7.0,
            },
            "reward_policy": {
                "realized_reward_multiplier": 1.30,
                "close_realized_bonus_multiplier": 1.65,
                "split_realized_bonus_multiplier": 1.15,
                "hold_drag_penalty_multiplier": 0.42,
                "pyramid_failure_penalty": 0.16,
                "pyramid_negative_exit_penalty": 0.38,
            },
        },
        "metals": {
            "profile_name": "scalp_metals_v2",
            "entry_filter": {
                "ema_mode": "relaxed",
                "require_vwap_alignment": True,
                "require_obv_confirmation": True,
                "allow_trend_fallback": True,
                "min_adx": 10.0,
                "trend_adx": 14.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 38,
                "stale_penalty": 1.35,
                "trend_penalty": 0.70,
                "range_penalty": 0.08,
            },
            "pyramiding_policy": {
                "max_additions": 3,
                "min_profit_to_add": 0.0007,
                "reward_bonus": 0.20,
            },
            "split_policy": {
                "max_splits": 4,
                "min_trade_return": 0.0032,
                "min_realized_pct": 0.020,
                "slbe_after_split": True,
                "failure_penalty": 0.75,
            },
            "slbe_policy": {
                "activation_return": 0.0018,
                "bonus": 8.0,
                "exit_bonus": 2.20,
            },
            "close_policy": {
                "winner_threshold": 0.0040,
                "strong_winner_threshold": 0.0078,
                "tp_like_threshold": 0.0036,
            },
            "activity_policy": {
                "min_entries": 1,
                "inactive_episode_penalty": 18.0,
                "insufficient_entries_penalty": 8.0,
            },
            "directional_policy": {
                "min_entry_share": 0.18,
                "max_directional_imbalance": 0.65,
                "imbalance_penalty": 7.0,
            },
            "reward_policy": {
                "realized_reward_multiplier": 1.40,
                "close_realized_bonus_multiplier": 1.90,
                "split_realized_bonus_multiplier": 1.25,
                "hold_drag_penalty_multiplier": 0.45,
                "pyramid_failure_penalty": 0.18,
                "pyramid_negative_exit_penalty": 0.42,
            },
        },
        "crypto": {
            "profile_name": "scalp_crypto_v2",
            "entry_filter": {
                "ema_mode": "relaxed",
                "require_vwap_alignment": True,
                "require_obv_confirmation": True,
                "allow_trend_fallback": True,
                "min_adx": 12.0,
                "trend_adx": 17.0,
            },
            "hold_policy": {
                "stale_penalty_after_steps": 36,
                "stale_penalty": 1.30,
                "trend_penalty": 0.62,
                "range_penalty": 0.08,
            },
            "pyramiding_policy": {
                "max_additions": 2,
                "min_profit_to_add": 0.0010,
                "reward_bonus": 0.18,
            },
            "split_policy": {
                "max_splits": 3,
                "min_trade_return": 0.0040,
                "min_realized_pct": 0.022,
                "slbe_after_split": True,
                "failure_penalty": 0.90,
            },
            "slbe_policy": {
                "activation_return": 0.0022,
                "bonus": 7.0,
                "exit_bonus": 1.80,
            },
            "close_policy": {
                "winner_threshold": 0.0055,
                "strong_winner_threshold": 0.0100,
                "tp_like_threshold": 0.0046,
            },
            "activity_policy": {
                "min_entries": 1,
                "inactive_episode_penalty": 18.0,
                "insufficient_entries_penalty": 8.0,
            },
            "directional_policy": {
                "min_entry_share": 0.18,
                "max_directional_imbalance": 0.68,
                "imbalance_penalty": 7.0,
            },
            "reward_policy": {
                "realized_reward_multiplier": 1.35,
                "close_realized_bonus_multiplier": 1.75,
                "split_realized_bonus_multiplier": 1.15,
                "hold_drag_penalty_multiplier": 0.38,
                "pyramid_failure_penalty": 0.20,
                "pyramid_negative_exit_penalty": 0.45,
            },
        },
    }

    selected = family_overrides.get(family, family_overrides["fx"])
    for section_name, section_payload in selected.items():
        if isinstance(section_payload, dict):
            current_section = dict(upgraded.get(section_name) or {})
            current_section.update(section_payload)
            upgraded[section_name] = current_section
        else:
            upgraded[section_name] = section_payload
    return upgraded


def resolve_feature_profile(horizon: str, family: str | None) -> dict[str, Any]:
    """Retourne le profil officiel de features pour un couple horizon/famille.

    Args:
        horizon (str): Horizon strategique cible.
        family (str | None): Famille d'actifs ciblee.

    Returns:
        dict[str, Any]: Profil stable et versionne pour le run.
    """
    normalized_horizon = str(horizon or "intraday").strip().lower()
    normalized_family = resolve_model_family(family=family)
    matrix = FEATURE_PROFILE_MATRIX.get(normalized_horizon, FEATURE_PROFILE_MATRIX["intraday"])
    profile = dict(matrix.get(normalized_family) or matrix.get("fx") or {})
    if normalized_horizon == "scalp":
        profile = _upgrade_scalp_feature_profile_v2(normalized_family, profile)
    profile["horizon"] = normalized_horizon
    profile["family"] = normalized_family
    profile["profile_name"] = str(
        profile.get("profile_name")
        or f"{normalized_horizon}_{normalized_family}_v1"
    )
    profile["profile_version"] = _extract_profile_version(profile.get("profile_name"))
    return profile


def resolve_position_mechanics_profile(horizon: str, family: str | None) -> dict[str, Any]:
    """Retourne le profil de mecanique de position pour un horizon/famille.

    Args:
        horizon (str): Horizon strategique cible.
        family (str | None): Famille d'actifs ciblee.

    Returns:
        dict[str, Any]: Regles de filtrage, hold, split, SLBE et pyramiding.
    """
    normalized_horizon = str(horizon or "intraday").strip().lower()
    normalized_family = resolve_model_family(family=family)
    matrix = POSITION_MECHANICS_PROFILES.get(normalized_horizon, POSITION_MECHANICS_PROFILES["intraday"])
    profile = json.loads(json.dumps(matrix.get(normalized_family) or matrix.get("fx") or {}))
    if normalized_horizon == "scalp":
        profile = _upgrade_scalp_position_mechanics_profile_v2(normalized_family, profile)
    profile["horizon"] = normalized_horizon
    profile["family"] = normalized_family
    profile["profile_name"] = str(
        profile.get("profile_name")
        or f"{normalized_horizon}_{normalized_family}_v1"
    )
    profile["profile_version"] = _extract_profile_version(profile.get("profile_name"))
    return _apply_position_mechanics_env_overrides(profile)


def _apply_position_mechanics_env_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    """Applique les surcharges runtime de mecanique de position.

    Ces surcharges servent a piloter rapidement les runs d'exploration V2
    sans dupliquer les profils horizon/famille dans le code.

    Args:
        profile (dict[str, Any]): Profil de base deja resolu.

    Returns:
        dict[str, Any]: Profil enrichi des surcharges runtime.
    """
    override_sources: list[str] = []

    def apply_value(
        section: str,
        key: str,
        env_name: str,
        parser,
    ) -> None:
        raw_value = os.getenv(env_name)
        if raw_value is None:
            return
        current_section = dict(profile.get(section) or {})
        current_value = current_section.get(key)
        parsed_value = parser(env_name, current_value)
        current_section[key] = parsed_value
        profile[section] = current_section
        override_sources.append(env_name)

    apply_value("entry_filter", "ema_mode", "MUZERO_ENTRY_EMA_MODE", lambda env_name, current: str(os.getenv(env_name) or current))
    apply_value(
        "entry_filter",
        "require_vwap_alignment",
        "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT",
        lambda env_name, current: _env_bool(env_name, bool(current)),
    )
    apply_value(
        "entry_filter",
        "require_obv_confirmation",
        "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION",
        lambda env_name, current: _env_bool(env_name, bool(current)),
    )
    apply_value(
        "entry_filter",
        "min_adx",
        "MUZERO_ENTRY_MIN_ADX",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "entry_filter",
        "trend_adx",
        "MUZERO_ENTRY_TREND_ADX",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "hold_policy",
        "stale_penalty_after_steps",
        "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS",
        lambda env_name, current: _env_int(env_name, int(current or 0)),
    )
    apply_value(
        "hold_policy",
        "stale_penalty",
        "MUZERO_HOLD_STALE_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "hold_policy",
        "trend_penalty",
        "MUZERO_HOLD_TREND_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "hold_policy",
        "range_penalty",
        "MUZERO_HOLD_RANGE_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "pyramiding_policy",
        "max_additions",
        "MUZERO_PYRAMID_MAX_ADDITIONS",
        lambda env_name, current: _env_int(env_name, int(current or 0)),
    )
    apply_value(
        "pyramiding_policy",
        "min_profit_to_add",
        "MUZERO_PYRAMID_MIN_PROFIT_TO_ADD",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "pyramiding_policy",
        "reward_bonus",
        "MUZERO_PYRAMID_REWARD_BONUS",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "split_policy",
        "max_splits",
        "MUZERO_SPLIT_MAX_SPLITS",
        lambda env_name, current: _env_int(env_name, int(current or 0)),
    )
    apply_value(
        "split_policy",
        "min_trade_return",
        "MUZERO_SPLIT_MIN_TRADE_RETURN",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "slbe_policy",
        "activation_return",
        "MUZERO_SLBE_ACTIVATION_RETURN",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "slbe_policy",
        "bonus",
        "MUZERO_SLBE_BONUS",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "close_policy",
        "winner_threshold",
        "MUZERO_CLOSE_WINNER_THRESHOLD",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "close_policy",
        "strong_winner_threshold",
        "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "close_policy",
        "tp_like_threshold",
        "MUZERO_CLOSE_TP_LIKE_THRESHOLD",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "entry_filter",
        "allow_trend_fallback",
        "MUZERO_ENTRY_ALLOW_TREND_FALLBACK",
        lambda env_name, current: _env_bool(env_name, bool(current)),
    )
    apply_value(
        "activity_policy",
        "min_entries",
        "MUZERO_ACTIVITY_MIN_ENTRIES",
        lambda env_name, current: _env_int(env_name, int(current or 0)),
    )
    apply_value(
        "activity_policy",
        "inactive_episode_penalty",
        "MUZERO_ACTIVITY_INACTIVE_EPISODE_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "activity_policy",
        "insufficient_entries_penalty",
        "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "directional_policy",
        "min_entry_share",
        "MUZERO_DIRECTIONAL_MIN_ENTRY_SHARE",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "directional_policy",
        "max_directional_imbalance",
        "MUZERO_DIRECTIONAL_MAX_IMBALANCE",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "directional_policy",
        "imbalance_penalty",
        "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "split_policy",
        "min_realized_pct",
        "MUZERO_SPLIT_MIN_REALIZED_PCT",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "split_policy",
        "failure_penalty",
        "MUZERO_SPLIT_FAILURE_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "slbe_policy",
        "exit_bonus",
        "MUZERO_SLBE_EXIT_BONUS",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "reward_policy",
        "realized_reward_multiplier",
        "MUZERO_REWARD_REALIZED_PNL_MULTIPLIER",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "reward_policy",
        "close_realized_bonus_multiplier",
        "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "reward_policy",
        "split_realized_bonus_multiplier",
        "MUZERO_REWARD_SPLIT_REALIZED_MULTIPLIER",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "reward_policy",
        "hold_drag_penalty_multiplier",
        "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "reward_policy",
        "pyramid_failure_penalty",
        "MUZERO_REWARD_PYRAMID_FAILURE_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    apply_value(
        "reward_policy",
        "pyramid_negative_exit_penalty",
        "MUZERO_REWARD_PYRAMID_NEGATIVE_EXIT_PENALTY",
        lambda env_name, current: _env_float(env_name, float(current or 0.0)),
    )
    if override_sources:
        profile["runtime_override_vars"] = override_sources
        profile["profile_name"] = f"{profile['profile_name']}_runtime"
        profile["profile_version"] = _extract_profile_version(profile.get("profile_name"), default="runtime")
    return profile


def _inventory_supports_timeframe(timeframes: set[str], timeframe: str) -> bool:
    """Indique si un inventaire CSV peut satisfaire un timeframe demande.

    Args:
        timeframes (set[str]): Timeframes disponibles pour un symbole.
        timeframe (str): Timeframe demande.

    Returns:
        bool: `True` si le timeframe est directement disponible ou reconstructible.
    """
    normalized = {str(item).upper() for item in timeframes}
    target = str(timeframe).upper()
    if target in normalized:
        return True
    fallback_matrix = {
        "M15": {"M5", "M1"},
        "H1": {"M15", "M5", "M1"},
        "D1": {"H1", "M15", "M5", "M1"},
        "W1": {"D1", "H1", "M15", "M5", "M1"},
    }
    return bool(normalized & fallback_matrix.get(target, set()))


def build_dataset_coverage(
    *,
    symbols: list[str],
    timeframe: str,
    data_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Calcule la couverture effective du dataset pour un run.

    Args:
        symbols (list[str]): Univers cible du run.
        timeframe (str): Timeframe principal du run.
        data_dir (str | os.PathLike[str] | None): Dossier CSV optionnel.

    Returns:
        dict[str, Any]: Couverture CSV/TimeDB et source effective pressentie.
    """
    normalized_symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    target_timeframe = str(timeframe).upper()
    csv_inventory = discover_history_inventory(data_dir)
    csv_available = [
        symbol
        for symbol in normalized_symbols
        if _inventory_supports_timeframe(csv_inventory.get(symbol, set()), target_timeframe)
    ]

    timescale_info: dict[str, Any] = {}
    timescale_available: list[str] = []
    try:
        from eva_lab.timescale_store import (
            describe_timescale_source,
            discover_timescale_inventory,
            ensure_timescale_ready,
        )

        timescale_info = describe_timescale_source()
        timescale_ready = ensure_timescale_ready() if bool(timescale_info.get("enabled", False)) else False
        timescale_inventory = discover_timescale_inventory()
        timescale_available = [
            symbol
            for symbol in normalized_symbols
            if target_timeframe in {str(item).upper() for item in timescale_inventory.get(symbol, set())}
        ]
    except Exception:
        timescale_info = {"enabled": False, "kind": "timescaledb", "state": "offline"}
        timescale_ready = False
        timescale_available = []

    csv_missing = [symbol for symbol in normalized_symbols if symbol not in csv_available]
    timescale_missing = [symbol for symbol in normalized_symbols if symbol not in timescale_available]
    timescale_enabled = bool(timescale_info.get("enabled", False))
    if not timescale_enabled:
        effective_source = "csv"
        effective_source_reason = "timescaledb_disabled"
    elif not timescale_ready:
        effective_source = "csv"
        effective_source_reason = "timescaledb_unreachable"
    elif timescale_missing:
        effective_source = "csv"
        effective_source_reason = "timescaledb_incomplete_coverage"
    else:
        effective_source = "timescaledb"
        effective_source_reason = "timescaledb_ready"
    effective_available = timescale_available if effective_source == "timescaledb" else csv_available
    effective_missing = timescale_missing if effective_source == "timescaledb" else csv_missing
    effective_ratio = len(effective_available) / max(len(normalized_symbols), 1)

    return {
        "timeframe": target_timeframe,
        "requested_symbols": list(normalized_symbols),
        "required_symbols": len(normalized_symbols),
        "csv": {
            "available_symbols": csv_available,
            "missing_symbols": csv_missing,
            "coverage_ratio": len(csv_available) / max(len(normalized_symbols), 1),
        },
        "timescaledb": {
            "enabled": timescale_enabled,
            "ready": timescale_ready,
            "state": str(timescale_info.get("state") or ("enabled" if timescale_enabled else "disabled")),
            "host": timescale_info.get("host"),
            "database": timescale_info.get("database"),
            "bars_table": timescale_info.get("bars_table"),
            "features_table": timescale_info.get("features_table"),
            "available_symbols": timescale_available,
            "missing_symbols": timescale_missing,
            "coverage_ratio": len(timescale_available) / max(len(normalized_symbols), 1),
        },
        "effective_source": effective_source,
        "effective_source_reason": effective_source_reason,
        "effective_symbols": effective_available,
        "missing_symbols": effective_missing,
        "coverage_ratio": effective_ratio,
        "all_symbols_available": len(effective_available) == len(normalized_symbols),
    }


def build_dataset_id(
    *,
    horizon: str,
    family: str,
    timeframe: str,
    symbols: list[str],
    source: str,
    feature_profile: str,
    history_bars: int,
) -> str:
    """Construit un identifiant immuable de dataset.

    Args:
        horizon (str): Horizon strategique.
        family (str): Famille d'actifs.
        timeframe (str): Timeframe principal.
        symbols (list[str]): Univers exact du run.
        source (str): Source historique (`csv`, `timescaledb`, etc.).
        feature_profile (str): Profil de features applique.
        history_bars (int): Fenetre historique chargee.

    Returns:
        str: Identifiant stable du dataset.
    """
    signature_payload = {
        "horizon": str(horizon).lower(),
        "family": resolve_model_family(family=family),
        "timeframe": str(timeframe).upper(),
        "symbols": list(symbols),
        "source": str(source).lower(),
        "feature_profile": str(feature_profile),
        "history_bars": int(history_bars),
    }
    signature = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"{signature_payload['horizon']}_"
        f"{signature_payload['family']}_"
        f"{signature_payload['timeframe'].lower()}_"
        f"{signature}"
    )


def build_dataset_descriptor(
    *,
    horizon: str,
    family: str,
    timeframe: str,
    symbols: list[str],
    source: str,
    feature_profile: dict[str, Any],
    mechanics_profile: dict[str, Any] | None = None,
    history_bars: int,
    dataset_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit le manifeste minimal d'un dataset d'entrainement.

    Args:
        horizon (str): Horizon strategique.
        family (str): Famille d'actifs.
        timeframe (str): Timeframe principal.
        symbols (list[str]): Univers exact.
        source (str): Source historique retenue.
        feature_profile (dict[str, Any]): Profil de features applique.
        mechanics_profile (dict[str, Any] | None): Profil de mecanique
            de position applique au run.
        history_bars (int): Fenetre historique effectivement chargee.
        dataset_coverage (dict[str, Any] | None): Couverture constatee
            sur les sources de donnees.

    Returns:
        dict[str, Any]: Descripteur stable pour les endpoints et rapports.
    """
    profile_name = str(feature_profile.get("profile_name") or "unknown")
    mechanics_profile_name = str((mechanics_profile or {}).get("profile_name") or "unknown")
    dataset_id = build_dataset_id(
        horizon=horizon,
        family=family,
        timeframe=timeframe,
        symbols=symbols,
        source=source,
        feature_profile=f"{profile_name}:{mechanics_profile_name}",
        history_bars=history_bars,
    )
    return {
        "dataset_id": dataset_id,
        "source": str(source).lower(),
        "horizon": str(horizon).lower(),
        "family": resolve_model_family(family=family),
        "timeframe": str(timeframe).upper(),
        "symbols": list(symbols),
        "symbols_count": len(symbols),
        "history_bars": int(history_bars),
        "feature_profile": profile_name,
        "feature_version": feature_profile.get("feature_version"),
        "mechanics_profile": mechanics_profile_name,
        "mechanics_profile_version": (mechanics_profile or {}).get("profile_version"),
        "entry_features": list(feature_profile.get("entry_features") or []),
        "audit_features": list(feature_profile.get("audit_features") or []),
        "dataset_coverage": dict(dataset_coverage or {}),
    }


def resolve_family_horizon_symbols(horizon: str, family: str) -> list[str]:
    """Retourne les symboles a utiliser pour un couple horizon/famille.

    Args:
        horizon (str): Horizon strategique demande.
        family (str): Famille d'actifs ciblee.

    Returns:
        list[str]: Symboles retenus pour le run cible.
    """
    supported_horizons = {"scalp", "intraday", "swing"}
    normalized_horizon = str(horizon).lower()
    if normalized_horizon not in supported_horizons:
        logger.warning("Horizon inconnu pour la resolution famille: %s", horizon)
        return []
    return get_model_family_symbols(family)


def resolve_family_training_symbols(
    horizon: str,
    family: str,
    data_dir: str | os.PathLike[str] | None = None,
    max_symbols: int = 0,
) -> list[str]:
    """Retourne les symboles exploitables pour une famille et un horizon.

    Args:
        horizon (str): Horizon strategique cible.
        family (str): Famille d'actifs ciblee.
        data_dir (str | os.PathLike[str] | None): Dossier d'historiques local.
        max_symbols (int): Limite optionnelle de symboles.

    Returns:
        list[str]: Symboles entrainables, tries et filtres par timeframe.
    """
    required_timeframe = get_horizon_timeframe(horizon)
    family_symbols = get_model_family_symbols(family)
    if not family_symbols:
        return []

    inventory = discover_history_inventory(data_dir)
    eligible = [
        symbol
        for symbol in family_symbols
        if symbol in inventory and _can_build_timeframe(inventory.get(symbol, set()), required_timeframe)
    ]
    if max_symbols > 0:
        return eligible[:max_symbols]
    return eligible


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
    inventory: dict[str, set[str]] = {}
    try:
        from eva_lab.timescale_store import discover_timescale_inventory

        timescale_inventory = discover_timescale_inventory()
        for symbol, timeframes in timescale_inventory.items():
            inventory.setdefault(symbol, set()).update(set(timeframes))
    except Exception as exc:
        logger.debug("Inventaire TimeDB ignore: %s", exc)

    history_dir = get_history_dir(data_dir)
    if not history_dir.exists():
        if inventory:
            return inventory
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
    override_env_names: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Retourne l'univers d'actifs effectivement entrainable.

    Args:
        data_dir (str | os.PathLike[str] | None): Dossier des CSV.
        required_timeframes (set[str] | None): Timeframes necessaires.
        max_symbols (int): Nombre max de symboles retournes. ``0`` desactive la limite.
        override_env_names (list[str] | tuple[str, ...] | None): Variables
            d'environnement a verifier avant l'inventaire automatique.

    Returns:
        list[str]: Liste triee de symboles exploitables.
    """
    env_candidates = list(override_env_names or [])
    if "TRAINING_SYMBOLS" not in env_candidates:
        env_candidates.append("TRAINING_SYMBOLS")
    manual_symbols, source_env = resolve_symbol_overrides(env_candidates)
    if manual_symbols:
        if source_env:
            logger.info(
                "Univers d'entrainement force via %s (%s symboles).",
                source_env,
                len(manual_symbols),
            )
        return manual_symbols[:max_symbols] if max_symbols > 0 else manual_symbols

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
    try:
        from eva_lab.timescale_store import load_history_frame_from_timescale

        frame = load_history_frame_from_timescale(
            symbol=symbol,
            timeframe=timeframe,
            limit=get_timeframe_history_bars(timeframe),
        )
        if frame is not None:
            return frame
    except Exception as exc:
        logger.debug("Lecture TimeDB ignoree pour %s %s: %s", symbol, timeframe, exc)

    history_dir = get_history_dir(data_dir)
    direct_path = history_dir / f"{symbol}_{timeframe}.csv"
    if direct_path.exists():
        frame = _read_history_frame(direct_path)
        frame.attrs["dataset_source"] = f"csv:{direct_path.name}"
        frame.attrs["dataset_timeframe"] = timeframe.upper()
        return frame

    if timeframe == "M5":
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "5min")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame

    if timeframe == "M15":
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "15min")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "15min")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame

    if timeframe == "H1":
        source_path = history_dir / f"{symbol}_M15.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1H")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1H")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1H")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame

    if timeframe == "D1":
        source_path = history_dir / f"{symbol}_H1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1D")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M15.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1D")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1D")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1D")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame

    if timeframe == "W1":
        source_path = history_dir / f"{symbol}_D1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1W")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_H1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1W")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M15.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1W")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M5.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1W")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame
        source_path = history_dir / f"{symbol}_M1.csv"
        if source_path.exists():
            frame = _resample_ohlcv(_read_history_frame(source_path), "1W")
            frame.attrs["dataset_source"] = f"csv_resampled:{source_path.name}"
            frame.attrs["dataset_timeframe"] = timeframe.upper()
            return frame

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
