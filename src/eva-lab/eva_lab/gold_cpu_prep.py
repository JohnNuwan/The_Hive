"""Utilitaires CPU pour preparer les artefacts Gold Monday."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eva_lab.muzero.replay_buffer import GameHistory
from eva_lab.shadow_dataset import load_shadow_games
from eva_lab.training_utils import MTF_HORIZONS, build_inventory_report, load_history_frame
from shared.indicators import IndicatorFactory

logger = logging.getLogger(__name__)

DEFAULT_CPU_PREP_DIR = Path(
    os.getenv("GOLD_CPU_PREP_DIR", "data/checkpoints/gold_cpu_prep")
)


def resolve_cpu_prep_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Retourne le dossier racine des artefacts CPU.

    Args:
        cache_dir (str | os.PathLike[str] | None): Dossier cible optionnel.

    Returns:
        Path: Dossier cree si necessaire.
    """
    target = Path(cache_dir) if cache_dir else DEFAULT_CPU_PREP_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _slugify_token(value: str) -> str:
    """Normalise un token pour un nom de fichier.

    Args:
        value (str): Valeur brute.

    Returns:
        str: Valeur nettoyee et stable.
    """
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_") or "na"


def _fingerprint(payload: dict[str, Any]) -> str:
    """Construit une empreinte courte et stable.

    Args:
        payload (dict[str, Any]): Signature a hacher.

    Returns:
        str: Empreinte SHA1 tronquee.
    """
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()[:12]


def _sorted_symbols(symbols: list[str]) -> list[str]:
    """Trie et nettoie une liste de symboles.

    Args:
        symbols (list[str]): Symboles bruts.

    Returns:
        list[str]: Symboles uniques et tries.
    """
    return sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})


def resolve_gnn_dataset_cache_path(
    symbols: list[str],
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Construit le chemin de cache du dataset GNN.

    Args:
        symbols (list[str]): Univers GNN cible.
        cache_dir (str | os.PathLike[str] | None): Dossier cible optionnel.

    Returns:
        Path: Chemin du cache pickle.
    """
    ordered_symbols = _sorted_symbols(symbols)
    signature = {
        "kind": "gnn_dataset",
        "symbols": ordered_symbols,
        "horizons": {
            timeframe: {
                "count": int(cfg["count"]),
                "seq_len": int(cfg["seq_len"]),
                "future": int(cfg["future"]),
            }
            for timeframe, cfg in sorted(MTF_HORIZONS.items())
        },
        "version": 1,
    }
    label = "_".join(_slugify_token(symbol) for symbol in ordered_symbols) or "empty"
    filename = f"gnn_dataset_{label}_{_fingerprint(signature)}.pkl"
    return resolve_cpu_prep_dir(cache_dir) / filename


def get_gnn_label(current_price: float, future_price: float, atr: float) -> int:
    """Retourne le label de tendance future d'un echantillon GNN.

    Args:
        current_price (float): Prix courant.
        future_price (float): Prix futur de reference.
        atr (float): Volatilite locale.

    Returns:
        int: Label discret ``0/1/2``.
    """
    delta = future_price - current_price
    threshold = max(atr * 0.4, current_price * 0.0005)
    if delta > threshold:
        return 0
    if delta < -threshold:
        return 1
    return 2


def compute_gnn_feature_sequences(
    frame: pd.DataFrame,
    seq_len: int,
    future_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les sequences de features GNN et leurs labels.

    Args:
        frame (pd.DataFrame): Historique source enrichissable.
        seq_len (int): Longueur de sequence.
        future_n (int): Horizon de prediction.

    Returns:
        tuple[np.ndarray, np.ndarray]: Features et labels synchronises.
    """
    closes = frame["close"].astype(float)
    highs = frame["high"].astype(float)
    lows = frame["low"].astype(float)
    volumes = frame["tick_volume"].astype(float)
    opens = frame["open"].astype(float)

    rsi = IndicatorFactory.rsi(closes, 14)
    adx = IndicatorFactory.adx(highs, lows, closes, 14)["adx"]
    vwap = IndicatorFactory.vwap(highs, lows, closes, volumes)
    macd_hist = IndicatorFactory.macd(closes)["histogram"]
    atr = IndicatorFactory.atr(highs, lows, closes, 14)
    bb_pct = IndicatorFactory.bollinger_bands(closes)["pct_b"]

    features: list[list[list[float]]] = []
    labels: list[int] = []
    start_idx = max(50, seq_len)

    for current_idx in range(start_idx, len(frame) - future_n):
        sequence_rows: list[list[float]] = []
        start_seq = current_idx - seq_len + 1
        for idx in range(start_seq, current_idx + 1):
            price = float(closes.iloc[idx])
            previous_price = float(closes.iloc[idx - 1]) if idx > 0 else price
            high_price = float(highs.iloc[idx])
            low_price = float(lows.iloc[idx])
            open_price = float(opens.iloc[idx])
            avg_volume = float(volumes.iloc[max(0, idx - 10):idx].mean()) if idx > 0 else 0.0

            row = [
                (price / previous_price) - 1.0 if previous_price else 0.0,
                float(rsi.iloc[idx]) / 100.0 if not np.isnan(rsi.iloc[idx]) else 0.5,
                float(adx.iloc[idx]) / 100.0 if not np.isnan(adx.iloc[idx]) else 0.2,
                float(macd_hist.iloc[idx]) / price if (not np.isnan(macd_hist.iloc[idx]) and price) else 0.0,
                float(bb_pct.iloc[idx]) if not np.isnan(bb_pct.iloc[idx]) else 0.5,
                float(volumes.iloc[idx]) / (avg_volume + 1e-5) if avg_volume else 1.0,
                (price - float(vwap.iloc[idx])) / price if (not np.isnan(vwap.iloc[idx]) and price) else 0.0,
                float(atr.iloc[idx]) / price if (not np.isnan(atr.iloc[idx]) and price) else 0.0,
                (price - low_price) / (high_price - low_price + 1e-8),
                (high_price - max(price, open_price)) / price if price else 0.0,
                (min(price, open_price) - low_price) / price if price else 0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            sequence_rows.append(row)

        future_price = float(closes.iloc[current_idx + future_n])
        current_price = float(closes.iloc[current_idx])
        atr_value = (
            float(atr.iloc[current_idx])
            if not np.isnan(atr.iloc[current_idx])
            else current_price * 0.001
        )
        labels.append(get_gnn_label(current_price, future_price, atr_value))
        features.append(sequence_rows)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def build_gnn_dataset(symbols: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Assemble le dataset GNN multi-actifs et multi-timeframes.

    Args:
        symbols (list[str]): Univers cible.

    Returns:
        tuple[dict[str, Any], list[str]]: Dataset serialisable et symboles valides.
    """
    dataset: dict[str, Any] = {}
    valid_symbols: list[str] = []

    for symbol in symbols:
        horizon_payload: dict[str, Any] = {}
        symbol_valid = True

        for timeframe, cfg in MTF_HORIZONS.items():
            frame = load_history_frame(symbol, timeframe)
            if frame is None:
                logger.warning("Historique absent pour %s sur %s.", symbol, timeframe)
                symbol_valid = False
                break

            clipped = frame.tail(int(cfg["count"])).copy()
            features, labels = compute_gnn_feature_sequences(
                clipped,
                int(cfg["seq_len"]),
                int(cfg["future"]),
            )
            if len(labels) < 64:
                logger.warning(
                    "Pas assez d'echantillons pour %s sur %s (%s).",
                    symbol,
                    timeframe,
                    len(labels),
                )
                symbol_valid = False
                break

            horizon_payload[timeframe] = {
                "features": features,
                "labels": labels,
            }

        if symbol_valid:
            dataset[symbol] = horizon_payload
            valid_symbols.append(symbol)
            logger.info(
                "Dataset GNN %s pret: M5=%s | H1=%s | D1=%s",
                symbol,
                len(horizon_payload["M5"]["labels"]),
                len(horizon_payload["H1"]["labels"]),
                len(horizon_payload["D1"]["labels"]),
            )

    return dataset, valid_symbols


def save_gnn_dataset_cache(
    *,
    symbols: list[str],
    dataset: dict[str, Any],
    valid_symbols: list[str],
    inventory: dict[str, list[str]],
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Persiste un cache CPU du dataset GNN.

    Args:
        symbols (list[str]): Univers demande.
        dataset (dict[str, Any]): Dataset serialisable.
        valid_symbols (list[str]): Symboles effectivement retenus.
        inventory (dict[str, list[str]]): Inventaire historique.
        cache_dir (str | os.PathLike[str] | None): Dossier cible optionnel.

    Returns:
        Path: Chemin final du cache.
    """
    cache_path = resolve_gnn_dataset_cache_path(symbols=symbols, cache_dir=cache_dir)
    payload = {
        "kind": "gnn_dataset",
        "requested_symbols": _sorted_symbols(symbols),
        "valid_symbols": list(valid_symbols),
        "inventory": inventory,
        "dataset": dataset,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    with cache_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    return cache_path


def load_gnn_dataset_cache(
    symbols: list[str],
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Charge un cache CPU GNN s'il existe.

    Args:
        symbols (list[str]): Univers demande.
        cache_dir (str | os.PathLike[str] | None): Dossier cible optionnel.

    Returns:
        dict[str, Any] | None: Payload du cache si valide.
    """
    cache_path = resolve_gnn_dataset_cache_path(symbols=symbols, cache_dir=cache_dir)
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)
    if list(payload.get("requested_symbols") or []) != _sorted_symbols(symbols):
        return None
    payload["cache_path"] = str(cache_path)
    return payload


def _compute_dreamer_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs utilises pour le pre-entrainement Dreamer.

    Args:
        frame (pd.DataFrame): Historique brut OHLCV.

    Returns:
        pd.DataFrame: Historique enrichi.
    """
    enriched = frame.copy()
    enriched["rsi"] = IndicatorFactory.rsi(enriched["close"], 14)
    macd_res = IndicatorFactory.macd(enriched["close"])
    enriched["macd"] = macd_res["macd"]
    enriched["macd_signal"] = macd_res["signal"]
    enriched["macd_hist"] = macd_res["histogram"]
    enriched["vwap"] = IndicatorFactory.vwap(
        enriched["high"],
        enriched["low"],
        enriched["close"],
        enriched["tick_volume"],
    )
    enriched["obv"] = IndicatorFactory.obv(enriched["close"], enriched["tick_volume"])
    enriched["momentum"] = IndicatorFactory.momentum(enriched["close"])
    enriched["trix"] = IndicatorFactory.trix(enriched["close"])
    stoch_res = IndicatorFactory.stochastic(
        enriched["high"],
        enriched["low"],
        enriched["close"],
    )
    enriched["stoch_k"] = stoch_res["percent_k"]
    enriched["stoch_d"] = stoch_res["percent_d"]
    enriched["cci"] = IndicatorFactory.cci(
        enriched["high"],
        enriched["low"],
        enriched["close"],
    )
    adx_res = IndicatorFactory.adx(
        enriched["high"],
        enriched["low"],
        enriched["close"],
    )
    enriched["adx"] = adx_res["adx"]
    enriched["adx_plus_di"] = adx_res["plus_di"]
    enriched["adx_minus_di"] = adx_res["minus_di"]
    ichi_res = IndicatorFactory.ichimoku(
        enriched["high"],
        enriched["low"],
        enriched["close"],
    )
    enriched["ichi_tenkan"] = ichi_res["tenkan_sen"]
    enriched["ichi_kijun"] = ichi_res["kijun_sen"]
    enriched["ichi_senkou_a"] = ichi_res["senkou_span_a"]
    enriched["ichi_senkou_b"] = ichi_res["senkou_span_b"]
    return enriched.bfill().fillna(0.0)


def _build_dreamer_history_games(
    *,
    symbol: str,
    frame: pd.DataFrame,
    sequence_length: int,
    sequence_stride: int,
    observation_size: int,
    action_space_size: int,
    rng_seed: int,
) -> tuple[list[GameHistory], int, dict[str, int]]:
    """Construit les episodes Dreamer issus d'un historique marche.

    Args:
        symbol (str): Symbole en cours.
        frame (pd.DataFrame): Historique brut.
        sequence_length (int): Longueur de sequence.
        sequence_stride (int): Pas entre deux sequences.
        observation_size (int): Taille du vecteur d'observation.
        action_space_size (int): Taille de l'espace d'actions.
        rng_seed (int): Graine locale stable.

    Returns:
        tuple[list[GameHistory], int, dict[str, int]]: Episodes, pas de temps
            et compteurs d'actions.
    """
    enriched = _compute_dreamer_indicators(frame)
    closes_seg = enriched["close"].values
    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    games: list[GameHistory] = []
    total_steps = 0
    rng = np.random.default_rng(rng_seed)

    for start_idx in range(0, len(enriched) - sequence_length, sequence_stride):
        end_idx = start_idx + sequence_length
        if end_idx > len(enriched):
            break

        seg_closes = closes_seg[start_idx:end_idx]
        actions = rng.choice([0, 1, 2], size=sequence_length, p=[0.35, 0.325, 0.325])
        game = GameHistory()
        initial_balance = 10000.0
        balance = initial_balance
        peak_balance = initial_balance
        position = 0
        entry_price = 0.0

        for index_in_segment in range(sequence_length):
            idx = start_idx + index_in_segment
            price = float(seg_closes[index_in_segment])
            obs_vec = np.zeros((observation_size,), dtype=np.float32)
            obs_vec[0] = price / 3000.0
            obs_vec[1] = float(enriched["rsi"].values[idx]) / 100.0
            features_list = [
                enriched["rsi"].values[idx],
                enriched["macd_hist"].values[idx],
                enriched["macd_signal"].values[idx],
                enriched["vwap"].values[idx],
                enriched["obv"].values[idx] / 10000.0,
                enriched["momentum"].values[idx],
                enriched["trix"].values[idx],
                enriched["stoch_k"].values[idx],
                enriched["stoch_d"].values[idx],
                enriched["cci"].values[idx],
                enriched["adx"].values[idx],
                enriched["adx_plus_di"].values[idx],
                enriched["adx_minus_di"].values[idx],
                enriched["ichi_tenkan"].values[idx],
                enriched["ichi_kijun"].values[idx],
                enriched["ichi_senkou_a"].values[idx],
                enriched["ichi_senkou_b"].values[idx],
            ]
            for feature_index, feature_value in enumerate(features_list):
                if feature_index + 2 < observation_size:
                    obs_vec[feature_index + 2] = float(feature_value)

            action_val = int(actions[index_in_segment])
            reward = 0.0
            if action_val == 1:
                action_counts["BUY"] += 1
            elif action_val == 2:
                action_counts["SELL"] += 1
            else:
                action_counts["HOLD"] += 1

            if index_in_segment < sequence_length - 1:
                next_price = float(seg_closes[index_in_segment + 1])
                ret = (next_price - price) / max(price, 1e-9) * 100
                if action_val == 1:
                    reward = ret - 0.02
                    if position == 0:
                        position = 1
                        entry_price = price
                elif action_val == 2:
                    reward = -ret - 0.02
                    if position == 0:
                        position = -1
                        entry_price = price
                elif action_val == 0 and position != 0:
                    trade_pnl = (
                        (price - entry_price) / max(entry_price, 1e-9) * 100
                        if position == 1
                        else (entry_price - price) / max(entry_price, 1e-9) * 100
                    )
                    balance += balance * trade_pnl / 100
                    position = 0

                peak_balance = max(peak_balance, balance)
                drawdown_pct = (peak_balance - balance) / max(peak_balance, 1e-9) * 100
                if drawdown_pct >= 4.0:
                    reward -= 15.0

            action_one_hot = np.zeros((action_space_size,), dtype=np.float32)
            if 0 <= action_val < action_space_size:
                action_one_hot[action_val] = 1.0
            game.store(obs_vec, action_one_hot, reward, [1 / 3] * 3, 0.0)

        games.append(game)
        total_steps += sequence_length

    logger.info(
        "Cache Dreamer pret pour %s: %s episodes, %s pas.",
        symbol,
        len(games),
        total_steps,
    )
    return games, total_steps, action_counts


def resolve_dreamer_replay_cache_path(
    *,
    horizon: str,
    family: str,
    symbols: list[str],
    sequence_length: int,
    sequence_stride: int,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Construit le chemin de cache du replay Dreamer.

    Args:
        horizon (str): Horizon cible.
        family (str): Famille cible.
        symbols (list[str]): Symboles cibles.
        sequence_length (int): Longueur de sequence.
        sequence_stride (int): Pas entre sequences.
        cache_dir (str | os.PathLike[str] | None): Dossier cible optionnel.

    Returns:
        Path: Chemin du cache pickle.
    """
    ordered_symbols = _sorted_symbols(symbols)
    signature = {
        "kind": "dreamer_replay",
        "horizon": str(horizon).lower(),
        "family": str(family).lower(),
        "symbols": ordered_symbols,
        "sequence_length": int(sequence_length),
        "sequence_stride": int(sequence_stride),
        "version": 1,
    }
    label = "_".join(_slugify_token(symbol) for symbol in ordered_symbols) or "empty"
    filename = (
        f"dreamer_replay_{_slugify_token(horizon)}_{_slugify_token(family)}_"
        f"len{int(sequence_length)}_stride{int(sequence_stride)}_{label}_{_fingerprint(signature)}.pkl"
    )
    return resolve_cpu_prep_dir(cache_dir) / filename


def build_dreamer_replay_cache_payload(
    *,
    symbols: list[str],
    horizon: str,
    family: str,
    data_dir: str | os.PathLike[str] | None,
    shadow_data_dirs: list[str],
    observation_size: int,
    action_space_size: int,
    sequence_length: int,
    sequence_stride: int,
) -> dict[str, Any]:
    """Construit le replay Dreamer serialisable pour Monday Gold.

    Args:
        symbols (list[str]): Symboles cibles.
        horizon (str): Horizon cible.
        family (str): Famille cible.
        data_dir (str | os.PathLike[str] | None): Dossier CSV de secours.
        shadow_data_dirs (list[str]): Dossiers shadow.
        observation_size (int): Taille de l'observation.
        action_space_size (int): Taille de l'espace d'actions.
        sequence_length (int): Longueur de sequence.
        sequence_stride (int): Pas entre sequences.

    Returns:
        dict[str, Any]: Payload complet de cache.
    """
    ordered_symbols = _sorted_symbols(symbols)
    games: list[GameHistory] = []
    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    total_steps = 0
    history_games = 0

    for index, symbol in enumerate(ordered_symbols):
        frame = load_history_frame(symbol, "M5", data_dir=data_dir)
        if frame is None or frame.empty:
            logger.warning("Historique Dreamer absent pour %s sur M5.", symbol)
            continue
        symbol_games, symbol_steps, symbol_counts = _build_dreamer_history_games(
            symbol=symbol,
            frame=frame,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            observation_size=observation_size,
            action_space_size=action_space_size,
            rng_seed=1000 + index * 97 + sequence_length * 13 + sequence_stride,
        )
        games.extend(symbol_games)
        total_steps += symbol_steps
        history_games += len(symbol_games)
        for action_name, count in symbol_counts.items():
            action_counts[action_name] += int(count)

    shadow_games = load_shadow_games(
        shadow_data_dirs,
        observation_size=observation_size,
        action_space_size=action_space_size,
    )
    valid_shadow_games = [game for game in shadow_games if len(game) >= sequence_length]
    games.extend(valid_shadow_games)
    total_steps += sum(len(game) for game in valid_shadow_games)

    return {
        "kind": "dreamer_replay",
        "horizon": str(horizon).lower(),
        "family": str(family).lower(),
        "requested_symbols": ordered_symbols,
        "sequence_length": int(sequence_length),
        "sequence_stride": int(sequence_stride),
        "observation_size": int(observation_size),
        "action_space_size": int(action_space_size),
        "history_games": int(history_games),
        "shadow_games": int(len(valid_shadow_games)),
        "total_steps": int(total_steps),
        "action_counts": action_counts,
        "games": games,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }


def save_dreamer_replay_cache(
    payload: dict[str, Any],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Persiste un replay Dreamer preconstruit.

    Args:
        payload (dict[str, Any]): Payload a serialiser.
        cache_dir (str | os.PathLike[str] | None): Dossier cible optionnel.

    Returns:
        Path: Chemin final du cache.
    """
    cache_path = resolve_dreamer_replay_cache_path(
        horizon=str(payload.get("horizon") or "scalp"),
        family=str(payload.get("family") or "mixed"),
        symbols=list(payload.get("requested_symbols") or []),
        sequence_length=int(payload.get("sequence_length") or 0),
        sequence_stride=int(payload.get("sequence_stride") or 0),
        cache_dir=cache_dir,
    )
    with cache_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    return cache_path


def load_dreamer_replay_cache(
    *,
    horizon: str,
    family: str,
    symbols: list[str],
    sequence_length: int,
    sequence_stride: int,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Charge un cache replay Dreamer s'il existe.

    Args:
        horizon (str): Horizon cible.
        family (str): Famille cible.
        symbols (list[str]): Symboles cibles.
        sequence_length (int): Longueur de sequence.
        sequence_stride (int): Pas entre sequences.
        cache_dir (str | os.PathLike[str] | None): Dossier cible optionnel.

    Returns:
        dict[str, Any] | None: Payload du cache si valide.
    """
    cache_path = resolve_dreamer_replay_cache_path(
        horizon=horizon,
        family=family,
        symbols=symbols,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        cache_dir=cache_dir,
    )
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)
    if list(payload.get("requested_symbols") or []) != _sorted_symbols(symbols):
        return None
    payload["cache_path"] = str(cache_path)
    return payload


def build_default_gnn_cache_payload(symbols: list[str]) -> tuple[dict[str, Any], Path]:
    """Construit et persiste le cache GNN pour un univers cible.

    Args:
        symbols (list[str]): Univers GNN cible.

    Returns:
        tuple[dict[str, Any], Path]: Payload construit et chemin final.
    """
    dataset, valid_symbols = build_gnn_dataset(symbols)
    inventory = build_inventory_report()
    cache_path = save_gnn_dataset_cache(
        symbols=symbols,
        dataset=dataset,
        valid_symbols=valid_symbols,
        inventory=inventory,
    )
    payload = load_gnn_dataset_cache(symbols) or {}
    return payload, cache_path
