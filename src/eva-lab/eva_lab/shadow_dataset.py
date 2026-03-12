"""
Outils de chargement du dataset Shadow Learning et des imports MT5.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eva_lab.muzero.replay_buffer import GameHistory


ACTION_MAP = {
    "HOLD": 0,
    "BUY": 1,
    "SELL": 2,
    "SPLIT": 3,
    "CLOSE": 4,
}


def resolve_shadow_files(data_dirs: Iterable[str | Path]) -> list[Path]:
    """
    Liste recursivement les fichiers JSONL exploitables pour le Shadow Learning.

    Args:
        data_dirs (Iterable[str | Path]): Dossiers racines a inspecter.

    Returns:
        list[Path]: Fichiers trouves, tries de facon stable.
    """
    files: list[Path] = []
    for data_dir in data_dirs:
        root = Path(data_dir)
        if not root.exists():
            continue
        files.extend(sorted(path for path in root.rglob("*.jsonl") if path.is_file()))
    return sorted(files)


def load_shadow_episodes(data_dirs: Iterable[str | Path]) -> list[list[dict[str, Any]]]:
    """
    Charge et groupe les transitions shadow en episodes coherents.

    Args:
        data_dirs (Iterable[str | Path]): Dossiers a charger.

    Returns:
        list[list[dict[str, Any]]]: Episodes tries chronologiquement.
    """
    grouped: dict[str, list[tuple[datetime, int, int, dict[str, Any]]]] = defaultdict(list)
    files = resolve_shadow_files(data_dirs)

    for file_index, path in enumerate(files):
        with path.open("r", encoding="utf-8") as file_obj:
            for line_index, raw_line in enumerate(file_obj):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                metadata = payload.get("metadata") or {}
                timestamp = _parse_timestamp(payload.get("timestamp"))
                episode_id = (
                    str(metadata.get("episode_id") or "").strip()
                    or str(metadata.get("position_id") or "").strip()
                    or str(metadata.get("position_ticket") or "").strip()
                    or str(metadata.get("ticket") or "").strip()
                    or f"{path.name}:{line_index}"
                )
                grouped[episode_id].append((timestamp, file_index, line_index, payload))

    ordered_episodes: list[list[dict[str, Any]]] = []
    for items in grouped.values():
        items.sort(key=lambda item: (item[0], item[1], item[2]))
        ordered_episodes.append([payload for _, _, _, payload in items])

    ordered_episodes.sort(
        key=lambda episode: _parse_timestamp((episode[0].get("timestamp") if episode else None))
    )
    return ordered_episodes


def load_shadow_games(
    data_dirs: Iterable[str | Path],
    observation_size: int,
    action_space_size: int,
) -> list[GameHistory]:
    """
    Convertit les episodes shadow en parties compatibles avec le replay buffer.

    Args:
        data_dirs (Iterable[str | Path]): Dossiers contenant les fichiers JSONL.
        observation_size (int): Taille attendue du vecteur d'observation.
        action_space_size (int): Taille de l'espace d'actions.

    Returns:
        list[GameHistory]: Parties prêtes a etre injectees dans le replay buffer.
    """
    games: list[GameHistory] = []
    for episode in load_shadow_episodes(data_dirs):
        game = build_game_from_shadow_episode(
            episode,
            observation_size=observation_size,
            action_space_size=action_space_size,
        )
        if len(game) > 0:
            games.append(game)
    return games


def build_game_from_shadow_episode(
    episode: list[dict[str, Any]],
    observation_size: int,
    action_space_size: int,
) -> GameHistory:
    """
    Convertit un episode shadow en ``GameHistory``.

    Args:
        episode (list[dict[str, Any]]): Transitions d'un episode.
        observation_size (int): Taille cible des observations.
        action_space_size (int): Nombre d'actions supportees.

    Returns:
        GameHistory: Episode encode pour MuZero/Dreamer.
    """
    game = GameHistory()
    uniform_policy = np.full(
        action_space_size,
        1.0 / max(action_space_size, 1),
        dtype=np.float32,
    )

    for transition in episode:
        observation = transition.get("observation", {}) or {}
        action_payload = transition.get("action", {}) or {}
        action_name = str(action_payload.get("type", "HOLD")).upper()
        action_index = ACTION_MAP.get(action_name, 0)

        action_one_hot = np.zeros(action_space_size, dtype=np.float32)
        if 0 <= action_index < action_space_size:
            action_one_hot[action_index] = 1.0

        reward = float(transition.get("reward", 0.0) or 0.0)
        obs_vec = build_observation_vector(observation, observation_size)
        game.store(obs_vec, action_one_hot, reward, uniform_policy, 0.0)

    return game


def build_observation_vector(observation: dict[str, Any], observation_size: int) -> np.ndarray:
    """
    Convertit une observation shadow en vecteur numerique aligne avec MuZero.

    Args:
        observation (dict[str, Any]): Observation source.
        observation_size (int): Taille cible du vecteur.

    Returns:
        np.ndarray: Vecteur ``float32`` normalise.
    """
    candle = observation.get("latest_candle", {}) or {}
    indicators = observation.get("indicators", {}) or {}

    price = _safe_float(observation.get("price", candle.get("close", 0.0)))
    close_price = _safe_float(candle.get("close", price), fallback=price)
    open_price = _safe_float(candle.get("open", close_price), fallback=close_price)
    high_price = _safe_float(candle.get("high", close_price), fallback=close_price)
    low_price = _safe_float(candle.get("low", close_price), fallback=close_price)
    volume = _safe_float(candle.get("tick_volume", candle.get("volume", 0.0)))
    spread = _safe_float(candle.get("spread", 0.0))

    obs_vec = np.zeros(observation_size, dtype=np.float32)
    base_values = np.array(
        [
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            _indicator(indicators, "EMA_200", fallback=close_price),
            _indicator(indicators, "RSI", fallback=50.0),
            _indicator(indicators, "MACD_Hist", fallback=0.0),
            _indicator(indicators, "VWAP", fallback=close_price),
            _indicator(indicators, "OBV", fallback=0.0),
            _indicator(indicators, "Momentum", fallback=0.0),
            _indicator(indicators, "TRIX", fallback=0.0),
            _indicator(indicators, "Stoch_K", fallback=50.0),
            _indicator(indicators, "Stoch_D", fallback=50.0),
            _indicator(indicators, "CCI", fallback=0.0),
            _indicator(indicators, "ADX", fallback=0.0),
            _indicator(indicators, "ADX_Plus_DI", fallback=0.0),
            _indicator(indicators, "ADX_Minus_DI", fallback=0.0),
            _indicator(indicators, "Ichi_Tenkan", fallback=close_price),
            _indicator(indicators, "Ichi_Kijun", fallback=close_price),
            _indicator(indicators, "Ichi_Senkou_A", fallback=close_price),
            _indicator(indicators, "Ichi_Senkou_B", fallback=close_price),
            _indicator(indicators, "ATR", fallback=0.0),
            _indicator(indicators, "BB_Pct", fallback=0.5),
            spread / max(close_price, 1e-8),
            _indicator(indicators, "Return_1", fallback=0.0),
        ],
        dtype=np.float32,
    )
    obs_vec[: min(len(base_values), observation_size)] = base_values[:observation_size]
    return obs_vec


def _indicator(indicators: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    """
    Recupere une valeur d'indicateur avec tolerance sur la casse des cles.

    Args:
        indicators (dict[str, Any]): Mapping des indicateurs.
        key (str): Cle cible.
        fallback (float): Valeur de secours.

    Returns:
        float: Valeur convertie en flottant.
    """
    variants = {
        key,
        key.lower(),
        key.upper(),
        key.replace("_", ""),
        key.replace("_", "").lower(),
    }
    for candidate in variants:
        if candidate in indicators:
            return _safe_float(indicators.get(candidate), fallback=fallback)
    return fallback


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """
    Convertit une valeur arbitraire en flottant.

    Args:
        value (Any): Valeur source.
        fallback (float): Valeur retournee si la conversion echoue.

    Returns:
        float: Valeur convertie.
    """
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_timestamp(value: Any) -> datetime:
    """
    Convertit un timestamp inconnu en objet ``datetime`` comparable.

    Args:
        value (Any): Valeur source.

    Returns:
        datetime: Timestamp normalise.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.min
