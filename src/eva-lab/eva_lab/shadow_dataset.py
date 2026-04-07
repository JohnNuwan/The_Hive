"""
Outils de chargement du dataset Shadow Learning et des imports MT5.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
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

DEFAULT_EPISODE_WEIGHTING_PROFILE = {
    "base_weight": 1.0,
    "winner_bonus": 0.15,
    "loser_bonus": 0.35,
    "nemesis_bonus": 0.55,
    "risk_symbol_bonus": 0.25,
    "seed_candidate_bonus": 0.45,
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
    *,
    winner_symbols: Iterable[str] | None = None,
    risk_symbols: Iterable[str] | None = None,
    seed_model_versions: Iterable[str] | None = None,
    seed_checkpoints: Iterable[str] | None = None,
    allowed_symbols: Iterable[str] | None = None,
    max_games: int | None = None,
    weighting_profile: dict[str, float] | None = None,
    include_weighting_summary: bool = False,
) -> list[GameHistory] | tuple[list[GameHistory], dict[str, Any]]:
    """
    Convertit les episodes shadow en parties compatibles avec le replay buffer.

    Args:
        data_dirs (Iterable[str | Path]): Dossiers contenant les fichiers JSONL.
        observation_size (int): Taille attendue du vecteur d'observation.
        action_space_size (int): Taille de l'espace d'actions.
        winner_symbols (Iterable[str] | None): Symboles gagnants de la review.
        risk_symbols (Iterable[str] | None): Symboles a risque de la review.
        seed_model_versions (Iterable[str] | None): Versions seed a surponderer.
        seed_checkpoints (Iterable[str] | None): Checkpoints seed a surponderer.
        allowed_symbols (Iterable[str] | None): Univers autorise pour le run.
        max_games (int | None): Nombre maximal d'episodes retenus.
        weighting_profile (dict[str, float] | None): Profil de ponderation.
        include_weighting_summary (bool): Retourne aussi un resume de ponderation.

    Returns:
        list[GameHistory] | tuple[list[GameHistory], dict[str, Any]]: Jeux
        compatibles replay, avec resume optionnel.
    """
    winner_set = _normalize_symbol_set(winner_symbols)
    risk_set = _normalize_symbol_set(risk_symbols)
    seed_model_set = _normalize_text_set(seed_model_versions)
    seed_checkpoint_set = _normalize_text_set(seed_checkpoints)
    allowed_set = _normalize_symbol_set(allowed_symbols)
    effective_profile = _build_weighting_profile(weighting_profile)

    episodes = load_shadow_episodes(data_dirs)
    if max_games and max_games > 0:
        episodes = episodes[-max_games:]

    games: list[GameHistory] = []
    weighted_episode_counts: Counter[str] = Counter()
    weighted_priority_total = 0.0
    for episode in episodes:
        game = build_game_from_shadow_episode(
            episode,
            observation_size=observation_size,
            action_space_size=action_space_size,
            winner_symbols=winner_set,
            risk_symbols=risk_set,
            seed_model_versions=seed_model_set,
            seed_checkpoints=seed_checkpoint_set,
            allowed_symbols=allowed_set,
            weighting_profile=effective_profile,
        )
        if len(game) <= 0:
            continue
        games.append(game)
        metadata = dict(game.metadata or {})
        for tag in list(metadata.get("episode_tags") or []):
            weighted_episode_counts[str(tag)] += 1
        weighted_priority_total += float(metadata.get("episode_weight") or 0.0)

    if not include_weighting_summary:
        return games
    return games, {
        "review_loaded": bool(winner_set or risk_set or seed_model_set or seed_checkpoint_set),
        "winner_symbols": sorted(winner_set),
        "risk_symbols": sorted(risk_set),
        "seed_model_versions": sorted(seed_model_set),
        "seed_checkpoints": sorted(seed_checkpoint_set),
        "episodes_loaded": len(games),
        "weighted_episode_counts": dict(weighted_episode_counts),
        "weighting_profile": effective_profile,
        "weighted_priority_total": round(weighted_priority_total, 4),
    }


def build_game_from_shadow_episode(
    episode: list[dict[str, Any]],
    observation_size: int,
    action_space_size: int,
    *,
    winner_symbols: Iterable[str] | None = None,
    risk_symbols: Iterable[str] | None = None,
    seed_model_versions: Iterable[str] | None = None,
    seed_checkpoints: Iterable[str] | None = None,
    allowed_symbols: Iterable[str] | None = None,
    weighting_profile: dict[str, float] | None = None,
) -> GameHistory:
    """
    Convertit un episode shadow en ``GameHistory``.

    Args:
        episode (list[dict[str, Any]]): Transitions d'un episode.
        observation_size (int): Taille cible des observations.
        action_space_size (int): Nombre d'actions supportees.
        winner_symbols (Iterable[str] | None): Symboles gagnants de la review.
        risk_symbols (Iterable[str] | None): Symboles a risque de la review.
        seed_model_versions (Iterable[str] | None): Versions seed a prioriser.
        seed_checkpoints (Iterable[str] | None): Checkpoints seed a prioriser.
        allowed_symbols (Iterable[str] | None): Univers autorise du run courant.
        weighting_profile (dict[str, float] | None): Profil de ponderation.

    Returns:
        GameHistory: Episode encode pour MuZero/Dreamer.
    """
    classification = classify_shadow_episode(
        episode,
        winner_symbols=winner_symbols,
        risk_symbols=risk_symbols,
        seed_model_versions=seed_model_versions,
        seed_checkpoints=seed_checkpoints,
        allowed_symbols=allowed_symbols,
        weighting_profile=weighting_profile,
    )
    game = GameHistory()
    game.metadata = classification
    if not classification.get("allowed", True):
        return game

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

        reward = float(transition.get("reward", 0.0) or 0.0)
        obs_vec = build_observation_vector(observation, observation_size)
        game.store(
            obs_vec,
            action_index,
            reward,
            uniform_policy,
            0.0,
            priority=float(classification.get("episode_weight") or 1.0),
        )

    return game


def classify_shadow_episode(
    episode: list[dict[str, Any]],
    *,
    winner_symbols: Iterable[str] | None = None,
    risk_symbols: Iterable[str] | None = None,
    seed_model_versions: Iterable[str] | None = None,
    seed_checkpoints: Iterable[str] | None = None,
    allowed_symbols: Iterable[str] | None = None,
    weighting_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Classe un episode shadow pour la ponderation d'entrainement.

    Args:
        episode (list[dict[str, Any]]): Episode complet a analyser.
        winner_symbols (Iterable[str] | None): Symboles gagnants de la review.
        risk_symbols (Iterable[str] | None): Symboles a risque de la review.
        seed_model_versions (Iterable[str] | None): Versions seed a prioriser.
        seed_checkpoints (Iterable[str] | None): Checkpoints seed a prioriser.
        allowed_symbols (Iterable[str] | None): Univers autorise.
        weighting_profile (dict[str, float] | None): Profil de ponderation.

    Returns:
        dict[str, Any]: Classe de l'episode, poids et etiquettes associees.
    """
    profile = _build_weighting_profile(weighting_profile)
    winner_set = _normalize_symbol_set(winner_symbols)
    risk_set = _normalize_symbol_set(risk_symbols)
    seed_model_set = _normalize_text_set(seed_model_versions)
    seed_checkpoint_set = _normalize_text_set(seed_checkpoints)
    allowed_set = _normalize_symbol_set(allowed_symbols)

    if not episode:
        return {
            "allowed": False,
            "symbol": None,
            "done": False,
            "pnl": 0.0,
            "nemesis_type": None,
            "episode_weight": profile["base_weight"],
            "episode_tags": [],
            "winner_symbols": sorted(winner_set),
            "risk_symbols": sorted(risk_set),
            "seed_model_versions": sorted(seed_model_set),
            "seed_checkpoints": sorted(seed_checkpoint_set),
        }

    last_transition = dict(episode[-1] or {})
    first_transition = dict(episode[0] or {})
    metadata = dict(last_transition.get("metadata") or {})
    symbol = (
        str(last_transition.get("symbol") or metadata.get("symbol") or "").strip().upper()
        or str(first_transition.get("symbol") or (first_transition.get("metadata") or {}).get("symbol") or "").strip().upper()
        or None
    )
    allowed = not allowed_set or bool(symbol and symbol in allowed_set)
    done = bool(last_transition.get("done", False))
    pnl = _safe_float(last_transition.get("reward", last_transition.get("pnl", 0.0)))
    nemesis_type = str(
        metadata.get("nemesis_type")
        or metadata.get("nemesis_type_hint")
        or ""
    ).strip()
    model_version = str(
        metadata.get("model_version")
        or metadata.get("selection")
        or ""
    ).strip()
    checkpoint = str(metadata.get("checkpoint") or "").strip()
    checkpoint_name = Path(checkpoint).name if checkpoint else ""

    episode_tags: list[str] = []
    weight = float(profile["base_weight"])
    if done and pnl > 0.0:
        episode_tags.append("winner_episode")
        weight += float(profile["winner_bonus"])
    if done and pnl < 0.0:
        episode_tags.append("loser_episode")
        weight += float(profile["loser_bonus"])
    if nemesis_type:
        episode_tags.append("nemesis_episode")
        weight += float(profile["nemesis_bonus"])
    if symbol and symbol in risk_set:
        episode_tags.append("risk_symbol_episode")
        weight += float(profile["risk_symbol_bonus"])
    if (
        model_version
        and model_version.lower() in seed_model_set
    ) or (
        checkpoint
        and (
            checkpoint.lower() in seed_checkpoint_set
            or checkpoint_name.lower() in seed_checkpoint_set
        )
    ):
        episode_tags.append("seed_candidate_episode")
        weight += float(profile["seed_candidate_bonus"])
    if not episode_tags:
        episode_tags.append("neutral_episode")

    return {
        "allowed": allowed,
        "symbol": symbol,
        "done": done,
        "pnl": pnl,
        "nemesis_type": nemesis_type or None,
        "episode_weight": round(max(weight, float(profile["base_weight"])), 4),
        "episode_tags": episode_tags,
        "winner_symbols": sorted(winner_set),
        "risk_symbols": sorted(risk_set),
        "seed_model_versions": sorted(seed_model_set),
        "seed_checkpoints": sorted(seed_checkpoint_set),
    }


def summarize_shadow_weighting(
    data_dirs: Iterable[str | Path],
    *,
    winner_symbols: Iterable[str] | None = None,
    risk_symbols: Iterable[str] | None = None,
    seed_model_versions: Iterable[str] | None = None,
    seed_checkpoints: Iterable[str] | None = None,
    allowed_symbols: Iterable[str] | None = None,
    max_episodes: int | None = None,
    weighting_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Resume la ponderation des episodes shadow sans construire le replay complet.

    Args:
        data_dirs (Iterable[str | Path]): Dossiers shadow a analyser.
        winner_symbols (Iterable[str] | None): Symboles gagnants de la review.
        risk_symbols (Iterable[str] | None): Symboles a risque de la review.
        seed_model_versions (Iterable[str] | None): Versions seed a prioriser.
        seed_checkpoints (Iterable[str] | None): Checkpoints seed a prioriser.
        allowed_symbols (Iterable[str] | None): Univers autorise pour le run.
        max_episodes (int | None): Nombre maximal d'episodes analyses.
        weighting_profile (dict[str, float] | None): Profil de ponderation.

    Returns:
        dict[str, Any]: Resume compact des episodes ponderes.
    """
    winner_set = _normalize_symbol_set(winner_symbols)
    risk_set = _normalize_symbol_set(risk_symbols)
    seed_model_set = _normalize_text_set(seed_model_versions)
    seed_checkpoint_set = _normalize_text_set(seed_checkpoints)
    allowed_set = _normalize_symbol_set(allowed_symbols)
    effective_profile = _build_weighting_profile(weighting_profile)
    episodes = load_shadow_episodes(data_dirs)
    if max_episodes and max_episodes > 0:
        episodes = episodes[-max_episodes:]

    counts: Counter[str] = Counter()
    weighted_priority_total = 0.0
    episodes_loaded = 0
    for episode in episodes:
        classification = classify_shadow_episode(
            episode,
            winner_symbols=winner_set,
            risk_symbols=risk_set,
            seed_model_versions=seed_model_set,
            seed_checkpoints=seed_checkpoint_set,
            allowed_symbols=allowed_set,
            weighting_profile=effective_profile,
        )
        if not classification.get("allowed", True):
            continue
        episodes_loaded += 1
        for tag in list(classification.get("episode_tags") or []):
            counts[str(tag)] += 1
        weighted_priority_total += float(classification.get("episode_weight") or 0.0)

    return {
        "review_loaded": bool(winner_set or risk_set or seed_model_set or seed_checkpoint_set),
        "winner_symbols": sorted(winner_set),
        "risk_symbols": sorted(risk_set),
        "seed_model_versions": sorted(seed_model_set),
        "seed_checkpoints": sorted(seed_checkpoint_set),
        "episodes_loaded": episodes_loaded,
        "weighted_episode_counts": dict(counts),
        "weighting_profile": effective_profile,
        "weighted_priority_total": round(weighted_priority_total, 4),
    }


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


def _build_weighting_profile(profile: dict[str, float] | None) -> dict[str, float]:
    """
    Normalise un profil de ponderation des episodes shadow.

    Args:
        profile (dict[str, float] | None): Profil brut eventuel.

    Returns:
        dict[str, float]: Profil exploitable et borne.
    """
    merged = dict(DEFAULT_EPISODE_WEIGHTING_PROFILE)
    for key, value in dict(profile or {}).items():
        try:
            merged[key] = max(float(value), 0.0)
        except (TypeError, ValueError):
            continue
    return merged


def _normalize_symbol_set(symbols: Iterable[str] | None) -> set[str]:
    """
    Normalise une liste de symboles en majuscules sans doublons.

    Args:
        symbols (Iterable[str] | None): Symboles bruts.

    Returns:
        set[str]: Ensemble nettoye.
    """
    normalized: set[str] = set()
    for symbol in symbols or []:
        candidate = str(symbol or "").strip().upper()
        if candidate:
            normalized.add(candidate)
    return normalized


def _normalize_text_set(values: Iterable[str] | None) -> set[str]:
    """
    Normalise une liste generique de textes en minuscules sans doublons.

    Args:
        values (Iterable[str] | None): Valeurs brutes a nettoyer.

    Returns:
        set[str]: Ensemble nettoye.
    """
    normalized: set[str] = set()
    for value in values or []:
        candidate = str(value or "").strip().lower()
        if candidate:
            normalized.add(candidate)
    return normalized


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
