"""Entraine MuZero sur historique reel puis execute la selection ADN."""

from __future__ import annotations

import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

package_root = Path(__file__).resolve().parents[1]
shared_root = package_root.parent / 'shared'
for candidate in (package_root, shared_root):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eva_lab.arena import Arena
from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.genetic_updater import GeneticUpdater
from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.checkpoint_utils import (
    archive_muzero_artifacts,
    recommend_muzero_seed_for_v66,
)
from eva_lab.muzero.environment import TradingEnvironment
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.timescale_store import record_arena_result, record_training_dataset
from eva_lab.training_notifier import send_horizon_summary, send_training_horizon_started
from eva_lab.training_status import (
    append_training_log,
    merge_training_status,
    write_arena_summary,
    load_training_status,
    mark_step_running,
    set_gold_precheck,
    write_precheck_summary,
    write_terminal_summary,
)
from eva_lab.training_utils import (
    build_inventory_report,
    build_muzero_market_context,
    get_horizon_history_bars,
    infer_family_from_symbols,
    load_history_frame,
    resolve_model_family,
    resolve_feature_profile,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.train_muzero")


def _resolve_collection_runtime_budget(
    symbol: str,
    config: MuZeroConfigV3,
) -> dict[str, float | int | str]:
    """Resout un budget de collecte adapte a la famille du symbole.

    Les indices deviennent tres couteux avec un MCTS profond. Sans budget
    dedie, la collecte passe surtout son temps a expirer sur `US30.cash`
    avant de produire des episodes exploitables.

    Args:
        symbol (str): Symbole collecte.
        config (MuZeroConfigV3): Configuration MuZero courante.

    Returns:
        dict[str, float | int | str]: Budget effectif de collecte.
    """

    family = resolve_model_family(symbol=symbol)
    simulations = int(
        getattr(config, "collection_num_simulations", config.num_simulations)
        or config.num_simulations
    )
    max_moves = int(
        getattr(config, "collection_max_moves", config.max_moves)
        or config.max_moves
    )
    max_episode_seconds = float(
        getattr(config, "collection_max_episode_seconds", 0.0) or 0.0
    )

    normalized_symbol = str(symbol or "").strip().upper()

    if family == "indices":
        simulations = max(
            32,
            min(
                simulations,
                # int(float(...)) supporte les valeurs flottantes dans .env (ex: "192.0")
                int(float(os.getenv("MUZERO_COLLECTION_NUM_SIMULATIONS_INDICES", "192"))),
            ),
        )
        max_moves = max(
            48,
            min(
                max_moves,
                int(float(os.getenv("MUZERO_COLLECTION_MAX_MOVES_INDICES", "96"))),
            ),
        )
        max_episode_seconds = max(
            120.0,
            min(
                max_episode_seconds,
                float(os.getenv("MUZERO_COLLECTION_MAX_EPISODE_SECONDS_INDICES", "180")),
            ),
        )
    elif family == "metals":
        simulations = max(
            32,
            min(
                simulations,
                int(float(os.getenv("MUZERO_COLLECTION_NUM_SIMULATIONS_METALS", "160"))),
            ),
        )
        max_moves = max(
            48,
            min(
                max_moves,
                int(float(os.getenv("MUZERO_COLLECTION_MAX_MOVES_METALS", "96"))),
            ),
        )
        max_episode_seconds = max(
            120.0,
            min(
                max_episode_seconds,
                float(os.getenv("MUZERO_COLLECTION_MAX_EPISODE_SECONDS_METALS", "180")),
            ),
        )

    if normalized_symbol == "XAUUSD":
        simulations = max(
            16,
            min(
                simulations,
                # Toujours convertir via float() pour éviter ValueError sur les valeurs "128.0"
                int(float(os.getenv("MUZERO_COLLECTION_NUM_SIMULATIONS_XAUUSD", str(simulations)))),
            ),
        )
        max_moves = max(
            32,
            min(
                max_moves,
                int(float(os.getenv("MUZERO_COLLECTION_MAX_MOVES_XAUUSD", str(max_moves)))),
            ),
        )
        max_episode_seconds = max(
            90.0,
            min(
                max_episode_seconds,
                float(
                    os.getenv(
                        "MUZERO_COLLECTION_MAX_EPISODE_SECONDS_XAUUSD",
                        str(max_episode_seconds),
                    )
                ),
            ),
        )

    if normalized_symbol == "EURUSD":
        simulations = max(
            16,
            min(
                simulations,
                int(float(os.getenv("MUZERO_COLLECTION_NUM_SIMULATIONS_EURUSD", str(simulations)))),
            ),
        )
        max_moves = max(
            32,
            min(
                max_moves,
                int(float(os.getenv("MUZERO_COLLECTION_MAX_MOVES_EURUSD", str(max_moves)))),
            ),
        )
        max_episode_seconds = max(
            90.0,
            min(
                max_episode_seconds,
                float(
                    os.getenv(
                        "MUZERO_COLLECTION_MAX_EPISODE_SECONDS_EURUSD",
                        str(max_episode_seconds),
                    )
                ),
            ),
        )

    return {
        "family": family,
        "collection_num_simulations": simulations,
        "collection_max_moves": max_moves,
        "collection_max_episode_seconds": max_episode_seconds,
    }


def build_environment(
    symbol: str,
    config: MuZeroConfigV3,
    *,
    training_progress_step: int = 0,
    for_collection: bool = False,
) -> TradingEnvironment | None:
    """Construit l'environnement MuZero a partir de l'historique du symbole.

    Args:
        symbol (str): Symbole de marche cible.
        config (MuZeroConfigV3): Configuration MuZero active.
        training_progress_step (int): Etape d'optimisation a refléter dans
            le curriculum de self-play.
        for_collection (bool): Active les garde-fous specifiques a la
            collecte, notamment un horizon d'episode plus court.

    Returns:
        TradingEnvironment | None: Environnement pret ou ``None`` si
            l'historique est inutilisable.
    """
    frame = load_history_frame(symbol, config.primary_timeframe)
    if frame is None:
        logger.warning("Historique absent pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    history_bars = get_horizon_history_bars(config.horizon, env_prefix="MUZERO_HISTORY", fallback=4000)
    market_data, day_labels = build_muzero_market_context(frame.tail(history_bars))
    if market_data.shape[0] < 240:
        logger.warning("Historique insuffisant pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    collection_budget = (
        _resolve_collection_runtime_budget(symbol, config)
        if for_collection
        else {}
    )
    max_steps_budget = (
        int(
            collection_budget.get(
                "collection_max_moves",
                getattr(config, "collection_max_moves", config.max_moves),
            )
        )
        if for_collection
        else int(config.max_moves)
    )
    max_steps = min(max_steps_budget, market_data.shape[0] - 101)
    env = TradingEnvironment(
        data=market_data,
        day_labels=day_labels,
        symbol=symbol,
        config=config,
        max_steps=max_steps,
        training_mode=True,
        training_progress_step=int(training_progress_step or 0),
    )
    setattr(env, "dataset_source", str(frame.attrs.get("dataset_source") or getattr(config, "dataset_source", "csv")))
    setattr(env, "dataset_last_bar_at", frame.attrs.get("dataset_last_bar_at"))
    setattr(env, "dataset_lag_hours", frame.attrs.get("dataset_lag_hours"))
    setattr(env, "dataset_stale", frame.attrs.get("dataset_stale"))
    if for_collection:
        setattr(
            env,
            "collection_num_simulations",
            int(
                collection_budget.get(
                    "collection_num_simulations",
                    getattr(config, "collection_num_simulations", config.num_simulations),
                )
            ),
        )
        setattr(
            env,
            "collection_max_episode_seconds",
            float(
                collection_budget.get(
                    "collection_max_episode_seconds",
                    getattr(config, "collection_max_episode_seconds", 0.0) or 0.0,
                )
            ),
        )
        setattr(env, "collection_budget_family", str(collection_budget.get("family") or "mixed"))
    return env


def _collect_single_muzero_game(
    *,
    symbol: str,
    config: MuZeroConfigV3,
    params: Any,
    opt_state: Any,
    training_step_count: int,
) -> tuple[Any | None, dict[str, object], str | None]:
    """Collecte une partie MuZero isolee pour un symbole donne.

    Cette variante sert a parallelliser la collecte entre plusieurs
    parties independantes sans partager le replay buffer principal.

    Args:
        symbol (str): Symbole cible.
        config (MuZeroConfigV3): Configuration MuZero active.
        params (Any): Poids MuZero courants.
        opt_state (Any): Etat d'optimiseur courant.
        training_step_count (int): Etape de progression a refleter.

    Returns:
        tuple[Any | None, dict[str, object], str | None]: Partie collectee,
        resume d'episode et raison d'arret eventuelle.
    """
    env = build_environment(
        symbol,
        config,
        training_progress_step=int(training_step_count or 0),
        for_collection=True,
    )
    if env is None:
        return None, {}, "historique_inutilisable"

    worker_agent = JAXMuZeroAgent(config)
    worker_agent.params = params
    worker_agent.opt_state = opt_state
    worker_agent.training_step_count = int(training_step_count or 0)
    game = worker_agent.play_game(
        env,
        exploration=True,
        progress_callback=None,
    )
    summary = dict(env.get_summary() or {})
    stopped_reason = str(game.metadata.get("stopped_reason") or "").strip()
    return game, summary, (stopped_reason or None)


def _is_gold_proxy_precheck_enabled(
    *,
    gate_profile: str,
    trial_mode: str | None,
    focus_symbols: list[str],
) -> bool:
    """Retourne vrai si le run courant doit executer un precheck Gold.

    Args:
        gate_profile (str): Profil de gate du run.
        trial_mode (str | None): Mode courant du trial.
        focus_symbols (list[str]): Univers explicite du run.

    Returns:
        bool: ``True`` si le precheck Gold est pertinent, ``False`` sinon.
    """

    normalized_focus = [str(symbol).strip().upper() for symbol in focus_symbols if str(symbol).strip()]
    return (
        str(gate_profile or "").strip().lower() == "gold_demo"
        and str(trial_mode or "").strip().lower() == "proxy_ga"
        and normalized_focus == ["XAUUSD"]
    )


def _evaluate_gold_precheck_verdict(
    *,
    metrics: dict[str, object],
    mechanics: dict[str, object],
) -> dict[str, object]:
    """Etablit un verdict de precheck Gold a partir des metriques intermediaires.

    La politique reste prudente: on coupe uniquement les echantillons
    manifestement faibles. Les cas ambigus continuent jusqu'au proxy complet.

    Args:
        metrics (dict[str, object]): Metriques consolidees du challenger.
        mechanics (dict[str, object]): Metriques de mecanique de position.

    Returns:
        dict[str, object]: Verdict structure ``pass``, ``fail`` ou
            ``inconclusive`` avec raison et mode d'echec stable.
    """

    evaluation_games = int(metrics.get("evaluation_games", 0) or 0)
    total_trades = int(metrics.get("total_trades", 0) or 0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)
    directional_bias = str(metrics.get("directional_bias") or "inactive").strip().lower()
    long_entries = int(metrics.get("long_entries", 0) or 0)
    short_entries = int(metrics.get("short_entries", 0) or 0)
    try:
        directional_imbalance = float(metrics.get("directional_imbalance", 1.0))
    except (TypeError, ValueError):
        directional_imbalance = 1.0

    if evaluation_games <= 0 or total_trades <= 0:
        return {
            "status": "fail",
            "reason": "aucun_trade_exploitable",
            "failure_mode": "inactive",
        }

    if long_entries <= 0 or short_entries <= 0:
        return {
            "status": "fail",
            "reason": "direction_absente",
            "failure_mode": directional_bias or "inactive",
        }

    if directional_imbalance > 0.75:
        return {
            "status": "fail",
            "reason": "desequilibre_directionnel_extreme",
            "failure_mode": directional_bias or "inactive",
        }

    if directional_bias in {"buy_heavy", "sell_heavy"} and profit_factor < 1.0:
        return {
            "status": "fail",
            "reason": "biais_directionnel_extreme",
            "failure_mode": directional_bias,
        }

    if (
        return_pct <= 0.0
        and net_realized_pct < 0.0
        and profit_factor < 0.90
        and close_quality_score <= 0.05
        and hold_drag_score >= 0.90
    ):
        return {
            "status": "fail",
            "reason": "profil_non_rentable_et_sorties_degradees",
            "failure_mode": "unprofitable",
        }

    if close_quality_score <= 0.02 and hold_drag_score >= 1.00 and total_trades >= 4:
        return {
            "status": "fail",
            "reason": "sorties_trop_degradees",
            "failure_mode": "bad_exit",
        }

    if (
        total_trades >= 6
        and return_pct > 0.0
        and net_realized_pct >= 0.0
        and profit_factor >= 1.0
        and directional_bias == "balanced"
        and close_quality_score >= 0.20
        and hold_drag_score <= 0.80
    ):
        return {
            "status": "pass",
            "reason": "signal_prometteur",
            "failure_mode": None,
        }

    return {
        "status": "inconclusive",
        "reason": "signal_ambigu_a_confirmer",
        "failure_mode": None,
    }


def _build_lineage(
    *,
    resume_source: str | None,
    resume_checkpoint_path: str | None,
    ga_parent_champion_id: str | None,
    ga_campaign_id: str | None,
    ga_trial: str | None,
    ga_scope: str | None,
    ga_generation: int | None,
) -> dict[str, object]:
    """Construit une lineage stable pour les checkpoints et manifestes MuZero."""

    lineage = {
        "resume_source": resume_source,
        "resume_checkpoint_path": resume_checkpoint_path,
        "parent_champion_id": ga_parent_champion_id,
        "ga_campaign_id": ga_campaign_id,
        "ga_trial": ga_trial,
        "ga_scope": ga_scope,
        "ga_generation": ga_generation,
    }
    return {
        str(key): value
        for key, value in lineage.items()
        if value is not None
    }


def _build_resume_candidates(
    *,
    explicit_resume_path: str | None,
    ga_seed_checkpoint_path: str | None,
    latest_path: Path,
) -> list[dict[str, str]]:
    """Construit les sources de reprise MuZero dans l'ordre de priorite."""

    candidates: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for source_name, source_path in (
        ("explicit_resume", explicit_resume_path),
        ("ga_seed_checkpoint", ga_seed_checkpoint_path),
        ("latest", str(latest_path)),
    ):
        normalized_path = str(source_path or "").strip()
        if not normalized_path or normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        candidates.append({"source": source_name, "path": normalized_path})
    return candidates


def _to_metric_float(
    metrics: dict[str, object] | object,
    key: str | None = None,
    default: float = 0.0,
) -> float:
    """Convertit une metrique libre en flottant robuste.

    Args:
        metrics (dict[str, object] | object): Dictionnaire source ou valeur
            scalaire brute.
        key (str | None): Nom de la metrique si ``metrics`` est un
            dictionnaire.
        default (float): Valeur de repli.

    Returns:
        float: Valeur convertie ou repli.
    """
    raw_value: object
    if isinstance(metrics, dict) and key is not None:
        raw_value = metrics.get(key, default)
    else:
        raw_value = metrics
    try:
        return float(raw_value if raw_value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _coerce_metric_mapping(
    raw_value: object,
    *,
    default: dict[str, object] | None = None,
) -> dict[str, object]:
    """Convertit une metrique libre en dictionnaire robuste.

    Certaines metriques `*_by_symbol` sont persistees sous forme de chaine
    Python (`"{'EURUSD': 0.41}"`) dans les snapshots intermediaires. Le
    precheck policy doit accepter ces deux formats sans interrompre le run.

    Args:
        raw_value (object): Valeur source brute.
        default (dict[str, object] | None): Valeur de repli.

    Returns:
        dict[str, object]: Dictionnaire exploitable, ou le repli si la
            conversion echoue.
    """
    fallback = dict(default or {})
    if raw_value is None:
        return fallback
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if isinstance(raw_value, str):
        payload = raw_value.strip()
        if not payload:
            return fallback
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed_value = parser(payload)
            except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
                continue
            if isinstance(parsed_value, dict):
                return dict(parsed_value)
        return fallback
    try:
        return dict(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _normalize_rate(value: float) -> float:
    """Normalise un taux qui peut etre exprime en ratio ou en pourcentage.

    Args:
        value (float): Valeur brute.

    Returns:
        float: Valeur ramenee dans `[0, 1]` si necessaire.
    """
    return value / 100.0 if value > 1.0 else value


def _resolve_policy_head_weight_sum(config: MuZeroConfigV3 | Any | None) -> float:
    """Retourne le nombre effectif de tetes policy apres ponderation.

    Args:
        config (MuZeroConfigV3 | Any | None): Configuration MuZero active.

    Returns:
        float: Nombre effectif de tetes policy pris en compte.
    """
    if config is None:
        return 6.0
    root_weight = max(0.0, float(getattr(config, "policy_loss_root_weight", 1.0) or 1.0))
    unroll_weight = max(0.0, float(getattr(config, "policy_loss_unroll_weight", 0.85) or 0.85))
    num_unroll_steps = max(0, int(getattr(config, "num_unroll_steps", 5) or 5))
    return max(1.0, root_weight + (unroll_weight * num_unroll_steps))


def _resolve_loss_pol_per_head(
    metrics: dict[str, object],
    *,
    config: MuZeroConfigV3 | Any | None,
    default: float,
) -> float:
    """Retourne la loss policy par tete, meme si la metrique dediee manque.

    Args:
        metrics (dict[str, object]): Dictionnaire de metriques source.
        config (MuZeroConfigV3 | Any | None): Configuration MuZero active.
        default (float): Valeur de repli si la conversion echoue.

    Returns:
        float: Loss policy moyenne par tete.
    """
    explicit_per_head = _to_metric_float(metrics, "loss_pol_per_head", default=float("nan"))
    if explicit_per_head == explicit_per_head:
        return explicit_per_head
    loss_pol = _to_metric_float(metrics, "loss_pol", default=default)
    return loss_pol / _resolve_policy_head_weight_sum(config)


def _build_arena_cutover_status(step: int) -> dict[str, object]:
    """Construit l'etat de la fenetre de cutover Arena tardive.

    Args:
        step (int): Etape d'optimisation courante.

    Returns:
        dict[str, object]: Indicateurs de readiness Arena et fenetre active.
    """
    ordered_steps = (10000, 12000, 14000)
    if step < ordered_steps[0]:
        return {
            "arena_cutover_ready": False,
            "screen_window": "before_ckpt10000",
        }
    if step < ordered_steps[1]:
        return {
            "arena_cutover_ready": True,
            "screen_window": "ckpt10000_to_11999",
        }
    if step < ordered_steps[2]:
        return {
            "arena_cutover_ready": True,
            "screen_window": "ckpt12000_to_13999",
        }
    return {
        "arena_cutover_ready": True,
        "screen_window": "ckpt14000_plus",
    }


def _evaluate_policy_precheck_window(
    *,
    history: list[dict[str, object]],
    config: MuZeroConfigV3,
    step: int,
    stage: str,
) -> dict[str, object]:
    """Evalue la fenetre glissante de policy avant autorisation Arena.

    Args:
        history (list[dict[str, object]]): Historique recent des metriques.
        config (MuZeroConfigV3): Configuration MuZero courante.
        step (int): Etape a laquelle le precheck est evalue.
        stage (str): Etiquette du contexte (`mid_run`, `pre_arena`, etc.).

    Returns:
        dict[str, object]: Verdict structure du precheck policy avec
            le niveau autorise (`full_ready`, `screen_only` ou `blocked`).
    """
    if not history:
        return {
            "status": "blocked",
            "reason": "fenetre_policy_absente",
            "step": step,
            "stage": stage,
            "window_size": 0,
            "checks": {},
            "full_checks": {},
            "screen_checks": {},
            "medians": {},
        }

    def _compute_metric_trend(metric_key: str, default: float = 0.0) -> float:
        """Calcule une derive simple entre les deux moities de la fenetre.

        Args:
            metric_key (str): Cle de metrique a comparer.
            default (float): Valeur de repli si la metrique est absente.

        Returns:
            float: Difference ``median(fin) - median(debut)``.
        """
        if len(history) < 2:
            return 0.0
        pivot = max(1, len(history) // 2)
        left = [_to_metric_float(item, metric_key, default=default) for item in history[:pivot]]
        right = [_to_metric_float(item, metric_key, default=default) for item in history[pivot:]]
        if not left or not right:
            return 0.0
        return statistics.median(right) - statistics.median(left)

    loss_values = [_to_metric_float(item, "loss_pol", default=999.0) for item in history]
    loss_per_head_values = [
        _resolve_loss_pol_per_head(item, config=config, default=999.0)
        for item in history
    ]
    loss_root_values = [_to_metric_float(item, "loss_pol_root", default=999.0) for item in history]
    loss_unroll_mean_values = [
        _to_metric_float(item, "loss_pol_unroll_mean", default=999.0) for item in history
    ]
    top1_values = [_to_metric_float(item, "policy_top1_share", default=0.0) for item in history]
    entropy_values = [_to_metric_float(item, "policy_entropy", default=1.0) for item in history]
    root_mask_values = [_to_metric_float(item, "root_mask_rate", default=1.0) for item in history]
    post_veto_values = [_to_metric_float(item, "post_veto_to_hold_rate", default=1.0) for item in history]
    soft_ratio_values = [
        _to_metric_float(item, "soft_penalty_to_bonus_ratio", default=999.0)
        for item in history
    ]
    close_quality_values = [_to_metric_float(item, "close_quality_score", default=0.0) for item in history]
    split_efficiency_values = [_to_metric_float(item, "split_efficiency", default=0.0) for item in history]
    split_runner_capture_values = [
        _to_metric_float(item, "split_runner_capture_rate", default=0.0)
        for item in history
    ]
    pyramid_efficiency_values = [_to_metric_float(item, "pyramid_efficiency", default=0.0) for item in history]
    pyramid_exit_capture_values = [
        _to_metric_float(item, "pyramid_exit_capture_rate", default=0.0)
        for item in history
    ]
    slbe_capture_values = [_to_metric_float(item, "slbe_capture_rate", default=0.0) for item in history]
    hold_drag_values = [_to_metric_float(item, "hold_drag_score", default=1.0) for item in history]
    balanced_values = [
        _normalize_rate(_to_metric_float(item, "balanced_episode_rate", default=0.0))
        for item in history
    ]
    long_share_values = [_to_metric_float(item, "long_entry_share", default=0.0) for item in history]
    short_share_values = [_to_metric_float(item, "short_entry_share", default=0.0) for item in history]
    good_close_symbol_counts: list[float] = []
    min_symbol_close_quality = float(
        getattr(config, "policy_precheck_min_symbol_close_quality_score", 0.25) or 0.25
    )
    min_symbol_close_events = int(
        getattr(config, "policy_precheck_min_symbol_close_events", 6) or 6
    )
    for item in history:
        close_quality_by_symbol = _coerce_metric_mapping(item.get("close_quality_by_symbol"))
        close_events_by_symbol = _coerce_metric_mapping(item.get("close_events_by_symbol"))
        good_symbol_count = 0
        for symbol, raw_quality in close_quality_by_symbol.items():
            try:
                quality_value = float(raw_quality or 0.0)
            except (TypeError, ValueError):
                quality_value = 0.0
            try:
                close_events = int(float(close_events_by_symbol.get(symbol, 0) or 0))
            except (TypeError, ValueError):
                close_events = 0
            if close_events >= min_symbol_close_events and quality_value >= min_symbol_close_quality:
                good_symbol_count += 1
        good_close_symbol_counts.append(float(good_symbol_count))

    medians = {
        "loss_pol": statistics.median(loss_values),
        "loss_pol_per_head": statistics.median(loss_per_head_values),
        "loss_pol_root": statistics.median(loss_root_values),
        "loss_pol_unroll_mean": statistics.median(loss_unroll_mean_values),
        "policy_top1_share": statistics.median(top1_values),
        "policy_entropy": statistics.median(entropy_values),
        "root_mask_rate": statistics.median(root_mask_values),
        "post_veto_to_hold_rate": statistics.median(post_veto_values),
        "soft_penalty_to_bonus_ratio": statistics.median(soft_ratio_values),
        "close_quality_score": statistics.median(close_quality_values),
        "split_efficiency": statistics.median(split_efficiency_values),
        "split_runner_capture_rate": statistics.median(split_runner_capture_values),
        "pyramid_efficiency": statistics.median(pyramid_efficiency_values),
        "pyramid_exit_capture_rate": statistics.median(pyramid_exit_capture_values),
        "slbe_capture_rate": statistics.median(slbe_capture_values),
        "hold_drag_score": statistics.median(hold_drag_values),
        "balanced_episode_rate": statistics.median(balanced_values),
        "long_entry_share": statistics.median(long_share_values),
        "short_entry_share": statistics.median(short_share_values),
        "good_close_symbols": statistics.median(good_close_symbol_counts) if good_close_symbol_counts else 0.0,
    }
    trends = {
        "loss_pol_trend": _compute_metric_trend("loss_pol", default=999.0),
        "loss_pol_per_head_trend": (
            statistics.median(loss_per_head_values[max(1, len(loss_per_head_values) // 2):])
            - statistics.median(loss_per_head_values[:max(1, len(loss_per_head_values) // 2)])
            if len(loss_per_head_values) >= 2
            else 0.0
        ),
        "loss_pol_root_trend": _compute_metric_trend("loss_pol_root", default=999.0),
        "loss_pol_unroll_mean_trend": _compute_metric_trend(
            "loss_pol_unroll_mean",
            default=999.0,
        ),
        "root_mask_rate_trend": _compute_metric_trend("root_mask_rate", default=1.0),
        "split_runner_capture_trend": _compute_metric_trend("split_runner_capture_rate", default=0.0),
        "pyramid_exit_capture_trend": _compute_metric_trend("pyramid_exit_capture_rate", default=0.0),
        "close_quality_score_trend": _compute_metric_trend("close_quality_score", default=0.0),
    }
    late_policy_convergence = (
        stage == "pre_arena"
        and trends["loss_pol_root_trend"] <= -0.01
        and trends["loss_pol_unroll_mean_trend"] <= -0.01
    )
    loss_pol_per_head_limit = float(
        getattr(config, "policy_precheck_max_loss_pol_per_head", 1.12) or 1.12
    )
    has_v6_metrics = any("close_quality_score" in item for item in history)
    has_symbol_metrics = any("close_quality_by_symbol" in item for item in history)
    full_checks = {
        "loss_pol_per_head": (
            medians["loss_pol_per_head"] <= loss_pol_per_head_limit
            or late_policy_convergence
        ),
        "policy_top1_share": medians["policy_top1_share"] >= float(
            getattr(config, "policy_precheck_min_top1_share", 0.75) or 0.75
        ),
        "policy_entropy": medians["policy_entropy"] <= float(
            getattr(config, "policy_precheck_max_policy_entropy", 1.0) or 1.0
        ),
        "root_mask_rate": medians["root_mask_rate"] <= float(
            getattr(config, "policy_precheck_max_root_mask_rate", 0.05) or 0.05
        ),
        "root_mask_rate_trend": trends["root_mask_rate_trend"] <= float(
            getattr(config, "policy_precheck_max_root_mask_rate_trend", 0.02) or 0.02
        ),
        "post_veto_to_hold_rate": medians["post_veto_to_hold_rate"] <= float(
            getattr(config, "policy_precheck_max_post_veto_rate", 0.01) or 0.01
        ),
        "balanced_episode_rate": medians["balanced_episode_rate"] >= float(
            getattr(config, "policy_precheck_min_balanced_episode_rate", 0.85) or 0.85
        ),
        "long_entry_share": medians["long_entry_share"] >= float(
            getattr(config, "policy_precheck_min_long_entry_share", 0.35) or 0.35
        ),
        "short_entry_share": medians["short_entry_share"] >= float(
            getattr(config, "policy_precheck_min_short_entry_share", 0.35) or 0.35
        ),
        "close_quality_score": not has_v6_metrics or medians["close_quality_score"] >= float(
            getattr(config, "policy_precheck_min_close_quality_score", 0.40) or 0.40
        ),
        "split_efficiency": not has_v6_metrics or medians["split_efficiency"] >= float(
            getattr(config, "policy_precheck_min_split_efficiency", 0.35) or 0.35
        ),
        "pyramid_efficiency": not has_v6_metrics or medians["pyramid_efficiency"] >= float(
            getattr(config, "policy_precheck_min_pyramid_efficiency", 0.35) or 0.35
        ),
        "slbe_capture_rate": not has_v6_metrics or medians["slbe_capture_rate"] >= float(
            getattr(config, "policy_precheck_min_slbe_capture_rate", 0.45) or 0.45
        ),
        "hold_drag_score": not has_v6_metrics or medians["hold_drag_score"] <= float(
            getattr(config, "policy_precheck_max_hold_drag_score", 0.10) or 0.10
        ),
        "good_close_symbols": not has_symbol_metrics or medians["good_close_symbols"] >= float(
            getattr(config, "policy_precheck_min_good_close_symbols", 5) or 5
        ),
    }
    screen_checks = {
        "loss_pol_per_head": medians["loss_pol_per_head"] <= float(
            getattr(config, "policy_screen_max_loss_pol_per_head", 1.20) or 1.20
        ),
        "policy_top1_share": medians["policy_top1_share"] >= float(
            getattr(config, "policy_screen_min_top1_share", 0.88) or 0.88
        ),
        "policy_entropy": medians["policy_entropy"] <= float(
            getattr(config, "policy_screen_max_policy_entropy", 0.45) or 0.45
        ),
        "root_mask_rate": medians["root_mask_rate"] <= float(
            getattr(config, "policy_screen_max_root_mask_rate", 0.05) or 0.05
        ),
        "root_mask_rate_trend": trends["root_mask_rate_trend"] <= float(
            getattr(config, "policy_precheck_max_root_mask_rate_trend", 0.02) or 0.02
        ),
        "post_veto_to_hold_rate": medians["post_veto_to_hold_rate"] <= float(
            getattr(config, "policy_screen_max_post_veto_rate", 0.01) or 0.01
        ),
        "balanced_episode_rate": medians["balanced_episode_rate"] >= float(
            getattr(config, "policy_screen_min_balanced_episode_rate", 0.85) or 0.85
        ),
        "long_entry_share": medians["long_entry_share"] >= float(
            getattr(config, "policy_screen_min_long_entry_share", 0.35) or 0.35
        ),
        "short_entry_share": medians["short_entry_share"] >= float(
            getattr(config, "policy_screen_min_short_entry_share", 0.35) or 0.35
        ),
    }
    if has_v6_metrics and has_symbol_metrics:
        screen_checks["good_close_symbols"] = medians["good_close_symbols"] >= float(
            getattr(config, "policy_precheck_min_good_close_symbols", 5) or 5
        )
    full_failed_check = next((name for name, passed in full_checks.items() if not passed), None)
    screen_failed_check = next((name for name, passed in screen_checks.items() if not passed), None)
    if full_failed_check is None:
        status = "full_ready"
        reason = "eligible_full"
        checks = full_checks
    elif screen_failed_check is None:
        status = "screen_only"
        reason = "eligible_screen"
        checks = screen_checks
    else:
        status = "blocked"
        reason = screen_failed_check
        checks = screen_checks
    return {
        "status": status,
        "reason": reason,
        "step": step,
        "stage": stage,
        "window_size": len(history),
        "checks": checks,
        "full_checks": full_checks,
        "screen_checks": screen_checks,
        "medians": medians,
        "trends": trends,
        "arena_cutover_ready": _build_arena_cutover_status(step).get("arena_cutover_ready"),
        "screen_window": _build_arena_cutover_status(step).get("screen_window"),
        "late_policy_convergence": late_policy_convergence,
        "latest_metrics": dict(history[-1] or {}),
    }


def _evaluate_seed_viability_window(
    *,
    history: list[dict[str, object]],
    config: MuZeroConfigV3,
    step: int,
    horizon: str,
    weights_dir: Path,
    trial_mode: str | None = None,
) -> dict[str, object]:
    """Evalue si le seed courant reste pedagogiquement viable pour V6.10.

    Args:
        history (list[dict[str, object]]): Historique recent des metriques.
        config (MuZeroConfigV3): Configuration MuZero active.
        step (int): Etape courante d'optimisation.
        horizon (str): Horizon MuZero en cours.
        weights_dir (Path): Dossier des checkpoints afin de proposer un seed
            de repli si le seed courant echoue.
        trial_mode (str | None): Etage de qualification seed en cours.

    Returns:
        dict[str, object]: Verdict structure sur la viabilite du seed,
            avec recommandation de checkpoint si la fenetre est jugee
            non productive.
    """
    normalized_trial_mode = str(trial_mode or "").strip().lower()
    if normalized_trial_mode not in {"offensive_bootstrap", "seed_short_mixed"}:
        recommendation = recommend_muzero_seed_for_v66(
            weights_dir=weights_dir,
            horizon=horizon,
        )
        return {
            "allowed": True,
            "status": "skipped",
            "reason": "seed_viability_not_applicable",
            "step": step,
            "window_size": len(history),
            "training_steps": int(getattr(config, "training_steps", 0) or 0),
            "window_min_step": None,
            "window_max_step": None,
            "seed_stage": "full_run",
            "metrics": {},
            "checks": {},
            "recommended_seed_for_v66": recommendation.get("recommended_seed_for_v66"),
            "seed_selection_reason": recommendation.get("seed_selection_reason"),
        }
    seed_stage = (
        "offensive_bootstrap"
        if normalized_trial_mode == "offensive_bootstrap"
        else "seed_short_mixed"
    )
    min_step = int(getattr(config, "seed_viability_min_step", 6000) or 6000)
    max_step = int(getattr(config, "seed_viability_max_step", 8000) or 8000)
    training_steps = int(getattr(config, "training_steps", max_step) or max_step)
    effective_min_step = max(1, min(min_step, training_steps))
    effective_max_step = max(effective_min_step, min(max_step, training_steps))
    recommendation = recommend_muzero_seed_for_v66(
        weights_dir=weights_dir,
        horizon=horizon,
    )
    result = {
        "allowed": True,
        "status": "monitoring",
        "reason": "before_seed_window",
        "step": step,
        "window_size": len(history),
        "training_steps": training_steps,
        "window_min_step": effective_min_step,
        "window_max_step": effective_max_step,
        "seed_stage": seed_stage,
        "metrics": {},
        "checks": {},
        "recommended_seed_for_v66": recommendation.get("recommended_seed_for_v66"),
        "seed_selection_reason": recommendation.get("seed_selection_reason"),
    }
    if step < effective_min_step and step < training_steps:
        return result

    if not history:
        result.update(
            {
                "allowed": False,
                "status": "seed_not_viable_for_v66",
                "reason": "seed_window_without_history",
            }
        )
        return result

    def _compute_metric_trend(metric_key: str, default: float = 0.0) -> float:
        """Calcule une derive simple entre les deux moities de la fenetre.

        Args:
            metric_key (str): Cle de metrique a comparer.
            default (float): Valeur de repli si la metrique est absente.

        Returns:
            float: Difference ``median(fin) - median(debut)``.
        """
        if len(history) < 2:
            return 0.0
        pivot = max(1, len(history) // 2)
        left = [_to_metric_float(item, metric_key, default=default) for item in history[:pivot]]
        right = [_to_metric_float(item, metric_key, default=default) for item in history[pivot:]]
        if not left or not right:
            return 0.0
        return statistics.median(right) - statistics.median(left)

    loss_values = [_to_metric_float(item, "loss_pol", default=999.0) for item in history]
    root_mask_values = [_to_metric_float(item, "root_mask_rate", default=1.0) for item in history]
    split_runner_values = [
        _to_metric_float(item, "split_runner_capture_rate", default=0.0)
        for item in history
    ]
    split_zone_values = [
        _to_metric_float(item, "split_zone_capture_rate", default=0.0)
        for item in history
    ]
    split_monetization_capture_values = [
        _to_metric_float(item, "split_monetization_capture_rate", default=0.0)
        for item in history
    ]
    split_monetization_window_values = [
        _to_metric_float(item, "split_monetization_window_count", default=0.0)
        for item in history
    ]
    runner_extension_values = [
        _to_metric_float(item, "runner_extension_capture_rate", default=0.0)
        for item in history
    ]
    runner_profit_hold_capture_values = [
        _to_metric_float(item, "runner_profit_hold_capture_rate", default=0.0)
        for item in history
    ]
    runner_profit_hold_window_values = [
        _to_metric_float(item, "runner_profit_hold_window_count", default=0.0)
        for item in history
    ]
    runner_giveback_ratio_values = [
        _to_metric_float(item, "runner_giveback_ratio", default=1.0)
        for item in history
    ]
    profit_peak_giveback_ratio_values = [
        _to_metric_float(item, "profit_peak_giveback_ratio", default=1.0)
        for item in history
    ]
    pyramid_exit_values = [
        _to_metric_float(item, "pyramid_exit_capture_rate", default=0.0)
        for item in history
    ]
    pyramid_add_values = [
        _to_metric_float(item, "pyramid_add_capture_rate", default=0.0)
        for item in history
    ]
    pyramid_monetization_capture_values = [
        _to_metric_float(item, "pyramid_monetization_capture_rate", default=0.0)
        for item in history
    ]
    pyramid_monetization_window_values = [
        _to_metric_float(item, "pyramid_monetization_window_count", default=0.0)
        for item in history
    ]
    close_quality_values = [
        _to_metric_float(item, "close_quality_score", default=0.0)
        for item in history
    ]
    slbe_capture_values = [
        _to_metric_float(item, "slbe_capture_rate", default=0.0)
        for item in history
    ]
    split_tp_zone_opportunity_values = [
        _to_metric_float(item, "split_tp_zone_opportunity_count", default=0.0)
        for item in history
    ]
    runner_extension_opportunity_values = [
        _to_metric_float(item, "runner_extension_opportunity_count", default=0.0)
        for item in history
    ]
    pyramid_add_opportunity_values = [
        _to_metric_float(item, "pyramid_add_opportunity_count", default=0.0)
        for item in history
    ]
    loss_per_head_values = [
        _resolve_loss_pol_per_head(item, config=config, default=999.0)
        for item in history
    ]
    medians = {
        "loss_pol": statistics.median(loss_values),
        "loss_pol_per_head": statistics.median(loss_per_head_values),
        "root_mask_rate": statistics.median(root_mask_values),
        "split_runner_capture_rate": statistics.median(split_runner_values),
        "split_zone_capture_rate": statistics.median(split_zone_values),
        "split_monetization_capture_rate": statistics.median(split_monetization_capture_values),
        "split_monetization_window_count": statistics.median(split_monetization_window_values),
        "runner_extension_capture_rate": statistics.median(runner_extension_values),
        "runner_profit_hold_capture_rate": statistics.median(runner_profit_hold_capture_values),
        "runner_profit_hold_window_count": statistics.median(runner_profit_hold_window_values),
        "runner_giveback_ratio": statistics.median(runner_giveback_ratio_values),
        "profit_peak_giveback_ratio": statistics.median(profit_peak_giveback_ratio_values),
        "pyramid_exit_capture_rate": statistics.median(pyramid_exit_values),
        "pyramid_add_capture_rate": statistics.median(pyramid_add_values),
        "pyramid_monetization_capture_rate": statistics.median(pyramid_monetization_capture_values),
        "pyramid_monetization_window_count": statistics.median(pyramid_monetization_window_values),
        "close_quality_score": statistics.median(close_quality_values),
        "slbe_capture_rate": statistics.median(slbe_capture_values),
        "split_tp_zone_opportunity_count": statistics.median(split_tp_zone_opportunity_values),
        "runner_extension_opportunity_count": statistics.median(runner_extension_opportunity_values),
        "pyramid_add_opportunity_count": statistics.median(pyramid_add_opportunity_values),
    }
    trends = {
        "loss_pol_trend": _compute_metric_trend("loss_pol", default=999.0),
        "loss_pol_per_head_trend": (
            statistics.median(loss_per_head_values[max(1, len(loss_per_head_values) // 2):])
            - statistics.median(loss_per_head_values[:max(1, len(loss_per_head_values) // 2)])
            if len(loss_per_head_values) >= 2
            else 0.0
        ),
        "root_mask_rate_trend": _compute_metric_trend("root_mask_rate", default=1.0),
        "split_runner_capture_trend": _compute_metric_trend("split_runner_capture_rate", default=0.0),
        "split_zone_capture_trend": _compute_metric_trend("split_zone_capture_rate", default=0.0),
        "split_monetization_capture_trend": _compute_metric_trend(
            "split_monetization_capture_rate",
            default=0.0,
        ),
        "runner_extension_capture_trend": _compute_metric_trend(
            "runner_extension_capture_rate",
            default=0.0,
        ),
        "runner_profit_hold_capture_trend": _compute_metric_trend(
            "runner_profit_hold_capture_rate",
            default=0.0,
        ),
        "pyramid_exit_capture_trend": _compute_metric_trend("pyramid_exit_capture_rate", default=0.0),
        "pyramid_add_capture_trend": _compute_metric_trend("pyramid_add_capture_rate", default=0.0),
        "pyramid_monetization_capture_trend": _compute_metric_trend(
            "pyramid_monetization_capture_rate",
            default=0.0,
        ),
    }
    loss_pol_improvement = -float(trends["loss_pol_trend"])
    zero_capture_with_opportunities = (
        (
            medians["split_monetization_window_count"]
            + medians["runner_profit_hold_window_count"]
            + medians["pyramid_monetization_window_count"]
        )
        > 0.0
        and medians["split_monetization_capture_rate"] <= 0.0
        and medians["runner_profit_hold_capture_rate"] <= 0.0
        and medians["pyramid_monetization_capture_rate"] <= 0.0
    )
    if seed_stage == "offensive_bootstrap":
        checks = {
            "loss_pol_per_head": medians["loss_pol_per_head"] <= float(
                getattr(config, "seed_bootstrap_max_loss_pol_per_head", 1.16) or 1.16
            ),
            "root_mask_rate": medians["root_mask_rate"] < float(
                getattr(config, "seed_bootstrap_max_root_mask_rate", 0.01) or 0.01
            ),
            "split_monetization_window_count": medians["split_monetization_window_count"] >= float(
                getattr(config, "seed_bootstrap_min_split_monetization_window_count", 1.0) or 1.0
            ),
            "runner_profit_hold_window_count": medians["runner_profit_hold_window_count"] >= float(
                getattr(config, "seed_bootstrap_min_runner_profit_hold_window_count", 1.0) or 1.0
            ),
            "pyramid_monetization_window_count": medians["pyramid_monetization_window_count"] >= float(
                getattr(config, "seed_bootstrap_min_pyramid_monetization_window_count", 1.0) or 1.0
            ),
            "profit_peak_giveback_ratio": medians["profit_peak_giveback_ratio"] < float(
                getattr(config, "seed_bootstrap_max_profit_peak_giveback_ratio", 0.80) or 0.80
            ),
        }
    else:
        checks = {
            "loss_pol_per_head": medians["loss_pol_per_head"] <= float(
                getattr(config, "seed_mixed_max_loss_pol_per_head", 1.10) or 1.10
            ),
            "root_mask_rate": medians["root_mask_rate"] < float(
                getattr(config, "seed_mixed_max_root_mask_rate", 0.06) or 0.06
            ),
            "split_capture": (
                medians["split_runner_capture_rate"] > float(
                    getattr(config, "seed_mixed_min_split_runner_capture_rate", 0.05) or 0.05
                )
                or medians["split_monetization_capture_rate"] > float(
                    getattr(config, "seed_mixed_min_split_monetization_capture_rate", 0.12) or 0.12
                )
            ),
            "pyramid_capture": (
                medians["pyramid_exit_capture_rate"] > float(
                    getattr(config, "seed_mixed_min_pyramid_exit_capture_rate", 0.03) or 0.03
                )
                or medians["pyramid_monetization_capture_rate"] > float(
                    getattr(config, "seed_mixed_min_pyramid_monetization_capture_rate", 0.08) or 0.08
                )
            ),
            "close_quality_score": medians["close_quality_score"] >= float(
                getattr(config, "seed_mixed_min_close_quality_score", 0.40) or 0.40
            ),
            "slbe_capture_rate": medians["slbe_capture_rate"] >= float(
                getattr(config, "seed_mixed_min_slbe_capture_rate", 0.45) or 0.45
            ),
        }
    result.update(
        {
            "metrics": {
                **medians,
                "loss_pol_improvement": loss_pol_improvement,
            },
            "trends": trends,
            "checks": checks,
        }
    )
    if (
        zero_capture_with_opportunities
        and seed_stage != "offensive_bootstrap"
        and step >= effective_min_step
    ):
        result.update(
            {
                "allowed": False,
                "status": "seed_not_viable_for_v66",
                "reason": "seed_not_viable_offensive_zero_capture",
            }
        )
        return result
    failed_check = next((name for name, passed in checks.items() if not passed), None)
    if failed_check is not None and step >= effective_min_step:
        result.update(
            {
                "allowed": False,
                "status": "seed_not_viable_for_v66",
                "reason": failed_check,
            }
        )
        return result

    if step >= effective_max_step or step >= training_steps:
        result.update(
            {
                "status": "seed_viability_passed",
                "reason": "seed_window_passed",
            }
        )
        return result

    if step >= effective_min_step:
        result.update(
            {
                "status": "monitoring",
                "reason": "within_seed_window",
            }
        )
        return result

    return result


def _detect_arena_plateau(
    *,
    history: list[dict[str, object]],
    config: MuZeroConfigV3,
    step: int,
) -> dict[str, object]:
    """Detecte un plateau utile pour couper un run avant 40k steps.

    Args:
        history (list[dict[str, object]]): Historique recent des metriques.
        config (MuZeroConfigV3): Configuration MuZero courante.
        step (int): Etape courante d'optimisation.

    Returns:
        dict[str, object]: Verdict structure du detecteur de plateau.
    """
    min_step = int(getattr(config, "arena_plateau_min_step", 10000) or 10000)
    window_size = int(getattr(config, "arena_plateau_window_size", 500) or 500)
    if step < min_step or len(history) < max(4, window_size):
        return {
            "allowed": False,
            "reason": "insufficient_history",
            "step": step,
        }

    window = list(history[-window_size:])
    pivot = max(1, len(window) // 2)

    def _window_delta(metric_key: str, default: float = 0.0) -> float:
        left = [_to_metric_float(item, metric_key, default=default) for item in window[:pivot]]
        right = [_to_metric_float(item, metric_key, default=default) for item in window[pivot:]]
        if not left or not right:
            return 0.0
        return statistics.median(left) - statistics.median(right)

    loss_pol_improvement = _window_delta("loss_pol", default=999.0)
    split_runner_improvement = _window_delta("split_runner_capture_rate", default=0.0)
    pyramid_exit_improvement = _window_delta("pyramid_exit_capture_rate", default=0.0)
    close_quality_improvement = _window_delta("close_quality_score", default=0.0)
    checks = {
        "loss_pol_plateau": loss_pol_improvement <= float(
            getattr(config, "arena_plateau_max_loss_pol_improvement", 0.10) or 0.10
        ),
        "split_runner_plateau": split_runner_improvement <= float(
            getattr(config, "arena_plateau_min_split_runner_improvement", 0.02) or 0.02
        ),
        "pyramid_exit_plateau": pyramid_exit_improvement <= float(
            getattr(config, "arena_plateau_min_pyramid_exit_improvement", 0.02) or 0.02
        ),
        "close_quality_plateau": close_quality_improvement <= float(
            getattr(config, "arena_plateau_min_close_quality_improvement", 0.02) or 0.02
        ),
    }
    plateau_detected = all(checks.values())
    return {
        "allowed": plateau_detected,
        "reason": "plateau_confirmed" if plateau_detected else "improving_or_noisy",
        "step": step,
        "window_size": len(window),
        "checks": checks,
        "metrics": {
            "loss_pol_improvement": loss_pol_improvement,
            "split_runner_capture_improvement": split_runner_improvement,
            "pyramid_exit_capture_improvement": pyramid_exit_improvement,
            "close_quality_score_improvement": close_quality_improvement,
        },
    }


def _summarize_policy_window(history: list[dict[str, object]]) -> dict[str, float]:
    """Resume une fenetre de metrics pour le tri des checkpoints.

    Args:
        history (list[dict[str, object]]): Fenetre glissante a resumer.

    Returns:
        dict[str, float]: Medians utiles pour le classement.
    """
    if not history:
        return {
            "loss_pol": 999.0,
            "policy_top1_share": 0.0,
            "policy_entropy": 1.0,
            "root_mask_rate": 1.0,
            "post_veto_to_hold_rate": 1.0,
            "soft_penalty_to_bonus_ratio": 999.0,
        }
    return {
        "loss_pol": statistics.median(
            [_to_metric_float(item, "loss_pol", default=999.0) for item in history]
        ),
        "policy_top1_share": statistics.median(
            [_to_metric_float(item, "policy_top1_share", default=0.0) for item in history]
        ),
        "policy_entropy": statistics.median(
            [_to_metric_float(item, "policy_entropy", default=1.0) for item in history]
        ),
        "root_mask_rate": statistics.median(
            [_to_metric_float(item, "root_mask_rate", default=1.0) for item in history]
        ),
        "post_veto_to_hold_rate": statistics.median(
            [_to_metric_float(item, "post_veto_to_hold_rate", default=1.0) for item in history]
        ),
        "soft_penalty_to_bonus_ratio": statistics.median(
            [
                _to_metric_float(item, "soft_penalty_to_bonus_ratio", default=999.0)
                for item in history
            ]
        ),
    }


def _extract_checkpoint_step(path: Path) -> int | None:
    """Extrait l'etape numerique d'un checkpoint MuZero.

    Args:
        path (Path): Chemin de checkpoint.

    Returns:
        int | None: Etape numerique si le nom est conforme.
    """
    stem = str(path.stem)
    if "_ckpt_" not in stem:
        return None
    raw_step = stem.rsplit("_ckpt_", 1)[-1]
    try:
        return int(raw_step)
    except ValueError:
        return None


def _select_recent_screen_checkpoints(
    *,
    history: list[dict[str, object]],
    weights_dir: Path,
    horizon: str,
    last_step: int,
    config: MuZeroConfigV3,
) -> list[dict[str, object]]:
    """Selectionne la grappe recente de checkpoints la plus prometteuse.

    Args:
        history (list[dict[str, object]]): Historique recent des metrics.
        weights_dir (Path): Repertoire des checkpoints.
        horizon (str): Horizon MuZero courant.
        last_step (int): Derniere etape d'optimisation terminee.
        config (MuZeroConfigV3): Configuration MuZero.

    Returns:
        list[dict[str, object]]: Checkpoints tries par etape croissante.
    """
    recent_steps = max(
        int(getattr(config, "arena_screen_recent_steps", 2500) or 2500),
        int(getattr(config, "checkpoint_interval", 500) or 500),
    )
    candidate_count = max(1, int(getattr(config, "arena_screen_candidate_count", 5) or 5))
    window_size = max(1, int(getattr(config, "arena_screen_window_size", 500) or 500))
    recent_floor = max(0, int(last_step) - recent_steps)

    available_checkpoints: list[dict[str, object]] = []
    for checkpoint_path in sorted(weights_dir.glob(f"muzero_{horizon}_ckpt_*.pkl")):
        checkpoint_step = _extract_checkpoint_step(checkpoint_path)
        if checkpoint_step is None or checkpoint_step > int(last_step):
            continue
        available_checkpoints.append(
            {
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": checkpoint_path,
            }
        )
    if not available_checkpoints:
        return []

    target_steps = [
        int(step)
        for step in list(getattr(config, "arena_screen_target_steps", []) or [])
        if int(step) <= int(last_step)
    ]
    if target_steps:
        checkpoints_by_step = {
            int(item["checkpoint_step"]): dict(item)
            for item in available_checkpoints
        }
        explicit_targets = [
            {
                **dict(checkpoints_by_step[step]),
                "selection_anchor_step": step,
                "selection_window": {
                    "mode": "v6_12_late_target",
                    "target_step": step,
                    "target_steps": target_steps,
                    "last_step": int(last_step),
                },
            }
            for step in target_steps
            if step in checkpoints_by_step
        ]
        if explicit_targets:
            return sorted(
                explicit_targets[-candidate_count:],
                key=lambda item: int(item["checkpoint_step"]),
            )

    recent_checkpoints = [
        dict(item)
        for item in available_checkpoints
        if int(item["checkpoint_step"]) >= recent_floor
    ]
    if not recent_checkpoints:
        recent_checkpoints = [dict(item) for item in available_checkpoints[-candidate_count:]]

    recent_history = [
        dict(item)
        for item in history
        if int(item.get("training_step", 0) or 0) >= recent_floor
    ]
    if not recent_history:
        recent_history = [dict(item) for item in history]
    recent_history.sort(key=lambda item: int(item.get("training_step", 0) or 0))

    effective_window = min(window_size, len(recent_history))
    if effective_window <= 0:
        return recent_checkpoints[-candidate_count:]

    candidate_windows: list[dict[str, object]] = []
    for index in range(0, len(recent_history) - effective_window + 1):
        window = recent_history[index:index + effective_window]
        center_step = int(
            statistics.median([int(item.get("training_step", 0) or 0) for item in window])
        )
        candidate_windows.append(
            {
                "start_step": int(window[0].get("training_step", 0) or 0),
                "end_step": int(window[-1].get("training_step", 0) or 0),
                "center_step": center_step,
                "summary": _summarize_policy_window(window),
            }
        )

    best_window = min(
        candidate_windows,
        key=lambda item: (
            float(dict(item.get("summary") or {}).get("loss_pol", 999.0)),
            -float(dict(item.get("summary") or {}).get("policy_top1_share", 0.0)),
            float(dict(item.get("summary") or {}).get("soft_penalty_to_bonus_ratio", 999.0)),
            float(dict(item.get("summary") or {}).get("root_mask_rate", 1.0)),
            float(dict(item.get("summary") or {}).get("post_veto_to_hold_rate", 1.0)),
            abs(int(item.get("center_step") or 0) - int(last_step)),
        ),
    )
    anchor_step = int(best_window.get("center_step") or int(last_step))
    ranked = sorted(
        recent_checkpoints,
        key=lambda item: (
            abs(int(item["checkpoint_step"]) - anchor_step),
            abs(int(last_step) - int(item["checkpoint_step"])),
            -int(item["checkpoint_step"]),
        ),
    )
    selected = sorted(
        [dict(item) for item in ranked[:candidate_count]],
        key=lambda item: int(item["checkpoint_step"]),
    )
    for item in selected:
        item["selection_anchor_step"] = anchor_step
        item["selection_window"] = dict(best_window)
    return selected


def _build_screen_candidate_sort_key(candidate: dict[str, object]) -> tuple[float, ...]:
    """Construit la cle de tri du gagnant provisoire du screen Arena.

    Args:
        candidate (dict[str, object]): Resultat brut d'un checkpoint evalue.

    Returns:
        tuple[float, ...]: Cle deterministe de classement.
    """
    battle_report = dict(candidate.get("battle_report") or {})
    challenger = dict(battle_report.get("challenger") or {})
    metrics = dict(challenger.get("metrics") or {})
    outcome = 1.0 if str(battle_report.get("outcome") or "").upper() == "VICTORY" else 0.0
    return (
        outcome,
        _to_metric_float(challenger.get("score")),
        _to_metric_float(metrics.get("return_pct")),
        _to_metric_float(metrics.get("profit_factor")),
        -_to_metric_float(metrics.get("max_drawdown_pct"), default=100.0),
        float(int(candidate.get("checkpoint_step") or 0)),
    )


def _select_best_screen_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Retourne le meilleur checkpoint issu du screen automatique.

    Args:
        candidates (list[dict[str, object]]): Resultats evalues.

    Returns:
        dict[str, object]: Candidat gagnant.

    Raises:
        ValueError: Si la liste est vide.
    """
    if not candidates:
        raise ValueError("Impossible de selectionner un checkpoint sans candidats.")
    ranked = sorted(candidates, key=_build_screen_candidate_sort_key, reverse=True)
    return dict(ranked[0])


def _build_champion_league_table(
    *,
    engine: str,
    horizon: str,
    screen_results: list[dict[str, object]],
    selected_candidate_id: str | None,
    battle_report: dict[str, object] | None = None,
    promotion_result: dict[str, object] | None = None,
    screen_gate: dict[str, object] | None = None,
    family_probe_report: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Construit une table unique des candidats vus par le cycle.

    Args:
        engine (str): Moteur source du cycle.
        horizon (str): Horizon evalue.
        screen_results (list[dict[str, object]]): Resultats des screens.
        selected_candidate_id (str | None): Identifiant du candidat retenu.
        battle_report (dict[str, object] | None): Rapport full Arena.
        promotion_result (dict[str, object] | None): Resultat de promotion.
        screen_gate (dict[str, object] | None): Gate du screen.
        family_probe_report (dict[str, object] | None): Rapport des probes.

    Returns:
        list[dict[str, object]]: Lignes de ligue triees par etape.
    """
    rows: list[dict[str, object]] = []
    selected_path = str(
        (battle_report or {}).get("challenger", {}).get("path")
        if isinstance((battle_report or {}).get("challenger"), dict)
        else ""
    )
    for item in screen_results:
        report = dict(item.get("battle_report") or {})
        challenger = dict(report.get("challenger") or {})
        metrics = dict(challenger.get("metrics") or {})
        checkpoint_step = int(item.get("checkpoint_step") or 0)
        checkpoint_path = str(item.get("checkpoint_path") or "")
        is_selected = bool(
            selected_path
            and checkpoint_path
            and Path(selected_path).resolve() == Path(checkpoint_path).resolve()
        )
        candidate_id = (
            selected_candidate_id
            if is_selected and selected_candidate_id
            else f"{engine}_{horizon}_ckpt{checkpoint_step}"
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "algo_source": engine,
                "horizon": horizon,
                "stage": "screen",
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": checkpoint_path,
                "score_proxy": _to_metric_float(challenger.get("score")),
                "score_nemesis": _to_metric_float(
                    dict(report.get("nemesis_validation") or {}).get("score"),
                    default=0.0,
                ),
                "score_arena": None,
                "outcome": str(report.get("outcome") or "UNKNOWN"),
                "promotion_possible": bool(
                    str(report.get("outcome") or "").upper() == "VICTORY"
                    and (
                        not is_selected
                        or bool((screen_gate or {}).get("allowed", False))
                    )
                ),
                "profit_factor": _to_metric_float(metrics, "profit_factor"),
                "return_pct": _to_metric_float(metrics, "return_pct"),
                "drawdown_pct": _to_metric_float(metrics, "max_drawdown_pct", default=100.0),
                "reason": _describe_battle_rejection(report, screen_gate if is_selected else None),
            }
        )

    if battle_report:
        challenger = dict(battle_report.get("challenger") or {})
        metrics = dict(challenger.get("metrics") or {})
        promotion_gate = dict((promotion_result or {}).get("promotion_gate") or {})
        rows.append(
            {
                "candidate_id": selected_candidate_id or str(challenger.get("id") or ""),
                "algo_source": engine,
                "horizon": horizon,
                "stage": "full_arena",
                "checkpoint_step": None,
                "checkpoint_path": str(challenger.get("path") or ""),
                "score_proxy": None,
                "score_nemesis": _to_metric_float(
                    dict(battle_report.get("nemesis_validation") or {}).get("score"),
                    default=0.0,
                ),
                "score_arena": _to_metric_float(challenger.get("score")),
                "outcome": str(battle_report.get("outcome") or "UNKNOWN"),
                "promotion_possible": bool(
                    str(battle_report.get("outcome") or "").upper() == "VICTORY"
                    and promotion_gate.get("allowed", False)
                    and str((promotion_result or {}).get("status") or "") == "promoted"
                ),
                "profit_factor": _to_metric_float(metrics, "profit_factor"),
                "return_pct": _to_metric_float(metrics, "return_pct"),
                "drawdown_pct": _to_metric_float(metrics, "max_drawdown_pct", default=100.0),
                "reason": _describe_battle_rejection(battle_report, promotion_gate),
            }
        )

    if family_probe_report:
        summary = dict(family_probe_report.get("family_probe_summary") or {})
        if summary:
            rows.append(
                {
                    "candidate_id": selected_candidate_id,
                    "algo_source": engine,
                    "horizon": horizon,
                    "stage": "family_probe",
                    "checkpoint_step": None,
                    "checkpoint_path": None,
                    "score_proxy": None,
                    "score_nemesis": None,
                    "score_arena": None,
                    "outcome": "VICTORY" if summary.get("allowed", False) else "DEFEAT",
                    "promotion_possible": bool(summary.get("allowed", False)),
                    "profit_factor": None,
                    "return_pct": None,
                    "drawdown_pct": None,
                    "reason": str(summary.get("reason") or "family_probe"),
                }
            )

    return sorted(rows, key=lambda row: (str(row.get("stage") or ""), int(row.get("checkpoint_step") or 0)))


def _build_champion_rejection_report(
    *,
    screen_gate: dict[str, object] | None = None,
    promotion_gate: dict[str, object] | None = None,
    battle_report: dict[str, object] | None = None,
    family_probe_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Explique pourquoi le cycle n'a pas produit de champion.

    Args:
        screen_gate (dict[str, object] | None): Gate du screen tardif.
        promotion_gate (dict[str, object] | None): Gate finale.
        battle_report (dict[str, object] | None): Rapport full Arena.
        family_probe_report (dict[str, object] | None): Rapport famille.

    Returns:
        dict[str, object]: Diagnostic compact et prochain override conseille.
    """
    gate = dict(promotion_gate or screen_gate or {})
    metrics = dict(gate.get("metrics") or {})
    battle = dict(battle_report or {})
    challenger = dict(battle.get("challenger") or {})
    if not metrics:
        metrics = dict(challenger.get("metrics") or {})

    failing_checks = [
        key
        for key, allowed in dict(gate.get("checks") or {}).items()
        if allowed is False
    ]
    family_summary = dict((family_probe_report or {}).get("family_probe_summary") or {})
    nemesis = dict(battle.get("nemesis_validation") or {})
    weak_metrics = _rank_weak_champion_metrics(metrics)
    symbol_blockers = _extract_symbol_blockers(metrics)
    next_overrides = _recommend_next_champion_overrides(
        failing_checks=failing_checks,
        weak_metrics=weak_metrics,
        nemesis_validation=nemesis,
        family_probe_summary=family_summary,
    )
    return {
        "status": "champion_found" if str((battle or {}).get("outcome") or "").upper() == "VICTORY" and gate.get("allowed", False) else "blocked",
        "reason": (
            str(gate.get("reason") or "")
            or str(family_summary.get("reason") or "")
            or _describe_battle_rejection(battle, gate)
        ),
        "failure_mode": str(gate.get("failure_mode") or ""),
        "failing_checks": failing_checks,
        "weak_metrics": weak_metrics,
        "symbol_blockers": symbol_blockers,
        "nemesis_validation": nemesis,
        "family_probe_summary": family_summary,
        "next_overrides": next_overrides,
    }


def _describe_battle_rejection(
    battle_report: dict[str, object],
    gate: dict[str, object] | None = None,
) -> str:
    """Retourne une raison lisible de rejet Arena.

    Args:
        battle_report (dict[str, object]): Rapport Arena.
        gate (dict[str, object] | None): Gate associee.

    Returns:
        str: Raison compacte.
    """
    if gate and str(gate.get("reason") or "").strip():
        return str(gate.get("reason"))
    validation = dict(battle_report.get("validation") or {})
    if str(battle_report.get("outcome") or "").upper() == "VICTORY":
        return "victory"
    if not validation.get("sample_size_ok", True):
        return "sample_size"
    if not validation.get("nemesis_ok", True):
        return "nemesis_failed"
    if not validation.get("inverse_ok", True):
        return "inverse_beats_challenger"
    if not validation.get("directional_ok", True):
        return "directional_collapse"
    return "score_edge_or_profitability"


def _rank_weak_champion_metrics(metrics: dict[str, object]) -> list[dict[str, object]]:
    """Classe les metriques les plus faibles pour piloter le prochain run."""

    candidates = [
        ("profit_factor", _to_metric_float(metrics, "profit_factor"), 1.20, "augmenter_pf"),
        ("return_pct", _to_metric_float(metrics, "return_pct"), 0.0, "forcer_profit_net"),
        ("expectancy_pct", _to_metric_float(metrics, "expectancy_pct"), 0.0, "augmenter_expectancy"),
        ("close_quality_score", _to_metric_float(metrics, "close_quality_score"), 0.40, "corriger_sorties"),
        ("split_runner_capture_rate", _to_metric_float(metrics, "split_runner_capture_rate"), 0.20, "forcer_split_runner"),
        ("pyramid_exit_capture_rate", _to_metric_float(metrics, "pyramid_exit_capture_rate"), 0.20, "corriger_pyramid_exit"),
        ("slbe_capture_rate", _to_metric_float(metrics, "slbe_capture_rate"), 0.30, "renforcer_slbe"),
    ]
    weak = [
        {
            "metric": name,
            "value": value,
            "target": target,
            "override_hint": hint,
        }
        for name, value, target, hint in candidates
        if value < target
    ]
    return sorted(weak, key=lambda item: float(item["value"]) - float(item["target"]))[:5]


def _extract_symbol_blockers(metrics: dict[str, object]) -> list[dict[str, object]]:
    """Extrait les symboles qui bloquent le plus la selection."""

    blockers: list[dict[str, object]] = []
    for symbol, payload in _coerce_metric_mapping(metrics.get("metrics_by_symbol")).items():
        symbol_metrics = _coerce_metric_mapping(payload)
        profit_factor = _to_metric_float(symbol_metrics, "profit_factor")
        return_pct = _to_metric_float(symbol_metrics, "return_pct")
        close_quality = _to_metric_float(symbol_metrics, "close_quality_score")
        if profit_factor < 1.0 or return_pct <= 0.0 or close_quality < 0.25:
            blockers.append(
                {
                    "symbol": symbol,
                    "profit_factor": profit_factor,
                    "return_pct": return_pct,
                    "close_quality_score": close_quality,
                }
            )
    return sorted(
        blockers,
        key=lambda item: (
            _to_metric_float(item, "profit_factor"),
            _to_metric_float(item, "return_pct"),
        ),
    )[:7]


def _recommend_next_champion_overrides(
    *,
    failing_checks: list[str],
    weak_metrics: list[dict[str, object]],
    nemesis_validation: dict[str, object],
    family_probe_summary: dict[str, object],
) -> list[str]:
    """Transforme les echecs en overrides concrets pour le cycle suivant."""

    recommendations: list[str] = []
    weak_names = {str(item.get("metric") or "") for item in weak_metrics}
    if "profit_factor" in weak_names or "return_pct" in weak_names:
        recommendations.append("augmenter le poids profit_factor/expectancy dans le screen et la fitness GA")
    if "split_runner_capture_rate" in weak_names or "split_runner_capture_rate" in failing_checks:
        recommendations.append("surponderer les episodes split + runner profitable dans le replay")
    if "close_quality_score" in weak_names or "close_quality_score" in failing_checks:
        recommendations.append("durcir le curriculum de sorties: TP partiel, SLBE, close tardif")
    if "pyramid_exit_capture_rate" in weak_names:
        recommendations.append("reduire les pyramids decoratives et favoriser les exits profitables")
    if nemesis_validation and not bool(nemesis_validation.get("allowed", True)):
        recommendations.append("injecter les slices Nemesis perdantes dans le prochain proxy GA")
    if family_probe_summary and not bool(family_probe_summary.get("allowed", True)):
        recommendations.append("forcer un univers multi-famille avant full Arena")
    return recommendations[:5]


@contextmanager
def _temporary_env_overrides(overrides: dict[str, str]) -> Any:
    """Applique temporairement des variables d'environnement.

    Args:
        overrides (dict[str, str]): Variables a surcharger.

    Yields:
        Any: Contexte d'execution temporaire.
    """
    previous_values: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous_values[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _evaluate_screen_winner_gate(
    battle_report: dict[str, object],
    config: MuZeroConfigV3,
) -> dict[str, object]:
    """Valide le gagnant du screen avant la full Arena.

    Args:
        battle_report (dict[str, object]): Rapport Arena du checkpoint gagnant.
        config (MuZeroConfigV3): Configuration MuZero.

    Returns:
        dict[str, object]: Verdict intermediaire avant full Arena.
    """
    challenger_metrics = _coerce_metric_mapping(
        _coerce_metric_mapping(battle_report.get("challenger")).get("metrics")
    )
    mechanics_metrics = _coerce_metric_mapping(
        challenger_metrics.get("metrics_by_position_mechanics")
    )
    metrics_by_symbol = _coerce_metric_mapping(challenger_metrics.get("metrics_by_symbol"))
    split_opportunities = int(
        mechanics_metrics.get(
            "split_opportunity_count",
            challenger_metrics.get("split_opportunity_count", 0),
        )
        or 0
    )
    pyramid_opportunities = int(
        mechanics_metrics.get(
            "pyramid_opportunity_count",
            challenger_metrics.get("pyramid_opportunity_count", 0),
        )
        or 0
    )
    slbe_triggered = int(
        mechanics_metrics.get(
            "slbe_triggered",
            challenger_metrics.get("slbe_triggered", 0),
        )
        or 0
    )
    profitable_symbols = 0
    directional_collapse_detected = False
    split_symbol_checks: list[bool] = []
    split_runner_symbol_checks: list[bool] = []
    pyramid_exit_symbol_checks: list[bool] = []
    slbe_symbol_checks: list[bool] = []
    close_symbol_checks: list[bool] = []
    min_profitable_symbols = int(
        getattr(config, "arena_screen_min_profitable_symbols", 5) or 5
    )
    min_symbol_profit_factor = float(
        getattr(config, "arena_screen_min_symbol_profit_factor", 1.0) or 1.0
    )
    min_symbol_return_pct = float(
        getattr(config, "arena_screen_min_symbol_return_pct", 0.0) or 0.0
    )
    min_symbol_split_efficiency = float(
        getattr(config, "arena_screen_min_symbol_split_efficiency", 0.20) or 0.20
    )
    min_symbol_split_runner_capture_rate = float(
        getattr(config, "arena_screen_min_symbol_split_runner_capture_rate", 0.20) or 0.20
    )
    min_symbol_pyramid_exit_capture_rate = float(
        getattr(config, "arena_screen_min_symbol_pyramid_exit_capture_rate", 0.20) or 0.20
    )
    min_symbol_slbe_capture_rate = float(
        getattr(config, "arena_screen_min_symbol_slbe_capture_rate", 0.25) or 0.25
    )
    min_symbol_close_quality_score = float(
        getattr(config, "arena_screen_min_symbol_close_quality_score", 0.20) or 0.20
    )
    min_symbol_close_events = int(
        getattr(config, "arena_screen_min_symbol_close_events", 6) or 6
    )
    for symbol_metrics_raw in metrics_by_symbol.values():
        symbol_metrics = _coerce_metric_mapping(symbol_metrics_raw)
        symbol_mechanics = _coerce_metric_mapping(
            symbol_metrics.get("metrics_by_position_mechanics")
        )
        if (
            _to_metric_float(symbol_metrics, "profit_factor") >= min_symbol_profit_factor
            or _to_metric_float(symbol_metrics, "return_pct") > min_symbol_return_pct
        ):
            profitable_symbols += 1
        if bool(symbol_metrics.get("directional_collapse", False)):
            directional_collapse_detected = True
        symbol_split_executed = int(
            symbol_mechanics.get("split_executed", symbol_metrics.get("split_executed", 0)) or 0
        )
        if symbol_split_executed >= 3:
            split_symbol_checks.append(
                _to_metric_float(
                    symbol_mechanics,
                    "split_efficiency",
                    default=_to_metric_float(symbol_metrics, "split_efficiency", default=0.0),
                ) >= min_symbol_split_efficiency
            )
            split_runner_symbol_checks.append(
                _to_metric_float(
                    symbol_mechanics,
                    "split_runner_capture_rate",
                    default=_to_metric_float(
                        symbol_metrics,
                        "split_runner_capture_rate",
                        default=0.0,
                    ),
                ) >= min_symbol_split_runner_capture_rate
            )
        symbol_pyramids_opened = int(
            symbol_mechanics.get("pyramids_opened", symbol_metrics.get("pyramids_opened", 0)) or 0
        )
        if symbol_pyramids_opened >= 3:
            pyramid_exit_symbol_checks.append(
                _to_metric_float(
                    symbol_mechanics,
                    "pyramid_exit_capture_rate",
                    default=_to_metric_float(
                        symbol_metrics,
                        "pyramid_exit_capture_rate",
                        default=0.0,
                    ),
                ) >= min_symbol_pyramid_exit_capture_rate
            )
        symbol_slbe_triggered = int(
            symbol_mechanics.get("slbe_triggered", symbol_metrics.get("slbe_triggered", 0)) or 0
        )
        if symbol_slbe_triggered >= 3:
            slbe_symbol_checks.append(
                _to_metric_float(
                    symbol_mechanics,
                    "slbe_capture_rate",
                    default=_to_metric_float(symbol_metrics, "slbe_capture_rate", default=0.0),
                ) >= min_symbol_slbe_capture_rate
            )
        symbol_close_events = int(
            symbol_mechanics.get("close_winner_count", symbol_metrics.get("close_winner_count", 0)) or 0
        ) + int(
            symbol_mechanics.get("close_loser_count", symbol_metrics.get("close_loser_count", 0)) or 0
        )
        if symbol_close_events >= min_symbol_close_events:
            close_symbol_checks.append(
                _to_metric_float(
                    symbol_mechanics,
                    "close_quality_score",
                    default=_to_metric_float(symbol_metrics, "close_quality_score", default=0.0),
                ) >= min_symbol_close_quality_score
            )
    checks = {
        "screen_victory": str(battle_report.get("outcome") or "").strip().upper()
        == "VICTORY",
        "profit_factor": _to_metric_float(challenger_metrics, "profit_factor") >= float(
            getattr(config, "arena_screen_min_profit_factor", 1.20) or 1.20
        ),
        "return_pct": _to_metric_float(challenger_metrics, "return_pct") > float(
            getattr(config, "arena_screen_min_return_pct", 0.0) or 0.0
        ),
        "expectancy_pct": _to_metric_float(challenger_metrics, "expectancy_pct") > float(
            getattr(config, "arena_screen_min_expectancy_pct", 0.0) or 0.0
        ),
        "positive_episode_rate": _to_metric_float(challenger_metrics, "positive_episode_rate") >= float(
            getattr(config, "arena_screen_min_positive_episode_rate", 55.0) or 55.0
        ),
        "directional_bias": str(challenger_metrics.get("directional_bias") or "inactive").strip().lower()
        == "balanced",
        "hold_drag_score": _to_metric_float(
            mechanics_metrics,
            "hold_drag_score",
            default=_to_metric_float(challenger_metrics, "hold_drag_score"),
        ) <= float(
            getattr(config, "arena_screen_max_hold_drag_score", 0.80) or 0.80
        ),
        "close_quality_score": _to_metric_float(
            mechanics_metrics,
            "close_quality_score",
            default=_to_metric_float(challenger_metrics, "close_quality_score"),
        ) >= float(
            getattr(config, "arena_screen_min_close_quality_score", 0.35) or 0.35
        ),
        "split_efficiency": (
            True
            if split_opportunities < int(getattr(config, "arena_screen_min_split_opportunities", 3) or 3)
            else _to_metric_float(
                mechanics_metrics,
                "split_efficiency",
                default=_to_metric_float(challenger_metrics, "split_efficiency"),
            ) >= float(
                getattr(config, "arena_screen_min_split_efficiency", 0.35) or 0.35
            )
        ),
        "split_runner_capture_rate": (
            True
            if split_opportunities < int(getattr(config, "arena_screen_min_split_opportunities", 3) or 3)
            else _to_metric_float(
                mechanics_metrics,
                "split_runner_capture_rate",
                default=_to_metric_float(challenger_metrics, "split_runner_capture_rate"),
            ) >= float(
                getattr(config, "arena_screen_min_split_runner_capture_rate", 0.20) or 0.20
            )
        ),
        "pyramid_efficiency": (
            True
            if pyramid_opportunities < int(getattr(config, "arena_screen_min_pyramid_opportunities", 3) or 3)
            else _to_metric_float(
                mechanics_metrics,
                "pyramid_efficiency",
                default=_to_metric_float(challenger_metrics, "pyramid_efficiency"),
            ) >= float(
                getattr(config, "arena_screen_min_pyramid_efficiency", 0.35) or 0.35
            )
        ),
        "pyramid_exit_capture_rate": (
            True
            if pyramid_opportunities < int(getattr(config, "arena_screen_min_pyramid_opportunities", 3) or 3)
            else _to_metric_float(
                mechanics_metrics,
                "pyramid_exit_capture_rate",
                default=_to_metric_float(challenger_metrics, "pyramid_exit_capture_rate"),
            ) >= float(
                getattr(config, "arena_screen_min_pyramid_exit_capture_rate", 0.20) or 0.20
            )
        ),
        "slbe_capture_rate": (
            True
            if slbe_triggered < int(getattr(config, "arena_screen_min_slbe_triggered", 3) or 3)
            else _to_metric_float(
                mechanics_metrics,
                "slbe_capture_rate",
                default=_to_metric_float(challenger_metrics, "slbe_capture_rate"),
            ) >= float(
                getattr(config, "arena_screen_min_slbe_capture_rate", 0.30) or 0.30
            )
        ),
        "inverse_edge": (
            True
            if "edge_vs_inverse_pf" not in battle_report
            else (
                float(battle_report.get("edge_vs_inverse_pf", 0.0) or 0.0) > 0.0
                and float(battle_report.get("edge_vs_inverse_return_pct", 0.0) or 0.0) > 0.0
                and int(battle_report.get("edge_vs_inverse_profitable_symbols", 0) or 0)
                >= int(getattr(config, "arena_inverse_min_profitable_symbols", 5) or 5)
            )
        ),
        "profitable_symbols": profitable_symbols >= min_profitable_symbols,
        "directional_collapse_by_symbol": not directional_collapse_detected,
        "split_efficiency_by_symbol": all(split_symbol_checks) if split_symbol_checks else True,
        "split_runner_capture_rate_by_symbol": all(split_runner_symbol_checks) if split_runner_symbol_checks else True,
        "pyramid_exit_capture_rate_by_symbol": all(pyramid_exit_symbol_checks) if pyramid_exit_symbol_checks else True,
        "slbe_capture_rate_by_symbol": all(slbe_symbol_checks) if slbe_symbol_checks else True,
        "close_quality_score_by_symbol": all(close_symbol_checks) if close_symbol_checks else True,
    }
    failure_reason = next((name for name, passed in checks.items() if not passed), "eligible")
    return {
        "allowed": all(checks.values()),
        "status": "eligible" if all(checks.values()) else "blocked",
        "reason": failure_reason,
        "checks": checks,
        "metrics": {
            **challenger_metrics,
            "profitable_symbols": profitable_symbols,
            "directional_collapse_detected": directional_collapse_detected,
        },
    }


def _build_terminal_failure_summary(
    *,
    run_id: str | None,
    engine: str,
    horizon: str,
    family: str | None,
    feature_profile_name: str | None,
    ga_trial: str | None,
    ga_campaign_id: str | None,
    ga_scope: str | None,
    ga_parent_champion_id: str | None,
    trial_mode: str | None,
    dataset_id: str | None,
    dataset_source: str | None,
    focus_symbols: list[str],
    gate_profile: str,
    latest_checkpoint: str | None,
    resume_source: str | None,
    artifact_compatibility: dict[str, Any],
    reason: str,
    failure_mode: str = "artifact_incompatible",
    lineage: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Construit un resume terminal de blocage avant le train MuZero."""

    return {
        "run_id": run_id,
        "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
        "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
        "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
        "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
        "engine": engine,
        "horizon": horizon,
        "family": family,
        "feature_profile": feature_profile_name,
        "ga_trial": ga_trial,
        "ga_campaign_id": ga_campaign_id,
        "ga_scope": ga_scope,
        "ga_parent_champion_id": ga_parent_champion_id,
        "seed_parent_champion_id": ga_parent_champion_id,
        "trial_mode": trial_mode,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "focus_symbols": focus_symbols,
        "gate_profile": gate_profile,
        "terminal_status": "failed",
        "failed_step": "checkpoint_resume",
        "failure_mode": failure_mode,
        "promotion_gate": {},
        "metrics": {},
        "metrics_by_symbol": {},
        "metrics_by_position_mechanics": {},
        "training_metrics": {},
        "challenger_path": None,
        "latest_checkpoint": latest_checkpoint,
        "battle_report_path": None,
        "live_comparison": {},
        "resume_source": resume_source,
        "artifact_compatibility": dict(artifact_compatibility or {}),
        "checkpoint_schema_version": artifact_compatibility.get("schema_version"),
        "lineage": dict(lineage or {}),
        "artifact_state": {
            "arena_report_present": False,
            "battle_report_present": False,
            "promotion_present": False,
            "candidate_checkpoint_present": False,
            "latest_checkpoint_present": bool(latest_checkpoint),
        },
        "latest_candidate": None,
        "latest_verdict": {
            "status": "failed",
            "reason": reason,
            "failure_mode": failure_mode,
        },
        "reason": reason,
    }


def _build_training_metrics_payload(
    *,
    base_metrics: dict[str, object] | None,
    family: str | None,
    dataset_id: str | None,
    dataset_source: str | None,
    feature_profile_name: str | None,
    mechanics_profile_version: str | None,
    dataset_coverage: dict[str, Any],
    focus_symbols: list[str],
    ga_campaign_id: str | None,
    ga_trial: str | None,
    ga_scope: str | None,
    ga_parent_champion_id: str | None,
    resume_source: str | None,
    checkpoint_schema_version: int | None,
    artifact_compatibility: dict[str, Any] | None,
    lineage: dict[str, object] | None,
) -> dict[str, object]:
    """Construit un bloc de metriques stable pour la promotion MuZero."""

    payload = {
        **dict(base_metrics or {}),
        "family": family,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "feature_profile": feature_profile_name,
        "mechanics_profile_version": mechanics_profile_version,
        "dataset_coverage": dict(dataset_coverage or {}),
        "focus_symbols": list(focus_symbols),
        "ga_campaign_id": ga_campaign_id,
        "ga_trial": ga_trial,
        "ga_scope": ga_scope,
        "ga_parent_champion_id": ga_parent_champion_id,
        "seed_parent_champion_id": ga_parent_champion_id,
        "resume_source": resume_source,
        "checkpoint_schema_version": checkpoint_schema_version,
        "artifact_compatibility": dict(artifact_compatibility or {}),
        "lineage": dict(lineage or {}),
    }
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }


def _resolve_resume_checkpoint(
    *,
    agent: JAXMuZeroAgent,
    explicit_resume_path: str | None,
    ga_seed_checkpoint_path: str | None,
    latest_path: Path,
    mechanics_only_mode: bool,
    archive_root: Path,
) -> dict[str, Any]:
    """Charge le meilleur checkpoint de reprise compatible pour MuZero."""

    last_incompatible: dict[str, Any] | None = None
    for candidate in _build_resume_candidates(
        explicit_resume_path=explicit_resume_path,
        ga_seed_checkpoint_path=ga_seed_checkpoint_path,
        latest_path=latest_path,
    ):
        candidate_source = str(candidate["source"])
        candidate_path = Path(candidate["path"])
        if not candidate_path.exists():
            continue

        compatibility = agent.inspect_checkpoint(str(candidate_path))
        if compatibility.get("allowed", False):
            agent.load(str(candidate_path))
            return {
                "resume_path": str(candidate_path),
                "resume_source": candidate_source,
                "artifact_compatibility": compatibility,
                "loaded": True,
            }

        last_incompatible = {
            "resume_path": str(candidate_path),
            "resume_source": candidate_source,
            "artifact_compatibility": compatibility,
            "loaded": False,
        }
        if candidate_source == "ga_seed_checkpoint" or (
            mechanics_only_mode and candidate_source == "explicit_resume"
        ):
            return last_incompatible

        if candidate_source == "latest":
            archive_report = archive_muzero_artifacts(
                archive_root=archive_root,
                paths=[candidate_path],
                reason="checkpoint_incompatible",
                metadata={
                    "source": candidate_source,
                    "compatibility": compatibility,
                },
            )
            logger.warning(
                "Checkpoint latest MuZero archive apres incompatibilite: %s",
                archive_report.get("archive_dir"),
            )
            append_training_log(
                (
                    "MuZero: checkpoint latest incompatible archive avant cold start: "
                    f"{compatibility.get('reason')}"
                ),
                level="WARNING",
                source="muzero",
            )

    return {
        "resume_path": None,
        "resume_source": "cold_start",
        "artifact_compatibility": last_incompatible.get("artifact_compatibility") if last_incompatible else {},
        "loaded": False,
    }


def main() -> dict[str, object]:
    """Orchestre l'entrainement MuZero d'un horizon strategique."""
    config = MuZeroConfigV3()
    horizon = config.horizon
    engine = "muzero"
    step_name = f"muzero_{horizon}"
    initial_family = str(getattr(config, "model_family", "")).strip() or None
    feature_profile_name = (
        str((getattr(config, "feature_profile", {}) or {}).get("profile_name") or "").strip() or None
    )
    mechanics_profile_version = str(getattr(config, "mechanics_profile_version", "")).strip() or None
    dataset_id = str(getattr(config, "dataset_id", "")).strip() or None
    dataset_source = str(getattr(config, "dataset_source", "")).strip() or None
    dataset_coverage = dict(getattr(config, "dataset_coverage", {}) or {})
    ga_status = str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None
    ga_generation = (
        int(os.getenv("TRAINING_GA_GENERATION", "0"))
        if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
        else None
    )
    ga_trial = str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None
    ga_campaign_id = str(os.getenv("TRAINING_GA_CAMPAIGN_ID", "")).strip() or None
    ga_scope = str(os.getenv("TRAINING_GA_SCOPE", "")).strip() or None
    ga_parent_champion_id = str(os.getenv("TRAINING_GA_PARENT_CHAMPION_ID", "")).strip() or None
    ga_defer_promotion = str(os.getenv("TRAINING_GA_DEFER_PROMOTION", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    explicit_resume_path = str(os.getenv("TRAINING_RESUME_CHECKPOINT_PATH", "")).strip() or None
    ga_seed_checkpoint_path = str(os.getenv("TRAINING_GA_SEED_CHECKPOINT_PATH", "")).strip() or None
    ga_genome_raw = str(os.getenv("TRAINING_GA_GENOME_JSON", "")).strip()
    ga_genome: dict[str, object] = {}
    if ga_genome_raw:
        try:
            decoded_genome = json.loads(ga_genome_raw)
        except json.JSONDecodeError:
            logger.warning("Genotype GA invalide ignore pour le run %s.", step_name)
        else:
            if isinstance(decoded_genome, dict):
                ga_genome = decoded_genome
    trial_mode = str(os.getenv("TRAINING_TRIAL_MODE", "")).strip() or None
    trial_cost_profile = str(os.getenv("TRAINING_TRIAL_COST_PROFILE", "")).strip() or None
    gate_profile = str(os.getenv("TRAINING_GATE_PROFILE", "")).strip() or "standard"
    mechanics_only_mode = bool(
        ga_scope == "mechanics_only"
        and ga_seed_checkpoint_path
        and ga_status in {"proxy", "proxy_ga", "final", "full"}
    )
    focus_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in config.symbols if str(symbol).strip()))
    replay_cache_key = f"{engine}:{horizon}:{initial_family or 'global'}:{mechanics_profile_version or 'default'}"
    gold_precheck_enabled = _is_gold_proxy_precheck_enabled(
        gate_profile=gate_profile,
        trial_mode=trial_mode,
        focus_symbols=focus_symbols,
    )
    gold_precheck_step = max(1, int(os.getenv("MUZERO_GOLD_PRECHECK_STEP", "3000")))
    gold_precheck_games = max(1, int(os.getenv("MUZERO_GOLD_PRECHECK_GAMES", "6")))
    logger.info("Demarrage MuZero horizon=%s | timeframe=%s", horizon, config.primary_timeframe)
    logger.info("Peripheriques JAX: %s", jax.devices())
    runtime_devices = ", ".join(str(device) for device in jax.devices())
    runtime_message = (
        f"MuZero {horizon}: runtime pid={os.getpid()} "
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')} "
        f"JAX_PLATFORMS={os.getenv('JAX_PLATFORMS', 'auto')} "
        f"devices=[{runtime_devices}]"
    )
    logger.info(runtime_message)
    collection_budget_message = (
        f"MuZero {horizon}: budget collecte "
        f"sims={int(getattr(config, 'collection_num_simulations', config.num_simulations) or config.num_simulations)} "
        f"max_moves={int(getattr(config, 'collection_max_moves', config.max_moves) or config.max_moves)} "
        f"timeout_episode={float(getattr(config, 'collection_max_episode_seconds', 0.0) or 0.0):.0f}s "
        f"timeout_step={float(getattr(config, 'collection_max_step_seconds', 0.0) or 0.0):.0f}s."
    )
    logger.info(collection_budget_message)
    logger.info("Inventaire historique: %s", build_inventory_report())
    logger.info("Univers MuZero: %s", config.symbols)
    append_training_log(
        f"MuZero {horizon} demarre sur {len(config.symbols)} symboles.",
        source="muzero",
    )
    append_training_log(collection_budget_message, source="muzero")
    append_training_log(runtime_message, source="muzero")
    send_training_horizon_started(horizon, len(config.symbols))
    set_gold_precheck(None)
    mark_step_running(
        step_name,
        engine=engine,
        phase="initialisation",
        horizon=horizon,
        family=initial_family,
        symbol_total=len(config.symbols),
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=feature_profile_name,
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
        ga_campaign_id=ga_campaign_id,
        ga_scope=ga_scope,
        ga_parent_champion_id=ga_parent_champion_id,
        trial_mode=trial_mode,
        trial_cost_profile=trial_cost_profile,
        focus_symbols=focus_symbols,
        gate_profile=gate_profile,
        replay_cache_status="warming",
        replay_cache_key=replay_cache_key,
        replay_cache_entries=0,
        replay_cache_source="memoire",
        dataset_coverage=dataset_coverage,
    )
    record_training_dataset(
        dict(getattr(config, "dataset_descriptor", {}) or {}),
        metadata={
            "engine": engine,
            "run_trigger": str(os.getenv("TRAINING_RUN_TRIGGER", "manual") or "manual"),
            "ga_status": ga_status,
            "ga_generation": str(ga_generation) if ga_generation is not None else None,
            "ga_trial": ga_trial,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "gate_profile": gate_profile,
        },
    )

    agent = JAXMuZeroAgent(config)
    if getattr(config, "use_league", False) and getattr(agent, "league_buffer", None) is not None:
        logger.info(
            "[MuZero %s] Initialisation de l'AlphaStar League. %d trajectoires historiques chargees.",
            horizon,
            len(agent.league_buffer)
        )
    weights_dir = Path(config.weights_path)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    latest_path = weights_dir / f"muzero_{horizon}_latest.pkl"
    resume_resolution = _resolve_resume_checkpoint(
        agent=agent,
        explicit_resume_path=explicit_resume_path,
        ga_seed_checkpoint_path=ga_seed_checkpoint_path,
        latest_path=latest_path,
        mechanics_only_mode=mechanics_only_mode,
        archive_root=weights_dir / "archive" / "incompatible" / horizon,
    )
    resume_source = str(resume_resolution.get("resume_source") or "cold_start")
    resume_checkpoint_path = (
        str(resume_resolution.get("resume_path") or "").strip() or None
    )
    artifact_compatibility = dict(resume_resolution.get("artifact_compatibility") or {})
    checkpoint_schema_version = artifact_compatibility.get("schema_version")
    lineage = _build_lineage(
        resume_source=resume_source,
        resume_checkpoint_path=resume_checkpoint_path,
        ga_parent_champion_id=ga_parent_champion_id,
        ga_campaign_id=ga_campaign_id,
        ga_trial=ga_trial,
        ga_scope=ga_scope,
        ga_generation=ga_generation,
    )
    if resume_checkpoint_path:
        logger.info("Reprise MuZero depuis %s (%s).", resume_checkpoint_path, resume_source)
    else:
        logger.info("MuZero demarre a froid (%s).", resume_source)

    if mechanics_only_mode and resume_source != "ga_seed_checkpoint":
        failure_reason = (
            "Campagne GA seedee bloquee: aucun champion MuZero seed compatible n'a ete charge."
        )
        append_training_log(failure_reason, level="ERROR", source="muzero")
        active_run_id = str(load_training_status().get("run_id") or "").strip() or None
        mark_step_running(
            step_name,
            engine=engine,
            phase="initialisation",
            horizon=horizon,
            family=initial_family,
            symbol_total=len(config.symbols),
            dataset_id=dataset_id,
            dataset_source=dataset_source,
            feature_profile=feature_profile_name,
            mechanics_profile_version=mechanics_profile_version,
            ga_status=ga_status,
            ga_generation=ga_generation,
            ga_trial=ga_trial,
            ga_campaign_id=ga_campaign_id,
            ga_scope=ga_scope,
            ga_parent_champion_id=ga_parent_champion_id,
            seed_parent_champion_id=ga_parent_champion_id,
            trial_mode=trial_mode,
            trial_cost_profile=trial_cost_profile,
            focus_symbols=focus_symbols,
            gate_profile=gate_profile,
            resume_source=resume_source,
            checkpoint_schema_version=checkpoint_schema_version,
            artifact_compatibility=artifact_compatibility,
            lineage=lineage,
            replay_cache_status="warming",
            replay_cache_key=replay_cache_key,
            replay_cache_entries=0,
            replay_cache_source="memoire",
            dataset_coverage=dataset_coverage,
        )
        terminal_summary = _build_terminal_failure_summary(
            run_id=active_run_id,
            engine=engine,
            horizon=horizon,
            family=initial_family,
            feature_profile_name=feature_profile_name,
            ga_trial=ga_trial,
            ga_campaign_id=ga_campaign_id,
            ga_scope=ga_scope,
            ga_parent_champion_id=ga_parent_champion_id,
            trial_mode=trial_mode,
            dataset_id=dataset_id,
            dataset_source=dataset_source,
            focus_symbols=focus_symbols,
            gate_profile=gate_profile,
            latest_checkpoint=resume_checkpoint_path,
            resume_source=resume_source,
            artifact_compatibility=artifact_compatibility,
            reason=failure_reason,
            lineage=lineage,
        )
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.error("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        raise RuntimeError(failure_reason)

    mark_step_running(
        step_name,
        engine=engine,
        phase="initialisation",
        horizon=horizon,
        family=initial_family,
        symbol_total=len(config.symbols),
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=feature_profile_name,
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
        ga_campaign_id=ga_campaign_id,
        ga_scope=ga_scope,
        ga_parent_champion_id=ga_parent_champion_id,
        seed_parent_champion_id=ga_parent_champion_id,
        trial_mode=trial_mode,
        trial_cost_profile=trial_cost_profile,
        focus_symbols=focus_symbols,
        gate_profile=gate_profile,
        resume_source=resume_source,
        checkpoint_schema_version=checkpoint_schema_version,
        artifact_compatibility=artifact_compatibility,
        lineage=lineage,
        replay_cache_status="warming",
        replay_cache_key=replay_cache_key,
        replay_cache_entries=0,
        replay_cache_source="memoire",
        dataset_coverage=dataset_coverage,
    )

    games_per_symbol = int(os.getenv("MUZERO_GAMES_PER_SYMBOL", "12"))
    valid_symbols: list[str] = []
    total_games = 0

    family = infer_family_from_symbols(config.symbols, family=getattr(config, "model_family", None))
    feature_profile = resolve_feature_profile(horizon, family)
    dataset_source = str(getattr(config, "dataset_source", "csv") or "csv")
    dataset_descriptor = dict(getattr(config, "dataset_descriptor", {}) or {})
    dataset_id = str(dataset_descriptor.get("dataset_id") or "")
    start_time = datetime.now()
    last_metrics: dict[str, object] | None = None
    last_optimization_step = 0
    policy_precheck_history: deque[dict[str, object]] = deque(
        maxlen=max(
            int(getattr(config, "policy_precheck_window_size", 500) or 500),
            int(getattr(config, "arena_screen_recent_steps", 2500) or 2500),
        )
    )
    policy_precheck_payload: dict[str, object] | None = None
    policy_precheck_executed = False
    reanalyze_games_total = 0
    reanalyze_positions_total = 0
    gold_precheck_payload: dict[str, object] | None = None
    gold_precheck_executed = False
    killed_after_precheck = False
    directional_collapse_payload: dict[str, object] | None = None
    killed_after_directional_collapse = False
    killed_after_policy_precheck = False
    seed_viability_payload: dict[str, object] | None = None
    killed_after_seed_viability = False
    plateau_payload: dict[str, object] | None = None
    plateau_checkpoint_saved = False
    stopped_for_plateau = False

    if mechanics_only_mode:
        valid_symbols = [str(symbol).strip() for symbol in config.symbols if str(symbol).strip()]
        total_games = 0
        last_metrics = {
            "mode": "seeded_ga_mechanics_only",
            "seed_checkpoint_path": ga_seed_checkpoint_path,
            "feature_profile": str(feature_profile.get("profile_name") or "").strip() or None,
            "mechanics_profile_version": mechanics_profile_version,
            "resume_source": resume_source,
            "checkpoint_schema_version": checkpoint_schema_version,
            "artifact_compatibility": artifact_compatibility,
            "lineage": lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
        }
        append_training_log(
            (
                f"MuZero {horizon}: mode GA seede mecanique active. "
                f"Checkpoint fixe={ga_seed_checkpoint_path}."
            ),
            source="muzero",
        )
    else:
        logger.info("Phase 1 - collecte historique par self-play guide")
        collection_parallel_games = max(
            1,
            min(
                games_per_symbol,
                int(getattr(config, "collection_parallel_games", 1) or 1),
            ),
        )
        if collection_parallel_games > 1:
            append_training_log(
                (
                    f"MuZero {horizon}: collecte parallele active "
                    f"({collection_parallel_games} parties simultanees par symbole)."
                ),
                source="muzero",
            )
        for symbol_index, symbol in enumerate(config.symbols, start=1):
            env = build_environment(
                symbol,
                config,
                training_progress_step=int(agent.training_step_count),
                for_collection=True,
            )
            if env is None:
                continue
            valid_symbols.append(symbol)
            append_training_log(
                f"MuZero {horizon}: collecte sur {symbol} ({symbol_index}/{len(config.symbols)}).",
                source="muzero",
            )
            if collection_parallel_games > 1:
                with ThreadPoolExecutor(
                    max_workers=collection_parallel_games,
                    thread_name_prefix=f"muzero-collect-{symbol_index}",
                ) as collection_pool:
                    futures = [
                        collection_pool.submit(
                            _collect_single_muzero_game,
                            symbol=symbol,
                            config=config,
                            params=agent.params,
                            opt_state=agent.opt_state,
                            training_step_count=int(agent.training_step_count),
                        )
                        for _ in range(games_per_symbol)
                    ]
                    for game_index, future in enumerate(futures):
                        mark_step_running(
                            step_name,
                            engine=engine,
                            phase="collecte",
                            horizon=horizon,
                            family=initial_family,
                            symbol=symbol,
                            symbol_index=symbol_index,
                            symbol_total=len(config.symbols),
                            part_index=game_index + 1,
                            part_total=games_per_symbol,
                            dataset_id=dataset_id,
                            dataset_source=dataset_source,
                            feature_profile=feature_profile_name,
                            mechanics_profile_version=mechanics_profile_version,
                            ga_status=ga_status,
                            ga_generation=ga_generation,
                            ga_trial=ga_trial,
                            ga_campaign_id=ga_campaign_id,
                            ga_scope=ga_scope,
                            ga_parent_champion_id=ga_parent_champion_id,
                            trial_mode=trial_mode,
                            trial_cost_profile=trial_cost_profile,
                            focus_symbols=focus_symbols,
                            gate_profile=gate_profile,
                            replay_cache_status="warming",
                            replay_cache_key=replay_cache_key,
                            replay_cache_entries=agent.replay_buffer.size,
                            replay_cache_source="memoire",
                            dataset_coverage=dataset_coverage,
                        )
                        game, summary, stopped_reason = future.result()
                        if game is None:
                            continue
                        if len(game) > 0:
                            agent.replay_buffer.save_game(game)
                            if getattr(config, "use_league", False) and getattr(agent, "league_buffer", None) is not None:
                                import random
                                profit_factor = float((game.metadata or {}).get("profit_factor", 1.0) or 1.0)
                                if profit_factor > 1.2 or random.random() < 0.1:
                                    agent.league_buffer.save_game(game, f"champion_{horizon}")
                        total_games += 1
                        if stopped_reason:
                            logger.warning(
                                "[%s] %s partie %s/%s interrompue (%s).",
                                horizon,
                                symbol,
                                game_index + 1,
                                games_per_symbol,
                                stopped_reason,
                            )
                            append_training_log(
                                f"MuZero {horizon}: {symbol} partie {game_index + 1}/{games_per_symbol} interrompue ({stopped_reason}).",
                                level="WARNING",
                                source="muzero",
                            )
                        logger.info(
                            "[%s] %s partie %s/%s | return=%.2f%% | trades=%s | buffer=%s",
                            horizon,
                            symbol,
                            game_index + 1,
                            games_per_symbol,
                            summary.get("return_pct", 0.0),
                            summary.get("total_trades", 0),
                            agent.replay_buffer.size,
                        )
                continue
            for game_index in range(games_per_symbol):
                def _report_collection_heartbeat(heartbeat: dict[str, object]) -> None:
                    """Diffuse un heartbeat de collecte pour la supervision.

                    Args:
                        heartbeat (dict[str, object]): Metriques intra-partie
                            publiees par l'agent MuZero.
                    """
                    mark_step_running(
                        step_name,
                        engine=engine,
                        phase="collecte",
                        horizon=horizon,
                        family=initial_family,
                        symbol=symbol,
                        symbol_index=symbol_index,
                        symbol_total=len(config.symbols),
                        part_index=game_index + 1,
                        part_total=games_per_symbol,
                        episode_step_current=int(heartbeat.get("steps", 0) or 0),
                        episode_step_total=int(
                            heartbeat.get("max_moves", getattr(env, "max_steps_per_episode", config.max_moves))
                            or getattr(env, "max_steps_per_episode", config.max_moves)
                        ),
                        episode_elapsed_seconds=float(heartbeat.get("elapsed_seconds", 0.0) or 0.0),
                        dataset_id=dataset_id,
                        dataset_source=dataset_source,
                        feature_profile=feature_profile_name,
                        mechanics_profile_version=mechanics_profile_version,
                        ga_status=ga_status,
                        ga_generation=ga_generation,
                        ga_trial=ga_trial,
                        ga_campaign_id=ga_campaign_id,
                        ga_scope=ga_scope,
                        ga_parent_champion_id=ga_parent_champion_id,
                        trial_mode=trial_mode,
                        trial_cost_profile=trial_cost_profile,
                        focus_symbols=focus_symbols,
                        gate_profile=gate_profile,
                        replay_cache_status="warming",
                        replay_cache_key=replay_cache_key,
                        replay_cache_entries=agent.replay_buffer.size,
                        replay_cache_source="memoire",
                        dataset_coverage=dataset_coverage,
                    )

                mark_step_running(
                    step_name,
                    engine=engine,
                    phase="collecte",
                    horizon=horizon,
                    family=initial_family,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    symbol_total=len(config.symbols),
                    part_index=game_index + 1,
                    part_total=games_per_symbol,
                    dataset_id=dataset_id,
                    dataset_source=dataset_source,
                    feature_profile=feature_profile_name,
                    mechanics_profile_version=mechanics_profile_version,
                    ga_status=ga_status,
                    ga_generation=ga_generation,
                    ga_trial=ga_trial,
                    ga_campaign_id=ga_campaign_id,
                    ga_scope=ga_scope,
                    ga_parent_champion_id=ga_parent_champion_id,
                    trial_mode=trial_mode,
                    trial_cost_profile=trial_cost_profile,
                    focus_symbols=focus_symbols,
                    gate_profile=gate_profile,
                    replay_cache_status="warming",
                    replay_cache_key=replay_cache_key,
                    replay_cache_entries=agent.replay_buffer.size,
                    replay_cache_source="memoire",
                    dataset_coverage=dataset_coverage,
                )
                # La collecte doit suivre l'etape reelle de reprise pour
                # conserver un curriculum coherent avec le checkpoint seed.
                env.training_progress_step = int(agent.training_step_count)
                game = agent.play_game(
                    env,
                    exploration=True,
                    progress_callback=_report_collection_heartbeat,
                )
                summary = env.get_summary()
                total_games += 1
                stopped_reason = str(game.metadata.get("stopped_reason") or "").strip()
                if stopped_reason:
                    logger.warning(
                        "[%s] %s partie %s/%s interrompue (%s).",
                        horizon,
                        symbol,
                        game_index + 1,
                        games_per_symbol,
                        stopped_reason,
                    )
                    append_training_log(
                        f"MuZero {horizon}: {symbol} partie {game_index + 1}/{games_per_symbol} interrompue ({stopped_reason}).",
                        level="WARNING",
                        source="muzero",
                    )
                logger.info(
                    "[%s] %s partie %s/%s | return=%.2f%% | trades=%s | buffer=%s",
                    horizon,
                    symbol,
                    game_index + 1,
                    games_per_symbol,
                    summary.get("return_pct", 0.0),
                    summary.get("total_trades", 0),
                    agent.replay_buffer.size,
                )

        if not valid_symbols:
            raise RuntimeError("Aucun symbole valide pour MuZero.")

        logger.info("Phase 2 - optimisation profonde (%s steps)", config.training_steps)
        append_training_log(
            f"MuZero {horizon}: optimisation profonde sur {config.training_steps} steps.",
            source="muzero",
        )
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="muzero-batch-prefetch") as prefetch_pool:
            current_prepared_step = agent.prepare_training_step()

            for step in range(1, config.training_steps + 1):
                last_optimization_step = step
                mark_step_running(
                    step_name,
                    engine=engine,
                    phase="optimisation",
                    horizon=horizon,
                    family=family,
                    symbol_total=len(valid_symbols),
                    training_step_current=step,
                    training_step_total=config.training_steps,
                    dataset_id=dataset_id,
                    dataset_source=dataset_source,
                    feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
                    mechanics_profile_version=mechanics_profile_version,
                    ga_status=ga_status,
                    ga_generation=ga_generation,
                    ga_trial=ga_trial,
                    ga_campaign_id=ga_campaign_id,
                    ga_scope=ga_scope,
                    ga_parent_champion_id=ga_parent_champion_id,
                    trial_mode=trial_mode,
                    trial_cost_profile=trial_cost_profile,
                    focus_symbols=focus_symbols,
                    gate_profile=gate_profile,
                    replay_cache_status="memoire",
                    replay_cache_key=f"{engine}:{horizon}:{family}:{mechanics_profile_version or 'default'}",
                    replay_cache_entries=agent.replay_buffer.size,
                    replay_cache_source="memoire",
                    dataset_coverage=dataset_coverage,
                )
                if current_prepared_step is None:
                    logger.warning("MuZero sans batch suffisant, arret a l'etape %s.", step)
                    append_training_log(
                        f"MuZero {horizon}: arret anticipe a l'etape {step} faute de batch suffisant.",
                        level="WARNING",
                        source="muzero",
                    )
                    break

                should_reanalyze = (
                    int(getattr(config, "reanalyze_every_steps", 0) or 0) > 0
                    and step % int(getattr(config, "reanalyze_every_steps", 0) or 1) == 0
                )
                if should_reanalyze:
                    metrics = agent.train_step(current_prepared_step)
                    current_prepared_step = None
                else:
                    next_prepared_future = prefetch_pool.submit(agent.prepare_training_step)
                    metrics = agent.train_step(current_prepared_step)
                    current_prepared_step = next_prepared_future.result()

                if metrics is None:
                    logger.warning("MuZero sans batch suffisant, arret a l'etape %s.", step)
                    append_training_log(
                        f"MuZero {horizon}: arret anticipe a l'etape {step} faute de batch suffisant.",
                        level="WARNING",
                        source="muzero",
                    )
                    break

                reanalyzed_this_step = 0
                reanalyze_started_at = perf_counter()
                if should_reanalyze:
                    reanalyzed_this_step = agent.reanalyze_recent_games(
                        int(getattr(config, "reanalyze_max_games", 0) or 0)
                    )
                    reanalyze_games_total += reanalyzed_this_step
                    reanalyze_positions_total += int(
                        getattr(agent, "last_reanalyze_positions_count", 0) or 0
                    )
                    if reanalyzed_this_step > 0:
                        logger.info(
                            "[%s] reanalyse de %s parties, %s positions, %s simulations a l'etape %s.",
                            horizon,
                            reanalyzed_this_step,
                            int(getattr(agent, "last_reanalyze_positions_count", 0) or 0),
                            int(getattr(agent, "last_reanalyze_num_simulations", 0) or 0),
                            step,
                        )
                    current_prepared_step = agent.prepare_training_step()
                reanalyze_ms = (perf_counter() - reanalyze_started_at) * 1000.0 if should_reanalyze else 0.0

                metrics["reanalyze_ms"] = reanalyze_ms
                metrics["reanalyze_games_count"] = float(reanalyze_games_total)
                metrics["reanalyze_positions_count"] = float(reanalyze_positions_total)
                metrics["reanalyze_num_simulations"] = float(
                    int(getattr(agent, "last_reanalyze_num_simulations", 0) or 0)
                )
                phase_durations_ms = {
                    "batch_prepare_ms": float(metrics.get("batch_prepare_ms", 0.0) or 0.0),
                    "device_put_ms": float(metrics.get("device_put_ms", 0.0) or 0.0),
                    "update_ms": float(metrics.get("update_ms", 0.0) or 0.0),
                    "reanalyze_ms": float(reanalyze_ms),
                }
                last_metrics = metrics
                arena_cutover_status = _build_arena_cutover_status(step)
                policy_precheck_history.append(
                    {
                        **dict(metrics),
                        "training_step": step,
                    }
                )
                merge_training_status(
                    {
                        "latest_metrics": {
                            **dict(last_metrics),
                            **arena_cutover_status,
                        },
                        "train_step_phase": "optimisation",
                        "phase_durations_ms": phase_durations_ms,
                        **arena_cutover_status,
                    }
                )

                seed_viability_payload = _evaluate_seed_viability_window(
                    history=list(policy_precheck_history),
                    config=config,
                    step=step,
                    horizon=horizon,
                    weights_dir=weights_dir,
                    trial_mode=trial_mode,
                )
                seed_viability_metrics = dict(seed_viability_payload.get("metrics") or {})
                merge_training_status(
                    {
                        "seed_viability_status": seed_viability_payload.get("status"),
                        "seed_viability_reason": seed_viability_payload.get("reason"),
                        "seed_stage": seed_viability_payload.get("seed_stage"),
                        "recommended_seed_for_v66": seed_viability_payload.get("recommended_seed_for_v66"),
                        "seed_selection_reason": seed_viability_payload.get("seed_selection_reason"),
                        "seed_viability": dict(seed_viability_payload),
                        "latest_metrics": {
                            **dict(last_metrics),
                            **arena_cutover_status,
                            "seed_viability_status": seed_viability_payload.get("status"),
                            "seed_viability_reason": seed_viability_payload.get("reason"),
                            "seed_stage": seed_viability_payload.get("seed_stage"),
                            "recommended_seed_for_v66": seed_viability_payload.get("recommended_seed_for_v66"),
                            "seed_selection_reason": seed_viability_payload.get("seed_selection_reason"),
                            "seed_loss_pol_improvement": seed_viability_metrics.get("loss_pol_improvement"),
                        },
                    }
                )
                if seed_viability_payload.get("status") == "seed_not_viable_for_v66":
                    merge_training_status(
                        {
                            "status": "seed_not_viable_for_v66",
                            "reason": "seed_not_viable_for_v66",
                        }
                    )
                    append_training_log(
                        (
                            f"MuZero {horizon}: arret anticipe a l'etape {step} "
                            "car le seed V6.10 est juge non productif "
                            f"({seed_viability_payload.get('reason')})."
                        ),
                        level="WARNING",
                        source="muzero",
                    )
                    killed_after_seed_viability = True
                    break

                plateau_payload = _detect_arena_plateau(
                    history=list(policy_precheck_history),
                    config=config,
                    step=step,
                )
                if bool(plateau_payload.get("allowed")) and not plateau_checkpoint_saved:
                    checkpoint_path = weights_dir / f"muzero_{horizon}_ckpt_{step}.pkl"
                    agent.save(
                        str(checkpoint_path),
                        artifact_kind="arena_plateau_checkpoint",
                        lineage={
                            **lineage,
                            "plateau_step": step,
                            "plateau_reason": plateau_payload.get("reason"),
                        },
                    )
                    merge_training_status(
                        {
                            "latest_metrics": {
                                **dict(last_metrics),
                                "plateau_detected": True,
                                "plateau_checkpoint_path": str(checkpoint_path),
                            },
                            "plateau_status": {
                                **dict(plateau_payload),
                                "checkpoint_path": str(checkpoint_path),
                            },
                        }
                    )
                    append_training_log(
                        (
                            f"MuZero {horizon}: plateau confirme a l'etape {step} "
                            f"-> checkpoint {checkpoint_path.name}."
                        ),
                        level="INFO",
                        source="muzero",
                    )
                    plateau_checkpoint_saved = True
                    if bool(getattr(config, "arena_plateau_stop_enabled", False)):
                        append_training_log(
                            (
                                f"MuZero {horizon}: bascule anticipee vers Arena activee "
                                "par MUZERO_ARENA_PLATEAU_STOP_ENABLED."
                            ),
                            level="WARNING",
                            source="muzero",
                        )
                        stopped_for_plateau = True
                        break
                    append_training_log(
                        (
                            f"MuZero {horizon}: le plateau est conserve comme signal de screen, "
                            "mais le run continue vers les checkpoints tardifs."
                        ),
                        level="INFO",
                        source="muzero",
                    )

                collapse_check_step = int(
                    getattr(config, "directional_collapse_check_step", 4000) or 4000
                )
                collapse_stop_step = int(
                    getattr(config, "directional_collapse_stop_step", 8000) or 8000
                )
                collapse_max_imbalance = float(
                    getattr(config, "directional_collapse_max_imbalance", 0.80) or 0.80
                )
                long_entry_share = float(metrics.get("long_entry_share", 0.0) or 0.0)
                short_entry_share = float(metrics.get("short_entry_share", 0.0) or 0.0)
                try:
                    directional_imbalance = float(metrics.get("directional_imbalance", 1.0))
                except (TypeError, ValueError):
                    directional_imbalance = 1.0
                directional_bias = str(metrics.get("directional_bias") or "inactive").strip().lower()
                policy_precheck_step = int(
                    getattr(config, "policy_precheck_step", 12000) or 12000
                )

                if (
                    not policy_precheck_executed
                    and policy_precheck_step > 0
                    and step >= policy_precheck_step
                ):
                    policy_precheck_payload = _evaluate_policy_precheck_window(
                        history=list(policy_precheck_history),
                        config=config,
                        step=step,
                        stage="mid_run",
                    )
                    merge_training_status(
                        {
                            "policy_precheck": dict(policy_precheck_payload),
                            "latest_metrics": {
                                **dict(last_metrics),
                                **arena_cutover_status,
                                "policy_precheck_passed": policy_precheck_payload.get("status") != "blocked",
                                "policy_precheck_mode": policy_precheck_payload.get("status"),
                                "root_mask_rate_trend": dict(
                                    policy_precheck_payload.get("trends") or {}
                                ).get("root_mask_rate_trend"),
                                "loss_pol_trend": dict(
                                    policy_precheck_payload.get("trends") or {}
                                ).get("loss_pol_trend"),
                                "loss_pol_root": dict(
                                    policy_precheck_payload.get("medians") or {}
                                ).get("loss_pol_root"),
                                "loss_pol_unroll_mean": dict(
                                    policy_precheck_payload.get("medians") or {}
                                ).get("loss_pol_unroll_mean"),
                                "loss_pol_root_trend": dict(
                                    policy_precheck_payload.get("trends") or {}
                                ).get("loss_pol_root_trend"),
                                "loss_pol_unroll_mean_trend": dict(
                                    policy_precheck_payload.get("trends") or {}
                                ).get("loss_pol_unroll_mean_trend"),
                            },
                            "arena_cutover_ready": arena_cutover_status.get("arena_cutover_ready"),
                            "screen_window": arena_cutover_status.get("screen_window"),
                        }
                    )
                    policy_precheck_executed = True
                    append_training_log(
                        (
                            f"MuZero {horizon}: precheck policy {policy_precheck_payload.get('status')} "
                            f"a l'etape {step} ({policy_precheck_payload.get('reason')})."
                        ),
                        level="WARNING" if policy_precheck_payload.get("status") == "blocked" else "INFO",
                        source="muzero",
                    )
                    if policy_precheck_payload.get("status") == "blocked":
                        merge_training_status(
                            {
                                "status": "policy_precheck_failed",
                                "reason": "policy_precheck_failed",
                            }
                        )
                        killed_after_policy_precheck = True
                        break

                if step >= collapse_check_step and (long_entry_share <= 0.0 or short_entry_share <= 0.0):
                    directional_collapse_payload = {
                        "status": "directional_collapse",
                        "step": step,
                        "reason": "direction_absente_apres_phase_apprentissage",
                        "failure_mode": directional_bias or "inactive",
                        "metrics": dict(metrics),
                    }
                    merge_training_status(
                        {
                            "status": "directional_collapse",
                            "latest_metrics": {
                                **dict(last_metrics),
                                "directional_collapse": True,
                            },
                        }
                    )
                    append_training_log(
                        (
                            f"MuZero {horizon}: arret anticipe a l'etape {step} "
                            "car une direction reste absente."
                        ),
                        level="WARNING",
                        source="muzero",
                    )
                    killed_after_directional_collapse = True
                    break

                if step >= collapse_stop_step and directional_imbalance > collapse_max_imbalance:
                    directional_collapse_payload = {
                        "status": "directional_collapse",
                        "step": step,
                        "reason": "desequilibre_directionnel_extreme",
                        "failure_mode": directional_bias or "inactive",
                        "metrics": dict(metrics),
                    }
                    merge_training_status(
                        {
                            "status": "directional_collapse",
                            "latest_metrics": {
                                **dict(last_metrics),
                                "directional_collapse": True,
                            },
                        }
                    )
                    append_training_log(
                        (
                            f"MuZero {horizon}: arret anticipe a l'etape {step} "
                            "pour collapse directionnel."
                        ),
                        level="WARNING",
                        source="muzero",
                    )
                    killed_after_directional_collapse = True
                    break

                if (
                    gold_precheck_enabled
                    and not gold_precheck_executed
                    and step >= gold_precheck_step
                ):
                    checkpoint_path = weights_dir / f"muzero_{horizon}_gold_precheck_{step}.pkl"
                    agent.save(
                        str(checkpoint_path),
                        artifact_kind="gold_precheck",
                        lineage=lineage,
                    )
                    append_training_log(
                        (
                            f"MuZero {horizon}: lancement du precheck Gold a l'etape "
                            f"{step} sur {gold_precheck_games} segments."
                        ),
                        source="muzero",
                    )
                    running_precheck = {
                        "status": "running",
                        "run_id": str(load_training_status().get("run_id") or "").strip() or None,
                        "trial_id": ga_trial,
                        "engine": engine,
                        "horizon": horizon,
                        "family": family,
                        "feature_profile": str(feature_profile.get("profile_name") or "").strip() or None,
                        "step": step,
                        "eval_symbols": list(focus_symbols),
                        "games": gold_precheck_games,
                        "reason": "precheck_en_cours",
                    }
                    set_gold_precheck(running_precheck)
                    arena = Arena(weights_dir=config.weights_path)
                    precheck_report = arena.evaluate_candidate(
                        checkpoint_path.stem,
                        horizon=horizon,
                        eval_symbols=list(focus_symbols),
                        games_per_symbol=gold_precheck_games,
                    )
                    precheck_metrics = dict(((precheck_report.get("challenger") or {}).get("metrics")) or {})
                    precheck_mechanics = dict(precheck_metrics.get("metrics_by_position_mechanics") or {})
                    verdict = _evaluate_gold_precheck_verdict(
                        metrics=precheck_metrics,
                        mechanics=precheck_mechanics,
                    )
                    gold_precheck_payload = {
                        "status": verdict.get("status"),
                        "run_id": str(load_training_status().get("run_id") or "").strip() or None,
                        "trial_id": ga_trial,
                        "engine": engine,
                        "horizon": horizon,
                        "family": family,
                        "feature_profile": str(feature_profile.get("profile_name") or "").strip() or None,
                        "step": step,
                        "eval_symbols": list(precheck_report.get("eval_symbols") or focus_symbols),
                        "games": int(precheck_report.get("games_per_symbol") or gold_precheck_games),
                        "metrics": precheck_metrics,
                        "metrics_by_position_mechanics": precheck_mechanics,
                        "reason": verdict.get("reason"),
                        "failure_mode": verdict.get("failure_mode"),
                    }
                    precheck_path = write_precheck_summary(gold_precheck_payload)
                    gold_precheck_payload["path"] = str(precheck_path)
                    set_gold_precheck(gold_precheck_payload)
                    gold_precheck_executed = True
                    append_training_log(
                        (
                            f"MuZero {horizon}: precheck Gold {gold_precheck_payload.get('status')} "
                            f"({gold_precheck_payload.get('reason')})."
                        ),
                        level="WARNING" if gold_precheck_payload.get("status") == "fail" else "INFO",
                        source="muzero",
                    )
                    if gold_precheck_payload.get("status") == "fail":
                        logger.warning(
                            "MuZero %s coupe apres precheck Gold a l'etape %s: %s",
                            horizon,
                            step,
                            gold_precheck_payload.get("reason"),
                        )
                        killed_after_precheck = True
                        break

                if step % 50 == 0:
                    elapsed = max((datetime.now() - start_time).total_seconds(), 1.0)
                    logger.info(
                        (
                            "[%s] step %05d/%05d | loss=%.4f | val=%.4f | rew=%.4f | "
                            "pol=%.4f | ent=%.4f | top1=%.4f | legal=%.2f | masked=%.2f | "
                            "root_mask=%.2f | post_veto=%.2f | soft_ratio=%.2f | soft_net=%.3f | "
                            "prep=%.1fms | put=%.1fms | upd=%.1fms | reanalyze=%.1fms | "
                            "mode=%s | %.2f steps/s"
                        ),
                        horizon,
                        step,
                        config.training_steps,
                        float(metrics["loss_total"]),
                        float(metrics["loss_val"]),
                        float(metrics["loss_rew"]),
                        float(metrics["loss_pol"]),
                        float(metrics.get("policy_entropy", 0.0)),
                        float(metrics.get("policy_top1_share", 0.0)),
                        float(metrics.get("root_legal_action_count", 0.0)),
                        float(metrics.get("invalid_root_action_masked_rate", 0.0)),
                        float(metrics.get("root_mask_rate", 0.0)),
                        float(metrics.get("post_veto_to_hold_rate", 0.0)),
                        float(metrics.get("soft_penalty_to_bonus_ratio", 0.0)),
                        float(metrics.get("soft_penalty_net", 0.0)),
                        float(metrics.get("batch_prepare_ms", 0.0)),
                        float(metrics.get("device_put_ms", 0.0)),
                        float(metrics.get("update_ms", 0.0)),
                        float(metrics.get("reanalyze_ms", 0.0)),
                        str(metrics.get("gpu_target_mode") or "auto"),
                        step / elapsed,
                    )
                    append_training_log(
                        "MuZero "
                        f"{horizon}: step {step}/{config.training_steps} | "
                        f"loss={float(metrics['loss_total']):.4f} | "
                        f"pol={float(metrics['loss_pol']):.4f} | "
                        f"root_mask={float(metrics.get('root_mask_rate', 0.0)):.4f} | "
                        f"post_veto={float(metrics.get('post_veto_to_hold_rate', 0.0)):.4f} | "
                        f"soft_ratio={float(metrics.get('soft_penalty_to_bonus_ratio', 0.0)):.3f} | "
                        f"ent={float(metrics.get('policy_entropy', 0.0)):.4f} | "
                        f"prep_ms={float(metrics.get('batch_prepare_ms', 0.0)):.1f} | "
                        f"upd_ms={float(metrics.get('update_ms', 0.0)):.1f}",
                        source="muzero",
                    )

                if step % config.checkpoint_interval == 0:
                    checkpoint_path = weights_dir / f"muzero_{horizon}_ckpt_{step}.pkl"
                    agent.save(
                        str(checkpoint_path),
                        artifact_kind="intermediate_checkpoint",
                        lineage=lineage,
                    )
                    logger.info("Checkpoint MuZero sauvegarde: %s", checkpoint_path)

        agent.save(
            str(latest_path),
            artifact_kind="latest",
            lineage=lineage,
        )
        logger.info("Checkpoint latest mis a jour: %s", latest_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    challenger_id = f"gen_{horizon}_{timestamp}"
    challenger_path = weights_dir / f"{challenger_id}.pkl"
    challenger_lineage = {
        **lineage,
        "challenger_id": challenger_id,
    }
    agent.save(
        str(challenger_path),
        artifact_kind="challenger",
        lineage=challenger_lineage,
    )
    challenger_compatibility = agent.inspect_checkpoint(str(challenger_path))
    challenger_checkpoint_schema_version = challenger_compatibility.get("schema_version")
    training_metrics_payload = _build_training_metrics_payload(
        base_metrics=last_metrics,
        family=family,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile_name=str(feature_profile.get("profile_name") or "").strip() or None,
        mechanics_profile_version=mechanics_profile_version,
        dataset_coverage=dataset_coverage,
        focus_symbols=focus_symbols,
        ga_campaign_id=ga_campaign_id,
        ga_trial=ga_trial,
        ga_scope=ga_scope,
        ga_parent_champion_id=ga_parent_champion_id,
        resume_source=resume_source,
        checkpoint_schema_version=challenger_checkpoint_schema_version,
        artifact_compatibility=challenger_compatibility,
        lineage=challenger_lineage,
    )
    if seed_viability_payload:
        training_metrics_payload.update(
            {
                "seed_viability_status": seed_viability_payload.get("status"),
                "seed_viability_reason": seed_viability_payload.get("reason"),
                "seed_stage": seed_viability_payload.get("seed_stage"),
                "recommended_seed_for_v66": seed_viability_payload.get("recommended_seed_for_v66"),
                "seed_selection_reason": seed_viability_payload.get("seed_selection_reason"),
            }
        )
    latest_checkpoint_reference = (
        str(ga_seed_checkpoint_path)
        if mechanics_only_mode and ga_seed_checkpoint_path
        else str(latest_path)
    )
    latest_checkpoint_present = bool(
        latest_checkpoint_reference and Path(latest_checkpoint_reference).exists()
    )
    active_run_id = str(load_training_status().get("run_id") or "").strip() or None

    if (
        not mechanics_only_mode
        and not killed_after_directional_collapse
        and not killed_after_seed_viability
        and not killed_after_precheck
        and last_metrics is not None
    ):
        final_arena_cutover_status = _build_arena_cutover_status(last_optimization_step)
        final_policy_precheck = _evaluate_policy_precheck_window(
            history=list(policy_precheck_history),
            config=config,
            step=last_optimization_step,
            stage="pre_arena",
        )
        policy_precheck_payload = dict(final_policy_precheck)
        if stopped_for_plateau:
            if policy_precheck_payload.get("status") == "full_ready":
                policy_precheck_payload.update(
                    {
                        "status": "screen_only",
                        "reason": "plateau_screen_cutover",
                        "plateau_status": dict(plateau_payload or {}),
                    }
                )
            elif policy_precheck_payload.get("status") == "blocked":
                screen_checks = dict(policy_precheck_payload.get("screen_checks") or {})
                screen_failed_check = next(
                    (name for name, passed in screen_checks.items() if not passed),
                    None,
                )
                if screen_failed_check is None:
                    policy_precheck_payload.update(
                        {
                            "status": "screen_only",
                            "reason": "plateau_screen_cutover",
                            "plateau_status": dict(plateau_payload or {}),
                        }
                    )
        merge_training_status(
            {
                "policy_precheck": dict(policy_precheck_payload),
                "latest_metrics": {
                    **dict(last_metrics),
                    **final_arena_cutover_status,
                    "policy_precheck_passed": policy_precheck_payload.get("status") != "blocked",
                    "policy_precheck_mode": policy_precheck_payload.get("status"),
                    "root_mask_rate_trend": dict(
                        policy_precheck_payload.get("trends") or {}
                    ).get("root_mask_rate_trend"),
                    "loss_pol_trend": dict(
                        policy_precheck_payload.get("trends") or {}
                    ).get("loss_pol_trend"),
                    "loss_pol_root": dict(
                        policy_precheck_payload.get("medians") or {}
                    ).get("loss_pol_root"),
                    "loss_pol_unroll_mean": dict(
                        policy_precheck_payload.get("medians") or {}
                    ).get("loss_pol_unroll_mean"),
                    "loss_pol_root_trend": dict(
                        policy_precheck_payload.get("trends") or {}
                    ).get("loss_pol_root_trend"),
                    "loss_pol_unroll_mean_trend": dict(
                        policy_precheck_payload.get("trends") or {}
                    ).get("loss_pol_unroll_mean_trend"),
                },
                "arena_cutover_ready": final_arena_cutover_status.get("arena_cutover_ready"),
                "screen_window": final_arena_cutover_status.get("screen_window"),
            }
        )
        if policy_precheck_payload.get("status") == "blocked":
            merge_training_status(
                {
                    "status": "policy_precheck_failed",
                    "reason": "policy_precheck_failed",
                }
            )
            killed_after_policy_precheck = True

    if killed_after_directional_collapse:
        promotion_gate = {
            "allowed": False,
            "status": "blocked",
            "reason": str((directional_collapse_payload or {}).get("reason") or "directional_collapse"),
            "gate_profile": gate_profile,
            "failure_mode": str((directional_collapse_payload or {}).get("failure_mode") or "inactive"),
            "checks": {
                "directional_collapse": False,
            },
            "metrics": dict((directional_collapse_payload or {}).get("metrics") or {}),
        }
        promotion_result = {
            "status": "skipped",
            "reason": "directional_collapse",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(challenger_path),
            "champion_paths": [],
            "promotion_gate": promotion_gate,
            "artifact_compatibility": challenger_compatibility,
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "resume_source": resume_source,
            "lineage": challenger_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
            "ga_campaign_id": ga_campaign_id,
            "ga_trial": ga_trial,
            "ga_scope": ga_scope,
        }
        promoter = ChampionPromoter(weights_dir=config.weights_path, results_dir=config.results_path)
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status="blocked",
            challenger_id=challenger_id,
            challenger_path=str(challenger_path),
            latest_checkpoint=str(latest_path),
            battle_report=None,
            training_metrics=training_metrics_payload,
            promotion_gate=promotion_gate,
            promotion_result=promotion_result,
            artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
            checkpoint_schema_version=challenger_checkpoint_schema_version,
            resume_source=resume_source,
            lineage=challenger_lineage,
        )
        terminal_summary = {
            "run_id": active_run_id,
            "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
            "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
            "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
            "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
            "engine": engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "terminal_status": "completed",
            "failed_step": (directional_collapse_payload or {}).get("step"),
            "failure_mode": promotion_gate.get("failure_mode"),
            "arena_outcome": None,
            "promotion_gate": promotion_gate,
            "metrics": dict((directional_collapse_payload or {}).get("metrics") or {}),
            "metrics_by_symbol": {},
            "metrics_by_position_mechanics": {},
            "training_metrics": training_metrics_payload,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "artifact_state": {
                "precheck_report_present": False,
                "arena_report_present": False,
                "battle_report_present": False,
                "promotion_present": True,
                "candidate_checkpoint_present": challenger_path.exists(),
                "latest_checkpoint_present": latest_checkpoint_present,
            },
            "challenger_path": str(challenger_path),
            "latest_checkpoint": str(latest_path),
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "directional_collapse",
                "reason": promotion_gate.get("reason"),
                "failure_mode": promotion_gate.get("failure_mode"),
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        return {
            "engine": engine,
            "horizon": horizon,
            "timeframe": config.primary_timeframe,
            "symbols": valid_symbols,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "dataset_descriptor": dataset_descriptor,
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "games_per_symbol": games_per_symbol,
            "total_games": total_games,
            "latest_checkpoint": latest_checkpoint_reference,
            "challenger_path": str(challenger_path),
            "training_metrics": training_metrics_payload,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "precheck": dict(gold_precheck_payload or {}),
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
        }

    if killed_after_seed_viability:
        seed_metrics = dict((seed_viability_payload or {}).get("metrics") or {})
        promotion_gate = {
            "allowed": False,
            "status": "blocked",
            "reason": "seed_not_viable_for_v66",
            "gate_profile": gate_profile,
            "failure_mode": str((seed_viability_payload or {}).get("reason") or "seed_not_viable_for_v66"),
            "checks": dict((seed_viability_payload or {}).get("checks") or {}),
            "metrics": seed_metrics,
        }
        promotion_result = {
            "status": "skipped",
            "reason": "seed_not_viable_for_v66",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(challenger_path),
            "champion_paths": [],
            "promotion_gate": promotion_gate,
            "artifact_compatibility": challenger_compatibility,
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "resume_source": resume_source,
            "lineage": challenger_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
            "ga_campaign_id": ga_campaign_id,
            "ga_trial": ga_trial,
            "ga_scope": ga_scope,
            "seed_viability": dict(seed_viability_payload or {}),
            "recommended_seed_for_v66": (seed_viability_payload or {}).get("recommended_seed_for_v66"),
            "seed_selection_reason": (seed_viability_payload or {}).get("seed_selection_reason"),
        }
        promoter = ChampionPromoter(weights_dir=config.weights_path, results_dir=config.results_path)
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status="blocked",
            challenger_id=challenger_id,
            challenger_path=str(challenger_path),
            latest_checkpoint=str(latest_path),
            battle_report=None,
            training_metrics=training_metrics_payload,
            promotion_gate=promotion_gate,
            promotion_result=promotion_result,
            artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
            checkpoint_schema_version=challenger_checkpoint_schema_version,
            resume_source=resume_source,
            lineage=challenger_lineage,
        )
        terminal_summary = {
            "run_id": active_run_id,
            "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
            "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
            "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
            "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
            "engine": engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "terminal_status": "completed",
            "failed_step": (seed_viability_payload or {}).get("step"),
            "failure_mode": promotion_gate.get("failure_mode"),
            "arena_outcome": None,
            "promotion_gate": promotion_gate,
            "metrics": seed_metrics,
            "metrics_by_symbol": {},
            "metrics_by_position_mechanics": {},
            "training_metrics": training_metrics_payload,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "artifact_state": {
                "precheck_report_present": False,
                "arena_report_present": False,
                "battle_report_present": False,
                "promotion_present": True,
                "candidate_checkpoint_present": challenger_path.exists(),
                "latest_checkpoint_present": latest_checkpoint_present,
            },
            "challenger_path": str(challenger_path),
            "latest_checkpoint": str(latest_path),
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "seed_not_viable_for_v66",
                "reason": promotion_gate.get("reason"),
                "failure_mode": promotion_gate.get("failure_mode"),
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
            "policy_precheck": dict(policy_precheck_payload or {}),
            "seed_viability": dict(seed_viability_payload or {}),
            "recommended_seed_for_v66": (seed_viability_payload or {}).get("recommended_seed_for_v66"),
            "seed_selection_reason": (seed_viability_payload or {}).get("seed_selection_reason"),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        append_training_log(
            (
                f"MuZero {horizon}: arena annulee car le seed V6.10 est non viable "
                f"({(seed_viability_payload or {}).get('reason')})."
            ),
            level="WARNING",
            source="muzero",
        )
        return {
            "engine": engine,
            "horizon": horizon,
            "timeframe": config.primary_timeframe,
            "symbols": valid_symbols,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "dataset_descriptor": dataset_descriptor,
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "games_per_symbol": games_per_symbol,
            "total_games": total_games,
            "latest_checkpoint": latest_checkpoint_reference,
            "challenger_path": str(challenger_path),
            "training_metrics": training_metrics_payload,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "precheck": dict(gold_precheck_payload or {}),
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
        }

    if killed_after_policy_precheck:
        policy_metrics = dict((policy_precheck_payload or {}).get("medians") or {})
        promotion_gate = {
            "allowed": False,
            "status": "blocked",
            "reason": "policy_precheck_failed",
            "gate_profile": gate_profile,
            "failure_mode": str((policy_precheck_payload or {}).get("reason") or "policy_precheck_failed"),
            "checks": dict((policy_precheck_payload or {}).get("checks") or {}),
            "metrics": policy_metrics,
        }
        promotion_result = {
            "status": "skipped",
            "reason": "policy_precheck_failed",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(challenger_path),
            "champion_paths": [],
            "promotion_gate": promotion_gate,
            "artifact_compatibility": challenger_compatibility,
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "resume_source": resume_source,
            "lineage": challenger_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
            "ga_campaign_id": ga_campaign_id,
            "ga_trial": ga_trial,
            "ga_scope": ga_scope,
            "policy_precheck": dict(policy_precheck_payload or {}),
        }
        promoter = ChampionPromoter(weights_dir=config.weights_path, results_dir=config.results_path)
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status="blocked",
            challenger_id=challenger_id,
            challenger_path=str(challenger_path),
            latest_checkpoint=str(latest_path),
            battle_report=None,
            training_metrics=training_metrics_payload,
            promotion_gate=promotion_gate,
            promotion_result=promotion_result,
            artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
            checkpoint_schema_version=challenger_checkpoint_schema_version,
            resume_source=resume_source,
            lineage=challenger_lineage,
        )
        terminal_summary = {
            "run_id": active_run_id,
            "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
            "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
            "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
            "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
            "engine": engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "terminal_status": "completed",
            "failed_step": (policy_precheck_payload or {}).get("step"),
            "failure_mode": promotion_gate.get("failure_mode"),
            "arena_outcome": None,
            "promotion_gate": promotion_gate,
            "metrics": policy_metrics,
            "metrics_by_symbol": {},
            "metrics_by_position_mechanics": {},
            "training_metrics": training_metrics_payload,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "artifact_state": {
                "precheck_report_present": False,
                "arena_report_present": False,
                "battle_report_present": False,
                "promotion_present": True,
                "candidate_checkpoint_present": challenger_path.exists(),
                "latest_checkpoint_present": latest_checkpoint_present,
            },
            "challenger_path": str(challenger_path),
            "latest_checkpoint": str(latest_path),
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "policy_precheck_failed",
                "reason": promotion_gate.get("reason"),
                "failure_mode": promotion_gate.get("failure_mode"),
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
            "policy_precheck": dict(policy_precheck_payload or {}),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        append_training_log(
            (
                f"MuZero {horizon}: arena annulee apres echec du precheck policy "
                f"({(policy_precheck_payload or {}).get('reason')})."
            ),
            level="WARNING",
            source="muzero",
        )
        return {
            "engine": engine,
            "horizon": horizon,
            "timeframe": config.primary_timeframe,
            "symbols": valid_symbols,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "dataset_descriptor": dataset_descriptor,
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "games_per_symbol": games_per_symbol,
            "total_games": total_games,
            "latest_checkpoint": latest_checkpoint_reference,
            "challenger_path": str(challenger_path),
            "training_metrics": training_metrics_payload,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "precheck": dict(gold_precheck_payload or {}),
            "policy_precheck": dict(policy_precheck_payload or {}),
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
        }

    if killed_after_precheck:
        precheck_metrics = dict((gold_precheck_payload or {}).get("metrics") or {})
        precheck_mechanics = dict((gold_precheck_payload or {}).get("metrics_by_position_mechanics") or {})
        promotion_result = {
            "status": "skipped",
            "reason": "gold_precheck_fail",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(challenger_path),
            "champion_paths": [],
            "promotion_gate": {
                "allowed": False,
                "status": "blocked",
                "reason": "gold_precheck_fail",
                "gate_profile": gate_profile,
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            },
            "artifact_compatibility": challenger_compatibility,
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "resume_source": resume_source,
            "lineage": challenger_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
            "ga_campaign_id": ga_campaign_id,
            "ga_trial": ga_trial,
            "ga_scope": ga_scope,
        }
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status="blocked",
            challenger_id=challenger_id,
            challenger_path=str(challenger_path),
            latest_checkpoint=str(latest_path),
            battle_report=None,
            training_metrics=training_metrics_payload,
            promotion_gate=dict(promotion_result.get("promotion_gate") or {}),
            promotion_result=promotion_result,
            artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
            checkpoint_schema_version=challenger_checkpoint_schema_version,
            resume_source=resume_source,
            lineage=challenger_lineage,
        )
        terminal_summary = {
            "run_id": active_run_id,
            "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
            "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
            "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
            "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
            "engine": engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "terminal_status": "completed",
            "failed_step": None,
            "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            "arena_outcome": None,
            "promotion_gate": dict(promotion_result.get("promotion_gate") or {}),
            "metrics": precheck_metrics,
            "metrics_by_symbol": dict(precheck_metrics.get("metrics_by_symbol") or {}),
            "metrics_by_position_mechanics": precheck_mechanics,
            "training_metrics": training_metrics_payload,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "artifact_state": {
                "precheck_report_present": bool((gold_precheck_payload or {}).get("path")),
                "arena_report_present": False,
                "battle_report_present": False,
                "promotion_present": True,
                "candidate_checkpoint_present": challenger_path.exists(),
                "latest_checkpoint_present": latest_checkpoint_present,
            },
            "challenger_path": str(challenger_path),
            "latest_checkpoint": str(latest_path),
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "killed_after_precheck",
                "reason": (gold_precheck_payload or {}).get("reason"),
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        append_training_log(
            (
                f"MuZero {horizon}: trial coupe apres precheck Gold "
                f"({(gold_precheck_payload or {}).get('reason')})."
            ),
            level="WARNING",
            source="muzero",
        )
        return {
            "engine": engine,
            "horizon": horizon,
            "timeframe": config.primary_timeframe,
            "symbols": valid_symbols,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "dataset_descriptor": dataset_descriptor,
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "games_per_symbol": games_per_symbol,
            "total_games": total_games,
            "latest_checkpoint": latest_checkpoint_reference,
            "challenger_path": str(challenger_path),
            "training_metrics": training_metrics_payload,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "precheck": dict(gold_precheck_payload or {}),
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
        }

    bakeoff_no_arena = str(os.getenv("MUZERO_SKIP_ARENA_FOR_BAKEOFF", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if bakeoff_no_arena:
        promotion_gate = {
            "allowed": False,
            "status": "skipped",
            "reason": "bakeoff_no_arena",
            "gate_profile": gate_profile,
            "failure_mode": "bakeoff_no_arena",
            "checks": {},
            "metrics": dict(training_metrics_payload),
        }
        promotion_result = {
            "status": "skipped",
            "reason": "bakeoff_no_arena",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(challenger_path),
            "champion_paths": [],
            "promotion_gate": promotion_gate,
            "artifact_compatibility": challenger_compatibility,
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "resume_source": resume_source,
            "lineage": challenger_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
            "ga_campaign_id": ga_campaign_id,
            "ga_trial": ga_trial,
            "ga_scope": ga_scope,
            "recommended_seed_for_v66": (seed_viability_payload or {}).get("recommended_seed_for_v66"),
            "seed_selection_reason": (seed_viability_payload or {}).get("seed_selection_reason"),
        }
        promoter = ChampionPromoter(weights_dir=config.weights_path, results_dir=config.results_path)
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status="blocked",
            challenger_id=challenger_id,
            challenger_path=str(challenger_path),
            latest_checkpoint=str(latest_path),
            battle_report=None,
            training_metrics=training_metrics_payload,
            promotion_gate=promotion_gate,
            promotion_result=promotion_result,
            artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
            checkpoint_schema_version=challenger_checkpoint_schema_version,
            resume_source=resume_source,
            lineage=challenger_lineage,
        )
        terminal_summary = {
            "run_id": active_run_id,
            "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
            "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
            "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
            "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
            "engine": engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "terminal_status": "completed",
            "failed_step": last_optimization_step,
            "failure_mode": "bakeoff_no_arena",
            "arena_outcome": None,
            "promotion_gate": promotion_gate,
            "metrics": dict(training_metrics_payload),
            "metrics_by_symbol": {},
            "metrics_by_position_mechanics": {},
            "training_metrics": training_metrics_payload,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "artifact_state": {
                "precheck_report_present": bool((gold_precheck_payload or {}).get("path")),
                "arena_report_present": False,
                "battle_report_present": False,
                "promotion_present": True,
                "candidate_checkpoint_present": challenger_path.exists(),
                "latest_checkpoint_present": latest_checkpoint_present,
            },
            "challenger_path": str(challenger_path),
            "latest_checkpoint": str(latest_path),
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "bakeoff_no_arena",
                "reason": "bakeoff_no_arena",
                "failure_mode": "bakeoff_no_arena",
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
            "policy_precheck": dict(policy_precheck_payload or {}),
            "seed_viability": dict(seed_viability_payload or {}),
            "recommended_seed_for_v66": (seed_viability_payload or {}).get("recommended_seed_for_v66"),
            "seed_selection_reason": (seed_viability_payload or {}).get("seed_selection_reason"),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        append_training_log(
            f"MuZero {horizon}: run de bake-off termine sans Arena (MUZERO_SKIP_ARENA_FOR_BAKEOFF).",
            level="WARNING",
            source="muzero",
        )
        return {
            "engine": engine,
            "horizon": horizon,
            "timeframe": config.primary_timeframe,
            "symbols": valid_symbols,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "dataset_descriptor": dataset_descriptor,
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "games_per_symbol": games_per_symbol,
            "total_games": total_games,
            "latest_checkpoint": latest_checkpoint_reference,
            "challenger_path": str(challenger_path),
            "training_metrics": training_metrics_payload,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": challenger_lineage,
            "precheck": dict(gold_precheck_payload or {}),
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
        }

    logger.info("Phase 3 - arena ADN")
    append_training_log(
        f"MuZero {horizon}: lancement de l'arena ADN.",
        source="muzero",
    )
    mark_step_running(
        step_name,
        engine=engine,
        phase="arena",
        horizon=horizon,
        family=family,
        symbol_total=len(valid_symbols),
        part_total=games_per_symbol,
        training_step_current=config.training_steps if last_metrics is not None else None,
        training_step_total=config.training_steps,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
        ga_campaign_id=ga_campaign_id,
        ga_scope=ga_scope,
        ga_parent_champion_id=ga_parent_champion_id,
        trial_mode=trial_mode,
        trial_cost_profile=trial_cost_profile,
        replay_cache_status="memoire",
        replay_cache_key=f"{engine}:{horizon}:{family}:{mechanics_profile_version or 'default'}",
        replay_cache_entries=agent.replay_buffer.size,
        replay_cache_source="memoire",
        dataset_coverage=dataset_coverage,
    )
    genetic = GeneticUpdater()
    arena = Arena(weights_dir=config.weights_path)
    promoter = ChampionPromoter(weights_dir=config.weights_path, results_dir=config.results_path)
    live_path, live_meta = promoter.resolve_live_checkpoint(horizon)
    live_champion_id = str(live_meta.get("live_champion_id") or "").strip()
    champion_reference = None
    if live_path is not None:
        champion_reference = str(live_path)
    elif live_champion_id:
        champion_reference = live_champion_id
    else:
        champion_reference = genetic.get_champion(horizon=horizon)
    selected_candidate_id = challenger_id
    selected_challenger_path = Path(challenger_path)
    selected_latest_checkpoint = latest_checkpoint_reference
    selected_checkpoint_step: int | None = None
    selected_challenger_lineage = dict(challenger_lineage)
    screen_results: list[dict[str, object]] = []
    screen_gate: dict[str, object] | None = None
    family_probe_report: dict[str, object] | None = None
    champion_league_table: list[dict[str, object]] = []
    champion_rejection_report: dict[str, object] = {}
    policy_precheck_mode = str((policy_precheck_payload or {}).get("status") or "full_ready").strip().lower()

    if policy_precheck_mode == "screen_only":
        screen_candidates = _select_recent_screen_checkpoints(
            history=list(policy_precheck_history),
            weights_dir=weights_dir,
            horizon=horizon,
            last_step=last_optimization_step,
            config=config,
        )
        if not screen_candidates:
            screen_gate = {
                "allowed": False,
                "status": "blocked",
                "reason": "screen_candidates_absent",
                "checks": {},
                "metrics": {},
            }
        else:
            screen_budget = {
                "ARENA_GAMES_PER_SYMBOL": str(
                    int(getattr(config, "arena_screen_games_per_symbol", 4) or 4)
                ),
                "ARENA_MIN_GAMES": str(int(getattr(config, "arena_screen_min_games", 14) or 14)),
                "ARENA_MIN_SYMBOLS": str(int(getattr(config, "arena_screen_min_symbols", 7) or 7)),
            }
            screen_labels = ", ".join(
                f"ckpt{int(item.get('checkpoint_step') or 0)}"
                for item in screen_candidates
            )
            append_training_log(
                f"MuZero {horizon}: screen Arena automatique sur {screen_labels}.",
                source="muzero",
            )
            with _temporary_env_overrides(screen_budget):
                for candidate in screen_candidates:
                    checkpoint_step = int(candidate.get("checkpoint_step") or 0)
                    checkpoint_path = Path(str(candidate.get("checkpoint_path") or ""))
                    battle_report_screen = arena.battle(
                        str(checkpoint_path),
                        champion_reference,
                        horizon=horizon,
                    )
                    report_payload = {
                        "kind": "auto_screen",
                        "generated_at": datetime.now().isoformat(),
                        "source_run_id": active_run_id,
                        "engine": engine,
                        "horizon": horizon,
                        "checkpoint_step": checkpoint_step,
                        "checkpoint_path": str(checkpoint_path),
                        "champion_reference": champion_reference,
                        "symbols": list(valid_symbols),
                        "selection_window": dict(candidate.get("selection_window") or {}),
                        "selection_anchor_step": candidate.get("selection_anchor_step"),
                        "budget": {
                            "games_per_symbol": int(screen_budget["ARENA_GAMES_PER_SYMBOL"]),
                            "min_games": int(screen_budget["ARENA_MIN_GAMES"]),
                            "min_symbols": int(screen_budget["ARENA_MIN_SYMBOLS"]),
                        },
                        "battle_report": battle_report_screen,
                    }
                    report_path = results_dir / f"screen_{active_run_id}_ckpt{checkpoint_step}.json"
                    report_path.write_text(
                        json.dumps(report_payload, indent=2, default=float),
                        encoding="utf-8",
                    )
                    screen_results.append(
                        {
                            "checkpoint_step": checkpoint_step,
                            "checkpoint_path": str(checkpoint_path),
                            "report_path": str(report_path),
                            "battle_report": battle_report_screen,
                            "selection_window": dict(candidate.get("selection_window") or {}),
                            "selection_anchor_step": candidate.get("selection_anchor_step"),
                        }
                    )
            winner = _select_best_screen_candidate(screen_results)
            screen_gate = _evaluate_screen_winner_gate(
                dict(winner.get("battle_report") or {}),
                config,
            )
            selected_checkpoint_step = int(winner.get("checkpoint_step") or 0)
            selected_candidate_id = f"{challenger_id}_ckpt{selected_checkpoint_step}"
            selected_challenger_path = Path(str(winner.get("checkpoint_path") or challenger_path))
            selected_latest_checkpoint = str(selected_challenger_path)
            selected_challenger_lineage = {
                **challenger_lineage,
                "selected_checkpoint_step": selected_checkpoint_step,
                "selection_method": "auto_screen_recent_best_window",
            }
            append_training_log(
                (
                    f"MuZero {horizon}: gagnant screen ckpt{selected_checkpoint_step} | "
                    f"gate={screen_gate.get('status')} ({screen_gate.get('reason')})."
                ),
                level="WARNING" if not bool(screen_gate.get("allowed")) else "INFO",
                source="muzero",
            )

        if not bool((screen_gate or {}).get("allowed")):
            promotion_gate = {
                "allowed": False,
                "status": "blocked",
                "reason": str((screen_gate or {}).get("reason") or "screen_only_gate_failed"),
                "gate_profile": gate_profile,
                "failure_mode": str((screen_gate or {}).get("reason") or "screen_only_gate_failed"),
                "checks": dict((screen_gate or {}).get("checks") or {}),
                "metrics": dict((screen_gate or {}).get("metrics") or {}),
            }
            promotion_result = {
                "status": "skipped",
                "reason": "screen_only_gate_failed",
                "engine": engine,
                "horizon": horizon,
                "source_path": str(selected_challenger_path),
                "champion_paths": [],
                "promotion_gate": promotion_gate,
                "artifact_compatibility": challenger_compatibility,
                "checkpoint_schema_version": challenger_checkpoint_schema_version,
                "resume_source": resume_source,
                "lineage": selected_challenger_lineage,
                "seed_parent_champion_id": ga_parent_champion_id,
                "ga_campaign_id": ga_campaign_id,
                "ga_trial": ga_trial,
                "ga_scope": ga_scope,
                "policy_precheck": dict(policy_precheck_payload or {}),
                "screen_results": list(screen_results),
            }
            best_screen_report = (
                dict((_select_best_screen_candidate(screen_results).get("battle_report") or {}))
                if screen_results
                else None
            )
            champion_league_table = _build_champion_league_table(
                engine=engine,
                horizon=horizon,
                screen_results=screen_results,
                selected_candidate_id=selected_candidate_id,
                battle_report=best_screen_report,
                promotion_result=promotion_result,
                screen_gate=screen_gate,
            )
            champion_rejection_report = _build_champion_rejection_report(
                screen_gate=screen_gate,
                promotion_gate=promotion_gate,
                battle_report=best_screen_report,
            )
            promotion_result["champion_league_table"] = champion_league_table
            promotion_result["champion_rejection_report"] = champion_rejection_report
            promoter.persist_challenger_manifest(
                engine=engine,
                horizon=horizon,
                status="blocked",
                challenger_id=selected_candidate_id,
                challenger_path=str(selected_challenger_path),
                latest_checkpoint=str(selected_latest_checkpoint),
                battle_report=best_screen_report,
                training_metrics=training_metrics_payload,
                promotion_gate=promotion_gate,
                promotion_result=promotion_result,
                artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
                checkpoint_schema_version=challenger_checkpoint_schema_version,
                resume_source=resume_source,
                lineage=selected_challenger_lineage,
            )
            terminal_summary = {
                "run_id": active_run_id,
                "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
                "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
                "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
                "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
                "engine": engine,
                "horizon": horizon,
                "family": family,
                "feature_profile": feature_profile.get("profile_name"),
                "mechanics_profile_version": mechanics_profile_version,
                "ga_trial": ga_trial,
                "ga_campaign_id": ga_campaign_id,
                "ga_scope": ga_scope,
                "ga_parent_champion_id": ga_parent_champion_id,
                "seed_parent_champion_id": ga_parent_champion_id,
                "trial_mode": trial_mode,
                "trial_cost_profile": trial_cost_profile,
                "dataset_id": dataset_id,
                "dataset_source": dataset_source,
                "focus_symbols": focus_symbols,
                "gate_profile": gate_profile,
                "terminal_status": "completed",
                "failed_step": selected_checkpoint_step,
                "failure_mode": promotion_gate.get("failure_mode"),
                "arena_outcome": None,
                "promotion_gate": promotion_gate,
                "metrics": dict((screen_gate or {}).get("metrics") or {}),
                "metrics_by_symbol": {},
                "metrics_by_position_mechanics": {},
                "training_metrics": training_metrics_payload,
                "resume_source": resume_source,
                "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
                "checkpoint_schema_version": challenger_checkpoint_schema_version,
                "lineage": selected_challenger_lineage,
                "artifact_state": {
                    "precheck_report_present": False,
                    "arena_report_present": bool(screen_results),
                    "battle_report_present": bool(screen_results),
                    "promotion_present": True,
                    "candidate_checkpoint_present": selected_challenger_path.exists(),
                    "latest_checkpoint_present": Path(str(selected_latest_checkpoint)).exists(),
                },
                "challenger_path": str(selected_challenger_path),
                "latest_checkpoint": str(selected_latest_checkpoint),
                "latest_candidate": selected_candidate_id,
                "latest_verdict": {
                    "status": "screen_only_gate_failed",
                    "reason": promotion_gate.get("reason"),
                    "failure_mode": promotion_gate.get("failure_mode"),
                },
                "gold_precheck": dict(gold_precheck_payload or {}),
                "precheck_status": (gold_precheck_payload or {}).get("status"),
                "policy_precheck": dict(policy_precheck_payload or {}),
                "policy_precheck_mode": policy_precheck_mode,
                "screen_results": list(screen_results),
                "screen_gate": dict(screen_gate or {}),
                "champion_league_table": champion_league_table,
                "champion_rejection_report": champion_rejection_report,
            }
            terminal_summary_path = write_terminal_summary(terminal_summary)
            logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
            return {
                "engine": engine,
                "horizon": horizon,
                "timeframe": config.primary_timeframe,
                "symbols": valid_symbols,
                "family": family,
                "feature_profile": feature_profile.get("profile_name"),
                "mechanics_profile_version": mechanics_profile_version,
                "dataset_id": dataset_id,
                "dataset_source": dataset_source,
                "focus_symbols": focus_symbols,
                "gate_profile": gate_profile,
                "dataset_descriptor": dataset_descriptor,
                "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
                "games_per_symbol": games_per_symbol,
                "total_games": total_games,
                "latest_checkpoint": str(selected_latest_checkpoint),
                "challenger_path": str(selected_challenger_path),
                "training_metrics": training_metrics_payload,
                "ga_status": ga_status,
                "ga_generation": ga_generation,
                "ga_trial": ga_trial,
                "ga_campaign_id": ga_campaign_id,
                "ga_scope": ga_scope,
                "ga_parent_champion_id": ga_parent_champion_id,
                "seed_parent_champion_id": ga_parent_champion_id,
                "trial_mode": trial_mode,
                "trial_cost_profile": trial_cost_profile,
                "resume_source": resume_source,
                "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
                "checkpoint_schema_version": challenger_checkpoint_schema_version,
                "lineage": selected_challenger_lineage,
                "precheck": dict(gold_precheck_payload or {}),
                "policy_precheck": dict(policy_precheck_payload or {}),
                "policy_precheck_mode": policy_precheck_mode,
                "screen_results": list(screen_results),
                "screen_gate": dict(screen_gate or {}),
                "champion_league_table": champion_league_table,
                "champion_rejection_report": champion_rejection_report,
                "promotion": promotion_result,
                "terminal_summary_path": str(terminal_summary_path),
            }

    arena_candidate_id = selected_candidate_id
    arena_candidate_path = Path(selected_challenger_path)
    arena_latest_checkpoint = (
        Path(str(selected_latest_checkpoint))
        if selected_latest_checkpoint
        else latest_path
    )
    arena_lineage = dict(selected_challenger_lineage)
    family_probe_report = arena.run_family_probes(
        arena_candidate_id,
        horizon=horizon,
        engine=engine,
        weights_path=arena_candidate_path,
    )
    merge_training_status(
        {
            "family_probe_status": dict(family_probe_report.get("family_probe_summary") or {}),
        }
    )
    if not bool(dict(family_probe_report.get("family_probe_summary") or {}).get("allowed")):
        family_probe_summary = dict(family_probe_report.get("family_probe_summary") or {})
        promotion_gate = {
            "allowed": False,
            "status": "blocked",
            "reason": str(family_probe_summary.get("reason") or "family_probe_failed"),
            "gate_profile": gate_profile,
            "failure_mode": str(family_probe_summary.get("reason") or "family_probe_failed"),
            "checks": {},
            "metrics": family_probe_summary,
        }
        promotion_result = {
            "status": "skipped",
            "reason": "family_probe_failed",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(arena_candidate_path),
            "champion_paths": [],
            "promotion_gate": promotion_gate,
            "artifact_compatibility": challenger_compatibility,
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "resume_source": resume_source,
            "lineage": arena_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
            "ga_campaign_id": ga_campaign_id,
            "ga_trial": ga_trial,
            "ga_scope": ga_scope,
            "policy_precheck": dict(policy_precheck_payload or {}),
            "policy_precheck_mode": policy_precheck_mode,
            "screen_results": list(screen_results),
            "screen_gate": dict(screen_gate or {}),
            "family_probe_report": dict(family_probe_report or {}),
        }
        champion_league_table = _build_champion_league_table(
            engine=engine,
            horizon=horizon,
            screen_results=screen_results,
            selected_candidate_id=arena_candidate_id,
            promotion_result=promotion_result,
            screen_gate=screen_gate,
            family_probe_report=family_probe_report,
        )
        champion_rejection_report = _build_champion_rejection_report(
            screen_gate=screen_gate,
            promotion_gate=promotion_gate,
            family_probe_report=family_probe_report,
        )
        promotion_result["champion_league_table"] = champion_league_table
        promotion_result["champion_rejection_report"] = champion_rejection_report
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status="blocked",
            challenger_id=arena_candidate_id,
            challenger_path=str(arena_candidate_path),
            latest_checkpoint=str(arena_latest_checkpoint),
            battle_report=None,
            training_metrics=training_metrics_payload,
            promotion_gate=promotion_gate,
            promotion_result=promotion_result,
            artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
            checkpoint_schema_version=challenger_checkpoint_schema_version,
            resume_source=resume_source,
            lineage=arena_lineage,
        )
        terminal_summary = {
            "run_id": active_run_id,
            "engine": engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "gate_profile": gate_profile,
            "terminal_status": "completed",
            "failure_mode": promotion_gate.get("failure_mode"),
            "arena_outcome": None,
            "promotion_gate": promotion_gate,
            "training_metrics": training_metrics_payload,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": arena_lineage,
            "challenger_path": str(arena_candidate_path),
            "latest_checkpoint": str(arena_latest_checkpoint),
            "latest_candidate": arena_candidate_id,
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
            "policy_precheck": dict(policy_precheck_payload or {}),
            "policy_precheck_mode": policy_precheck_mode,
            "screen_results": list(screen_results),
            "screen_gate": dict(screen_gate or {}),
            "family_probe_report": dict(family_probe_report or {}),
            "champion_league_table": champion_league_table,
            "champion_rejection_report": champion_rejection_report,
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        return {
            "engine": engine,
            "horizon": horizon,
            "timeframe": config.primary_timeframe,
            "symbols": valid_symbols,
            "family": family,
            "feature_profile": feature_profile.get("profile_name"),
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_id": dataset_id,
            "dataset_source": dataset_source,
            "focus_symbols": focus_symbols,
            "gate_profile": gate_profile,
            "dataset_descriptor": dataset_descriptor,
            "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
            "games_per_symbol": games_per_symbol,
            "total_games": total_games,
            "latest_checkpoint": str(arena_latest_checkpoint),
            "challenger_path": str(arena_candidate_path),
            "training_metrics": training_metrics_payload,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "ga_campaign_id": ga_campaign_id,
            "ga_scope": ga_scope,
            "ga_parent_champion_id": ga_parent_champion_id,
            "seed_parent_champion_id": ga_parent_champion_id,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "resume_source": resume_source,
            "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or {}),
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "lineage": arena_lineage,
            "precheck": dict(gold_precheck_payload or {}),
            "policy_precheck": dict(policy_precheck_payload or {}),
            "policy_precheck_mode": policy_precheck_mode,
            "screen_results": list(screen_results),
            "screen_gate": dict(screen_gate or {}),
            "family_probe_report": dict(family_probe_report or {}),
            "champion_league_table": champion_league_table,
            "champion_rejection_report": champion_rejection_report,
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
        }

    battle_report = arena.battle(str(arena_candidate_path), champion_reference, horizon=horizon)
    logger.info("Verdict ADN %s: %s", horizon, battle_report["outcome"])
    logger.info(
        "Validation Arena %s: %s",
        horizon,
        battle_report.get("validation", {}),
    )

    challenger_metrics = battle_report["challenger"]["metrics"]
    registry_metrics = {
        "win_rate": {horizon: challenger_metrics.get("win_rate", 0.0)},
        "return_pct": {horizon: challenger_metrics.get("return_pct", 0.0)},
        "battles_won": {horizon: 1 if battle_report["outcome"] == "VICTORY" else 0},
        "horizon_accuracy": {horizon: challenger_metrics.get("win_rate", 0.0) / 100.0},
    }
    if ga_defer_promotion:
        promotion_gate = promoter.evaluate_promotion_gate(
            battle_report,
            gate_profile="standard",
        )
        live_comparison = promoter._compare_with_live_champion(
            horizon=horizon,
            candidate_gate=promotion_gate,
            engine=engine,
            challenger_id=arena_candidate_id,
        )
        promotion_result = {
            "status": "candidate_only",
            "reason": "deferred_ga_selection",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(arena_candidate_path),
            "champion_paths": [],
            "promotion_gate": promotion_gate,
            "artifact_compatibility": challenger_compatibility,
            "checkpoint_schema_version": challenger_checkpoint_schema_version,
            "requested_gate_profile": promoter.normalize_gate_profile(gate_profile or "standard"),
            "live_gate_profile": "standard",
            "live_comparison": live_comparison,
            "promotion_state": "candidate_only",
            "deferred_promotion": True,
            "ga_campaign_id": ga_campaign_id,
            "ga_trial": ga_trial,
            "ga_scope": ga_scope,
            "resume_source": resume_source,
            "lineage": arena_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
            "policy_precheck": dict(policy_precheck_payload or {}),
            "policy_precheck_mode": policy_precheck_mode,
            "screen_results": list(screen_results),
            "screen_gate": dict(screen_gate or {}),
            "family_probe_report": dict(family_probe_report or {}),
        }
    else:
        promotion_result = promoter.promote_muzero_challenger(
            challenger_path=arena_candidate_path,
            horizon=horizon,
            battle_report=battle_report,
            training_metrics=training_metrics_payload,
            latest_checkpoint=arena_latest_checkpoint,
            challenger_id=arena_candidate_id,
            gate_profile=gate_profile,
            promotion_metadata={
                "resume_source": resume_source,
                "lineage": arena_lineage,
                "seed_parent_champion_id": ga_parent_champion_id,
                "ga_campaign_id": ga_campaign_id,
                "ga_trial": ga_trial,
                "ga_scope": ga_scope,
                "policy_precheck": dict(policy_precheck_payload or {}),
                "policy_precheck_mode": policy_precheck_mode,
                "screen_results": list(screen_results),
                "screen_gate": dict(screen_gate or {}),
                "family_probe_report": dict(family_probe_report or {}),
            },
        )
    promotion_result.setdefault("policy_precheck", dict(policy_precheck_payload or {}))
    promotion_result.setdefault("policy_precheck_mode", policy_precheck_mode)
    promotion_result.setdefault("screen_results", list(screen_results))
    promotion_result.setdefault("screen_gate", dict(screen_gate or {}))
    promotion_result.setdefault("family_probe_report", dict(family_probe_report or {}))
    champion_league_table = _build_champion_league_table(
        engine=engine,
        horizon=horizon,
        screen_results=screen_results,
        selected_candidate_id=arena_candidate_id,
        battle_report=battle_report,
        promotion_result=promotion_result,
        screen_gate=screen_gate,
        family_probe_report=family_probe_report,
    )
    champion_rejection_report = _build_champion_rejection_report(
        screen_gate=screen_gate,
        promotion_gate=dict(promotion_result.get("promotion_gate") or {}),
        battle_report=battle_report,
        family_probe_report=family_probe_report,
    )
    promotion_result.setdefault("champion_league_table", champion_league_table)
    promotion_result.setdefault("champion_rejection_report", champion_rejection_report)
    logger.info("Promotion live %s: %s", horizon, promotion_result.get("status"))
    append_training_log(
        "MuZero "
        f"{horizon}: arena={battle_report.get('outcome')} | "
        f"promotion={promotion_result.get('status')} | "
        f"gate={promotion_result.get('reason') or promotion_result.get('promotion_gate', {}).get('reason') or 'aucun'}",
        source="muzero",
    )
    genetic.register_new_generation(
        gen_id=arena_candidate_id,
        metrics=registry_metrics,
        is_champion=promotion_result.get("status") == "promoted",
        horizon=horizon,
    )
    if promotion_result.get("status") != "promoted":
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status=(
                "candidate_only"
                if promotion_result.get("status") == "candidate_only"
                else "blocked"
            ),
            challenger_id=arena_candidate_id,
            challenger_path=str(arena_candidate_path),
            latest_checkpoint=str(arena_latest_checkpoint),
            battle_report=battle_report,
            training_metrics=training_metrics_payload,
            promotion_gate=dict(promotion_result.get("promotion_gate") or {}),
            promotion_result=promotion_result,
            artifact_compatibility=dict(promotion_result.get("artifact_compatibility") or {}),
            checkpoint_schema_version=(
                promotion_result.get("checkpoint_schema_version")
                or challenger_checkpoint_schema_version
            ),
            resume_source=resume_source,
            lineage=arena_lineage,
        )
    champion_paths = promotion_result.get("champion_paths", [])

    report_path = results_dir / f"arena_{horizon}_latest.json"
    report_payload = {
        "run_id": active_run_id,
        "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
        "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
        "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
        "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
        "engine": engine,
        "horizon": horizon,
        "timeframe": config.primary_timeframe,
        "symbols": valid_symbols,
        "family": family,
        "feature_profile": feature_profile.get("profile_name"),
        "mechanics_profile_version": str(getattr(config, "mechanics_profile_version", "") or "") or None,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "focus_symbols": focus_symbols,
        "gate_profile": gate_profile,
        "dataset_descriptor": dataset_descriptor,
        "dataset_coverage": dict(getattr(config, "dataset_coverage", {}) or {}),
        "games_per_symbol": games_per_symbol,
        "total_games": total_games,
        "latest_checkpoint": str(arena_latest_checkpoint),
        "challenger_path": str(arena_candidate_path),
        "live_champion_reference": champion_reference,
        "live_champion_id": live_champion_id or None,
        "champion_paths": champion_paths,
        "training_metrics": training_metrics_payload,
        "ga_status": str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None,
        "ga_generation": (
            int(os.getenv("TRAINING_GA_GENERATION", "0"))
            if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
            else None
        ),
        "ga_trial": str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None,
        "ga_campaign_id": ga_campaign_id,
        "ga_scope": ga_scope,
        "ga_parent_champion_id": ga_parent_champion_id,
        "seed_parent_champion_id": ga_parent_champion_id,
        "ga_defer_promotion": ga_defer_promotion,
        "ga_genome": ga_genome,
        "trial_mode": trial_mode,
        "trial_cost_profile": trial_cost_profile,
        "resume_source": resume_source,
        "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or challenger_compatibility),
        "checkpoint_schema_version": (
            promotion_result.get("checkpoint_schema_version")
            or challenger_checkpoint_schema_version
        ),
        "lineage": arena_lineage,
        "gold_precheck": dict(gold_precheck_payload or {}),
        "policy_precheck": dict(policy_precheck_payload or {}),
        "policy_precheck_mode": policy_precheck_mode,
        "screen_results": list(screen_results),
        "screen_gate": dict(screen_gate or {}),
        "family_probe_report": dict(family_probe_report or {}),
        "champion_league_table": champion_league_table,
        "champion_rejection_report": champion_rejection_report,
        "selected_checkpoint_step": selected_checkpoint_step,
        "battle_report": battle_report,
        "promotion": promotion_result,
    }
    unique_report_path = write_arena_summary(report_payload)
    report_payload["battle_report_path"] = str(unique_report_path)
    report_path.write_text(json.dumps(report_payload, indent=2, default=float), encoding="utf-8")
    logger.info("Rapport MuZero ecrit dans %s", report_path)
    promotion_gate = dict(promotion_result.get("promotion_gate") or {})
    challenger_payload = dict((battle_report.get("challenger") or {}))
    challenger_metrics_full = dict(challenger_payload.get("metrics") or {})
    terminal_summary = {
        "run_id": active_run_id,
        "sequence_id": str(os.getenv("TRAINING_SEQUENCE_ID", "")).strip() or None,
        "sequence_profile": str(os.getenv("TRAINING_SEQUENCE_PROFILE", "")).strip() or None,
        "window_id": str(os.getenv("TRAINING_WINDOW_ID", "")).strip() or None,
        "trial_id": str(os.getenv("TRAINING_TRIAL_ID", "")).strip() or ga_trial,
        "engine": engine,
        "horizon": horizon,
        "family": family,
        "feature_profile": feature_profile.get("profile_name"),
        "mechanics_profile_version": mechanics_profile_version,
        "ga_trial": ga_trial,
        "ga_campaign_id": ga_campaign_id,
        "ga_scope": ga_scope,
        "ga_parent_champion_id": ga_parent_champion_id,
        "seed_parent_champion_id": ga_parent_champion_id,
        "ga_genome": ga_genome,
        "trial_mode": trial_mode,
        "trial_cost_profile": trial_cost_profile,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "focus_symbols": focus_symbols,
        "gate_profile": gate_profile,
        "terminal_status": "completed",
        "failed_step": None,
        "failure_mode": (
            str(promotion_gate.get("failure_mode") or "").strip()
            or ("arena_defeat" if str(battle_report.get("outcome") or "").upper() != "VICTORY" else None)
        ),
        "arena_outcome": battle_report.get("outcome"),
        "promotion_gate": promotion_gate,
        "metrics": challenger_metrics_full,
        "metrics_by_symbol": dict(challenger_metrics_full.get("metrics_by_symbol") or {}),
        "metrics_by_position_mechanics": dict(
            challenger_metrics_full.get("metrics_by_position_mechanics") or {}
        ),
        "training_metrics": training_metrics_payload,
        "challenger_path": str(arena_candidate_path),
        "latest_checkpoint": str(arena_latest_checkpoint),
        "battle_report_path": str(unique_report_path),
        "live_comparison": dict(promotion_result.get("live_comparison") or {}),
        "resume_source": resume_source,
        "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or challenger_compatibility),
        "checkpoint_schema_version": (
            promotion_result.get("checkpoint_schema_version")
            or challenger_checkpoint_schema_version
        ),
        "lineage": arena_lineage,
        "artifact_state": {
            "arena_report_present": True,
            "battle_report_present": True,
            "promotion_present": bool(promotion_result),
            "candidate_checkpoint_present": arena_candidate_path.exists(),
            "latest_checkpoint_present": arena_latest_checkpoint.exists(),
        },
        "latest_candidate": arena_candidate_id,
        "latest_verdict": {
            "status": promotion_result.get("status"),
            "reason": promotion_result.get("reason") or promotion_gate.get("reason"),
            "failure_mode": promotion_gate.get("failure_mode"),
        },
        "gold_precheck": dict(gold_precheck_payload or {}),
        "precheck_status": (gold_precheck_payload or {}).get("status"),
        "policy_precheck": dict(policy_precheck_payload or {}),
        "policy_precheck_mode": policy_precheck_mode,
        "screen_results": list(screen_results),
        "screen_gate": dict(screen_gate or {}),
        "family_probe_report": dict(family_probe_report or {}),
        "champion_league_table": champion_league_table,
        "champion_rejection_report": champion_rejection_report,
        "selected_checkpoint_step": selected_checkpoint_step,
    }
    terminal_summary_path = write_terminal_summary(terminal_summary)
    logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
    record_arena_result(
        report_payload,
        metadata={
            "engine": engine,
            "run_trigger": str(os.getenv("TRAINING_RUN_TRIGGER", "manual") or "manual"),
            "ga_status": report_payload.get("ga_status"),
            "ga_generation": report_payload.get("ga_generation"),
            "ga_trial": report_payload.get("ga_trial"),
            "trial_mode": report_payload.get("trial_mode"),
            "trial_cost_profile": report_payload.get("trial_cost_profile"),
        },
    )
    send_horizon_summary(horizon, report_payload, promotion_result)
    return report_payload


if __name__ == "__main__":
    summary = main()
    logger.info("MuZero termine: %s", summary)

