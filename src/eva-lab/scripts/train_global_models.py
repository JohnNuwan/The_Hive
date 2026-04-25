"""Entraine MuZero sur historique reel puis execute la selection ADN."""

from __future__ import annotations

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
from eva_lab.muzero.checkpoint_utils import archive_muzero_artifacts
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
    resolve_feature_profile,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.train_muzero")



def build_environment(symbol: str, config: MuZeroConfigV3) -> TradingEnvironment | None:
    """Construit l'environnement MuZero a partir de l'historique du symbole."""
    frame = load_history_frame(symbol, config.primary_timeframe)
    if frame is None:
        logger.warning("Historique absent pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    history_bars = get_horizon_history_bars(config.horizon, env_prefix="MUZERO_HISTORY", fallback=4000)
    market_data, day_labels = build_muzero_market_context(frame.tail(history_bars))
    if market_data.shape[0] < 240:
        logger.warning("Historique insuffisant pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    max_steps = min(config.max_moves, market_data.shape[0] - 101)
    env = TradingEnvironment(
        data=market_data,
        day_labels=day_labels,
        symbol=symbol,
        config=config,
        max_steps=max_steps,
        training_mode=True,
        training_progress_step=0,
    )
    setattr(env, "dataset_source", str(frame.attrs.get("dataset_source") or getattr(config, "dataset_source", "csv")))
    return env



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
    metrics: dict[str, object],
    key: str,
    default: float = 0.0,
) -> float:
    """Convertit une metrique libre en flottant robuste.

    Args:
        metrics (dict[str, object]): Dictionnaire source.
        key (str): Nom de la metrique.
        default (float): Valeur de repli.

    Returns:
        float: Valeur convertie ou repli.
    """
    try:
        return float(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _normalize_rate(value: float) -> float:
    """Normalise un taux qui peut etre exprime en ratio ou en pourcentage.

    Args:
        value (float): Valeur brute.

    Returns:
        float: Valeur ramenee dans `[0, 1]` si necessaire.
    """
    return value / 100.0 if value > 1.0 else value


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

    loss_values = [_to_metric_float(item, "loss_pol", default=999.0) for item in history]
    top1_values = [_to_metric_float(item, "policy_top1_share", default=0.0) for item in history]
    entropy_values = [_to_metric_float(item, "policy_entropy", default=1.0) for item in history]
    root_mask_values = [_to_metric_float(item, "root_mask_rate", default=1.0) for item in history]
    post_veto_values = [_to_metric_float(item, "post_veto_to_hold_rate", default=1.0) for item in history]
    soft_ratio_values = [
        _to_metric_float(item, "soft_penalty_to_bonus_ratio", default=999.0)
        for item in history
    ]
    balanced_values = [
        _normalize_rate(_to_metric_float(item, "balanced_episode_rate", default=0.0))
        for item in history
    ]
    long_share_values = [_to_metric_float(item, "long_entry_share", default=0.0) for item in history]
    short_share_values = [_to_metric_float(item, "short_entry_share", default=0.0) for item in history]

    medians = {
        "loss_pol": statistics.median(loss_values),
        "policy_top1_share": statistics.median(top1_values),
        "policy_entropy": statistics.median(entropy_values),
        "root_mask_rate": statistics.median(root_mask_values),
        "post_veto_to_hold_rate": statistics.median(post_veto_values),
        "soft_penalty_to_bonus_ratio": statistics.median(soft_ratio_values),
        "balanced_episode_rate": statistics.median(balanced_values),
        "long_entry_share": statistics.median(long_share_values),
        "short_entry_share": statistics.median(short_share_values),
    }
    full_checks = {
        "loss_pol": medians["loss_pol"] <= float(getattr(config, "policy_precheck_max_loss_pol", 5.8) or 5.8),
        "policy_top1_share": medians["policy_top1_share"] >= float(
            getattr(config, "policy_precheck_min_top1_share", 0.75) or 0.75
        ),
        "policy_entropy": medians["policy_entropy"] <= float(
            getattr(config, "policy_precheck_max_policy_entropy", 1.0) or 1.0
        ),
        "root_mask_rate": medians["root_mask_rate"] <= float(
            getattr(config, "policy_precheck_max_root_mask_rate", 0.05) or 0.05
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
    }
    screen_checks = {
        "loss_pol": medians["loss_pol"] <= float(getattr(config, "policy_screen_max_loss_pol", 6.6) or 6.6),
        "policy_top1_share": medians["policy_top1_share"] >= float(
            getattr(config, "policy_screen_min_top1_share", 0.88) or 0.88
        ),
        "policy_entropy": medians["policy_entropy"] <= float(
            getattr(config, "policy_screen_max_policy_entropy", 0.45) or 0.45
        ),
        "root_mask_rate": medians["root_mask_rate"] <= float(
            getattr(config, "policy_screen_max_root_mask_rate", 0.05) or 0.05
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
        "latest_metrics": dict(history[-1] or {}),
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
    challenger_metrics = dict((dict(battle_report.get("challenger") or {})).get("metrics") or {})
    checks = {
        "profit_factor": _to_metric_float(challenger_metrics.get("profit_factor")) >= float(
            getattr(config, "arena_screen_min_profit_factor", 1.20) or 1.20
        ),
        "return_pct": _to_metric_float(challenger_metrics.get("return_pct")) > float(
            getattr(config, "arena_screen_min_return_pct", 0.0) or 0.0
        ),
        "expectancy_pct": _to_metric_float(challenger_metrics.get("expectancy_pct")) > float(
            getattr(config, "arena_screen_min_expectancy_pct", 0.0) or 0.0
        ),
        "positive_episode_rate": _to_metric_float(challenger_metrics.get("positive_episode_rate")) >= float(
            getattr(config, "arena_screen_min_positive_episode_rate", 55.0) or 55.0
        ),
        "directional_bias": str(challenger_metrics.get("directional_bias") or "inactive").strip().lower()
        == "balanced",
    }
    failure_reason = next((name for name, passed in checks.items() if not passed), "eligible")
    return {
        "allowed": all(checks.values()),
        "status": "eligible" if all(checks.values()) else "blocked",
        "reason": failure_reason,
        "checks": checks,
        "metrics": challenger_metrics,
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
    logger.info("Inventaire historique: %s", build_inventory_report())
    logger.info("Univers MuZero: %s", config.symbols)
    append_training_log(
        f"MuZero {horizon} demarre sur {len(config.symbols)} symboles.",
        source="muzero",
    )
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
        for symbol_index, symbol in enumerate(config.symbols, start=1):
            env = build_environment(symbol, config)
            if env is None:
                continue
            valid_symbols.append(symbol)
            append_training_log(
                f"MuZero {horizon}: collecte sur {symbol} ({symbol_index}/{len(config.symbols)}).",
                source="muzero",
            )
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
                        episode_step_total=int(heartbeat.get("max_moves", config.max_moves) or config.max_moves),
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
                policy_precheck_history.append(
                    {
                        **dict(metrics),
                        "training_step": step,
                    }
                )
                merge_training_status(
                    {
                        "latest_metrics": dict(last_metrics),
                        "train_step_phase": "optimisation",
                        "phase_durations_ms": phase_durations_ms,
                    }
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
                                "policy_precheck_passed": policy_precheck_payload.get("status") != "blocked",
                                "policy_precheck_mode": policy_precheck_payload.get("status"),
                            },
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
        and not killed_after_precheck
        and last_metrics is not None
    ):
        final_policy_precheck = _evaluate_policy_precheck_window(
            history=list(policy_precheck_history),
            config=config,
            step=last_optimization_step,
            stage="pre_arena",
        )
        policy_precheck_payload = dict(final_policy_precheck)
        merge_training_status(
            {
                "policy_precheck": dict(policy_precheck_payload),
                "latest_metrics": {
                    **dict(last_metrics),
                    "policy_precheck_passed": policy_precheck_payload.get("status") != "blocked",
                    "policy_precheck_mode": policy_precheck_payload.get("status"),
                },
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
            promoter.persist_challenger_manifest(
                engine=engine,
                horizon=horizon,
                status="blocked",
                challenger_id=selected_candidate_id,
                challenger_path=str(selected_challenger_path),
                latest_checkpoint=str(selected_latest_checkpoint),
                battle_report=dict((_select_best_screen_candidate(screen_results).get("battle_report") or {}))
                if screen_results
                else None,
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
            },
        )
    promotion_result.setdefault("policy_precheck", dict(policy_precheck_payload or {}))
    promotion_result.setdefault("policy_precheck_mode", policy_precheck_mode)
    promotion_result.setdefault("screen_results", list(screen_results))
    promotion_result.setdefault("screen_gate", dict(screen_gate or {}))
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

