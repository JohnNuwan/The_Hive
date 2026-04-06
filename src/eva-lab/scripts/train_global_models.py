"""Entraine MuZero sur historique reel puis execute la selection ADN."""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

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
from eva_lab.muzero.collector import (
    CollectorEnvironmentPayload,
    collect_games_parallel,
)
from eva_lab.muzero.environment import TradingEnvironment
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.shadow_dataset import load_shadow_games
from eva_lab.timescale_store import record_arena_result, record_training_dataset
from eva_lab.training_notifier import send_horizon_summary, send_training_horizon_started
from eva_lab.training_status import (
    append_training_log,
    load_training_status,
    mark_step_running,
    set_gold_precheck,
    set_training_weighting,
    set_training_runtime_state,
    write_precheck_summary,
    write_terminal_summary,
)
from eva_lab.training_utils import (
    build_inventory_report,
    build_muzero_market_data,
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
    market_data = build_muzero_market_data(frame.tail(history_bars))
    if market_data.shape[0] < 240:
        logger.warning("Historique insuffisant pour %s sur %s.", symbol, config.primary_timeframe)
        return None

    max_steps = min(config.max_moves, market_data.shape[0] - 101)
    env = TradingEnvironment(data=market_data, symbol=symbol, config=config, max_steps=max_steps)
    setattr(env, "dataset_source", str(frame.attrs.get("dataset_source") or getattr(config, "dataset_source", "csv")))
    return env


def _load_shadow_replay_bundle(
    *,
    agent: JAXMuZeroAgent,
    config: MuZeroConfigV3,
    focus_symbols: list[str],
) -> dict[str, object]:
    """Charge un replay shadow pondere dans le buffer MuZero.

    Args:
        agent (JAXMuZeroAgent): Agent MuZero actif.
        config (MuZeroConfigV3): Configuration du run courant.
        focus_symbols (list[str]): Univers explicitement retenu pour le run.

    Returns:
        dict[str, object]: Resume compact du chargement shadow.
    """
    shadow_enabled = str(os.getenv("MUZERO_SHADOW_REPLAY_ENABLED", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    raw_data_dirs = str(os.getenv("TRAINING_SHADOW_DIRS", "data/shadow_learning") or "").strip()
    shadow_dirs = [
        candidate.strip()
        for chunk in raw_data_dirs.split(os.pathsep)
        for candidate in chunk.split(",")
        if candidate.strip()
    ]
    replay_capacity = max(int(getattr(agent.replay_buffer, "max_games", 0) or 0), 1)
    max_games_default = max(replay_capacity // 2, 1)
    max_games_raw = int(str(os.getenv("MUZERO_SHADOW_MAX_GAMES", str(max_games_default))).strip() or max_games_default)
    max_games = max_games_raw if max_games_raw > 0 else None
    winner_symbols = [
        item.strip()
        for item in str(os.getenv("TRAINING_WINNER_SYMBOLS", "") or "").split(",")
        if item.strip()
    ]
    risk_symbols = [
        item.strip()
        for item in str(os.getenv("TRAINING_RISK_SYMBOLS", "") or "").split(",")
        if item.strip()
    ]
    weighting_profile = {
        "base_weight": float(str(os.getenv("TRAINING_EPISODE_WEIGHT_BASE", "1.0")).strip() or 1.0),
        "winner_bonus": float(
            str(os.getenv("TRAINING_EPISODE_WEIGHT_WINNER_BONUS", "0.15")).strip() or 0.15
        ),
        "loser_bonus": float(
            str(os.getenv("TRAINING_EPISODE_WEIGHT_LOSER_BONUS", "0.35")).strip() or 0.35
        ),
        "nemesis_bonus": float(
            str(os.getenv("TRAINING_EPISODE_WEIGHT_NEMESIS_BONUS", "0.55")).strip() or 0.55
        ),
        "risk_symbol_bonus": float(
            str(os.getenv("TRAINING_EPISODE_WEIGHT_RISK_BONUS", "0.25")).strip() or 0.25
        ),
    }
    observation_size = 1
    for dimension in tuple(getattr(config, "observation_shape", ()) or ()):
        observation_size *= max(int(dimension or 1), 1)

    summary: dict[str, object] = {
        "enabled": shadow_enabled,
        "data_dirs": shadow_dirs,
        "max_games": max_games,
        "focus_symbols": list(focus_symbols),
        "winner_symbols": list(winner_symbols),
        "risk_symbols": list(risk_symbols),
        "weighting_profile": dict(weighting_profile),
        "episodes_loaded": 0,
        "loaded_into_replay": 0,
        "weighted_episode_counts": {},
        "weighted_priority_total": 0.0,
        "replay_buffer_size_after_load": agent.replay_buffer.size,
    }
    if not shadow_enabled:
        summary["reason"] = "shadow_replay_disabled"
        return summary
    if not shadow_dirs:
        summary["reason"] = "shadow_dirs_missing"
        return summary

    games, weighting_summary = load_shadow_games(
        shadow_dirs,
        observation_size=observation_size,
        action_space_size=int(getattr(config, "action_space_size", 5) or 5),
        winner_symbols=winner_symbols,
        risk_symbols=risk_symbols,
        allowed_symbols=focus_symbols,
        max_games=max_games,
        weighting_profile=weighting_profile,
        include_weighting_summary=True,
    )
    for game in games:
        agent.replay_buffer.save_game(game)

    summary.update(dict(weighting_summary or {}))
    summary["loaded_into_replay"] = len(games)
    summary["episodes_loaded"] = int(weighting_summary.get("episodes_loaded", len(games)) or len(games))
    summary["replay_buffer_size_after_load"] = agent.replay_buffer.size
    summary["reason"] = "shadow_replay_loaded" if games else "shadow_replay_empty"
    return summary


def _build_traceback_tail(exc: Exception, max_lines: int = 12) -> list[str]:
    """Construit une queue compacte de traceback pour le statut runtime.

    Args:
        exc (Exception): Exception a serialiser.
        max_lines (int): Nombre maximal de lignes utiles a conserver.

    Returns:
        list[str]: Lignes finales du traceback.
    """

    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    flattened = [line.strip() for line in "".join(lines).splitlines() if line.strip()]
    return flattened[-max(max_lines, 1):]


def _snapshot_agent_training_state(agent: JAXMuZeroAgent) -> dict[str, object]:
    """Capture l'etat d'optimisation MuZero pour un precheck reversible.

    Args:
        agent (JAXMuZeroAgent): Agent a figer temporairement.

    Returns:
        dict[str, object]: Etat serialisable des poids et de l'optimiseur.
    """

    return {
        "params": pickle.loads(pickle.dumps(agent.params)),
        "opt_state": pickle.loads(pickle.dumps(agent.opt_state)),
    }


def _restore_agent_training_state(agent: JAXMuZeroAgent, snapshot: dict[str, object]) -> None:
    """Restaure l'etat d'optimisation MuZero apres un precheck.

    Args:
        agent (JAXMuZeroAgent): Agent a restaurer.
        snapshot (dict[str, object]): Instantane precedemment capture.
    """

    agent.params = snapshot["params"]
    agent.opt_state = snapshot["opt_state"]


def _read_gpu_memory_snapshot() -> dict[str, float] | None:
    """Lit l'occupation memoire du premier GPU visible.

    Returns:
        dict[str, float] | None: Memoire utilisee, totale et ratio, ou
        ``None`` si la sonde est indisponible.
    """

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    first_line = str(result.stdout or "").strip().splitlines()
    if not first_line:
        return None
    try:
        used_raw, total_raw = [part.strip() for part in first_line[0].split(",", 1)]
        used_mb = float(used_raw)
        total_mb = float(total_raw)
    except (TypeError, ValueError):
        return None
    if total_mb <= 0.0:
        return None
    return {
        "used_mb": round(used_mb, 3),
        "total_mb": round(total_mb, 3),
        "usage_ratio": round(used_mb / total_mb, 6),
    }


def _autotune_muzero_batch_size(
    *,
    agent: JAXMuZeroAgent,
    config: MuZeroConfigV3,
) -> dict[str, object]:
    """Calibre automatiquement le batch JAX avant l'optimisation longue.

    Args:
        agent (JAXMuZeroAgent): Agent MuZero deja initialise.
        config (MuZeroConfigV3): Configuration courante du run.

    Returns:
        dict[str, object]: Profil choisi et details des candidats testes.
    """

    current_batch_size = max(int(getattr(config, "batch_size", 32) or 32), 1)
    candidates = []
    for candidate in list(getattr(config, "batch_autotune_candidates", []) or []):
        try:
            normalized = max(int(candidate), 1)
        except (TypeError, ValueError):
            continue
        if normalized not in candidates:
            candidates.append(normalized)
    if current_batch_size not in candidates:
        candidates.insert(0, current_batch_size)
    gpu_memory_limit_ratio = float(
        str(os.getenv("MUZERO_BATCH_AUTOTUNE_GPU_MAX_RATIO", "0.85")).strip() or 0.85
    )
    if not getattr(config, "batch_autotune_enabled", False):
        return {
            "enabled": False,
            "selected_batch_size": current_batch_size,
            "baseline_batch_size": current_batch_size,
            "candidates": candidates,
            "gpu_memory_limit_ratio": gpu_memory_limit_ratio,
            "reason": "batch_autotune_disabled",
            "candidate_results": [],
        }

    base_snapshot = _snapshot_agent_training_state(agent)
    baseline_batch_size = current_batch_size
    selected_batch_size = current_batch_size
    stable_reference_throughput: float | None = None
    candidate_results: list[dict[str, object]] = []

    append_training_log(
        (
            "MuZero %s: autotune batch JAX sur %s."
            % (config.horizon, ",".join(str(item) for item in candidates))
        ),
        source="muzero",
    )
    set_training_runtime_state(
        train_step_phase="batch_autotune",
        jax_batch_profile={
            "enabled": True,
            "status": "running",
            "selected_batch_size": current_batch_size,
            "baseline_batch_size": baseline_batch_size,
            "candidates": list(candidates),
            "gpu_memory_limit_ratio": gpu_memory_limit_ratio,
            "candidate_results": [],
        },
    )

    try:
        for candidate_batch in candidates:
            config.batch_size = candidate_batch
            _restore_agent_training_state(agent, base_snapshot)
            try:
                warmup_result = agent.train_step()
                if warmup_result is None:
                    raise RuntimeError(
                        "BATCH_AUTOTUNE_PRECHECK_FAILED: replay insuffisant pour le candidat "
                        f"{candidate_batch}."
                    )

                measured_results: list[dict[str, object]] = []
                total_wall_time_ms = 0.0
                for _ in range(3):
                    step_started_at = time.perf_counter()
                    step_result = agent.train_step()
                    if step_result is None:
                        raise RuntimeError(
                            "BATCH_AUTOTUNE_PRECHECK_FAILED: aucun batch produit pour le candidat "
                            f"{candidate_batch}."
                        )
                    measured_results.append(step_result)
                    total_wall_time_ms += max(
                        (time.perf_counter() - step_started_at) * 1000.0,
                        0.001,
                    )

                gpu_snapshot = _read_gpu_memory_snapshot()
                prepare_batch_ms = round(
                    sum(
                        float(item.get("phase_durations_ms", {}).get("prepare_batch", 0.0) or 0.0)
                        for item in measured_results
                    )
                    / max(len(measured_results), 1),
                    3,
                )
                update_fn_ms = round(
                    sum(
                        float(item.get("phase_durations_ms", {}).get("update_fn", 0.0) or 0.0)
                        for item in measured_results
                    )
                    / max(len(measured_results), 1),
                    3,
                )
                throughput = round(
                    (candidate_batch * len(measured_results)) / max(total_wall_time_ms / 1000.0, 1e-6),
                    3,
                )
                memory_ratio = float((gpu_snapshot or {}).get("usage_ratio", 0.0) or 0.0)
                candidate_result = {
                    "batch_size": candidate_batch,
                    "status": "stable",
                    "prepare_batch_ms": prepare_batch_ms,
                    "update_fn_ms": update_fn_ms,
                    "samples_per_second": throughput,
                    "gpu_memory": gpu_snapshot,
                    "warmup_phase_durations_ms": dict(
                        warmup_result.get("phase_durations_ms") or {}
                    ),
                    "measured_steps": len(measured_results),
                }
                if gpu_snapshot and memory_ratio > gpu_memory_limit_ratio:
                    candidate_result["status"] = "rejected_memory"
                    candidate_result["reason"] = (
                        "Le candidat depasse la limite memoire GPU "
                        f"({memory_ratio:.2%} > {gpu_memory_limit_ratio:.2%})."
                    )
                else:
                    if stable_reference_throughput is None:
                        selected_batch_size = candidate_batch
                        stable_reference_throughput = throughput
                    elif candidate_batch > selected_batch_size and throughput >= (
                        stable_reference_throughput * 1.10
                    ):
                        selected_batch_size = candidate_batch
                        stable_reference_throughput = throughput
                    else:
                        candidate_result["selection_reason"] = "gain_insuffisant"
                candidate_results.append(candidate_result)
            except Exception as exc:
                candidate_results.append(
                    {
                        "batch_size": candidate_batch,
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
            finally:
                _restore_agent_training_state(agent, base_snapshot)
    finally:
        config.batch_size = selected_batch_size
        _restore_agent_training_state(agent, base_snapshot)

    result = {
        "enabled": True,
        "status": "completed",
        "baseline_batch_size": baseline_batch_size,
        "selected_batch_size": selected_batch_size,
        "candidates": list(candidates),
        "gpu_memory_limit_ratio": gpu_memory_limit_ratio,
        "candidate_results": candidate_results,
    }
    set_training_runtime_state(jax_batch_profile=result)
    append_training_log(
        (
            f"MuZero {config.horizon}: batch JAX selectionne={selected_batch_size} "
            f"(baseline={baseline_batch_size})."
        ),
        source="muzero",
    )
    logger.info(
        "Autotune batch MuZero %s: baseline=%s | selection=%s | candidats=%s",
        config.horizon,
        baseline_batch_size,
        selected_batch_size,
        candidate_results,
    )
    return result


def _collect_precheck_warmup_games(
    *,
    agent: JAXMuZeroAgent,
    config: MuZeroConfigV3,
    required_entries: int,
    max_wall_time_seconds: float | None,
) -> dict[str, object]:
    """Complete un replay minimal avant le precheck d'optimisation.

    Args:
        agent (JAXMuZeroAgent): Agent MuZero courant.
        config (MuZeroConfigV3): Configuration du run.
        required_entries (int): Nombre minimal d'episodes dans le replay.
        max_wall_time_seconds (float | None): Garde-fou de collecte.

    Returns:
        dict[str, object]: Resume compact du warmup.

    Raises:
        RuntimeError: Si aucun buffer minimal ne peut etre produit.
    """

    attempts = 0
    collected_symbols: list[str] = []
    max_attempts = max(len(config.symbols) * max(required_entries, 1), 1)
    while agent.replay_buffer.size < required_entries and attempts < max_attempts:
        symbol = config.symbols[attempts % len(config.symbols)]
        attempts += 1
        env = build_environment(symbol, config)
        if env is None:
            continue
        agent.play_game(
            env,
            exploration=True,
            collection_mode="policy_only",
            max_wall_time_seconds=max_wall_time_seconds,
        )
        collected_symbols.append(symbol)
    if agent.replay_buffer.size < required_entries:
        raise RuntimeError(
            "OPTIMISATION_PRECHECK_FAILED: buffer insuffisant pour le precheck "
            f"({agent.replay_buffer.size}/{required_entries})."
        )
    return {
        "attempts": attempts,
        "collected_symbols": collected_symbols,
        "buffer_size": agent.replay_buffer.size,
    }


def _record_muzero_failure(
    *,
    exc: Exception,
    failed_phase: str,
    run_id: str,
    step_name: str,
    engine: str,
    horizon: str,
    family: str,
    feature_profile: dict[str, object],
    mechanics_profile_version: str | None,
    dataset_id: str,
    dataset_source: str,
    focus_symbols: list[str],
    gate_profile: str,
    symbol_universe: list[str],
    ga_trial: str | None,
    trial_mode: str | None,
    trial_cost_profile: str | None,
    resume_checkpoint_path: str | None,
    resume_step: int | None,
    last_metrics: dict[str, object] | None = None,
) -> Path:
    """Persiste un echec MuZero exploitable par le runtime et l'operateur.

    Args:
        exc (Exception): Exception capturee.
        failed_phase (str): Sous-phase runtime ayant echoue.
        run_id (str): Identifiant du run courant.
        step_name (str): Etape nightly courante.
        engine (str): Moteur d'entrainement.
        horizon (str): Horizon MuZero concerne.
        family (str): Famille de symboles retenue.
        feature_profile (dict[str, object]): Profil de features courant.
        mechanics_profile_version (str | None): Version mecanique active.
        dataset_id (str): Identifiant de dataset.
        dataset_source (str): Source effective de dataset.
        focus_symbols (list[str]): Symboles prioritaires du run.
        gate_profile (str): Profil de gate actif.
        symbol_universe (list[str]): Univers reel du run.
        ga_trial (str | None): Trial GA associe.
        trial_mode (str | None): Mode de trial.
        trial_cost_profile (str | None): Profil de cout.
        resume_checkpoint_path (str | None): Checkpoint de reprise.
        resume_step (int | None): Step de reprise.
        last_metrics (dict[str, object] | None): Metriques partielles si disponibles.

    Returns:
        Path: Chemin du resume terminal ecrit.
    """

    exception_type = type(exc).__name__
    exception_message = str(exc)
    traceback_tail = _build_traceback_tail(exc)
    runtime_snapshot = dict(load_training_status() or {})
    error_context = {
        "step_name": step_name,
        "failed_phase": failed_phase,
        "run_id": run_id or None,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }
    set_training_runtime_state(
        train_step_phase=failed_phase,
        failed_phase=failed_phase,
        exception_type=exception_type,
        exception_message=exception_message,
        traceback_tail=traceback_tail,
        stall_detected=True,
        stall_reason=exception_message,
        last_nonzero_exit=error_context,
    )
    append_training_log(
        (
            f"MuZero {horizon}: echec pendant {failed_phase} | "
            f"{exception_type}: {exception_message}"
        ),
        level="ERROR",
        source="muzero",
    )
    if traceback_tail:
        append_training_log(
            "Traceback MuZero: " + " || ".join(traceback_tail[-3:]),
            level="ERROR",
            source="muzero",
        )

    terminal_summary = {
        "run_id": run_id or None,
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
        "trial_mode": trial_mode,
        "trial_cost_profile": trial_cost_profile,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "focus_symbols": focus_symbols,
        "symbol_universe": list(symbol_universe),
        "gate_profile": gate_profile,
        "terminal_status": "error",
        "failed_step": failed_phase,
        "failure_mode": "optimization_pipeline_error",
        "arena_outcome": None,
        "promotion_gate": {
            "allowed": False,
            "status": "blocked",
            "reason": failed_phase,
            "failure_mode": "optimization_pipeline_error",
        },
        "metrics": dict(last_metrics or {}),
        "metrics_by_symbol": dict((last_metrics or {}).get("metrics_by_symbol") or {}),
        "artifact_state": {
            "arena_report_present": False,
            "battle_report_present": False,
            "promotion_present": False,
            "candidate_checkpoint_present": False,
        },
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_step": resume_step,
        "latest_verdict": {
            "status": "error",
            "reason": exception_message,
            "failure_mode": "optimization_pipeline_error",
        },
        "collector_profile": {
            "collector_mode": runtime_snapshot.get("collector_mode"),
            "collector_workers": runtime_snapshot.get("collector_workers"),
            "collector_active_symbols": list(runtime_snapshot.get("collector_active_symbols") or []),
            "collector_queue_depth": runtime_snapshot.get("collector_queue_depth"),
            "inference_batch_profile": dict(runtime_snapshot.get("inference_batch_profile") or {}),
        },
        "batch_autotune_result": dict(runtime_snapshot.get("jax_batch_profile") or {}),
        "runtime_failure": {
            **error_context,
            "traceback_tail": traceback_tail,
            "model_family": family,
            "symbol_universe": list(symbol_universe),
        },
    }
    return write_terminal_summary(terminal_summary)



def _parse_proxy_precheck_steps(raw_steps: str) -> list[int]:
    """Normalise la liste des etapes de precheck proxy.

    Args:
        raw_steps (str): Valeur CSV issue de l'environnement.

    Returns:
        list[int]: Etapes strictement positives, triees et dedoublonnees.
    """

    steps: list[int] = []
    for raw_part in str(raw_steps or "").split(","):
        candidate = str(raw_part).strip()
        if not candidate:
            continue
        try:
            step = int(candidate)
        except ValueError:
            continue
        if step > 0 and step not in steps:
            steps.append(step)
    return sorted(steps)


def _is_gold_proxy_precheck_enabled(
    *,
    gate_profile: str,
    trial_mode: str | None,
    focus_symbols: list[str],
) -> bool:
    """Retourne vrai si le run courant doit executer un precheck proxy.

    Args:
        gate_profile (str): Profil de gate du run.
        trial_mode (str | None): Mode courant du trial.
        focus_symbols (list[str]): Univers explicite du run.

    Returns:
        bool: ``True`` si le precheck proxy est pertinent, ``False`` sinon.
    """

    if str(trial_mode or "").strip().lower() != "proxy_ga":
        return False
    if str(os.getenv("MUZERO_PROXY_PRECHECK_ENABLED", "")).strip():
        return str(os.getenv("MUZERO_PROXY_PRECHECK_ENABLED", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    normalized_focus = [str(symbol).strip().upper() for symbol in focus_symbols if str(symbol).strip()]
    return (
        str(gate_profile or "").strip().lower() == "gold_demo"
        and normalized_focus == ["XAUUSD"]
    )


def _evaluate_gold_precheck_verdict(
    *,
    metrics: dict[str, object],
    mechanics: dict[str, object],
) -> dict[str, object]:
    """Etablit un verdict de precheck proxy a partir des metriques intermediaires.

    Args:
        metrics (dict[str, object]): Metriques consolidees du challenger.
        mechanics (dict[str, object]): Metriques de mecanique de position.

    Returns:
        dict[str, object]: Verdict structure ``pass`` ou ``fail`` avec raison stable.
    """

    evaluation_games = int(metrics.get("evaluation_games", 0) or 0)
    total_trades = int(metrics.get("total_trades", 0) or 0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
    net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
    directional_imbalance = float(metrics.get("directional_imbalance", 1.0) or 1.0)
    close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
    hold_drag_score = float(mechanics.get("hold_drag_score", 0.0) or 0.0)
    directional_bias = str(metrics.get("directional_bias") or "inactive").strip().lower()

    if evaluation_games <= 0 or total_trades <= 0:
        return {
            "status": "fail",
            "reason": "insufficient_sample",
            "failure_mode": "inactive",
        }

    if profit_factor <= 1.0:
        return {
            "status": "fail",
            "reason": "profit_factor_too_low",
            "failure_mode": "unprofitable",
        }

    if return_pct <= 0.0:
        return {
            "status": "fail",
            "reason": "return_pct_non_positive",
            "failure_mode": "unprofitable",
        }

    if net_realized_pct <= 0.0:
        return {
            "status": "fail",
            "reason": "net_realized_pct_non_positive",
            "failure_mode": "unprofitable",
        }

    if close_quality_score < 0.40:
        return {
            "status": "fail",
            "reason": "close_quality_too_low",
            "failure_mode": "bad_exit",
        }

    if hold_drag_score > 0.80:
        return {
            "status": "fail",
            "reason": "hold_drag_too_high",
            "failure_mode": "bad_exit",
        }

    if directional_imbalance > 0.70:
        return {
            "status": "fail",
            "reason": "directional_imbalance_too_high",
            "failure_mode": directional_bias or "directional_imbalance",
        }

    if directional_bias in {"buy_heavy", "sell_heavy"} and return_pct <= 0.0:
        return {
            "status": "fail",
            "reason": "directional_bias_negative",
            "failure_mode": directional_bias,
        }

    return {"status": "pass", "reason": "proxy_precheck_pass", "failure_mode": None}


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
    trial_mode = str(os.getenv("TRAINING_TRIAL_MODE", "")).strip() or None
    trial_cost_profile = str(os.getenv("TRAINING_TRIAL_COST_PROFILE", "")).strip() or None
    gate_profile = str(os.getenv("TRAINING_GATE_PROFILE", "")).strip() or "standard"
    focus_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in config.symbols if str(symbol).strip()))
    replay_cache_key = f"{engine}:{horizon}:{initial_family or 'global'}:{mechanics_profile_version or 'default'}"
    gold_precheck_enabled = _is_gold_proxy_precheck_enabled(
        gate_profile=gate_profile,
        trial_mode=trial_mode,
        focus_symbols=focus_symbols,
    )
    proxy_precheck_steps = _parse_proxy_precheck_steps(
        str(
            os.getenv(
                "MUZERO_PROXY_PRECHECK_STEPS",
                os.getenv("MUZERO_GOLD_PRECHECK_STEP", "3000"),
            )
        )
    )
    if not proxy_precheck_steps:
        proxy_precheck_steps = [3000]
    gold_precheck_games = max(
        1,
        int(
            os.getenv(
                "MUZERO_PROXY_PRECHECK_GAMES",
                os.getenv("MUZERO_GOLD_PRECHECK_GAMES", "6"),
            )
        ),
    )
    resume_checkpoint_path = str(os.getenv("MUZERO_RESUME_CHECKPOINT_PATH", "")).strip() or None
    resume_step_override = int(str(os.getenv("MUZERO_RESUME_STEP", "0")).strip() or 0)
    collection_timeout_seconds = max(
        0.0,
        float(str(os.getenv("MUZERO_COLLECTION_GAME_TIMEOUT_SECONDS", "240")).strip() or 0.0),
    )
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
    active_run_id = str(load_training_status().get("run_id") or "").strip() or None
    resume_step = 0
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
    set_training_runtime_state(
        last_successful_step=None,
        last_successful_step_at=None,
        train_step_phase="initialisation",
        phase_durations_ms={},
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=None,
        stall_detected=False,
        stall_reason=None,
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
    if resume_checkpoint_path:
        resume_path = Path(resume_checkpoint_path)
        if not resume_path.exists():
            raise ValueError(f"Checkpoint de reprise introuvable: {resume_checkpoint_path}")
        resume_metadata = agent.load(str(resume_path))
        resume_step = max(int(resume_metadata.get("step") or 0), resume_step_override)
        logger.info("Reprise MuZero explicite depuis %s a l'etape %s", resume_path, resume_step)
        append_training_log(
            f"MuZero {horizon}: reprise depuis {resume_path.name} a l'etape {resume_step}.",
            source="muzero",
        )
        set_training_runtime_state(
            resume_checkpoint_path=resume_checkpoint_path,
            resume_step=resume_step if resume_step > 0 else None,
        )
    elif latest_path.exists():
        try:
            agent.load(str(latest_path))
            logger.info("Reprise MuZero depuis %s", latest_path)
        except Exception as exc:
            logger.warning("Checkpoint MuZero ignore: %s", exc)

    existing_training_weighting = dict(load_training_status().get("training_weighting") or {})
    shadow_weighting = _load_shadow_replay_bundle(
        agent=agent,
        config=config,
        focus_symbols=focus_symbols,
    )
    merged_training_weighting = dict(existing_training_weighting)
    merged_training_weighting.update(shadow_weighting)
    if existing_training_weighting.get("gnn_focus_symbol") and not merged_training_weighting.get("gnn_focus_symbol"):
        merged_training_weighting["gnn_focus_symbol"] = existing_training_weighting.get("gnn_focus_symbol")
    set_training_weighting(merged_training_weighting)
    append_training_log(
        (
            f"MuZero {horizon}: replay shadow "
            f"{shadow_weighting.get('loaded_into_replay', 0)} episode(s) charge(s) "
            f"depuis {len(list(shadow_weighting.get('data_dirs') or []))} dossier(s)."
        ),
        source="muzero",
    )
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
        trial_mode=trial_mode,
        trial_cost_profile=trial_cost_profile,
        focus_symbols=focus_symbols,
        gate_profile=gate_profile,
        replay_cache_status="memoire",
        replay_cache_key=replay_cache_key,
        replay_cache_entries=agent.replay_buffer.size,
        replay_cache_source="memoire",
        shadow_buffer_size=int(shadow_weighting.get("episodes_loaded", 0) or 0),
        dataset_coverage=dataset_coverage,
    )
    required_precheck_entries = max(int(getattr(config, "batch_size", 1) or 1) // 10, 1)
    precheck_family = infer_family_from_symbols(
        focus_symbols or list(config.symbols),
        family=initial_family,
    )
    precheck_feature_profile = resolve_feature_profile(horizon, precheck_family)
    precheck_runtime_state = {
        "failed_phase": None,
        "exception_type": None,
        "exception_message": None,
        "traceback_tail": [],
        "last_nonzero_exit": None,
        "stall_detected": False,
        "stall_reason": None,
    }
    append_training_log(
        (
            f"MuZero {horizon}: precheck optimisation sur replay "
            f"(buffer={agent.replay_buffer.size}, minimum={required_precheck_entries})."
        ),
        source="muzero",
    )
    mark_step_running(
        step_name,
        engine=engine,
        phase="optimisation_precheck",
        horizon=horizon,
        family=precheck_family,
        symbol_total=len(config.symbols),
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=(str(precheck_feature_profile.get("profile_name") or "").strip() or None),
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
        trial_mode=trial_mode,
        trial_cost_profile=trial_cost_profile,
        focus_symbols=focus_symbols,
        gate_profile=gate_profile,
        replay_cache_status="memoire",
        replay_cache_key=replay_cache_key,
        replay_cache_entries=agent.replay_buffer.size,
        replay_cache_source="memoire",
        shadow_buffer_size=int(shadow_weighting.get("episodes_loaded", 0) or 0),
        dataset_coverage=dataset_coverage,
    )
    set_training_runtime_state(
        train_step_phase="optimisation_precheck",
        phase_durations_ms={},
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=resume_step if resume_step > 0 else None,
        **precheck_runtime_state,
    )
    precheck_warmup = None
    if agent.replay_buffer.size < required_precheck_entries:
        precheck_warmup = _collect_precheck_warmup_games(
            agent=agent,
            config=config,
            required_entries=required_precheck_entries,
            max_wall_time_seconds=collection_timeout_seconds or None,
        )
        append_training_log(
            (
                f"MuZero {horizon}: warmup precheck effectue "
                f"({precheck_warmup.get('buffer_size')} episodes, "
                f"symboles={','.join(precheck_warmup.get('collected_symbols') or []) or 'aucun'})."
            ),
            source="muzero",
        )
    precheck_snapshot = _snapshot_agent_training_state(agent)
    precheck_phase_name = "optimisation_precheck"

    def _trace_precheck(phase_name: str) -> None:
        """Publie la sous-phase exacte du precheck d'optimisation MuZero."""

        nonlocal precheck_phase_name
        precheck_phase_name = phase_name
        set_training_runtime_state(
            train_step_phase=phase_name,
            failed_phase=None,
            exception_type=None,
            exception_message=None,
            traceback_tail=[],
            last_nonzero_exit=None,
            stall_detected=False,
            stall_reason=None,
        )

    _trace_precheck("optimisation_enter")
    try:
        precheck_result = agent.train_step(trace_hook=_trace_precheck)
        if precheck_result is None:
            raise RuntimeError(
                "OPTIMISATION_PRECHECK_FAILED: aucun batch exploitable n'a ete prepare pour le precheck."
            )
    except Exception as exc:
        terminal_summary_path = _record_muzero_failure(
            exc=exc,
            failed_phase=precheck_phase_name,
            run_id=str(load_training_status().get("run_id") or "").strip(),
            step_name=step_name,
            engine=engine,
            horizon=horizon,
            family=precheck_family,
            feature_profile=precheck_feature_profile,
            mechanics_profile_version=mechanics_profile_version,
            dataset_id=dataset_id or "",
            dataset_source=dataset_source or "",
            focus_symbols=focus_symbols,
            gate_profile=gate_profile,
            symbol_universe=focus_symbols or list(config.symbols),
            ga_trial=ga_trial,
            trial_mode=trial_mode,
            trial_cost_profile=trial_cost_profile,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_step=resume_step if resume_step > 0 else None,
        )
        logger.exception("Precheck optimisation MuZero en echec: %s", exc)
        append_training_log(
            f"MuZero {horizon}: echec du precheck optimisation, resume={terminal_summary_path.name}.",
            level="ERROR",
            source="muzero",
        )
        raise
    finally:
        _restore_agent_training_state(agent, precheck_snapshot)
    append_training_log(
        (
            f"MuZero {horizon}: precheck optimisation valide "
            f"(buffer={precheck_result.get('buffer_size')}, "
            f"durations={precheck_result.get('phase_durations_ms')})."
        ),
        source="muzero",
    )
    set_training_runtime_state(
        train_step_phase="optimisation_precheck_done",
        phase_durations_ms=dict(precheck_result.get("phase_durations_ms") or {}),
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=resume_step if resume_step > 0 else None,
        failed_phase=None,
        exception_type=None,
        exception_message=None,
        traceback_tail=[],
        last_nonzero_exit=None,
        stall_detected=False,
        stall_reason=None,
    )

    games_per_symbol = int(os.getenv("MUZERO_GAMES_PER_SYMBOL", "12"))
    if resume_step > 0:
        games_per_symbol = max(
            1,
            int(
                str(
                    os.getenv(
                        "MUZERO_RESUME_GAMES_PER_SYMBOL",
                        str(games_per_symbol),
                    )
                ).strip()
                or games_per_symbol
            ),
        )
    collection_mode = str(
        os.getenv(
            "MUZERO_RESUME_COLLECTION_MODE" if resume_step > 0 else "MUZERO_COLLECTION_MODE",
            "policy_only" if resume_step > 0 else "mcts",
        )
        or ("policy_only" if resume_step > 0 else "mcts")
    ).strip().lower()
    if collection_mode not in {"mcts", "policy_only"}:
        raise ValueError(f"Mode de collecte MuZero invalide: {collection_mode}")
    valid_symbols: list[str] = []
    total_games = 0
    collector_env_payloads: list[CollectorEnvironmentPayload] = []
    collector_profile: dict[str, object] = {
        "collector_mode": "sequential",
        "collector_workers": 1,
        "collector_queue_depth": int(getattr(config, "collector_queue_depth", 0) or 0),
        "collector_active_symbols": [],
        "inference_batch_profile": {
            "mode": "local_inline",
            "batch_max": 1,
            "batch_timeout_ms": 0,
            "total_requests": 0,
            "total_batches": 0,
        },
    }
    for symbol in config.symbols:
        env = build_environment(symbol, config)
        if env is None:
            continue
        valid_symbols.append(symbol)
        collector_env_payloads.append(
            CollectorEnvironmentPayload(
                symbol=symbol,
                market_data=np.asarray(env.data, dtype=np.float32),
                max_steps=int(env.max_steps_per_episode),
                dataset_source=str(
                    getattr(env, "dataset_source", None)
                    or getattr(config, "dataset_source", "csv")
                    or "csv"
                ),
            )
        )

    logger.info("Phase 1 - collecte historique par self-play guide")
    if resume_step > 0:
        logger.info(
            "Collecte MuZero de reprise en mode %s avec %s parties par symbole.",
            collection_mode,
            games_per_symbol,
        )
        append_training_log(
            (
                f"MuZero {horizon}: collecte de reprise en mode {collection_mode} "
                f"avec {games_per_symbol} parties par symbole."
            ),
            source="muzero",
        )

    if not valid_symbols:
        raise RuntimeError("Aucun symbole valide pour MuZero.")

    symbol_positions = {symbol: index + 1 for index, symbol in enumerate(valid_symbols)}
    collector_heartbeat_state = {
        "last_log_at": 0.0,
    }

    def _publish_collection_progress(payload: dict[str, object]) -> None:
        """Propage l'avancement de collecte vers le statut runtime."""

        active_symbols = [
            str(symbol).strip()
            for symbol in list(payload.get("collector_active_symbols") or [])
            if str(symbol).strip()
        ]
        live_inference_batch_profile = dict(
            payload.get("inference_batch_profile")
            or collector_profile.get("inference_batch_profile")
            or {}
        )
        collector_profile["collector_active_symbols"] = list(active_symbols)
        collector_profile["inference_batch_profile"] = dict(live_inference_batch_profile)
        set_training_runtime_state(
            collector_mode=str(collector_profile.get("collector_mode") or "sequential"),
            collector_workers=int(collector_profile.get("collector_workers") or 1),
            collector_active_symbols=active_symbols,
            collector_queue_depth=int(collector_profile.get("collector_queue_depth") or 0),
            inference_batch_profile=dict(live_inference_batch_profile),
            gpu_owner="muzero",
        )
        event_name = str(payload.get("event") or "").strip().lower()
        symbol = str(payload.get("symbol") or "").strip()
        if event_name == "symbol_start" and symbol:
            append_training_log(
                f"MuZero {horizon}: collecte sur {symbol} ({symbol_positions.get(symbol, 0)}/{len(valid_symbols)}).",
                source="muzero",
            )
            return
        if event_name == "symbol_done" and symbol:
            append_training_log(
                f"MuZero {horizon}: collecte terminee sur {symbol}.",
                source="muzero",
            )
            return
        if event_name == "collector_heartbeat":
            primary_symbol = active_symbols[0] if active_symbols else None
            mark_step_running(
                step_name,
                engine=engine,
                phase="collecte",
                horizon=horizon,
                family=initial_family,
                symbol=primary_symbol,
                symbol_index=symbol_positions.get(primary_symbol, 0) if primary_symbol else None,
                symbol_total=len(valid_symbols),
                dataset_id=dataset_id,
                dataset_source=dataset_source,
                feature_profile=feature_profile_name,
                mechanics_profile_version=mechanics_profile_version,
                ga_status=ga_status,
                ga_generation=ga_generation,
                ga_trial=ga_trial,
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
            now = time.perf_counter()
            if (now - float(collector_heartbeat_state["last_log_at"])) >= 60.0:
                append_training_log(
                    (
                        f"MuZero {horizon}: heartbeat collecte | actifs={active_symbols} "
                        f"| parties={int(payload.get('total_games') or 0)} "
                        f"| requetes_gpu={int(live_inference_batch_profile.get('total_requests') or 0)} "
                        f"| batchs_gpu={int(live_inference_batch_profile.get('total_batches') or 0)}"
                    ),
                    source="muzero",
                )
                collector_heartbeat_state["last_log_at"] = now
            return
        if event_name != "game_result" or not symbol:
            return
        mark_step_running(
            step_name,
            engine=engine,
            phase="collecte",
            horizon=horizon,
            family=initial_family,
            symbol=symbol,
            symbol_index=int(payload.get("symbol_index") or symbol_positions.get(symbol, 0)),
            symbol_total=len(valid_symbols),
            part_index=int(payload.get("part_index") or 0),
            part_total=games_per_symbol,
            dataset_id=dataset_id,
            dataset_source=dataset_source,
            feature_profile=feature_profile_name,
            mechanics_profile_version=mechanics_profile_version,
            ga_status=ga_status,
            ga_generation=ga_generation,
            ga_trial=ga_trial,
            trial_mode=trial_mode,
            trial_cost_profile=trial_cost_profile,
            focus_symbols=focus_symbols,
            gate_profile=gate_profile,
            replay_cache_status="warming",
            replay_cache_key=replay_cache_key,
            replay_cache_entries=int(payload.get("replay_entries") or agent.replay_buffer.size),
            replay_cache_source="memoire",
            dataset_coverage=dataset_coverage,
        )

    try:
        if str(getattr(config, "collector_mode", "") or "").strip().lower() == "batched_symbol_workers":
            collector_profile.update(
                {
                    "collector_mode": "batched_symbol_workers",
                    "collector_workers": min(
                        max(int(getattr(config, "collector_workers", 1) or 1), 1),
                        len(collector_env_payloads),
                    ),
                    "collector_queue_depth": int(getattr(config, "collector_queue_depth", 0) or 0),
                }
            )
            set_training_runtime_state(
                collector_mode=str(collector_profile.get("collector_mode") or "batched_symbol_workers"),
                collector_workers=int(collector_profile.get("collector_workers") or 1),
                collector_active_symbols=list(valid_symbols),
                collector_queue_depth=int(collector_profile.get("collector_queue_depth") or 0),
                inference_batch_profile=dict(collector_profile.get("inference_batch_profile") or {}),
                gpu_owner="muzero",
            )
            parallel_collection_result = collect_games_parallel(
                agent=agent,
                config=config,
                environments=collector_env_payloads,
                games_per_symbol=games_per_symbol,
                collection_mode=collection_mode,
                max_wall_time_seconds=collection_timeout_seconds or None,
                collector_workers=int(collector_profile.get("collector_workers") or 1),
                queue_depth=int(getattr(config, "collector_queue_depth", 0) or 0),
                inference_batch_max=int(getattr(config, "inference_batch_max", 64) or 64),
                inference_batch_timeout_ms=int(getattr(config, "inference_batch_timeout_ms", 2) or 2),
                progress_callback=_publish_collection_progress,
                log_callback=logger.info,
            )
            collector_profile.update(dict(parallel_collection_result or {}))
            total_games = int(collector_profile.get("total_games") or 0)
            valid_symbols = [
                str(symbol).strip()
                for symbol in list(collector_profile.get("valid_symbols") or valid_symbols)
                if str(symbol).strip()
            ]
        else:
            collector_profile.update(
                {
                    "collector_mode": "sequential",
                    "collector_workers": 1,
                }
            )
            set_training_runtime_state(
                collector_mode="sequential",
                collector_workers=1,
                collector_active_symbols=list(valid_symbols),
                collector_queue_depth=int(collector_profile.get("collector_queue_depth") or 0),
                inference_batch_profile=dict(collector_profile.get("inference_batch_profile") or {}),
                gpu_owner="muzero",
            )
            for symbol_index, env_payload in enumerate(collector_env_payloads, start=1):
                _publish_collection_progress(
                    {
                        "event": "symbol_start",
                        "symbol": env_payload.symbol,
                        "collector_active_symbols": [item.symbol for item in collector_env_payloads[symbol_index - 1 :]],
                    }
                )
                env = TradingEnvironment(
                    data=np.asarray(env_payload.market_data, dtype=np.float32),
                    symbol=env_payload.symbol,
                    config=config,
                    max_steps=int(env_payload.max_steps),
                )
                setattr(env, "dataset_source", env_payload.dataset_source)
                for game_index in range(games_per_symbol):
                    mark_step_running(
                        step_name,
                        engine=engine,
                        phase="collecte",
                        horizon=horizon,
                        family=initial_family,
                        symbol=env_payload.symbol,
                        symbol_index=symbol_index,
                        symbol_total=len(valid_symbols),
                        part_index=game_index + 1,
                        part_total=games_per_symbol,
                        dataset_id=dataset_id,
                        dataset_source=dataset_source,
                        feature_profile=feature_profile_name,
                        mechanics_profile_version=mechanics_profile_version,
                        ga_status=ga_status,
                        ga_generation=ga_generation,
                        ga_trial=ga_trial,
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
                    agent.play_game(
                        env,
                        exploration=True,
                        collection_mode=collection_mode,
                        max_wall_time_seconds=collection_timeout_seconds or None,
                    )
                    summary = env.get_summary()
                    total_games += 1
                    logger.info(
                        "[%s] %s partie %s/%s | return=%.2f%% | trades=%s | buffer=%s",
                        horizon,
                        env_payload.symbol,
                        game_index + 1,
                        games_per_symbol,
                        summary.get("return_pct", 0.0),
                        summary.get("total_trades", 0),
                        agent.replay_buffer.size,
                    )
                _publish_collection_progress(
                    {
                        "event": "symbol_done",
                        "symbol": env_payload.symbol,
                        "collector_active_symbols": [item.symbol for item in collector_env_payloads[symbol_index:]],
                    }
                )
    except Exception as exc:
        terminal_summary_path = _record_muzero_failure(
            exc=exc,
            failed_phase="collecte_parallel" if collector_profile.get("collector_mode") == "batched_symbol_workers" else "collecte",
            run_id=str(load_training_status().get("run_id") or "").strip(),
            step_name=step_name,
            engine=engine,
            horizon=horizon,
            family=initial_family,
            feature_profile=dict(getattr(config, "feature_profile", {}) or {}),
            mechanics_profile_version=mechanics_profile_version,
            dataset_id=dataset_id or "",
            dataset_source=dataset_source or "",
            focus_symbols=focus_symbols,
            gate_profile=gate_profile,
            symbol_universe=valid_symbols or list(config.symbols),
            ga_trial=ga_trial,
            trial_mode=trial_mode,
            trial_cost_profile=trial_cost_profile,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_step=resume_step if resume_step > 0 else None,
        )
        logger.exception("MuZero %s en erreur pendant la collecte: %s", horizon, exc)
        append_training_log(
            f"MuZero {horizon}: echec collecte, resume={terminal_summary_path.name}.",
            level="ERROR",
            source="muzero",
        )
        raise

    collector_profile["collector_active_symbols"] = []
    set_training_runtime_state(
        collector_mode=str(collector_profile.get("collector_mode") or "sequential"),
        collector_workers=int(collector_profile.get("collector_workers") or 1),
        collector_active_symbols=[],
        collector_queue_depth=int(collector_profile.get("collector_queue_depth") or 0),
        inference_batch_profile=dict(collector_profile.get("inference_batch_profile") or {}),
        gpu_owner="muzero",
    )
    append_training_log(
        (
            f"MuZero {horizon}: collecte terminee "
            f"({total_games} parties, mode={collector_profile.get('collector_mode')}, "
            f"workers={collector_profile.get('collector_workers')})."
        ),
        source="muzero",
    )

    family = infer_family_from_symbols(valid_symbols, family=getattr(config, "model_family", None))
    feature_profile = resolve_feature_profile(horizon, family)
    dataset_source = str(getattr(config, "dataset_source", "csv") or "csv")
    dataset_descriptor = dict(getattr(config, "dataset_descriptor", {}) or {})
    dataset_id = str(dataset_descriptor.get("dataset_id") or "")
    batch_autotune_result = _autotune_muzero_batch_size(
        agent=agent,
        config=config,
    )

    logger.info("Phase 2 - optimisation profonde (%s steps)", config.training_steps)
    start_time = datetime.now()
    last_metrics = None
    gold_precheck_payload: dict[str, object] | None = None
    start_optimisation_step = min(resume_step, config.training_steps)
    last_successful_step = start_optimisation_step
    executed_precheck_steps = {
        step for step in proxy_precheck_steps if start_optimisation_step >= step
    } if gold_precheck_enabled else set()
    killed_after_precheck = False
    if executed_precheck_steps:
        append_training_log(
            (
                f"MuZero {horizon}: precheck proxy saute pour les etapes "
                f"{sorted(executed_precheck_steps)} car la reprise commence apres ces seuils."
            ),
            source="muzero",
        )
    append_training_log(
        f"MuZero {horizon}: optimisation profonde sur {config.training_steps} steps.",
        source="muzero",
    )
    mark_step_running(
        step_name,
        engine=engine,
        phase="optimisation",
        horizon=horizon,
        family=family,
        symbol_total=len(valid_symbols),
        training_step_current=start_optimisation_step,
        training_step_total=config.training_steps,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
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
    set_training_runtime_state(
        last_successful_step=start_optimisation_step if start_optimisation_step > 0 else None,
        last_successful_step_at=datetime.now().isoformat(),
        train_step_phase="optimisation_enter",
        phase_durations_ms={},
        collector_mode=str(collector_profile.get("collector_mode") or "sequential"),
        collector_workers=int(collector_profile.get("collector_workers") or 1),
        collector_active_symbols=[],
        collector_queue_depth=int(collector_profile.get("collector_queue_depth") or 0),
        inference_batch_profile=dict(collector_profile.get("inference_batch_profile") or {}),
        jax_batch_profile=dict(batch_autotune_result or {}),
        gpu_owner="muzero",
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=start_optimisation_step if start_optimisation_step > 0 else None,
        failed_phase=None,
        exception_type=None,
        exception_message=None,
        traceback_tail=[],
        stall_detected=False,
        stall_reason=None,
        last_nonzero_exit=None,
    )

    current_failed_phase = "optimisation_enter"

    def _trace_train_step(phase_name: str) -> None:
        """Publie la sous-phase exacte du `train_step` MuZero."""

        nonlocal current_failed_phase
        current_failed_phase = phase_name
        set_training_runtime_state(
            train_step_phase=phase_name,
            failed_phase=None,
            exception_type=None,
            exception_message=None,
            traceback_tail=[],
            last_nonzero_exit=None,
            stall_detected=False,
            stall_reason=None,
        )

    for step in range(start_optimisation_step + 1, config.training_steps + 1):
        current_failed_phase = "optimisation_enter"
        try:
            step_result = agent.train_step(trace_hook=_trace_train_step)
        except Exception as exc:
            terminal_summary_path = _record_muzero_failure(
                exc=exc,
                failed_phase=current_failed_phase,
                run_id=str(load_training_status().get("run_id") or "").strip(),
                step_name=step_name,
                engine=engine,
                horizon=horizon,
                family=family,
                feature_profile=feature_profile,
                mechanics_profile_version=mechanics_profile_version,
                dataset_id=dataset_id,
                dataset_source=dataset_source,
                focus_symbols=focus_symbols,
                gate_profile=gate_profile,
                symbol_universe=valid_symbols,
                ga_trial=ga_trial,
                trial_mode=trial_mode,
                trial_cost_profile=trial_cost_profile,
                resume_checkpoint_path=resume_checkpoint_path,
                resume_step=start_optimisation_step if start_optimisation_step > 0 else None,
                last_metrics=last_metrics,
            )
            logger.exception("MuZero %s en erreur pendant %s: %s", horizon, current_failed_phase, exc)
            append_training_log(
                f"MuZero {horizon}: erreur optimisation, resume={terminal_summary_path.name}.",
                level="ERROR",
                source="muzero",
            )
            raise
        if step_result is None:
            logger.warning("MuZero sans batch suffisant, arret a l'etape %s.", step)
            append_training_log(
                f"MuZero {horizon}: arret anticipe a l'etape {step} faute de batch suffisant.",
                level="WARNING",
                source="muzero",
            )
            set_training_runtime_state(
                train_step_phase="waiting_for_batch",
                stall_detected=False,
                stall_reason=None,
            )
            break
        metrics = dict(step_result.get("metrics") or {})
        phase_durations_ms = dict(step_result.get("phase_durations_ms") or {})
        last_metrics = metrics
        last_successful_step = step
        step_completed_at = datetime.now().isoformat()
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
        set_training_runtime_state(
            last_successful_step=last_successful_step,
            last_successful_step_at=step_completed_at,
            train_step_phase="completed",
            phase_durations_ms=phase_durations_ms,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_step=start_optimisation_step if start_optimisation_step > 0 else None,
            failed_phase=None,
            exception_type=None,
            exception_message=None,
            traceback_tail=[],
            last_nonzero_exit=None,
            stall_detected=False,
            stall_reason=None,
        )

        next_precheck_step = next(
            (
                candidate_step
                for candidate_step in proxy_precheck_steps
                if candidate_step not in executed_precheck_steps and step >= candidate_step
            ),
            None,
        )
        if gold_precheck_enabled and next_precheck_step is not None:
            checkpoint_path = weights_dir / f"muzero_{horizon}_proxy_precheck_{next_precheck_step}.pkl"
            agent.save(
                str(checkpoint_path),
                step=step,
                run_id=active_run_id,
                trial_id=ga_trial,
                gate_profile=gate_profile,
                focus_symbols=focus_symbols,
            )
            append_training_log(
                (
                    f"MuZero {horizon}: lancement du precheck proxy a l'etape "
                    f"{next_precheck_step} sur {gold_precheck_games} segments."
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
                "step": next_precheck_step,
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
                "step": next_precheck_step,
                "eval_symbols": list(precheck_report.get("eval_symbols") or focus_symbols),
                "games": int(precheck_report.get("games_per_symbol") or gold_precheck_games),
                "metrics": precheck_metrics,
                "metrics_by_position_mechanics": precheck_mechanics,
                "reason": verdict.get("reason"),
                "failure_mode": verdict.get("failure_mode"),
                "early_kill_reason": verdict.get("reason") if verdict.get("status") == "fail" else None,
            }
            precheck_path = write_precheck_summary(gold_precheck_payload)
            gold_precheck_payload["path"] = str(precheck_path)
            set_gold_precheck(gold_precheck_payload)
            executed_precheck_steps.add(next_precheck_step)
            append_training_log(
                (
                    f"MuZero {horizon}: precheck proxy {gold_precheck_payload.get('status')} "
                    f"({gold_precheck_payload.get('reason')})."
                ),
                level="WARNING" if gold_precheck_payload.get("status") == "fail" else "INFO",
                source="muzero",
            )
            if gold_precheck_payload.get("status") == "fail":
                logger.warning(
                    "MuZero %s coupe apres precheck proxy a l'etape %s: %s",
                    horizon,
                    next_precheck_step,
                    gold_precheck_payload.get("reason"),
                )
                killed_after_precheck = True
                break

        if step % 50 == 0:
            elapsed = max((datetime.now() - start_time).total_seconds(), 1.0)
            logger.info(
                "[%s] step %05d/%05d | loss=%.4f | val=%.4f | rew=%.4f | pol=%.4f | %.2f steps/s",
                horizon,
                step,
                config.training_steps,
                float(metrics["loss_total"]),
                float(metrics["loss_val"]),
                float(metrics["loss_rew"]),
                float(metrics["loss_pol"]),
                step / elapsed,
            )
            append_training_log(
                "MuZero "
                f"{horizon}: step {step}/{config.training_steps} | "
                f"loss={float(metrics['loss_total']):.4f}",
                source="muzero",
            )

        if step % config.checkpoint_interval == 0:
            checkpoint_path = weights_dir / f"muzero_{horizon}_ckpt_{step}.pkl"
            agent.save(
                str(checkpoint_path),
                step=step,
                run_id=active_run_id,
                trial_id=ga_trial,
                gate_profile=gate_profile,
                focus_symbols=focus_symbols,
            )
            logger.info("Checkpoint MuZero sauvegarde: %s", checkpoint_path)

    final_checkpoint_step = last_successful_step if last_successful_step > 0 else start_optimisation_step
    agent.save(
        str(latest_path),
        step=final_checkpoint_step,
        run_id=active_run_id,
        trial_id=ga_trial,
        gate_profile=gate_profile,
        focus_symbols=focus_symbols,
    )
    logger.info("Checkpoint latest mis a jour: %s", latest_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    challenger_id = f"gen_{horizon}_{timestamp}"
    challenger_path = weights_dir / f"{challenger_id}.pkl"
    agent.save(
        str(challenger_path),
        step=final_checkpoint_step,
        run_id=active_run_id,
        trial_id=ga_trial,
        gate_profile=gate_profile,
        focus_symbols=focus_symbols,
    )
    active_run_id = str(load_training_status().get("run_id") or "").strip() or active_run_id

    if killed_after_precheck:
        precheck_metrics = dict((gold_precheck_payload or {}).get("metrics") or {})
        precheck_mechanics = dict((gold_precheck_payload or {}).get("metrics_by_position_mechanics") or {})
        promotion_result = {
            "status": "skipped",
            "reason": "proxy_precheck_fail",
            "promotion_gate": {
                "allowed": False,
                "status": "blocked",
                "reason": "proxy_precheck_fail",
                "gate_profile": gate_profile,
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
                "early_kill_reason": (gold_precheck_payload or {}).get("early_kill_reason"),
            },
        }
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
            "artifact_state": {
                "precheck_report_present": bool((gold_precheck_payload or {}).get("path")),
                "arena_report_present": False,
                "battle_report_present": False,
                "promotion_present": True,
                "candidate_checkpoint_present": challenger_path.exists(),
            },
            "resume_checkpoint_path": resume_checkpoint_path,
            "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
            "latest_candidate": challenger_id,
            "latest_verdict": {
                "status": "killed_after_precheck",
                "reason": (gold_precheck_payload or {}).get("reason"),
                "failure_mode": (gold_precheck_payload or {}).get("failure_mode"),
            },
            "gold_precheck": dict(gold_precheck_payload or {}),
            "precheck_status": (gold_precheck_payload or {}).get("status"),
            "early_kill_reason": (gold_precheck_payload or {}).get("early_kill_reason"),
            "collector_profile": dict(collector_profile or {}),
            "batch_autotune_result": dict(batch_autotune_result or {}),
        }
        terminal_summary_path = write_terminal_summary(terminal_summary)
        logger.info("Resume terminal MuZero ecrit dans %s", terminal_summary_path)
        append_training_log(
            (
                f"MuZero {horizon}: trial coupe apres precheck proxy "
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
            "latest_checkpoint": str(latest_path),
            "challenger_path": str(challenger_path),
            "training_metrics": last_metrics,
            "ga_status": ga_status,
            "ga_generation": ga_generation,
            "ga_trial": ga_trial,
            "trial_mode": trial_mode,
            "trial_cost_profile": trial_cost_profile,
            "resume_checkpoint_path": resume_checkpoint_path,
            "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
            "precheck": dict(gold_precheck_payload or {}),
            "early_kill_reason": (gold_precheck_payload or {}).get("early_kill_reason"),
            "promotion": promotion_result,
            "terminal_summary_path": str(terminal_summary_path),
            "collector_profile": dict(collector_profile or {}),
            "batch_autotune_result": dict(batch_autotune_result or {}),
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
        training_step_current=last_successful_step if last_metrics is not None else None,
        training_step_total=config.training_steps,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        feature_profile=(str(feature_profile.get("profile_name") or "").strip() or None),
        mechanics_profile_version=mechanics_profile_version,
        ga_status=ga_status,
        ga_generation=ga_generation,
        ga_trial=ga_trial,
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

    battle_report = arena.battle(challenger_id, champion_reference, horizon=horizon)
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
    promotion_result = promoter.promote_muzero_challenger(
        challenger_path=challenger_path,
        horizon=horizon,
        battle_report=battle_report,
        training_metrics=last_metrics,
        latest_checkpoint=latest_path,
        challenger_id=challenger_id,
        gate_profile=gate_profile,
    )
    logger.info("Promotion live %s: %s", horizon, promotion_result.get("status"))
    append_training_log(
        "MuZero "
        f"{horizon}: arena={battle_report.get('outcome')} | "
        f"promotion={promotion_result.get('status')} | "
        f"gate={promotion_result.get('reason') or promotion_result.get('promotion_gate', {}).get('reason') or 'aucun'}",
        source="muzero",
    )
    genetic.register_new_generation(
        gen_id=challenger_id,
        metrics=registry_metrics,
        is_champion=promotion_result.get("status") == "promoted",
        horizon=horizon,
    )
    champion_paths = promotion_result.get("champion_paths", [])

    report_path = results_dir / f"arena_{horizon}_latest.json"
    report_payload = {
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
        "latest_checkpoint": str(latest_path),
        "challenger_path": str(challenger_path),
        "live_champion_reference": champion_reference,
        "live_champion_id": live_champion_id or None,
        "champion_paths": champion_paths,
        "training_metrics": last_metrics,
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
        "ga_status": str(os.getenv("TRAINING_GA_STATUS", "")).strip() or None,
        "ga_generation": (
            int(os.getenv("TRAINING_GA_GENERATION", "0"))
            if str(os.getenv("TRAINING_GA_GENERATION", "")).strip()
            else None
        ),
        "ga_trial": str(os.getenv("TRAINING_GA_TRIAL", "")).strip() or None,
        "trial_mode": trial_mode,
        "trial_cost_profile": trial_cost_profile,
        "gold_precheck": dict(gold_precheck_payload or {}),
        "collector_profile": dict(collector_profile or {}),
        "batch_autotune_result": dict(batch_autotune_result or {}),
        "battle_report": battle_report,
        "promotion": promotion_result,
    }
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
        "ga_trial": ga_trial,
        "trial_mode": trial_mode,
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
        "artifact_state": {
            "arena_report_present": True,
            "battle_report_present": True,
            "promotion_present": bool(promotion_result),
            "candidate_checkpoint_present": challenger_path.exists(),
        },
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_step": start_optimisation_step if start_optimisation_step > 0 else None,
        "latest_candidate": challenger_id,
        "latest_verdict": {
            "status": promotion_result.get("status"),
            "reason": promotion_result.get("reason") or promotion_gate.get("reason"),
            "failure_mode": promotion_gate.get("failure_mode"),
        },
        "gold_precheck": dict(gold_precheck_payload or {}),
        "precheck_status": (gold_precheck_payload or {}).get("status"),
        "collector_profile": dict(collector_profile or {}),
        "batch_autotune_result": dict(batch_autotune_result or {}),
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
    try:
        summary = main()
        logger.info("MuZero termine: %s", summary)
    except Exception as exc:  # pragma: no cover - diagnostic operateur
        logger.exception("MuZero en erreur terminale: %s", exc)
        raise

