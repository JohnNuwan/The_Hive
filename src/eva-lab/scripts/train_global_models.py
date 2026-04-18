"""Entraine MuZero sur historique reel puis execute la selection ADN."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
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

    if evaluation_games <= 0 or total_trades <= 0:
        return {
            "status": "fail",
            "reason": "aucun_trade_exploitable",
            "failure_mode": "inactive",
        }

    if directional_bias in {"buy_heavy", "sell_heavy"} and return_pct <= 0.0 and profit_factor < 0.95:
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
    reanalyze_games_total = 0
    reanalyze_positions_total = 0
    gold_precheck_payload: dict[str, object] | None = None
    gold_precheck_executed = False
    killed_after_precheck = False

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
                agent.play_game(env, exploration=True)
                summary = env.get_summary()
                total_games += 1
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
                merge_training_status(
                    {
                        "latest_metrics": dict(last_metrics),
                        "train_step_phase": "optimisation",
                        "phase_durations_ms": phase_durations_ms,
                    }
                )

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
    if ga_defer_promotion:
        promotion_gate = promoter.evaluate_promotion_gate(
            battle_report,
            gate_profile="standard",
        )
        live_comparison = promoter._compare_with_live_champion(
            horizon=horizon,
            candidate_gate=promotion_gate,
            engine=engine,
            challenger_id=challenger_id,
        )
        promotion_result = {
            "status": "candidate_only",
            "reason": "deferred_ga_selection",
            "engine": engine,
            "horizon": horizon,
            "source_path": str(challenger_path),
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
            "lineage": challenger_lineage,
            "seed_parent_champion_id": ga_parent_champion_id,
        }
    else:
        promotion_result = promoter.promote_muzero_challenger(
            challenger_path=challenger_path,
            horizon=horizon,
            battle_report=battle_report,
            training_metrics=training_metrics_payload,
            latest_checkpoint=(Path(ga_seed_checkpoint_path) if mechanics_only_mode and ga_seed_checkpoint_path else latest_path),
            challenger_id=challenger_id,
            gate_profile=gate_profile,
            promotion_metadata={
                "resume_source": resume_source,
                "lineage": challenger_lineage,
                "seed_parent_champion_id": ga_parent_champion_id,
                "ga_campaign_id": ga_campaign_id,
                "ga_trial": ga_trial,
                "ga_scope": ga_scope,
            },
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
    if promotion_result.get("status") != "promoted":
        promoter.persist_challenger_manifest(
            engine=engine,
            horizon=horizon,
            status=(
                "candidate_only"
                if promotion_result.get("status") == "candidate_only"
                else "blocked"
            ),
            challenger_id=challenger_id,
            challenger_path=str(challenger_path),
            latest_checkpoint=latest_checkpoint_reference,
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
            lineage=challenger_lineage,
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
        "latest_checkpoint": latest_checkpoint_reference,
        "challenger_path": str(challenger_path),
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
        "lineage": challenger_lineage,
        "gold_precheck": dict(gold_precheck_payload or {}),
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
        "challenger_path": str(challenger_path),
        "latest_checkpoint": latest_checkpoint_reference,
        "battle_report_path": str(unique_report_path),
        "live_comparison": dict(promotion_result.get("live_comparison") or {}),
        "resume_source": resume_source,
        "artifact_compatibility": dict(promotion_result.get("artifact_compatibility") or challenger_compatibility),
        "checkpoint_schema_version": (
            promotion_result.get("checkpoint_schema_version")
            or challenger_checkpoint_schema_version
        ),
        "lineage": challenger_lineage,
        "artifact_state": {
            "arena_report_present": True,
            "battle_report_present": True,
            "promotion_present": bool(promotion_result),
            "candidate_checkpoint_present": challenger_path.exists(),
            "latest_checkpoint_present": latest_checkpoint_present,
        },
        "latest_candidate": challenger_id,
        "latest_verdict": {
            "status": promotion_result.get("status"),
            "reason": promotion_result.get("reason") or promotion_gate.get("reason"),
            "failure_mode": promotion_gate.get("failure_mode"),
        },
        "gold_precheck": dict(gold_precheck_payload or {}),
        "precheck_status": (gold_precheck_payload or {}).get("status"),
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

