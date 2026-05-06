"""Notifications Telegram pour les entrainements EVA Lab."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.training_status import load_training_status
from shared.telegram_client import TelegramClient

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """Lit un booleen depuis l'environnement.

    Args:
        name (str): Nom de la variable d'environnement.
        default (bool): Valeur de repli.

    Returns:
        bool: Valeur booleenne normalisee.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _format_metric(value: Any, suffix: str = "") -> str:
    """Formate une metrique numerique pour Telegram.

    Args:
        value (Any): Valeur brute.
        suffix (str): Suffixe optionnel.

    Returns:
        str: Chaine lisible.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"n/a{suffix}"
    return f"{numeric:.2f}{suffix}"


def _format_share_metric(value: Any) -> str:
    """Formate une part ou un ratio en pourcentage lisible.

    Args:
        value (Any): Valeur brute, generalement comprise entre 0 et 1.

    Returns:
        str: Pourcentage lisible.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a%"
    if abs(numeric) <= 1.0:
        numeric *= 100.0
    return f"{numeric:.2f}%"


def _format_ratio(current: Any, total: Any) -> str:
    """Formate un ratio courant/total.

    Args:
        current (Any): Valeur courante.
        total (Any): Valeur maximale.

    Returns:
        str: Ratio lisible ou ``n/a``.
    """
    try:
        left = int(float(current))
        right = int(float(total))
    except (TypeError, ValueError):
        return "n/a"
    if right <= 0:
        return str(left)
    return f"{left}/{right}"


def _format_timestamp(value: Any) -> str:
    """Formate un horodatage ISO pour Telegram.

    Args:
        value (Any): Valeur brute a convertir.

    Returns:
        str: Horodatage lisible.
    """
    raw_value = str(value or "").strip()
    if not raw_value:
        return "n/a"
    try:
        timestamp = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return raw_value
    return timestamp.strftime("%d/%m/%Y %H:%M")


def _parse_iso_datetime(value: Any) -> datetime | None:
    """Convertit un horodatage ISO en objet ``datetime``.

    Args:
        value (Any): Valeur brute a convertir.

    Returns:
        datetime | None: Date convertie ou ``None`` si invalide.
    """
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _env_int(name: str, default: int) -> int:
    """Lit un entier depuis l'environnement avec repli robuste.

    Args:
        name (str): Nom de la variable d'environnement.
        default (int): Valeur de repli.

    Returns:
        int: Valeur entiere normalisee.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Valeur entiere invalide pour %s=%s. Repli=%s.", name, raw_value, default)
        return default


def _resolve_digest_horizons(horizons: list[str] | None = None) -> list[str]:
    """Retourne les horizons a afficher dans le digest.

    Args:
        horizons (list[str] | None): Horizons explicites optionnels.

    Returns:
        list[str]: Horizons normalises et dedupliques.
    """
    if horizons:
        raw_items = list(horizons)
    else:
        raw_items = os.getenv("MUZERO_HORIZONS", "scalp,intraday,swing").split(",")

    resolved: list[str] = []
    for item in raw_items:
        horizon = str(item or "").strip().lower()
        if horizon and horizon not in resolved:
            resolved.append(horizon)
    return resolved or ["scalp"]


def _humanize_token(value: Any, default: str = "n/a") -> str:
    """Normalise un identifiant technique en texte lisible.

    Args:
        value (Any): Valeur brute a humaniser.
        default (str): Valeur de repli.

    Returns:
        str: Texte court plus lisible.
    """
    text = str(value or "").strip()
    if not text:
        return default
    return text.replace("_", " ")


def _shorten_identifier(value: Any, default: str = "aucun", max_length: int = 44) -> str:
    """Raccourcit un identifiant ou chemin pour Telegram.

    Args:
        value (Any): Identifiant brut.
        default (str): Valeur de repli si vide.
        max_length (int): Longueur maximale affichee.

    Returns:
        str: Identifiant compact et lisible.
    """
    raw_value = str(value or "").strip()
    if not raw_value:
        return default
    candidate = raw_value
    if "/" in raw_value or "\\" in raw_value:
        candidate = Path(raw_value).name
    if candidate.lower().endswith(".pkl"):
        candidate = candidate[:-4]
    candidate = candidate.replace("_", "-")
    if len(candidate) <= max_length:
        return candidate
    head = max_length // 2 - 2
    tail = max_length - head - 3
    return f"{candidate[:head]}...{candidate[-tail:]}"


def _resolve_digest_state_path() -> Path:
    """Retourne le chemin de persistance du digest training.

    Returns:
        Path: Fichier de cache local au digest.
    """
    raw_value = str(os.getenv("TRAINING_DIGEST_STATE_PATH", "")).strip()
    if raw_value:
        return Path(raw_value)
    return Path.cwd() / "data" / "checkpoints" / "training_digest_state.json"


def _bucket_metric(value: Any, quantum: float) -> float | None:
    """Normalise une metrique dans un seau grossier.

    Args:
        value (Any): Valeur brute a bucketiser.
        quantum (float): Largeur du seau.

    Returns:
        float | None: Valeur bucketisee ou ``None`` si invalide.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if quantum <= 0:
        return round(numeric, 6)
    return round(round(numeric / quantum) * quantum, 6)


def _build_digest_change_snapshot(
    *,
    run_status: dict[str, Any],
    horizon_statuses: dict[str, dict[str, Any]] | None,
    horizons: list[str] | None,
) -> dict[str, Any]:
    """Construit une vue semantique pour detecter un vrai changement.

    Le digest Telegram contient volontairement un horodatage courant et de
    nombreuses metriques fines. Cette signature ne retient que les transitions
    materielles afin d'eviter les messages repetitifs.

    Args:
        run_status (dict[str, Any]): Statut courant du training.
        horizon_statuses (dict[str, dict[str, Any]] | None): Statuts par horizon.
        horizons (list[str] | None): Horizons affiches.

    Returns:
        dict[str, Any]: Signature semantique compacte.
    """
    current_step = dict(run_status.get("current_step") or {})
    latest_metrics = dict(run_status.get("latest_metrics") or {})
    arena_progress = dict(run_status.get("arena_progress") or {})
    family_probe_status = dict(run_status.get("family_probe_status") or {})
    resolved_horizons = _resolve_digest_horizons(horizons)

    phase = str(current_step.get("phase") or "").strip().lower()
    optimisation_step = current_step.get("training_step_current")
    optimisation_bucket = None
    if optimisation_step is not None:
        optimisation_bucket_size = max(_env_int("TELEGRAM_TRAINING_DIGEST_STEP_BUCKET", 500), 1)
        try:
            optimisation_bucket = (
                int(float(optimisation_step)) // optimisation_bucket_size
            ) * optimisation_bucket_size
        except (TypeError, ValueError):
            optimisation_bucket = None

    horizon_snapshot: dict[str, dict[str, Any]] = {}
    for horizon in resolved_horizons:
        payload = dict((horizon_statuses or {}).get(horizon) or {})
        metrics = dict(payload.get("candidate_metrics") or {})
        mechanics = dict(metrics.get("metrics_by_position_mechanics") or {})
        horizon_snapshot[horizon] = {
            "selection": payload.get("selection"),
            "live": payload.get("live_champion_id"),
            "candidate": payload.get("candidate_id"),
            "gate_reason": payload.get("gate_reason"),
            "profit_factor": _bucket_metric(metrics.get("profit_factor"), 0.05),
            "return_pct": _bucket_metric(metrics.get("return_pct"), 0.05),
            "split_runner_capture_rate": _bucket_metric(
                mechanics.get("split_runner_capture_rate"), 0.05
            ),
            "pyramid_exit_capture_rate": _bucket_metric(
                mechanics.get("pyramid_exit_capture_rate"), 0.05
            ),
        }

    return {
        "run_id": run_status.get("run_id"),
        "status": run_status.get("status"),
        "strategy": run_status.get("strategy"),
        "reason": run_status.get("reason"),
        "resume_source": run_status.get("resume_source"),
        "current_step": {
            "name": current_step.get("name"),
            "phase": phase,
            "horizon": current_step.get("horizon"),
            "symbol": current_step.get("symbol"),
            "symbol_index": current_step.get("symbol_index"),
            "symbol_total": current_step.get("symbol_total"),
            "part_index": current_step.get("part_index"),
            "part_total": current_step.get("part_total"),
            "optimisation_bucket": optimisation_bucket,
            "train_step_phase": run_status.get("train_step_phase"),
        },
        "seed": {
            "status": latest_metrics.get("seed_viability_status"),
            "reason": latest_metrics.get("seed_viability_reason"),
            "recommended": latest_metrics.get("recommended_seed_for_v66"),
        },
        "metrics": {
            "loss_pol": _bucket_metric(latest_metrics.get("loss_pol"), 0.25),
            "loss_pol_per_head": _bucket_metric(latest_metrics.get("loss_pol_per_head"), 0.05),
            "root_mask_rate": _bucket_metric(latest_metrics.get("root_mask_rate"), 0.01),
            "split_runner_capture_rate": _bucket_metric(
                latest_metrics.get("split_runner_capture_rate"), 0.05
            ),
            "pyramid_exit_capture_rate": _bucket_metric(
                latest_metrics.get("pyramid_exit_capture_rate"), 0.05
            ),
            "close_quality_score": _bucket_metric(
                latest_metrics.get("close_quality_score"), 0.05
            ),
            "runner_giveback_pct": _bucket_metric(
                latest_metrics.get("runner_giveback_pct"), 0.05
            ),
        },
        "precheck_status": run_status.get("precheck_status"),
        "family_probe_status": {
            "reason": family_probe_status.get("reason"),
            "ready_families": family_probe_status.get("ready_families"),
            "positive_families": family_probe_status.get("positive_families"),
        },
        "arena_progress": {
            "current_role": arena_progress.get("current_role"),
            "current_symbol": arena_progress.get("current_symbol"),
            "symbol_index": arena_progress.get("symbol_index"),
            "symbol_total": arena_progress.get("symbol_total"),
            "challenger_score": _bucket_metric(arena_progress.get("challenger_score"), 0.5),
            "champion_score": _bucket_metric(arena_progress.get("champion_score"), 0.5),
        },
        "horizons": horizon_snapshot,
    }


def _build_digest_signature(
    *,
    run_status: dict[str, Any],
    horizon_statuses: dict[str, dict[str, Any]] | None,
    horizons: list[str] | None,
) -> tuple[str, dict[str, Any]]:
    """Construit la signature de changement du digest.

    Args:
        run_status (dict[str, Any]): Statut courant du training.
        horizon_statuses (dict[str, dict[str, Any]] | None): Statuts par horizon.
        horizons (list[str] | None): Horizons a afficher.

    Returns:
        tuple[str, dict[str, Any]]: Signature SHA-256 et snapshot semantique.
    """
    snapshot = _build_digest_change_snapshot(
        run_status=run_status,
        horizon_statuses=horizon_statuses,
        horizons=horizons,
    )
    serialized = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), snapshot


def _load_digest_state() -> dict[str, Any]:
    """Charge l'etat du dernier digest envoye.

    Returns:
        dict[str, Any]: Etat persiste ou dictionnaire vide.
    """
    path = _resolve_digest_state_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Lecture du cache digest impossible pour %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_digest_state(payload: dict[str, Any]) -> None:
    """Persiste l'etat du dernier digest envoye.

    Args:
        payload (dict[str, Any]): Etat a persister.
    """
    path = _resolve_digest_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _should_send_digest(
    *,
    signature: str,
    run_status: dict[str, Any],
) -> tuple[bool, str]:
    """Determine si le digest Telegram doit etre envoye.

    Args:
        signature (str): Signature semantique courante.
        run_status (dict[str, Any]): Statut courant du training.

    Returns:
        tuple[bool, str]: Decision d'envoi et raison associee.
    """
    if not _env_flag("TELEGRAM_TRAINING_DIGEST_ONLY_ON_CHANGE", True):
        return True, "always_send"

    state = _load_digest_state()
    previous_signature = str(state.get("signature") or "").strip()
    force_after_minutes = max(_env_int("TELEGRAM_TRAINING_DIGEST_FORCE_AFTER_MINUTES", 180), 1)
    now = datetime.now()
    last_sent_at = _parse_iso_datetime(state.get("last_sent_at"))
    minutes_since_last_send = None
    if last_sent_at is not None:
        minutes_since_last_send = max((now - last_sent_at).total_seconds() / 60.0, 0.0)

    if not previous_signature:
        return True, "first_digest"
    if previous_signature != signature:
        return True, "material_change"
    if minutes_since_last_send is None:
        return True, "missing_last_sent_at"
    if minutes_since_last_send >= force_after_minutes:
        return True, "forced_refresh"

    run_id = str(run_status.get("run_id") or "").strip() or "n/a"
    logger.info(
        "Digest training Telegram ignore: aucun changement materiel pour %s (%.1f min).",
        run_id,
        minutes_since_last_send,
    )
    return False, "unchanged"


def build_training_digest_message(
    *,
    run_status: dict[str, Any] | None = None,
    horizon_statuses: dict[str, dict[str, Any]] | None = None,
    horizons: list[str] | None = None,
) -> str:
    """Construit un digest Telegram compact de l'etat training.

    Args:
        run_status (dict[str, Any] | None): Statut courant explicite du run.
        horizon_statuses (dict[str, dict[str, Any]] | None): Statuts champion
            deja resolves par horizon.
        horizons (list[str] | None): Horizons a inclure si les statuts doivent
            etre resolves automatiquement.

    Returns:
        str: Message pret pour Telegram.
    """
    status = dict(run_status or load_training_status() or {})
    current_step = dict(status.get("current_step") or {})
    latest_metrics = dict(status.get("latest_metrics") or {})
    arena_progress = dict(status.get("arena_progress") or {})
    policy_precheck_payload = dict(status.get("policy_precheck") or {})
    family_probe_status = dict(status.get("family_probe_status") or {})
    resolved_horizons = _resolve_digest_horizons(horizons)

    if horizon_statuses is None:
        promoter = ChampionPromoter()
        resolved_horizon_statuses: dict[str, dict[str, Any]] = {}
        for horizon in resolved_horizons:
            try:
                resolved_horizon_statuses[horizon] = promoter.build_horizon_status(horizon)
            except Exception as exc:
                logger.warning("Lecture du statut champion impossible pour %s: %s", horizon, exc)
                resolved_horizon_statuses[horizon] = {"horizon": horizon, "gate_reason": "lecture_impossible"}
        horizon_statuses = resolved_horizon_statuses

    primary_horizon = str(current_step.get("horizon") or "").strip().lower()
    if not primary_horizon or primary_horizon not in resolved_horizons:
        primary_horizon = resolved_horizons[0]

    lines = [
        "POINT ENTRAINEMENT",
        _format_timestamp(datetime.now().isoformat()),
        "",
        "RUN",
        f"- id: {_shorten_identifier(status.get('run_id'), default='aucun', max_length=56)}",
        (
            f"- statut: {_humanize_token(status.get('status'), default='inconnu')} | "
            f"strategie: {_humanize_token(status.get('strategy'))} | "
            f"reprise: {_humanize_token(status.get('resume_source'))}"
        ),
        f"- raison: {_humanize_token(status.get('reason'))}",
    ]

    if current_step:
        lines.extend(
            [
                "",
                "ACTIF",
                (
                    f"- etape: {_humanize_token(current_step.get('name'))} | "
                    f"phase: {_humanize_token(current_step.get('phase'))} | "
                    f"horizon: {_humanize_token(current_step.get('horizon'))}"
                ),
                (
                    f"- progression: {current_step.get('symbol') or 'n/a'} "
                    f"({_format_ratio(current_step.get('symbol_index'), current_step.get('symbol_total'))}) | "
                    f"partie {_format_ratio(current_step.get('part_index'), current_step.get('part_total'))} | "
                    f"episode {_format_ratio(current_step.get('episode_step_current'), current_step.get('episode_step_total'))}"
                ),
                f"- replay: {_format_ratio(status.get('replay_cache_entries'), 196)}",
                f"- maj: {_format_timestamp(current_step.get('updated_at') or status.get('updated_at'))}",
            ]
        )

    if latest_metrics:
        lines.extend(
            [
                "",
                "METRIQUES CLES",
                (
                    f"- policy: total={_format_metric(latest_metrics.get('loss_pol'))} | "
                    f"par_tete={_format_metric(latest_metrics.get('loss_pol_per_head'))} | "
                    f"top1={_format_share_metric(latest_metrics.get('policy_top1_share'))} | "
                    f"entropy={_format_metric(latest_metrics.get('policy_entropy'))}"
                ),
                (
                    f"- defensif: root_mask={_format_share_metric(latest_metrics.get('root_mask_rate'))} | "
                    f"close_q={_format_metric(latest_metrics.get('close_quality_score'))} | "
                    f"slbe={_format_metric(latest_metrics.get('slbe_capture_rate'))} | "
                    f"hold_drag={_format_metric(latest_metrics.get('hold_drag_score'))}"
                ),
                (
                    f"- offensif: split_cap={_format_metric(latest_metrics.get('split_runner_capture_rate'))} | "
                    f"runner_win={_format_metric(latest_metrics.get('runner_profit_hold_window_count'))} | "
                    f"pyramid_cap={_format_metric(latest_metrics.get('pyramid_exit_capture_rate'))} | "
                    f"peak_giveback={_format_metric(latest_metrics.get('profit_peak_giveback_ratio'))}"
                ),
                (
                    f"- fenetres: split={_format_metric(latest_metrics.get('split_monetization_window_count'))} | "
                    f"runner={_format_metric(latest_metrics.get('runner_profit_hold_window_count'))} | "
                    f"pyramid={_format_metric(latest_metrics.get('pyramid_monetization_window_count'))}"
                ),
                (
                    f"- seed: etage={_humanize_token(latest_metrics.get('seed_stage'))} | "
                    f"statut={_humanize_token(latest_metrics.get('seed_viability_status'))} | "
                    f"raison={_humanize_token(latest_metrics.get('seed_viability_reason'))}"
                ),
                f"- seed_reco: {_shorten_identifier(latest_metrics.get('recommended_seed_for_v66'))}",
            ]
        )

    if policy_precheck_payload:
        trends = dict(policy_precheck_payload.get("trends") or {})
        lines.extend(
            [
                "",
                "PRECHECK POLICY",
                (
                    f"- statut: {_humanize_token(policy_precheck_payload.get('status'))} | "
                    f"raison: {_humanize_token(policy_precheck_payload.get('reason'))}"
                ),
                (
                    f"- tendances: loss={_format_metric(trends.get('loss_pol_trend'))} | "
                    f"root_mask={_format_metric(trends.get('root_mask_rate_trend'))} | "
                    f"split_runner={_format_metric(trends.get('split_runner_capture_trend'))} | "
                    f"pyramid_exit={_format_metric(trends.get('pyramid_exit_capture_trend'))}"
                ),
            ]
        )

    if family_probe_status:
        lines.extend(
            [
                "",
                "FAMILY PROBES",
                (
                    f"- statut: {_humanize_token(family_probe_status.get('reason'))} | "
                    f"pretes: {_format_metric(family_probe_status.get('ready_families'))}/"
                    f"{_format_metric(family_probe_status.get('required_ready_families'))} | "
                    f"positives: {_format_metric(family_probe_status.get('positive_families'))}/"
                    f"{_format_metric(family_probe_status.get('required_positive_families'))}"
                ),
            ]
        )

    if arena_progress:
        lines.extend(
            [
                "",
                "ARENA",
                (
                    f"- statut: {_humanize_token(arena_progress.get('status'), default='running')} | "
                    f"role: {_humanize_token(arena_progress.get('current_role'))}"
                ),
                (
                    f"- progression: {arena_progress.get('current_symbol') or 'n/a'} "
                    f"({_format_ratio(arena_progress.get('symbol_index'), arena_progress.get('symbol_total'))}) | "
                    f"challenger={_format_metric(arena_progress.get('challenger_score'))} | "
                    f"champion={_format_metric(arena_progress.get('champion_score'))}"
                ),
            ]
        )

    primary_payload = dict((horizon_statuses or {}).get(primary_horizon) or {})
    primary_metrics = dict(primary_payload.get("candidate_metrics") or {})
    primary_mechanics = dict(primary_metrics.get("metrics_by_position_mechanics") or {})
    lines.extend(
        [
            "",
            f"MUZERO {primary_horizon.upper()}",
            (
                f"- selection: {_humanize_token(primary_payload.get('selection'))} | "
                f"live: {_shorten_identifier(primary_payload.get('live_champion_id'))}"
            ),
            (
                f"- candidat: {_shorten_identifier(primary_payload.get('candidate_id'))} | "
                f"gate: {_humanize_token(primary_payload.get('gate_reason'))}"
            ),
            (
                f"- perf: PF={_format_metric(primary_metrics.get('profit_factor'))} | "
                f"Ret={_format_metric(primary_metrics.get('return_pct'), '%')} | "
                f"WR={_format_metric(primary_metrics.get('win_rate'), '%')}"
            ),
            (
                f"- meca: close_q={_format_metric(primary_mechanics.get('close_quality_score'))} | "
                f"split_runner={_format_metric(primary_mechanics.get('split_runner_capture_rate'))} | "
                f"pyramid_exit={_format_metric(primary_mechanics.get('pyramid_exit_capture_rate'))} | "
                f"slbe={_format_metric(primary_mechanics.get('slbe_capture_rate'))}"
            ),
        ]
    )

    secondary_lines: list[str] = []
    for horizon in resolved_horizons:
        if horizon == primary_horizon:
            continue
        payload = dict((horizon_statuses or {}).get(horizon) or {})
        selection = str(payload.get("selection") or "").strip().lower()
        live_id = payload.get("live_champion_id")
        candidate_id = payload.get("candidate_id")
        if not selection and not live_id and not candidate_id:
            continue
        secondary_lines.append(
            f"- {horizon}: {_humanize_token(payload.get('selection'))} | "
            f"live={_shorten_identifier(live_id)} | gate={_humanize_token(payload.get('gate_reason'))}"
        )
    if secondary_lines:
        lines.extend(["", "AUTRES HORIZONS", *secondary_lines])

    return "\n".join(lines)


def send_training_digest(
    *,
    run_status: dict[str, Any] | None = None,
    horizon_statuses: dict[str, dict[str, Any]] | None = None,
    horizons: list[str] | None = None,
) -> bool:
    """Envoie un digest Telegram compact de l'etat training.

    Args:
        run_status (dict[str, Any] | None): Statut courant explicite du run.
        horizon_statuses (dict[str, dict[str, Any]] | None): Statuts champion
            deja resolves par horizon.
        horizons (list[str] | None): Horizons a afficher.

    Returns:
        bool: ``True`` si un digest a ete envoye, sinon ``False``.
    """
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING", True):
        return False
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING_DIGEST", True):
        return False

    status = dict(run_status or load_training_status() or {})
    message = build_training_digest_message(
        run_status=status,
        horizon_statuses=horizon_statuses,
        horizons=horizons,
    )
    signature, snapshot = _build_digest_signature(
        run_status=status,
        horizon_statuses=horizon_statuses,
        horizons=horizons,
    )
    should_send, reason = _should_send_digest(signature=signature, run_status=status)
    if not should_send:
        return False

    try:
        TelegramClient().send_sync(message)
        _save_digest_state(
            {
                "last_sent_at": datetime.now().isoformat(),
                "signature": signature,
                "reason": reason,
                "run_id": status.get("run_id"),
                "snapshot": snapshot,
            }
        )
        return True
    except Exception as exc:
        logger.warning("Notification Telegram de digest training ignoree: %s", exc)
        return False


def send_training_run_started(
    *,
    run_id: str,
    strategy: str,
    reason: str,
    trigger: str,
    universe: dict[str, Any] | None = None,
) -> None:
    """Diffuse le demarrage d'un run nightly.

    Args:
        run_id (str): Identifiant du run.
        strategy (str): Strategie retenue (`skip`, `refresh`, `research`).
        reason (str): Raison de la strategie.
        trigger (str): Origine du lancement.
        universe (dict[str, Any] | None): Resume d'univers optionnel.
    """
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING", True):
        return

    family_counts = dict((universe or {}).get("family_counts") or {})
    lines = [
        "DEMARRAGE ENTRAINEMENT",
        f"Run: {run_id}",
        f"Strategie: {strategy}",
        f"Raison: {reason}",
        f"Declencheur: {trigger}",
    ]
    if family_counts:
        lines.extend(
            [
                "",
                "Univers:",
                f"- Forex: {family_counts.get('forex', 0)}",
                f"- Crypto: {family_counts.get('crypto', 0)}",
                f"- CFD index: {family_counts.get('index_cfd', 0)}",
                f"- Metaux: {family_counts.get('metal', 0)}",
            ]
        )

    try:
        TelegramClient().send_sync("\n".join(lines))
    except Exception as exc:
        logger.warning("Notification Telegram de demarrage ignoree: %s", exc)


def send_training_horizon_started(horizon: str, symbol_total: int) -> None:
    """Diffuse le debut d'un horizon MuZero.

    Args:
        horizon (str): Horizon strategique courant.
        symbol_total (int): Nombre de symboles prevus.
    """
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING", True):
        return

    try:
        TelegramClient().send_sync(
            "\n".join(
                [
                    "PHASE MUZERO",
                    f"Horizon: {horizon}",
                    f"Symboles planifies: {symbol_total}",
                ]
            )
        )
    except Exception as exc:
        logger.warning("Notification Telegram d'horizon ignoree: %s", exc)


def send_horizon_summary(
    horizon: str,
    report_payload: dict[str, Any],
    promotion_result: dict[str, Any],
) -> None:
    """Envoie un resume Telegram d'un candidat MuZero.

    Args:
        horizon (str): Horizon strategique traite.
        report_payload (dict[str, Any]): Rapport complet du run.
        promotion_result (dict[str, Any]): Verdict de promotion live.
    """
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING", True):
        return

    battle_report = report_payload.get("battle_report", {}) or {}
    challenger = battle_report.get("challenger", {}) or {}
    metrics = challenger.get("metrics", {}) or {}
    mechanics = dict(metrics.get("metrics_by_position_mechanics") or {})
    validation = battle_report.get("validation", {}) or {}
    promotion_gate = promotion_result.get("promotion_gate", {}) or {}
    live_universe = report_payload.get("symbols", []) or []
    live_status = ChampionPromoter().build_horizon_status(horizon)
    candidate_id = (
        challenger.get("id")
        or promotion_result.get("challenger_id")
        or live_status.get("candidate_id")
        or "inconnu"
    )
    live_champion_id = live_status.get("live_champion_id") or "aucun"
    selection = str(live_status.get("selection") or promotion_result.get("status") or "unknown")

    lines = [
        "RAPPORT D'ENTRAINEMENT",
        f"Horizon: {horizon}",
        f"Champion candidat: {candidate_id}",
        f"Champion live: {live_champion_id}",
        f"Arena: {battle_report.get('outcome', 'UNKNOWN')}",
        f"Promotion live: {promotion_result.get('status', 'inconnue')}",
        f"Selection live: {selection}",
        f"Blocage: {promotion_gate.get('reason', promotion_result.get('reason', 'aucun'))}",
        f"Univers evalue: {len(live_universe)} symboles",
        "",
        "Metriques:",
        f"- Win rate: {_format_metric(metrics.get('win_rate'), '%')}",
        f"- Return: {_format_metric(metrics.get('return_pct'), '%')}",
        f"- Profit factor: {_format_metric(metrics.get('profit_factor'))}",
        f"- Trades: {_format_metric(metrics.get('total_trades'))}",
        f"- Eval games: {_format_metric(metrics.get('evaluation_games'))}",
        f"- Eval symbols: {_format_metric(metrics.get('evaluation_symbols'))}",
        f"- Expectancy: {_format_metric(metrics.get('expectancy_pct'), '%')}",
        f"- Drawdown max: {_format_metric(metrics.get('max_drawdown_pct'), '%')}",
        f"- Episodes positifs: {_format_metric(metrics.get('positive_episode_rate'), '%')}",
        f"- Hold drag: {_format_metric(mechanics.get('hold_drag_score'))}",
        f"- Close quality: {_format_metric(mechanics.get('close_quality_score'))}",
        f"- Split eff.: {_format_metric(mechanics.get('split_efficiency'))}",
        f"- Pyramid eff.: {_format_metric(mechanics.get('pyramid_efficiency'))}",
        f"- SLBE capture: {_format_metric(mechanics.get('slbe_capture_rate'))}",
        "",
        "Validation:",
        f"- Echantillon suffisant: {bool(validation.get('sample_size_ok', False))}",
        f"- Min games: {_format_metric(validation.get('min_games'))}",
        f"- Min symbols: {_format_metric(validation.get('min_symbols'))}",
        f"- Edge minimal: {_format_metric(validation.get('score_edge_required'))}",
    ]

    try:
        TelegramClient().send_sync("\n".join(lines))
    except Exception as exc:
        logger.warning("Notification Telegram training ignoree: %s", exc)


def send_nightly_summary(summary: dict[str, Any]) -> None:
    """Diffuse le resume final d'une sequence nocturne.

    Args:
        summary (dict[str, Any]): Resume de `train_nightly_stack.py`.
    """
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING", True):
        return

    steps = summary.get("steps", []) or []
    promoter = ChampionPromoter()
    horizons = [
        item.strip().lower()
        for item in os.getenv("MUZERO_HORIZONS", "scalp,intraday,swing").split(",")
        if item.strip()
    ]
    lines = [
        "RESUME NIGHTLY",
        f"Statut: {summary.get('status', 'inconnu')}",
        f"Strategie: {summary.get('strategy', 'n/a')}",
        f"Raison: {summary.get('reason', 'n/a')}",
        f"Declencheur: {summary.get('trigger', 'n/a')}",
        f"Debut: {summary.get('started_at', 'n/a')}",
        f"Fin: {summary.get('finished_at', 'n/a')}",
        "",
        "Etapes:",
    ]
    for step in steps:
        name = step.get("name", "unknown")
        status = step.get("status", "unknown")
        error = step.get("error")
        if error:
            lines.append(f"- {name}: {status} ({error})")
        else:
            lines.append(f"- {name}: {status}")

    lines.append("")
    lines.append("Champions:")
    for horizon in horizons:
        try:
            status = promoter.build_horizon_status(horizon)
        except Exception as exc:
            logger.warning("Lecture du statut champion impossible pour %s: %s", horizon, exc)
            lines.append(f"- {horizon}: lecture impossible")
            continue
        lines.append(
            "- "
            f"{horizon}: candidat={status.get('candidate_id') or 'aucun'} | "
            f"live={status.get('live_champion_id') or 'aucun'} | "
            f"selection={status.get('selection') or 'none'} | "
            f"gate={status.get('gate_reason') or 'n/a'}"
        )

    try:
        TelegramClient().send_sync("\n".join(lines))
    except Exception as exc:
        logger.warning("Notification Telegram nightly ignoree: %s", exc)
