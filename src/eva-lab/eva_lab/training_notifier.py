"""Notifications Telegram pour les entrainements EVA Lab."""

from __future__ import annotations

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

    lines = [
        "POINT ENTRAINEMENT",
        _format_timestamp(datetime.now().isoformat()),
        "",
        "RUN",
        f"- id: {_shorten_identifier(status.get('run_id'), default='aucun', max_length=56)}",
        f"- statut: {_humanize_token(status.get('status'), default='inconnu')}",
        f"- strategie: {_humanize_token(status.get('strategy'))}",
        f"- raison: {_humanize_token(status.get('reason'))}",
        f"- reprise: {_humanize_token(status.get('resume_source'))}",
    ]

    if current_step:
        lines.extend(
            [
                "",
                "ACTIF",
                f"- etape: {_humanize_token(current_step.get('name'))}",
                f"- phase: {_humanize_token(current_step.get('phase'))} | horizon: {_humanize_token(current_step.get('horizon'))}",
                f"- symbole: {current_step.get('symbol') or 'n/a'} ({_format_ratio(current_step.get('symbol_index'), current_step.get('symbol_total'))})",
                f"- partie: {_format_ratio(current_step.get('part_index'), current_step.get('part_total'))} | episode: {_format_ratio(current_step.get('episode_step_current'), current_step.get('episode_step_total'))}",
                f"- replay: {_format_ratio(status.get('replay_cache_entries'), 196)}",
                f"- maj: {_format_timestamp(current_step.get('updated_at') or status.get('updated_at'))}",
            ]
        )

    if latest_metrics:
        lines.extend(
            [
                "",
                "METRIQUES",
                f"- policy: loss_pol={_format_metric(latest_metrics.get('loss_pol'))} | top1={_format_share_metric(latest_metrics.get('policy_top1_share'))} | entropy={_format_metric(latest_metrics.get('policy_entropy'))}",
                f"- filtres: root_mask={_format_share_metric(latest_metrics.get('root_mask_rate'))} | post_veto={_format_share_metric(latest_metrics.get('post_veto_to_hold_rate'))}",
                f"- reward: bonus_doux={_format_share_metric(latest_metrics.get('soft_entry_bonus_rate'))} | penalite_douce={_format_share_metric(latest_metrics.get('soft_entry_penalty_rate'))}",
                f"- shaping: ratio={_format_metric(latest_metrics.get('soft_penalty_to_bonus_ratio'))} | net={_format_metric(latest_metrics.get('soft_penalty_net'))}",
                f"- equilibre: eq={_format_share_metric(latest_metrics.get('balanced_episode_rate'))} | long={_format_share_metric(latest_metrics.get('long_entry_share'))} | short={_format_share_metric(latest_metrics.get('short_entry_share'))}",
            ]
        )

    if arena_progress:
        lines.extend(
            [
                "",
                "Arena:",
                f"- statut: {_humanize_token(arena_progress.get('status'), default='running')}",
                f"- role: {_humanize_token(arena_progress.get('current_role'))}",
                f"- symbole: {arena_progress.get('current_symbol') or 'n/a'} ({_format_ratio(arena_progress.get('symbol_index'), arena_progress.get('symbol_total'))})",
                f"- scores: challenger={_format_metric(arena_progress.get('challenger_score'))} | champion={_format_metric(arena_progress.get('champion_score'))}",
            ]
        )

    lines.extend(["", "CHAMPIONS MUZERO"])
    for horizon in resolved_horizons:
        horizon_payload = dict((horizon_statuses or {}).get(horizon) or {})
        directional = dict(horizon_payload.get("directional_metrics") or {})
        candidate_metrics = dict(horizon_payload.get("candidate_metrics") or {})
        lines.extend(
            [
                f"{str(horizon or '').upper() or 'N/A'}",
                f"- selection: {_humanize_token(horizon_payload.get('selection'))}",
                f"- live: {_shorten_identifier(horizon_payload.get('live_champion_id'))}",
                f"- candidat: {_shorten_identifier(horizon_payload.get('candidate_id'))}",
                f"- gate: {_humanize_token(horizon_payload.get('gate_reason'))}",
                f"- PF={_format_metric(candidate_metrics.get('profit_factor'))} | Ret={_format_metric(candidate_metrics.get('return_pct'), '%')} | WR={_format_metric(candidate_metrics.get('win_rate'), '%')}",
                f"- Bias={_humanize_token(directional.get('directional_bias'))}",
            ]
        )

    return "\n".join(lines)


def send_training_digest(
    *,
    run_status: dict[str, Any] | None = None,
    horizon_statuses: dict[str, dict[str, Any]] | None = None,
    horizons: list[str] | None = None,
) -> None:
    """Envoie un digest Telegram compact de l'etat training.

    Args:
        run_status (dict[str, Any] | None): Statut courant explicite du run.
        horizon_statuses (dict[str, dict[str, Any]] | None): Statuts champion
            deja resolves par horizon.
        horizons (list[str] | None): Horizons a afficher.
    """
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING", True):
        return
    if not _env_flag("TELEGRAM_NOTIFY_TRAINING_DIGEST", True):
        return

    try:
        TelegramClient().send_sync(
            build_training_digest_message(
                run_status=run_status,
                horizon_statuses=horizon_statuses,
                horizons=horizons,
            )
        )
    except Exception as exc:
        logger.warning("Notification Telegram de digest training ignoree: %s", exc)


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
