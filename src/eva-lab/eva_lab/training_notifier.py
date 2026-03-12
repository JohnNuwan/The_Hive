"""Notifications Telegram pour les entrainements EVA Lab."""

from __future__ import annotations

import logging
import os
from typing import Any

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

    lines = [
        "RAPPORT D'ENTRAINEMENT",
        f"Horizon: {horizon}",
        f"Candidat: {challenger.get('id', 'inconnu')}",
        f"Arena: {battle_report.get('outcome', 'UNKNOWN')}",
        f"Promotion live: {promotion_result.get('status', 'inconnue')}",
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
    lines = [
        "RESUME NIGHTLY",
        f"Statut: {summary.get('status', 'inconnu')}",
        f"Strategie: {summary.get('strategy', 'n/a')}",
        f"Raison: {summary.get('reason', 'n/a')}",
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

    try:
        TelegramClient().send_sync("\n".join(lines))
    except Exception as exc:
        logger.warning("Notification Telegram nightly ignoree: %s", exc)
