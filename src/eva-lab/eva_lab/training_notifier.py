"""Notifications Telegram pour les entrainements EVA Lab."""

from __future__ import annotations

import logging
import os
from typing import Any

from eva_lab.champion_promoter import ChampionPromoter
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
