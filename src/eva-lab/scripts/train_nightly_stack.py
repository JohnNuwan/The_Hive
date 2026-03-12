"""Orchestre la sequence nocturne complete des entrainements trading."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from eva_lab.champion_promoter import ChampionPromoter
from eva_lab.training_notifier import send_nightly_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.nightly_training")

WORKDIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = WORKDIR / "data" / "checkpoints" / "nightly_training_summary.json"
SHADOW_DIR = WORKDIR / "data" / "shadow_learning"


def _env_flag(name: str, default: bool) -> bool:
    """Interprete une variable d'environnement booleenne.

    Args:
        name (str): Nom de la variable a lire.
        default (bool): Valeur de repli si la variable est absente.

    Returns:
        bool: Valeur booleenne normalisee.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Lit un entier depuis l'environnement avec repli robuste.

    Args:
        name (str): Nom de la variable.
        default (int): Valeur de repli.

    Returns:
        int: Valeur entiere exploitable.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Valeur entiere invalide pour %s=%s. Repli=%s.", name, raw_value, default)
        return default


def _parse_iso_datetime(value: object) -> datetime | None:
    """Convertit une date ISO en objet ``datetime``.

    Args:
        value (object): Valeur brute a convertir.

    Returns:
        datetime | None: Date valide ou ``None`` si la conversion echoue.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _hours_since(timestamp: datetime | None) -> float | None:
    """Calcule l'age en heures d'un evenement.

    Args:
        timestamp (datetime | None): Date de reference.

    Returns:
        float | None: Age en heures ou ``None``.
    """
    if timestamp is None:
        return None
    return max((datetime.now() - timestamp).total_seconds() / 3600.0, 0.0)


def _load_previous_summary() -> dict[str, object] | None:
    """Charge le dernier resume nightly si disponible.

    Returns:
        dict[str, object] | None: Resume precedent ou ``None``.
    """
    if not SUMMARY_PATH.exists():
        return None
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Lecture du resume nightly impossible: %s", exc)
        return None


def _collect_shadow_learning_stats() -> dict[str, object]:
    """Mesure le volume et la fraicheur des donnees Shadow Learning.

    Returns:
        dict[str, object]: Compteurs de lignes et date de modification.
    """
    latest_modified: datetime | None = None
    total_records = 0
    jsonl_files = sorted(SHADOW_DIR.rglob("*.jsonl")) if SHADOW_DIR.exists() else []

    for file_path in jsonl_files:
        try:
            modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
            latest_modified = (
                modified_at if latest_modified is None or modified_at > latest_modified else latest_modified
            )
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                total_records += sum(1 for line in handle if line.strip())
        except OSError as exc:
            logger.warning("Lecture Shadow Learning impossible pour %s: %s", file_path, exc)

    return {
        "records": total_records,
        "latest_modified_at": latest_modified.isoformat() if latest_modified else None,
        "files": len(jsonl_files),
    }


def _resolve_horizons() -> list[str]:
    """Retourne les horizons strategiques a traiter.

    Returns:
        list[str]: Horizons ordonnes et nettoyes.
    """
    return [
        item.strip().lower()
        for item in os.getenv("MUZERO_HORIZONS", "scalp,intraday,swing").split(",")
        if item.strip()
    ]


def _build_champion_snapshot(promoter: ChampionPromoter, horizons: list[str]) -> dict[str, object]:
    """Assemble l'etat des champions par horizon.

    Args:
        promoter (ChampionPromoter): Promoteur central des champions.
        horizons (list[str]): Horizons a inspecter.

    Returns:
        dict[str, object]: Snapshot d'etat par horizon.
    """
    snapshot: dict[str, object] = {}
    for horizon in horizons:
        status = promoter.build_horizon_status(horizon)
        promotion_gate = status.get("promotion_gate", {}) or {}
        live_checkpoint = status.get("live_checkpoint", {}) or {}
        snapshot[horizon] = {
            "allowed": bool(promotion_gate.get("allowed", False)),
            "reason": promotion_gate.get("reason", "unknown"),
            "selection": status.get("selection"),
            "champion_id": status.get("champion_id"),
            "live_checkpoint_exists": bool(live_checkpoint.get("exists", False)),
            "live_checkpoint_modified_at": live_checkpoint.get("modified_at"),
            "live_universe_count": ((status.get("live_universe") or {}).get("count")) or 0,
        }
    return snapshot


def decide_training_strategy() -> dict[str, object]:
    """Choisit automatiquement la strategie nightly.

    La decision suit trois modes:
    - ``skip``: aucun retrain utile ce soir.
    - ``refresh``: retrain de maintien sur echantillon reduit.
    - ``research``: run massif pour chercher un nouveau champion.

    Returns:
        dict[str, object]: Strategie retenue, raison et contexte de decision.
    """
    automation_mode = os.getenv("TRAINING_AUTOMATION_MODE", "smart").strip().lower()
    previous_summary = _load_previous_summary() or {}
    previous_finished_at = _parse_iso_datetime(previous_summary.get("finished_at"))
    shadow_stats = _collect_shadow_learning_stats()
    shadow_latest = _parse_iso_datetime(shadow_stats.get("latest_modified_at"))
    shadow_age_hours = _hours_since(shadow_latest)
    new_shadow_since_last = bool(
        previous_finished_at is None
        or (shadow_latest is not None and shadow_latest > previous_finished_at)
    )
    min_shadow_records = _env_int("TRAINING_MIN_SHADOW_RECORDS", 25)
    refresh_after_hours = _env_int("TRAINING_REFRESH_AFTER_HOURS", 24)
    max_champion_age_hours = _env_int("TRAINING_MAX_CHAMPION_AGE_HOURS", 72)
    promoter = ChampionPromoter(
        weights_dir=str(WORKDIR / "data" / "muzero" / "weights"),
        results_dir=str(WORKDIR / "data" / "muzero" / "results"),
    )
    horizons = _resolve_horizons()
    champion_snapshot = _build_champion_snapshot(promoter, horizons)
    deployable_horizons = [
        horizon
        for horizon, status in champion_snapshot.items()
        if isinstance(status, dict)
        and status.get("allowed")
        and status.get("live_checkpoint_exists")
    ]
    checkpoint_dates = [
        _parse_iso_datetime((status or {}).get("live_checkpoint_modified_at"))
        for status in champion_snapshot.values()
        if isinstance(status, dict)
    ]
    checkpoint_dates = [value for value in checkpoint_dates if value is not None]
    oldest_live_hours = max((_hours_since(value) or 0.0) for value in checkpoint_dates) if checkpoint_dates else None

    decision: dict[str, object] = {
        "mode": automation_mode,
        "strategy": "research",
        "reason": "default_research",
        "shadow_stats": shadow_stats,
        "champion_snapshot": champion_snapshot,
        "deployable_horizons": deployable_horizons,
        "previous_finished_at": previous_summary.get("finished_at"),
        "oldest_live_hours": oldest_live_hours,
    }

    if automation_mode in {"always", "force_research"}:
        decision["reason"] = "forced_research"
        return decision

    if automation_mode == "disabled":
        decision["strategy"] = "skip"
        decision["reason"] = "automation_disabled"
        return decision

    if not deployable_horizons:
        decision["strategy"] = "research"
        decision["reason"] = "no_deployable_champion"
        return decision

    if oldest_live_hours is not None and oldest_live_hours >= max_champion_age_hours:
        decision["strategy"] = "research"
        decision["reason"] = "champion_stale"
        return decision

    if (
        new_shadow_since_last
        and int(shadow_stats.get("records", 0)) >= min_shadow_records
        and (shadow_age_hours is None or shadow_age_hours <= refresh_after_hours + 12)
    ):
        decision["strategy"] = "refresh"
        decision["reason"] = "new_shadow_data"
        return decision

    if previous_finished_at is None:
        decision["strategy"] = "refresh"
        decision["reason"] = "no_previous_nightly"
        return decision

    since_last_hours = _hours_since(previous_finished_at)
    if since_last_hours is not None and since_last_hours >= refresh_after_hours:
        decision["strategy"] = "refresh"
        decision["reason"] = "refresh_window_elapsed"
        return decision

    decision["strategy"] = "skip"
    decision["reason"] = "champion_recent_and_no_new_data"
    return decision


def _set_env_default(name: str, value: str) -> None:
    """Definit une variable d'environnement sans ecraser un choix explicite.

    Args:
        name (str): Nom de la variable.
        value (str): Valeur par defaut a appliquer.
    """
    if not os.getenv(name):
        os.environ[name] = value


def apply_training_strategy(decision: dict[str, object]) -> None:
    """Applique les parametres adaptes a la strategie nightly choisie.

    Args:
        decision (dict[str, object]): Strategie issue de `decide_training_strategy`.
    """
    strategy = str(decision.get("strategy", "research")).lower()
    if strategy == "research":
        _set_env_default("TRAINING_PROFILE", "research")
        _set_env_default("RUN_TRAIN_GNN", "0")
        _set_env_default("RUN_TRAIN_MUZERO", "1")
        _set_env_default("RUN_TRAIN_DREAMER", "0")
        _set_env_default("MUZERO_TRAINING_STEPS", "32000")
        _set_env_default("MUZERO_GAMES_PER_SYMBOL", "20")
        _set_env_default("ARENA_GAMES_PER_SYMBOL", "8")
        _set_env_default("ARENA_MIN_GAMES", "24")
        _set_env_default("ARENA_MIN_SYMBOLS", "6")
        _set_env_default("MUZERO_MAX_SYMBOLS", "0")
        _set_env_default("ARENA_MAX_SYMBOLS", "0")
        return

    if strategy == "refresh":
        _set_env_default("TRAINING_PROFILE", "refresh")
        _set_env_default("RUN_TRAIN_GNN", "0")
        _set_env_default("RUN_TRAIN_MUZERO", "1")
        _set_env_default("RUN_TRAIN_DREAMER", "0")
        _set_env_default("MUZERO_TRAINING_STEPS", "8000")
        _set_env_default("MUZERO_GAMES_PER_SYMBOL", "8")
        _set_env_default("ARENA_GAMES_PER_SYMBOL", "4")
        _set_env_default("ARENA_MIN_GAMES", "12")
        _set_env_default("ARENA_MIN_SYMBOLS", "3")
        _set_env_default("MUZERO_MAX_SYMBOLS", "6")
        _set_env_default("ARENA_MAX_SYMBOLS", "6")


def persist_summary(summary: dict[str, object]) -> None:
    """Ecrit le resume courant sur disque pour garder une trace meme en cas d'echec.

    Args:
        summary (dict[str, object]): Resume courant de la sequence nightly.
    """
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def append_step(
    summary: dict[str, object],
    name: str,
    status: str,
    error: str | None = None,
) -> None:
    """Ajoute le resultat d'une etape dans le resume JSON.

    Args:
        summary (dict[str, object]): Resume global en construction.
        name (str): Nom de l'etape.
        status (str): Statut final de l'etape.
        error (str | None): Erreur eventuelle.
    """
    step: dict[str, object] = {"name": name, "status": status}
    if error:
        step["error"] = error
    summary.setdefault("steps", []).append(step)
    persist_summary(summary)


def run_step(name: str, command: list[str], extra_env: dict[str, str] | None = None) -> None:
    """Execute une etape d'entrainement dans un processus isole.

    Args:
        name (str): Nom de l'etape.
        command (list[str]): Commande a lancer.
        extra_env (dict[str, str] | None): Variables d'environnement additionnelles.

    Raises:
        RuntimeError: Si le sous-processus se termine en erreur.
    """
    env = os.environ.copy()
    pythonpath_entries = [str(WORKDIR), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([entry for entry in pythonpath_entries if entry])
    if extra_env:
        env.update(extra_env)

    logger.info("Debut etape %s: %s", name, command)
    result = subprocess.run(command, cwd=WORKDIR, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Echec de l'etape {name} (code {result.returncode}).")
    logger.info("Etape %s terminee avec succes.", name)


def main() -> dict[str, object]:
    """Lance GNN, MuZero multi-horizon puis Dreamer offline.

    Returns:
        dict[str, object]: Resume complet de la sequence nightly.
    """
    decision = decide_training_strategy()
    summary: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "workdir": str(WORKDIR),
        "steps": [],
        "status": "running",
        "strategy": decision.get("strategy"),
        "reason": decision.get("reason"),
        "decision": decision,
    }
    persist_summary(summary)
    logger.info(
        "Strategie nightly retenue: %s (%s).",
        summary["strategy"],
        summary["reason"],
    )

    if summary["strategy"] == "skip":
        summary["status"] = "skipped"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        send_nightly_summary(summary)
        logger.info("Sequence nightly ignoree: %s", summary["reason"])
        return summary

    apply_training_strategy(decision)

    run_gnn = _env_flag("RUN_TRAIN_GNN", True)
    run_muzero = _env_flag("RUN_TRAIN_MUZERO", True)
    run_dreamer = _env_flag("RUN_TRAIN_DREAMER", True)

    try:
        if run_gnn:
            run_step("gnn", [sys.executable, "scripts/train_gnn.py"])
            append_step(summary, "gnn", "ok")

        if run_muzero:
            horizons = _resolve_horizons()
            for horizon in horizons:
                step_name = f"muzero_{horizon}"
                run_step(
                    step_name,
                    [sys.executable, "scripts/train_global_models.py"],
                    extra_env={"MUZERO_HORIZON": horizon},
                )
                append_step(summary, step_name, "ok")

        if run_dreamer:
            run_step(
                "dreamer_offline",
                [sys.executable, "-m", "eva_lab.muzero.offline_trainer"],
                extra_env={"DREAMER_EPOCHS": os.getenv("DREAMER_EPOCHS", "1500")},
            )
            append_step(summary, "dreamer_offline", "ok")

        summary["status"] = "ok"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        send_nightly_summary(summary)
        logger.info("Resume nocturne ecrit dans %s", SUMMARY_PATH)
        return summary
    except Exception as exc:
        logger.exception("Sequence nocturne en echec: %s", exc)
        summary["status"] = "error"
        summary["error"] = str(exc)
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        send_nightly_summary(summary)
        raise


if __name__ == "__main__":
    report = main()
    logger.info("Sequence nocturne terminee: %s", report)
