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
from eva_lab.training_notifier import send_nightly_summary, send_training_run_started
from eva_lab.training_status import (
    append_training_log,
    build_training_universe_summary,
    finalize_training_status,
    mark_skip_status,
    mark_step_finished,
    mark_step_running,
    reset_training_status,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.nightly_training")

WORKDIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = WORKDIR / "data" / "checkpoints" / "nightly_training_summary.json"
LOCK_PATH = WORKDIR / "data" / "checkpoints" / "nightly_training.lock"
SHADOW_DIR = WORKDIR / "data" / "shadow_learning"


def _enforce_parent_runtime_env() -> None:
    """Impose un profil CPU-only au parent nightly.

    Le parent ne doit pas initialiser JAX sur GPU pendant la detection de
    strategie, la lecture des manifestes ou la supervision. Le vrai trainer
    MuZero rebasculera explicitement en mode GPU dans `run_step`.
    """

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.05")


def _targets_muzero_trainer(command: list[str]) -> bool:
    """Retourne `True` si la commande cible le trainer MuZero global ou Dreamer offline."""

    return any("scripts/train_global_models.py" in str(token) or "eva_lab.muzero.offline_trainer" in str(token) for token in command)


def _build_muzero_child_runtime_env(base_env: dict[str, str]) -> dict[str, str]:
    """Construit l'environnement GPU-only du sous-processus MuZero.

    Args:
        base_env (dict[str, str]): Environnement parent deja resolu.

    Returns:
        dict[str, str]: Environnement complet du trainer MuZero.
    """

    child_env = dict(base_env)
    child_env["CUDA_VISIBLE_DEVICES"] = str(
        base_env.get(
            "TRAINING_CHILD_CUDA_VISIBLE_DEVICES",
            base_env.get("TRAINING_GPU_DEVICE", "1"),
        )
    ).strip()
    child_env["JAX_PLATFORMS"] = str(
        base_env.get("TRAINING_CHILD_JAX_PLATFORMS", "cuda")
    ).strip() or "cuda"
    child_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(
        base_env.get("TRAINING_CHILD_XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    ).strip() or "false"
    child_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
        base_env.get("TRAINING_CHILD_XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    ).strip() or "0.85"
    return child_env


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


def _load_json_file(path: Path) -> dict[str, object] | None:
    """Charge un fichier JSON si present.

    Args:
        path (Path): Chemin cible.

    Returns:
        dict[str, object] | None: Charge utile JSON ou ``None``.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Lecture JSON impossible pour %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _load_previous_summary() -> dict[str, object] | None:
    """Charge le dernier resume nightly si disponible.

    Returns:
        dict[str, object] | None: Resume precedent ou ``None``.
    """
    return _load_json_file(SUMMARY_PATH)


def _load_lock_payload() -> dict[str, object] | None:
    """Charge le verrou actif si present.

    Returns:
        dict[str, object] | None: Charge utile du verrou ou ``None``.
    """
    return _load_json_file(LOCK_PATH)


def _write_lock_payload(payload: dict[str, object]) -> None:
    """Ecrit le verrou de run actif sur disque.

    Args:
        payload (dict[str, object]): Metadonnees du run actif.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_lock_stale(payload: dict[str, object] | None) -> bool:
    """Determine si un verrou parait obsolete.

    Args:
        payload (dict[str, object] | None): Verrou charge.

    Returns:
        bool: ``True`` si le verrou doit etre purge.
    """
    if not payload:
        return True
    started_at = _parse_iso_datetime(payload.get("started_at"))
    age_hours = _hours_since(started_at)
    if age_hours is None:
        return True
    stale_after_hours = _env_int("TRAINING_RUN_LOCK_MAX_AGE_HOURS", 18)
    return age_hours >= stale_after_hours


def _record_skip_event(reason: str, trigger: str, lock_payload: dict[str, object] | None) -> None:
    """Ajoute un evenement de skip dans le resume nightly sans ecraser un run actif.

    Args:
        reason (str): Raison explicite du skip.
        trigger (str): Origine du lancement (`cron`, `manual`, etc.).
        lock_payload (dict[str, object] | None): Verrou qui a provoque le skip.
    """
    summary = _load_previous_summary() or {}
    if not summary:
        summary = {
            "started_at": datetime.now().isoformat(),
            "status": "skipped",
            "strategy": "n/a",
            "reason": reason,
            "steps": [],
        }
    skip_event = {
        "trigger": trigger,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "lock": lock_payload or {},
    }
    skip_events = list(summary.get("skip_events") or [])
    skip_events.append(skip_event)
    summary["skip_events"] = skip_events[-20:]
    summary["last_skip_event"] = skip_event
    if summary.get("status") in {None, "skipped"}:
        summary["status"] = "skipped"
        summary["reason"] = reason
        summary["finished_at"] = datetime.now().isoformat()
    persist_summary(summary)
    mark_skip_status(reason, trigger, lock_payload)


def acquire_run_lock() -> tuple[bool, dict[str, object] | None]:
    """Acquiert le verrou nightly si possible.

    Returns:
        tuple[bool, dict[str, object] | None]: Etat d'acquisition et charge utile.
    """
    if _env_flag("NIGHTLY_RUN_LOCK_ALREADY_HELD", False):
        payload = _load_lock_payload() or {
            "mode": "external",
            "trigger": os.getenv("TRAINING_RUN_TRIGGER", "external"),
            "started_at": datetime.now().isoformat(),
        }
        return True, payload

    existing_lock = _load_lock_payload()
    if existing_lock and not _is_lock_stale(existing_lock):
        return False, existing_lock

    if LOCK_PATH.exists():
        try:
            LOCK_PATH.unlink()
            logger.warning("Verrou nightly obsolete supprime: %s", LOCK_PATH)
        except OSError as exc:
            logger.warning("Suppression du verrou nightly impossible: %s", exc)
            return False, existing_lock

    payload = {
        "pid": os.getpid(),
        "trigger": os.getenv("TRAINING_RUN_TRIGGER", "manual"),
        "started_at": datetime.now().isoformat(),
        "holder": "train_nightly_stack",
        "workdir": str(WORKDIR),
    }
    _write_lock_payload(payload)
    return True, payload


def release_run_lock(lock_payload: dict[str, object] | None) -> None:
    """Libere le verrou nightly si le processus en est proprietaire.

    Args:
        lock_payload (dict[str, object] | None): Charge utile capturee a l'acquisition.
    """
    if not lock_payload:
        return
    if lock_payload.get("mode") == "external":
        return
    current_lock = _load_lock_payload()
    if current_lock and current_lock.get("pid") != lock_payload.get("pid"):
        return
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Liberation du verrou nightly impossible: %s", exc)


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
        # Mode recherche massif : GNN actif pour produire de meilleurs embeddings
        _set_env_default("RUN_TRAIN_GNN", "1")
        _set_env_default("RUN_TRAIN_MUZERO", "1")
        _set_env_default("RUN_TRAIN_DREAMER", "1")
        # Arena et metriques actives pour promouvoir des champions a chaque run
        _set_env_default("RUN_ARENA", "1")
        _set_env_default("RUN_EXPORT_METRICS", "1")
        _set_env_default("MUZERO_TRAINING_STEPS", "8000")
        _set_env_default("MUZERO_GAMES_PER_SYMBOL", "10")
        _set_env_default("ARENA_GAMES_PER_SYMBOL", "4")
        _set_env_default("ARENA_MIN_GAMES", "12")
        _set_env_default("ARENA_MIN_SYMBOLS", "4")
        _set_env_default("MUZERO_MAX_SYMBOLS", "0")
        _set_env_default("ARENA_MAX_SYMBOLS", "0")
        return

    if strategy == "refresh":
        _set_env_default("TRAINING_PROFILE", "refresh")
        _set_env_default("RUN_TRAIN_GNN", "1")
        _set_env_default("RUN_TRAIN_MUZERO", "1")
        _set_env_default("RUN_TRAIN_DREAMER", "1")
        _set_env_default("RUN_ARENA", "1")
        _set_env_default("RUN_EXPORT_METRICS", "1")
        _set_env_default("MUZERO_TRAINING_STEPS", "4000")
        _set_env_default("MUZERO_GAMES_PER_SYMBOL", "5")
        _set_env_default("ARENA_GAMES_PER_SYMBOL", "2")
        _set_env_default("ARENA_MIN_GAMES", "6")
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
    mark_step_finished(name, status, error)


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
    if _targets_muzero_trainer(command):
        env = _build_muzero_child_runtime_env(env)
        logger.info(
            "Etape %s executee en mode GPU cible (CUDA_VISIBLE_DEVICES=%s, JAX_PLATFORMS=%s).",
            name,
            env.get("CUDA_VISIBLE_DEVICES"),
            env.get("JAX_PLATFORMS"),
        )
        append_training_log(
            (
                f"Etape {name}: runtime MuZero cible "
                f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} "
                f"JAX_PLATFORMS={env.get('JAX_PLATFORMS')}."
            ),
            source="nightly",
        )

    logger.info("Debut etape %s: %s", name, command)
    mark_step_running(name, phase="demarrage")
    append_training_log(
        f"Debut de l'etape {name}.",
        source="nightly",
    )
    result = subprocess.run(command, cwd=WORKDIR, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Echec de l'etape {name} (code {result.returncode}).")
    logger.info("Etape %s terminee avec succes.", name)


def _stop_vllm_container() -> None:
    """Arrete temporairement le conteneur vLLM pour liberer la VRAM sur le GPU 0."""
    logger.info("Arrêt temporaire du conteneur vLLM sur le GPU 0...")
    try:
        root_dir = WORKDIR.parent.parent
        res = subprocess.run(
            ["docker", "compose", "stop", "vllm"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0:
            logger.info("Conteneur vLLM arrete avec succes via docker compose stop vllm.")
            return
        
        logger.warning("Echec de docker compose stop vllm (%s). Tentative de docker stop direct...", res.stderr.strip() or "inconnu")
        subprocess.run(["docker", "stop", "the_hive-vllm-1"], capture_output=True, check=False)
        subprocess.run(["docker", "stop", "vllm"], capture_output=True, check=False)
    except Exception as exc:
        logger.warning("Impossible d'arreter le conteneur vLLM : %s. Poursuite de la sequence nocturne.", exc)


def _start_vllm_container() -> None:
    """Redemarre le conteneur vLLM a la fin de la sequence."""
    logger.info("Redémarrage du conteneur vLLM...")
    try:
        root_dir = WORKDIR.parent.parent
        res = subprocess.run(
            ["docker", "compose", "start", "vllm"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0:
            logger.info("Conteneur vLLM redemarre avec succes.")
            return
        
        subprocess.run(["docker", "start", "the_hive-vllm-1"], capture_output=True, check=False)
        subprocess.run(["docker", "start", "vllm"], capture_output=True, check=False)
    except Exception as exc:
        logger.error("Impossible de redemarrer le conteneur vLLM : %s", exc)


def main() -> dict[str, object]:
    """Lance GNN, MuZero multi-horizon puis Dreamer offline.

    Returns:
        dict[str, object]: Resume complet de la sequence nightly.
    """
    _enforce_parent_runtime_env()
    decision = decide_training_strategy()
    trigger = os.getenv("TRAINING_RUN_TRIGGER", "manual")
    summary: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "workdir": str(WORKDIR),
        "steps": [],
        "status": "running",
        "strategy": decision.get("strategy"),
        "reason": decision.get("reason"),
        "decision": decision,
        "trigger": trigger,
        "lock_file": str(LOCK_PATH),
    }
    run_id = f"nightly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    lock_payload: dict[str, object] | None = None
    lock_acquired, lock_payload = acquire_run_lock()
    if not lock_acquired:
        logger.warning("Sequence nightly ignoree: verrou deja actif (%s).", lock_payload)
        _record_skip_event("run_already_active", trigger, lock_payload)
        return {
            "status": "skipped",
            "reason": "run_already_active",
            "trigger": trigger,
            "active_lock": lock_payload,
        }

    summary["lock"] = lock_payload
    persist_summary(summary)
    logger.info(
        "Strategie nightly retenue: %s (%s).",
        summary["strategy"],
        summary["reason"],
    )
    append_training_log(
        f"Strategie nightly retenue: {summary['strategy']} ({summary['reason']}).",
        source="nightly",
    )

    # Purge des vieux checkpoints MuZero incompatibles avant de lancer l'entrainement
    try:
        from scripts.purge_incompatible_checkpoints import purge_incompatible_checkpoints
        purge_report = purge_incompatible_checkpoints()
        if purge_report.get("archived", 0) > 0:
            logger.warning(
                "%d checkpoints MuZero incompatibles archives dans %s.",
                purge_report["archived"],
                purge_report.get("archive_dir", "?"),
            )
            append_training_log(
                f"Purge MuZero: {purge_report['archived']} checkpoints incompatibles archives.",
                source="nightly",
            )
    except Exception as purge_exc:
        logger.warning("Purge MuZero ignoree: %s", purge_exc)

    if summary["strategy"] == "skip":
        summary["status"] = "skipped"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        mark_skip_status(str(summary["reason"]), trigger, lock_payload)
        send_nightly_summary(summary)
        logger.info("Sequence nightly ignoree: %s", summary["reason"])
        release_run_lock(lock_payload)
        return summary

    apply_training_strategy(decision)
    universe_summary = build_training_universe_summary()
    reset_training_status(
        run_id=run_id,
        trigger=trigger,
        strategy=str(summary.get("strategy") or "research"),
        reason=str(summary.get("reason") or "manual"),
        universe=universe_summary,
    )
    send_training_run_started(
        run_id=run_id,
        strategy=str(summary.get("strategy") or "research"),
        reason=str(summary.get("reason") or "manual"),
        trigger=trigger,
        universe=universe_summary,
    )

    run_jepa = _env_flag("RUN_TRAIN_JEPA", True)
    run_gnn = _env_flag("RUN_TRAIN_GNN", True)
    run_muzero = _env_flag("RUN_TRAIN_MUZERO", True)
    run_dreamer = _env_flag("RUN_TRAIN_DREAMER", True)
    run_arena = _env_flag("RUN_ARENA", True)
    run_export_metrics = _env_flag("RUN_EXPORT_METRICS", True)
    # En mode entraînement massif, on ne touche pas au vLLM
    # Positionner TRAINING_DISABLE_VLLM_TOGGLE=0 dans .env pour réactiver
    disable_vllm_toggle = _env_flag("TRAINING_DISABLE_VLLM_TOGGLE", True)

    try:
        # Gestion vLLM : uniquement si le toggle n'est pas désactivé
        if disable_vllm_toggle:
            logger.info(
                "Mode entraînement massif : vLLM non touché (TRAINING_DISABLE_VLLM_TOGGLE=1). "
                "Positionner TRAINING_DISABLE_VLLM_TOGGLE=0 pour réactiver la rotation vLLM."
            )
        else:
            _stop_vllm_container()

        if run_jepa:
            logger.info("Lancement du pre-entrainement auto-supervise VICReg (Market-JEPA)...")
            run_step("jepa_pretrain", [sys.executable, "scripts/train_jepa.py"], extra_env={"CUDA_VISIBLE_DEVICES": "0"})
            append_step(summary, "jepa_pretrain", "ok")

        if run_gnn:
            run_step("gnn", [sys.executable, "scripts/train_gnn.py"],
                     extra_env={"CUDA_VISIBLE_DEVICES": "0", "JAX_PLATFORMS": "cpu"})
            append_step(summary, "gnn", "ok")

        if run_muzero:
            horizons = _resolve_horizons()
            for horizon in horizons:
                step_name = f"muzero_{horizon}"
                # Entraînement parallèle JAX sur les deux GPU (0 et 1)
                run_step(
                    step_name,
                    [sys.executable, "scripts/train_global_models.py"],
                    extra_env={
                        "MUZERO_HORIZON": horizon,
                        "CUDA_VISIBLE_DEVICES": "1",
                        "TRAINING_CHILD_CUDA_VISIBLE_DEVICES": "1"
                    },
                )
                append_step(summary, step_name, "ok")

        if run_dreamer:
            run_step(
                "dreamer_offline",
                [sys.executable, "-m", "eva_lab.muzero.offline_trainer"],
                extra_env={
                    "DREAMER_EPOCHS": os.getenv("DREAMER_EPOCHS", "1500"),
                    "CUDA_VISIBLE_DEVICES": "0",
                    "TRAINING_CHILD_CUDA_VISIBLE_DEVICES": "0"
                },
            )
            append_step(summary, "dreamer_offline", "ok")

        # Arena : compare challenger vs champion et promeut si victoire
        if run_arena:
            try:
                logger.info("Lancement de l'Arena : comparaison challenger vs champion...")
                run_step(
                    "arena_promote",
                    [sys.executable, "scripts/run_arena_promote.py"],
                    extra_env={"CUDA_VISIBLE_DEVICES": "0"},
                )
                append_step(summary, "arena_promote", "ok")
            except Exception as arena_exc:
                # L'Arena ne bloque pas la sequence si elle echoue
                logger.warning("Arena echouee : %s. La sequence continue.", arena_exc)
                append_step(summary, "arena_promote", "warning", str(arena_exc))

        # Export metriques JSON structurees (lisibles par Nexus/Dashboard)
        if run_export_metrics:
            try:
                run_step(
                    "export_metrics",
                    [sys.executable, "scripts/export_training_metrics.py"],
                )
                append_step(summary, "export_metrics", "ok")
            except Exception as metrics_exc:
                logger.warning("Export metriques echoue : %s.", metrics_exc)
                append_step(summary, "export_metrics", "warning", str(metrics_exc))

        # Couplage AlphaEvolve Feedback Bridge (Live Bridging)
        try:
            logger.info("Executing AlphaEvolve feedback bridge...")
            run_step(
                "alphaevolve_bridge",
                [sys.executable, "scripts/apply_alphaevolve_best.py"]
            )
            append_step(summary, "alphaevolve_bridge", "ok")
        except Exception as bridge_exc:
            logger.warning("AlphaEvolve feedback bridge failed: %s", bridge_exc)
            append_step(summary, "alphaevolve_bridge", "error")

        # Hermes Loss Auditor
        try:
            logger.info("Executing Hermes Loss Auditor...")
            run_step(
                "hermes_loss_auditor",
                [sys.executable, "scripts/hermes_loss_auditor.py"]
            )
            append_step(summary, "hermes_loss_auditor", "ok")
        except Exception as audit_exc:
            logger.warning("Hermes Loss Auditor failed: %s", audit_exc)
            append_step(summary, "hermes_loss_auditor", "error")

        # Red Team : analyse les trades live et detecte les faiblesses du champion
        if run_arena:
            try:
                logger.info("Lancement du Red Team : analyse des hard negatifs live...")
                run_step(
                    "redteam",
                    [sys.executable, "scripts/run_redteam.py", "--window", "30"],
                    extra_env={"CUDA_VISIBLE_DEVICES": "0"},
                )
                append_step(summary, "redteam", "ok")
            except Exception as redteam_exc:
                logger.warning("Red Team echouee : %s.", redteam_exc)
                append_step(summary, "redteam", "warning", str(redteam_exc))

        summary["status"] = "ok"
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        finalize_training_status("ok", reason=str(summary.get("reason") or "complete"))
        send_nightly_summary(summary)
        logger.info("Resume nocturne ecrit dans %s", SUMMARY_PATH)
        return summary
    except Exception as exc:
        logger.exception("Sequence nocturne en echec: %s", exc)
        summary["status"] = "error"
        summary["error"] = str(exc)
        summary["finished_at"] = datetime.now().isoformat()
        persist_summary(summary)
        finalize_training_status("error", reason=str(exc))
        send_nightly_summary(summary)
        raise
    finally:
        # Redemarrage vLLM uniquement si le toggle est desactive
        if not disable_vllm_toggle:
            _start_vllm_container()
        release_run_lock(lock_payload)


if __name__ == "__main__":
    report = main()
    logger.info("Sequence nocturne terminee: %s", report)
