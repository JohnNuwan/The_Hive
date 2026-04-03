"""Orchestre une chasse MuZero multi-generations pour la nuit.

Cette commande ne modifie pas le champion live en place. Elle utilise le
checkpoint champion courant comme seed, enchaine plusieurs generations proxy
courtes avec crossover et mutations, puis lance un full 7 symboles avec le
meilleur survivant si un candidat credible emerge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import paramiko


ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = Path(__file__).resolve().parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))


def _load_proxmox_module() -> Any:
    """Charge le module de lancement Proxmox existant.

    Returns:
        Any: Module charge dynamiquement.

    Raises:
        RuntimeError: Si le module ne peut pas etre charge.
    """

    module_path = DEPLOY_DIR / "start_training_proxmox.py"
    spec = importlib.util.spec_from_file_location("start_training_proxmox", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Chargement impossible du module {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROXMOX = _load_proxmox_module()

PROFILE = "scalp_fx_v2"
ENGINE = "muzero"
HUNT_PREFIX = "muzero_night_hunt"
TIMESCALE_PROXY_SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD"]
TIMESCALE_FULL_SYMBOLS = list(PROXMOX.FAST_MUZERO_FULL_SYMBOLS)
TIMESCALE_REQUIRED_TIMEFRAMES = ["M5", "H1", "D1"]
PROXY_STEPS = "1800"
PROXY_PRECHECK_STEPS = "600,1200"
PROXY_PRECHECK_GAMES = "2"
PROXY_GAMES_PER_SYMBOL = "2"
PROXY_ARENA_GAMES_PER_SYMBOL = "2"
PROXY_ARENA_MIN_GAMES = "6"
FULL_STEPS = "12000"
FULL_GAMES_PER_SYMBOL = "6"
FULL_ARENA_GAMES_PER_SYMBOL = "6"
FULL_ARENA_MIN_GAMES = "18"
DEFAULT_MAX_GENERATIONS = 3
DEFAULT_GENERATION_SIZE = 3
DEFAULT_BEST_COUNT = 2
DEFAULT_POLL_SECONDS = 30

BASE_TRIAL_IDS = [
    "directional_guard",
    "balanced_activity",
    "close_recovery",
]

ANTI_BIAS_OVERRIDES = {
    "MUZERO_DIRECTIONAL_MAX_IMBALANCE": "0.56",
    "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY": "22.0",
    "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY": "14.0",
    "MUZERO_ACTIVITY_MIN_ENTRIES": "2",
    "MUZERO_HOLD_TREND_PENALTY": "0.72",
    "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER": "0.66",
}

MUTATION_SPACE = {
    "MUZERO_DIRECTIONAL_MAX_IMBALANCE": ("float", 0.48, 0.62, 0.02),
    "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY": ("float", 16.0, 28.0, 2.0),
    "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY": ("float", 8.0, 18.0, 2.0),
    "MUZERO_ACTIVITY_MIN_ENTRIES": ("int", 1, 4, 1),
    "MUZERO_HOLD_TREND_PENALTY": ("float", 0.58, 0.88, 0.04),
    "MUZERO_REWARD_HOLD_DRAG_MULTIPLIER": ("float", 0.45, 0.85, 0.05),
    "MUZERO_SPLIT_MIN_TRADE_RETURN": ("float", 0.0018, 0.0032, 0.0002),
    "MUZERO_SPLIT_MIN_REALIZED_PCT": ("float", 0.014, 0.03, 0.002),
    "MUZERO_SPLIT_FAILURE_PENALTY": ("float", 0.35, 0.6, 0.03),
    "MUZERO_CLOSE_WINNER_THRESHOLD": ("float", 0.0026, 0.0042, 0.0002),
    "MUZERO_CLOSE_TP_LIKE_THRESHOLD": ("float", 0.0022, 0.0036, 0.0002),
    "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": ("float", 0.0055, 0.0078, 0.0003),
    "MUZERO_REWARD_CLOSE_REALIZED_MULTIPLIER": ("float", 1.3, 2.0, 0.1),
    "MUZERO_ENTRY_MIN_ADX": ("int", 8, 12, 1),
}


@dataclass
class HuntContext:
    """Regroupe le contexte d'execution local de la chasse nocturne."""

    hunt_id: str
    hunt_dir: Path
    log_path: Path
    seed_checkpoint_path: str
    seed_champion_id: str
    proxy_symbols: list[str]
    full_symbols: list[str]
    coverage_generated_at: str | None
    deadline: datetime
    random_seed: int


def _now_paris() -> datetime:
    """Retourne l'heure locale du poste d'execution."""

    return datetime.now().astimezone()


def _now_tag() -> str:
    """Construit un identifiant horodate compact."""

    return _now_paris().strftime("%Y%m%d_%H%M%S")


def _append_log(context: HuntContext, message: str) -> None:
    """Ajoute une ligne de log locale et l'affiche.

    Args:
        context (HuntContext): Contexte d'execution.
        message (str): Message a persister.
    """

    timestamp = _now_paris().isoformat(timespec="seconds")
    line = f"{timestamp} | {message}"
    context.log_path.parent.mkdir(parents=True, exist_ok=True)
    with context.log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _fetch_timescaledb_coverage() -> dict[str, Any]:
    """Charge la couverture TimeDB publiee par EVA Lab.

    Returns:
        dict[str, Any]: Bloc `timescaledb` expose par l'API du Lab.

    Raises:
        RuntimeError: Si le rapport de couverture est indisponible.
    """

    coverage_url = f"http://{PROXMOX.HOST}:8600/timescaledb/coverage"
    with urllib.request.urlopen(coverage_url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    coverage = dict(payload.get("timescaledb") or {})
    if not coverage:
        raise RuntimeError("Rapport de couverture TimeDB indisponible.")
    return {
        **coverage,
        "generated_at": payload.get("generated_at"),
    }


def _is_symbol_fully_covered(symbol_payload: dict[str, Any]) -> bool:
    """Determine si un symbole couvre tous les timeframes requis.

    Args:
        symbol_payload (dict[str, Any]): Bloc de couverture d'un symbole.

    Returns:
        bool: `True` si tous les timeframes requis sont disponibles.
    """

    available = {
        str(timeframe).strip().upper()
        for timeframe in list(symbol_payload.get("available_timeframes") or [])
        if str(timeframe).strip()
    }
    missing = {
        str(timeframe).strip().upper()
        for timeframe in list(symbol_payload.get("missing_timeframes") or [])
        if str(timeframe).strip()
    }
    return not missing and set(TIMESCALE_REQUIRED_TIMEFRAMES).issubset(available)


def _resolve_timescaledb_universes() -> tuple[list[str], list[str], str | None]:
    """Valide les univers proxy/full utilisables uniquement via TimeDB.

    Returns:
        tuple[list[str], list[str], str | None]: Univers proxy, univers full
            et horodatage du rapport de couverture.

    Raises:
        RuntimeError: Si la couverture TimeDB n'est pas complete.
    """

    coverage = _fetch_timescaledb_coverage()
    by_symbol = dict(coverage.get("by_symbol") or {})

    missing_full = [
        symbol
        for symbol in TIMESCALE_FULL_SYMBOLS
        if not _is_symbol_fully_covered(dict(by_symbol.get(symbol) or {}))
    ]
    if missing_full:
        raise RuntimeError(
            "Couverture TimeDB incomplete pour la vague du soir: "
            + ", ".join(missing_full)
        )

    missing_proxy = [
        symbol
        for symbol in TIMESCALE_PROXY_SYMBOLS
        if not _is_symbol_fully_covered(dict(by_symbol.get(symbol) or {}))
    ]
    if missing_proxy:
        raise RuntimeError(
            "Couverture TimeDB incomplete pour les symboles proxy: "
            + ", ".join(missing_proxy)
        )

    return (
        list(TIMESCALE_PROXY_SYMBOLS),
        list(TIMESCALE_FULL_SYMBOLS),
        str(coverage.get("generated_at") or "").strip() or None,
    )


def _load_base_trials() -> dict[str, dict[str, Any]]:
    """Indexe les essais MuZero V3 deja definis dans le projet."""

    return {
        str(item.get("trial_id") or "").strip(): dict(item)
        for item in PROXMOX._get_v3_trial_catalog(PROFILE)
        if str(item.get("trial_id") or "").strip()
    }


def _round_numeric(value: float, precision: int) -> str:
    """Formate une valeur numerique en chaine courte."""

    return f"{round(value, precision):.{precision}f}"


def _mutate_value(
    key: str,
    current_value: str | None,
    rng: random.Random,
) -> str:
    """Mutile un parametre dans des bornes defensables.

    Args:
        key (str): Nom du parametre.
        current_value (str | None): Valeur de depart.
        rng (random.Random): Generateur pseudo-aleatoire local.

    Returns:
        str: Nouvelle valeur formatee.
    """

    kind, lower, upper, delta = MUTATION_SPACE[key]
    if kind == "int":
        base_value = int(float(current_value or lower))
        candidate = base_value + int(rng.choice([-delta, delta]))
        candidate = max(int(lower), min(int(upper), int(candidate)))
        return str(candidate)
    precision = 4 if abs(float(delta)) < 0.001 else 3
    base_value = float(current_value or lower)
    candidate = base_value + float(rng.choice([-delta, delta]))
    candidate = max(float(lower), min(float(upper), float(candidate)))
    return _round_numeric(candidate, precision)


def _blend_parent_overrides(
    parent_a: dict[str, str],
    parent_b: dict[str, str],
    *,
    rng: random.Random,
) -> dict[str, str]:
    """Construit un enfant par mix gene-a-gene de deux parents."""

    child = dict(ANTI_BIAS_OVERRIDES)
    keys = set(parent_a) | set(parent_b) | set(MUTATION_SPACE)
    for key in sorted(keys):
        if key.startswith("TRAINING_") or key.startswith("RUN_TRAIN_"):
            continue
        if key in {
            "MUZERO_RESUME_CHECKPOINT_PATH",
            "MUZERO_RESUME_STEP",
            "MUZERO_DATASET_SOURCE",
            "MUZERO_SYMBOLS",
            "MUZERO_SYMBOLS_SCALP",
            "ARENA_SYMBOLS",
            "ARENA_SYMBOLS_SCALP",
            "TRAINING_FOCUS_SYMBOLS",
            "MUZERO_TRAINING_STEPS",
            "MUZERO_GAMES_PER_SYMBOL",
            "ARENA_GAMES_PER_SYMBOL",
            "ARENA_MIN_GAMES",
            "MUZERO_PROXY_PRECHECK_STEPS",
            "MUZERO_PROXY_PRECHECK_GAMES",
        }:
            continue
        if key in parent_a and key in parent_b:
            child[key] = parent_a[key] if rng.random() < 0.5 else parent_b[key]
        elif key in parent_a:
            child[key] = parent_a[key]
        elif key in parent_b:
            child[key] = parent_b[key]
    return child


def _mutate_overrides(
    overrides: dict[str, str],
    *,
    rng: random.Random,
    mutation_count: int = 4,
) -> dict[str, str]:
    """Applique quelques mutations bornees a un genome mecanique."""

    mutated = dict(overrides)
    candidate_keys = [key for key in MUTATION_SPACE if key in mutated or key in ANTI_BIAS_OVERRIDES]
    if not candidate_keys:
        candidate_keys = list(MUTATION_SPACE)
    rng.shuffle(candidate_keys)
    for key in candidate_keys[:mutation_count]:
        mutated[key] = _mutate_value(key, mutated.get(key), rng)
    return mutated


def _build_proxy_runtime_overrides(
    *,
    trial_id: str,
    genome_overrides: dict[str, str],
    generation: int,
    seed_checkpoint_path: str,
    proxy_symbols: list[str],
) -> dict[str, str]:
    """Construit l'environnement proxy a envoyer au serveur."""

    trial_definition = {"trial_id": trial_id, "overrides": genome_overrides}
    symbol_csv = ",".join(proxy_symbols)
    symbol_count = str(len(proxy_symbols))
    runtime_overrides = PROXMOX._build_v4_profile_overrides(
        PROFILE,
        engine=ENGINE,
        mode="proxy_ga",
        trial=trial_definition,
    )
    runtime_overrides.update(
        {
            "TRAINING_RUN_TRIGGER": f"manual_{HUNT_PREFIX}_g{generation:02d}_{trial_id}",
            "TRAINING_GA_GENERATION": str(generation),
            "TRAINING_GA_TRIAL": trial_id,
            "TRAINING_TRIAL_ID": trial_id,
            "MUZERO_RESUME_CHECKPOINT_PATH": seed_checkpoint_path,
            "MUZERO_RESUME_STEP": "0",
            "MUZERO_DATASET_SOURCE": "timescaledb",
            "MUZERO_TRAINING_STEPS": PROXY_STEPS,
            "MUZERO_PROXY_PRECHECK_STEPS": PROXY_PRECHECK_STEPS,
            "MUZERO_PROXY_PRECHECK_GAMES": PROXY_PRECHECK_GAMES,
            "MUZERO_GAMES_PER_SYMBOL": PROXY_GAMES_PER_SYMBOL,
            "ARENA_GAMES_PER_SYMBOL": PROXY_ARENA_GAMES_PER_SYMBOL,
            "ARENA_MIN_GAMES": PROXY_ARENA_MIN_GAMES,
            "TRAINING_FOCUS_SYMBOLS": symbol_csv,
            "MUZERO_SYMBOLS": symbol_csv,
            "MUZERO_SYMBOLS_SCALP": symbol_csv,
            "ARENA_SYMBOLS": symbol_csv,
            "ARENA_SYMBOLS_SCALP": symbol_csv,
            "MUZERO_MAX_SYMBOLS": symbol_count,
            "ARENA_MAX_SYMBOLS": symbol_count,
            "ARENA_MIN_SYMBOLS": symbol_count,
            "MUZERO_PROMOTION_MIN_EVAL_SYMBOLS": symbol_count,
            "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": symbol_count,
            "MUZERO_LIVE_TOP_SYMBOLS": symbol_count,
            "TRAINING_TIMESCALE_ENABLED": "1",
            "TRAINING_GATE_PROFILE": "standard",
            "NIGHTLY_KEEP_VLLM": "0",
        }
    )
    return runtime_overrides


def _build_full_runtime_overrides(
    *,
    trial_id: str,
    genome_overrides: dict[str, str],
    seed_checkpoint_path: str,
    full_symbols: list[str],
) -> dict[str, str]:
    """Construit l'environnement full final 7 symboles."""

    symbol_csv = ",".join(full_symbols)
    symbol_count = str(len(full_symbols))
    runtime_overrides = PROXMOX._build_muzero_full_7_overrides()
    runtime_overrides.update(
        {
            "TRAINING_RUN_TRIGGER": f"manual_{HUNT_PREFIX}_full_{trial_id}",
            "TRAINING_ENGINE": ENGINE,
            "TRAINING_GA_STATUS": "full",
            "TRAINING_GA_GENERATION": "99",
            "TRAINING_GA_TRIAL": trial_id,
            "TRAINING_TRIAL_MODE": "full",
            "TRAINING_TRIAL_COST_PROFILE": "full",
            "TRAINING_TRIAL_ID": trial_id,
            "MUZERO_RESUME_CHECKPOINT_PATH": seed_checkpoint_path,
            "MUZERO_RESUME_STEP": "0",
            "MUZERO_DATASET_SOURCE": "timescaledb",
            "MUZERO_TRAINING_STEPS": FULL_STEPS,
            "MUZERO_GAMES_PER_SYMBOL": FULL_GAMES_PER_SYMBOL,
            "ARENA_GAMES_PER_SYMBOL": FULL_ARENA_GAMES_PER_SYMBOL,
            "ARENA_MIN_GAMES": FULL_ARENA_MIN_GAMES,
            "TRAINING_FOCUS_SYMBOLS": symbol_csv,
            "MUZERO_SYMBOLS": symbol_csv,
            "MUZERO_SYMBOLS_SCALP": symbol_csv,
            "ARENA_SYMBOLS": symbol_csv,
            "ARENA_SYMBOLS_SCALP": symbol_csv,
            "MUZERO_MAX_SYMBOLS": symbol_count,
            "ARENA_MAX_SYMBOLS": symbol_count,
            "ARENA_MIN_SYMBOLS": str(max(3, len(full_symbols) - 1)),
            "MUZERO_PROMOTION_MIN_EVAL_SYMBOLS": str(max(3, len(full_symbols) - 1)),
            "MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS": symbol_count,
            "MUZERO_LIVE_TOP_SYMBOLS": symbol_count,
            "TRAINING_TIMESCALE_ENABLED": "1",
            "TRAINING_GATE_PROFILE": "standard",
            "NIGHTLY_KEEP_VLLM": "0",
        }
    )
    runtime_overrides.update(ANTI_BIAS_OVERRIDES)
    runtime_overrides.update(genome_overrides)
    return runtime_overrides


def _build_generation_one_trials(
    *,
    context: HuntContext,
) -> list[dict[str, Any]]:
    """Construit la premiere generation a partir des essais de base."""

    base_trials = _load_base_trials()
    trials: list[dict[str, Any]] = []
    for trial_id in BASE_TRIAL_IDS:
        source_trial = dict(base_trials.get(trial_id) or {})
        genome_overrides = dict(ANTI_BIAS_OVERRIDES)
        genome_overrides.update(dict(source_trial.get("overrides") or {}))
        trials.append(
            {
                "trial_id": f"g01_{trial_id}",
                "overrides": genome_overrides,
                "runtime_overrides": _build_proxy_runtime_overrides(
                    trial_id=f"g01_{trial_id}",
                    genome_overrides=genome_overrides,
                    generation=1,
                    seed_checkpoint_path=context.seed_checkpoint_path,
                    proxy_symbols=context.proxy_symbols,
                ),
                "notes": f"seed:{trial_id}",
            }
        )
    return trials


def _select_proxy_survivors(results: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    """Retient les survivants les plus defensables d'une generation proxy."""

    completed = [
        item
        for item in results
        if str(item.get("terminal_status") or "").strip().lower() == "completed"
        and not bool(item.get("killed_after_precheck"))
        and str(item.get("failure_mode") or "").strip().lower() not in {"soft_hang", "infra_error"}
    ]
    if completed:
        return completed[:count]
    return results[:count]


def _build_next_generation_trials(
    *,
    context: HuntContext,
    generation: int,
    parents: list[dict[str, Any]],
    generation_size: int,
) -> list[dict[str, Any]]:
    """Construit une generation par crossover et mutations autour des meilleurs."""

    rng = random.Random(context.random_seed + generation)
    trials: list[dict[str, Any]] = []
    parent_a = dict((parents[0] or {}).get("runtime_overrides") or {})
    parent_b = dict((parents[min(1, len(parents) - 1)] or {}).get("runtime_overrides") or {})
    if parent_a:
        elite_genome = {
            key: value
            for key, value in parent_a.items()
            if key in MUTATION_SPACE or key in ANTI_BIAS_OVERRIDES
        }
        trials.append(
            {
                "trial_id": f"g{generation:02d}_elite",
                "overrides": elite_genome,
                "runtime_overrides": _build_proxy_runtime_overrides(
                    trial_id=f"g{generation:02d}_elite",
                    genome_overrides=elite_genome,
                    generation=generation,
                    seed_checkpoint_path=context.seed_checkpoint_path,
                    proxy_symbols=context.proxy_symbols,
                ),
                "notes": "elite_survivante",
            }
        )
    if parent_b and len(trials) < generation_size:
        second_genome = {
            key: value
            for key, value in parent_b.items()
            if key in MUTATION_SPACE or key in ANTI_BIAS_OVERRIDES
        }
        trials.append(
            {
                "trial_id": f"g{generation:02d}_second",
                "overrides": second_genome,
                "runtime_overrides": _build_proxy_runtime_overrides(
                    trial_id=f"g{generation:02d}_second",
                    genome_overrides=second_genome,
                    generation=generation,
                    seed_checkpoint_path=context.seed_checkpoint_path,
                    proxy_symbols=context.proxy_symbols,
                ),
                "notes": "second_survivant",
            }
        )

    while len(trials) < generation_size:
        child_genome = _blend_parent_overrides(parent_a, parent_b, rng=rng)
        child_genome = _mutate_overrides(child_genome, rng=rng, mutation_count=4)
        child_id = f"g{generation:02d}_mix_{len(trials) + 1}"
        trials.append(
            {
                "trial_id": child_id,
                "overrides": child_genome,
                "runtime_overrides": _build_proxy_runtime_overrides(
                    trial_id=child_id,
                    genome_overrides=child_genome,
                    generation=generation,
                    seed_checkpoint_path=context.seed_checkpoint_path,
                    proxy_symbols=context.proxy_symbols,
                ),
                "notes": "crossover_mutation",
            }
        )
    return trials


def _build_proxy_sequence_config(
    *,
    sequence_id: str,
    stdout_log_path: str,
    stderr_log_path: str,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construit une sequence V4 limitee aux proxies d'une generation."""

    return {
        "sequence_id": sequence_id,
        "sequence_name": HUNT_PREFIX,
        "profiles": [PROFILE],
        "retry_limit": 1,
        "stall_timeout_seconds": 900,
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
        "catalogs": {
            ENGINE: {
                PROFILE: trials,
            }
        },
        "steps": [
            {
                "kind": "window",
                "profile": PROFILE,
                "engine": ENGINE,
                "mode": "proxy_ga",
            }
        ],
    }


def _build_full_sequence_config(
    *,
    sequence_id: str,
    stdout_log_path: str,
    stderr_log_path: str,
    trial_id: str,
    runtime_overrides: dict[str, str],
) -> dict[str, Any]:
    """Construit la sequence V4 d'un full final unique."""

    return {
        "sequence_id": sequence_id,
        "sequence_name": f"{HUNT_PREFIX}_full",
        "profiles": [PROFILE],
        "retry_limit": 1,
        "stall_timeout_seconds": 1200,
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
        "catalogs": {
            ENGINE: {
                PROFILE: [],
            }
        },
        "full_catalogs": {
            ENGINE: {
                PROFILE: {
                    trial_id: runtime_overrides,
                }
            }
        },
        "manual_finalists": [trial_id],
        "steps": [
            {
                "kind": "window",
                "profile": PROFILE,
                "engine": ENGINE,
                "mode": "full",
            }
        ],
    }


def _remote_finalists_path() -> str:
    """Retourne le chemin distant du fichier de finalistes MuZero."""

    return f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/v4_{ENGINE}_{PROFILE}_finalists.json"


def _candidate_ssh_keys() -> list[Path]:
    """Retourne les cles locales a tester avant le repli mot de passe."""

    user_home = Path.home()
    raw_candidates = [
        os.getenv("HIVE_SSH_KEY_PATH"),
        user_home / ".ssh" / "the_hive_banker_tunnel",
        user_home / ".ssh" / "id_ed25519",
        user_home / ".ssh" / "id_rsa",
    ]
    candidates: list[Path] = []
    for raw in raw_candidates:
        if not raw:
            continue
        path = Path(raw)
        if path.exists():
            candidates.append(path)
    return candidates


def _connect_client() -> paramiko.SSHClient:
    """Ouvre une session SSH vers le serveur Proxmox.

    La connexion essaye d'abord les cles locales deja configurees. Le mot de
    passe reste un repli pour les postes qui n'ont pas encore la cle projet.

    Returns:
        paramiko.SSHClient: Session SSH prete a l'emploi.

    Raises:
        RuntimeError: Si aucune authentification distante ne fonctionne.
    """

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last_error: Exception | None = None
    for key_path in _candidate_ssh_keys():
        try:
            client.connect(
                PROXMOX.HOST,
                username=PROXMOX.USER,
                key_filename=str(key_path),
                timeout=15,
            )
            return client
        except Exception as exc:  # pragma: no cover - dependant de l'environnement local
            last_error = exc
    ssh_password = str(os.getenv("HIVE_SSH_PASSWORD") or "").strip()
    if ssh_password:
        try:
            client.connect(PROXMOX.HOST, username=PROXMOX.USER, password=ssh_password, timeout=15)
            return client
        except Exception as exc:  # pragma: no cover - dependant de l'environnement local
            last_error = exc
    raise RuntimeError(f"Connexion SSH distante impossible: {last_error!s}")


def _write_remote_config(
    client: paramiko.SSHClient,
    *,
    remote_path: str,
    payload: dict[str, Any],
) -> None:
    """Ecrit une configuration JSON de sequence sur le serveur."""

    sftp = client.open_sftp()
    try:
        PROXMOX._write_remote_text_file(
            sftp,
            remote_path,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )
    finally:
        sftp.close()


def _launch_remote_sequence(
    client: paramiko.SSHClient,
    *,
    config_payload: dict[str, Any],
    context: HuntContext,
) -> dict[str, str]:
    """Lance le superviseur V4 distant avec un JSON de sequence."""

    sequence_id = str(config_payload["sequence_id"])
    stdout_log_path = str(config_payload["stdout_log_path"])
    stderr_log_path = str(config_payload["stderr_log_path"])
    remote_archive_dir = PROXMOX._prepare_remote_v4_sequence_workspace(client, sequence_id)
    remote_config_path = f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/sequence_config_{sequence_id}.json"
    _write_remote_config(client, remote_path=remote_config_path, payload=config_payload)

    supervisor_exports = PROXMOX.build_runtime_exports(PROXMOX._build_v4_supervisor_overrides())
    supervisor_prefix = f"{supervisor_exports}; " if supervisor_exports else ""
    launch_cmd = (
        "bash -lc "
        + PROXMOX.shlex.quote(
            "set -euo pipefail\n"
            f"cd {PROXMOX.REMOTE_DIR}\n"
            f"mkdir -p {PROXMOX.REMOTE_V4_SEQUENCE_DIR}\n"
            f"{supervisor_prefix}"
            "nohup env "
            f"PYTHONPATH={PROXMOX.REMOTE_DIR}/src/eva-lab:{PROXMOX.REMOTE_DIR}/src/shared "
            f"python3 {PROXMOX.REMOTE_V4_SEQUENCE_RUNNER} --config {remote_config_path} "
            f"> {stdout_log_path} 2> {stderr_log_path} < /dev/null &\n"
            "echo $!"
        )
    )
    output, error, code = PROXMOX.run_command(client, launch_cmd, timeout=30)
    if code != 0:
        raise RuntimeError(error or output or "Lancement du superviseur distant impossible.")
    pid_lines = [line.strip() for line in output.splitlines() if line.strip()]
    remote_pid = pid_lines[-1] if pid_lines else "inconnu"
    _append_log(
        context,
        (
            f"Sequence distante demarree | sequence_id={sequence_id} | pid={remote_pid} | "
            f"archive={remote_archive_dir}"
        ),
    )
    return {
        "sequence_id": sequence_id,
        "remote_pid": remote_pid,
        "remote_config_path": remote_config_path,
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
    }


def _wait_for_sequence_completion(
    *,
    sequence_id: str,
    context: HuntContext,
    poll_seconds: int,
) -> dict[str, Any]:
    """Attend la fin d'une sequence distante V4."""

    last_state: tuple[str | None, str | None, str | None] | None = None
    while True:
        status = dict(PROXMOX._fetch_remote_sequence_status() or {})
        state = str(status.get("state") or "").strip() or None
        current_trial = str(status.get("current_trial") or "").strip() or None
        status_label = str(status.get("status") or "").strip() or None
        fingerprint = (state, current_trial, status_label)
        if fingerprint != last_state:
            _append_log(
                context,
                f"Superviseur | sequence={status.get('sequence_id')} | state={state} | trial={current_trial} | status={status_label}",
            )
            last_state = fingerprint
        if str(status.get("sequence_id") or "").strip() == sequence_id and state in {"completed", "paused"}:
            return status
        time.sleep(max(5, poll_seconds))


def _download_remote_json(
    client: paramiko.SSHClient,
    remote_path: str,
    *,
    missing_ok: bool = False,
) -> dict[str, Any]:
    """Charge un JSON distant via SFTP.

    Args:
        client (paramiko.SSHClient): Session SSH ouverte.
        remote_path (str): Chemin JSON distant.
        missing_ok (bool): Autorise l'absence du fichier.

    Returns:
        dict[str, Any]: Charge utile JSON ou dictionnaire vide si absent.
    """

    sftp = client.open_sftp()
    try:
        try:
            with sftp.file(remote_path, "r") as remote_file:
                payload = json.loads(remote_file.read().decode("utf-8"))
        except OSError:
            if missing_ok:
                return {}
            raise
    finally:
        sftp.close()
    return payload if isinstance(payload, dict) else {}


def _persist_local_json(path: Path, payload: dict[str, Any]) -> None:
    """Ecrit un JSON localement pour audit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _filter_sequence_results(
    payload: dict[str, Any],
    *,
    sequence_id: str,
) -> list[dict[str, Any]]:
    """Extrait et trie uniquement les resultats d'une sequence donnee.

    Args:
        payload (dict[str, Any]): Charge utile brute des resultats proxy.
        sequence_id (str): Identifiant de sequence a retenir.

    Returns:
        list[dict[str, Any]]: Resultats de la sequence tries par score decroissant.
    """

    results = [
        item
        for item in list(payload.get("results") or [])
        if str(item.get("sequence_id") or "").strip() == sequence_id
    ]
    results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return results


def _write_remote_finalists(
    client: paramiko.SSHClient,
    *,
    best_result: dict[str, Any],
) -> dict[str, Any]:
    """Persiste un unique finaliste afin d'autoriser la fenetre `full`.

    Args:
        client (paramiko.SSHClient): Session SSH ouverte vers le serveur.
        best_result (dict[str, Any]): Meilleur survivor proxy retenu.

    Returns:
        dict[str, Any]: Charge utile finalistes ecrite cote serveur.
    """

    finalists_payload = {
        "sequence_id": f"{HUNT_PREFIX}_manual_full",
        "profile": PROFILE,
        "engine": ENGINE,
        "generated_at": _now_paris().isoformat(timespec="seconds"),
        "finalists": [best_result],
        "eligible_trial_count": 1,
    }
    _write_remote_config(client, remote_path=_remote_finalists_path(), payload=finalists_payload)
    return finalists_payload


def _run_proxy_generation(
    *,
    client: paramiko.SSHClient,
    context: HuntContext,
    generation: int,
    trials: list[dict[str, Any]],
    poll_seconds: int,
) -> list[dict[str, Any]]:
    """Execute une generation proxy complete et renvoie ses resultats tries."""

    sequence_id = f"{HUNT_PREFIX}_g{generation:02d}_{_now_tag()}"
    stdout_log_path = f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/sequence_{sequence_id}.out.log"
    stderr_log_path = f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/sequence_{sequence_id}.err.log"
    config_payload = _build_proxy_sequence_config(
        sequence_id=sequence_id,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        trials=trials,
    )
    _persist_local_json(context.hunt_dir / f"{sequence_id}_config.json", config_payload)
    launch_info = _launch_remote_sequence(client, config_payload=config_payload, context=context)
    sequence_status = _wait_for_sequence_completion(
        sequence_id=sequence_id,
        context=context,
        poll_seconds=poll_seconds,
    )
    remote_results_path = f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/v4_{ENGINE}_{PROFILE}_proxy_results.json"
    remote_finalists_path = f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/v4_{ENGINE}_{PROFILE}_finalists.json"
    proxy_results_payload = _download_remote_json(client, remote_results_path, missing_ok=True)
    finalists_payload = _download_remote_json(client, remote_finalists_path, missing_ok=True)
    _persist_local_json(context.hunt_dir / f"{sequence_id}_proxy_results.json", proxy_results_payload)
    _persist_local_json(context.hunt_dir / f"{sequence_id}_finalists.json", finalists_payload)
    _persist_local_json(context.hunt_dir / f"{sequence_id}_launch.json", launch_info)
    _persist_local_json(context.hunt_dir / f"{sequence_id}_status.json", sequence_status)
    results = _filter_sequence_results(proxy_results_payload, sequence_id=sequence_id)
    sequence_state = str(sequence_status.get("state") or "").strip().lower()
    if sequence_state == "paused" and not results:
        _append_log(
            context,
            (
                f"Generation {generation} en pause sans resultats scorables | "
                f"trial={sequence_status.get('current_trial')} | erreur={sequence_status.get('last_error')}"
            ),
        )
        return []
    _append_log(
        context,
        (
            f"Generation {generation} terminee | essais={len(results)} | "
            f"meilleur={results[0]['trial_id'] if results else 'aucun'} | "
            f"score={results[0].get('score') if results else 'n/a'}"
        ),
    )
    return results


def _launch_full_from_best(
    *,
    client: paramiko.SSHClient,
    context: HuntContext,
    best_result: dict[str, Any],
) -> dict[str, str]:
    """Lance un full 7 symboles a partir du meilleur survivor proxy."""

    trial_id = str(best_result.get("trial_id") or "best_final").strip() or "best_final"
    genome_overrides = {
        key: value
        for key, value in dict(best_result.get("runtime_overrides") or {}).items()
        if key in MUTATION_SPACE or key in ANTI_BIAS_OVERRIDES
    }
    runtime_overrides = _build_full_runtime_overrides(
        trial_id=trial_id,
        genome_overrides=genome_overrides,
        seed_checkpoint_path=context.seed_checkpoint_path,
        full_symbols=context.full_symbols,
    )
    finalists_payload = _write_remote_finalists(client, best_result=best_result)
    sequence_id = f"{HUNT_PREFIX}_full_{_now_tag()}"
    stdout_log_path = f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/sequence_{sequence_id}.out.log"
    stderr_log_path = f"{PROXMOX.REMOTE_V4_SEQUENCE_DIR}/sequence_{sequence_id}.err.log"
    config_payload = _build_full_sequence_config(
        sequence_id=sequence_id,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        trial_id=trial_id,
        runtime_overrides=runtime_overrides,
    )
    _persist_local_json(context.hunt_dir / f"{sequence_id}_config.json", config_payload)
    _persist_local_json(context.hunt_dir / f"{sequence_id}_finalists.json", finalists_payload)
    launch_info = _launch_remote_sequence(client, config_payload=config_payload, context=context)
    _persist_local_json(context.hunt_dir / f"{sequence_id}_launch.json", launch_info)
    return launch_info


def _best_promising_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Retourne le meilleur candidat assez propre pour meriter un full."""

    for item in results:
        failure_mode = str(item.get("failure_mode") or "").strip().lower()
        if failure_mode in {
            "buy_heavy",
            "sell_heavy",
            "unprofitable",
            "inactive",
            "insufficient_sample",
            "bad_exit",
            "arena_defeat",
        }:
            continue
        if bool(item.get("killed_after_precheck")):
            continue
        metrics = dict(item.get("metrics") or {})
        mechanics = dict(item.get("mechanics") or {})
        profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        return_pct = float(metrics.get("return_pct", 0.0) or 0.0)
        net_realized_pct = float(metrics.get("net_realized_pct", 0.0) or 0.0)
        directional_imbalance = float(metrics.get("directional_imbalance", 1.0) or 1.0)
        close_quality_score = float(mechanics.get("close_quality_score", 0.0) or 0.0)
        if (
            profit_factor <= 1.0
            or return_pct <= 0.0
            or net_realized_pct <= 0.0
            or directional_imbalance > 0.65
            or close_quality_score < 0.35
            or float(item.get("score", 0.0) or 0.0) <= 0.0
        ):
            continue
        return item
    return None


def _build_context(*, hours: float, random_seed: int | None) -> HuntContext:
    """Construit le contexte local de la chasse nocturne."""

    hunt_id = f"{HUNT_PREFIX}_{_now_tag()}"
    hunt_dir = ROOT / "data" / "checkpoints" / "night_hunts" / hunt_id
    hunt_dir.mkdir(parents=True, exist_ok=True)
    log_path = hunt_dir / "hunt.log"
    status_payload = json.loads(
        PROXMOX.urllib.request.urlopen(f"http://{PROXMOX.HOST}:8600/champions/status", timeout=20).read().decode("utf-8")
    )
    scalp = dict((status_payload.get("horizons") or {}).get("scalp") or {})
    live_checkpoint = dict(scalp.get("live_checkpoint") or {})
    seed_checkpoint_path = str(live_checkpoint.get("path") or "").strip()
    seed_champion_id = str(scalp.get("live_champion_id") or "").strip()
    proxy_symbols, full_symbols, coverage_generated_at = _resolve_timescaledb_universes()
    if not seed_checkpoint_path or not seed_champion_id:
        raise RuntimeError("Champion live scalp introuvable pour initialiser la chasse MuZero.")
    return HuntContext(
        hunt_id=hunt_id,
        hunt_dir=hunt_dir,
        log_path=log_path,
        seed_checkpoint_path=seed_checkpoint_path,
        seed_champion_id=seed_champion_id,
        proxy_symbols=proxy_symbols,
        full_symbols=full_symbols,
        coverage_generated_at=coverage_generated_at,
        deadline=_now_paris() + timedelta(hours=hours),
        random_seed=random_seed if random_seed is not None else int(time.time()),
    )


def _ensure_remote_idle() -> None:
    """Refuse le lancement si un entrainement distant tourne deja."""

    active_run = dict(PROXMOX._read_active_remote_run() or {})
    if bool(active_run.get("active")):
        raise RuntimeError(
            f"Un run distant est deja actif ({active_run.get('run_id')}). Stoppez-le avant de lancer la chasse."
        )


def main() -> int:
    """Point d'entree CLI.

    Returns:
        int: Code shell de retour.
    """

    parser = argparse.ArgumentParser(description="Lance une chasse MuZero multi-generations pour la nuit.")
    parser.add_argument("--hours", type=float, default=6.0, help="Budget horaire avant de lancer le full.")
    parser.add_argument(
        "--max-generations",
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help="Nombre maximal de generations proxy.",
    )
    parser.add_argument(
        "--generation-size",
        type=int,
        default=DEFAULT_GENERATION_SIZE,
        help="Nombre d'essais proxy par generation.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="Frequence de polling du superviseur distant.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed optionnelle pour reproduire les mutations.",
    )
    args = parser.parse_args()

    context = _build_context(hours=args.hours, random_seed=args.random_seed)
    _append_log(
        context,
        (
            f"Initialisation de la chasse | champion_seed={context.seed_champion_id} | "
            f"checkpoint={context.seed_checkpoint_path} | proxy={context.proxy_symbols} | "
            f"full={context.full_symbols} | couverture={context.coverage_generated_at} | "
            f"deadline={context.deadline.isoformat(timespec='seconds')}"
        ),
    )
    _ensure_remote_idle()

    client = _connect_client()
    try:
        PROXMOX._sync_remote_training_payload(client, profile_hint=None)
        _append_log(context, "Payload distant synchronise pour la chasse nocturne.")

        trials = _build_generation_one_trials(context=context)
        best_overall: dict[str, Any] | None = None
        parent_pool: list[dict[str, Any]] = []

        for generation in range(1, max(1, args.max_generations) + 1):
            if _now_paris() >= context.deadline:
                _append_log(context, "Budget horaire atteint avant la generation suivante.")
                break
            if generation > 1:
                if not parent_pool:
                    _append_log(context, "Aucun survivant exploitable, arret des generations proxy.")
                    break
                trials = _build_next_generation_trials(
                    context=context,
                    generation=generation,
                    parents=parent_pool,
                    generation_size=max(2, args.generation_size),
                )
            _append_log(
                context,
                f"Lancement generation {generation} | essais={[item['trial_id'] for item in trials]}",
            )
            generation_results = _run_proxy_generation(
                client=client,
                context=context,
                generation=generation,
                trials=trials,
                poll_seconds=max(10, args.poll_seconds),
            )
            if generation_results:
                if best_overall is None or float(generation_results[0].get("score", 0.0)) > float(
                    best_overall.get("score", 0.0)
                ):
                    best_overall = generation_results[0]
            parent_pool = _select_proxy_survivors(generation_results, count=DEFAULT_BEST_COUNT)
            if best_overall:
                _append_log(
                    context,
                    (
                        f"Meilleur cumul provisoire | trial={best_overall.get('trial_id')} | "
                        f"score={best_overall.get('score')} | failure_mode={best_overall.get('failure_mode')}"
                    ),
                )

        candidate_for_full = _best_promising_result(parent_pool or ([best_overall] if best_overall else []))
        if candidate_for_full is None:
            _append_log(context, "Aucun survivant assez propre pour lancer un full 7.")
            return 0

        _append_log(
            context,
            (
                f"Lancement du full 7 final | trial={candidate_for_full.get('trial_id')} | "
                f"score_proxy={candidate_for_full.get('score')}"
            ),
        )
        launch_info = _launch_full_from_best(
            client=client,
            context=context,
            best_result=candidate_for_full,
        )
        _append_log(
            context,
            (
                f"Full 7 en cours | sequence_id={launch_info['sequence_id']} | "
                f"stdout={launch_info['stdout_log_path']}"
            ),
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
