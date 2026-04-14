"""Orchestre une campagne GA seedee MuZero sans interrompre DreamerV3.

Ce superviseur tourne sur le poste Windows local. Il attend la fin du run
Dreamer actif, evalue des genomes mecaniques autour du champion MuZero live
courant, puis ne promeut qu'un finaliste s'il bat reellement le live.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import paramiko

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.deploy import start_training_proxmox as remote_training

LOG_DIR = ROOT_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "seeded_muzero_ga_campaign.log"

LAB_BASE_URL = f"http://{remote_training.HOST}:8600"
TRAINING_STATUS_URL = f"{LAB_BASE_URL}/training/status"
CHAMPIONS_STATUS_URL = f"{LAB_BASE_URL}/champions/status"
GA_TRIAL_URL = f"{LAB_BASE_URL}/internal/ga-trial"
GA_CAMPAIGN_URL = f"{LAB_BASE_URL}/internal/ga-seeded-campaign"

SCALP_MULTI_UNIVERSE_SYMBOLS = [
    "XAUUSD",
    "US30.cash",
    "GER40.cash",
    "EURUSD",
    "US100.cash",
    "US500.cash",
    "BTCUSD",
]

PROXY_TRIAL_COUNT = 24
FINALIST_COUNT = 4
PROXY_GAMES_PER_SYMBOL = 3
PROXY_MIN_GAMES = 10
FINAL_GAMES_PER_SYMBOL = 8
FINAL_MIN_GAMES = 24
FINAL_MIN_SYMBOLS = 7
GENERATION_INDEX = 1
REQUIRED_SYMBOL_WIN_COUNT = 4
REQUIRED_SYMBOL_COVERAGE = len(SCALP_MULTI_UNIVERSE_SYMBOLS)

PARAMETER_SPECS: dict[str, dict[str, Any]] = {
    "MUZERO_ENTRY_MIN_ADX": {"type": "float", "default": 22.0, "min": 10.0, "max": 40.0, "step": 2.0},
    "MUZERO_ENTRY_TREND_ADX": {"type": "float", "default": 28.0, "min": 16.0, "max": 45.0, "step": 2.0},
    "MUZERO_ENTRY_REQUIRE_VWAP_ALIGNMENT": {"type": "bool", "default": True},
    "MUZERO_ENTRY_REQUIRE_OBV_CONFIRMATION": {"type": "bool", "default": True},
    "MUZERO_ENTRY_ALLOW_TREND_FALLBACK": {"type": "bool", "default": True},
    "MUZERO_ACTIVITY_MIN_ENTRIES": {"type": "int", "default": 12, "min": 4, "max": 24, "step": 2},
    "MUZERO_ACTIVITY_INACTIVE_EPISODE_PENALTY": {"type": "float", "default": 0.40, "min": 0.05, "max": 1.20, "step": 0.05},
    "MUZERO_ACTIVITY_INSUFFICIENT_ENTRIES_PENALTY": {"type": "float", "default": 0.25, "min": 0.05, "max": 1.00, "step": 0.05},
    "MUZERO_DIRECTIONAL_MIN_ENTRY_SHARE": {"type": "float", "default": 0.20, "min": 0.10, "max": 0.35, "step": 0.01},
    "MUZERO_DIRECTIONAL_MAX_IMBALANCE": {"type": "float", "default": 0.60, "min": 0.45, "max": 0.80, "step": 0.02},
    "MUZERO_DIRECTIONAL_IMBALANCE_PENALTY": {"type": "float", "default": 0.40, "min": 0.05, "max": 1.50, "step": 0.05},
    "MUZERO_HOLD_STALE_PENALTY_AFTER_STEPS": {"type": "int", "default": 12, "min": 4, "max": 32, "step": 2},
    "MUZERO_HOLD_STALE_PENALTY": {"type": "float", "default": 0.08, "min": 0.0, "max": 0.40, "step": 0.02},
    "MUZERO_HOLD_TREND_PENALTY": {"type": "float", "default": 0.02, "min": 0.0, "max": 0.20, "step": 0.01},
    "MUZERO_HOLD_RANGE_PENALTY": {"type": "float", "default": 0.04, "min": 0.0, "max": 0.25, "step": 0.01},
    "MUZERO_PYRAMID_MAX_ADDITIONS": {"type": "int", "default": 2, "min": 0, "max": 4, "step": 1},
    "MUZERO_PYRAMID_MIN_PROFIT_TO_ADD": {"type": "float", "default": 0.20, "min": 0.05, "max": 0.80, "step": 0.05},
    "MUZERO_PYRAMID_REWARD_BONUS": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.50, "step": 0.02},
    "MUZERO_REWARD_PYRAMID_FAILURE_PENALTY": {"type": "float", "default": 0.20, "min": 0.0, "max": 0.80, "step": 0.05},
    "MUZERO_REWARD_PYRAMID_NEGATIVE_EXIT_PENALTY": {"type": "float", "default": 0.30, "min": 0.0, "max": 1.00, "step": 0.05},
    "MUZERO_SPLIT_MAX_SPLITS": {"type": "int", "default": 2, "min": 0, "max": 4, "step": 1},
    "MUZERO_SPLIT_MIN_TRADE_RETURN": {"type": "float", "default": 0.10, "min": 0.02, "max": 0.40, "step": 0.02},
    "MUZERO_SPLIT_MIN_REALIZED_PCT": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.25, "step": 0.01},
    "MUZERO_SPLIT_FAILURE_PENALTY": {"type": "float", "default": 0.20, "min": 0.0, "max": 0.80, "step": 0.05},
    "MUZERO_SLBE_ACTIVATION_RETURN": {"type": "float", "default": 0.15, "min": 0.02, "max": 0.40, "step": 0.02},
    "MUZERO_SLBE_BONUS": {"type": "float", "default": 0.10, "min": 0.0, "max": 0.40, "step": 0.02},
    "MUZERO_SLBE_EXIT_BONUS": {"type": "float", "default": 0.12, "min": 0.0, "max": 0.40, "step": 0.02},
    "MUZERO_CLOSE_WINNER_THRESHOLD": {"type": "float", "default": 0.15, "min": 0.05, "max": 0.40, "step": 0.02},
    "MUZERO_CLOSE_STRONG_WINNER_THRESHOLD": {"type": "float", "default": 0.30, "min": 0.10, "max": 0.80, "step": 0.05},
    "MUZERO_CLOSE_TP_LIKE_THRESHOLD": {"type": "float", "default": 0.45, "min": 0.20, "max": 1.20, "step": 0.05},
}


def _append_log(message: str) -> None:
    """Journalise un message local de campagne."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _fetch_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    """Charge un document JSON via HTTP."""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Lecture impossible de {url}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON invalide depuis {url}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Reponse inattendue depuis {url}")
    return payload


def _post_json(url: str, payload: dict[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    """Envoie un payload JSON et decode la reponse."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST impossible vers {url}: {exc}") from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON invalide depuis {url}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"Reponse inattendue depuis {url}")
    return body


def _fetch_training_run() -> dict[str, Any]:
    """Retourne le bloc `run` du training distant."""

    payload = _fetch_json(TRAINING_STATUS_URL)
    run = payload.get("run") or {}
    return run if isinstance(run, dict) else {}


def _dreamer_is_active(run: dict[str, Any]) -> bool:
    """Retourne vrai si le run courant est un Dreamer actif."""

    return bool(run.get("active")) and str(run.get("engine") or "").strip().lower() == "dreamer"


def _wait_for_dreamer_completion(*, poll_seconds: int = 60, timeout_hours: int = 24) -> None:
    """Attend la fin du Dreamer actif avant de lancer la campagne seedee."""

    deadline = time.time() + timeout_hours * 3600
    announced = False
    while time.time() < deadline:
        run = _fetch_training_run()
        if _dreamer_is_active(run):
            if not announced:
                _append_log(
                    f"Dreamer actif detecte ({run.get('run_id')}); attente de fin avant la campagne GA seedee."
                )
                announced = True
            time.sleep(poll_seconds)
            continue
        if bool(run.get("active")):
            _append_log(
                f"Un autre run GPU reste actif ({run.get('engine')} / {run.get('trigger')}); attente."
            )
            time.sleep(poll_seconds)
            continue
        return
    raise TimeoutError("Dreamer est reste actif trop longtemps; campagne GA seedee annulee.")


def _normalize_remote_path(raw_path: str | None) -> str | None:
    """Convertit un chemin conteneur ou relatif vers le chemin hote Proxmox."""

    path = str(raw_path or "").strip()
    if not path:
        return None
    if path.startswith("/app/eva-lab/"):
        return path.replace("/app/eva-lab", remote_training.REMOTE_DIR, 1)
    if path.startswith("/home/"):
        return path
    return str(Path(remote_training.REMOTE_DIR) / path)


def _open_remote_client() -> paramiko.SSHClient:
    """Ouvre une session SSH vers le serveur GPU."""

    ssh_password, _ = remote_training._require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(remote_training.HOST, username=remote_training.USER, password=ssh_password, timeout=20)
    return client


def _read_remote_json(path: str) -> dict[str, Any]:
    """Lit un fichier JSON sur le serveur distant."""

    remote_path = _normalize_remote_path(path)
    if not remote_path:
        raise RuntimeError("Chemin JSON distant absent.")
    client = _open_remote_client()
    try:
        command = (
            "python3 - <<'PY'\n"
            "import json\n"
            "from pathlib import Path\n"
            f"print(Path({remote_path!r}).read_text(encoding='utf-8'))\n"
            "PY"
        )
        output, error, code = remote_training.run_command(client, command, timeout=60)
        if code != 0:
            raise RuntimeError(error or output or f"Lecture distante impossible pour {remote_path}.")
        payload = json.loads(output)
    finally:
        client.close()
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON inattendu pour {remote_path}.")
    return payload


def _load_local_env_defaults() -> dict[str, str]:
    """Charge le fichier `.env` local pour extraire les valeurs live courantes."""

    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _build_base_genome() -> dict[str, Any]:
    """Construit le genome seed a partir des valeurs live courantes."""

    env_values = _load_local_env_defaults()
    genome: dict[str, Any] = {}
    for name, spec in PARAMETER_SPECS.items():
        raw_value = env_values.get(name, os.getenv(name, spec["default"]))
        if spec["type"] == "bool":
            genome[name] = str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
        elif spec["type"] == "int":
            genome[name] = int(float(raw_value))
        else:
            genome[name] = float(raw_value)
    return genome


def _mutate_gene(randomizer: random.Random, value: Any, spec: dict[str, Any]) -> Any:
    """Mutile un gene dans des bornes prudentes."""

    gene_type = spec["type"]
    if gene_type == "bool":
        return (not bool(value)) if randomizer.random() < 0.35 else bool(value)

    step = float(spec.get("step", 1.0))
    lower = float(spec["min"])
    upper = float(spec["max"])
    span = max(step, (upper - lower) * 0.18)
    mutated = float(value) + randomizer.uniform(-span, span)
    mutated = min(max(mutated, lower), upper)
    if gene_type == "int":
        return int(round(mutated / step) * step)
    rounded = round(mutated / step) * step
    return round(rounded, 6)


def _build_mutated_genome(base_genome: dict[str, Any], *, seed: int, intensity: float = 1.0) -> dict[str, Any]:
    """Construit un genome derive du seed live avec mutation bornee."""

    randomizer = random.Random(seed)
    mutated: dict[str, Any] = {}
    mutation_count = 0
    for name, spec in PARAMETER_SPECS.items():
        current_value = base_genome[name]
        if randomizer.random() <= min(0.75, 0.35 + 0.15 * intensity):
            mutated[name] = _mutate_gene(randomizer, current_value, spec)
            mutation_count += 1
        else:
            mutated[name] = current_value
    if mutation_count == 0:
        first_key = next(iter(PARAMETER_SPECS))
        mutated[first_key] = _mutate_gene(randomizer, base_genome[first_key], PARAMETER_SPECS[first_key])
    return mutated


def _compute_seeded_ga_fitness(metrics: dict[str, Any]) -> float:
    """Calcule le score proxy exact de la campagne."""

    def _value(name: str, default: float = 0.0) -> float:
        try:
            return float(metrics.get(name, default) or default)
        except (TypeError, ValueError):
            return default

    return (
        8.0 * _value("return_pct")
        + 18.0 * max(0.0, _value("profit_factor") - 1.0)
        + 120.0 * _value("expectancy_pct")
        + 0.08 * _value("win_rate")
        + 0.06 * _value("positive_episode_rate")
        - 1.5 * _value("max_drawdown_pct")
        + 6.0 * _value("close_quality_score")
        + 4.0 * _value("split_efficiency")
        + 4.0 * _value("pyramid_efficiency")
        + 4.0 * _value("slbe_capture_rate")
        - 8.0 * _value("hold_drag_score")
        - 10.0 * _value("directional_imbalance")
    )


def _hard_reject_trial(metrics: dict[str, Any]) -> str | None:
    """Retourne la raison d'un rejet proxy immediat."""

    directional_bias = str(metrics.get("directional_bias") or "").strip().lower()
    if directional_bias in {"inactive", ""}:
        return "inactive"
    if directional_bias in {"sell_heavy", "buy_heavy"}:
        return directional_bias
    if float(metrics.get("profit_factor", 0.0) or 0.0) <= 1.0:
        return "profit_factor"
    if float(metrics.get("return_pct", 0.0) or 0.0) <= 0.0:
        return "return_pct"
    if float(metrics.get("net_realized_pct", 0.0) or 0.0) <= 0.0:
        return "net_realized_pct"
    if float(metrics.get("directional_imbalance", 1.0) or 1.0) > 0.70:
        return "directional_imbalance"
    if float(metrics.get("close_quality_score", 0.0) or 0.0) < 0.40:
        return "close_quality_score"
    return None


def _publish_campaign_state(state: dict[str, Any]) -> None:
    """Persiste l'etat de campagne dans le Lab distant."""

    _post_json(GA_CAMPAIGN_URL, state, timeout=30)


def _publish_ga_trial(payload: dict[str, Any]) -> None:
    """Persiste un resultat de trial dans TimeDB via le Lab."""

    _post_json(GA_TRIAL_URL, payload, timeout=30)


def _build_seed_reference_from_status(muzero_status: dict[str, Any]) -> dict[str, Any]:
    """Valide et extrait la reference seed depuis `/champions/status`."""

    promotion_state = str(muzero_status.get("promotion_state") or "").strip().lower()
    selection = str(muzero_status.get("selection") or "").strip().lower()
    live_champion_id = str(muzero_status.get("live_champion_id") or "").strip()
    checkpoint_info = dict(muzero_status.get("champion_checkpoint") or {})
    seed_checkpoint_path = str(checkpoint_info.get("path") or "").strip()
    artifact_compatibility = dict(muzero_status.get("artifact_compatibility") or {})

    if promotion_state != "promoted":
        raise RuntimeError("Aucun champion MuZero scalp promu n'est disponible pour le seed GA.")
    if selection != "champion":
        raise RuntimeError("Le seed GA MuZero refuse les artefacts live non champions.")
    if not artifact_compatibility.get("allowed", False):
        reason = str(artifact_compatibility.get("reason") or "artifact_incompatible").strip()
        raise RuntimeError(
            f"Le champion MuZero scalp est incompatible et ne peut pas servir de seed: {reason}"
        )
    if not live_champion_id or not seed_checkpoint_path:
        raise RuntimeError("Champion MuZero live introuvable pour initialiser la campagne seedee.")

    return {
        "champion_id": live_champion_id,
        "checkpoint_path": seed_checkpoint_path,
        "metrics": dict((muzero_status.get("promotion_gate") or {}).get("metrics") or {}),
        "mechanics_profile_version": str(muzero_status.get("mechanics_profile_version") or "").strip() or None,
        "feature_profile": str(muzero_status.get("feature_profile") or "").strip() or None,
        "artifact_compatibility": artifact_compatibility,
        "checkpoint_schema_version": muzero_status.get("checkpoint_schema_version"),
        "resume_source": str(muzero_status.get("resume_source") or "").strip() or None,
        "lineage": dict(muzero_status.get("lineage") or {}),
        "seed_parent_champion_id": str(muzero_status.get("seed_parent_champion_id") or "").strip() or None,
    }


def _extract_seed_reference() -> dict[str, Any]:
    """Charge la baseline live MuZero et ses metriques de reference."""

    payload = _fetch_json(CHAMPIONS_STATUS_URL, timeout=30)
    muzero_status = dict(((payload.get("engines") or {}).get("muzero") or {}).get("scalp") or {})
    return _build_seed_reference_from_status(muzero_status)


def _build_seeded_overrides(
    *,
    trigger: str,
    trial_id: str,
    gate_profile: str,
    ga_status: str,
    trial_mode: str,
    cost_profile: str,
    genome: dict[str, Any],
    campaign_id: str,
    seed_reference: dict[str, Any],
    arena_games_per_symbol: int,
    arena_min_games: int,
    arena_min_symbols: int,
) -> dict[str, str]:
    """Construit les overrides d'un run seede mecanique."""

    overrides = remote_training._build_muzero_scalp_multi_universe_full_overrides(SCALP_MULTI_UNIVERSE_SYMBOLS)
    overrides.update(
        {
            "TRAINING_RUN_TRIGGER": trigger,
            "TRAINING_TRIAL_MODE": trial_mode,
            "TRAINING_TRIAL_COST_PROFILE": cost_profile,
            "TRAINING_GA_STATUS": ga_status,
            "TRAINING_GA_GENERATION": str(GENERATION_INDEX),
            "TRAINING_GA_TRIAL": trial_id,
            "TRAINING_GA_CAMPAIGN_ID": campaign_id,
            "TRAINING_GA_SCOPE": "mechanics_only",
            "TRAINING_GA_PARENT_CHAMPION_ID": str(seed_reference["champion_id"]),
            "TRAINING_GA_DEFER_PROMOTION": "1",
            "TRAINING_GA_SEED_CHECKPOINT_PATH": str(seed_reference["checkpoint_path"]),
            "TRAINING_RESUME_CHECKPOINT_PATH": str(seed_reference["checkpoint_path"]),
            "TRAINING_RESUME_STEP": "0",
            "TRAINING_GA_GENOME_JSON": json.dumps(genome, ensure_ascii=True, separators=(",", ":")),
            "TRAINING_GATE_PROFILE": gate_profile,
            "ARENA_GAMES_PER_SYMBOL": str(arena_games_per_symbol),
            "ARENA_MIN_GAMES": str(arena_min_games),
            "ARENA_MIN_SYMBOLS": str(arena_min_symbols),
        }
    )
    for name, value in genome.items():
        overrides[name] = "1" if isinstance(value, bool) and value else ("0" if isinstance(value, bool) else str(value))
    return overrides


def _wait_for_run_completion_with_summary(run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attend un run distant et charge son resume terminal."""

    completion = remote_training._wait_for_remote_run_completion(run_id, poll_interval_seconds=60)
    summary_path = str(completion.get("terminal_summary_path") or "").strip()
    if not summary_path:
        raise RuntimeError(f"Resume terminal absent pour le run {run_id}.")
    summary = _read_remote_json(summary_path)
    return completion, summary


def _launch_seeded_run(overrides: dict[str, str], *, expected_trigger: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Lance un run seede, attend sa fin et retourne son resume."""

    previous_run_id = str((_fetch_training_run().get("run_id") or "")).strip() or None
    client = _open_remote_client()
    try:
        remote_training._sync_remote_training_payload(client, profile_hint=None)
        remote_training._launch_remote_training_process(client, overrides)
    finally:
        client.close()
    run_id = remote_training._wait_for_remote_run_start(previous_run_id, expected_trigger, timeout_seconds=300)
    completion, summary = _wait_for_run_completion_with_summary(run_id)
    return run_id, completion, summary


def _build_trial_record(
    *,
    trial_id: str,
    campaign_id: str,
    ga_status: str,
    trial_mode: str,
    genome: dict[str, Any],
    run_id: str,
    summary: dict[str, Any],
    fitness_score: float,
    rejection_reason: str | None,
) -> dict[str, Any]:
    """Construit la charge persistable d'un trial GA."""

    promotion_gate = dict(summary.get("promotion_gate") or {})
    latest_verdict = dict(summary.get("latest_verdict") or {})
    return {
        "trial_id": trial_id,
        "engine": "muzero",
        "sequence_id": str(summary.get("sequence_id") or "").strip() or None,
        "profile": "seeded_muzero_ga",
        "horizon": "scalp",
        "family": str(summary.get("family") or "").strip() or None,
        "feature_profile": str(summary.get("feature_profile") or "").strip() or None,
        "mechanics_profile_version": str(summary.get("mechanics_profile_version") or "").strip() or None,
        "ga_generation": GENERATION_INDEX,
        "ga_trial": trial_id,
        "trial_mode": trial_mode,
        "trial_cost_profile": "proxy" if trial_mode == "proxy_ga" else "full",
        "params": genome,
        "fitness_score": fitness_score,
        "failure_mode": rejection_reason or str(promotion_gate.get("failure_mode") or latest_verdict.get("failure_mode") or "").strip() or None,
        "run_id": run_id,
        "dataset_id": str(summary.get("dataset_id") or "").strip() or None,
        "finished_at": summary.get("terminal_at"),
        "campaign_id": campaign_id,
        "resume_source": str(summary.get("resume_source") or "").strip() or None,
        "artifact_compatibility": dict(summary.get("artifact_compatibility") or {}),
        "checkpoint_schema_version": summary.get("checkpoint_schema_version"),
        "lineage": dict(summary.get("lineage") or {}),
        "seed_parent_champion_id": (
            str(summary.get("seed_parent_champion_id") or "").strip()
            or str((summary.get("lineage") or {}).get("parent_champion_id") or "").strip()
            or None
        ),
        "promotion_gate": promotion_gate,
        "live_comparison": dict(summary.get("live_comparison") or {}),
        "metrics": dict(summary.get("metrics") or {}),
        "metrics_by_symbol": dict(summary.get("metrics_by_symbol") or {}),
        "payload": summary,
    }


def _select_proxy_finalists(proxy_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retourne les quatre meilleurs proxy non rejetes."""

    eligible = [result for result in proxy_results if not result.get("hard_reject_reason")]
    eligible.sort(key=lambda item: float(item.get("fitness_score") or 0.0), reverse=True)
    return eligible[:FINALIST_COUNT]


def _select_campaign_winner(final_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choisit le meilleur finaliste admissible face au live courant."""

    candidates: list[dict[str, Any]] = []
    for result in final_results:
        summary = dict(result.get("summary") or {})
        promotion_gate = dict(summary.get("promotion_gate") or {})
        live_comparison = dict(summary.get("live_comparison") or {})
        artifact_compatibility = dict(summary.get("artifact_compatibility") or {})
        if not artifact_compatibility.get("allowed", False):
            continue
        if not promotion_gate.get("allowed", False):
            continue
        if not live_comparison.get("allowed", False):
            continue
        symbol_wins = dict(live_comparison.get("symbol_wins_vs_live") or {})
        compared_symbols = list(symbol_wins.get("symbols_compared") or [])
        win_count = int(symbol_wins.get("win_count") or 0)
        if len(compared_symbols) < REQUIRED_SYMBOL_COVERAGE:
            continue
        if win_count < REQUIRED_SYMBOL_WIN_COUNT:
            continue
        candidates.append(result)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            int((((item.get("summary") or {}).get("live_comparison") or {}).get("symbol_wins_vs_live") or {}).get("win_count") or 0),
            float(item.get("fitness_score") or 0.0),
        ),
        reverse=True,
    )
    return candidates[0]


def _promote_remote_winner(seed_reference: dict[str, Any], winner: dict[str, Any], campaign_id: str) -> dict[str, Any]:
    """Declenche la promotion distante du finaliste retenu."""

    summary = dict(winner.get("summary") or {})
    battle_report_path = _normalize_remote_path(str(summary.get("battle_report_path") or ""))
    challenger_path = _normalize_remote_path(str(summary.get("challenger_path") or ""))
    latest_checkpoint = _normalize_remote_path(str(summary.get("latest_checkpoint") or ""))
    if not battle_report_path or not challenger_path:
        raise RuntimeError("Artefacts du gagnant incomplets pour la promotion distante.")

    metadata = {
        "parent_champion_id": seed_reference.get("champion_id"),
        "seed_checkpoint_path": seed_reference.get("checkpoint_path"),
        "seed_checkpoint_schema_version": seed_reference.get("checkpoint_schema_version"),
        "seed_artifact_compatibility": dict(seed_reference.get("artifact_compatibility") or {}),
        "seed_resume_source": seed_reference.get("resume_source"),
        "seed_lineage": dict(seed_reference.get("lineage") or {}),
        "ga_campaign_id": campaign_id,
        "ga_scope": "mechanics_only",
        "ga_generation": GENERATION_INDEX,
        "ga_trial": winner.get("trial_id"),
        "ga_genome": winner.get("genome"),
        "ga_fitness": winner.get("fitness_score"),
        "symbol_wins_vs_live": dict(((summary.get("live_comparison") or {}).get("symbol_wins_vs_live") or {})),
    }

    script = (
        "import json\n"
        "from pathlib import Path\n"
        "from eva_lab.champion_promoter import ChampionPromoter\n"
        f"report_wrapper = json.loads(Path({battle_report_path!r}).read_text(encoding='utf-8'))\n"
        "battle_report = dict(report_wrapper.get('battle_report') or report_wrapper)\n"
        "promoter = ChampionPromoter("
        f"weights_dir={str(Path(remote_training.REMOTE_DIR) / 'data' / 'muzero' / 'weights')!r}, "
        f"results_dir={str(Path(remote_training.REMOTE_DIR) / 'data' / 'muzero' / 'results')!r})\n"
        "result = promoter.promote_muzero_challenger("
        f"challenger_path={challenger_path!r}, "
        "horizon='scalp', "
        "battle_report=battle_report, "
        f"training_metrics=json.loads({json.dumps(json.dumps(dict(summary.get('training_metrics') or {}), ensure_ascii=True))}), "
        f"latest_checkpoint={latest_checkpoint!r}, "
        f"challenger_id={str(summary.get('latest_candidate') or '')!r}, "
        "gate_profile='standard', "
        f"promotion_metadata=json.loads({json.dumps(json.dumps(metadata, ensure_ascii=True))}), "
        f"live_comparison_override=json.loads({json.dumps(json.dumps(dict(summary.get('live_comparison') or {}), ensure_ascii=True))}))\n"
        "print(json.dumps(result, ensure_ascii=True))\n"
    )
    command = (
        "cd "
        + remote_training.REMOTE_DIR
        + " && PYTHONPATH="
        + f"{remote_training.REMOTE_DIR}/src/eva-lab:{remote_training.REMOTE_DIR}/src/shared "
        + "python3 - <<'PY'\n"
        + script
        + "PY"
    )

    client = _open_remote_client()
    try:
        output, error, code = remote_training.run_command(client, command, timeout=120)
        if code != 0:
            raise RuntimeError(error or output or "Promotion distante impossible.")
        promotion_result = json.loads(output.strip())
    finally:
        client.close()
    if not isinstance(promotion_result, dict):
        raise RuntimeError("Reponse de promotion distante invalide.")
    return promotion_result


def main() -> int:
    """Execute la campagne GA seedee MuZero en serie apres Dreamer."""

    _append_log("Debut de la campagne GA seedee MuZero.")
    _wait_for_dreamer_completion()

    campaign_id = f"seeded_muzero_ga_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        seed_reference = _extract_seed_reference()
    except RuntimeError as exc:
        blocking_reason = str(exc).strip() or "seed_incompatible"
        _append_log(f"Campagne seedee bloquee: {blocking_reason}")
        _publish_campaign_state(
            {
                "campaign_id": campaign_id,
                "status": "seed_blocked",
                "scope": "mechanics_only",
                "reason": blocking_reason,
                "generation": GENERATION_INDEX,
                "trial_count": 0,
                "finalists": [],
                "selected_challenger_id": None,
                "selected_trial_id": None,
                "promotion_state": "blocked",
                "universe": SCALP_MULTI_UNIVERSE_SYMBOLS,
            }
        )
        return 0

    base_genome = _build_base_genome()
    _publish_campaign_state(
        {
            "campaign_id": campaign_id,
            "status": "seed_ready",
            "scope": "mechanics_only",
            "seed_champion_id": seed_reference.get("champion_id"),
            "seed_checkpoint_path": seed_reference.get("checkpoint_path"),
            "seed_metrics": seed_reference.get("metrics"),
            "seed_mechanics_profile_version": seed_reference.get("mechanics_profile_version"),
            "seed_feature_profile": seed_reference.get("feature_profile"),
            "seed_checkpoint_schema_version": seed_reference.get("checkpoint_schema_version"),
            "seed_artifact_compatibility": seed_reference.get("artifact_compatibility"),
            "seed_resume_source": seed_reference.get("resume_source"),
            "seed_lineage": seed_reference.get("lineage"),
            "seed_parent_champion_id": seed_reference.get("seed_parent_champion_id"),
            "generation": GENERATION_INDEX,
            "trial_count": 0,
            "finalists": [],
            "selected_challenger_id": None,
            "promotion_state": "candidate_only",
            "universe": SCALP_MULTI_UNIVERSE_SYMBOLS,
            "base_genome": base_genome,
        }
    )

    proxy_results: list[dict[str, Any]] = []
    for index in range(1, PROXY_TRIAL_COUNT + 1):
        trial_id = f"seeded_proxy_g{GENERATION_INDEX:02d}_t{index:02d}"
        trigger = f"manual_muzero_seeded_ga_proxy_{GENERATION_INDEX:02d}_{index:02d}"
        genome = _build_mutated_genome(base_genome, seed=index)
        _append_log(f"Lancement du proxy {trial_id}.")
        overrides = _build_seeded_overrides(
            trigger=trigger,
            trial_id=trial_id,
            gate_profile="gold_demo",
            ga_status="proxy",
            trial_mode="proxy_ga",
            cost_profile="proxy",
            genome=genome,
            campaign_id=campaign_id,
            seed_reference=seed_reference,
            arena_games_per_symbol=PROXY_GAMES_PER_SYMBOL,
            arena_min_games=PROXY_MIN_GAMES,
            arena_min_symbols=3,
        )
        run_id, _completion, summary = _launch_seeded_run(overrides, expected_trigger=trigger)
        metrics = dict(summary.get("metrics") or {})
        fitness_score = _compute_seeded_ga_fitness(metrics)
        rejection_reason = _hard_reject_trial(metrics)
        _publish_ga_trial(
            _build_trial_record(
                trial_id=trial_id,
                campaign_id=campaign_id,
                ga_status="proxy",
                trial_mode="proxy_ga",
                genome=genome,
                run_id=run_id,
                summary=summary,
                fitness_score=fitness_score,
                rejection_reason=rejection_reason,
            )
        )
        proxy_results.append(
            {
                "trial_id": trial_id,
                "run_id": run_id,
                "genome": genome,
                "fitness_score": fitness_score,
                "hard_reject_reason": rejection_reason,
                "summary": summary,
            }
        )
        _publish_campaign_state(
            {
                "campaign_id": campaign_id,
                "status": "proxy_running",
                "trial_count": len(proxy_results),
                "latest_proxy_trial": {
                    "trial_id": trial_id,
                    "run_id": run_id,
                    "fitness_score": fitness_score,
                    "hard_reject_reason": rejection_reason,
                },
            }
    )

    finalists = _select_proxy_finalists(proxy_results)
    if not finalists:
        _append_log("Aucun proxy compatible n'a produit de finaliste admissible.")
        _publish_campaign_state(
            {
                "campaign_id": campaign_id,
                "status": "completed",
                "promotion_state": "blocked",
                "selected_challenger_id": None,
                "selected_trial_id": None,
                "reason": "no_proxy_finalist",
                "trial_count": len(proxy_results),
            }
        )
        return 0

    _append_log(f"{len(finalists)} finalistes retenus apres la phase proxy.")
    _publish_campaign_state(
        {
            "campaign_id": campaign_id,
            "status": "finalist_selection",
            "trial_count": len(proxy_results),
            "finalists": [
                {
                    "trial_id": item.get("trial_id"),
                    "run_id": item.get("run_id"),
                    "fitness_score": item.get("fitness_score"),
                }
                for item in finalists
            ],
        }
    )

    final_results: list[dict[str, Any]] = []
    for rank, finalist in enumerate(finalists, start=1):
        trial_id = f"seeded_final_g{GENERATION_INDEX:02d}_r{rank:02d}"
        trigger = f"manual_muzero_seeded_ga_final_{GENERATION_INDEX:02d}_{rank:02d}"
        _append_log(f"Lancement du finaliste {trial_id} derive de {finalist.get('trial_id')}.")
        overrides = _build_seeded_overrides(
            trigger=trigger,
            trial_id=trial_id,
            gate_profile="standard",
            ga_status="final",
            trial_mode="full",
            cost_profile="full",
            genome=dict(finalist.get("genome") or {}),
            campaign_id=campaign_id,
            seed_reference=seed_reference,
            arena_games_per_symbol=FINAL_GAMES_PER_SYMBOL,
            arena_min_games=FINAL_MIN_GAMES,
            arena_min_symbols=FINAL_MIN_SYMBOLS,
        )
        run_id, _completion, summary = _launch_seeded_run(overrides, expected_trigger=trigger)
        metrics = dict(summary.get("metrics") or {})
        fitness_score = _compute_seeded_ga_fitness(metrics)
        _publish_ga_trial(
            _build_trial_record(
                trial_id=trial_id,
                campaign_id=campaign_id,
                ga_status="final",
                trial_mode="full",
                genome=dict(finalist.get("genome") or {}),
                run_id=run_id,
                summary=summary,
                fitness_score=fitness_score,
                rejection_reason=None,
            )
        )
        final_results.append(
            {
                "trial_id": trial_id,
                "run_id": run_id,
                "genome": dict(finalist.get("genome") or {}),
                "fitness_score": fitness_score,
                "summary": summary,
            }
        )

    winner = _select_campaign_winner(final_results)
    if winner is None:
        _append_log("Aucun finaliste ne bat le live actuel; aucune promotion.")
        _publish_campaign_state(
            {
                "campaign_id": campaign_id,
                "status": "completed",
                "promotion_state": "blocked",
                "selected_challenger_id": None,
                "selected_trial_id": None,
                "reason": "no_candidate_passed_live_comparison",
            }
        )
        return 0

    promotion_result = _promote_remote_winner(seed_reference, winner, campaign_id)
    _append_log(
        f"Promotion finale du trial {winner.get('trial_id')} -> statut={promotion_result.get('status')}."
    )
    _publish_campaign_state(
        {
            "campaign_id": campaign_id,
            "status": "completed",
            "promotion_state": str(promotion_result.get("status") or "unknown"),
            "selected_challenger_id": (
                str(promotion_result.get("challenger_id") or "")
                or str((promotion_result.get("battle_report") or {}).get("challenger", {}).get("id") or "")
                or None
            ),
            "selected_trial_id": winner.get("trial_id"),
            "promotion_result": promotion_result,
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _append_log(f"Echec de la campagne GA seedee MuZero: {exc}")
        raise
