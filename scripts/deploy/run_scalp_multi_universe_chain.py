"""Orchestre la chaine `MuZero -> Dreamer -> GNN` sur l'univers scalp 7 symboles.

Ce script est destine a tourner sur le poste Windows local. Il ne remplace pas
le live actuel; il pilote seulement les lancements distants en serie pour
eviter les collisions GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import paramiko

ROOT_DIR = Path(__file__).resolve().parents[2]
START_TRAINING_SCRIPT = ROOT_DIR / "scripts" / "deploy" / "start_training_proxmox.py"
LOG_DIR = ROOT_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "scalp_multi_universe_chain.log"

TRAINING_STATUS_URL = "http://192.168.1.6:8600/training/status"
GNN_REFRESH_URL = "http://192.168.1.6:8600/gnn/refresh"

MUZERO_TRIGGER = "manual_muzero_scalp_multi_universe_full"
DREAMER_TRIGGER = "manual_dreamer_scalp_multi_universe_full"

GPU_HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
GPU_USER = os.getenv("HIVE_SSH_USER", "aza")
GPU_PASSWORD = os.getenv("HIVE_SSH_PASSWORD")

SCALP_MULTI_UNIVERSE_SYMBOLS = [
    "EURUSD",
    "XAUUSD",
    "GBPUSD",
    "USDJPY",
    "US30.cash",
    "GER40.cash",
    "US500.cash",
]

GNN_REFRESH_PAYLOAD = {
    "symbols": SCALP_MULTI_UNIVERSE_SYMBOLS,
    "focus_symbol": "XAUUSD",
    "context_symbols": [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "US30.cash",
        "GER40.cash",
        "US500.cash",
    ],
    "deployment_class": "consultative",
    "epochs": 300,
    "batch_size": 32,
    "checkpoint_every": 25,
    "max_symbols": 7,
}


def _append_log(message: str) -> None:
    """Ecrit un message dans le journal local et sur stdout.

    Args:
        message (str): Ligne a journaliser.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _fetch_json(url: str, *, timeout: int = 30) -> dict[str, object]:
    """Charge un JSON HTTP en utilisant uniquement la bibliotheque standard.

    Args:
        url (str): Endpoint a interroger.
        timeout (int): Duree maximale d'attente en secondes.

    Returns:
        dict[str, object]: Corps JSON decode.

    Raises:
        RuntimeError: Si l'endpoint ne repond pas ou renvoie un JSON invalide.
    """

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Endpoint indisponible: {url} ({exc})") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Reponse JSON invalide depuis {url}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Reponse inattendue depuis {url}")
    return payload


def _post_json(url: str, payload: dict[str, object], *, timeout: int = 30) -> dict[str, object]:
    """Envoie un payload JSON et retourne la reponse decodee.

    Args:
        url (str): Endpoint cible.
        payload (dict[str, object]): Corps JSON.
        timeout (int): Duree maximale d'attente en secondes.

    Returns:
        dict[str, object]: Reponse JSON decodee.

    Raises:
        RuntimeError: Si l'appel HTTP echoue.
    """

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
        raise RuntimeError(f"Echec de POST sur {url}: {exc}") from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Reponse JSON invalide depuis {url}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"Reponse inattendue depuis {url}")
    return body


def _fetch_training_run() -> dict[str, object]:
    """Retourne l'etat du run d'entrainement courant.

    Returns:
        dict[str, object]: Bloc `run` issu de `/training/status`.
    """

    payload = _fetch_json(TRAINING_STATUS_URL)
    run = payload.get("run") or {}
    return run if isinstance(run, dict) else {}


def _wait_for_trigger_completion(trigger: str, *, poll_seconds: int = 60, timeout_hours: int = 48) -> dict[str, object]:
    """Attend la fin d'un run identifie par son trigger.

    Args:
        trigger (str): Trigger du run a suivre.
        poll_seconds (int): Intervalle de polling en secondes.
        timeout_hours (int): Delai maximal d'attente.

    Returns:
        dict[str, object]: Dernier etat du run observe.

    Raises:
        TimeoutError: Si le run ne termine pas dans le delai.
    """

    deadline = time.time() + timeout_hours * 3600
    seen_active = False
    while time.time() < deadline:
        run = _fetch_training_run()
        current_trigger = str(run.get("trigger") or "")
        active = bool(run.get("active"))
        status = str(run.get("status") or "")
        run_id = str(run.get("run_id") or "")
        step = run.get("current_step") or {}
        step_name = str(step.get("name") or "")
        phase = str(step.get("phase") or "")

        if current_trigger == trigger:
            if active:
                seen_active = True
                _append_log(
                    f"Run {run_id} toujours actif pour {trigger} | etape={step_name} | phase={phase}."
                )
            elif seen_active:
                _append_log(f"Run {run_id} termine pour {trigger} avec statut={status or 'inconnu'}.")
                return run
        elif seen_active and not active:
            _append_log(f"Le run suivi pour {trigger} n'est plus actif. Etat final observe: {status or 'inconnu'}.")
            return run

        time.sleep(poll_seconds)

    raise TimeoutError(f"Attente depassee pour le trigger {trigger}")


def _wait_for_trigger_start(trigger: str, *, poll_seconds: int = 15, timeout_minutes: int = 15) -> dict[str, object]:
    """Attend le demarrage effectif d'un run identifie par son trigger.

    Args:
        trigger (str): Trigger attendu.
        poll_seconds (int): Intervalle de polling en secondes.
        timeout_minutes (int): Delai maximal d'attente.

    Returns:
        dict[str, object]: Etat du run au moment de son demarrage.

    Raises:
        TimeoutError: Si le trigger n'apparait pas.
    """

    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        run = _fetch_training_run()
        if str(run.get("trigger") or "") == trigger and bool(run.get("active")):
            _append_log(f"Run detecte pour {trigger}: {run.get('run_id')}.")
            return run
        time.sleep(poll_seconds)
    raise TimeoutError(f"Demarrage non detecte pour le trigger {trigger}")


def _launch_training(argument: str, trigger: str) -> None:
    """Lance un run distant via le script de deploiement existant.

    Args:
        argument (str): Option CLI a transmettre.
        trigger (str): Trigger attendu pour le run.

    Raises:
        RuntimeError: Si le processus local echoue.
    """

    command = [sys.executable, str(START_TRAINING_SCRIPT), argument]
    _append_log(f"Lancement de {trigger} via {' '.join(command)}.")
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.stdout.strip():
        _append_log(completed.stdout.strip())
    if completed.stderr.strip():
        _append_log(f"stderr {trigger}: {completed.stderr.strip()}")
    if completed.returncode != 0:
        raise RuntimeError(f"Echec du lancement {trigger} (code {completed.returncode})")


def _query_gpu_state() -> tuple[int | None, int | None]:
    """Retourne l'utilisation GPU et la memoire utilisee sur le serveur.

    Returns:
        tuple[int | None, int | None]: Pourcentage GPU et memoire utilisee en Mo.

    Raises:
        RuntimeError: Si la connexion SSH n'est pas possible.
    """

    if not GPU_PASSWORD:
        raise RuntimeError("HIVE_SSH_PASSWORD manquant pour verifier l'etat du GPU.")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(GPU_HOST, username=GPU_USER, password=GPU_PASSWORD, timeout=20)
    try:
        command = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used "
            "--format=csv,noheader,nounits | head -n 1"
        )
        stdin, stdout, stderr = client.exec_command(command, timeout=30)
        output = stdout.read().decode("utf-8", "replace").strip()
        error = stderr.read().decode("utf-8", "replace").strip()
        if error:
            raise RuntimeError(error)
        if not output:
            return None, None
        values = [segment.strip() for segment in output.split(",")]
        if len(values) < 2:
            return None, None
        return int(values[0]), int(values[1])
    finally:
        client.close()


def _wait_for_gpu_free(*, poll_seconds: int = 60, timeout_minutes: int = 60) -> None:
    """Attend un etat GPU compatible avec un refresh GNN.

    Args:
        poll_seconds (int): Intervalle de polling en secondes.
        timeout_minutes (int): Delai maximal d'attente.

    Raises:
        TimeoutError: Si le GPU reste occupe.
    """

    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        run = _fetch_training_run()
        if bool(run.get("active")):
            _append_log("GPU reserve par un entrainement actif; attente avant le refresh GNN.")
            time.sleep(poll_seconds)
            continue
        utilization, memory_used = _query_gpu_state()
        _append_log(f"Etat GPU observe: utilisation={utilization}%, memoire={memory_used} Mo.")
        if utilization is not None and memory_used is not None and utilization <= 10 and memory_used <= 1024:
            return
        time.sleep(poll_seconds)
    raise TimeoutError("GPU encore occupe apres attente pour le refresh GNN.")


def main() -> int:
    """Execute la chaine `MuZero -> Dreamer -> GNN` en serie.

    Returns:
        int: Code retour du script.
    """

    _append_log("Debut de la chaine scalp multi-univers 7 symboles.")

    run = _fetch_training_run()
    if bool(run.get("active")) and str(run.get("trigger") or "") == MUZERO_TRIGGER:
        _append_log(f"MuZero full deja actif: {run.get('run_id')}.")
    elif bool(run.get("active")):
        _append_log(
            "Un autre run GPU est deja actif. La chaine attendra sa liberation avant de lancer Dreamer."
        )
    else:
        _append_log("Aucun run MuZero actif detecte; la chaine passe directement a Dreamer.")

    if bool(run.get("active")) and str(run.get("trigger") or "") == MUZERO_TRIGGER:
        _wait_for_trigger_completion(MUZERO_TRIGGER)

    run = _fetch_training_run()
    if bool(run.get("active")) and str(run.get("trigger") or "") == DREAMER_TRIGGER:
        _append_log(f"Dreamer full deja actif: {run.get('run_id')}.")
    else:
        if bool(run.get("active")):
            _append_log(
                f"Run actif inattendu avant Dreamer ({run.get('trigger')}). Attente de sa fin avant de poursuivre."
            )
            _wait_for_trigger_completion(str(run.get("trigger") or ""), poll_seconds=60, timeout_hours=24)
        _launch_training("--dreamer-scalp-full-7", DREAMER_TRIGGER)
        _wait_for_trigger_start(DREAMER_TRIGGER)

    _wait_for_trigger_completion(DREAMER_TRIGGER)
    _wait_for_gpu_free()

    response = _post_json(GNN_REFRESH_URL, GNN_REFRESH_PAYLOAD, timeout=30)
    _append_log(f"Refresh GNN declenche: {json.dumps(response, ensure_ascii=False)}")
    _append_log("Chaine scalp multi-univers terminee.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _append_log(f"Echec de la chaine: {exc}")
        raise
