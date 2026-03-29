"""Lance la nuit Gold manuelle sur le serveur sans dependre de V4.

Ce lanceur reste cote local. Il prepare la configuration de la nuit,
synchronise le payload critique sur Proxmox, puis demarre en arriere-plan
le runner autonome ``gold_manual_remote_runner.py``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path
from typing import Any

import paramiko

import start_training_proxmox as proxmox

REMOTE_RUNNER_PATH = f"{proxmox.REMOTE_DIR}/scripts/deploy/gold_manual_remote_runner.py"
REMOTE_STATE_PATH = f"{proxmox.REMOTE_DIR}/data/checkpoints/gold_manual_state.json"
DEFAULT_DREAMER_PROXY_TRIAL = "gold_balanced_short_seq"


def _merge_runtime_overrides(runtime_overrides: dict[str, str]) -> dict[str, str]:
    """Ajoute les variables de base Gold au bloc d'environnement.

    Args:
        runtime_overrides (dict[str, str]): Overrides moteur.

    Returns:
        dict[str, str]: Overrides complets avec la source TimeDB forcee.
    """

    merged = dict(proxmox._build_v4_supervisor_overrides())
    merged.update(runtime_overrides)
    return merged


def _build_remote_config(
    *,
    run_group_id: str,
    resume_checkpoint_path: str,
    resume_step: int,
    skip_gnn: bool,
) -> dict[str, Any]:
    """Construit la configuration JSON consommee par le runner distant.

    Args:
        run_group_id (str): Identifiant unique de la nuit Gold.
        resume_checkpoint_path (str): Checkpoint MuZero explicite.
        resume_step (int): Etape de reprise associee.
        skip_gnn (bool): Indique si le refresh GNN doit etre ignore.

    Returns:
        dict[str, Any]: Configuration serialisable du bypass Gold.
    """

    muzero_trials = {
        str(trial.get("trial_id") or "").strip(): dict(trial)
        for trial in proxmox._get_v3_trial_catalog(proxmox.GOLD_MONDAY_PROFILE)
    }
    primary_trial = dict(muzero_trials["momentum_close"])
    fallback_trial = dict(muzero_trials["balanced_activity"])
    full_catalog = {
        trial_id: (
            lambda overrides: {
                **overrides,
                "MUZERO_COLLECTION_GAME_TIMEOUT_SECONDS": "240",
            }
        )(
            _merge_runtime_overrides(
                proxmox._build_gold_monday_muzero_overrides(
                    proxmox.GOLD_MONDAY_PROFILE,
                    mode="full",
                    trial=trial,
                )
            )
        )
        for trial_id, trial in muzero_trials.items()
    }

    smoke_trials = []
    for trial in proxmox._get_gold_monday_dreamer_smoke_trials():
        smoke_trials.append(
            {
                "trial_id": str(trial.get("trial_id") or "").strip(),
                "runtime_overrides": _merge_runtime_overrides(
                    proxmox._build_gold_monday_dreamer_overrides(
                        proxmox.GOLD_MONDAY_PROFILE,
                        mode="smoke",
                        trial=trial,
                    )
                ),
            }
        )

    proxy_trial = next(
        (
            dict(trial)
            for trial in proxmox._get_gold_monday_dreamer_proxy_trials()
            if str(trial.get("trial_id") or "").strip() == DEFAULT_DREAMER_PROXY_TRIAL
        ),
        None,
    )
    if proxy_trial is None:
        raise RuntimeError("Trial Dreamer proxy Gold introuvable.")

    return {
        "run_group_id": run_group_id,
        "state_path": REMOTE_STATE_PATH,
        "focus_symbols": [proxmox.GOLD_MONDAY_FOCUS_SYMBOL],
        "gate_profile": "gold_demo",
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_step": resume_step,
        "skip_gnn": skip_gnn,
        "manual_env": {
            "sequence_profile": "gold_manual_night",
            "supervisor_state": "manual_gold_night",
        },
        "muzero": {
            "primary_proxy": {
                "trial_id": "momentum_close",
                "runtime_overrides": (
                    lambda overrides: {
                        **overrides,
                        "MUZERO_RESUME_COLLECTION_MODE": "policy_only",
                        "MUZERO_COLLECTION_GAME_TIMEOUT_SECONDS": "180",
                    }
                )(
                    _merge_runtime_overrides(
                        proxmox._build_gold_monday_muzero_overrides(
                            proxmox.GOLD_MONDAY_PROFILE,
                            mode="proxy_ga",
                            trial=primary_trial,
                        )
                    )
                ),
            },
            "fallback_proxy": {
                "trial_id": "balanced_activity",
                "runtime_overrides": (
                    lambda overrides: {
                        **overrides,
                        "MUZERO_COLLECTION_GAME_TIMEOUT_SECONDS": "240",
                    }
                )(
                    _merge_runtime_overrides(
                        proxmox._build_gold_monday_muzero_overrides(
                            proxmox.GOLD_MONDAY_PROFILE,
                            mode="proxy_ga",
                            trial=fallback_trial,
                        )
                    )
                ),
            },
            "full_catalog": full_catalog,
        },
        "dreamer": {
            "smoke_trials": smoke_trials,
            "proxy_trial": {
                "trial_id": str(proxy_trial.get("trial_id") or "").strip(),
                "runtime_overrides": _merge_runtime_overrides(
                    proxmox._build_gold_monday_dreamer_overrides(
                        proxmox.GOLD_MONDAY_PROFILE,
                        mode="proxy_ga",
                        trial=proxy_trial,
                    )
                ),
            },
            "full_trial": {
                "trial_id": str(proxy_trial.get("trial_id") or "").strip(),
                "runtime_overrides": _merge_runtime_overrides(
                    proxmox._build_gold_monday_dreamer_overrides(
                        proxmox.GOLD_MONDAY_PROFILE,
                        mode="full",
                        trial=proxy_trial,
                    )
                ),
            },
        },
        "gnn_request": {
            "symbols": [
                proxmox.GOLD_MONDAY_FOCUS_SYMBOL,
                *list(proxmox.GOLD_MONDAY_CONTEXT_SYMBOLS),
            ],
            "focus_symbol": proxmox.GOLD_MONDAY_FOCUS_SYMBOL,
            "context_symbols": list(proxmox.GOLD_MONDAY_CONTEXT_SYMBOLS),
            "deployment_class": "consultative_gold",
            "epochs": 300,
            "batch_size": 32,
            "checkpoint_every": 25,
            "max_symbols": 5,
        },
    }


def _write_remote_text_file(
    sftp: paramiko.SFTPClient,
    remote_path: str,
    content: str,
) -> None:
    """Ecrit un fichier texte UTF-8 sur le serveur.

    Args:
        sftp (paramiko.SFTPClient): Session SFTP active.
        remote_path (str): Fichier distant cible.
        content (str): Contenu a ecrire.
    """

    proxmox.ensure_remote_parent(sftp, remote_path)
    with sftp.file(remote_path, "w") as remote_file:
        remote_file.write(content)


def _stop_remote_manual_runners(client: paramiko.SSHClient, sudo_password: str) -> None:
    """Arrete les anciens runners Gold manuels cote serveur.

    Args:
        client (paramiko.SSHClient): Session SSH active.
        sudo_password (str): Mot de passe sudo distant.
    """

    cleanup_body = (
        "pkill -f 'scripts/deploy/gold_manual_remote_runner.py' || true; "
        "pkill -f 'scripts/deploy/v4_sequence_runner.py' || true"
    )
    cleanup_command = f"echo {shlex.quote(sudo_password)} | sudo -S bash -lc {shlex.quote(cleanup_body)}"
    proxmox.run_command(client, cleanup_command, timeout=30)


def _resolve_remote_runner_pid(
    client: paramiko.SSHClient,
    *,
    remote_config_path: str,
) -> str | None:
    """Retourne le PID distant du runner Gold s'il tourne deja.

    Args:
        client (paramiko.SSHClient): Session SSH active.
        remote_config_path (str): Fichier de configuration du runner.

    Returns:
        str | None: PID trouve ou ``None``.
    """

    command = (
        "ps -eo pid,cmd | "
        f"grep -F -- {shlex.quote(remote_config_path)} | "
        "grep 'gold_manual_remote_runner.py' | grep -v grep | awk '{print $1}' | head -n 1"
    )
    output, _, code = proxmox.run_command(client, command, timeout=30)
    if code != 0:
        return None
    pid = str(output).strip()
    return pid or None


def launch_gold_manual_remote(*, resume_checkpoint_path: str, resume_step: int, skip_gnn: bool) -> None:
    """Synchronise le payload puis lance le runner Gold manuel sur Proxmox.

    Args:
        resume_checkpoint_path (str): Checkpoint MuZero explicite.
        resume_step (int): Etape de reprise associee.
        skip_gnn (bool): Ignore le refresh GNN si vrai.
    """

    run_group_id = f"gold_manual_{time.strftime('%Y%m%d_%H%M%S')}"
    config_payload = _build_remote_config(
        run_group_id=run_group_id,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=resume_step,
        skip_gnn=skip_gnn,
    )
    remote_config_path = f"{proxmox.REMOTE_DIR}/data/checkpoints/gold_manual_config_{run_group_id}.json"
    remote_stdout_path = f"{proxmox.REMOTE_DIR}/data/checkpoints/gold_manual_runner_{run_group_id}.out.log"
    remote_stderr_path = f"{proxmox.REMOTE_DIR}/data/checkpoints/gold_manual_runner_{run_group_id}.err.log"

    ssh_password, sudo_password = proxmox._require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(proxmox.HOST, username=proxmox.USER, password=ssh_password, timeout=20)
        proxmox._sync_remote_training_payload(client, profile_hint=proxmox.GOLD_MONDAY_PROFILE)
        _stop_remote_manual_runners(client, sudo_password)
        sftp = client.open_sftp()
        try:
            _write_remote_text_file(
                sftp,
                remote_config_path,
                json.dumps(config_payload, indent=2, ensure_ascii=False),
            )
        finally:
            sftp.close()

        launch_body = (
            f"cd {proxmox.REMOTE_DIR} && "
            f"nohup env PYTHONPATH={proxmox.REMOTE_DIR}/src/eva-lab:{proxmox.REMOTE_DIR}/src/shared "
            f"python3 {REMOTE_RUNNER_PATH} --config {remote_config_path} "
            f"> {remote_stdout_path} 2> {remote_stderr_path} < /dev/null & echo $!"
        )
        launch_command = f"echo {shlex.quote(sudo_password)} | sudo -S bash -lc {shlex.quote(launch_body)}"
        try:
            output, error, code = proxmox.run_command(client, launch_command, timeout=120)
            if code != 0:
                raise RuntimeError(error or output or "Lancement du runner Gold manuel impossible.")
            pid_lines = [line.strip() for line in output.splitlines() if line.strip()]
            remote_pid = pid_lines[-1] if pid_lines else "inconnu"
        except TimeoutError:
            time.sleep(5)
            remote_pid = _resolve_remote_runner_pid(client, remote_config_path=remote_config_path)
            if remote_pid is None:
                raise

        print(f"Runner Gold manuel lance sur {proxmox.HOST}.")
        print(f" - run_group_id: {run_group_id}")
        print(f" - pid distant: {remote_pid}")
        print(f" - config: {remote_config_path}")
        print(f" - state: {REMOTE_STATE_PATH}")
        print(f" - stdout: {remote_stdout_path}")
        print(f" - stderr: {remote_stderr_path}")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    """Analyse les arguments du lanceur Gold manuel.

    Returns:
        argparse.Namespace: Arguments normalises.
    """

    parser = argparse.ArgumentParser(description="Lance le runner Gold manuel sur le serveur.")
    parser.add_argument(
        "--resume-checkpoint",
        default="data/muzero/weights/muzero_scalp_ckpt_7000.pkl",
        help="Checkpoint MuZero explicite a reutiliser pour le proxy principal.",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=7000,
        help="Etape de reprise associee au checkpoint MuZero.",
    )
    parser.add_argument(
        "--skip-gnn",
        action="store_true",
        help="Ignore le refresh GNN pour reserver tout le temps a MuZero et Dreamer.",
    )
    return parser.parse_args()


def main() -> int:
    """Point d'entree CLI du lanceur Gold manuel.

    Returns:
        int: Code de sortie systeme.
    """

    args = parse_args()
    launch_gold_manual_remote(
        resume_checkpoint_path=str(args.resume_checkpoint or "").strip(),
        resume_step=int(args.resume_step),
        skip_gnn=bool(args.skip_gnn),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
