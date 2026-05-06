"""Installe un cron de digest training Telegram avec cadence parametrable."""

from __future__ import annotations

import os
import stat
import sys
import time

from start_training_proxmox import (
    HOST,
    LOCAL_ROOT,
    PASS,
    REMOTE_DIR,
    USER,
    ensure_remote_parent,
    upload_file,
)


REMOTE_CRON_WRAPPER = f"{REMOTE_DIR}/scripts/run_training_digest_cron.sh"
CRON_LOG = f"{REMOTE_DIR}/data/checkpoints/hive_training_digest.log"
REMOTE_WRAPPER = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "echo \"[training-digest] $(date -Is) demarrage\" \n"
    "eval \"$(python3 - <<'PY'\n"
    "from pathlib import Path\n"
    "import shlex\n"
    "target_keys = {\n"
    "    'TELEGRAM_BOT_TOKEN',\n"
    "    'TELEGRAM_CHAT_ID',\n"
    "    'TELEGRAM_TOPIC_ID',\n"
    "    'TELEGRAM_NOTIFY_TRAINING',\n"
    "    'TELEGRAM_NOTIFY_TRAINING_DIGEST',\n"
    "    'TELEGRAM_TRAINING_DIGEST_ONLY_ON_CHANGE',\n"
    "    'TELEGRAM_TRAINING_DIGEST_FORCE_AFTER_MINUTES',\n"
    "    'TELEGRAM_TRAINING_DIGEST_STEP_BUCKET',\n"
    "    'TRAINING_DIGEST_STATE_PATH',\n"
    "}\n"
    f"path = Path('{REMOTE_DIR}/.env')\n"
    "if not path.exists():\n"
    "    raise SystemExit(0)\n"
    "for raw_line in path.read_text(encoding='utf-8', errors='ignore').splitlines():\n"
    "    line = raw_line.strip()\n"
    "    if not line or line.startswith('#') or '=' not in line:\n"
    "        continue\n"
    "    key, value = line.split('=', 1)\n"
    "    key = key.strip()\n"
    "    if key not in target_keys:\n"
    "        continue\n"
    "    value = value.strip()\n"
    "    if (value.startswith('\"') and value.endswith('\"')) or (value.startswith(\"'\") and value.endswith(\"'\")):\n"
    "        value = value[1:-1]\n"
    "    print(f'export {key}={shlex.quote(value)}')\n"
    "PY\n"
    ")\"\n"
    "export TELEGRAM_NOTIFY_TRAINING=\"${TELEGRAM_NOTIFY_TRAINING:-1}\"\n"
    "export TELEGRAM_NOTIFY_TRAINING_DIGEST=\"${TELEGRAM_NOTIFY_TRAINING_DIGEST:-1}\"\n"
    "export TELEGRAM_TRAINING_DIGEST_ONLY_ON_CHANGE=\"${TELEGRAM_TRAINING_DIGEST_ONLY_ON_CHANGE:-1}\"\n"
    "export TELEGRAM_TRAINING_DIGEST_FORCE_AFTER_MINUTES=\"${TELEGRAM_TRAINING_DIGEST_FORCE_AFTER_MINUTES:-180}\"\n"
    "export TELEGRAM_TRAINING_DIGEST_STEP_BUCKET=\"${TELEGRAM_TRAINING_DIGEST_STEP_BUCKET:-500}\"\n"
    "export TRAINING_DIGEST_STATE_PATH=\"${TRAINING_DIGEST_STATE_PATH:-/app/eva-lab/data/checkpoints/training_digest_state.json}\"\n"
    "LAB_CONTAINER=\"$(docker ps --filter label=com.docker.compose.service=lab --format '{{.Names}}' | head -n 1)\"\n"
    "if [ -z \"$LAB_CONTAINER\" ]; then\n"
    "  echo 'Conteneur lab introuvable pour le digest training.' >&2\n"
    "  exit 1\n"
    "fi\n"
    "echo \"[training-digest] $(date -Is) conteneur=$LAB_CONTAINER topic=${TELEGRAM_TOPIC_ID:-none}\" \n"
    "docker exec "
    "-e TELEGRAM_BOT_TOKEN "
    "-e TELEGRAM_CHAT_ID "
    "-e TELEGRAM_TOPIC_ID "
    "-e TELEGRAM_NOTIFY_TRAINING "
    "-e TELEGRAM_NOTIFY_TRAINING_DIGEST "
    "-e TELEGRAM_TRAINING_DIGEST_ONLY_ON_CHANGE "
    "-e TELEGRAM_TRAINING_DIGEST_FORCE_AFTER_MINUTES "
    "-e TELEGRAM_TRAINING_DIGEST_STEP_BUCKET "
    "-e TRAINING_DIGEST_STATE_PATH "
    "\"$LAB_CONTAINER\" "
    "bash -lc 'cd /app/eva-lab && export PYTHONUNBUFFERED=1 && export PYTHONPATH=/app/shared:/app/eva-lab:${PYTHONPATH:-} && python -c \"from eva_lab.training_notifier import send_training_digest; send_training_digest()\"' \n"
    "echo \"[training-digest] $(date -Is) succes\" \n"
)
SYNC_FILES = (
    LOCAL_ROOT / "scripts" / "send_training_digest.py",
    LOCAL_ROOT / "src" / "eva-lab" / "eva_lab" / "training_notifier.py",
    LOCAL_ROOT / "src" / "shared" / "shared" / "telegram_client.py",
)


def _build_cron_entry(interval_minutes: int) -> str:
    """Construit l'entree cron a partir d'un intervalle en minutes.

    Args:
        interval_minutes (int): Frequence demandee.

    Returns:
        str: Ligne cron prete a installer.

    Raises:
        ValueError: Si l'intervalle n'est pas supporte.
    """
    normalized = max(int(interval_minutes), 1)
    if normalized < 60:
        if 60 % normalized != 0:
            raise ValueError(
                "TRAINING_DIGEST_CRON_MINUTES doit diviser 60 pour rester compatible cron."
            )
        schedule = f"*/{normalized} * * * *"
    elif normalized % 60 == 0:
        hours = normalized // 60
        if hours == 1:
            schedule = "0 * * * *"
        elif 24 % hours == 0:
            schedule = f"0 */{hours} * * *"
        else:
            raise ValueError(
                "TRAINING_DIGEST_CRON_MINUTES doit produire un pas horaire compatible cron."
            )
    else:
        raise ValueError("TRAINING_DIGEST_CRON_MINUTES doit etre un multiple de 60 au-dela d'une heure.")
    return f"{schedule} {REMOTE_CRON_WRAPPER} >> {CRON_LOG} 2>&1"


def install_cron() -> None:
    """Synchronise le digest training puis installe le cron distant.

    Raises:
        RuntimeError: Si l'installation distante echoue.
    """
    if not PASS:
        raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

    print(f"Connexion a {HOST} en tant que {USER}...")
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
        print("Connexion SSH etablie.")
        sftp = client.open_sftp()

        for local_path in SYNC_FILES:
            remote_path = f"{REMOTE_DIR}/{local_path.relative_to(LOCAL_ROOT).as_posix()}"
            upload_file(sftp, local_path, remote_path)

        ensure_remote_parent(sftp, REMOTE_CRON_WRAPPER)
        with sftp.file(REMOTE_CRON_WRAPPER, "w") as remote_file:
            remote_file.write(REMOTE_WRAPPER)
        sftp.chmod(
            REMOTE_CRON_WRAPPER,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH,
        )
        sftp.close()
        print("Digest training et wrapper cron synchronises.")

        cron_interval_minutes = max(int(os.getenv("TRAINING_DIGEST_CRON_MINUTES", "60")), 1)
        cron_entry = _build_cron_entry(cron_interval_minutes)
        cron_cmd = (
            f"(crontab -l 2>/dev/null | grep -v 'run_training_digest_cron.sh'; "
            f"echo '{cron_entry}') | crontab -"
        )
        stdin, stdout, stderr = client.exec_command(cron_cmd)
        _ = stdout.read().decode("utf-8", "ignore")
        cron_err = stderr.read().decode("utf-8", "ignore")
        if cron_err.strip():
            raise RuntimeError(f"Installation cron en echec: {cron_err.strip()}")

        time.sleep(1)
        stdin, stdout, stderr = client.exec_command("crontab -l")
        current_crontab = stdout.read().decode("utf-8", "ignore")
        print("Cron digest training installe avec succes.\n")
        print(current_crontab)
    except Exception as exc:
        print(f"Erreur d'installation du cron digest training: {exc}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    install_cron()
