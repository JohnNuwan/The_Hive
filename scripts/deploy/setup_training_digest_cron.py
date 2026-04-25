"""Installe un cron de digest training Telegram toutes les 15 minutes."""

from __future__ import annotations

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
CRON_LOG = "/var/log/hive_training_digest.log"
CRON_ENTRY = f"*/15 * * * * {REMOTE_CRON_WRAPPER} >> {CRON_LOG} 2>&1"
REMOTE_WRAPPER = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "eval \"$(python3 - <<'PY'\n"
    "from pathlib import Path\n"
    "import shlex\n"
    "target_keys = {\n"
    "    'TELEGRAM_BOT_TOKEN',\n"
    "    'TELEGRAM_CHAT_ID',\n"
    "    'TELEGRAM_TOPIC_ID',\n"
    "    'TELEGRAM_NOTIFY_TRAINING',\n"
    "    'TELEGRAM_NOTIFY_TRAINING_DIGEST',\n"
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
    "LAB_CONTAINER=\"$(docker ps --filter label=com.docker.compose.service=lab --format '{{.Names}}' | head -n 1)\"\n"
    "if [ -z \"$LAB_CONTAINER\" ]; then\n"
    "  echo 'Conteneur lab introuvable pour le digest training.' >&2\n"
    "  exit 1\n"
    "fi\n"
    "exec docker exec "
    "-e TELEGRAM_BOT_TOKEN "
    "-e TELEGRAM_CHAT_ID "
    "-e TELEGRAM_TOPIC_ID "
    "-e TELEGRAM_NOTIFY_TRAINING "
    "-e TELEGRAM_NOTIFY_TRAINING_DIGEST "
    "\"$LAB_CONTAINER\" "
    "bash -lc 'cd /app/eva-lab && export PYTHONUNBUFFERED=1 && python -c \"from eva_lab.training_notifier import send_training_digest; send_training_digest()\"'\n"
)
SYNC_FILES = (
    LOCAL_ROOT / "scripts" / "send_training_digest.py",
    LOCAL_ROOT / "src" / "eva-lab" / "eva_lab" / "training_notifier.py",
)


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

        cron_cmd = (
            f"(crontab -l 2>/dev/null | grep -v 'run_training_digest_cron.sh'; "
            f"echo '{CRON_ENTRY}') | crontab -"
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
