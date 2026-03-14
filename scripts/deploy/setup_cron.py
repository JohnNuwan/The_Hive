"""Installe le cron nightly Debian pour l'entrainement EVA Lab."""

from __future__ import annotations

import os
import stat
import sys
import time

import paramiko

from start_training_proxmox import (
    HOST,
    LOCAL_ROOT,
    PASS,
    REMOTE_DIR,
    REMOTE_ENV_LOADER,
    REMOTE_LAUNCH_SCRIPT,
    REMOTE_SCRIPT,
    SYNC_FILES,
    USER,
    ensure_remote_parent,
    upload_file,
)

REMOTE_CRON_WRAPPER = f"{REMOTE_DIR}/scripts/run_nightly_training_cron.sh"
CRON_LOG = "/var/log/hive_nightly_training.log"
CRON_ENTRY = f"30 23 * * * {REMOTE_CRON_WRAPPER} >> {CRON_LOG} 2>&1"
REMOTE_WRAPPER = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    f"cd \"{REMOTE_DIR}\"\n"
    f"{REMOTE_ENV_LOADER}\n"
    "export TRAINING_RUN_TRIGGER=cron\n"
    f"exec \"{REMOTE_SCRIPT}\"\n"
)

# Le cron n'a besoin que du code et des scripts de lancement.
# Synchroniser l'historique complet ici rend l'installation fragile et lente,
# alors que les donnees sont poussees par le lanceur de training ou le collecteur.
CRON_SYNC_FILES = tuple(SYNC_FILES)


if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def install_cron() -> None:
    """Synchronise les artefacts nightly puis installe le cron Debian.

    Raises:
        RuntimeError: Si l'installation distante echoue.
    """
    print(f"Connexion a {HOST} en tant que {USER}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
        print("Connexion SSH etablie.")
        sftp = client.open_sftp()

        for relative_path in CRON_SYNC_FILES:
            local_path = LOCAL_ROOT / relative_path
            remote_path = f"{REMOTE_DIR}/{relative_path.as_posix()}"
            upload_file(sftp, local_path, remote_path)

        ensure_remote_parent(sftp, REMOTE_SCRIPT)
        with sftp.file(REMOTE_SCRIPT, "w") as remote_file:
            remote_file.write(REMOTE_LAUNCH_SCRIPT)
        sftp.chmod(
            REMOTE_SCRIPT,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH,
        )

        ensure_remote_parent(sftp, REMOTE_CRON_WRAPPER)
        with sftp.file(REMOTE_CRON_WRAPPER, "w") as remote_file:
            remote_file.write(REMOTE_WRAPPER)
        sftp.chmod(
            REMOTE_CRON_WRAPPER,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH,
        )
        sftp.close()
        print("Scripts nightly et wrapper cron synchronises.")

        cron_cmd = (
            f"(crontab -l 2>/dev/null | grep -v 'run_nightly_training_cron.sh' | grep -v 'auto_train_gnn'; "
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
        print("Cron Debian installe avec succes.\n")
        print(current_crontab)
    except Exception as exc:
        print(f"Erreur d'installation cron: {exc}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    install_cron()
