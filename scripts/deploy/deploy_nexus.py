"""Deploie le frontend Nexus a distance sans toucher au trainer.

Le conteneur `nexus` embarque le dossier `dist` produit localement. Ce script:
- lance le build local du frontend;
- synchronise les artefacts necessaires sur le serveur;
- reconstruit uniquement le service `nexus`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import paramiko

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

LOCAL_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from scripts.deploy.start_training_proxmox import (
    HOST,
    REMOTE_DIR,
    USER,
    _require_remote_credentials,
    ensure_remote_parent,
    run_command,
    upload_file,
    upload_tree,
)

NEXUS_ROOT = LOCAL_ROOT / "src" / "eva-nexus"
SYNC_FILES = [
    Path("src/eva-nexus/Dockerfile"),
    Path("src/eva-nexus/nginx.conf"),
    Path("src/eva-nexus/package.json"),
    Path("src/eva-nexus/package-lock.json"),
    Path("src/eva-nexus/tsconfig.json"),
    Path("src/eva-nexus/tsconfig.node.json"),
    Path("src/eva-nexus/vite.config.ts"),
]
SYNC_DIRS = [
    Path("src/eva-nexus/dist"),
]


def parse_args() -> argparse.Namespace:
    """Analyse les options du deploiement Nexus.

    Returns:
        argparse.Namespace: Arguments normalises.
    """
    parser = argparse.ArgumentParser(description="Construit localement puis deploie Nexus a distance.")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="N'execute pas `npm run build` localement avant synchronisation.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="N'envoie pas les artefacts avant le `docker compose up`.",
    )
    return parser.parse_args()


def build_local_nexus(skip_build: bool = False) -> None:
    """Construit localement le bundle `dist` de Nexus.

    Args:
        skip_build (bool): Desactive la construction si vrai.

    Raises:
        RuntimeError: Si le build local echoue.
    """
    if skip_build:
        return
    npm_command = "npm.cmd" if os.name == "nt" else "npm"
    result = subprocess.run(
        [npm_command, "run", "build"],
        cwd=NEXUS_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Le build local de Nexus a echoue.")


def sync_nexus_artifacts(sftp: paramiko.SFTPClient) -> None:
    """Synchronise les artefacts `nexus` vers le serveur.

    Args:
        sftp (paramiko.SFTPClient): Canal SFTP actif.
    """
    for relative_path in SYNC_FILES:
        local_path = LOCAL_ROOT / relative_path
        remote_path = f"{REMOTE_DIR}/{relative_path.as_posix()}"
        ensure_remote_parent(sftp, remote_path)
        upload_file(sftp, local_path, remote_path)
    for relative_dir in SYNC_DIRS:
        local_dir = LOCAL_ROOT / relative_dir
        remote_dir = f"{REMOTE_DIR}/{relative_dir.as_posix()}"
        upload_tree(sftp, local_dir, remote_dir)


def deploy_nexus(*, skip_build: bool = False, no_sync: bool = False) -> None:
    """Construit et deploie le service `nexus`.

    Args:
        skip_build (bool): N'exectute pas le build local si vrai.
        no_sync (bool): N'envoie pas les artefacts si vrai.
    """
    build_local_nexus(skip_build=skip_build)
    ssh_password, sudo_password = _require_remote_credentials()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        print(f"Connexion SSH etablie vers {HOST}.")

        if not no_sync:
            sftp = client.open_sftp()
            sync_nexus_artifacts(sftp)
            sftp.close()
            print("Artefacts Nexus synchronises.")

        command = (
            f"echo '{sudo_password}' | sudo -S bash -lc "
            f"'cd {REMOTE_DIR} && docker compose up -d --build nexus'"
        )
        output, error, code = run_command(client, command, timeout=180)
        if code != 0:
            raise RuntimeError(error or output or f"Code {code}")
        print("Deploiement Nexus termine.")
        if output.strip():
            print(output.strip())
        if error.strip():
            print(error.strip())
    finally:
        client.close()


if __name__ == "__main__":
    args = parse_args()
    try:
        deploy_nexus(skip_build=args.no_build, no_sync=args.no_sync)
    except Exception as exc:
        print(f"Erreur: {exc}")
        raise SystemExit(1) from exc
