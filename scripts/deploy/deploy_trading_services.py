"""Deploie des services cibles a distance sans redemarrage global.

Ce script prepare le socle de correction trading apres un run:
- synchronisation des sources `eva-lab` et `eva-banker` utiles au live;
- synchronisation optionnelle de `eva-researcher` pour les vues Nexus;
- reconstruction ciblee de `lab`, `live-inference`, `banker` et/ou `researcher`;
- aucune action destructive sur le reste de la stack.
"""

from __future__ import annotations

import argparse
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
    ensure_remote_parent,
    run_command,
    upload_file,
    _require_remote_credentials,
)

SYNC_FILES = [
    Path("docker-compose.yml"),
    Path("scripts/cpu_assist.py"),
    Path("scripts/cpu_scheduler.py"),
    Path("scripts/sql/timescaledb_v51.sql"),
    Path("src/eva-lab/eva_lab/arena.py"),
    Path("src/eva-lab/eva_lab/champion_promoter.py"),
    Path("src/eva-lab/eva_lab/dreamer_gate.py"),
    Path("src/eva-lab/eva_lab/gnn_registry.py"),
    Path("src/eva-lab/eva_lab/live_inference_main.py"),
    Path("src/eva-lab/eva_lab/live_inference_models.py"),
    Path("src/eva-lab/eva_lab/main.py"),
    Path("src/eva-lab/eva_lab/timescale_store.py"),
    Path("src/eva-lab/eva_lab/training_status.py"),
    Path("src/eva-lab/eva_lab/training_utils.py"),
    Path("src/eva-lab/eva_lab/muzero/config.py"),
    Path("src/eva-lab/eva_lab/muzero/environment.py"),
    Path("src/eva-lab/scripts/train_global_models.py"),
    Path("src/eva-lab/scripts/train_nightly_stack.py"),
    Path("src/eva-lab/pyproject.toml"),
    Path("src/eva-lab/Dockerfile"),
    Path("src/eva-core/eva_core/main.py"),
    Path("src/eva-core/pyproject.toml"),
    Path("src/eva-core/Dockerfile"),
    Path("src/shared/pyproject.toml"),
    Path("src/shared/shared/__init__.py"),
    Path("src/shared/shared/config.py"),
    Path("src/shared/shared/memory_bridge.py"),
    Path("src/shared/shared/memory_graph.py"),
    Path("src/shared/shared/models.py"),
    Path("src/shared/shared/redis_client.py"),
    Path("src/eva-banker/eva_banker/brain.py"),
    Path("src/eva-banker/eva_banker/hydra_terminal_main.py"),
    Path("src/eva-banker/eva_banker/main.py"),
    Path("src/eva-banker/eva_banker/nemesis.py"),
    Path("src/eva-banker/eva_banker/strategist.py"),
    Path("src/eva-banker/eva_banker/services/hydra.py"),
    Path("src/eva-banker/eva_banker/services/multi_account.py"),
    Path("src/eva-banker/eva_banker/services/mt5.py"),
    Path("src/eva-banker/eva_banker/services/risk.py"),
    Path("src/eva-banker/eva_banker/services/traderepublic_client.py"),
    Path("src/eva-banker/pyproject.toml"),
    Path("src/eva-banker/Dockerfile"),
    Path("scripts/deploy/setup_wireguard.py"),
    Path("scripts/deploy/setup_hydra_wine.py"),
    Path("infra/wireguard/wg0.conf.template"),
    Path("infra/wireguard/peer-client.conf.template"),
    Path("infra/wireguard/peers.sample.json"),
    Path("infra/hydra/systemd/hydra-terminal@.service.template"),
    Path("infra/hydra/bin/run_hydra_terminal.sh.template"),
    Path("infra/hydra/examples/hydra_accounts.sample.json"),
    Path("src/eva-researcher/eva_researcher/main.py"),
    Path("src/eva-researcher/eva_researcher/report_generator.py"),
    Path("src/eva-researcher/eva_researcher/__init__.py"),
    Path("src/eva-researcher/eva_researcher/services/context_engine.py"),
    Path("src/eva-researcher/eva_researcher/services/ingestion.py"),
    Path("src/eva-researcher/eva_researcher/services/pea_analyzer.py"),
    Path("src/eva-researcher/eva_researcher/services/search.py"),
    Path("src/eva-researcher/eva_researcher/services/__init__.py"),
    Path("src/eva-researcher/pyproject.toml"),
    Path("src/eva-researcher/Dockerfile"),
]
ALLOWED_SERVICES = {"core", "lab", "banker", "live-inference", "researcher", "cpu-scheduler", "timescaledb"}


def parse_args() -> argparse.Namespace:
    """Analyse les options du deploiement cible.

    Returns:
        argparse.Namespace: Arguments normalises du script.
    """
    parser = argparse.ArgumentParser(
        description="Deploie `core`, `lab`, `live-inference`, `banker`, `researcher`, `cpu-scheduler` et `timescaledb` a distance sans restart global."
    )
    parser.add_argument(
        "--service",
        action="append",
        dest="services",
        help="Service cible a reconstruire (`core`, `lab`, `live-inference`, `banker`, `researcher`, `cpu-scheduler` ou `timescaledb`). Repetable.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="N'envoie pas les fichiers source avant le `docker compose up`.",
    )
    return parser.parse_args()


def _normalize_services(raw_services: list[str] | None) -> list[str]:
    """Normalise la liste des services a reconstruire.

    Args:
        raw_services (list[str] | None): Valeurs brutes passees en ligne de commande.

    Returns:
        list[str]: Liste dedoublonnee de services valides.

    Raises:
        ValueError: Si un service demande n'est pas autorise.
    """
    requested = raw_services or ["core", "lab", "live-inference", "banker"]
    services: list[str] = []
    seen: set[str] = set()
    for item in requested:
        service = str(item or "").strip().lower()
        if not service:
            continue
        if service not in ALLOWED_SERVICES:
            raise ValueError(f"Service non supporte: {service}")
        if service in seen:
            continue
        services.append(service)
        seen.add(service)
    return services


def sync_sources(sftp: paramiko.SFTPClient) -> None:
    """Synchronise les fichiers trading utiles au deploiement cible.

    Args:
        sftp (paramiko.SFTPClient): Canal SFTP distant.
    """
    for relative_path in SYNC_FILES:
        local_path = LOCAL_ROOT / relative_path
        remote_path = f"{REMOTE_DIR}/{relative_path.as_posix()}"
        ensure_remote_parent(sftp, remote_path)
        upload_file(sftp, local_path, remote_path)


def deploy_services(services: list[str], no_sync: bool = False) -> None:
    """Deploie les services demandes sur le serveur distant.

    Args:
        services (list[str]): Services a reconstruire.
        no_sync (bool): Desactive la synchronisation des sources si vrai.
    """
    ssh_password, sudo_password = _require_remote_credentials()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=ssh_password, timeout=15)
        print(f"Connexion SSH etablie vers {HOST}.")
        if not no_sync:
            sftp = client.open_sftp()
            sync_sources(sftp)
            sftp.close()
            print("Sources synchronisees.")

        service_list = " ".join(services)
        command = (
            f"echo '{sudo_password}' | sudo -S bash -lc "
            f"'cd {REMOTE_DIR} && docker compose up -d --build {service_list}'"
        )
        output, error, code = run_command(client, command, timeout=180)
        if code != 0:
            raise RuntimeError(error or output or f"Code {code}")
        print(f"Deploiement cible termine: {service_list}")
        if output.strip():
            print(output.strip())
        if error.strip():
            print(error.strip())
    finally:
        client.close()


if __name__ == "__main__":
    args = parse_args()
    try:
        deploy_services(_normalize_services(args.services), no_sync=args.no_sync)
    except Exception as exc:
        print(f"Erreur: {exc}")
        raise SystemExit(1) from exc
