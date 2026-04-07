"""Deploie des services cibles a distance sans redemarrage global.

Ce script prepare le socle de correction trading apres un run:
- synchronisation des sources `eva-lab` et `eva-banker` utiles au live;
- synchronisation optionnelle de `eva-researcher` pour les vues Nexus;
- reconstruction ciblee de `lab`, `live-inference`, `banker` et/ou `researcher`;
- aucune action destructive sur le reste de la stack.
"""

from __future__ import annotations

import argparse
import json
import shlex
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
    Path("src/eva-lab/eva_lab/muzero/dreamer_agent.py"),
    Path("src/eva-lab/eva_lab/muzero/dreamer_networks.py"),
    Path("src/eva-lab/eva_lab/muzero/dreamer_trainer.py"),
    Path("src/eva-lab/eva_lab/muzero/imagination.py"),
    Path("src/eva-lab/eva_lab/muzero/rssm.py"),
    Path("src/eva-lab/scripts/train_global_models.py"),
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
    Path("src/eva-banker/eva_banker/main.py"),
    Path("src/eva-banker/eva_banker/strategist.py"),
    Path("src/eva-banker/eva_banker/services/risk.py"),
    Path("src/eva-banker/eva_banker/services/traderepublic_client.py"),
    Path("src/eva-banker/pyproject.toml"),
    Path("src/eva-banker/Dockerfile"),
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
        description="Deploie `lab`, `live-inference` et `banker` par defaut, sans redemarrage global."
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verifie les endpoints critiques apres le deploiement cible.",
    )
    parser.add_argument(
        "--banker-force-maintenance",
        choices=["true", "false"],
        help="Force explicitement le banker en maintenance ou le reactive.",
    )
    parser.add_argument(
        "--banker-ensemble-enabled",
        choices=["true", "false"],
        help="Active ou desactive explicitement le mode ensemble MuZero/Dreamer.",
    )
    parser.add_argument(
        "--banker-ensemble-min-edge",
        type=float,
        help="Surcharge la marge minimale de l'ensemble pour le live.",
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
    requested = raw_services or ["lab", "live-inference", "banker"]
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


def _build_compose_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Construit les surcharges d'environnement a injecter a `docker compose`.

    Args:
        args (argparse.Namespace): Arguments CLI deja normalises.

    Returns:
        dict[str, str]: Variables shell a appliquer avant le deploiement.
    """
    overrides: dict[str, str] = {}
    if args.banker_force_maintenance is not None:
        overrides["BANKER_FORCE_MAINTENANCE"] = args.banker_force_maintenance
    if args.banker_ensemble_enabled is not None:
        overrides["BANKER_ENSEMBLE_ENABLED"] = args.banker_ensemble_enabled
        overrides["LIVE_ENSEMBLE_ENABLED"] = args.banker_ensemble_enabled
    if args.banker_ensemble_min_edge is not None:
        normalized_edge = f"{float(args.banker_ensemble_min_edge):.4f}".rstrip("0").rstrip(".")
        overrides["BANKER_ENSEMBLE_MIN_EDGE"] = normalized_edge
        overrides["ENSEMBLE_MIN_EDGE"] = normalized_edge
    return overrides


def _build_remote_http_probe(port: int, path: str) -> str:
    """Construit une sonde HTTP distante executee en Python standard.

    Args:
        port (int): Port cible sur l'hote distant.
        path (str): Route HTTP a verifier.

    Returns:
        str: Commande shell distante qui imprime un JSON de verification.
    """
    return f"""python3 - <<'PY'
import json
import urllib.error
import urllib.request

url = "http://127.0.0.1:{port}{path}"
payload = {{"ok": False, "url": url}}
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        raw = response.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {{"raw": raw[:2000]}}
        payload = {{
            "ok": True,
            "url": url,
            "status_code": response.getcode(),
            "body": body,
        }}
except Exception as exc:
    payload["error"] = str(exc)
print(json.dumps(payload, ensure_ascii=False))
PY"""


def verify_services(client: paramiko.SSHClient, services: list[str]) -> dict[str, object]:
    """Verifie les endpoints critiques des services de trading deploies.

    Args:
        client (paramiko.SSHClient): Client SSH deja connecte.
        services (list[str]): Services cibles du deploiement.

    Returns:
        dict[str, object]: Resume JSON des checks HTTP distants.
    """
    probes: list[tuple[str, int, str]] = []
    if "lab" in services:
        probes.extend(
            [
                ("lab_champions", 8600, "/champions/status"),
                ("lab_dreamer", 8600, "/dreamer/status"),
            ]
        )
    if "live-inference" in services:
        probes.extend(
            [
                ("live_inference_health", 8610, "/health"),
                ("live_inference_status", 8610, "/status"),
            ]
        )
    if "banker" in services:
        probes.extend(
            [
                ("banker_health", 8100, "/health"),
                ("banker_trading", 8100, "/trading/status"),
            ]
        )

    results: dict[str, object] = {}
    for label, port, path in probes:
        command = _build_remote_http_probe(port=port, path=path)
        output, error, code = run_command(client, command, timeout=30)
        if code != 0:
            results[label] = {
                "ok": False,
                "url": f"http://127.0.0.1:{port}{path}",
                "error": error.strip() or output.strip() or f"Code {code}",
            }
            continue
        try:
            results[label] = json.loads(output.strip())
        except json.JSONDecodeError:
            results[label] = {
                "ok": False,
                "url": f"http://127.0.0.1:{port}{path}",
                "error": output.strip() or "Reponse JSON invalide",
            }
    return results


def deploy_services(
    services: list[str],
    no_sync: bool = False,
    verify: bool = False,
    compose_env_overrides: dict[str, str] | None = None,
) -> None:
    """Deploie les services demandes sur le serveur distant.

    Args:
        services (list[str]): Services a reconstruire.
        no_sync (bool): Desactive la synchronisation des sources si vrai.
        verify (bool): Active les checks HTTP post-deploiement si vrai.
        compose_env_overrides (dict[str, str] | None): Variables shell a injecter
            avant `docker compose`.
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
        env_prefix = ""
        if compose_env_overrides:
            env_prefix = " ".join(
                f"{name}={shlex.quote(str(value))}"
                for name, value in compose_env_overrides.items()
            )
            env_prefix = f"{env_prefix} "
        command = (
            f"echo '{sudo_password}' | sudo -S bash -lc "
            f"'cd {REMOTE_DIR} && {env_prefix}docker compose up -d --build {service_list}'"
        )
        output, error, code = run_command(client, command, timeout=180)
        if code != 0:
            raise RuntimeError(error or output or f"Code {code}")
        print(f"Deploiement cible termine: {service_list}")
        if output.strip():
            print(output.strip())
        if error.strip():
            print(error.strip())
        if verify:
            verification = verify_services(client, services)
            print("Verification HTTP distante:")
            print(json.dumps(verification, indent=2, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    args = parse_args()
    try:
        deploy_services(
            _normalize_services(args.services),
            no_sync=args.no_sync,
            verify=args.verify,
            compose_env_overrides=_build_compose_env_overrides(args),
        )
    except Exception as exc:
        print(f"Erreur: {exc}")
        raise SystemExit(1) from exc
