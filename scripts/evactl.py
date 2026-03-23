#!/usr/bin/env python3
"""
Pilote les operations locales d'administration de THE HIVE.

Cette CLI fournit une facade legere pour:
- verifier l'etat des services exposes localement;
- consulter les logs locaux;
- activer un verrouillage d'urgence local;
- deleguer les charges CPU sures vers `cpu_assist.py`.

Elle n'orchestre aucun redemarrage Docker et n'intervient jamais directement
sur le training market actif.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib import error, request

from cpu_assist import DEFAULT_PORTS, build_service_urls


ROOT_DIR = Path(__file__).resolve().parent.parent
CPU_ASSIST_PATH = Path(__file__).resolve().with_name("cpu_assist.py")
LOG_CANDIDATES = (
    ROOT_DIR / "logs",
    ROOT_DIR / "data" / "logs",
)
LOCKDOWN_CANDIDATES = (
    ROOT_DIR / "data" / "runtime" / "lockdown.mode",
    ROOT_DIR / "lockdown.mode",
)


def iter_lockdown_paths() -> Iterable[Path]:
    """Retourne les chemins de verrouillage a verifier."""

    return LOCKDOWN_CANDIDATES


def get_active_lockdown_path() -> Path | None:
    """Retourne le chemin de verrouillage present s'il existe."""

    for path in iter_lockdown_paths():
        if path.exists():
            return path
    return None


def get_primary_lockdown_path() -> Path:
    """Retourne le chemin principal ou ecrire le verrouillage."""

    return LOCKDOWN_CANDIDATES[0]


def fetch_json(url: str, timeout: float) -> tuple[str, int | None, str]:
    """Recupere un endpoint JSON ou texte de maniere robuste.

    Args:
        url (str): URL cible.
        timeout (float): Timeout HTTP en secondes.

    Returns:
        tuple[str, int | None, str]: Statut logique, code HTTP eventuel et detail.
    """

    req = request.Request(url=url, method="GET", headers={"Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return "UP", response.status, "Disponible"
    except error.HTTPError as exc:
        return "WARN", exc.code, f"HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover - depend de l'environnement local
        return "DOWN", None, str(exc)


def check_status(args: argparse.Namespace) -> int:
    """Affiche l'etat des services exposes localement.

    Args:
        args (argparse.Namespace): Arguments CLI.

    Returns:
        int: Code de retour processus.
    """

    service_urls = build_service_urls(args.host)
    selected = set(args.service or service_urls.keys())
    print(f"[EvaCTL] Etat de la ruche au {datetime.now().isoformat()}")
    print("-" * 92)
    print(f"{'SERVICE':<14} | {'URL':<33} | {'ETAT':<6} | {'CODE':<6} | DETAIL")
    print("-" * 92)

    overall_ok = True
    for service in sorted(selected):
        base_url = service_urls.get(service)
        if base_url is None:
            overall_ok = False
            print(f"{service:<14} | {'-':<33} | {'N/A':<6} | {'-':<6} | Service inconnu")
            continue
        status, status_code, detail = fetch_json(base_url.rstrip("/") + "/health", args.timeout)
        if status != "UP":
            overall_ok = False
        code_text = str(status_code) if status_code is not None else "-"
        print(f"{service:<14} | {base_url.rstrip('/'): <33} | {status:<6} | {code_text:<6} | {detail}")

    print("-" * 92)
    lockdown_path = get_active_lockdown_path()
    if lockdown_path is not None:
        print(f"VERROUILLAGE LOCAL ACTIF: {lockdown_path}")
        overall_ok = False
    else:
        print("Aucun verrouillage local actif.")
    return 0 if overall_ok else 1


def trigger_panic(args: argparse.Namespace) -> int:
    """Active un verrouillage local d'urgence.

    Args:
        args (argparse.Namespace): Arguments CLI.

    Returns:
        int: Code de retour processus.
    """

    target = get_primary_lockdown_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    print("[EvaCTL] Activation du verrouillage local.")
    print(f"Raison: {args.reason}")
    if not args.force:
        confirm = input("Confirmer la creation du verrouillage local ? (oui/NON): ")
        if confirm.strip().lower() != "oui":
            print("Operation annulee.")
            return 1

    payload = {
        "timestamp": datetime.now().isoformat(),
        "reason": args.reason,
        "triggered_by": "evactl",
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Verrouillage ecrit dans {target}")
    return 0


def show_logs(args: argparse.Namespace) -> int:
    """Affiche les derniers logs locaux d'un service.

    Args:
        args (argparse.Namespace): Arguments CLI.

    Returns:
        int: Code de retour processus.
    """

    candidates = [directory / f"{args.service}.log" for directory in LOG_CANDIDATES]
    for path in candidates:
        if path.exists():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-args.tail :] if args.tail > 0 else lines
            print(f"[EvaCTL] Logs pour {args.service} depuis {path}")
            print("\n".join(tail))
            return 0

    print(f"Aucun fichier de log trouve pour {args.service}.")
    for path in candidates:
        print(f"- {path}")
    return 1


def run_cpu(args: argparse.Namespace) -> int:
    """Delegue une commande CPU sure vers `cpu_assist.py`.

    Args:
        args (argparse.Namespace): Arguments CLI.

    Returns:
        int: Code de retour du sous-processus.
    """

    command = [sys.executable, str(CPU_ASSIST_PATH), *args.cpu_args]
    print(f"[EvaCTL] Delegation CPU: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(ROOT_DIR), check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur principal.

    Returns:
        argparse.ArgumentParser: Parseur configure.
    """

    parser = argparse.ArgumentParser(description="CLI d'administration locale pour THE HIVE.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_status = subparsers.add_parser("status", help="Verifie l'etat des services exposes.")
    parser_status.add_argument("--host", default="127.0.0.1", help="Hote cible pour les endpoints HTTP.")
    parser_status.add_argument("--timeout", type=float, default=2.0, help="Timeout HTTP en secondes.")
    parser_status.add_argument(
        "--service",
        action="append",
        choices=sorted(DEFAULT_PORTS.keys()),
        help="Limite le controle a un service donne. Peut etre repete.",
    )

    parser_panic = subparsers.add_parser("panic", help="Active un verrouillage local d'urgence.")
    parser_panic.add_argument("--reason", default="Intervention manuelle", help="Raison du verrouillage.")
    parser_panic.add_argument("--force", action="store_true", help="Supprime la demande de confirmation interactive.")

    parser_logs = subparsers.add_parser("logs", help="Affiche les logs locaux d'un service.")
    parser_logs.add_argument("--service", default="core", help="Nom du service a afficher.")
    parser_logs.add_argument("--tail", type=int, default=100, help="Nombre de lignes a afficher.")

    parser_cpu = subparsers.add_parser("cpu", help="Delegue aux charges CPU sures de cpu_assist.")
    parser_cpu.add_argument("cpu_args", nargs=argparse.REMAINDER, help="Arguments transmis tels quels a cpu_assist.py.")

    return parser


def main() -> int:
    """Point d'entree principal.

    Returns:
        int: Code de retour processus.
    """

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        return check_status(args)
    if args.command == "panic":
        return trigger_panic(args)
    if args.command == "logs":
        return show_logs(args)
    if args.command == "cpu":
        return run_cpu(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
