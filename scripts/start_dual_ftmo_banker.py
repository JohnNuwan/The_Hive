"""
Prepare et lance deux instances locales de The Banker sur deux terminaux MT5 distincts.

Le script cree :
- une instance maitre sur le compte FTMO principal ;
- une instance fille sur le compte FTMO challenge a recopier proportionnellement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


def _log(message: str) -> None:
    """Affiche un message simple pour le provisioning local."""
    print(message)


def _detect_source_terminal_dir(explicit_path: str | None) -> Path:
    """
    Resolve le terminal FTMO source a dupliquer.

    Args:
        explicit_path (str | None): Repertoire ou executable explicite fourni en argument.

    Returns:
        Path: Repertoire du terminal MT5 source.

    Raises:
        FileNotFoundError: Si aucun terminal FTMO exploitable n'est trouve.
    """
    candidates: list[Path] = []
    if explicit_path:
        explicit = Path(explicit_path).expanduser().resolve()
        candidates.append(explicit.parent if explicit.is_file() else explicit)

    candidates.extend(
        [
            Path(r"C:\Program Files\FTMO Global Markets MT5 Terminal"),
            Path(r"C:\Program Files (x86)\FTMO Global Markets MT5 Terminal"),
        ]
    )

    for candidate in candidates:
        terminal_exe = candidate / "terminal64.exe"
        if terminal_exe.exists():
            return candidate

    raise FileNotFoundError("Aucun terminal FTMO source n'a ete trouve sur cette machine.")


def _ensure_terminal_copy(source_dir: Path, target_dir: Path) -> Path:
    """
    Duplique le terminal source vers un repertoire portable dedie.

    Args:
        source_dir (Path): Repertoire source du terminal FTMO.
        target_dir (Path): Repertoire cible de l'instance.

    Returns:
        Path: Chemin absolu de `terminal64.exe` dans l'instance cible.

    Raises:
        RuntimeError: Si la duplication du terminal echoue.
    """
    terminal_path = target_dir / "terminal64.exe"
    if terminal_path.exists():
        return terminal_path

    target_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "robocopy",
        str(source_dir),
        str(target_dir),
        "/E",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode > 7:
        raise RuntimeError(
            f"Echec de duplication du terminal MT5 vers {target_dir}: {result.stdout or result.stderr}"
        )
    if not terminal_path.exists():
        raise RuntimeError(f"Le terminal cible {terminal_path} est absent apres duplication.")
    return terminal_path


def _format_env_value(value: Any) -> str:
    """
    Formate une valeur pour un fichier `.env` local.

    Args:
        value (Any): Valeur brute a serialiser.

    Returns:
        str: Valeur encadree et echappee pour python-dotenv.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace("\\", "/").replace('"', '\\"')
    return f'"{text}"'


def _write_override_env(path: Path, values: dict[str, Any]) -> None:
    """
    Ecrit un fichier d'override local pour une instance Banker.

    Args:
        path (Path): Fichier `.env.*.local` cible.
        values (dict[str, Any]): Variables d'environnement a ecrire.
    """
    lines = [
        "# Fichier local genere automatiquement pour une instance Banker.",
        "# Ce fichier est ignore par Git.",
        "",
    ]
    for key, value in values.items():
        lines.append(f"{key}={_format_env_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wait_for_health(port: int, timeout_seconds: int = 120) -> bool:
    """
    Attend qu'une instance Banker reponde sur son endpoint `/health`.

    Args:
        port (int): Port HTTP local de l'instance.
        timeout_seconds (int): Delai maximum d'attente.

    Returns:
        bool: True si l'instance est joignable avant le timeout.
    """
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1)
    return False


def _start_banker_process(
    repo_root: Path,
    env_file: Path,
    port: int,
    instance_name: str,
    enable_tunnel: bool,
) -> subprocess.Popen[str]:
    """
    Lance une instance Banker via `banker.bat` avec un override local.

    Args:
        repo_root (Path): Racine du depot.
        env_file (Path): Fichier `.env` local de l'instance.
        port (int): Port HTTP a exposer.
        instance_name (str): Nom lisible de l'instance.
        enable_tunnel (bool): Active le tunnel SSH si True.

    Returns:
        subprocess.Popen[str]: Processus `cmd.exe` lance.
    """
    env = os.environ.copy()
    env["BANKER_ENV_FILE"] = str(env_file)
    env["BANKER_API_PORT"] = str(port)
    env["BANKER_INSTANCE_NAME"] = instance_name
    env["BANKER_ENABLE_TUNNEL"] = "true" if enable_tunnel else "false"
    env.setdefault("BANKER_BIND_HOST", "0.0.0.0")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    return subprocess.Popen(
        ["cmd.exe", "/c", "banker.bat"],
        cwd=str(repo_root),
        env=env,
        creationflags=creationflags,
        text=True,
    )


def _build_copy_targets_json(
    follower_port: int,
    follower_login: int,
    follower_server: str,
    follower_phase: str,
    follower_balance_reference: int,
) -> str:
    """
    Construit la configuration de copie proportionnelle vers le compte fille.

    Args:
        follower_port (int): Port HTTP local du banker fille.
        follower_login (int): Login MT5 du compte fille.
        follower_server (str): Serveur MT5 du compte fille.
        follower_phase (str): Phase prop firm du compte fille.
        follower_balance_reference (int): Capital de reference du compte fille.

    Returns:
        str: Configuration JSON serialisee pour `BANKER_COPY_TARGETS_JSON`.
    """
    payload = [
        {
            "name": "FTMO Challenge 50K",
            "banker_base_url": f"http://127.0.0.1:{follower_port}",
            "allocation_ratio": "1.0",
            "enabled": True,
            "phase": follower_phase,
            "broker": "FTMO",
            "server": follower_server,
            "login": follower_login,
            "terminal_label": "MT5 FTMO Challenge 50K",
            "balance_reference": str(follower_balance_reference),
            "use_equity_for_sizing": True,
        }
    ]
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def main() -> int:
    """
    Point d'entree principal du provisionnement dual-banker FTMO.

    Returns:
        int: Code retour shell.
    """
    parser = argparse.ArgumentParser(description="Prepare deux instances Banker FTMO locales.")
    parser.add_argument("--follower-login", type=int, required=True, help="Login MT5 du compte challenge.")
    parser.add_argument("--follower-password", required=True, help="Mot de passe MT5 du compte challenge.")
    parser.add_argument("--follower-server", required=True, help="Serveur MT5 du compte challenge.")
    parser.add_argument("--follower-phase", default="challenge", help="Phase du compte challenge.")
    parser.add_argument(
        "--follower-balance-reference",
        type=int,
        default=50000,
        help="Capital de reference du compte challenge pour la copie proportionnelle.",
    )
    parser.add_argument("--master-port", type=int, default=8100, help="Port local du banker maitre.")
    parser.add_argument("--follower-port", type=int, default=8110, help="Port local du banker challenge.")
    parser.add_argument(
        "--source-terminal-dir",
        default="",
        help="Repertoire du terminal FTMO source a dupliquer si la detection automatique echoue.",
    )
    parser.add_argument("--no-launch", action="store_true", help="Prepare seulement les fichiers sans lancer les processus.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_env_path = repo_root / ".env"
    if not base_env_path.exists():
        raise FileNotFoundError(f"Le fichier de base {base_env_path} est introuvable.")

    base_env = dotenv_values(base_env_path)
    master_login = int(str(base_env.get("MT5_LOGIN") or "0"))
    master_server = str(base_env.get("MT5_SERVER") or "").strip()
    master_password = str(base_env.get("MT5_PASSWORD") or "").strip()
    if master_login <= 0 or not master_server or not master_password:
        raise RuntimeError("Le compte FTMO principal n'est pas complet dans .env.")

    source_terminal_dir = _detect_source_terminal_dir(args.source_terminal_dir)
    terminal_root = Path(os.environ.get("LOCALAPPDATA", str(repo_root / "tmp"))) / "TheHive" / "MT5"
    master_terminal_exe = source_terminal_dir / "terminal64.exe"
    follower_terminal_exe = _ensure_terminal_copy(source_terminal_dir, terminal_root / "FTMO-Server3-Challenge50K")

    master_env_file = repo_root / ".env.banker.master.local"
    follower_env_file = repo_root / ".env.banker.ftmo50k.local"

    master_copy_targets_json = _build_copy_targets_json(
        follower_port=args.follower_port,
        follower_login=args.follower_login,
        follower_server=args.follower_server,
        follower_phase=args.follower_phase,
        follower_balance_reference=args.follower_balance_reference,
    )

    _write_override_env(
        master_env_file,
        {
            "BANKER_API_PORT": args.master_port,
            "BANKER_INSTANCE_NAME": "FTMO Master 10K",
            "BANKER_FOLLOWER_MODE": False,
            "MOCK_MT5": False,
            "PAPER_TRADING": False,
            "MT5_LOGIN": master_login,
            "MT5_SERVER": master_server,
            "MT5_TERMINAL_PATH": master_terminal_exe.as_posix(),
            "MT5_TERMINAL_PORTABLE": False,
            "BANKER_COPY_TARGETS_JSON": master_copy_targets_json,
        },
    )

    _write_override_env(
        follower_env_file,
        {
            "BANKER_API_PORT": args.follower_port,
            "BANKER_INSTANCE_NAME": "FTMO Challenge 50K",
            "BANKER_FOLLOWER_MODE": True,
            "MOCK_MT5": False,
            "PAPER_TRADING": False,
            "MT5_LOGIN": args.follower_login,
            "MT5_PASSWORD": args.follower_password,
            "MT5_SERVER": args.follower_server,
            "MT5_TERMINAL_PATH": follower_terminal_exe.as_posix(),
            "MT5_TERMINAL_PORTABLE": True,
            "BANKER_COPY_TARGETS_JSON": "",
        },
    )

    _log(f"Profil maitre ecrit dans {master_env_file}")
    _log(f"Profil challenge ecrit dans {follower_env_file}")
    _log(f"Terminal maitre: {master_terminal_exe}")
    _log(f"Terminal challenge: {follower_terminal_exe}")

    if args.no_launch:
        _log("Preparation terminee sans lancement des instances.")
        return 0

    follower_process = _start_banker_process(
        repo_root=repo_root,
        env_file=follower_env_file,
        port=args.follower_port,
        instance_name="FTMO Challenge 50K",
        enable_tunnel=False,
    )
    _log(f"Instance challenge demarree (PID {follower_process.pid}) sur le port {args.follower_port}.")
    if not _wait_for_health(args.follower_port):
        raise RuntimeError("Le banker challenge ne repond pas sur /health.")

    master_process = _start_banker_process(
        repo_root=repo_root,
        env_file=master_env_file,
        port=args.master_port,
        instance_name="FTMO Master 10K",
        enable_tunnel=True,
    )
    _log(f"Instance maitre demarree (PID {master_process.pid}) sur le port {args.master_port}.")
    if not _wait_for_health(args.master_port):
        raise RuntimeError("Le banker maitre ne repond pas sur /health.")

    _log("Les deux instances Banker FTMO sont en ligne.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
