"""Lanceur Banker avec duplication des logs vers un fichier persistant."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_LOGS_DIR = Path("logs")


def _slugify_instance_name(value: str) -> str:
    """Construit un slug stable pour nommer les fichiers de log.

    Args:
        value (str): Nom libre de l'instance.

    Returns:
        str: Identifiant fichier ASCII simple.
    """

    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(value or "").strip())
    compact = "_".join(segment for segment in normalized.split("_") if segment)
    return compact or "banker"


def resolve_log_file_path() -> Path:
    """Determine le chemin du fichier de log pour l'instance courante.

    Returns:
        Path: Fichier de log absolu ou relatif au workspace.
    """

    explicit_path = str(os.getenv("BANKER_LOG_FILE") or "").strip()
    if explicit_path:
        return Path(explicit_path)

    instance_name = str(os.getenv("BANKER_INSTANCE_NAME") or "").strip()
    if not instance_name:
        env_file = str(os.getenv("BANKER_ENV_FILE") or "").strip()
        instance_name = Path(env_file).stem if env_file else "banker"
    return DEFAULT_LOGS_DIR / f"{_slugify_instance_name(instance_name)}.log"


def build_uvicorn_command(host: str, port: int, env_file: str) -> list[str]:
    """Construit la commande uvicorn a executer.

    Args:
        host (str): Hote de bind local.
        port (int): Port HTTP du Banker.
        env_file (str): Fichier d'environnement cible.

    Returns:
        list[str]: Ligne de commande complete.
    """

    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "uvicorn",
        "eva_banker.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--env-file",
        env_file,
        "--no-access-log",
    ]


def safe_write_stdout(text: str) -> None:
    """Ecrit du texte dans stdout de maniere securisee face aux erreurs d'encodage."""
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        encoded = text.encode(encoding, errors="replace").decode(encoding)
        sys.stdout.write(encoded)


def stream_process_output(process: subprocess.Popen[str], log_file_path: Path) -> int:
    """Recopie la sortie d'un sous-processus vers la console et un fichier.

    Args:
        process (subprocess.Popen[str]): Processus uvicorn deja demarre.
        log_file_path (Path): Fichier de destination des logs.

    Returns:
        int: Code retour du processus.
    """

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    start_line = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Demarrage du logger Banker vers {log_file_path}\n"
    )
    safe_write_stdout(start_line)
    sys.stdout.flush()

    with log_file_path.open("a", encoding="utf-8") as handle:
        handle.write(start_line)
        handle.flush()
        if process.stdout is None:
            return process.wait()

        for line in process.stdout:
            handle.write(line)
            handle.flush()
            safe_write_stdout(line)
            sys.stdout.flush()

    return process.wait()


def run_banker(host: str, port: int, env_file: str) -> int:
    """Demarre uvicorn et duplique ses logs vers un fichier persistant.

    Args:
        host (str): Hote de bind local.
        port (int): Port HTTP d'ecoute.
        env_file (str): Fichier `.env` a utiliser.

    Returns:
        int: Code retour du processus uvicorn.
    """

    os.environ["BANKER_ENV_FILE"] = env_file
    log_file_path = resolve_log_file_path()
    command = build_uvicorn_command(host=host, port=port, env_file=env_file)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        return stream_process_output(process, log_file_path=log_file_path)
    except KeyboardInterrupt:
        process.terminate()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=10)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI du lanceur Banker.

    Returns:
        argparse.ArgumentParser: Parseur configure.
    """

    parser = argparse.ArgumentParser(description="Lance The Banker avec logs persistants.")
    parser.add_argument("--host", required=True, help="Hote de bind du Banker.")
    parser.add_argument("--port", required=True, type=int, help="Port HTTP du Banker.")
    parser.add_argument("--env-file", required=True, help="Fichier `.env` du Banker.")
    return parser


def main() -> int:
    """Point d'entree CLI du lanceur Banker.

    Returns:
        int: Code retour du processus.
    """

    parser = _build_argument_parser()
    args = parser.parse_args()
    return run_banker(host=str(args.host), port=int(args.port), env_file=str(args.env_file))


if __name__ == "__main__":
    raise SystemExit(main())
