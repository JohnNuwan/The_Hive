"""Verifie les logs et artefacts de l'entrainement nocturne sur Proxmox."""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
REMOTE_DIR = "/home/aza/The_Hive"
LOG_PATH = f"{REMOTE_DIR}/hive_nightly_training.log"
SUMMARY_PATH = f"{REMOTE_DIR}/data/checkpoints/nightly_training_summary.json"

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def safe_print(text: str) -> None:
    """Affiche un texte en tolerant les caracteres non supportes par la console Windows."""
    sys.stdout.buffer.write(text.encode("utf-8", "replace"))
    if not text.endswith("\n"):
        sys.stdout.buffer.write(b"\n")


def main() -> None:
    """Affiche les derniers logs et le resume d'entrainement distant."""
    safe_print("Connexion au serveur Proxmox pour verification...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)

        for title, command in [
            ("Logs nocturnes", f"tail -n 60 {LOG_PATH}"),
            ("Resume JSON", f"cat {SUMMARY_PATH} 2>/dev/null || echo 'Resume indisponible pour le moment.'"),
            ("Etat GPU", "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader"),
        ]:
            safe_print(f"\n=== {title} ===")
            stdin, stdout, stderr = client.exec_command(command)
            safe_print(stdout.read().decode("utf-8", "ignore"))
            err = stderr.read().decode("utf-8", "ignore")
            if err:
                safe_print(err)

    finally:
        client.close()
        safe_print("Deconnexion.")


if __name__ == "__main__":
    main()
