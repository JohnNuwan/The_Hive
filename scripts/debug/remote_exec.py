"""Execution distante via SSH (Paramiko) pour les scripts de debug/deploiement."""

import os
import sys

import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def main() -> int:
    """Execute une commande distante et retourne le code de sortie shell."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug/remote_exec.py <commande distante>")
        return 2

    cmd = " ".join(sys.argv[1:])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
        _, stdout, stderr = client.exec_command(cmd, get_pty=True)

        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")

        if out:
            print(out, end="")
        if err:
            print(err, end="", file=sys.stderr)

        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())