"""
Script to install the nightly cron job for MTF-GNN training on Proxmox.
Run once: python scripts/deploy/setup_cron.py
"""
import os
import paramiko
import sys
import time

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
REMOTE_DIR = "/home/aza/The_Hive"

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

CRON_ENTRY = f"0 23 * * * {REMOTE_DIR}/scripts/auto_train_gnn.sh >> /var/log/hive_gnn_training.log 2>&1"


def setup_cron():
    print(f"Connecting to {HOST} as {USER}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
        print("Connected!")

        stdin, stdout, stderr = client.exec_command(f"cd {REMOTE_DIR} && git pull origin feat/sprint-6")
        print(stdout.read().decode())
        time.sleep(1)

        stdin, stdout, stderr = client.exec_command(f"chmod +x {REMOTE_DIR}/scripts/auto_train_gnn.sh")
        stdout.read()
        print("Script is now executable.")

        stdin, stdout, stderr = client.exec_command(f"sed -i 's/\\r$//' {REMOTE_DIR}/scripts/auto_train_gnn.sh")
        stdout.read()
        print("Line endings fixed.")

        update_path_cmd = f"sed -i 's|PROJECT_DIR.*|PROJECT_DIR=\"{REMOTE_DIR}\"|' {REMOTE_DIR}/scripts/auto_train_gnn.sh"
        stdin, stdout, stderr = client.exec_command(update_path_cmd)
        stdout.read()
        print("Script path set.")

        cron_cmd = (
            f"(crontab -l 2>/dev/null | grep -v 'auto_train_gnn'; "
            f"echo '{CRON_ENTRY}') | crontab -"
        )
        stdin, stdout, stderr = client.exec_command(cron_cmd)
        err = stderr.read().decode()
        if err:
            print(f"Warning: {err}")
        print("Cron entry installed!")

        stdin, stdout, stderr = client.exec_command("crontab -l")
        print("\nCurrent crontab:")
        print(stdout.read().decode())

    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)
    finally:
        client.close()
        print("Connection closed.")


if __name__ == "__main__":
    setup_cron()