import os
import paramiko
import sys

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def start_training():
    print(f"Connecting to Proxmox Server {HOST} to start training...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected! Launching offline_trainer.py in background inside Lab container...")

        train_cmd = (
            f"cd ~/The_Hive && echo '{SUDO_PASS}' | sudo -S docker compose exec "
            f"-e PYTHONPATH=/app/eva-lab -d lab python eva_lab/muzero/offline_trainer.py"
        )
        stdin, stdout, stderr = client.exec_command(train_cmd)

        output = stdout.read().decode()
        error = stderr.read().decode()

        if output:
            print("Output:", output)
        if error and not error.startswith("[sudo] password for"):
            print("Stderr:", error)

        print("Training launched successfully on the server!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    start_training()