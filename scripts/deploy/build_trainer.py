"""
Script to create required directories and verify the eva-trainer Docker image.
Also tests a direct docker run (bypassing Swarm).
"""
import os
import paramiko
import sys

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
REMOTE_DIR = "/home/aza/The_Hive"

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def run_cmd(client, cmd, timeout=300, print_live=True):
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=timeout)
    output = ""
    for line in iter(stdout.readline, ""):
        if print_live:
            print(line, end="")
        output += line
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, output


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=15)
    print("Connected to Proxmox")

    run_cmd(client, f"cd {REMOTE_DIR} && git pull origin feat/sprint-6")
    run_cmd(client, f"mkdir -p {REMOTE_DIR}/src/eva-lab/data/models")
    print("Data directory created")

    run_cmd(client, f"echo '{SUDO_PASS}' | sudo -S mkdir -p /mnt/data/docker_payload/volumes && echo OK || true")
    run_cmd(client, f"cd {REMOTE_DIR} && echo '{SUDO_PASS}' | sudo -S docker compose build eva-trainer", timeout=600)

    print("\nTesting with plain docker run (no Swarm):")
    exit_code, _ = run_cmd(
        client,
        f"echo '{SUDO_PASS}' | sudo -S docker run --rm --gpus all "
        f"-v {REMOTE_DIR}/src/eva-lab:/app/eva-lab "
        f"thehive/eva-trainer:latest "
        f"python -c \"import torch; "
        f"print('CUDA:', torch.cuda.is_available(), "
        f"'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\"",
        timeout=120,
    )
    if exit_code == 0:
        print("\neva-trainer works with direct docker run")
    else:
        print(f"\nExit code {exit_code}")

    client.close()
    print("\nDone")


if __name__ == "__main__":
    main()