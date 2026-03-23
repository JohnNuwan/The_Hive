import os
import paramiko
import sys

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def deploy():
    print(f"Connecting to Proxmox Server {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected! Fetching remote changes and resetting branch...")

        git_cmds = "cd ~/The_Hive && git fetch origin main && git reset --hard origin/main"
        stdin, stdout, stderr = client.exec_command(git_cmds)
        print(stdout.read().decode())
        print(stderr.read().decode())

        print("Rebuilding Docker containers Lab, Core, and Nexus...")
        docker_cmd = f"cd ~/The_Hive && echo '{SUDO_PASS}' | sudo -S docker compose -f docker-compose.yml up -d --build lab core nexus"
        stdin, stdout, stderr = client.exec_command(docker_cmd, get_pty=True)

        for line in iter(stdout.readline, ""):
            print(line, end="")

        err = stderr.read().decode()
        if err:
            print(f"Stderr: {err}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    deploy()