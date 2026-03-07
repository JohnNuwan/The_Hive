import os
import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
REMOTE_DIR = "/home/aza/The_Hive"

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def run_interactive(client, command):
    print(f"Running: {command}")
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    return stdout.channel.recv_exit_status()


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

print("\n--- Stopping the_hive_lite ---")
run_interactive(client, f"cd {REMOTE_DIR} && echo '{SUDO_PASS}' | sudo -S docker compose -p the_hive_lite down --remove-orphans")

print("\n--- Updating COMPOSE_PROJECT_NAME in .env ---")
run_interactive(client, f"sed -i 's/COMPOSE_PROJECT_NAME=the_hive_lite/COMPOSE_PROJECT_NAME=the_hive/' {REMOTE_DIR}/.env")

print("\n--- Starting the_hive project ---")
run_interactive(client, f"cd {REMOTE_DIR} && echo '{SUDO_PASS}' | sudo -S docker compose up -d")

print("\n--- Verifying container names ---")
stdin, stdout, stderr = client.exec_command("docker ps --format '{{.Names}}' | head -n 10")
print(stdout.read().decode())

client.close()
print("\nDone.")