import os
import paramiko
import sys

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

print("Triggering docker compose up --build --no-deps --no-cache for muse and nexus...")
cmd = (
    f"cd ~/The_Hive && echo '{SUDO_PASS}' | sudo -S docker compose -f docker-compose.yml build --no-cache muse nexus "
    f"&& echo '{SUDO_PASS}' | sudo -S docker compose -f docker-compose.yml up -d --no-deps muse nexus"
)
_, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
err = stderr.read().decode()
if err:
    print("Stderr:", err)

client.close()
print("\nDone!")