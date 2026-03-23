import os
import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
REMOTE_DIR = "/home/aza/The_Hive"

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

print("\n--- Testing builder/substrate rebuild ---")
stdin, stdout, stderr = client.exec_command(
    f"cd {REMOTE_DIR} && echo '{SUDO_PASS}' | sudo -S docker compose build substrate builder",
    get_pty=True,
)
for line in iter(stdout.readline, ""):
    print(line, end="")

client.close()