import os
import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
LOCAL_FILE = "docker-compose.yml"
REMOTE_FILE = "/home/aza/The_Hive/docker-compose.yml"

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

with open(LOCAL_FILE, "r", encoding="utf-8-sig") as f:
    content = f.read()

sftp = client.open_sftp()
with sftp.open(REMOTE_FILE, "w") as f:
    f.write(content)
sftp.close()

print(f"Uploaded {LOCAL_FILE} to {REMOTE_FILE} cleanly.")

print("\n--- Starting the_hive project ---")
stdin, stdout, stderr = client.exec_command(
    f"cd /home/aza/The_Hive && echo '{SUDO_PASS}' | sudo -S docker compose up -d",
    get_pty=True,
)
for line in iter(stdout.readline, ""):
    print(line, end="")

client.close()
print("\nDone.")