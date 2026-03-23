import os

import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

_, o, _ = client.exec_command(
    f"echo '{SUDO_PASS}' | sudo -S docker ps --filter name=hive-muse --filter name=hive-nexus --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
)
print(o.read().decode())

client.close()