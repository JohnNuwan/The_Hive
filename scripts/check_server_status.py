import os
import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def run(client, cmd):
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    return out, err


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

print("=== DOCKER PROCESSES ===")
out, _ = run(client, "docker ps --format '{{.Names}}\t{{.Status}}'")
print(out)

print("\n=== TRAINING LOGS (/var/log/hive_gnn_training.log) ===")
out, _ = run(client, "tail -n 100 /var/log/hive_gnn_training.log")
print(out or "Log is empty")

print("\n=== SCRIPT CONTENT (scripts/auto_train_gnn.sh) ===")
out, _ = run(client, "cat /home/aza/The_Hive/scripts/auto_train_gnn.sh")
print(out)

client.close()