import os
import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def run(client, cmd):
    _, stdout, stderr = client.exec_command(cmd)
    return stdout.read().decode().strip()


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

print("=== RUNNING CONTAINERS ===")
print(run(client, "docker ps --format '{{.Names}}'"))

print("\n=== NETWORK: hive-net ===")
print(run(client, "docker network inspect the_hive_lite_hive-net --format '{{range .Containers}}{{.Name}} -> {{.IPv4Address}}{{println}}{{end}}'"))

print("\n=== HOST PORTS (6379, 6333, 7687, 11434) ===")
print(run(client, "netstat -tuln | grep -E '6379|6333|7687|11434'"))

client.close()