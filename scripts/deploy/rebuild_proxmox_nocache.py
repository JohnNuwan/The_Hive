import paramiko
import sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)

print("Triggering docker compose up --build --no-deps --no-cache for muse and nexus...")
# --no-cache ensures docker doesn't use a stale cached layer that misses dist/
cmd = "cd ~/The_Hive && echo 'Kumara-42/600' | sudo -S docker compose -f docker-compose.yml build --no-cache muse nexus && echo 'Kumara-42/600' | sudo -S docker compose -f docker-compose.yml up -d --no-deps muse nexus"
_, stdout, stderr = client.exec_command(cmd, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
err = stderr.read().decode()
if err:
    print("Stderr:", err)
    
client.close()
print("\nDone!")
