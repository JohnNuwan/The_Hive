"""Diagnose ComfyUI port availability from the muse container perspective."""
import os
import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)


def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    stdin.write(f"{SUDO_PASS}\n")
    stdin.flush()
    return stdout.read().decode("utf-8", errors="replace").strip()


print("=== ComfyUI containers ===")
print(run("sudo -S docker ps --format '{{.Names}}\\t{{.Ports}}\\t{{.Status}}' | grep -i comfy"))

print("\n=== Listening on 8188 ===")
print(run("ss -tlnp | grep 8188"))

print("\n=== Curl ComfyUI from muse container ===")
print(run("sudo -S docker exec hive-muse curl -s http://192.168.1.6:8188/system_stats --max-time 3 2>&1 | head -c 200"))

print("\n=== Curl ComfyUI from host ===")
print(run("curl -s http://localhost:8188/system_stats --max-time 3 2>&1 | head -c 200"))

print("\n=== Docker networks ComfyUI ===")
print(run("sudo -S docker inspect eva_comfyui --format '{{json .HostConfig.PortBindings}}' 2>/dev/null || echo 'Container eva_comfyui not found'"))

c.close()